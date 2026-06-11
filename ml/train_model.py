"""
Bitcoin 4H Prediction — XGBoost Training
=========================================
Trains two models:
  1. Classifier  → direction (up / down)     target: target_direction_4h
  2. Regressor   → magnitude (4H return %)   target: target_return_4h

Both use walk-forward split (time-ordered 70/10/20).
  Train 70% → Val 10% (early stopping only) → Test 20% (final eval, never touched during training)
No random split — prevents future data leaking into training.

Usage:
    python3 train_model.py
    python3 train_model.py --db-path ./feature_store.duckdb
    python3 train_model.py --train-ratio 0.70 --val-ratio 0.10
"""

import argparse
import json
import os
import sys
import pickle
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
except ImportError:
    # Allow importing this module's constants/helpers (e.g. EXCLUDE_COLS, the
    # split functions) without xgboost installed — eda.py imports EXCLUDE_COLS
    # for a validation check and must not require the training stack. Training
    # itself checks for xgb and errors clearly in main().
    xgb = None
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH

# ── Columns excluded from features ───────────────────────────────────────────
EXCLUDE_COLS = {
    "open_time",
    "target_return_4h",
    "target_direction_4h",
    # Raw absolute prices — not predictive (use returns / deviations instead)
    "open", "high", "low", "close",
    # OI columns — 97%+ missing
    "sum_open_interest", "oi_zscore", "oi_delta_norm",
}

MAX_NAN_PCT = 0.50   # drop any feature with >50% missing values


# ── Data loading ──────────────────────────────────────────────────────────────


def load_features(db_path: str, table: str = "ml_features_clean") -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if table not in tables:
        available = [t for t in tables if "ml_features" in t]
        print(f"ERROR: table '{table}' not found.")
        print(f"Available ml_features tables: {available}")
        print(f"Run eda.py first to create ml_features_clean.")
        sys.exit(1)
    df = con.execute(f"SELECT * FROM {table} ORDER BY open_time").df()
    con.close()
    print(f"Table:      {table}")
    print(f"Loaded:     {len(df):,} rows × {df.shape[1]} columns")
    print(f"Date range: {pd.to_datetime(df['open_time'].min(), unit='ms')} → "
          f"{pd.to_datetime(df['open_time'].max(), unit='ms')}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    # Keep only numeric columns — XGBoost cannot handle strings/objects
    numeric_cols = df[cols].select_dtypes(include=[np.number]).columns.tolist()
    dropped_type = [c for c in cols if c not in numeric_cols]
    if dropped_type:
        print(f"Dropped {len(dropped_type)} non-numeric features: {dropped_type}")
    cols = numeric_cols
    # Drop columns with too many NaNs
    nan_pct = df[cols].isnull().mean()
    dropped_nan = [c for c in cols if nan_pct[c] > MAX_NAN_PCT]
    cols = [c for c in cols if nan_pct[c] <= MAX_NAN_PCT]
    if dropped_nan:
        print(f"Dropped {len(dropped_nan)} high-NaN features: {dropped_nan}")
    print(f"Features used: {len(cols)}")
    return cols


# ── Walk-forward split (3-way) ────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.10):
    """
    Time-ordered 3-way split:
      Train  → model learns from this
      Val    → early stopping only (model sees loss but NOT test labels)
      Test   → final evaluation, never touched during training

    Using test set for early stopping (the old approach) leaks future data
    into the model and inflates reported accuracy.
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"test_ratio={test_ratio:.3f} — train_ratio + val_ratio must be < 1.0 "
            f"(got {train_ratio} + {val_ratio} = {train_ratio + val_ratio})"
        )

    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    if train_end < 100:
        raise ValueError(f"Train set too small: {train_end} rows. Need ≥ 100.")
    if (val_end - train_end) < 50:
        raise ValueError(f"Val set too small: {val_end - train_end} rows. Need ≥ 50.")
    if (n - val_end) < 50:
        raise ValueError(f"Test set too small: {n - val_end} rows. Need ≥ 50.")

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    # Verify no time overlap (guaranteed by construction, but assert-free check)
    if train.index[-1] >= val.index[0]:
        raise ValueError("Train/val time overlap — data may not be sorted by open_time")
    if val.index[-1] >= test.index[0]:
        raise ValueError("Val/test time overlap — data may not be sorted by open_time")

    return train, val, test


# ── Walk-forward cross-validation (Codex point 12) ────────────────────────────

def make_walk_forward_folds(n: int, n_folds: int, min_block: int = 100):
    """
    Anchored / expanding-window walk-forward fold boundaries.

    The timeline is cut into (n_folds + 1) equal blocks. Fold i trains on
    blocks 0..i and tests on block i+1, so train ALWAYS precedes test in time
    and every fold's test block is a DISTINCT, later, untouched period:

        fold 1:  train [#####]            test [-----]
        fold 2:  train [##########]       test      [-----]
        fold 3:  train [###############]  test           [-----]

    Reporting mean ± std of the metric across these folds shows whether the
    edge is stable through time or just an artifact of one lucky 20% split.

    Returns a list of (train_end, test_end) index boundaries, or [] if the data
    is too small to give each block at least `min_block` rows.
    """
    if n_folds < 2:
        return []
    anchor = n // (n_folds + 1)
    if anchor < min_block:
        return []
    folds = []
    for i in range(1, n_folds + 1):
        train_end = i * anchor
        test_end  = (i + 1) * anchor if i < n_folds else n
        folds.append((train_end, test_end))
    return folds


def walk_forward_cv(df, feature_cols, n_folds: int, val_ratio: float,
                    model_factory=None):
    """
    Run anchored walk-forward CV for the DIRECTION classifier — the headline
    out-of-sample test for a trading model. `model_factory` returns a fresh
    estimator each fold (defaults to XGBClassifier(**CLF_PARAMS)); injectable so
    the logic can be unit-tested without XGBoost installed.
    """
    print("\n" + "=" * 65)
    print(f"  WALK-FORWARD CV — CLASSIFIER ({n_folds} folds, expanding window)")
    print("=" * 65)

    if model_factory is None:
        model_factory = lambda: xgb.XGBClassifier(**CLF_PARAMS)

    data = df[["open_time"] + feature_cols + ["target_direction_4h"]].copy()
    data = data.dropna(subset=["target_direction_4h"])
    data = data[data["target_direction_4h"] != 0]
    data["y"] = (data["target_direction_4h"] == 1).astype(int)
    data = data.reset_index(drop=True)

    folds = make_walk_forward_folds(len(data), n_folds)
    if not folds:
        print(f"  Not enough rows ({len(data):,}) for {n_folds}-fold CV — skipping.")
        return None

    print(f"\n  {'Fold':<5}{'TrainN':>9}{'TestN':>8}{'Acc':>9}{'Major':>8}{'Edge':>8}   Test window")
    print(f"  {'-'*5}{'-'*9}{'-'*8}{'-'*9}{'-'*8}{'-'*8}   {'-'*23}")

    accs, edges, fold_log = [], [], []
    for k, (tr_end, te_end) in enumerate(folds, 1):
        tr = data.iloc[:tr_end]
        te = data.iloc[tr_end:te_end]
        # carve a val tail from train for early stopping (keeps test untouched)
        v_start = int(len(tr) * (1 - val_ratio))
        tr_in, va = tr.iloc[:v_start], tr.iloc[v_start:]

        if min(tr_in["y"].nunique(), va["y"].nunique(), te["y"].nunique()) < 2:
            print(f"  {k:<5}{len(tr):>9,}{len(te):>8,}   single-class split — skipped")
            fold_log.append({"fold": k, "skipped": True})
            continue

        model = model_factory()
        try:
            model.fit(tr_in[feature_cols], tr_in["y"],
                      eval_set=[(va[feature_cols], va["y"])], verbose=False)
        except TypeError:
            model.fit(tr_in[feature_cols], tr_in["y"])   # estimators w/o eval_set

        pred = model.predict(te[feature_cols])
        acc  = accuracy_score(te["y"], pred)
        maj  = max(te["y"].mean(), 1 - te["y"].mean())
        edge = acc - maj
        t0 = pd.to_datetime(te["open_time"].iloc[0],  unit="ms").strftime("%Y-%m-%d")
        t1 = pd.to_datetime(te["open_time"].iloc[-1], unit="ms").strftime("%Y-%m-%d")
        accs.append(acc); edges.append(edge)
        fold_log.append({"fold": k, "train_n": len(tr), "test_n": len(te),
                         "acc": round(float(acc), 4), "majority": round(float(maj), 4),
                         "edge": round(float(edge), 4), "test_start": t0, "test_end": t1})
        print(f"  {k:<5}{len(tr):>9,}{len(te):>8,}{acc:>9.4f}{maj:>8.4f}{edge*100:>+7.2f}%   {t0}→{t1}")

    if not accs:
        print("\n  All folds skipped (single-class). CV inconclusive.")
        return {"folds": fold_log}

    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    mean_edge, std_edge = float(np.mean(edges)), float(np.std(edges))
    print(f"  {'-'*60}")
    print(f"  Mean accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean edge over majority: {mean_edge*100:+.2f}% ± {std_edge*100:.2f}%")
    if mean_edge <= 0:
        print(f"  ⚠️  No positive edge on average — model does not beat the naive baseline out-of-sample.")
    elif std_edge > abs(mean_edge):
        print(f"  ⚠️  Edge is smaller than its fold-to-fold volatility — not yet reliable.")
    else:
        print(f"  ✅  Positive, reasonably stable edge across {len(accs)} untouched periods.")

    return {"folds": fold_log, "mean_acc": mean_acc, "std_acc": std_acc,
            "mean_edge": mean_edge, "std_edge": std_edge}


# ── XGBoost hyperparameters ───────────────────────────────────────────────────
# Conservative settings to avoid overfitting on financial time series.
# max_depth=4, min_child_weight=10, gamma=1 prevent the model memorising noise.

CLF_PARAMS = dict(
    n_estimators=500,
    max_depth=3,           # reduced — less overfitting on noisy financial data
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=30,   # increased — needs more samples per leaf
    gamma=2,               # increased — harder split threshold
    reg_alpha=0.5,         # increased L1
    reg_lambda=2.0,        # increased L2
    eval_metric="logloss",
    early_stopping_rounds=50,  # more patience
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

REG_PARAMS = dict(
    # Regressor: NO early stopping — it fires at tree 0 because 4H returns are
    # too noisy for val loss to improve. Use a fixed n_estimators instead.
    # Previous values (min_child_weight=50, gamma=3) were so conservative that
    # no splits occurred — all importances came out 0.0. Reduced significantly.
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=10,   # was 50 — too conservative, caused all-zero importances
    gamma=0.0,             # was 0.5 — gamma is an ABSOLUTE min-loss-reduction; 4H
                           # return variance is ~1e-4, so ANY gamma>~1e-4 blocks
                           # every split → all-zero importances. Overfit control
                           # here comes from min_child_weight + reg_lambda + depth.
    reg_alpha=0.3,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)


# ── Classifier ────────────────────────────────────────────────────────────────

def train_classifier(df: pd.DataFrame, feature_cols: list[str], train_ratio: float, val_ratio: float):
    print("\n" + "=" * 65)
    print("  MODEL 1 — CLASSIFIER  (direction: UP or DOWN in 4H)")
    print("=" * 65)

    # Drop rows where target is NaN or 0 (flat / no signal)
    data = df[feature_cols + ["target_direction_4h"]].copy()
    data = data.dropna(subset=["target_direction_4h"])
    data = data[data["target_direction_4h"] != 0]

    # Encode: 1 = UP, 0 = DOWN
    data["target"] = (data["target_direction_4h"] == 1).astype(int)
    data = data.drop(columns=["target_direction_4h"])

    # 3-way split — val used ONLY for early stopping, test never seen during training
    train, val, test = walk_forward_split(data, train_ratio, val_ratio)

    X_train, y_train = train[feature_cols], train["target"]
    X_val,   y_val   = val[feature_cols],   val["target"]
    X_test,  y_test  = test[feature_cols],  test["target"]

    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"  Train UP: {y_train.mean():.1%} | Val UP: {y_val.mean():.1%} | Test UP: {y_test.mean():.1%}")

    # Fix 2: class-diversity check — single-class splits break XGBoost and give
    # meaningless results. Fail early with a clear message.
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        classes = set(y_split.unique())
        if len(classes) < 2:
            raise ValueError(
                f"{split_name} set contains only class {classes} — "
                f"both UP (1) and DOWN (0) must be present. "
                f"Adjust split ratios or add more data."
            )

    model = xgb.XGBClassifier(**CLF_PARAMS)
    # ✅ Early stopping uses VAL set — test set is never touched during training
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    # Fix 1: labels=[0, 1] guarantees a 2×2 matrix even if test set has one class
    cm   = confusion_matrix(y_test, y_pred, labels=[0, 1])

    # Baseline 1: always predict majority class
    majority     = int(y_test.mean() >= 0.5)
    baseline_acc = accuracy_score(y_test, [majority] * len(y_test))

    # Baseline 2: predict same direction as last 1H candle (naive momentum)
    if "is_bullish" in feature_cols:
        naive_pred   = test["is_bullish"].fillna(majority).astype(int).values
        naive_acc    = accuracy_score(y_test, naive_pred)
    else:
        naive_acc = baseline_acc

    print(f"\n  ── Results ──")
    print(f"  Accuracy:             {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Majority baseline:    {baseline_acc:.4f}  ({baseline_acc*100:.2f}%)")
    print(f"  Naive momentum base:  {naive_acc:.4f}  ({naive_acc*100:.2f}%)")
    print(f"  Edge over majority:   {(acc - baseline_acc)*100:+.2f}%")
    print(f"  Edge over momentum:   {(acc - naive_acc)*100:+.2f}%")
    print(f"  Precision:            {prec:.4f}  (of predicted UP → how many correct)")
    print(f"  Recall:               {rec:.4f}  (of actual UP moves → how many caught)")
    print(f"  F1 Score:             {f1:.4f}")

    print(f"\n  ── Confusion Matrix ──")
    print(f"                 Pred DOWN   Pred UP")
    print(f"  Actual DOWN     {cm[0,0]:>6}      {cm[0,1]:>6}")
    print(f"  Actual UP       {cm[1,0]:>6}      {cm[1,1]:>6}")

    imp = _importance_df(feature_cols, model.feature_importances_)
    _print_importance(imp, "Classifier")

    return model, imp


# ── Regressor ─────────────────────────────────────────────────────────────────

def train_regressor(df: pd.DataFrame, feature_cols: list[str], train_ratio: float, val_ratio: float):
    print("\n" + "=" * 65)
    print("  MODEL 2 — REGRESSOR  (4H return magnitude)")
    print("=" * 65)

    data = df[feature_cols + ["target_return_4h"]].dropna(subset=["target_return_4h"]).copy()

    # 3-way split — val used ONLY for early stopping
    train, val, test = walk_forward_split(data, train_ratio, val_ratio)

    X_train, y_train = train[feature_cols], train["target_return_4h"]
    X_val,   y_val   = val[feature_cols],   val["target_return_4h"]
    X_test,  y_test  = test[feature_cols],  test["target_return_4h"]

    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"  Target mean ± std:  {y_test.mean():.6f} ± {y_test.std():.6f}")

    model = xgb.XGBRegressor(**REG_PARAMS)
    # No early stopping for regressor — 4H returns are too noisy for val loss
    # to reliably improve. Fixed n_estimators with heavy regularization instead.
    model.fit(X_train, y_train, verbose=False)

    # Fix 5: use val set for diagnostics (was created but never used before)
    val_pred    = model.predict(X_val)
    val_mae     = mean_absolute_error(y_val, val_pred)
    val_dir_acc = (np.sign(val_pred) == np.sign(y_val)).mean()

    y_pred  = model.predict(X_test)
    mae     = mean_absolute_error(y_test, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    r2      = r2_score(y_test, y_pred)
    dir_acc = (np.sign(y_pred) == np.sign(y_test)).mean()

    # Baseline: always predict the mean return
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae  = mean_absolute_error(y_test, baseline_pred)
    baseline_dir  = (np.sign(baseline_pred) == np.sign(y_test)).mean()

    print(f"\n  ── Val Set Diagnostics ──")
    print(f"  Val MAE:           {val_mae:.6f}  ({val_mae*100:.4f}%)")
    print(f"  Val Dir Acc:       {val_dir_acc:.4f}  ({val_dir_acc*100:.2f}%)")

    print(f"\n  ── Test Results ──")
    print(f"  MAE:               {mae:.6f}  ({mae*100:.4f}%)")
    print(f"  RMSE:              {rmse:.6f}  ({rmse*100:.4f}%)")
    print(f"  R²:                {r2:.4f}  (1.0 = perfect, 0.0 = no better than mean)")
    print(f"  Directional Acc:   {dir_acc:.4f}  ({dir_acc*100:.2f}%)  ← did model get UP/DOWN right?")
    print(f"\n  ── Baseline (predict mean always) ──")
    print(f"  Baseline MAE:      {baseline_mae:.6f}")
    print(f"  Baseline Dir Acc:  {baseline_dir:.4f}  ({baseline_dir*100:.2f}%)")
    print(f"  MAE improvement:   {(1 - mae/baseline_mae)*100:+.1f}%")
    print(f"  Dir Acc edge:      {(dir_acc - baseline_dir)*100:+.2f}%")

    imp = _importance_df(feature_cols, model.feature_importances_)
    _print_importance(imp, "Regressor")

    return model, imp


# ── Helpers ───────────────────────────────────────────────────────────────────

def _importance_df(feature_cols: list[str], importances) -> pd.DataFrame:
    return pd.DataFrame({
        "feature":    feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def _print_importance(imp: pd.DataFrame, label: str, top_n: int = 15):
    print(f"\n  ── Top {top_n} Features ({label}) ──")
    for i, row in imp.head(top_n).iterrows():
        bar = "█" * int(row["importance"] * 400)
        print(f"  {i+1:>2}. {row['feature']:<38} {row['importance']:.4f}  {bar}")


def save_models(clf, clf_imp, reg, reg_imp, out_dir: str, metadata: dict):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    paths = {
        "classifier":     os.path.join(out_dir, f"clf_direction_{ts}.pkl"),
        "regressor":      os.path.join(out_dir, f"reg_return_{ts}.pkl"),
        "clf_importance": os.path.join(out_dir, f"clf_importance_{ts}.csv"),
        "reg_importance": os.path.join(out_dir, f"reg_importance_{ts}.csv"),
        # Fix 6: metadata JSON for reproducibility and inference
        "metadata":       os.path.join(out_dir, f"metadata_{ts}.json"),
    }

    with open(paths["classifier"], "wb") as f: pickle.dump(clf, f)
    with open(paths["regressor"],  "wb") as f: pickle.dump(reg, f)
    clf_imp.to_csv(paths["clf_importance"], index=False)
    reg_imp.to_csv(paths["reg_importance"], index=False)
    with open(paths["metadata"], "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n" + "=" * 65)
    print("  ✅ Models saved")
    print("=" * 65)
    for label, path in paths.items():
        print(f"  {label:<18} {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost classifier + regressor on ml_features")
    parser.add_argument("--db-path",      default=DB_PATH,  help="Path to feature_store.duckdb")
    parser.add_argument("--table",        default="ml_features_clean",
                        help="DuckDB table to train on (default: ml_features_clean — run eda.py first)")
    parser.add_argument("--train-ratio",  type=float, default=0.70, help="Train split ratio (default 0.70)")
    parser.add_argument("--val-ratio",    type=float, default=0.10, help="Val split ratio for early stopping (default 0.10)")
    parser.add_argument("--cv-folds",     type=int,   default=5, help="Walk-forward CV folds (default 5; <2 disables)")
    parser.add_argument("--no-cv",        action="store_true", help="Skip walk-forward cross-validation")
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    # Fix 3: replace brittle assert with a clear ValueError (assert can be
    # disabled with python -O and gives cryptic tracebacks on bad ratios)
    if test_ratio <= 0:
        raise ValueError(
            f"--train-ratio ({args.train_ratio}) + --val-ratio ({args.val_ratio}) "
            f"must sum to < 1.0 (test_ratio = {test_ratio:.3f})"
        )

    if xgb is None:
        raise SystemExit(
            "xgboost is required to train but is not installed in this environment.\n"
            "  Install it:  pip install xgboost\n"
            "  (eda.py and audit_data.py do NOT need xgboost — only training does.)"
        )

    print("Bitcoin 4H Prediction — XGBoost Training")
    print(f"DB:    {args.db_path}")
    print(f"Table: {args.table}")
    print(f"Split: {args.train_ratio:.0%} train / {args.val_ratio:.0%} val (early stop) / {test_ratio:.0%} test\n")

    df           = load_features(args.db_path, args.table)
    feature_cols = get_feature_cols(df)

    # Walk-forward CV first — the honest out-of-sample stability check.
    cv_results = None
    if not args.no_cv:
        cv_results = walk_forward_cv(df, feature_cols, args.cv_folds, args.val_ratio)

    # Then fit the final deployable models on the standard 70/10/20 split.
    clf, clf_imp = train_classifier(df, feature_cols, args.train_ratio, args.val_ratio)
    reg, reg_imp = train_regressor(df,  feature_cols, args.train_ratio, args.val_ratio)

    # Fix 6: build metadata for reproducibility and future inference
    import sklearn, scipy
    metadata = {
        "trained_at":      datetime.now().isoformat(),
        "table":           args.table,
        "db_path":         str(args.db_path),
        "feature_cols":    feature_cols,
        "n_features":      len(feature_cols),
        "n_rows":          len(df),
        "split": {
            "train_ratio": args.train_ratio,
            "val_ratio":   args.val_ratio,
            "test_ratio":  round(test_ratio, 4),
        },
        "date_range": {
            "start": str(pd.to_datetime(df["open_time"].min(), unit="ms")),
            "end":   str(pd.to_datetime(df["open_time"].max(), unit="ms")),
        },
        "clf_params":      CLF_PARAMS,
        "reg_params":      REG_PARAMS,
        "exclude_cols":    sorted(EXCLUDE_COLS),
        "max_nan_pct":     MAX_NAN_PCT,
        "walk_forward_cv": cv_results,
        "versions": {
            "xgboost":   xgb.__version__,
            "pandas":    pd.__version__,
            "numpy":     np.__version__,
            "sklearn":   sklearn.__version__,
        },
    }

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    save_models(clf, clf_imp, reg, reg_imp, out_dir, metadata)


if __name__ == "__main__":
    main()
