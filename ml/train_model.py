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
import os
import sys
import pickle
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
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


def load_features(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "ml_features" not in tables:
        print("ERROR: ml_features table not found. Run compute_features.py first.")
        sys.exit(1)
    df = con.execute("SELECT * FROM ml_features ORDER BY open_time").df()
    con.close()
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
    n = len(df)
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end].copy()
    val   = df.iloc[train_end:val_end].copy()
    test  = df.iloc[val_end:].copy()

    # Sanity checks — no time overlap
    assert train.index[-1] < val.index[0],  "Train/val overlap detected"
    assert val.index[-1]   < test.index[0], "Val/test overlap detected"

    return train, val, test


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
    # too noisy for val loss to improve. Use a small fixed n_estimators instead.
    n_estimators=150,
    max_depth=3,
    learning_rate=0.02,    # slower learning = more stable on noisy target
    subsample=0.7,
    colsample_bytree=0.6,
    min_child_weight=50,   # high — very conservative on financial noise
    gamma=3,
    reg_alpha=1.0,
    reg_lambda=3.0,
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

    model = xgb.XGBClassifier(**CLF_PARAMS)
    # ✅ Early stopping uses VAL set — test set is never touched during training
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)

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

    y_pred = model.predict(X_test)

    mae     = mean_absolute_error(y_test, y_pred)
    rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
    r2      = r2_score(y_test, y_pred)
    dir_acc = (np.sign(y_pred) == np.sign(y_test)).mean()

    # Baseline: always predict the mean return
    baseline_pred = np.full(len(y_test), y_train.mean())
    baseline_mae  = mean_absolute_error(y_test, baseline_pred)
    baseline_dir  = (np.sign(baseline_pred) == np.sign(y_test)).mean()

    print(f"\n  ── Results ──")
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


def save_models(clf, clf_imp, reg, reg_imp, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    paths = {
        "classifier":      os.path.join(out_dir, f"clf_direction_{ts}.pkl"),
        "regressor":       os.path.join(out_dir, f"reg_return_{ts}.pkl"),
        "clf_importance":  os.path.join(out_dir, f"clf_importance_{ts}.csv"),
        "reg_importance":  os.path.join(out_dir, f"reg_importance_{ts}.csv"),
    }

    with open(paths["classifier"], "wb") as f: pickle.dump(clf, f)
    with open(paths["regressor"],  "wb") as f: pickle.dump(reg, f)
    clf_imp.to_csv(paths["clf_importance"], index=False)
    reg_imp.to_csv(paths["reg_importance"], index=False)

    print("\n" + "=" * 65)
    print("  ✅ Models saved")
    print("=" * 65)
    for label, path in paths.items():
        print(f"  {label:<18} {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost classifier + regressor on ml_features")
    parser.add_argument("--db-path",      default=DB_PATH,  help="Path to feature_store.duckdb")
    parser.add_argument("--train-ratio",  type=float, default=0.70, help="Train split ratio (default 0.70)")
    parser.add_argument("--val-ratio",    type=float, default=0.10, help="Val split ratio for early stopping (default 0.10)")
    args = parser.parse_args()

    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    assert test_ratio > 0, "train-ratio + val-ratio must be < 1.0"

    print("Bitcoin 4H Prediction — XGBoost Training")
    print(f"DB:    {args.db_path}")
    print(f"Split: {args.train_ratio:.0%} train / {args.val_ratio:.0%} val (early stop) / {test_ratio:.0%} test\n")

    df           = load_features(args.db_path)
    feature_cols = get_feature_cols(df)

    clf, clf_imp = train_classifier(df, feature_cols, args.train_ratio, args.val_ratio)
    reg, reg_imp = train_regressor(df,  feature_cols, args.train_ratio, args.val_ratio)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    save_models(clf, clf_imp, reg, reg_imp, out_dir)


if __name__ == "__main__":
    main()
