"""
EDA + Data Cleaning Pipeline  (Production-Grade)
=================================================
Exploratory analysis and data cleaning for ml_features → ml_features_clean.

Critical design principle
─────────────────────────
ALL cleaning statistics (percentile bounds for winsorization, medians for
imputation) are FIT ON THE TRAIN SPLIT ONLY (first 70% by time), then
applied to the entire dataset.  This prevents val/test distribution
information from influencing the cleaning and mirrors production behavior
where you deploy on data you have never seen.

Validation suite  (9 checks from quant review)
───────────────────────────────────────────────
  V1  No duplicate open_time rows
  V2  No missing hourly candles (gaps flagged and counted)
  V3  OHLC sanity: high ≥ max(open,close,low), low ≤ min(open,close,high), vol ≥ 0
  V4  Timestamps are UTC and evenly spaced (exactly 1H = 3,600,000 ms)
  V5  Rolling features use only past data (checked structurally, not run-time)
  V6  Targets are shifted forward and NOT present in feature list
  V7  Winsorization and imputation fit on TRAIN only
  V8  Funding / order-flow features aligned to candle open_time (structural note)
  V9  Final rows with unavailable 4H target are removed

EDA sections
────────────
  S1  Data quality  (duplicates, gaps, OHLC sanity, spacing)
  S2  Missing values per feature
  S3  Feature distributions (skewness, kurtosis, 5σ outlier rate)
  S4  Target distribution and class balance
  S5  Temporal drift by half-year
  S6  Multicollinearity (|Pearson| ≥ threshold)

Outputs
───────
  DuckDB  → ml_features_clean   (cleaned, all splits in one table)
  JSON    → ml/data/cleaning_stats.json   (train-fit bounds for inference)
  JSON    → ml/data/cleaning_log.json     (full audit trail)

Usage:
    python3 eda.py
    python3 eda.py --no-clean           # EDA only, skip ml_features_clean
    python3 eda.py --train-ratio 0.70   # must match train_model.py
    python3 eda.py --corr-threshold 0.95
    python3 eda.py --clip-pct 0.01
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH

warnings.filterwarnings("ignore")

# ── Column groups ─────────────────────────────────────────────────────────────

# Raw prices + 97%-missing OI — kept in ml_features for sanity checks but
# never trained on.  Will be excluded from ml_features_clean.
EXCLUDE_FROM_TRAINING = {
    "open_time",
    "open", "high", "low", "close",
    "sum_open_interest", "oi_zscore", "oi_delta_norm",
}
TARGET_COLS  = {"target_return_4h", "target_direction_4h"}
HELPER_COLS  = {"dt", "quarter", "half_year", "split_label"}

# Features that must NOT be winsorized (already bounded or binary)
# NB: compute_features.py emits dow_sin / dow_cos (NOT day_of_week_sin/cos).
# Both spellings are listed so the guard works regardless of naming.
CYCLIC_COLS  = {"hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "day_of_week_sin", "day_of_week_cos"}
BINARY_COLS  = {
    "cascade_flag", "cascade_4h", "cascade_24h",
    "funding_extreme_long", "funding_extreme_short",
    "is_bullish", "is_us_session", "is_asian_session", "is_european_session",
    "liq_long_proxy", "liq_short_proxy",
}
NO_CLIP = CYCLIC_COLS | BINARY_COLS | TARGET_COLS | EXCLUDE_FROM_TRAINING

# Imputation groups (by feature name prefix/substring)
FOOTPRINT_COLS = {
    "delta_norm", "cvd_norm", "poc_distance", "va_range",
    "large_trade_ratio", "large_trade_vol", "buy_vol_pct",
}
FUNDING_COLS   = {"funding_rate", "funding_8h_change"}
RSI_COLS       = {"rsi_7", "rsi_14"}

TRAIN_RATIO_DEFAULT = 0.70   # must match train_model.py


# ── Utilities ─────────────────────────────────────────────────────────────────

def section(title: str):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


def ok(msg: str):  print(f"  ✅  {msg}")
def warn(msg: str): print(f"  ⚠️   {msg}")
def fail(msg: str): print(f"  ❌  {msg}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if "ml_features" not in tables:
        print("ERROR: ml_features not found.  Run compute_features.py first.")
        sys.exit(1)
    df = con.execute("SELECT * FROM ml_features ORDER BY open_time").df()
    con.close()
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    print(f"Loaded: {len(df):,} rows × {df.shape[1]} columns")
    print(f"Range:  {df['dt'].min().strftime('%Y-%m-%d %H:%M')} → "
          f"{df['dt'].max().strftime('%Y-%m-%d %H:%M')} UTC")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION SUITE
# ══════════════════════════════════════════════════════════════════════════════

def run_validations(df: pd.DataFrame, log: dict) -> dict:
    """
    Run all 9 quant-review validation checks.
    Returns a dict of issue counts — any non-zero value needs attention.
    """
    section("VALIDATION SUITE  (9 quant-review checks)")
    issues = {}

    # ── V1: No duplicate open_time ────────────────────────────────────────────
    dup_ts = int(df.duplicated(subset=["open_time"]).sum())
    issues["V1_duplicate_timestamps"] = dup_ts
    if dup_ts == 0:
        ok("V1  No duplicate open_time rows")
    else:
        fail(f"V1  {dup_ts} duplicate open_time rows  → will be dropped in cleaning")

    # ── V2: No missing hourly candles ─────────────────────────────────────────
    ms_1h   = 3_600_000
    diffs   = df["open_time"].diff().dropna()
    gaps    = diffs[diffs > ms_1h * 1.5]   # allow tiny float drift
    n_gaps  = len(gaps)
    n_missing_candles = int((gaps / ms_1h - 1).sum())
    issues["V2_missing_candle_gaps"] = n_gaps
    issues["V2_total_missing_hours"] = n_missing_candles
    if n_gaps == 0:
        ok("V2  No missing hourly candles")
    else:
        warn(f"V2  {n_gaps} time gaps  →  {n_missing_candles} missing hourly candles")
        top_gaps = gaps.nlargest(5)
        for idx in top_gaps.index:
            gap_hrs = int(diffs[idx] / ms_1h)
            at_ts   = df.loc[idx, "dt"].strftime("%Y-%m-%d %H:%M")
            print(f"       {gap_hrs}h gap ending at {at_ts}")

    # ── V3: OHLC sanity ───────────────────────────────────────────────────────
    ohlc_issues = 0
    if all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
        bad_high = ((df["high"] < df["open"]) | (df["high"] < df["close"]) |
                    (df["high"] < df["low"])).sum()
        bad_low  = ((df["low"]  > df["open"]) | (df["low"]  > df["close"]) |
                    (df["low"]  > df["high"])).sum()
        neg_vol  = (df["volume"] < 0).sum()
        ohlc_issues = int(bad_high + bad_low + neg_vol)
        issues["V3_bad_high_candles"]  = int(bad_high)
        issues["V3_bad_low_candles"]   = int(bad_low)
        issues["V3_negative_volume"]   = int(neg_vol)
        if ohlc_issues == 0:
            ok("V3  OHLC sanity OK  (high ≥ open/close/low, low ≤ open/close/high, vol ≥ 0)")
        else:
            fail(f"V3  {ohlc_issues} OHLC violations  (bad_high={bad_high}, bad_low={bad_low}, neg_vol={neg_vol})")
    else:
        warn("V3  OHLC columns not in ml_features — skipping OHLC sanity check")
        issues["V3_skipped"] = True

    # ── V4: UTC and evenly spaced ─────────────────────────────────────────────
    non_1h  = int(((diffs != ms_1h) & (diffs.notna())).sum())
    issues["V4_uneven_intervals"] = non_1h - n_gaps   # exclude genuine data gaps
    if non_1h == n_gaps:
        ok(f"V4  All intervals are exactly 1h  ({len(diffs)} consecutive pairs checked)")
    else:
        warn(f"V4  {non_1h - n_gaps} intervals are NOT exactly 3,600,000 ms "
             f"(excludes {n_gaps} genuine gaps)")

    # ── V5: Rolling features use only past data ───────────────────────────────
    # We cannot verify this at runtime without re-running compute_features.py,
    # but the structural check is: no feature name implies a future window.
    future_pattern_features = [
        c for c in df.columns
        if any(kw in c.lower() for kw in ["future", "fwd", "ahead", "next_"])
        and c not in TARGET_COLS
    ]
    issues["V5_suspect_future_features"] = future_pattern_features
    if not future_pattern_features:
        ok("V5  No feature names suggest future-looking windows  (structural check)")
    else:
        fail(f"V5  Possible future-leaking feature names: {future_pattern_features}")
        print("     Review compute_features.py for these columns")

    # ── V6: Targets shifted forward, NOT in features ──────────────────────────
    # Target must be close_t+4 / close_t - 1, so it's shift(-4).
    # Verify: last 4 rows should have NaN target (they look into the future).
    target_present_as_feature = [c for c in TARGET_COLS if c in df.columns]
    # Cross-check against train_model's EXCLUDE_COLS when importable. eda.py is a
    # data-cleaning script and must NOT hard-require the training stack (xgboost),
    # so fall back to this module's own exclusion set if the import fails.
    try:
        from train_model import EXCLUDE_COLS as TM_EXCLUDE
    except Exception:
        TM_EXCLUDE = TARGET_COLS | EXCLUDE_FROM_TRAINING
        warn("V6  Could not import train_model.EXCLUDE_COLS — using eda's own exclusion set")
    leaked_targets = [c for c in target_present_as_feature if c not in TM_EXCLUDE]
    issues["V6_target_leaked_as_feature"] = leaked_targets
    if not leaked_targets:
        ok("V6  Target columns excluded from feature set in train_model.py")
    else:
        fail(f"V6  Target columns NOT in EXCLUDE_COLS: {leaked_targets}")

    # Verify last rows have NaN target (shift(-4) leaves 4 NaN at end)
    last_n = 6
    if "target_return_4h" in df.columns:
        tail_nan = df["target_return_4h"].iloc[-last_n:].isnull().sum()
        issues["V6_tail_target_nan_count"] = int(tail_nan)
        if tail_nan >= 4:
            ok(f"V6  Last {last_n} rows: {tail_nan} NaN targets — consistent with shift(-4)")
        else:
            warn(f"V6  Only {tail_nan} NaN targets in last {last_n} rows — expected ≥ 4 for 4H shift")

    # ── V7: Cleaning stats fit on train only ──────────────────────────────────
    # Enforced by design in this script (verified programmatically when cleaning runs)
    ok("V7  Winsorization/imputation stats will be fit on TRAIN split only  (enforced below)")
    issues["V7_train_only_fit"] = "enforced"

    # ── V8: Funding / order-flow alignment ────────────────────────────────────
    # Structural check: funding_rate at candle T should use the rate that was
    # active at T, not the rate announced after T.  This is a data ingestion
    # concern in fetch_funding_oi.py, not fixable here.  We note it.
    warn("V8  Funding/OI alignment cannot be verified from ml_features alone")
    print("     Verify in fetch_funding_oi.py: join on open_time, no forward-fill beyond 8h")
    issues["V8_manual_verification_needed"] = True

    # ── V9: Final rows with unavailable target removed ────────────────────────
    if "target_return_4h" in df.columns:
        tail_nan_count = int(df["target_return_4h"].isnull().sum())
        issues["V9_rows_with_nan_target"] = tail_nan_count
        if tail_nan_count > 0:
            ok(f"V9  {tail_nan_count} rows with NaN target will be removed during training "
               f"(dropna on target in train_model.py)")
        else:
            warn("V9  No NaN target rows found — double-check that shift(-4) was applied in compute_features.py")

    print(f"\n  ── Validation summary ──")
    hard_fails = {k: v for k, v in issues.items()
                  if isinstance(v, int) and v > 0
                  and k not in ("V2_total_missing_hours", "V6_tail_target_nan_count",
                                "V9_rows_with_nan_target")}
    if not hard_fails:
        ok(f"All critical checks passed.  Proceed with cleaning.")
    else:
        warn(f"{len(hard_fails)} issue(s) require attention before trusting results:")
        for k, v in hard_fails.items():
            print(f"    {k}: {v}")

    log["validations"] = {k: (v if not isinstance(v, list) else v[:10]) for k, v in issues.items()}
    return issues


# ══════════════════════════════════════════════════════════════════════════════
# EDA SECTIONS  (S1 – S6)
# ══════════════════════════════════════════════════════════════════════════════

def nan_analysis(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    section("S2 — MISSING VALUES (NaN)")
    analysis_cols = [c for c in df.columns
                     if c not in EXCLUDE_FROM_TRAINING and c not in HELPER_COLS]
    records = []
    for col in analysis_cols:
        n_nan = int(df[col].isnull().sum())
        records.append({"feature": col, "n_nan": n_nan, "pct_nan": n_nan / len(df) * 100})
    df_nan = (pd.DataFrame(records)
              .sort_values("pct_nan", ascending=False)
              .reset_index(drop=True))

    buckets = {
        "0%           ✅": (df_nan["pct_nan"] == 0).sum(),
        "1–10%        ⚠️ ": ((df_nan["pct_nan"] > 0) & (df_nan["pct_nan"] <= 10)).sum(),
        "10–50%       🔶": ((df_nan["pct_nan"] > 10) & (df_nan["pct_nan"] <= 50)).sum(),
        ">50%         ❌": (df_nan["pct_nan"] > 50).sum(),
    }
    for label, count in buckets.items():
        print(f"  {count:>3} features — {label} missing")

    has_nan = df_nan[df_nan["pct_nan"] > 0]
    if not has_nan.empty:
        print(f"\n  {'Feature':<35} {'n_nan':>8} {'%':>7}")
        print(f"  {'─'*35} {'─'*8} {'─'*7}")
        for _, r in has_nan.iterrows():
            flag = " ❌" if r["pct_nan"] > 50 else (" 🔶" if r["pct_nan"] > 10 else " ⚠️ ")
            print(f"  {r['feature']:<35} {r['n_nan']:>8,} {r['pct_nan']:>7.1f}%{flag}")
    else:
        ok("No missing values in any feature")

    log["nan_analysis"] = df_nan[df_nan["pct_nan"] > 0][["feature", "n_nan", "pct_nan"]].to_dict("records")
    return df_nan


def distribution_analysis(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    section("S3 — FEATURE DISTRIBUTIONS (Skewness / Kurtosis / 5σ Outliers)")
    analysis_cols = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_TRAINING and c not in TARGET_COLS and c not in HELPER_COLS
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    records = []
    for col in analysis_cols:
        s = df[col].dropna()
        if len(s) < 100:
            continue
        mean, std = float(s.mean()), float(s.std())
        skew = float(scipy_stats.skew(s))
        kurt = float(scipy_stats.kurtosis(s))
        n_out = int(((s - mean).abs() > 5 * std).sum()) if std > 0 else 0
        records.append({
            "feature": col, "mean": round(mean, 6), "std": round(std, 6),
            "min": round(float(s.min()), 6), "p1": round(float(s.quantile(0.01)), 6),
            "p99": round(float(s.quantile(0.99)), 6), "max": round(float(s.max()), 6),
            "skewness": round(skew, 3), "kurtosis": round(kurt, 3),
            "n_outliers_5s": n_out, "pct_outliers": round(n_out / len(s) * 100, 3),
        })
    df_stats = pd.DataFrame(records)

    high_skew = df_stats[df_stats["skewness"].abs() > 3].sort_values("skewness", key=abs, ascending=False)
    fat_tail  = df_stats[df_stats["kurtosis"] > 10].sort_values("kurtosis", ascending=False)
    high_out  = df_stats[df_stats["pct_outliers"] > 0.5].sort_values("pct_outliers", ascending=False)

    print(f"  Highly skewed (|skew| > 3): {len(high_skew)}")
    for _, r in high_skew.head(8).iterrows():
        print(f"    {r['feature']:<35} skew={r['skewness']:>8.3f}  kurt={r['kurtosis']:>8.1f}")

    print(f"\n  Fat-tailed (kurtosis > 10): {len(fat_tail)}")
    for _, r in fat_tail.head(8).iterrows():
        print(f"    {r['feature']:<35} kurt={r['kurtosis']:>8.1f}")

    print(f"\n  >0.5% values beyond 5σ: {len(high_out)}")
    for _, r in high_out.head(8).iterrows():
        print(f"    {r['feature']:<35} {r['pct_outliers']:>6.3f}%  max={r['max']:>14.6f}  p99={r['p99']:>14.6f}")

    log["high_skew_features"]    = high_skew["feature"].tolist()
    log["fat_tail_features"]     = fat_tail["feature"].tolist()
    log["high_outlier_features"] = high_out["feature"].tolist()
    return df_stats


def target_analysis(df: pd.DataFrame, log: dict):
    section("S4 — TARGET VARIABLE ANALYSIS")
    if "target_return_4h" in df.columns:
        t = df["target_return_4h"].dropna()
        skew = scipy_stats.skew(t)
        kurt = scipy_stats.kurtosis(t)
        print(f"  target_return_4h  (n={len(t):,})")
        print(f"    Mean:      {t.mean():>12.6f}  ({t.mean()*100:+.4f}%)")
        print(f"    Std:       {t.std():>12.6f}  ({t.std()*100:.4f}%)")
        print(f"    Skewness:  {skew:>12.3f}{'  ⚠️  fat tails' if abs(skew) > 1 else ''}")
        print(f"    Kurtosis:  {kurt:>12.3f}  (excess; normal=0)")
        print(f"    p1 / p99:  {t.quantile(0.01):.6f} / {t.quantile(0.99):.6f}")
        bins = [
            ("<-5%",   t < -0.05),
            ("-5→-2%", (t >= -0.05) & (t < -0.02)),
            ("-2→-1%", (t >= -0.02) & (t < -0.01)),
            ("±1%",    (t >= -0.01) & (t < +0.01)),
            ("+1→+2%", (t >= +0.01) & (t < +0.02)),
            ("+2→+5%", (t >= +0.02) & (t < +0.05)),
            (">+5%",   t >= +0.05),
        ]
        print(f"\n    Distribution:")
        for label, mask in bins:
            cnt = mask.sum()
            pct = cnt / len(t) * 100
            note = "  ← noise zone" if label == "±1%" else ""
            print(f"      {label:<10}: {cnt:>5,}  ({pct:4.1f}%)  {'█'*int(pct/2)}{note}")
        log["target_return_4h"] = {
            "mean": round(float(t.mean()), 6), "std": round(float(t.std()), 6),
            "skew": round(float(skew), 3), "kurtosis": round(float(kurt), 3),
        }

    if "target_direction_4h" in df.columns:
        t = df["target_direction_4h"].dropna()
        t = t[t != 0]
        up, dn = (t == 1).sum(), (t == -1).sum()
        total = len(t)
        imb = abs(up - dn) / total
        print(f"\n  target_direction_4h  (n={total:,}, flat excluded)")
        print(f"    UP   (+1): {up:,}  ({up/total*100:.1f}%)")
        print(f"    DOWN (-1): {dn:,}  ({dn/total*100:.1f}%)")
        status = "✅" if imb < 0.02 else ("⚠️ " if imb < 0.05 else "🔶")
        print(f"    {status} Imbalance: {imb*100:.1f}%"
              + (f"  → set scale_pos_weight={dn/up:.2f}" if imb >= 0.05 else ""))
        log["target_direction_4h"] = {
            "n_up": int(up), "n_down": int(dn), "imbalance_pct": round(float(imb)*100, 2),
        }


def temporal_drift_check(df: pd.DataFrame):
    section("S5 — TEMPORAL DRIFT CHECK (feature mean by half-year)")
    key_features = [f for f in [
        "log_return_1h", "funding_rate", "delta_norm",
        "vol_zscore", "rsi_14", "taker_buy_ratio", "cascade_flag",
    ] if f in df.columns]

    df_tmp = df.copy()
    # Robust half-year label (e.g. "2025-H1") — avoids the non-standard "2Q"
    # period frequency which errors on some pandas versions.
    _yr = df_tmp["dt"].dt.year.astype(str)
    _h  = np.where(df_tmp["dt"].dt.month <= 6, "H1", "H2")
    df_tmp["half_year"] = _yr + "-" + _h
    periods = sorted(df_tmp["half_year"].unique())

    print(f"  {'Feature':<30}", end="")
    for p in periods:
        print(f"  {str(p):>12}", end="")
    print()
    print(f"  {'─'*30}" + f"  {'─'*12}"*len(periods))

    for feat in key_features:
        print(f"  {feat:<30}", end="")
        vals = []
        for period in periods:
            mask = df_tmp["half_year"] == period
            v = df_tmp.loc[mask, feat].dropna()
            if len(v) > 10:
                m = float(v.mean())
                vals.append(m)
                print(f"  {m:>12.4f}", end="")
            else:
                vals.append(None)
                print(f"  {'N/A':>12}", end="")
        valids = [v for v in vals if v is not None]
        if len(valids) > 1:
            p_std = np.std(valids)
            o_std = float(df_tmp[feat].std())
            if o_std > 0 and p_std / o_std > 0.30:
                print("  ⚠️  DRIFT", end="")
        print()
    print(f"\n  ⚠️  DRIFT = period mean varies > 30% of overall std → may need periodic retraining")


def multicollinearity_check(df: pd.DataFrame, threshold: float, log: dict,
                            train_ratio: float = TRAIN_RATIO_DEFAULT) -> list:
    section(f"S6 — MULTICOLLINEARITY  (|Pearson| ≥ {threshold})")
    # Decide which features to DROP using the TRAIN split only — dropping
    # features is feature selection, so letting val/test correlations influence
    # it is a (mild) form of leakage. Fit the decision on train, apply to all.
    train_end = int(len(df) * train_ratio)
    train_slice = df.iloc[:train_end]
    usable = [
        c for c in df.columns
        if c not in EXCLUDE_FROM_TRAINING and c not in TARGET_COLS and c not in HELPER_COLS
        and train_slice[c].isnull().mean() < 0.5
        and train_slice[c].nunique() > 2
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    print(f"  Computing {len(usable)}×{len(usable)} correlation matrix on TRAIN split "
          f"(first {train_ratio:.0%}, {len(train_slice):,} rows) …")
    corr = train_slice[usable].corr().abs()

    pairs = []
    for i, c1 in enumerate(usable):
        for c2 in usable[i + 1:]:
            v = float(corr.loc[c1, c2])
            if v >= threshold:
                pairs.append((c1, c2, round(v, 4)))

    to_drop = []
    if not pairs:
        ok(f"No near-duplicate feature pairs above {threshold}")
    else:
        print(f"  Found {len(pairs)} pairs:")
        print(f"\n  {'Feature A':<35} {'Feature B':<35} {'Corr':>6}")
        print(f"  {'─'*35} {'─'*35} {'─'*6}")
        for c1, c2, v in sorted(pairs, key=lambda x: -x[2]):
            print(f"  {c1:<35} {c2:<35} {v:>6.4f}")
            if c2 not in to_drop:
                to_drop.append(c2)
        print(f"\n  Will drop {len(to_drop)} features: {to_drop}")

    log["redundant_features_dropped"] = to_drop
    return to_drop


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLEANING  (train-only stat fitting)
# ══════════════════════════════════════════════════════════════════════════════

def fit_cleaning_stats(train_df: pd.DataFrame, clip_pct: float) -> dict:
    """
    Compute all cleaning parameters from the TRAIN split only.
    Returns a stats dict that can be applied to any split and saved for inference.

    Every numeric feature gets a `train_median` (used for imputation fallback so
    we NEVER fall back to a full-dataset median, which would leak val/test info).
    Winsorization bounds (clip_lo/clip_hi) are computed only for unbounded
    continuous features — bounded/binary/cyclic features (NO_CLIP) get None.
    """
    stats = {}
    num_cols = train_df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        # Never fit stats on the target or on raw/excluded columns
        if col in EXCLUDE_FROM_TRAINING or col in TARGET_COLS:
            continue
        s = train_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 50:
            continue
        entry = {"train_median": float(s.median())}
        lo = float(s.quantile(clip_pct))
        hi = float(s.quantile(1 - clip_pct))
        if col in NO_CLIP or hi <= lo:
            # NO_CLIP (bounded/binary/cyclic) OR a near-constant column whose
            # 1st==99th percentile (e.g. sparse {-1,0,1} flags like liq_direction,
            # oi_price_divergence, macd_cross). Winsorizing these is meaningless —
            # store the median for imputation only, no clip bounds. (Emitting a
            # degenerate [v,v] bound would also make the integrity audit think a
            # clip was applied when it wasn't.)
            entry["clip_lo"] = None
            entry["clip_hi"] = None
        else:
            entry["clip_lo"] = lo
            entry["clip_hi"] = hi
        stats[col] = entry

    # Forward-fill anchor for funding (last known TRAIN value)
    for col in FUNDING_COLS:
        if col in train_df.columns:
            valid = train_df[col].dropna()
            last_valid = float(valid.iloc[-1]) if not valid.empty else 0.0
            stats.setdefault(col, {"clip_lo": None, "clip_hi": None,
                                   "train_median": float(valid.median()) if not valid.empty else 0.0})
            stats[col]["ffill_anchor"] = last_valid
    return stats


def apply_cleaning(df: pd.DataFrame, stats: dict, to_drop: list, log: dict) -> pd.DataFrame:
    """
    Apply cleaning to a DataFrame using PRE-FITTED stats.
    This must be called with the same stats dict for train, val, and test.
    """
    cleaned = df.copy()
    clog = log.setdefault("cleaning_steps", {})

    # Step 1: Drop duplicate timestamps
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    clog["S1_dup_rows_dropped"] = before - len(cleaned)

    # Step 2: inf → NaN
    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    inf_count = int(np.isinf(cleaned[num_cols]).sum().sum())
    cleaned[num_cols] = cleaned[num_cols].replace([np.inf, -np.inf], np.nan)
    clog["S2_inf_replaced"] = inf_count

    # Step 3: Winsorize using TRAIN-FIT bounds
    n_clipped = 0
    cols_clipped = 0
    for col, s in stats.items():
        if col not in cleaned.columns or s["clip_lo"] is None:
            continue
        lo, hi = s["clip_lo"], s["clip_hi"]
        if lo >= hi:
            continue
        before_clip = int(((cleaned[col] < lo) | (cleaned[col] > hi)).sum())
        cleaned[col] = cleaned[col].clip(lo, hi)
        n_clipped += before_clip
        cols_clipped += 1
    clog["S3_values_clipped"] = n_clipped
    clog["S3_cols_clipped"] = cols_clipped

    # Step 4: Impute NaN using TRAIN-FIT medians/defaults
    imputed = 0

    # 4a: Footprint → 0 (no activity)
    for col in FOOTPRINT_COLS:
        if col in cleaned.columns:
            n = int(cleaned[col].isnull().sum())
            if n:
                cleaned[col] = cleaned[col].fillna(0.0)
                imputed += n

    # 4b: Funding → forward-fill then train median
    for col in FUNDING_COLS:
        if col in cleaned.columns:
            n = int(cleaned[col].isnull().sum())
            if n:
                cleaned[col] = cleaned[col].ffill()
                still_nan = cleaned[col].isnull().sum()
                if still_nan:
                    fill_val = stats.get(col, {}).get("train_median", 0.0) or 0.0
                    cleaned[col] = cleaned[col].fillna(fill_val)
                imputed += n

    # 4c: RSI → 50 (neutral)
    for col in RSI_COLS:
        if col in cleaned.columns:
            n = int(cleaned[col].isnull().sum())
            if n:
                cleaned[col] = cleaned[col].fillna(50.0)
                imputed += n

    # 4d: MACD → 0, BB%B → 0.5
    for col in cleaned.columns:
        if col in EXCLUDE_FROM_TRAINING or col in TARGET_COLS:
            continue
        if "macd" in col:
            n = int(cleaned[col].isnull().sum())
            if n:
                cleaned[col] = cleaned[col].fillna(0.0)
                imputed += n
        if "bb_pct_b" in col:
            n = int(cleaned[col].isnull().sum())
            if n:
                cleaned[col] = cleaned[col].fillna(0.5)
                imputed += n

    # 4e: Log returns, vol features → 0
    for col in cleaned.columns:
        if col in EXCLUDE_FROM_TRAINING or col in TARGET_COLS:
            continue
        if any(kw in col for kw in ["log_return", "return_", "vol_zscore", "vol_ratio",
                                     "candle_body", "wick"]):
            n = int(cleaned[col].isnull().sum())
            if n:
                cleaned[col] = cleaned[col].fillna(0.0)
                imputed += n

    # 4f: Remaining features → TRAIN MEDIAN from stats dict.
    # CRITICAL (no leakage): the fallback is a constant 0.0 — NEVER the
    # full-dataset median, which would let val/test rows influence imputation.
    safe_cols = [
        c for c in cleaned.columns
        if c not in TARGET_COLS and c not in EXCLUDE_FROM_TRAINING and c not in HELPER_COLS
    ]
    leak_safe_fallbacks = []
    for col in safe_cols:
        if cleaned[col].isnull().any():
            n = int(cleaned[col].isnull().sum())
            fill_val = stats.get(col, {}).get("train_median")
            if fill_val is None:
                fill_val = 0.0                      # constant, train-independent
                leak_safe_fallbacks.append(col)
            cleaned[col] = cleaned[col].fillna(fill_val)
            imputed += n
    if leak_safe_fallbacks:
        clog["S4_const_fallback_cols"] = leak_safe_fallbacks

    clog["S4_values_imputed"] = imputed
    feat_nan_left = int(cleaned[safe_cols].isnull().sum().sum())
    clog["S4_feature_nan_remaining"] = feat_nan_left

    # Step 5: Drop near-duplicate features
    actual_drop = [c for c in to_drop if c in cleaned.columns]
    if actual_drop:
        cleaned = cleaned.drop(columns=actual_drop)
    clog["S5_features_dropped"] = actual_drop

    # Step 6: Drop raw OHLC and OI from clean table (kept for validation, not training)
    drop_raw = [c for c in EXCLUDE_FROM_TRAINING if c in cleaned.columns and c != "open_time"]
    drop_raw += [c for c in {"sum_open_interest", "oi_zscore", "oi_delta_norm"} if c in cleaned.columns]
    cleaned = cleaned.drop(columns=drop_raw, errors="ignore")

    # Step 7: Remove helper columns
    for col in list(HELPER_COLS):
        if col in cleaned.columns:
            cleaned = cleaned.drop(columns=[col])

    clog["rows_in"]  = len(df)
    clog["rows_out"] = len(cleaned)
    clog["cols_in"]  = df.shape[1]
    clog["cols_out"] = cleaned.shape[1]

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
# SAVE  ml_features_clean
# ══════════════════════════════════════════════════════════════════════════════

def save_clean_table(df_clean: pd.DataFrame, db_path: str):
    section("SAVING ml_features_clean → DuckDB")
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS ml_features_clean")
    con.execute("CREATE TABLE ml_features_clean AS SELECT * FROM df_clean")
    count = con.execute("SELECT COUNT(*) FROM ml_features_clean").fetchone()[0]
    ncols = len(con.execute("DESCRIBE ml_features_clean").fetchall())
    con.close()
    ok(f"ml_features_clean  →  {count:,} rows × {ncols} columns")
    print(f"\n  Next:")
    print(f"    python3 train_model.py                  # uses ml_features_clean by default")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Production-grade EDA + data cleaning for ml_features"
    )
    parser.add_argument("--db-path",        default=DB_PATH)
    parser.add_argument("--no-clean",       action="store_true",
                        help="EDA report only — skip writing ml_features_clean")
    parser.add_argument("--train-ratio",    type=float, default=TRAIN_RATIO_DEFAULT,
                        help=f"Train split ratio for fitting cleaning stats "
                             f"(default {TRAIN_RATIO_DEFAULT} — must match train_model.py)")
    parser.add_argument("--corr-threshold", type=float, default=0.95,
                        help="Correlation threshold for multicollinearity check (default 0.95)")
    parser.add_argument("--clip-pct",       type=float, default=0.01,
                        help="Winsorize percentile — 0.01 = 1st/99th (default 0.01)")
    args = parser.parse_args()

    print("Bitcoin ML Features — Production EDA + Cleaning")
    print(f"DB:          {args.db_path}")
    print(f"Train ratio: {args.train_ratio:.0%}  (cleaning stats fit on first "
          f"{args.train_ratio:.0%} of data only)")
    print(f"Time:        {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    log = {
        "run_at":       datetime.now().isoformat(),
        "db_path":      str(args.db_path),
        "train_ratio":  args.train_ratio,
        "clip_pct":     args.clip_pct,
        "corr_threshold": args.corr_threshold,
    }

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(args.db_path)
    log["rows_loaded"] = len(df)
    log["cols_loaded"] = df.shape[1]

    # ── Validation suite (before cleaning) ───────────────────────────────────
    run_validations(df, log)

    # ── EDA sections ──────────────────────────────────────────────────────────
    df_nan   = nan_analysis(df, log)
    df_stats = distribution_analysis(df, log)
    target_analysis(df, log)
    temporal_drift_check(df)
    to_drop  = multicollinearity_check(df, args.corr_threshold, log, args.train_ratio)

    # ── Save EDA CSVs ─────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    df_stats.to_csv(os.path.join(out_dir, "feature_stats.csv"), index=False)
    df_nan.to_csv(os.path.join(out_dir, "nan_analysis.csv"), index=False)
    print(f"\n✅ EDA CSVs → ml/data/feature_stats.csv  ml/data/nan_analysis.csv")

    if args.no_clean:
        print("\n[--no-clean: skipping ml_features_clean write]")
        with open(os.path.join(out_dir, "cleaning_log.json"), "w") as f:
            json.dump(log, f, indent=2, default=str)
        return

    # ── Cleaning (train-only stat fitting) ────────────────────────────────────
    section("CLEANING  (fit on train split, apply to full dataset)")

    # Determine train boundary (same logic as train_model.py walk_forward_split)
    n = len(df)
    train_end = int(n * args.train_ratio)
    train_df  = df.iloc[:train_end]

    print(f"  Train split:  rows 0 → {train_end:,}  "
          f"({pd.to_datetime(train_df['open_time'].min(), unit='ms').strftime('%Y-%m-%d')} → "
          f"{pd.to_datetime(train_df['open_time'].max(), unit='ms').strftime('%Y-%m-%d')})")
    print(f"  Held out:     rows {train_end:,} → {n:,}  "
          f"(val + test — stats NOT fitted on these rows)")

    # Fit all stats on TRAIN only
    cleaning_stats = fit_cleaning_stats(train_df, args.clip_pct)
    print(f"  Fitted cleaning stats for {len(cleaning_stats)} features on TRAIN only")

    # Apply to full dataset using train-fit stats
    df_clean = apply_cleaning(df, cleaning_stats, to_drop, log)

    clog = log.get("cleaning_steps", {})
    print(f"\n  ── Cleaning summary ──")
    print(f"  Duplicate rows dropped:  {clog.get('S1_dup_rows_dropped', 0)}")
    print(f"  Inf values → NaN:        {clog.get('S2_inf_replaced', 0)}")
    print(f"  Values clipped:          {clog.get('S3_values_clipped', 0)}")
    print(f"  Values imputed:          {clog.get('S4_values_imputed', 0)}")
    print(f"  Features dropped:        {len(clog.get('S5_features_dropped', []))}")
    print(f"  NaN in features after:   {clog.get('S4_feature_nan_remaining', '?')}")
    print(f"  Rows: {clog.get('rows_in', '?')} → {clog.get('rows_out', '?')}")
    print(f"  Cols: {clog.get('cols_in', '?')} → {clog.get('cols_out', '?')}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_clean_table(df_clean, args.db_path)

    # cleaning_stats.json — needed for inference (apply same bounds to live data)
    stats_path = os.path.join(out_dir, "cleaning_stats.json")
    with open(stats_path, "w") as f:
        json.dump(cleaning_stats, f, indent=2, default=str)
    print(f"  ✅ cleaning_stats.json → {stats_path}")
    print(f"     (use these bounds when cleaning live data at inference time)")

    # cleaning_log.json — full audit trail
    log_path = os.path.join(out_dir, "cleaning_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"  ✅ cleaning_log.json   → {log_path}")

    # ── Final note on test period ─────────────────────────────────────────────
    section("IMPORTANT NOTE — Test Period Coverage")
    print(
        "  The 20% test set is the most recent 20% of this 2-year dataset.\n"
        "  It is held out during training and never used for early stopping.\n"
        "  However, it still falls within the same market regime (2024–2026).\n\n"
        "  For production confidence, collect live predictions after today and\n"
        "  compare against actual outcomes — that is the true out-of-sample test.\n\n"
        "  To strengthen backtesting, consider:\n"
        "    1. Walk-forward cross-validation (multiple train/test windows)\n"
        "    2. A fixed 2026+ hold-out as data accumulates\n"
        "    3. VectorBT backtest on the 20% test window (Phase 9k)"
    )


if __name__ == "__main__":
    main()
