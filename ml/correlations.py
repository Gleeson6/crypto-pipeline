"""
Feature Correlation Analysis
=============================
Analyzes which features in ml_features table correlate most strongly with
4H Bitcoin returns. Run this after compute_features.py has populated the table.

Outputs:
  - Terminal: ranked correlation table
  - ml/data/correlations.csv: full results for reference

Usage:
    python3 correlations.py
    python3 correlations.py --target target_return_4h
    python3 correlations.py --target target_direction_4h
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import duckdb
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from setup_db import DB_PATH


# Features to exclude from correlation (non-numeric / target cols)
EXCLUDE_COLS = {
    "open_time",
    "target_return_4h",
    "target_direction_4h",
    "target_return_1h",
    "target_direction_1h",
}


def load_features(db_path: str) -> pd.DataFrame:
    con = duckdb.connect(db_path, read_only=True)
    try:
        # Check table exists
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if "ml_features" not in tables:
            print("ERROR: ml_features table not found. Run compute_features.py first.")
            sys.exit(1)

        count = con.execute("SELECT COUNT(*) FROM ml_features").fetchone()[0]
        print(f"Loading {count:,} rows from ml_features...")

        df = con.execute("SELECT * FROM ml_features ORDER BY open_time").df()
        return df
    finally:
        con.close()


def compute_correlations(df: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in df.columns:
        print(f"ERROR: Target column '{target}' not found in ml_features.")
        print(f"Available targets: {[c for c in df.columns if 'target' in c]}")
        sys.exit(1)

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    results = []
    for col in feature_cols:
        series = df[col].dropna()
        target_series = df.loc[series.index, target].dropna()
        common = series.index.intersection(target_series.index)

        if len(common) < 100:
            continue

        x = series.loc[common]
        y = target_series.loc[common]

        # Pearson correlation
        pearson = float(np.corrcoef(x, y)[0, 1])

        # Spearman (rank) correlation — catches non-linear monotonic relationships
        from scipy.stats import spearmanr
        spearman, p_value = spearmanr(x, y)

        results.append({
            "feature":       col,
            "pearson":       round(pearson, 4),
            "spearman":      round(float(spearman), 4),
            "abs_pearson":   abs(pearson),
            "abs_spearman":  abs(float(spearman)),
            "p_value":       round(float(p_value), 6),
            "n_samples":     len(common),
        })

    df_corr = pd.DataFrame(results)
    df_corr = df_corr.sort_values("abs_spearman", ascending=False).reset_index(drop=True)
    return df_corr


def print_table(df_corr: pd.DataFrame, target: str, top_n: int = 30):
    print(f"\n{'='*70}")
    print(f"  Feature correlations with: {target}")
    print(f"  Sorted by |Spearman| (rank correlation — robust to outliers)")
    print(f"{'='*70}")
    print(f"{'Rank':<5} {'Feature':<35} {'Pearson':>8} {'Spearman':>9} {'p-value':>10}")
    print(f"{'-'*5} {'-'*35} {'-'*8} {'-'*9} {'-'*10}")

    for i, row in df_corr.head(top_n).iterrows():
        p_str = f"{row['p_value']:.2e}" if row['p_value'] < 0.001 else f"{row['p_value']:.4f}"
        sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
        print(f"{i+1:<5} {row['feature']:<35} {row['pearson']:>8.4f} {row['spearman']:>9.4f} {p_str:>10} {sig}")

    print(f"\nShowing top {min(top_n, len(df_corr))} of {len(df_corr)} features")


def group_summary(df_corr: pd.DataFrame):
    """Print average correlation strength by feature group."""
    groups = {
        "Price/OHLCV":    ["open", "high", "low", "close", "volume", "vwap", "rsi", "macd", "bb_", "ema", "sma", "atr"],
        "Funding/OI":     ["funding", "oi_", "open_interest"],
        "Footprint":      ["delta", "buy_vol", "sell_vol", "poc", "vah", "val", "cvd", "imbalance", "large_trade"],
        "Order Flow":     ["bid_ask", "trade_count", "buy_cnt", "sell_cnt", "aggressor"],
    }

    print(f"\n{'='*50}")
    print("  Average |Spearman| by Feature Group")
    print(f"{'='*50}")

    for group_name, keywords in groups.items():
        mask = df_corr["feature"].apply(
            lambda f: any(kw in f.lower() for kw in keywords)
        )
        subset = df_corr[mask]
        if len(subset) == 0:
            continue
        avg = subset["abs_spearman"].mean()
        best = subset.iloc[0]
        print(f"  {group_name:<18} {len(subset):>3} features  avg={avg:.4f}  best: {best['feature']} ({best['spearman']:.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  default="target_return_4h",
                        help="Target column (default: target_return_4h)")
    parser.add_argument("--top",     type=int, default=30,
                        help="Number of top features to display (default: 30)")
    parser.add_argument("--db-path", default=DB_PATH)
    args = parser.parse_args()

    print(f"Feature Correlation Analysis")
    print(f"DB: {args.db_path}")
    print(f"Target: {args.target}\n")

    df = load_features(args.db_path)
    print(f"Loaded. Shape: {df.shape}")
    print(f"Date range: {pd.to_datetime(df['open_time'].min(), unit='ms')} → "
          f"{pd.to_datetime(df['open_time'].max(), unit='ms')}")
    print(f"NaN summary (% missing per feature):")
    nan_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    high_nan = nan_pct[nan_pct > 10]
    if len(high_nan):
        for col, pct in high_nan.items():
            print(f"  {col}: {pct:.1f}% missing")
    else:
        print("  All features < 10% missing ✓")

    df_corr = compute_correlations(df, args.target)
    print_table(df_corr, args.target, top_n=args.top)
    group_summary(df_corr)

    # Save to CSV
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"correlations_{args.target}.csv")
    df_corr.to_csv(out_path, index=False)
    print(f"\n✅ Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
