#!/usr/bin/env python3
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("research_mnq_model_scan")

print("=" * 80)
print("LONG TRADE MODELS — Ranked by ROC-AUC at threshold 0.70")
print("=" * 80)

long_eval = pd.read_csv(OUTPUT_DIR / "model_eval_long.csv")
summary_long = long_eval[
    (long_eval["fold"] == "all") &
    (long_eval["threshold"] == 0.70)
].sort_values("roc_auc", ascending=False)

print(summary_long[
    [
        "model",
        "roc_auc",
        "avg_precision",
        "signals",
        "precision_at_thr",
        "mean_fwd_ret_at_thr",
        "directional_hitrate_at_thr",
    ]
].to_string(index=False))

print("\n" + "=" * 80)
print("SHORT TRADE MODELS — Ranked by ROC-AUC at threshold 0.70")
print("=" * 80)

short_eval = pd.read_csv(OUTPUT_DIR / "model_eval_short.csv")
summary_short = short_eval[
    (short_eval["fold"] == "all") &
    (short_eval["threshold"] == 0.70)
].sort_values("roc_auc", ascending=False)

print(summary_short[
    [
        "model",
        "roc_auc",
        "avg_precision",
        "signals",
        "precision_at_thr",
        "mean_fwd_ret_at_thr",
        "directional_hitrate_at_thr",
    ]
].to_string(index=False))

print("\n" + "=" * 80)
print("TOP FEATURES FOR LONG TRADES (Random Forest importance)")
print("=" * 80)

long_imp = pd.read_csv(OUTPUT_DIR / "feature_importance_long.csv")
print(long_imp.head(15).to_string(index=False))

print("\n" + "=" * 80)
print("TOP FEATURES FOR SHORT TRADES (Random Forest importance)")
print("=" * 80)

short_imp = pd.read_csv(OUTPUT_DIR / "feature_importance_short.csv")
print(short_imp.head(15).to_string(index=False))
