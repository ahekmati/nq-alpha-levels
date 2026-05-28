import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- CONFIG ---------------- #
BASE_DIR = Path("research_mnq_bear_model")
H1_FILE = BASE_DIR / "early_entry_diagnostics_v2" / "h1_with_relaxed_early_signals.csv"
OUTPUT_DIR = BASE_DIR / "signal_debug_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TOP_N_NEAR_MISSES = 40

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_MAGENTA = "\033[95m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"
# ---------------------------------------- #


def color(text, c):
    return f"{c}{text}{ANSI_RESET}"


def section(title):
    print()
    print(color(f"{'=' * 12} {title} {'=' * 12}", ANSI_BOLD + ANSI_CYAN))


def fmt(x, digits=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def load_data():
    if not H1_FILE.exists():
        raise FileNotFoundError(f"Missing {H1_FILE}")

    df = pd.read_csv(H1_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def summarize_feature_health(df: pd.DataFrame, cols):
    rows = []
    for c in cols:
        if c not in df.columns:
            rows.append({
                "column": c,
                "exists": 0,
                "nonnull_pct": np.nan,
                "min": np.nan,
                "p25": np.nan,
                "median": np.nan,
                "p75": np.nan,
                "max": np.nan,
                "nunique": np.nan,
            })
            continue

        s = df[c]
        numeric = pd.to_numeric(s, errors="coerce")
        rows.append({
            "column": c,
            "exists": 1,
            "nonnull_pct": 100.0 * s.notna().mean(),
            "min": numeric.min(),
            "p25": numeric.quantile(0.25) if numeric.notna().any() else np.nan,
            "median": numeric.median() if numeric.notna().any() else np.nan,
            "p75": numeric.quantile(0.75) if numeric.notna().any() else np.nan,
            "max": numeric.max(),
            "nunique": s.nunique(dropna=True),
        })
    return pd.DataFrame(rows)


def boolify(series):
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(0).astype(float).astype(int) == 1


def condition_counts(df: pd.DataFrame, conds):
    rows = []
    n = len(df)
    for c in conds:
        if c not in df.columns:
            rows.append({"condition": c, "passes": np.nan, "pass_rate_pct": np.nan})
            continue
        b = boolify(df[c])
        rows.append({
            "condition": c,
            "passes": int(b.sum()),
            "pass_rate_pct": 100.0 * b.mean() if n > 0 else np.nan
        })
    return pd.DataFrame(rows)


def cumulative_filter_counts(df: pd.DataFrame, cond_sequence, label):
    rows = []
    current = pd.Series(True, index=df.index)

    for i, c in enumerate(cond_sequence, 1):
        if c not in df.columns:
            rows.append({
                "pipeline": label,
                "step": i,
                "condition": c,
                "remaining_rows": np.nan,
                "remaining_pct": np.nan
            })
            break

        current = current & boolify(df[c])
        rows.append({
            "pipeline": label,
            "step": i,
            "condition": c,
            "remaining_rows": int(current.sum()),
            "remaining_pct": 100.0 * current.mean()
        })

    return pd.DataFrame(rows), current


def pairwise_intersections(df: pd.DataFrame, conds):
    rows = []
    for i, c1 in enumerate(conds):
        for c2 in conds[i + 1:]:
            if c1 not in df.columns or c2 not in df.columns:
                continue
            b1 = boolify(df[c1])
            b2 = boolify(df[c2])
            both = b1 & b2
            rows.append({
                "cond_1": c1,
                "cond_2": c2,
                "both_pass": int(both.sum()),
                "both_pass_pct": 100.0 * both.mean()
            })
    return pd.DataFrame(rows).sort_values("both_pass", ascending=False)


def near_miss_table(df: pd.DataFrame, cond_sequence, signal_name):
    work = df.copy()

    present_conds = [c for c in cond_sequence if c in work.columns]
    if not present_conds:
        return pd.DataFrame()

    for c in present_conds:
        work[f"pass_{c}"] = boolify(work[c]).astype(int)

    work["conditions_passed"] = work[[f"pass_{c}" for c in present_conds]].sum(axis=1)
    work["conditions_total"] = len(present_conds)
    work["conditions_failed"] = work["conditions_total"] - work["conditions_passed"]

    near = work[work["conditions_failed"] > 0].copy()
    near = near.sort_values(
        ["conditions_failed", "conditions_passed", "time"],
        ascending=[True, False, True]
    ).head(TOP_N_NEAR_MISSES)

    fail_cols = []
    for c in present_conds:
        fail_col = f"fail_{c}"
        near[fail_col] = np.where(near[f"pass_{c}"] == 0, 1, 0)
        fail_cols.append(fail_col)

    base_cols = [
        "time", "bear_window_id", "hours_from_bear_start",
        "conditions_passed", "conditions_total", "conditions_failed",
        "bear_prob", "exhaust_prob", "daily_model_rsi", "daily_model_adx",
        "daily_model_ema20_slope_5", "daily_model_ema50_slope_5",
        "daily_model_atr", "rsi70_streak", "ob_streak_not_making_higher_high",
        "rsi_rolling_over", "range_pos", "hh_8", "high", "close", "open",
        "h1_rsi", "h1_ema_20", "h1_ret_4"
    ]
    out_cols = [c for c in base_cols if c in near.columns] + fail_cols
    near["signal_name"] = signal_name
    out_cols = ["signal_name"] + out_cols
    return near[out_cols]


def threshold_probe(df: pd.DataFrame):
    rows = []

    probes = [
        ("range_pos_ge_0_60", df["range_pos"] >= 0.60),
        ("range_pos_ge_0_65", df["range_pos"] >= 0.65),
        ("range_pos_ge_0_68", df["range_pos"] >= 0.68),
        ("range_pos_ge_0_70", df["range_pos"] >= 0.70),
        ("high_ge_hh8", df["high"] >= df["hh_8"]),
        ("close_lt_open", df["close"] < df["open"]),
        ("h1_rsi_45_72", (df["h1_rsi"] >= 45) & (df["h1_rsi"] <= 72)),
        ("bear_prob_ge_0_35", df["bear_prob"] >= 0.35),
        ("bear_prob_ge_0_45", df["bear_prob"] >= 0.45),
        ("exhaust_prob_le_0_55", df["exhaust_prob"] <= 0.55),
        ("daily_rsi_38_62", (df["daily_model_rsi"] >= 38) & (df["daily_model_rsi"] <= 62)),
        ("daily_adx_ge_12", df["daily_model_adx"] >= 12),
        ("rsi70_streak_ge_2", df["rsi70_streak"].fillna(0) >= 2),
        ("overbought_stall", (df["ob_streak_not_making_higher_high"] == 1) | (df["rsi_rolling_over"] == 1)),
        ("close_back_below_ema20", (df["high"] >= df["h1_ema_20"]) & (df["close"] < df["h1_ema_20"])),
        ("h1_ret_4_gt_0", df["h1_ret_4"] > 0),
    ]

    n = len(df)
    for name, cond in probes:
        b = cond.fillna(False)
        rows.append({
            "probe": name,
            "passes": int(b.sum()),
            "pass_rate_pct": 100.0 * b.mean() if n > 0 else np.nan
        })

    return pd.DataFrame(rows).sort_values("passes", ascending=False)


def print_table(df: pd.DataFrame, title: str, head=None):
    section(title)
    if df.empty:
        print(color("No rows.", ANSI_RED))
        return
    if head is not None:
        df = df.head(head)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


def main():
    df = load_data()
    early = df[df["is_early_window"].fillna(0).astype(int) == 1].copy()

    section("Dataset")
    print(color(f"Rows total: {len(df)}", ANSI_YELLOW))
    print(color(f"Early-window rows: {len(early)}", ANSI_YELLOW))

    feature_cols = [
        "bear_prob", "exhaust_prob", "daily_model_rsi", "daily_model_adx",
        "daily_model_ema20_slope_5", "daily_model_ema50_slope_5", "daily_model_atr",
        "range_pos", "hh_8", "high", "close", "open", "h1_rsi", "h1_ema_20",
        "h1_ret_4", "rsi70_streak", "ob_streak_not_making_higher_high", "rsi_rolling_over"
    ]
    feature_health = summarize_feature_health(early, feature_cols)

    all_conditions = [
        "in_session",
        "soft_daily_weakening",
        "soft_slope_condition",
        "overbought_stall_condition",
        "upper_range_failure",
        "close_back_below_ema20_fail",
        "near_end_of_avg_ob_streak",
        "early_relaxed_signal_A",
        "early_relaxed_signal_B",
        "early_relaxed_signal_C",
        "first_rally_short_relaxed",
    ]
    condition_tbl = condition_counts(early, all_conditions)
    probe_tbl = threshold_probe(early)

    seq_A = [
        "is_early_window",
        "in_session",
        "upper_range_failure",
        "soft_daily_weakening",
        "soft_slope_condition",
    ]
    seq_B = [
        "is_early_window",
        "in_session",
        "upper_range_failure",
        "overbought_stall_condition",
    ]
    seq_C = [
        "is_early_window",
        "in_session",
        "upper_range_failure",
        "near_end_of_avg_ob_streak",
        "close_back_below_ema20_fail",
    ]
    seq_R = [
        "in_session",
        "hours_from_bear_start_ge_12",
        "hours_from_bear_start_le_144",
        "bear_prob_ge_0_45",
        "exhaust_prob_le_0_50",
        "daily_rsi_32_55",
        "daily_adx_ge_15",
        "high_ge_hh8",
        "close_lt_open",
        "close_lt_h1_ema20",
        "h1_ret_4_gt_0",
    ]

    early["hours_from_bear_start_ge_12"] = (early["hours_from_bear_start"] >= 12).astype(int)
    early["hours_from_bear_start_le_144"] = (early["hours_from_bear_start"] <= 144).astype(int)
    early["bear_prob_ge_0_45"] = (early["bear_prob"] >= 0.45).astype(int)
    early["exhaust_prob_le_0_50"] = (early["exhaust_prob"] <= 0.50).astype(int)
    early["daily_rsi_32_55"] = ((early["daily_model_rsi"] >= 32) & (early["daily_model_rsi"] <= 55)).astype(int)
    early["daily_adx_ge_15"] = (early["daily_model_adx"] >= 15).astype(int)
    early["high_ge_hh8"] = (early["high"] >= early["hh_8"]).astype(int)
    early["close_lt_open"] = (early["close"] < early["open"]).astype(int)
    early["close_lt_h1_ema20"] = (early["close"] < early["h1_ema_20"]).astype(int)
    early["h1_ret_4_gt_0"] = (early["h1_ret_4"] > 0).astype(int)

    cum_A, mask_A = cumulative_filter_counts(early, seq_A, "signal_A_pipeline")
    cum_B, mask_B = cumulative_filter_counts(early, seq_B, "signal_B_pipeline")
    cum_C, mask_C = cumulative_filter_counts(early, seq_C, "signal_C_pipeline")
    cum_R, mask_R = cumulative_filter_counts(early, seq_R, "first_rally_pipeline")

    pair_tbl = pairwise_intersections(
        early,
        [
            "in_session",
            "upper_range_failure",
            "soft_daily_weakening",
            "soft_slope_condition",
            "overbought_stall_condition",
            "near_end_of_avg_ob_streak",
            "close_back_below_ema20_fail",
            "high_ge_hh8",
            "close_lt_open",
            "close_lt_h1_ema20",
            "h1_ret_4_gt_0",
        ]
    )

    near_A = near_miss_table(early, seq_A, "early_relaxed_signal_A")
    near_B = near_miss_table(early, seq_B, "early_relaxed_signal_B")
    near_C = near_miss_table(early, seq_C, "early_relaxed_signal_C")
    near_R = near_miss_table(early, seq_R, "first_rally_short_relaxed")

    print_table(feature_health, "Feature Health")
    print_table(condition_tbl, "Condition Counts")
    print_table(probe_tbl, "Threshold Probe Counts")
    print_table(cum_A, "Cumulative Filter Counts: Signal A")
    print_table(cum_B, "Cumulative Filter Counts: Signal B")
    print_table(cum_C, "Cumulative Filter Counts: Signal C")
    print_table(cum_R, "Cumulative Filter Counts: First Rally")
    print_table(pair_tbl, "Pairwise Intersections", head=40)
    print_table(near_A, "Near Misses: Signal A", head=20)
    print_table(near_B, "Near Misses: Signal B", head=20)
    print_table(near_C, "Near Misses: Signal C", head=20)
    print_table(near_R, "Near Misses: First Rally", head=20)

    feature_health.to_csv(OUTPUT_DIR / "feature_health.csv", index=False)
    condition_tbl.to_csv(OUTPUT_DIR / "condition_counts.csv", index=False)
    probe_tbl.to_csv(OUTPUT_DIR / "threshold_probe_counts.csv", index=False)
    cum_A.to_csv(OUTPUT_DIR / "cumulative_counts_signal_A.csv", index=False)
    cum_B.to_csv(OUTPUT_DIR / "cumulative_counts_signal_B.csv", index=False)
    cum_C.to_csv(OUTPUT_DIR / "cumulative_counts_signal_C.csv", index=False)
    cum_R.to_csv(OUTPUT_DIR / "cumulative_counts_first_rally.csv", index=False)
    pair_tbl.to_csv(OUTPUT_DIR / "pairwise_intersections.csv", index=False)
    near_A.to_csv(OUTPUT_DIR / "near_misses_signal_A.csv", index=False)
    near_B.to_csv(OUTPUT_DIR / "near_misses_signal_B.csv", index=False)
    near_C.to_csv(OUTPUT_DIR / "near_misses_signal_C.csv", index=False)
    near_R.to_csv(OUTPUT_DIR / "near_misses_first_rally.csv", index=False)

    section("Saved Outputs")
    for p in [
        "feature_health.csv",
        "condition_counts.csv",
        "threshold_probe_counts.csv",
        "cumulative_counts_signal_A.csv",
        "cumulative_counts_signal_B.csv",
        "cumulative_counts_signal_C.csv",
        "cumulative_counts_first_rally.csv",
        "pairwise_intersections.csv",
        "near_misses_signal_A.csv",
        "near_misses_signal_B.csv",
        "near_misses_signal_C.csv",
        "near_misses_first_rally.csv",
    ]:
        print(color(f"- {OUTPUT_DIR / p}", ANSI_GREEN))

    section("Interpretation")
    print("- Feature Health tells you whether a field is mostly null, constant, or out of expected scale.")
    print("- Threshold Probe Counts show whether the raw building blocks occur individually.")
    print("- Cumulative counts reveal the exact step where each signal pipeline collapses to zero.")
    print("- Near misses show bars that almost qualified, which is usually the fastest way to redesign thresholds.")

if __name__ == "__main__":
    main()
