import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- CONFIG ---------------- #
BASE_DIR = Path("research_mnq_bear_model")
H1_FILE = BASE_DIR / "hybrid_backtest_outputs_v2" / "h1_execution_dataset_v2.csv"
OUTPUT_DIR = BASE_DIR / "early_entry_diagnostics"
OUTPUT_DIR.mkdir(exist_ok=True)

POINT_VALUE = 2.0
FIXED_STOP_POINTS = 200.0
FIXED_STOP_DOLLARS = FIXED_STOP_POINTS * POINT_VALUE
CONTRACTS = 1

EARLY_WINDOW_DAYS = 8
SESSION_START_HOUR_UTC = 14
SESSION_END_HOUR_UTC = 20

FORWARD_HOURS = [6, 12, 24, 48]
R_MULTS = [1.0, 1.5, 2.0]

BEAR_WINDOWS = [
    ("2020-02-18", "2020-03-23"),
    ("2020-09-01", "2020-09-24"),
    ("2020-10-13", "2020-11-02"),
    ("2021-02-15", "2021-03-05"),
    ("2021-04-26", "2021-05-13"),
    ("2021-12-27", "2022-03-15"),
    ("2022-04-04", "2023-01-10"),
    ("2023-07-30", "2023-08-20"),
    ("2024-03-24", "2024-04-21"),
    ("2024-07-10", "2024-08-08"),
    ("2025-02-17", "2025-04-07"),
    ("2026-01-28", "2026-03-20"),
]

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


def parse_windows():
    out = []
    for i, (s, e) in enumerate(BEAR_WINDOWS, 1):
        out.append({
            "bear_window_id": i,
            "start": pd.Timestamp(s, tz="UTC"),
            "end": pd.Timestamp(e, tz="UTC"),
        })
    return out


def load_data():
    if not H1_FILE.exists():
        raise FileNotFoundError(f"Missing {H1_FILE}")

    h1 = pd.read_csv(H1_FILE)
    h1["time"] = pd.to_datetime(h1["time"], utc=True)
    h1 = h1.sort_values("time").reset_index(drop=True)
    return h1


def add_window_flags(h1: pd.DataFrame):
    h = h1.copy()
    h["bear_window_id"] = np.nan
    h["is_early_window"] = 0
    h["hours_from_bear_start"] = np.nan
    h["days_from_bear_start"] = np.nan

    windows = parse_windows()

    for w in windows:
        start = w["start"]
        end = w["end"]
        early_end = start + pd.Timedelta(days=EARLY_WINDOW_DAYS)

        in_window = (h["time"] >= start) & (h["time"] <= end)
        h.loc[in_window, "bear_window_id"] = w["bear_window_id"]

        early_mask = (h["time"] >= start) & (h["time"] <= min(early_end, end))
        h.loc[early_mask, "is_early_window"] = 1
        h.loc[early_mask, "bear_window_id"] = w["bear_window_id"]

        hrs = (h.loc[in_window, "time"] - start) / pd.Timedelta(hours=1)
        dys = (h.loc[in_window, "time"] - start) / pd.Timedelta(days=1)
        h.loc[in_window, "hours_from_bear_start"] = hrs
        h.loc[in_window, "days_from_bear_start"] = dys

    return h


def add_diagnostic_conditions(h1: pd.DataFrame):
    h = h1.copy()

    h["hour_utc"] = h["time"].dt.hour
    h["in_session"] = (
        (h["hour_utc"] >= SESSION_START_HOUR_UTC) &
        (h["hour_utc"] <= SESSION_END_HOUR_UTC)
    ).astype(int)

    h["cond_bear_prob"] = (h["bear_prob"] >= 0.45).astype(int)
    h["cond_exhaust_low"] = (h["exhaust_prob"] <= 0.45).astype(int)
    h["cond_daily_rsi_band"] = ((h["daily_model_rsi"] >= 35) & (h["daily_model_rsi"] <= 60)).astype(int)
    h["cond_daily_adx"] = (h["daily_model_adx"] >= 15).astype(int)
    h["cond_not_too_extended_50"] = (h["daily_model_dist_ema50_atr"] >= -2.50).astype(int)
    h["cond_not_too_extended_200"] = (h["daily_model_dist_ema200_atr"] >= -3.25).astype(int)
    h["cond_daily_slopes_down"] = (
        (h["daily_model_ema20_slope_5"] < 0) &
        (h["daily_model_ema50_slope_5"] < 0)
    ).astype(int)

    h["cond_range_top"] = (h["range_pos"] >= 0.70).astype(int)
    h["cond_tag_recent_high"] = (h["high"] >= h["hh_8"]).astype(int)
    h["cond_red_bar"] = (h["close"] < h["open"]).astype(int)
    h["cond_h1_rsi_band"] = ((h["h1_rsi"] >= 48) & (h["h1_rsi"] <= 70)).astype(int)
    h["cond_above_ema20_intrabar"] = (h["high"] >= h["h1_ema_20"]).astype(int)
    h["cond_close_back_below_ema20"] = (h["close"] < h["h1_ema_20"]).astype(int)

    h["cond_first_days"] = (h["is_early_window"] == 1).astype(int)

    h["early_setup_core"] = (
        (h["cond_range_top"] == 1) &
        (h["cond_tag_recent_high"] == 1) &
        (h["cond_red_bar"] == 1) &
        (h["cond_h1_rsi_band"] == 1)
    ).astype(int)

    h["early_setup_full"] = (
        (h["in_session"] == 1) &
        (h["cond_bear_prob"] == 1) &
        (h["cond_exhaust_low"] == 1) &
        (h["cond_daily_rsi_band"] == 1) &
        (h["cond_daily_adx"] == 1) &
        (h["cond_not_too_extended_50"] == 1) &
        (h["cond_not_too_extended_200"] == 1) &
        (h["cond_daily_slopes_down"] == 1) &
        (h["cond_first_days"] == 1) &
        (h["cond_range_top"] == 1) &
        (h["cond_tag_recent_high"] == 1) &
        (h["cond_red_bar"] == 1) &
        (h["cond_h1_rsi_band"] == 1)
    ).astype(int)

    h["first_rally_short_setup"] = (
        (h["in_session"] == 1) &
        (h["cond_bear_prob"] == 1) &
        (h["cond_exhaust_low"] == 1) &
        (h["cond_daily_rsi_band"] == 1) &
        (h["cond_daily_adx"] == 1) &
        (h["cond_not_too_extended_50"] == 1) &
        (h["cond_not_too_extended_200"] == 1) &
        (h["cond_daily_slopes_down"] == 1) &
        (h["hours_from_bear_start"] >= 24) &
        (h["hours_from_bear_start"] <= 120) &
        (h["high"] >= h["hh_8"]) &
        (h["close"] < h["open"]) &
        (h["close"] < h["h1_ema_20"]) &
        (h["h1_rsi"] < 58) &
        (h["h1_ret_4"] > 0)
    ).astype(int)

    return h


def forward_return_stats(df: pd.DataFrame, signal_col: str):
    sig = df[df[signal_col] == 1].copy()
    rows = []

    if sig.empty:
        return pd.DataFrame()

    for horizon in FORWARD_HOURS:
        future_close = df["close"].shift(-horizon)
        tmp = sig.copy()
        tmp[f"fwd_points_{horizon}h"] = tmp["close"] - future_close.loc[tmp.index]
        tmp[f"fwd_pct_{horizon}h"] = (future_close.loc[tmp.index] / tmp["close"] - 1.0) * -100.0

        vals_pts = tmp[f"fwd_points_{horizon}h"].dropna()
        vals_pct = tmp[f"fwd_pct_{horizon}h"].dropna()

        rows.append({
            "signal": signal_col,
            "horizon_h": horizon,
            "count": len(vals_pts),
            "mean_points_short": vals_pts.mean() if len(vals_pts) else np.nan,
            "median_points_short": vals_pts.median() if len(vals_pts) else np.nan,
            "win_rate_short": (vals_pts > 0).mean() if len(vals_pts) else np.nan,
            "mean_pct_short": vals_pct.mean() if len(vals_pct) else np.nan,
        })

    return pd.DataFrame(rows)


def simulate_fixed_stop_event_trades(df: pd.DataFrame, signal_col: str, setup_name: str):
    sig = df[df[signal_col] == 1].copy()
    trades = []

    if sig.empty:
        return pd.DataFrame()

    for idx, row in sig.iterrows():
        entry_price = row["close"] - 2.0
        stop_price = entry_price + FIXED_STOP_POINTS

        for r_mult in R_MULTS:
            target_points = FIXED_STOP_POINTS * r_mult
            target_price = entry_price - target_points

            exit_reason = "time_exit_48h"
            exit_price = None
            exit_time = None

            max_forward = min(48, len(df) - idx - 1)
            if max_forward <= 0:
                continue

            future = df.iloc[idx + 1: idx + 1 + max_forward].copy()

            for _, frow in future.iterrows():
                if frow["high"] >= stop_price:
                    exit_price = stop_price + 2.0
                    exit_time = frow["time"]
                    exit_reason = "stop"
                    break
                if frow["low"] <= target_price:
                    exit_price = target_price - 2.0
                    exit_time = frow["time"]
                    exit_reason = f"target_{r_mult:.1f}R"
                    break

            if exit_price is None:
                last = future.iloc[-1]
                exit_price = last["close"] + 2.0
                exit_time = last["time"]

            gross_points = entry_price - exit_price
            gross_dollars = gross_points * POINT_VALUE * CONTRACTS
            net_dollars = gross_dollars - 1.50

            trades.append({
                "setup": setup_name,
                "signal_col": signal_col,
                "r_mult": r_mult,
                "entry_time": row["time"],
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "gross_points": gross_points,
                "gross_dollars": gross_dollars,
                "net_dollars": net_dollars,
                "exit_reason": exit_reason,
                "bear_window_id": row.get("bear_window_id", np.nan),
                "hours_from_bear_start": row.get("hours_from_bear_start", np.nan),
                "entry_bear_prob": row.get("bear_prob", np.nan),
                "entry_exhaust_prob": row.get("exhaust_prob", np.nan),
                "entry_daily_rsi": row.get("daily_model_rsi", np.nan),
                "entry_daily_adx": row.get("daily_model_adx", np.nan),
                "entry_range_pos": row.get("range_pos", np.nan),
            })

    return pd.DataFrame(trades)


def summarize_trade_set(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame()

    rows = []
    for r_mult, g in trades.groupby("r_mult"):
        wins = g[g["net_dollars"] > 0]
        losses = g[g["net_dollars"] < 0]

        gp = wins["net_dollars"].sum() if not wins.empty else 0.0
        gl = abs(losses["net_dollars"].sum()) if not losses.empty else 0.0
        pf = gp / gl if gl > 0 else np.nan

        rows.append({
            "setup": g["setup"].iloc[0],
            "r_mult": r_mult,
            "trades": len(g),
            "win_rate": (g["net_dollars"] > 0).mean(),
            "avg_net_dollars": g["net_dollars"].mean(),
            "total_net_dollars": g["net_dollars"].sum(),
            "profit_factor": pf,
            "avg_hours_from_bear_start": g["hours_from_bear_start"].mean(),
        })

    return pd.DataFrame(rows)


def condition_pass_table(df: pd.DataFrame):
    early = df[df["is_early_window"] == 1].copy()

    conds = [
        "in_session",
        "cond_bear_prob",
        "cond_exhaust_low",
        "cond_daily_rsi_band",
        "cond_daily_adx",
        "cond_not_too_extended_50",
        "cond_not_too_extended_200",
        "cond_daily_slopes_down",
        "cond_range_top",
        "cond_tag_recent_high",
        "cond_red_bar",
        "cond_h1_rsi_band",
        "cond_above_ema20_intrabar",
        "cond_close_back_below_ema20",
        "early_setup_core",
        "early_setup_full",
        "first_rally_short_setup",
    ]

    rows = []
    n = len(early)
    for c in conds:
        if c in early.columns:
            passed = int(early[c].fillna(0).sum())
            rows.append({
                "condition": c,
                "passes": passed,
                "pass_rate_pct": 100.0 * passed / n if n > 0 else np.nan
            })

    return pd.DataFrame(rows)


def bottleneck_by_window(df: pd.DataFrame):
    early = df[df["is_early_window"] == 1].copy()
    conds = [
        "in_session",
        "cond_bear_prob",
        "cond_exhaust_low",
        "cond_daily_rsi_band",
        "cond_daily_adx",
        "cond_not_too_extended_50",
        "cond_not_too_extended_200",
        "cond_daily_slopes_down",
        "cond_range_top",
        "cond_tag_recent_high",
        "cond_red_bar",
        "cond_h1_rsi_band",
    ]

    rows = []
    for window_id, g in early.groupby("bear_window_id"):
        for c in conds:
            rows.append({
                "bear_window_id": int(window_id),
                "condition": c,
                "pass_rate_pct": 100.0 * g[c].fillna(0).mean()
            })

    return pd.DataFrame(rows)


def print_summary_table(df: pd.DataFrame, title: str):
    section(title)
    if df.empty:
        print(color("No rows.", ANSI_RED))
        return
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


def main():
    h1 = load_data()
    h1 = add_window_flags(h1)
    h1 = add_diagnostic_conditions(h1)

    early_only = h1[h1["is_early_window"] == 1].copy()

    section("Dataset")
    print(color(f"H1 rows total: {len(h1)}", ANSI_YELLOW))
    print(color(f"Early-window rows: {len(early_only)}", ANSI_YELLOW))
    print(color(f"Fixed stop: {FIXED_STOP_POINTS:.0f} points = ${FIXED_STOP_DOLLARS:.0f} per 1 lot", ANSI_CYAN))

    pass_tbl = condition_pass_table(h1)
    bottleneck_tbl = bottleneck_by_window(h1)

    forward_early_core = forward_return_stats(early_only, "early_setup_core")
    forward_early_full = forward_return_stats(early_only, "early_setup_full")
    forward_first_rally = forward_return_stats(h1, "first_rally_short_setup")

    early_core_trades = simulate_fixed_stop_event_trades(early_only, "early_setup_core", "early_setup_core")
    early_full_trades = simulate_fixed_stop_event_trades(early_only, "early_setup_full", "early_setup_full")
    first_rally_trades = simulate_fixed_stop_event_trades(h1, "first_rally_short_setup", "first_rally_short_setup")

    early_core_summary = summarize_trade_set(early_core_trades)
    early_full_summary = summarize_trade_set(early_full_trades)
    first_rally_summary = summarize_trade_set(first_rally_trades)

    print_summary_table(pass_tbl, "Condition Pass Rates")
    print_summary_table(forward_early_core, "Forward Returns: Early Setup Core")
    print_summary_table(forward_early_full, "Forward Returns: Early Setup Full")
    print_summary_table(forward_first_rally, "Forward Returns: First Rally Short")
    print_summary_table(early_core_summary, "Fixed Stop Results: Early Setup Core")
    print_summary_table(early_full_summary, "Fixed Stop Results: Early Setup Full")
    print_summary_table(first_rally_summary, "Fixed Stop Results: First Rally Short")

    pass_tbl.to_csv(OUTPUT_DIR / "condition_pass_rates.csv", index=False)
    bottleneck_tbl.to_csv(OUTPUT_DIR / "condition_bottlenecks_by_window.csv", index=False)
    forward_early_core.to_csv(OUTPUT_DIR / "forward_returns_early_core.csv", index=False)
    forward_early_full.to_csv(OUTPUT_DIR / "forward_returns_early_full.csv", index=False)
    forward_first_rally.to_csv(OUTPUT_DIR / "forward_returns_first_rally.csv", index=False)
    early_core_trades.to_csv(OUTPUT_DIR / "trades_early_core_fixed_1lot.csv", index=False)
    early_full_trades.to_csv(OUTPUT_DIR / "trades_early_full_fixed_1lot.csv", index=False)
    first_rally_trades.to_csv(OUTPUT_DIR / "trades_first_rally_fixed_1lot.csv", index=False)
    early_core_summary.to_csv(OUTPUT_DIR / "summary_early_core_fixed_1lot.csv", index=False)
    early_full_summary.to_csv(OUTPUT_DIR / "summary_early_full_fixed_1lot.csv", index=False)
    first_rally_summary.to_csv(OUTPUT_DIR / "summary_first_rally_fixed_1lot.csv", index=False)

    section("Saved Outputs")
    for p in [
        "condition_pass_rates.csv",
        "condition_bottlenecks_by_window.csv",
        "forward_returns_early_core.csv",
        "forward_returns_early_full.csv",
        "forward_returns_first_rally.csv",
        "trades_early_core_fixed_1lot.csv",
        "trades_early_full_fixed_1lot.csv",
        "trades_first_rally_fixed_1lot.csv",
        "summary_early_core_fixed_1lot.csv",
        "summary_early_full_fixed_1lot.csv",
        "summary_first_rally_fixed_1lot.csv",
    ]:
        print(color(f"- {OUTPUT_DIR / p}", ANSI_GREEN))

    section("Interpretation")
    print("- early_setup_core tells you whether the raw 'sell near upper range failure early in the bear window' idea has directional edge.")
    print("- early_setup_full tells you whether the idea improves after adding daily regime filters.")
    print("- first_rally_short_setup is the practical fallback: wait for the first countertrend rally, then short the failure.")
    print("- The fixed-stop trade summaries are strategy-style diagnostics; the forward-return tables are event-study diagnostics.")

if __name__ == "__main__":
    main()
