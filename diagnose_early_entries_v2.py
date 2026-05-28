import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- CONFIG ---------------- #
BASE_DIR = Path("research_mnq_bear_model")
DAILY_FILE = BASE_DIR / "mnq_daily_research_dataset.csv"
H1_FILE = BASE_DIR / "hybrid_backtest_outputs_v2" / "h1_execution_dataset_v2.csv"
OUTPUT_DIR = BASE_DIR / "early_entry_diagnostics_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

POINT_VALUE = 2.0
CONTRACTS = 1
COMMISSION_RT = 1.50
SLIPPAGE_POINTS = 2.0

STOP_POINTS_LIST = [200.0, 300.0, 400.0]
TIME_EXITS_HOURS = [6, 12, 24, 48]
R_MULTS = [1.0, 1.5, 2.0]

EARLY_WINDOW_DAYS = 8
SESSION_START_HOUR_UTC = 14
SESSION_END_HOUR_UTC = 20

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
    rows = []
    for i, (s, e) in enumerate(BEAR_WINDOWS, 1):
        rows.append({
            "bear_window_id": i,
            "start": pd.Timestamp(s, tz="UTC"),
            "end": pd.Timestamp(e, tz="UTC"),
        })
    return pd.DataFrame(rows)


def load_data():
    if not DAILY_FILE.exists():
        raise FileNotFoundError(f"Missing {DAILY_FILE}")
    if not H1_FILE.exists():
        raise FileNotFoundError(f"Missing {H1_FILE}")

    daily = pd.read_csv(DAILY_FILE)
    h1 = pd.read_csv(H1_FILE)

    daily["time"] = pd.to_datetime(daily["time"], utc=True)
    h1["time"] = pd.to_datetime(h1["time"], utc=True)

    daily = daily.sort_values("time").reset_index(drop=True)
    h1 = h1.sort_values("time").reset_index(drop=True)

    return daily, h1


def add_bear_windows_daily(daily: pd.DataFrame):
    d = daily.copy()
    d["bear_window_id"] = np.nan
    d["days_to_bear_start"] = np.nan
    d["days_from_bear_start"] = np.nan

    windows = parse_windows()

    for _, w in windows.iterrows():
        start = w["start"]
        end = w["end"]

        in_window = (d["time"] >= start) & (d["time"] <= end)
        d.loc[in_window, "bear_window_id"] = w["bear_window_id"]
        d.loc[in_window, "days_from_bear_start"] = (
            (d.loc[in_window, "time"] - start) / pd.Timedelta(days=1)
        )

        pre_mask = (d["time"] < start)
        d.loc[pre_mask, f"days_to_bear_start_{int(w['bear_window_id'])}"] = (
            (start - d.loc[pre_mask, "time"]) / pd.Timedelta(days=1)
        )

    return d


def compute_rsi_overbought_streaks(daily: pd.DataFrame):
    d = daily.copy()
    d["rsi_gt_70"] = (d["rsi"] > 70).astype(int)

    streak = 0
    streak_list = []
    for val in d["rsi_gt_70"]:
        if val == 1:
            streak += 1
        else:
            streak = 0
        streak_list.append(streak)

    d["rsi70_streak"] = streak_list

    # Peak high while RSI > 70 streak active
    peak_vals = []
    running_peak = np.nan
    for is_ob, h in zip(d["rsi_gt_70"], d["high"]):
        if is_ob == 1:
            running_peak = h if pd.isna(running_peak) else max(running_peak, h)
        else:
            running_peak = np.nan
        peak_vals.append(running_peak)
    d["ob_streak_high"] = peak_vals

    d["ob_streak_not_making_higher_high"] = (
        (d["rsi70_streak"] >= 2) &
        (d["high"] < d["ob_streak_high"].shift(1).fillna(d["high"]))
    ).astype(int)

    d["rsi_rolling_over"] = ((d["rsi"] < d["rsi"].shift(1)) & (d["rsi"] > 55)).astype(int)
    return d


def rsi_streak_event_study(daily: pd.DataFrame):
    windows = parse_windows()
    rows = []

    for _, w in windows.iterrows():
        start = w["start"]
        pre = daily[daily["time"] < start].copy().tail(30)

        if pre.empty:
            continue

        rsi70 = pre[pre["rsi70_streak"] > 0].copy()
        if rsi70.empty:
            rows.append({
                "bear_window_id": int(w["bear_window_id"]),
                "bear_start": start,
                "had_rsi70_streak_preceding": 0,
                "last_rsi70_streak_len": np.nan,
                "days_from_last_rsi70_bar_to_bear_start": np.nan,
                "max_preceding_rsi70_streak": np.nan,
            })
            continue

        last_streak_len = rsi70["rsi70_streak"].iloc[-1]
        last_rsi70_time = rsi70["time"].iloc[-1]
        max_streak = rsi70["rsi70_streak"].max()

        rows.append({
            "bear_window_id": int(w["bear_window_id"]),
            "bear_start": start,
            "had_rsi70_streak_preceding": 1,
            "last_rsi70_streak_len": last_streak_len,
            "days_from_last_rsi70_bar_to_bear_start": (start - last_rsi70_time) / pd.Timedelta(days=1),
            "max_preceding_rsi70_streak": max_streak,
        })

    return pd.DataFrame(rows)


def add_window_flags_h1(h1: pd.DataFrame):
    h = h1.copy()
    h["bear_window_id"] = np.nan
    h["is_early_window"] = 0
    h["hours_from_bear_start"] = np.nan
    h["days_from_bear_start"] = np.nan

    windows = parse_windows()

    for _, w in windows.iterrows():
        start = w["start"]
        end = w["end"]
        early_end = min(start + pd.Timedelta(days=EARLY_WINDOW_DAYS), end)

        in_window = (h["time"] >= start) & (h["time"] <= end)
        early_mask = (h["time"] >= start) & (h["time"] <= early_end)

        h.loc[in_window, "bear_window_id"] = w["bear_window_id"]
        h.loc[early_mask, "is_early_window"] = 1
        h.loc[in_window, "hours_from_bear_start"] = (h.loc[in_window, "time"] - start) / pd.Timedelta(hours=1)
        h.loc[in_window, "days_from_bear_start"] = (h.loc[in_window, "time"] - start) / pd.Timedelta(days=1)

    return h


def merge_daily_context_to_h1(h1: pd.DataFrame, daily: pd.DataFrame):
    h = h1.copy()
    h["date"] = h["time"].dt.floor("D")

    d = daily.copy()
    d["date"] = d["time"].dt.floor("D")

    keep = [
        "date",
        "rsi",
        "rsi70_streak",
        "ob_streak_not_making_higher_high",
        "rsi_rolling_over",
        "high",
        "ob_streak_high",
    ]
    d = d[keep].copy().rename(columns={
        "rsi": "daily_raw_rsi",
        "high": "daily_raw_high",
    })

    h = h.merge(d, on="date", how="left")
    return h


def add_relaxed_early_signals(h1: pd.DataFrame, avg_rsi70_len: float):
    h = h1.copy()

    h["hour_utc"] = h["time"].dt.hour
    h["in_session"] = (
        (h["hour_utc"] >= SESSION_START_HOUR_UTC) &
        (h["hour_utc"] <= SESSION_END_HOUR_UTC)
    ).astype(int)

    h["near_end_of_avg_ob_streak"] = (
        h["rsi70_streak"].fillna(0) >= max(2, int(np.floor(avg_rsi70_len - 1)))
    ).astype(int)

    h["soft_daily_weakening"] = (
        (h["bear_prob"] >= 0.35) &
        (h["exhaust_prob"] <= 0.55) &
        (h["daily_model_rsi"] >= 38) &
        (h["daily_model_rsi"] <= 62) &
        (h["daily_model_adx"] >= 12)
    ).astype(int)

    h["soft_slope_condition"] = (
        (h["daily_model_ema20_slope_5"] <= 0.15 * h["daily_model_atr"].fillna(0)) &
        (h["daily_model_ema50_slope_5"] <= 0.15 * h["daily_model_atr"].fillna(0))
    ).astype(int)

    h["overbought_stall_condition"] = (
        (h["rsi70_streak"] >= 2) &
        (
            (h["ob_streak_not_making_higher_high"] == 1) |
            (h["rsi_rolling_over"] == 1)
        )
    ).astype(int)

    h["upper_range_failure"] = (
        (h["range_pos"] >= 0.68) &
        (h["high"] >= h["hh_8"]) &
        (h["close"] < h["open"]) &
        (h["h1_rsi"] >= 45) &
        (h["h1_rsi"] <= 72)
    ).astype(int)

    h["close_back_below_ema20_fail"] = (
        (h["high"] >= h["h1_ema_20"]) &
        (h["close"] < h["h1_ema_20"])
    ).astype(int)

    h["early_relaxed_signal_A"] = (
        (h["is_early_window"] == 1) &
        (h["in_session"] == 1) &
        (h["upper_range_failure"] == 1) &
        (h["soft_daily_weakening"] == 1) &
        (h["soft_slope_condition"] == 1)
    ).astype(int)

    h["early_relaxed_signal_B"] = (
        (h["is_early_window"] == 1) &
        (h["in_session"] == 1) &
        (h["upper_range_failure"] == 1) &
        (h["overbought_stall_condition"] == 1)
    ).astype(int)

    h["early_relaxed_signal_C"] = (
        (h["is_early_window"] == 1) &
        (h["in_session"] == 1) &
        (h["upper_range_failure"] == 1) &
        (h["near_end_of_avg_ob_streak"] == 1) &
        (h["close_back_below_ema20_fail"] == 1)
    ).astype(int)

    h["first_rally_short_relaxed"] = (
        (h["hours_from_bear_start"] >= 12) &
        (h["hours_from_bear_start"] <= 144) &
        (h["in_session"] == 1) &
        (h["bear_prob"] >= 0.45) &
        (h["exhaust_prob"] <= 0.50) &
        (h["daily_model_rsi"] >= 32) &
        (h["daily_model_rsi"] <= 55) &
        (h["daily_model_adx"] >= 15) &
        (h["high"] >= h["hh_8"]) &
        (h["close"] < h["open"]) &
        (h["close"] < h["h1_ema_20"]) &
        (h["h1_ret_4"] > 0)
    ).astype(int)

    return h


def forward_return_stats(df: pd.DataFrame, signal_col: str, horizons=(6, 12, 24, 48)):
    sig = df[df[signal_col] == 1].copy()
    rows = []

    if sig.empty:
        return pd.DataFrame()

    for horizon in horizons:
        future_close = df["close"].shift(-horizon)
        tmp = sig.copy()
        tmp["fwd_points_short"] = tmp["close"] - future_close.loc[tmp.index]
        tmp["fwd_pct_short"] = (future_close.loc[tmp.index] / tmp["close"] - 1.0) * -100.0

        vals_pts = tmp["fwd_points_short"].dropna()
        vals_pct = tmp["fwd_pct_short"].dropna()

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


def simulate_fixed_stop_event_trades(df: pd.DataFrame, signal_col: str):
    sig = df[df[signal_col] == 1].copy()
    trades = []

    if sig.empty:
        return pd.DataFrame()

    for idx, row in sig.iterrows():
        entry_price = row["close"] - SLIPPAGE_POINTS

        for stop_points in STOP_POINTS_LIST:
            stop_price = entry_price + stop_points

            for time_exit in TIME_EXITS_HOURS:
                max_forward = min(time_exit, len(df) - idx - 1)
                if max_forward <= 0:
                    continue

                future = df.iloc[idx + 1: idx + 1 + max_forward].copy()

                # pure fixed-stop / time-exit
                exit_reason = f"time_exit_{time_exit}h"
                exit_price = future.iloc[-1]["close"] + SLIPPAGE_POINTS
                exit_time = future.iloc[-1]["time"]

                for _, frow in future.iterrows():
                    if frow["high"] >= stop_price:
                        exit_price = stop_price + SLIPPAGE_POINTS
                        exit_time = frow["time"]
                        exit_reason = "stop"
                        break

                gross_points = entry_price - exit_price
                gross_dollars = gross_points * POINT_VALUE * CONTRACTS
                net_dollars = gross_dollars - COMMISSION_RT

                trades.append({
                    "signal": signal_col,
                    "mode": "stop_time",
                    "stop_points": stop_points,
                    "time_exit_h": time_exit,
                    "r_mult": np.nan,
                    "entry_time": row["time"],
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_points": gross_points,
                    "net_dollars": net_dollars,
                    "exit_reason": exit_reason,
                    "bear_window_id": row.get("bear_window_id", np.nan),
                    "hours_from_bear_start": row.get("hours_from_bear_start", np.nan),
                })

            for r_mult in R_MULTS:
                target_points = stop_points * r_mult
                target_price = entry_price - target_points

                max_forward = min(48, len(df) - idx - 1)
                if max_forward <= 0:
                    continue

                future = df.iloc[idx + 1: idx + 1 + max_forward].copy()

                exit_reason = "time_exit_48h"
                exit_price = future.iloc[-1]["close"] + SLIPPAGE_POINTS
                exit_time = future.iloc[-1]["time"]

                for _, frow in future.iterrows():
                    if frow["high"] >= stop_price:
                        exit_price = stop_price + SLIPPAGE_POINTS
                        exit_time = frow["time"]
                        exit_reason = "stop"
                        break
                    if frow["low"] <= target_price:
                        exit_price = target_price - SLIPPAGE_POINTS
                        exit_time = frow["time"]
                        exit_reason = f"target_{r_mult:.1f}R"
                        break

                gross_points = entry_price - exit_price
                gross_dollars = gross_points * POINT_VALUE * CONTRACTS
                net_dollars = gross_dollars - COMMISSION_RT

                trades.append({
                    "signal": signal_col,
                    "mode": "stop_target",
                    "stop_points": stop_points,
                    "time_exit_h": 48,
                    "r_mult": r_mult,
                    "entry_time": row["time"],
                    "exit_time": exit_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_points": gross_points,
                    "net_dollars": net_dollars,
                    "exit_reason": exit_reason,
                    "bear_window_id": row.get("bear_window_id", np.nan),
                    "hours_from_bear_start": row.get("hours_from_bear_start", np.nan),
                })

    return pd.DataFrame(trades)


def summarize_trade_grid(trades: pd.DataFrame):
    if trades.empty:
        return pd.DataFrame()

    rows = []
    grp_cols = ["signal", "mode", "stop_points", "time_exit_h", "r_mult"]

    for keys, g in trades.groupby(grp_cols, dropna=False):
        wins = g[g["net_dollars"] > 0]
        losses = g[g["net_dollars"] < 0]

        gp = wins["net_dollars"].sum() if not wins.empty else 0.0
        gl = abs(losses["net_dollars"].sum()) if not losses.empty else 0.0
        pf = gp / gl if gl > 0 else np.nan

        rows.append({
            "signal": keys[0],
            "mode": keys[1],
            "stop_points": keys[2],
            "time_exit_h": keys[3],
            "r_mult": keys[4],
            "trades": len(g),
            "win_rate": (g["net_dollars"] > 0).mean(),
            "avg_net_dollars": g["net_dollars"].mean(),
            "total_net_dollars": g["net_dollars"].sum(),
            "profit_factor": pf,
            "avg_hours_from_bear_start": g["hours_from_bear_start"].mean(),
        })

    return pd.DataFrame(rows).sort_values(
        ["signal", "total_net_dollars", "profit_factor", "win_rate"],
        ascending=[True, False, False, False]
    )


def signal_counts(df: pd.DataFrame, signal_cols):
    rows = []
    for c in signal_cols:
        rows.append({
            "signal": c,
            "count": int(df[c].sum()),
            "pass_rate_pct": 100.0 * df[c].mean()
        })
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame, title: str, head=None):
    section(title)
    if df.empty:
        print(color("No rows.", ANSI_RED))
        return
    if head is not None:
        df = df.head(head)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


def main():
    daily, h1 = load_data()

    daily = add_bear_windows_daily(daily)
    daily = compute_rsi_overbought_streaks(daily)

    rsi_study = rsi_streak_event_study(daily)
    avg_rsi70_len = rsi_study["last_rsi70_streak_len"].dropna().mean()
    if pd.isna(avg_rsi70_len):
        avg_rsi70_len = 3.0

    h1 = add_window_flags_h1(h1)
    h1 = merge_daily_context_to_h1(h1, daily)
    h1 = add_relaxed_early_signals(h1, avg_rsi70_len)

    early_only = h1[h1["is_early_window"] == 1].copy()

    signal_cols = [
        "early_relaxed_signal_A",
        "early_relaxed_signal_B",
        "early_relaxed_signal_C",
        "first_rally_short_relaxed",
    ]

    signal_count_tbl = signal_counts(h1, signal_cols)

    fwd_tables = []
    trade_tables = []
    trade_summaries = []

    for sig in signal_cols:
        fwd = forward_return_stats(h1, sig)
        trd = simulate_fixed_stop_event_trades(h1, sig)
        smy = summarize_trade_grid(trd)

        if not fwd.empty:
            fwd_tables.append(fwd)
        if not trd.empty:
            trade_tables.append(trd)
        if not smy.empty:
            trade_summaries.append(smy)

    fwd_all = pd.concat(fwd_tables, ignore_index=True) if fwd_tables else pd.DataFrame()
    trades_all = pd.concat(trade_tables, ignore_index=True) if trade_tables else pd.DataFrame()
    summary_all = pd.concat(trade_summaries, ignore_index=True) if trade_summaries else pd.DataFrame()

    print_table(rsi_study, "RSI > 70 Pre-Bear Study")
    section("RSI > 70 Streak Insight")
    print(color(f"Average last RSI>70 streak length before bear windows: {fmt(avg_rsi70_len, 2)} days", ANSI_YELLOW))
    print(color(f"Median last RSI>70 streak length: {fmt(rsi_study['last_rsi70_streak_len'].dropna().median(), 2)} days", ANSI_YELLOW))
    print(color(f"Average days from last RSI>70 bar to bear start: {fmt(rsi_study['days_from_last_rsi70_bar_to_bear_start'].dropna().mean(), 2)} days", ANSI_CYAN))

    print_table(signal_count_tbl, "Relaxed Signal Counts")
    print_table(fwd_all, "Forward Return Diagnostics")
    print_table(summary_all, "Fixed Stop / Exit Grid Results", head=40)

    rsi_study.to_csv(OUTPUT_DIR / "rsi70_pre_bear_study.csv", index=False)
    signal_count_tbl.to_csv(OUTPUT_DIR / "relaxed_signal_counts.csv", index=False)
    fwd_all.to_csv(OUTPUT_DIR / "forward_return_diagnostics_relaxed.csv", index=False)
    trades_all.to_csv(OUTPUT_DIR / "relaxed_signal_trades_grid.csv", index=False)
    summary_all.to_csv(OUTPUT_DIR / "relaxed_signal_trade_summary_grid.csv", index=False)
    h1.to_csv(OUTPUT_DIR / "h1_with_relaxed_early_signals.csv", index=False)

    section("Saved Outputs")
    for p in [
        "rsi70_pre_bear_study.csv",
        "relaxed_signal_counts.csv",
        "forward_return_diagnostics_relaxed.csv",
        "relaxed_signal_trades_grid.csv",
        "relaxed_signal_trade_summary_grid.csv",
        "h1_with_relaxed_early_signals.csv",
    ]:
        print(color(f"- {OUTPUT_DIR / p}", ANSI_GREEN))

    section("Interpretation")
    print("- Signal A = early upper-range failure with softer daily weakening filters.")
    print("- Signal B = early upper-range failure plus RSI>70 stall / rollover logic.")
    print("- Signal C = early upper-range failure near the average end of an RSI>70 streak.")
    print("- first_rally_short_relaxed = looser fallback for the first countertrend rally short.")
    print("- Compare stop-size sensitivity: if 200pt fails but 300-400pt works, the idea may be right but the stop too tight.")
    print("- Compare time-exit sensitivity: if 6-12h works better than 24-48h, the setup may be an early flush trade rather than a trend hold.")

if __name__ == "__main__":
    main()
