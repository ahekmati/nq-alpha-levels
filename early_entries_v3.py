import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- CONFIG ---------------- #
BASE_DIR = Path("research_mnq_bear_model")
H1_FILE = BASE_DIR / "early_entry_diagnostics_v2" / "h1_with_relaxed_early_signals.csv"
OUTPUT_DIR = BASE_DIR / "early_entry_v3_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

POINT_VALUE = 2.0
CONTRACTS = 1
COMMISSION_RT = 1.50
SLIPPAGE_POINTS = 2.0

STOP_POINTS_LIST = [200.0, 300.0, 400.0]
TIME_EXITS_HOURS = [6, 12, 24, 48]
R_MULTS = [1.0, 1.5, 2.0]

SESSION_START_HOUR_UTC = 14
SESSION_END_HOUR_UTC = 20
TOP_ROWS = 50

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


def print_table(df: pd.DataFrame, title: str, head=None):
    section(title)
    if df.empty:
        print(color("No rows.", ANSI_RED))
        return
    if head is not None:
        df = df.head(head)
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))


def load_data():
    if not H1_FILE.exists():
        raise FileNotFoundError(f"Missing {H1_FILE}")
    df = pd.read_csv(H1_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def add_session_flag(df: pd.DataFrame):
    out = df.copy()
    if "hour_utc" not in out.columns:
        out["hour_utc"] = out["time"].dt.hour
    out["in_session_v3"] = (
        (out["hour_utc"] >= SESSION_START_HOUR_UTC) &
        (out["hour_utc"] <= SESSION_END_HOUR_UTC)
    ).astype(int)
    return out


def add_upper_range_variants(df: pd.DataFrame):
    out = df.copy()

    out["upper_range_failure_v3"] = (
        (out["range_pos"] >= 0.65) &
        (out["high"] >= out["hh_8"]) &
        (out["close"] < out["open"]) &
        (out["h1_rsi"] >= 42) &
        (out["h1_rsi"] <= 75)
    ).astype(int)

    out["upper_range_failure_v3_loose"] = (
        (out["range_pos"] >= 0.60) &
        (out["high"] >= out["hh_8"]) &
        (out["close"] < out["open"]) &
        (out["h1_rsi"] >= 40) &
        (out["h1_rsi"] <= 78)
    ).astype(int)

    out["close_back_below_ema20_v3"] = (
        (out["high"] >= out["h1_ema_20"]) &
        (out["close"] < out["h1_ema_20"])
    ).astype(int)

    out["bear_context_soft_v3"] = (
        (out["bear_prob"] >= 0.30) &
        (out["exhaust_prob"] <= 0.65) &
        (out["daily_model_rsi"] >= 35) &
        (out["daily_model_rsi"] <= 65) &
        (out["daily_model_adx"] >= 10)
    ).astype(int)

    out["bear_context_mid_v3"] = (
        (out["bear_prob"] >= 0.40) &
        (out["exhaust_prob"] <= 0.55) &
        (out["daily_model_rsi"] >= 32) &
        (out["daily_model_rsi"] <= 58) &
        (out["daily_model_adx"] >= 12)
    ).astype(int)

    out["rsi70_context_tag_v3"] = (
        (out["rsi70_streak"].fillna(0) >= 2) |
        (out["overbought_stall_condition"].fillna(0).astype(int) == 1) |
        (out["near_end_of_avg_ob_streak"].fillna(0).astype(int) == 1)
    ).astype(int)

    out["stall_or_reject_tag_v3"] = (
        (out["overbought_stall_condition"].fillna(0).astype(int) == 1) |
        (out["close_back_below_ema20_v3"] == 1)
    ).astype(int)

    out["first_rally_window_v3"] = (
        (out["hours_from_bear_start"] >= 8) &
        (out["hours_from_bear_start"] <= 168)
    ).astype(int)

    out["positive_lookback_v3"] = (out["h1_ret_4"] > -0.001).astype(int)

    return out


def build_signals(df: pd.DataFrame):
    out = df.copy()

    # Core baseline: early upper-range rejection with soft bear context.
    out["signal_v3_A_core"] = (
        (out["is_early_window"] == 1) &
        (out["in_session_v3"] == 1) &
        (out["upper_range_failure_v3"] == 1) &
        (out["bear_context_soft_v3"] == 1)
    ).astype(int)

    # Add RSI/stall as a tag, not a gate.
    out["signal_v3_B_tagged"] = (
        (out["is_early_window"] == 1) &
        (out["in_session_v3"] == 1) &
        (out["upper_range_failure_v3"] == 1) &
        (out["bear_context_soft_v3"] == 1)
    ).astype(int)

    out["signal_v3_B_tagged_rsi_context"] = (
        (out["signal_v3_B_tagged"] == 1) &
        (out["rsi70_context_tag_v3"] == 1)
    ).astype(int)

    # Near end-of-streak OR rejection back under EMA20, not both.
    out["signal_v3_C_or_logic"] = (
        (out["is_early_window"] == 1) &
        (out["in_session_v3"] == 1) &
        (out["upper_range_failure_v3"] == 1) &
        (
            (out["near_end_of_avg_ob_streak"].fillna(0).astype(int) == 1) |
            (out["close_back_below_ema20_v3"] == 1)
        )
    ).astype(int)

    # Slightly looser structural version for sample generation.
    out["signal_v3_D_loose_structure"] = (
        (out["is_early_window"] == 1) &
        (out["in_session_v3"] == 1) &
        (out["upper_range_failure_v3_loose"] == 1) &
        (out["bear_context_soft_v3"] == 1)
    ).astype(int)

    # First rally relaxed, built to ensure we get enough observations.
    out["signal_v3_E_first_rally"] = (
        (out["is_early_window"] == 1) &
        (out["first_rally_window_v3"] == 1) &
        (out["in_session_v3"] == 1) &
        (out["bear_context_mid_v3"] == 1) &
        (out["high"] >= out["hh_8"]) &
        (out["close"] < out["open"]) &
        (
            (out["close"] < out["h1_ema_20"]) |
            (out["close_back_below_ema20_v3"] == 1)
        ) &
        (out["positive_lookback_v3"] == 1)
    ).astype(int)

    return out


def signal_counts(df: pd.DataFrame, signal_cols):
    rows = []
    for c in signal_cols:
        rows.append({
            "signal": c,
            "count": int(df[c].sum()),
            "pass_rate_pct": 100.0 * df[c].mean()
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


def tag_mix_summary(df: pd.DataFrame, signal_col: str):
    sig = df[df[signal_col] == 1].copy()
    if sig.empty:
        return pd.DataFrame()

    rows = [{
        "signal": signal_col,
        "count": len(sig),
        "pct_with_rsi70_context_tag": 100.0 * sig["rsi70_context_tag_v3"].mean(),
        "pct_with_stall_or_reject_tag": 100.0 * sig["stall_or_reject_tag_v3"].mean(),
        "mean_bear_prob": sig["bear_prob"].mean(),
        "mean_exhaust_prob": sig["exhaust_prob"].mean(),
        "mean_daily_rsi": sig["daily_model_rsi"].mean(),
        "mean_hours_from_bear_start": sig["hours_from_bear_start"].mean(),
    }]
    return pd.DataFrame(rows)


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
                    "rsi70_context_tag_v3": row.get("rsi70_context_tag_v3", 0),
                    "stall_or_reject_tag_v3": row.get("stall_or_reject_tag_v3", 0),
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
                    "rsi70_context_tag_v3": row.get("rsi70_context_tag_v3", 0),
                    "stall_or_reject_tag_v3": row.get("stall_or_reject_tag_v3", 0),
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
            "pct_rsi70_context_tag": 100.0 * g["rsi70_context_tag_v3"].mean(),
            "pct_stall_or_reject_tag": 100.0 * g["stall_or_reject_tag_v3"].mean(),
        })

    return pd.DataFrame(rows).sort_values(
        ["signal", "total_net_dollars", "profit_factor", "win_rate"],
        ascending=[True, False, False, False]
    )


def sample_rows(df: pd.DataFrame, signal_col: str):
    sig = df[df[signal_col] == 1].copy()
    if sig.empty:
        return pd.DataFrame()

    cols = [
        "time", "bear_window_id", "hours_from_bear_start",
        "close", "high", "open", "range_pos", "hh_8", "h1_rsi", "h1_ema_20",
        "bear_prob", "exhaust_prob", "daily_model_rsi", "daily_model_adx",
        "rsi70_streak", "overbought_stall_condition", "near_end_of_avg_ob_streak",
        "rsi70_context_tag_v3", "stall_or_reject_tag_v3", signal_col
    ]
    cols = [c for c in cols if c in sig.columns]
    return sig[cols].head(TOP_ROWS).copy()


def main():
    df = load_data()
    df = add_session_flag(df)
    df = add_upper_range_variants(df)
    df = build_signals(df)

    signal_cols = [
        "signal_v3_A_core",
        "signal_v3_B_tagged",
        "signal_v3_B_tagged_rsi_context",
        "signal_v3_C_or_logic",
        "signal_v3_D_loose_structure",
        "signal_v3_E_first_rally",
    ]

    count_tbl = signal_counts(df, signal_cols)

    tag_tables = []
    fwd_tables = []
    trade_tables = []
    summary_tables = []
    sample_tables = []

    for sig in signal_cols:
        tag_tbl = tag_mix_summary(df, sig)
        fwd_tbl = forward_return_stats(df, sig)
        trades = simulate_fixed_stop_event_trades(df, sig)
        smy_tbl = summarize_trade_grid(trades)
        samp_tbl = sample_rows(df, sig)

        if not tag_tbl.empty:
            tag_tables.append(tag_tbl)
        if not fwd_tbl.empty:
            fwd_tables.append(fwd_tbl)
        if not trades.empty:
            trade_tables.append(trades)
        if not smy_tbl.empty:
            summary_tables.append(smy_tbl)
        if not samp_tbl.empty:
            sample_tables.append(samp_tbl.assign(signal=sig))

    tags_all = pd.concat(tag_tables, ignore_index=True) if tag_tables else pd.DataFrame()
    fwd_all = pd.concat(fwd_tables, ignore_index=True) if fwd_tables else pd.DataFrame()
    trades_all = pd.concat(trade_tables, ignore_index=True) if trade_tables else pd.DataFrame()
    summary_all = pd.concat(summary_tables, ignore_index=True) if summary_tables else pd.DataFrame()
    samples_all = pd.concat(sample_tables, ignore_index=True) if sample_tables else pd.DataFrame()

    print_table(count_tbl, "Signal Counts")
    print_table(tags_all, "Tag Mix Summary")
    print_table(fwd_all, "Forward Return Diagnostics")
    print_table(summary_all, "Fixed Stop / Exit Grid Results", head=80)
    print_table(samples_all, "Sample Signal Rows", head=TOP_ROWS)

    count_tbl.to_csv(OUTPUT_DIR / "signal_counts_v3.csv", index=False)
    tags_all.to_csv(OUTPUT_DIR / "tag_mix_summary_v3.csv", index=False)
    fwd_all.to_csv(OUTPUT_DIR / "forward_return_diagnostics_v3.csv", index=False)
    trades_all.to_csv(OUTPUT_DIR / "signal_trades_grid_v3.csv", index=False)
    summary_all.to_csv(OUTPUT_DIR / "signal_trade_summary_grid_v3.csv", index=False)
    samples_all.to_csv(OUTPUT_DIR / "sample_signal_rows_v3.csv", index=False)
    df.to_csv(OUTPUT_DIR / "h1_with_early_signals_v3.csv", index=False)

    section("Saved Outputs")
    for p in [
        "signal_counts_v3.csv",
        "tag_mix_summary_v3.csv",
        "forward_return_diagnostics_v3.csv",
        "signal_trades_grid_v3.csv",
        "signal_trade_summary_grid_v3.csv",
        "sample_signal_rows_v3.csv",
        "h1_with_early_signals_v3.csv",
    ]:
        print(color(f"- {OUTPUT_DIR / p}", ANSI_GREEN))

    section("Interpretation")
    print("- signal_v3_A_core is the clean baseline: early upper-range failure plus soft bear context.")
    print("- signal_v3_B_tagged keeps RSI>70/stall logic as context, not a hard requirement.")
    print("- signal_v3_B_tagged_rsi_context isolates the subset where that RSI context is actually present.")
    print("- signal_v3_C_or_logic uses near-end-of-streak OR EMA20 rejection, not both.")
    print("- signal_v3_D_loose_structure is there to increase sample size and test whether structure alone has edge.")
    print("- signal_v3_E_first_rally is a looser first-rally short variant designed to avoid another zero-count result.")

if __name__ == "__main__":
    main()
