import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd


# ---------------- CONFIG ---------------- #
BASE_DIR = Path("research_mnq_bear_model")
SRC_FILE = BASE_DIR / "early_entry_v3_outputs" / "h1_with_early_signals_v3.csv"
OUT_DIR = BASE_DIR / "promotion_early_short_proto_v1"
OUT_DIR.mkdir(exist_ok=True)

POINT_VALUE = 2.0
CONTRACTS = 1
COMMISSION_RT = 1.50
SLIPPAGE_POINTS = 2.0

PROTO_SIGNAL_NAME = "early_short_proto_v1"
BENCHMARK_SIGNAL_NAME = "signal_v3_E_first_rally"

DEFAULT_STOP_POINTS = 300.0
DEFAULT_TIME_EXIT_H = 48
DEFAULT_R_MULT = 2.0
DEFAULT_MODE = "stop_time"
ENABLED = 1

TOP_ROWS = 100

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
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
    if not SRC_FILE.exists():
        raise FileNotFoundError(f"Missing {SRC_FILE}")
    df = pd.read_csv(SRC_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def stamp_proto_signal(df: pd.DataFrame):
    out = df.copy()

    required_cols = [
        "signal_v3_A_core",
        "signal_v3_E_first_rally",
        "rsi70_context_tag_v3",
        "stall_or_reject_tag_v3",
        "bear_prob",
        "exhaust_prob",
        "daily_model_rsi",
        "daily_model_adx",
        "hours_from_bear_start",
        "bear_window_id",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out[PROTO_SIGNAL_NAME] = out["signal_v3_A_core"].fillna(0).astype(int)
    out["benchmark_first_rally_v1"] = out["signal_v3_E_first_rally"].fillna(0).astype(int)

    out["proto_enabled"] = ENABLED
    out["proto_stop_points"] = DEFAULT_STOP_POINTS
    out["proto_time_exit_h"] = DEFAULT_TIME_EXIT_H
    out["proto_r_mult"] = DEFAULT_R_MULT
    out["proto_mode"] = DEFAULT_MODE

    out["proto_has_rsi70_context"] = out["rsi70_context_tag_v3"].fillna(0).astype(int)
    out["proto_has_stall_or_reject"] = out["stall_or_reject_tag_v3"].fillna(0).astype(int)

    return out


def simulate_trades(df: pd.DataFrame, signal_col: str, stop_points: float, time_exit_h: int, mode: str, r_mult: float):
    sig = df[df[signal_col] == 1].copy()
    trades = []

    if sig.empty:
        return pd.DataFrame()

    for idx, row in sig.iterrows():
        entry_price = row["close"] - SLIPPAGE_POINTS
        stop_price = entry_price + stop_points

        max_forward = min(time_exit_h, len(df) - idx - 1)
        if max_forward <= 0:
            continue

        future = df.iloc[idx + 1: idx + 1 + max_forward].copy()

        exit_reason = f"time_exit_{time_exit_h}h"
        exit_price = future.iloc[-1]["close"] + SLIPPAGE_POINTS
        exit_time = future.iloc[-1]["time"]

        if mode == "stop_target":
            target_points = stop_points * r_mult
            target_price = entry_price - target_points

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
        else:
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
            "entry_time": row["time"],
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_points": gross_points,
            "gross_dollars": gross_dollars,
            "net_dollars": net_dollars,
            "exit_reason": exit_reason,
            "bear_window_id": row.get("bear_window_id", np.nan),
            "hours_from_bear_start": row.get("hours_from_bear_start", np.nan),
            "bear_prob": row.get("bear_prob", np.nan),
            "exhaust_prob": row.get("exhaust_prob", np.nan),
            "daily_model_rsi": row.get("daily_model_rsi", np.nan),
            "daily_model_adx": row.get("daily_model_adx", np.nan),
            "proto_has_rsi70_context": row.get("proto_has_rsi70_context", 0),
            "proto_has_stall_or_reject": row.get("proto_has_stall_or_reject", 0),
            "stop_points": stop_points,
            "time_exit_h": time_exit_h,
            "mode": mode,
            "r_mult": r_mult,
        })

    return pd.DataFrame(trades)


def summarize_trades(trades: pd.DataFrame, label: str):
    if trades.empty:
        return pd.DataFrame([{
            "label": label,
            "trades": 0,
            "win_rate": np.nan,
            "avg_net_dollars": np.nan,
            "total_net_dollars": np.nan,
            "profit_factor": np.nan,
            "avg_gross_points": np.nan,
            "avg_hours_from_bear_start": np.nan,
            "pct_stop_exits": np.nan,
        }])

    wins = trades[trades["net_dollars"] > 0]
    losses = trades[trades["net_dollars"] < 0]

    gp = wins["net_dollars"].sum() if not wins.empty else 0.0
    gl = abs(losses["net_dollars"].sum()) if not losses.empty else 0.0
    pf = gp / gl if gl > 0 else np.nan

    return pd.DataFrame([{
        "label": label,
        "trades": len(trades),
        "win_rate": (trades["net_dollars"] > 0).mean(),
        "avg_net_dollars": trades["net_dollars"].mean(),
        "total_net_dollars": trades["net_dollars"].sum(),
        "profit_factor": pf,
        "avg_gross_points": trades["gross_points"].mean(),
        "avg_hours_from_bear_start": trades["hours_from_bear_start"].mean(),
        "pct_stop_exits": 100.0 * (trades["exit_reason"] == "stop").mean(),
    }])


def per_window_summary(proto_trades: pd.DataFrame, bench_trades: pd.DataFrame):
    proto = proto_trades.groupby("bear_window_id").agg(
        proto_trades=("net_dollars", "size"),
        proto_total_net_dollars=("net_dollars", "sum"),
        proto_avg_net_dollars=("net_dollars", "mean"),
        proto_win_rate=("net_dollars", lambda s: (s > 0).mean()),
        proto_avg_points=("gross_points", "mean"),
        proto_avg_hours_from_start=("hours_from_bear_start", "mean"),
    ).reset_index()

    bench = bench_trades.groupby("bear_window_id").agg(
        bench_trades=("net_dollars", "size"),
        bench_total_net_dollars=("net_dollars", "sum"),
        bench_avg_net_dollars=("net_dollars", "mean"),
        bench_win_rate=("net_dollars", lambda s: (s > 0).mean()),
        bench_avg_points=("gross_points", "mean"),
        bench_avg_hours_from_start=("hours_from_bear_start", "mean"),
    ).reset_index()

    out = pd.merge(proto, bench, on="bear_window_id", how="outer").sort_values("bear_window_id")
    return out


def build_live_rules_manifest():
    rows = [{
        "rule_name": PROTO_SIGNAL_NAME,
        "enabled": ENABLED,
        "source_signal_column": "signal_v3_A_core",
        "description": "Early upper-range failure in soft bear context promoted from v3 research.",
        "entry_side": "short",
        "timeframe": "H1",
        "session_filter": "14:00-20:00 UTC",
        "window_filter": "is_early_window == 1",
        "structure_filter": "upper_range_failure_v3 == 1",
        "context_filter": "bear_context_soft_v3 == 1",
        "optional_context_tag": "rsi70_context_tag_v3",
        "optional_rejection_tag": "stall_or_reject_tag_v3",
        "stop_points": DEFAULT_STOP_POINTS,
        "time_exit_h": DEFAULT_TIME_EXIT_H,
        "mode": DEFAULT_MODE,
        "r_mult_if_target_mode": DEFAULT_R_MULT,
        "notes": "RSI>70 context retained as tag only, not hard gate.",
    }]
    return pd.DataFrame(rows)


def build_promotion_manifest(df: pd.DataFrame, proto_trades: pd.DataFrame):
    first_time = df["time"].min()
    last_time = df["time"].max()

    return pd.DataFrame([{
        "artifact_name": PROTO_SIGNAL_NAME,
        "source_file": str(SRC_FILE),
        "dataset_start_utc": first_time,
        "dataset_end_utc": last_time,
        "selected_signal": "signal_v3_A_core",
        "benchmark_signal": BENCHMARK_SIGNAL_NAME,
        "enabled": ENABLED,
        "stop_points": DEFAULT_STOP_POINTS,
        "time_exit_h": DEFAULT_TIME_EXIT_H,
        "mode": DEFAULT_MODE,
        "r_mult": DEFAULT_R_MULT,
        "promoted_trade_count": len(proto_trades),
        "research_status": "promoted_to_proto",
        "rsi_context_role": "tag_only",
        "notes": "Promotion artifact created from early_entries_v3 results.",
    }])


def event_rows(df: pd.DataFrame, signal_col: str):
    sig = df[df[signal_col] == 1].copy()
    if sig.empty:
        return pd.DataFrame()

    cols = [
        "time", "bear_window_id", "hours_from_bear_start",
        "close", "high", "open", "range_pos", "hh_8", "h1_rsi", "h1_ema_20",
        "bear_prob", "exhaust_prob", "daily_model_rsi", "daily_model_adx",
        "proto_has_rsi70_context", "proto_has_stall_or_reject",
        signal_col
    ]
    cols = [c for c in cols if c in sig.columns]
    return sig[cols].copy()


def main():
    df = load_data()
    df = stamp_proto_signal(df)

    proto_trades = simulate_trades(
        df, PROTO_SIGNAL_NAME,
        stop_points=DEFAULT_STOP_POINTS,
        time_exit_h=DEFAULT_TIME_EXIT_H,
        mode=DEFAULT_MODE,
        r_mult=DEFAULT_R_MULT
    )

    bench_trades = simulate_trades(
        df, "benchmark_first_rally_v1",
        stop_points=DEFAULT_STOP_POINTS,
        time_exit_h=DEFAULT_TIME_EXIT_H,
        mode=DEFAULT_MODE,
        r_mult=DEFAULT_R_MULT
    )

    proto_summary = summarize_trades(proto_trades, PROTO_SIGNAL_NAME)
    bench_summary = summarize_trades(bench_trades, "benchmark_first_rally_v1")
    combined_summary = pd.concat([proto_summary, bench_summary], ignore_index=True)

    window_tbl = per_window_summary(proto_trades, bench_trades)
    live_rules = build_live_rules_manifest()
    promotion_manifest = build_promotion_manifest(df, proto_trades)

    proto_events = event_rows(df, PROTO_SIGNAL_NAME)
    bench_events = event_rows(df, "benchmark_first_rally_v1")

    print_table(combined_summary, "Promotion Summary")
    print_table(window_tbl, "Per Bear Window Summary")
    print_table(proto_events, "Proto Event Rows", head=TOP_ROWS)
    print_table(bench_events, "Benchmark Event Rows", head=TOP_ROWS)
    print_table(live_rules, "Live Rules Manifest")
    print_table(promotion_manifest, "Promotion Manifest")

    df.to_csv(OUT_DIR / "h1_execution_dataset_with_proto_v1.csv", index=False)
    proto_trades.to_csv(OUT_DIR / "early_short_proto_v1_trades.csv", index=False)
    bench_trades.to_csv(OUT_DIR / "benchmark_first_rally_v1_trades.csv", index=False)
    combined_summary.to_csv(OUT_DIR / "promotion_summary.csv", index=False)
    window_tbl.to_csv(OUT_DIR / "per_bear_window_summary.csv", index=False)
    proto_events.to_csv(OUT_DIR / "early_short_proto_v1_events.csv", index=False)
    bench_events.to_csv(OUT_DIR / "benchmark_first_rally_v1_events.csv", index=False)
    live_rules.to_csv(OUT_DIR / "live_rules_manifest.csv", index=False)
    promotion_manifest.to_csv(OUT_DIR / "promotion_manifest.csv", index=False)

    section("Saved Outputs")
    for p in [
        "h1_execution_dataset_with_proto_v1.csv",
        "early_short_proto_v1_trades.csv",
        "benchmark_first_rally_v1_trades.csv",
        "promotion_summary.csv",
        "per_bear_window_summary.csv",
        "early_short_proto_v1_events.csv",
        "benchmark_first_rally_v1_events.csv",
        "live_rules_manifest.csv",
        "promotion_manifest.csv",
    ]:
        print(color(f"- {OUT_DIR / p}", ANSI_GREEN))

    section("Interpretation")
    print("- early_short_proto_v1 is stamped directly from signal_v3_A_core.")
    print("- benchmark_first_rally_v1 is included so you can compare the promoted early signal against the looser first-rally short.")
    print("- live_rules_manifest.csv is the execution-facing rules file.")
    print("- promotion_manifest.csv freezes the selected profile and parameters so the result is reproducible.")

if __name__ == "__main__":
    main()
