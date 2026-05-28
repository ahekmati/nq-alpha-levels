#!/usr/bin/env python3
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =========================================================
# MT5 BACKEND LOADER
# =========================================================
MT5_BACKEND = None
mt5 = None

MT5_HOST = os.getenv("MT5_HOST", "localhost")
MT5_PORT = int(os.getenv("MT5_PORT", "18812"))

try:
    import MetaTrader5 as mt5
    MT5_BACKEND = "MetaTrader5"
except ImportError:
    try:
        from mt5linux import MetaTrader5
        mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
        MT5_BACKEND = "mt5linux"
    except ImportError:
        mt5 = None
        MT5_BACKEND = None


# =========================================================
# CONFIG
# =========================================================
SYMBOL = os.getenv("MT5_SYMBOL", "@MNQ")
BARS_H1 = int(os.getenv("BARS_H1", "60000"))
BARS_D1 = int(os.getenv("BARS_D1", "5000"))
SERVER_TO_GMT_HOURS = int(os.getenv("SERVER_TO_GMT_HOURS", "0"))

HOLD_BARS = int(os.getenv("HOLD_BARS", "72"))
DAILY_RSI_PERIOD = int(os.getenv("DAILY_RSI_PERIOD", "14"))

# sweep ranges
RSI_LEVELS = list(range(50, 86, 2))
ATR_THRESHOLDS = list(range(50, 201, 10))
ATR_PERIODS = [7, 14]

SESSION_STARTS = [12, 13, 14]
SESSION_ENDS = [20, 21, 22]
ENTRY_HOURS = [22, 23, 0]

MIN_SAMPLE_FILTER = int(os.getenv("MIN_SAMPLE_FILTER", "15"))

STOP_CANDIDATES = [60, 80, 100, 120, 150, 180, 200, 220, 250, 300]
TARGET_CANDIDATES = [80, 100, 120, 150, 180, 200, 250, 300, 350, 400, 500, 600]

OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "rsi2300_param_sweep")


# =========================================================
# INDICATORS
# =========================================================
def rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# =========================================================
# MT5 LOAD
# =========================================================
def init_mt5():
    if mt5 is None:
        raise RuntimeError("No MT5 backend found. Install MetaTrader5 or mt5linux.")
    print(f"[INFO] Using MT5 backend: {MT5_BACKEND}")
    ok = mt5.initialize()
    if not ok:
        err = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
        raise RuntimeError(f"mt5.initialize() failed: {err}")

def shutdown_mt5():
    if mt5 is not None:
        try:
            mt5.shutdown()
        except Exception:
            pass

def load_rates(symbol: str, timeframe, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        err = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
        raise RuntimeError(f"copy_rates_from_pos failed for {symbol}, tf={timeframe}, error={err}")
    df = pd.DataFrame(rates)
    if df.empty:
        raise RuntimeError(f"No rates returned for {symbol}, tf={timeframe}")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)

def add_gmt_columns(df: pd.DataFrame, server_to_gmt_hours: int = 0) -> pd.DataFrame:
    out = df.copy()
    out["time_gmt"] = out["time"] - pd.to_timedelta(server_to_gmt_hours, unit="h")
    out["gmt_hour"] = out["time_gmt"].dt.hour
    out["gmt_date"] = out["time_gmt"].dt.date
    out["weekday"] = out["time_gmt"].dt.weekday
    return out


# =========================================================
# FEATURES
# =========================================================
def build_daily_features(d1: pd.DataFrame) -> pd.DataFrame:
    d = d1.copy()
    d["d_rsi"] = rsi_wilder(d["close"], DAILY_RSI_PERIOD)
    d["day"] = d["time_gmt"].dt.date
    return d

def build_h1_features(h1: pd.DataFrame) -> pd.DataFrame:
    h = h1.copy()
    h["atr7"] = atr_wilder(h, 7)
    h["atr14"] = atr_wilder(h, 14)
    h["is_red"] = (h["close"] < h["open"]).astype(int)
    h["day"] = h["time_gmt"].dt.date
    return h

def merge_daily_into_h1(h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    daily_cols = ["day", "d_rsi"]
    return h1.merge(d1[daily_cols], on="day", how="left")


# =========================================================
# EVENT BUILD
# =========================================================
def has_qualified_session_candle(day_df: pd.DataFrame, atr_col: str, atr_threshold: float,
                                 session_start: int, session_end: int) -> bool:
    scan = day_df[
        (day_df["gmt_hour"] >= session_start) &
        (day_df["gmt_hour"] < session_end)
    ]
    if scan.empty:
        return False
    cond = (scan["is_red"] == 1) & (scan[atr_col] >= atr_threshold)
    return bool(cond.any())

def build_event_dataset(df: pd.DataFrame,
                        rsi_gate: float,
                        atr_col: str,
                        atr_threshold: float,
                        session_start: int,
                        session_end: int,
                        entry_hour: int) -> pd.DataFrame:
    rows = []

    for i in range(len(df) - HOLD_BARS - 2):
        row = df.iloc[i]

        if pd.isna(row["d_rsi"]):
            continue
        if row["gmt_hour"] != entry_hour:
            continue
        if row["d_rsi"] < rsi_gate:
            continue

        same_day = df[df["gmt_date"] == row["gmt_date"]]
        same_day = same_day[same_day.index < df.index[i]]
        if not has_qualified_session_candle(same_day, atr_col, atr_threshold, session_start, session_end):
            continue

        nxt = df.iloc[i + 1]
        future = df.iloc[i + 1:i + 1 + HOLD_BARS].copy()
        if future.empty:
            continue

        entry = float(nxt["open"])
        max_up = float(future["high"].max() - entry)
        max_down = float(entry - future["low"].min())
        final_move = float(future.iloc[-1]["close"] - entry)

        rows.append({
            "signal_time": row["time"],
            "entry_time": nxt["time"],
            "entry": entry,
            "gmt_date": row["gmt_date"],
            "entry_hour": entry_hour,
            "session_start": session_start,
            "session_end": session_end,
            "atr_col": atr_col,
            "atr_threshold": atr_threshold,
            "rsi_gate": rsi_gate,
            "d_rsi": row["d_rsi"],
            "max_favorable_points": max_up,
            "max_adverse_points": max_down,
            "final_move_points": final_move,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("signal_time").reset_index(drop=True)


# =========================================================
# SUMMARIES
# =========================================================
def summarize_subset(sub: pd.DataFrame) -> Dict:
    if sub.empty:
        return {
            "count": 0,
            "avg_favorable": np.nan,
            "median_favorable": np.nan,
            "p60_favorable": np.nan,
            "p70_favorable": np.nan,
            "p80_favorable": np.nan,
            "p90_favorable": np.nan,
            "avg_adverse": np.nan,
            "median_adverse": np.nan,
            "p60_adverse": np.nan,
            "p70_adverse": np.nan,
            "p80_adverse": np.nan,
            "p90_adverse": np.nan,
            "avg_final_move": np.nan,
            "median_final_move": np.nan,
            "hit_100": np.nan,
            "hit_150": np.nan,
            "hit_200": np.nan,
            "hit_250": np.nan,
            "hit_300": np.nan,
            "hit_400": np.nan,
        }

    fav = sub["max_favorable_points"]
    adv = sub["max_adverse_points"]
    fin = sub["final_move_points"]

    return {
        "count": len(sub),
        "avg_favorable": fav.mean(),
        "median_favorable": fav.median(),
        "p60_favorable": fav.quantile(0.60),
        "p70_favorable": fav.quantile(0.70),
        "p80_favorable": fav.quantile(0.80),
        "p90_favorable": fav.quantile(0.90),
        "avg_adverse": adv.mean(),
        "median_adverse": adv.median(),
        "p60_adverse": adv.quantile(0.60),
        "p70_adverse": adv.quantile(0.70),
        "p80_adverse": adv.quantile(0.80),
        "p90_adverse": adv.quantile(0.90),
        "avg_final_move": fin.mean(),
        "median_final_move": fin.median(),
        "hit_100": (fav >= 100).mean(),
        "hit_150": (fav >= 150).mean(),
        "hit_200": (fav >= 200).mean(),
        "hit_250": (fav >= 250).mean(),
        "hit_300": (fav >= 300).mean(),
        "hit_400": (fav >= 400).mean(),
    }


# =========================================================
# STOP / TARGET SCREEN
# =========================================================
def simulate_screening(sub: pd.DataFrame, stop_points: float, target_points: float) -> Dict:
    if sub.empty:
        return {
            "count": 0,
            "stop": stop_points,
            "target": target_points,
            "target_hit_rate": np.nan,
            "stop_hit_rate": np.nan,
            "ambiguous_both_hit_rate": np.nan,
            "screening_score": np.nan,
        }

    fav_hit = sub["max_favorable_points"] >= target_points
    stop_hit = sub["max_adverse_points"] >= stop_points

    both = fav_hit & stop_hit
    target_only = fav_hit & (~stop_hit)
    stop_only = stop_hit & (~fav_hit)
    neither = (~fav_hit) & (~stop_hit)

    screening_score = (
        target_only.mean() * target_points
        - stop_only.mean() * stop_points
        + neither.mean() * sub["final_move_points"].mean()
    )

    return {
        "count": len(sub),
        "stop": stop_points,
        "target": target_points,
        "target_hit_rate": fav_hit.mean(),
        "stop_hit_rate": stop_hit.mean(),
        "ambiguous_both_hit_rate": both.mean(),
        "screening_score": screening_score,
    }

def build_stop_target_grid(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stop in STOP_CANDIDATES:
        for target in TARGET_CANDIDATES:
            rows.append(simulate_screening(sub, stop, target))
    return pd.DataFrame(rows)


# =========================================================
# MAIN SWEEP
# =========================================================
def run_full_grid(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows = []
    saved_event_sets = {}

    for atr_period in ATR_PERIODS:
        atr_col = f"atr{atr_period}"

        for rsi_gate in RSI_LEVELS:
            for atr_threshold in ATR_THRESHOLDS:
                for session_start in SESSION_STARTS:
                    for session_end in SESSION_ENDS:
                        if session_end <= session_start:
                            continue
                        for entry_hour in ENTRY_HOURS:
                            events = build_event_dataset(
                                df=df,
                                rsi_gate=rsi_gate,
                                atr_col=atr_col,
                                atr_threshold=atr_threshold,
                                session_start=session_start,
                                session_end=session_end,
                                entry_hour=entry_hour
                            )
                            stats = summarize_subset(events)
                            row = {
                                "atr_period": atr_period,
                                "atr_col": atr_col,
                                "rsi_gate": rsi_gate,
                                "atr_threshold": atr_threshold,
                                "session_start": session_start,
                                "session_end": session_end,
                                "entry_hour": entry_hour,
                                **stats
                            }
                            rows.append(row)

                            if stats["count"] >= MIN_SAMPLE_FILTER:
                                key = f"atr{atr_period}_rsi{rsi_gate}_atr{atr_threshold}_ss{session_start}_se{session_end}_eh{entry_hour}"
                                saved_event_sets[key] = events

    return pd.DataFrame(rows), saved_event_sets


# =========================================================
# MAIN
# =========================================================
def main():
    if mt5 is None:
        raise RuntimeError("No MT5 backend import succeeded. Install MetaTrader5 or mt5linux.")

    init_mt5()
    try:
        TF_H1 = mt5.TIMEFRAME_H1
        TF_D1 = mt5.TIMEFRAME_D1

        print(f"[INFO] Using symbol={SYMBOL}, H1={BARS_H1}, D1={BARS_D1}")
        h1 = load_rates(SYMBOL, TF_H1, BARS_H1)
        d1 = load_rates(SYMBOL, TF_D1, BARS_D1)

        h1 = add_gmt_columns(h1, SERVER_TO_GMT_HOURS)
        d1 = add_gmt_columns(d1, SERVER_TO_GMT_HOURS)

        h1 = build_h1_features(h1)
        d1 = build_daily_features(d1)
        df = merge_daily_into_h1(h1, d1)

        print("[INFO] Running full RSI 23:00 grid sweep...")
        summary, event_sets = run_full_grid(df)

        if summary.empty:
            raise RuntimeError("No summary rows produced.")

        summary.to_csv(f"{OUTPUT_PREFIX}_summary_grid.csv", index=False)

        valid = summary[summary["count"] >= MIN_SAMPLE_FILTER].copy()
        if valid.empty:
            print(f"[WARN] No combinations met MIN_SAMPLE_FILTER={MIN_SAMPLE_FILTER}.")
            print(summary.sort_values("count", ascending=False).head(30).round(2).to_string(index=False))
            return

        best_by_final = valid.sort_values("avg_final_move", ascending=False).head(25)
        best_by_fav = valid.sort_values("avg_favorable", ascending=False).head(25)
        best_by_200hit = valid.sort_values(["hit_200", "avg_final_move"], ascending=[False, False]).head(25)
        best_by_300hit = valid.sort_values(["hit_300", "avg_final_move"], ascending=[False, False]).head(25)

        print("\n=== BEST BY AVG FINAL MOVE ===")
        print(best_by_final[[
            "atr_period", "rsi_gate", "atr_threshold", "session_start", "session_end", "entry_hour",
            "count", "avg_final_move", "avg_favorable", "avg_adverse", "hit_150", "hit_200", "hit_300", "hit_400"
        ]].round(2).to_string(index=False))

        print("\n=== BEST BY AVG FAVORABLE MOVE ===")
        print(best_by_fav[[
            "atr_period", "rsi_gate", "atr_threshold", "session_start", "session_end", "entry_hour",
            "count", "avg_final_move", "avg_favorable", "avg_adverse", "hit_150", "hit_200", "hit_300", "hit_400"
        ]].round(2).to_string(index=False))

        print("\n=== BEST BY HIT RATE TO 200 ===")
        print(best_by_200hit[[
            "atr_period", "rsi_gate", "atr_threshold", "session_start", "session_end", "entry_hour",
            "count", "avg_final_move", "avg_favorable", "avg_adverse", "hit_150", "hit_200", "hit_300"
        ]].round(2).to_string(index=False))

        print("\n=== BEST BY HIT RATE TO 300 ===")
        print(best_by_300hit[[
            "atr_period", "rsi_gate", "atr_threshold", "session_start", "session_end", "entry_hour",
            "count", "avg_final_move", "avg_favorable", "avg_adverse", "hit_200", "hit_250", "hit_300", "hit_400"
        ]].round(2).to_string(index=False))

        # pick top combo by avg_final_move for stop/target screening
        top = best_by_final.iloc[0]
        key = f"atr{int(top['atr_period'])}_rsi{int(top['rsi_gate'])}_atr{int(top['atr_threshold'])}_ss{int(top['session_start'])}_se{int(top['session_end'])}_eh{int(top['entry_hour'])}"
        top_events = event_sets.get(key, pd.DataFrame())

        if not top_events.empty:
            top_events.to_csv(f"{OUTPUT_PREFIX}_best_events.csv", index=False)

            print("\n=== TOP COMBO SELECTED FOR STOP/TARGET SCREEN ===")
            print(top.round(2).to_string())

            st = build_stop_target_grid(top_events)
            st.to_csv(f"{OUTPUT_PREFIX}_best_stop_target_grid.csv", index=False)

            best_st = st.sort_values("screening_score", ascending=False).head(25)
            print("\n=== BEST STOP/TARGET SCREEN FOR TOP COMBO ===")
            print(best_st.round(2).to_string(index=False))

            print("\n=== TOP COMBO EVENT SUMMARY ===")
            print(top_events[["max_favorable_points", "max_adverse_points", "final_move_points"]].describe().round(2).to_string())

            # breakeven candidates from excursion profile
            be_rows = []
            for be_trigger in [80, 100, 120, 150, 180, 200, 220, 250, 300]:
                be_rows.append({
                    "breakeven_trigger": be_trigger,
                    "fraction_reaching_trigger": (top_events["max_favorable_points"] >= be_trigger).mean(),
                    "fraction_pullback_exceeding_trigger_after_entry_proxy": (top_events["max_adverse_points"] >= be_trigger).mean(),
                })
            be_df = pd.DataFrame(be_rows)
            be_df.to_csv(f"{OUTPUT_PREFIX}_breakeven_candidates.csv", index=False)

            print("\n=== BREAKEVEN CANDIDATES ===")
            print(be_df.round(3).to_string(index=False))

        print("\nSaved files:")
        print(f" - {OUTPUT_PREFIX}_summary_grid.csv")
        if not top_events.empty:
            print(f" - {OUTPUT_PREFIX}_best_events.csv")
            print(f" - {OUTPUT_PREFIX}_best_stop_target_grid.csv")
            print(f" - {OUTPUT_PREFIX}_breakeven_candidates.csv")

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
