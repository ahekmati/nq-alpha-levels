#!/usr/bin/env python3
import os
import warnings
from typing import Optional, List, Dict, Tuple

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

SIGNAL_TF_NAME = os.getenv("SIGNAL_TF_NAME", "H1")
HOLD_BARS = int(os.getenv("HOLD_BARS", "72"))

FAST_MA = int(os.getenv("FAST_MA", "26"))
SLOW_MA = int(os.getenv("SLOW_MA", "150"))

H1_ATR_PERIOD = int(os.getenv("H1_ATR_PERIOD", "14"))
H1_MIN_ATR = float(os.getenv("H1_MIN_ATR", "100.0"))
MIN_PULLBACK_ATR = float(os.getenv("MIN_PULLBACK_ATR", "0.50"))

DAILY_RSI_PERIOD = int(os.getenv("DAILY_RSI_PERIOD", "10"))
DAILY_ATR_FAST_PERIOD = int(os.getenv("DAILY_ATR_FAST_PERIOD", "7"))
DAILY_ATR_STD_PERIOD = int(os.getenv("DAILY_ATR_STD_PERIOD", "14"))

# Parameter sweeps
RSI_LEVELS = list(range(50, 86, 2))               # 50, 52, ..., 84
ATR_GATE_LEVELS = list(range(150, 701, 25))       # 150 .. 700

# Stop/target exploration
STOP_CANDIDATES = [60, 80, 100, 120, 150, 180, 200, 220, 250, 300]
TARGET_CANDIDATES = [80, 100, 120, 150, 180, 200, 250, 300, 350, 400, 500, 600]

OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "dailydip_param_sweep")


# =========================================================
# INDICATORS
# =========================================================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def smma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

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

def close_location(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["low"]) / rng

def body_ratio(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["open"]).abs() / rng

def bars_since_last_true(flag: pd.Series) -> pd.Series:
    out = np.full(len(flag), np.nan)
    last_idx = None
    vals = flag.fillna(False).astype(bool).values
    for i, v in enumerate(vals):
        if v:
            last_idx = i
            out[i] = 0
        else:
            out[i] = np.nan if last_idx is None else i - last_idx
    return pd.Series(out, index=flag.index)


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
# FEATURE BUILD
# =========================================================
def build_daily_features(d1: pd.DataFrame) -> pd.DataFrame:
    d = d1.copy()
    d["rsi10"] = rsi_wilder(d["close"], DAILY_RSI_PERIOD)
    d["atr7"] = atr_wilder(d, DAILY_ATR_FAST_PERIOD)
    d["atr14"] = atr_wilder(d, DAILY_ATR_STD_PERIOD)
    d["ema50"] = ema(d["close"], 50)
    d["ema100"] = ema(d["close"], 100)
    d["ema200"] = ema(d["close"], 200)
    d["close_loc"] = close_location(d)
    d["body_ratio"] = body_ratio(d)
    d["prev_high"] = d["high"].shift(1)
    d["prev_low"] = d["low"].shift(1)
    d["prev_close"] = d["close"].shift(1)
    return d

def build_h1_features(h1: pd.DataFrame) -> pd.DataFrame:
    h = h1.copy()
    h["fast_ma"] = smma(h["close"], FAST_MA)
    h["slow_ma"] = smma(h["close"], SLOW_MA)
    h["atr14"] = atr_wilder(h, H1_ATR_PERIOD)
    h["is_red"] = (h["close"] < h["open"]).astype(int)
    h["is_green"] = (h["close"] > h["open"]).astype(int)
    h["rolling_10_high"] = h["high"].rolling(10).max().shift(1)
    h["rolling_20_high"] = h["high"].rolling(20).max().shift(1)
    h["pullback_depth_points"] = h["rolling_10_high"] - h["close"]
    h["pullback_depth_atr"] = h["pullback_depth_points"] / h["atr14"].replace(0, np.nan)
    h["bars_since_20bar_high"] = bars_since_last_true(h["high"] >= h["rolling_20_high"])
    h["close_loc"] = close_location(h)
    h["body_ratio"] = body_ratio(h)
    return h

def merge_daily_into_h1(h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    d = d1.copy()
    d["day"] = d["time_gmt"].dt.date

    h = h1.copy()
    h["day"] = h["time_gmt"].dt.date

    daily_cols = [
        "day", "rsi10", "atr7", "atr14", "ema50", "ema100", "ema200",
        "prev_high", "prev_low", "prev_close", "weekday"
    ]
    d = d[daily_cols].rename(columns={
        "atr7": "d_atr7",
        "atr14": "d_atr14",
        "weekday": "d_weekday"
    })

    m = h.merge(d, on="day", how="left")
    m["close_above_d_100"] = (m["close"] > m["ema100"]).astype(int)
    m["close_above_d_200"] = (m["close"] > m["ema200"]).astype(int)
    m["dist_prev_day_high"] = m["close"] - m["prev_high"]
    m["dist_prev_day_low"] = m["close"] - m["prev_low"]
    m["dist_prev_day_close"] = m["close"] - m["prev_close"]
    return m


# =========================================================
# BASE DAILY DIP EVENT LOGIC
# =========================================================
def base_daily_dip_signal(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    if i < 30:
        return False
    if pd.isna(row["atr14"]) or pd.isna(row["rsi10"]) or pd.isna(row["d_atr7"]) or pd.isna(row["d_atr14"]):
        return False
    if row["atr14"] < H1_MIN_ATR:
        return False
    if row["is_red"] != 1:
        return False
    if row["fast_ma"] <= row["slow_ma"]:
        return False
    if row["close_above_d_100"] != 1 and row["close_above_d_200"] != 1:
        return False
    if pd.isna(row["pullback_depth_atr"]) or row["pullback_depth_atr"] < MIN_PULLBACK_ATR:
        return False
    return True


# =========================================================
# EVENT EXTRACTION
# =========================================================
def build_event_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)

    for i in range(n - HOLD_BARS - 2):
        if not base_daily_dip_signal(df, i):
            continue

        sig = df.iloc[i]
        nxt = df.iloc[i + 1]
        future = df.iloc[i + 1:i + 1 + HOLD_BARS].copy()
        if future.empty:
            continue

        entry = float(nxt["open"])
        max_up = float(future["high"].max() - entry)
        max_down = float(entry - future["low"].min())
        final_move = float(future.iloc[-1]["close"] - entry)

        rows.append({
            "signal_time": sig["time"],
            "entry_time": nxt["time"],
            "entry": entry,
            "d_rsi10": sig["rsi10"],
            "d_atr7": sig["d_atr7"],
            "d_atr14": sig["d_atr14"],
            "h1_atr14": sig["atr14"],
            "pullback_depth_atr": sig["pullback_depth_atr"],
            "bars_since_20bar_high": sig["bars_since_20bar_high"],
            "close_loc": sig["close_loc"],
            "body_ratio": sig["body_ratio"],
            "dist_prev_day_high": sig["dist_prev_day_high"],
            "dist_prev_day_low": sig["dist_prev_day_low"],
            "dist_prev_day_close": sig["dist_prev_day_close"],
            "max_favorable_points": max_up,
            "max_adverse_points": max_down,
            "final_move_points": final_move,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("signal_time").reset_index(drop=True)


# =========================================================
# GATE SWEEP
# =========================================================
def apply_gates(events: pd.DataFrame, min_rsi: float, atr_col: str, max_atr: float) -> pd.DataFrame:
    return events[
        (events["d_rsi10"] >= min_rsi) &
        (events[atr_col] <= max_atr)
    ].copy()

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
    }

def build_rsi_sweep(events: pd.DataFrame, atr_col: str, fixed_atr_gate: float) -> pd.DataFrame:
    rows = []
    for rsi_level in RSI_LEVELS:
        sub = apply_gates(events, rsi_level, atr_col, fixed_atr_gate)
        stats = summarize_subset(sub)
        rows.append({
            "atr_col": atr_col,
            "fixed_atr_gate": fixed_atr_gate,
            "rsi_min": rsi_level,
            **stats
        })
    return pd.DataFrame(rows)

def build_atr_sweep(events: pd.DataFrame, fixed_rsi_gate: float, atr_col: str) -> pd.DataFrame:
    rows = []
    for atr_gate in ATR_GATE_LEVELS:
        sub = apply_gates(events, fixed_rsi_gate, atr_col, atr_gate)
        stats = summarize_subset(sub)
        rows.append({
            "atr_col": atr_col,
            "fixed_rsi_gate": fixed_rsi_gate,
            "atr_max": atr_gate,
            **stats
        })
    return pd.DataFrame(rows)

def build_joint_grid(events: pd.DataFrame, atr_col: str) -> pd.DataFrame:
    rows = []
    for rsi_level in RSI_LEVELS:
        for atr_gate in ATR_GATE_LEVELS:
            sub = apply_gates(events, rsi_level, atr_col, atr_gate)
            stats = summarize_subset(sub)

            score = np.nan
            if stats["count"] >= 20 and pd.notna(stats["avg_final_move"]):
                score = stats["avg_final_move"]

            rows.append({
                "atr_col": atr_col,
                "rsi_min": rsi_level,
                "atr_max": atr_gate,
                "score_avg_final_move": score,
                **stats
            })
    return pd.DataFrame(rows)


# =========================================================
# STOP/TARGET STUDY
# =========================================================
def simulate_first_touch(sub: pd.DataFrame, stop_points: float, target_points: float) -> Dict:
    # Approximation using excursions only:
    # counts target as feasible if MFE >= target
    # counts stop as feasible if MAE >= stop
    # if both hit within hold window, sequence is unknown in this compressed view
    # so this is a screening tool, not a perfect execution simulator.
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

    # rough screening score, not exact expectancy
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
            rows.append(simulate_first_touch(sub, stop, target))
    return pd.DataFrame(rows)


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

        events = build_event_dataset(df)
        if events.empty:
            raise RuntimeError("No Daily Dip events found. Relax the base signal rules or load more history.")

        print(f"[INFO] Base Daily Dip events found: {len(events)}")
        print("\n[INFO] Base event summary:")
        print(events[["max_favorable_points", "max_adverse_points", "final_move_points"]].describe().round(2).to_string())

        # Baseline using your current idea: RSI >= 70 and ATR <= 550
        baseline_atr7 = apply_gates(events, min_rsi=70, atr_col="d_atr7", max_atr=550)
        baseline_atr14 = apply_gates(events, min_rsi=70, atr_col="d_atr14", max_atr=550)

        print(f"\n[INFO] Baseline gates RSI>=70 & ATR7<=550 count:  {len(baseline_atr7)}")
        print(f"[INFO] Baseline gates RSI>=70 & ATR14<=550 count: {len(baseline_atr14)}")

        # 1) RSI sweep holding ATR gate at 550
        rsi_sweep_atr7 = build_rsi_sweep(events, "d_atr7", 550)
        rsi_sweep_atr14 = build_rsi_sweep(events, "d_atr14", 550)

        # 2) ATR sweep holding RSI at 70
        atr_sweep_atr7 = build_atr_sweep(events, 70, "d_atr7")
        atr_sweep_atr14 = build_atr_sweep(events, 70, "d_atr14")

        # 3) Joint grid for best combinations
        joint_atr7 = build_joint_grid(events, "d_atr7")
        joint_atr14 = build_joint_grid(events, "d_atr14")

        # Best by avg final move, with minimum sample threshold
        min_count = 20
        best_rsi_atr7 = rsi_sweep_atr7[rsi_sweep_atr7["count"] >= min_count].sort_values("avg_final_move", ascending=False).head(10)
        best_rsi_atr14 = rsi_sweep_atr14[rsi_sweep_atr14["count"] >= min_count].sort_values("avg_final_move", ascending=False).head(10)

        best_atr7 = atr_sweep_atr7[atr_sweep_atr7["count"] >= min_count].sort_values("avg_final_move", ascending=False).head(10)
        best_atr14 = atr_sweep_atr14[atr_sweep_atr14["count"] >= min_count].sort_values("avg_final_move", ascending=False).head(10)

        best_joint_atr7 = joint_atr7[joint_atr7["count"] >= min_count].sort_values("avg_final_move", ascending=False).head(20)
        best_joint_atr14 = joint_atr14[joint_atr14["count"] >= min_count].sort_values("avg_final_move", ascending=False).head(20)

        print("\n=== BEST RSI GATES WITH ATR7<=550 ===")
        print(best_rsi_atr7[[
            "rsi_min", "count", "avg_final_move", "avg_favorable", "avg_adverse",
            "hit_150", "hit_200", "hit_250", "hit_300"
        ]].round(2).to_string(index=False))

        print("\n=== BEST RSI GATES WITH ATR14<=550 ===")
        print(best_rsi_atr14[[
            "rsi_min", "count", "avg_final_move", "avg_favorable", "avg_adverse",
            "hit_150", "hit_200", "hit_250", "hit_300"
        ]].round(2).to_string(index=False))

        print("\n=== BEST ATR7 GATES WITH RSI>=70 ===")
        print(best_atr7[[
            "atr_max", "count", "avg_final_move", "avg_favorable", "avg_adverse",
            "hit_150", "hit_200", "hit_250", "hit_300"
        ]].round(2).to_string(index=False))

        print("\n=== BEST ATR14 GATES WITH RSI>=70 ===")
        print(best_atr14[[
            "atr_max", "count", "avg_final_move", "avg_favorable", "avg_adverse",
            "hit_150", "hit_200", "hit_250", "hit_300"
        ]].round(2).to_string(index=False))

        print("\n=== BEST JOINT RSI/ATR7 COMBINATIONS ===")
        print(best_joint_atr7[[
            "rsi_min", "atr_max", "count", "avg_final_move",
            "avg_favorable", "avg_adverse", "hit_200", "hit_250", "hit_300"
        ]].round(2).to_string(index=False))

        print("\n=== BEST JOINT RSI/ATR14 COMBINATIONS ===")
        print(best_joint_atr14[[
            "rsi_min", "atr_max", "count", "avg_final_move",
            "avg_favorable", "avg_adverse", "hit_200", "hit_250", "hit_300"
        ]].round(2).to_string(index=False))

        # Use best joint subsets for stop/target study
        best_joint_row_7 = best_joint_atr7.iloc[0] if not best_joint_atr7.empty else None
        best_joint_row_14 = best_joint_atr14.iloc[0] if not best_joint_atr14.empty else None

        if best_joint_row_7 is not None:
            sub7 = apply_gates(events, best_joint_row_7["rsi_min"], "d_atr7", best_joint_row_7["atr_max"])
            stop_target_7 = build_stop_target_grid(sub7)
            best_stop_target_7 = stop_target_7.sort_values("screening_score", ascending=False).head(20)
            print("\n=== ATR7 BEST STOP/TARGET SCREEN ===")
            print(best_stop_target_7.round(2).to_string(index=False))
        else:
            stop_target_7 = pd.DataFrame()

        if best_joint_row_14 is not None:
            sub14 = apply_gates(events, best_joint_row_14["rsi_min"], "d_atr14", best_joint_row_14["atr_max"])
            stop_target_14 = build_stop_target_grid(sub14)
            best_stop_target_14 = stop_target_14.sort_values("screening_score", ascending=False).head(20)
            print("\n=== ATR14 BEST STOP/TARGET SCREEN ===")
            print(best_stop_target_14.round(2).to_string(index=False))
        else:
            stop_target_14 = pd.DataFrame()

        # Save outputs
        events.to_csv(f"{OUTPUT_PREFIX}_events.csv", index=False)
        rsi_sweep_atr7.to_csv(f"{OUTPUT_PREFIX}_rsi_sweep_atr7.csv", index=False)
        rsi_sweep_atr14.to_csv(f"{OUTPUT_PREFIX}_rsi_sweep_atr14.csv", index=False)
        atr_sweep_atr7.to_csv(f"{OUTPUT_PREFIX}_atr_sweep_atr7.csv", index=False)
        atr_sweep_atr14.to_csv(f"{OUTPUT_PREFIX}_atr_sweep_atr14.csv", index=False)
        joint_atr7.to_csv(f"{OUTPUT_PREFIX}_joint_grid_atr7.csv", index=False)
        joint_atr14.to_csv(f"{OUTPUT_PREFIX}_joint_grid_atr14.csv", index=False)

        if not stop_target_7.empty:
            stop_target_7.to_csv(f"{OUTPUT_PREFIX}_stop_target_atr7.csv", index=False)
        if not stop_target_14.empty:
            stop_target_14.to_csv(f"{OUTPUT_PREFIX}_stop_target_atr14.csv", index=False)

        print("\nSaved files:")
        print(f" - {OUTPUT_PREFIX}_events.csv")
        print(f" - {OUTPUT_PREFIX}_rsi_sweep_atr7.csv")
        print(f" - {OUTPUT_PREFIX}_rsi_sweep_atr14.csv")
        print(f" - {OUTPUT_PREFIX}_atr_sweep_atr7.csv")
        print(f" - {OUTPUT_PREFIX}_atr_sweep_atr14.csv")
        print(f" - {OUTPUT_PREFIX}_joint_grid_atr7.csv")
        print(f" - {OUTPUT_PREFIX}_joint_grid_atr14.csv")
        if not stop_target_7.empty:
            print(f" - {OUTPUT_PREFIX}_stop_target_atr7.csv")
        if not stop_target_14.empty:
            print(f" - {OUTPUT_PREFIX}_stop_target_atr14.csv")

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
