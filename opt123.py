#!/usr/bin/env python3
import os
import warnings
from itertools import product

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
MIN_SAMPLE_FILTER = int(os.getenv("MIN_SAMPLE_FILTER", "15"))
OUTPUT_PREFIX = os.getenv("OUTPUT_PREFIX", "rsi2300_fastopt")

# Primary search space
RSI_LEVELS = list(range(55, 86, 2))
ATR_THRESHOLDS = list(range(60, 221, 10))
ATR_PERIODS = [7, 14]

# Test optimal ATR scan window
SESSION_STARTS = [8, 10, 12, 13, 14, 15]
SESSION_ENDS = [16, 18, 20, 21, 22, 23]

# Test optimal activation hour near your current region
ENTRY_HOURS = [21, 22, 23, 0, 1]

STOP_CANDIDATES = [80, 100, 120, 150, 180, 200, 220, 250, 300]
TARGET_CANDIDATES = [100, 120, 150, 180, 200, 250, 300, 350, 400, 500, 600]

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
# MT5
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
    return out

# =========================================================
# FEATURE BUILD
# =========================================================
def build_features(h1: pd.DataFrame, d1: pd.DataFrame):
    d = d1.copy()
    d["gmt_date"] = d["time_gmt"].dt.date
    d["d_rsi"] = rsi_wilder(d["close"], DAILY_RSI_PERIOD)

    h = h1.copy()
    h["gmt_date"] = h["time_gmt"].dt.date
    h["atr7"] = atr_wilder(h, 7)
    h["atr14"] = atr_wilder(h, 14)
    h["is_red"] = (h["close"] < h["open"]).astype(int)

    d_small = d[["gmt_date", "d_rsi"]].copy()
    h = h.merge(d_small, on="gmt_date", how="left")
    return h, d_small

# =========================================================
# PRECOMPUTE SESSION FLAGS
# =========================================================
def precompute_session_flags(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    session_pairs = [(s, e) for s in SESSION_STARTS for e in SESSION_ENDS if e > s]

    grouped = h.groupby("gmt_date", sort=True)

    for gmt_date, day_df in grouped:
        day_df = day_df.sort_values("time_gmt").copy()

        for s, e in session_pairs:
            scan = day_df[(day_df["gmt_hour"] >= s) & (day_df["gmt_hour"] < e)]
            if scan.empty:
                continue

            for atr_period in ATR_PERIODS:
                atr_col = f"atr{atr_period}"
                red_scan = scan[scan["is_red"] == 1]

                if red_scan.empty:
                    continue

                atr_values = red_scan[atr_col].dropna().values
                if len(atr_values) == 0:
                    continue

                rows.append({
                    "gmt_date": gmt_date,
                    "session_start": s,
                    "session_end": e,
                    "atr_period": atr_period,
                    "max_red_atr": float(np.max(atr_values)),
                    "count_red_candles": int(len(atr_values)),
                })

    flags = pd.DataFrame(rows)
    if flags.empty:
        raise RuntimeError("No session flags were built.")
    return flags

# =========================================================
# PRECOMPUTE ENTRY CANDIDATES
# =========================================================
def build_entry_candidates(h: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(h) - HOLD_BARS - 1):
        row = h.iloc[i]
        future = h.iloc[i + 1:i + 1 + HOLD_BARS]
        if future.empty:
            continue

        rows.append({
            "idx": i,
            "signal_time": row["time"],
            "signal_time_gmt": row["time_gmt"],
            "gmt_date": row["gmt_date"],
            "entry_hour": int(row["gmt_hour"]),
            "d_rsi": row["d_rsi"],
            "entry_time": h.iloc[i + 1]["time"],
            "entry": float(h.iloc[i + 1]["open"]),
            "max_favorable_points": float(future["high"].max() - h.iloc[i + 1]["open"]),
            "max_adverse_points": float(h.iloc[i + 1]["open"] - future["low"].min()),
            "final_move_points": float(future.iloc[-1]["close"] - h.iloc[i + 1]["open"]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No entry candidates built.")
    return out

# =========================================================
# SUMMARY
# =========================================================
def summarize_subset(sub: pd.DataFrame) -> dict:
    if sub.empty:
        return {
            "count": 0,
            "avg_favorable": np.nan,
            "avg_adverse": np.nan,
            "avg_final_move": np.nan,
            "median_favorable": np.nan,
            "median_adverse": np.nan,
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
        "avg_adverse": adv.mean(),
        "avg_final_move": fin.mean(),
        "median_favorable": fav.median(),
        "median_adverse": adv.median(),
        "median_final_move": fin.median(),
        "hit_100": (fav >= 100).mean(),
        "hit_150": (fav >= 150).mean(),
        "hit_200": (fav >= 200).mean(),
        "hit_250": (fav >= 250).mean(),
        "hit_300": (fav >= 300).mean(),
        "hit_400": (fav >= 400).mean(),
    }

# =========================================================
# FAST SWEEP
# =========================================================
def run_fast_sweep(entry_candidates: pd.DataFrame, session_flags: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    total = len(RSI_LEVELS) * len(ATR_THRESHOLDS) * len(ATR_PERIODS) * len([1 for s in SESSION_STARTS for e in SESSION_ENDS if e > s]) * len(ENTRY_HOURS)
    done = 0

    for atr_period in ATR_PERIODS:
        sfp = session_flags[session_flags["atr_period"] == atr_period].copy()

        for s, e in [(a, b) for a in SESSION_STARTS for b in SESSION_ENDS if b > a]:
            sf = sfp[(sfp["session_start"] == s) & (sfp["session_end"] == e)].copy()
            if sf.empty:
                continue

            merged = entry_candidates.merge(
                sf[["gmt_date", "max_red_atr", "count_red_candles"]],
                on="gmt_date",
                how="left"
            )

            for entry_hour in ENTRY_HOURS:
                m2 = merged[merged["entry_hour"] == entry_hour].copy()
                if m2.empty:
                    done += len(RSI_LEVELS) * len(ATR_THRESHOLDS)
                    continue

                for rsi_gate in RSI_LEVELS:
                    m3 = m2[m2["d_rsi"] >= rsi_gate].copy()
                    if m3.empty:
                        done += len(ATR_THRESHOLDS)
                        continue

                    for atr_threshold in ATR_THRESHOLDS:
                        sub = m3[m3["max_red_atr"] >= atr_threshold].copy()
                        stats = summarize_subset(sub)

                        rows.append({
                            "atr_period": atr_period,
                            "rsi_gate": rsi_gate,
                            "atr_threshold": atr_threshold,
                            "session_start": s,
                            "session_end": e,
                            "entry_hour": entry_hour,
                            **stats
                        })

                        done += 1
                        if done % 250 == 0:
                            print(f"[INFO] Progress: {done}/{total} combinations complete...")

    summary = pd.DataFrame(rows)
    valid = summary[summary["count"] >= MIN_SAMPLE_FILTER].copy()
    return summary, valid

# =========================================================
# STOP / TARGET SCREEN
# =========================================================
def simulate_screening(sub: pd.DataFrame, stop_points: float, target_points: float) -> dict:
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

        print("[INFO] Building features...")
        h, d = build_features(h1, d1)

        print("[INFO] Precomputing session flags...")
        session_flags = precompute_session_flags(h)

        print("[INFO] Precomputing entry candidates...")
        entry_candidates = build_entry_candidates(h)

        print("[INFO] Running fast optimization sweep...")
        summary, valid = run_fast_sweep(entry_candidates, session_flags)

        if summary.empty:
            raise RuntimeError("No summary rows produced.")

        summary.to_csv(f"{OUTPUT_PREFIX}_summary_grid.csv", index=False)

        if valid.empty:
            print(f"[WARN] No combinations met MIN_SAMPLE_FILTER={MIN_SAMPLE_FILTER}.")
            print(summary.sort_values("count", ascending=False).head(30).round(2).to_string(index=False))
            return

        best_by_final = valid.sort_values("avg_final_move", ascending=False).head(25)
        best_by_hit200 = valid.sort_values(["hit_200", "avg_final_move"], ascending=[False, False]).head(25)
        best_by_hit300 = valid.sort_values(["hit_300", "avg_final_move"], ascending=[False, False]).head(25)

        print("\n=== BEST BY AVG FINAL MOVE ===")
        print(best_by_final[[
            "atr_period", "rsi_gate", "atr_threshold", "session_start", "session_end", "entry_hour",
            "count", "avg_final_move", "avg_favorable", "avg_adverse", "hit_150", "hit_200", "hit_300", "hit_400"
        ]].round(2).to_string(index=False))

        print("\n=== BEST BY HIT RATE TO 200 ===")
        print(best_by_hit200[[
            "atr_period", "rsi_gate", "atr_threshold", "session_start", "session_end", "entry_hour",
            "count", "avg_final_move", "avg_favorable", "avg_adverse", "hit_150", "hit_200", "hit_300"
        ]].round(2).to_string(index=False))

        print("\n=== BEST BY HIT RATE TO 300 ===")
        print(best_by_hit300[[
            "atr_period", "rsi_gate", "atr_threshold", "session_start", "session_end", "entry_hour",
            "count", "avg_final_move", "avg_favorable", "avg_adverse", "hit_200", "hit_250", "hit_300", "hit_400"
        ]].round(2).to_string(index=False))

        top = best_by_final.iloc[0]
        print("\n=== TOP COMBINATION ===")
        print(top.round(2).to_string())

        # rebuild best subset quickly
        sf = session_flags[
            (session_flags["atr_period"] == int(top["atr_period"])) &
            (session_flags["session_start"] == int(top["session_start"])) &
            (session_flags["session_end"] == int(top["session_end"]))
        ][["gmt_date", "max_red_atr", "count_red_candles"]].copy()

        subset = entry_candidates.merge(sf, on="gmt_date", how="left")
        subset = subset[
            (subset["entry_hour"] == int(top["entry_hour"])) &
            (subset["d_rsi"] >= float(top["rsi_gate"])) &
            (subset["max_red_atr"] >= float(top["atr_threshold"]))
        ].copy()

        subset.to_csv(f"{OUTPUT_PREFIX}_best_events.csv", index=False)

        st = build_stop_target_grid(subset)
        st.to_csv(f"{OUTPUT_PREFIX}_best_stop_target_grid.csv", index=False)

        best_st = st.sort_values("screening_score", ascending=False).head(25)
        print("\n=== BEST STOP/TARGET SCREEN ===")
        print(best_st.round(2).to_string(index=False))

        be_rows = []
        for be_trigger in [80, 100, 120, 150, 180, 200, 220, 250, 300]:
            be_rows.append({
                "breakeven_trigger": be_trigger,
                "fraction_reaching_trigger": (subset["max_favorable_points"] >= be_trigger).mean(),
                "fraction_adverse_at_least_trigger": (subset["max_adverse_points"] >= be_trigger).mean(),
            })
        be_df = pd.DataFrame(be_rows)
        be_df.to_csv(f"{OUTPUT_PREFIX}_breakeven_candidates.csv", index=False)

        print("\n=== BREAKEVEN CANDIDATES ===")
        print(be_df.round(3).to_string(index=False))

        print("\nSaved files:")
        print(f" - {OUTPUT_PREFIX}_summary_grid.csv")
        print(f" - {OUTPUT_PREFIX}_best_events.csv")
        print(f" - {OUTPUT_PREFIX}_best_stop_target_grid.csv")
        print(f" - {OUTPUT_PREFIX}_breakeven_candidates.csv")

    finally:
        shutdown_mt5()

if __name__ == "__main__":
    main()
