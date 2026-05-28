#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import signal
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

OUTDIR = Path("research_mnq_swing_meanrev_recovery")
OUTDIR.mkdir(parents=True, exist_ok=True)

STOP_REQUESTED = False

def _handle_sigint(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\nSIGINT received. Will stop after current parameter set and save partial results...")

signal.signal(signal.SIGINT, _handle_sigint)

@dataclass
class StrategyParams:
    regime_mode: str
    daily_ma_len: int
    daily_rsi_len: int
    daily_rsi_bull_min: float
    daily_rsi_bear_max: float
    z_window: int
    z_long_entry: float
    z_short_entry: float
    h1_rsi_len: int
    h1_rsi_long_max: float
    h1_rsi_short_min: float
    atr_len: int
    stop_atr_mult: float
    tp_atr_mult: float
    max_hold_bars: int
    min_reentry_gap_bars: int
    use_vwap_filter: bool
    vwap_buffer_atr: float

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / period, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd.replace(0, np.nan)

def rolling_vwap(df: pd.DataFrame, window: int = 24) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["tick_volume"].replace(0, np.nan).fillna(1.0)
    return (tp * vol).rolling(window).sum() / vol.rolling(window).sum().replace(0, np.nan)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else 0.0

def build_daily_context(d1: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    d = d1.copy()
    d["d_ma"] = d["close"].rolling(p.daily_ma_len).mean()
    d["d_rsi"] = rsi(d["close"], p.daily_rsi_len)
    d["d_atr"] = atr(d, 14)
    d["bull_regime"] = ((d["close"] > d["d_ma"]) & (d["d_rsi"] >= p.daily_rsi_bull_min)).astype(int)
    d["bear_regime"] = ((d["close"] < d["d_ma"]) & (d["d_rsi"] <= p.daily_rsi_bear_max)).astype(int)
    d["date"] = d["time"].dt.floor("D")
    return d[["date", "bull_regime", "bear_regime", "d_ma", "d_rsi", "d_atr"]].copy()

def build_hourly_features(h1: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    h = h1.copy()
    h["h_atr"] = atr(h, p.atr_len)
    h["h_rsi"] = rsi(h["close"], p.h1_rsi_len)
    h["z"] = zscore(h["close"], p.z_window)
    h["ma_fast"] = h["close"].rolling(12).mean()
    h["vwap24"] = rolling_vwap(h, 24)
    h["dist_fast_atr"] = (h["close"] - h["ma_fast"]) / h["h_atr"].replace(0, np.nan)
    h["vwap_dist_atr"] = (h["close"] - h["vwap24"]) / h["h_atr"].replace(0, np.nan)
    h["date"] = h["time"].dt.floor("D")
    return h

def merge_context(h1f: pd.DataFrame, dctx: pd.DataFrame) -> pd.DataFrame:
    return h1f.merge(dctx, on="date", how="left").sort_values("time").reset_index(drop=True)

def build_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    x = df.copy()
    if p.regime_mode == "both":
        long_regime = x["bull_regime"] == 1
        short_regime = x["bear_regime"] == 1
    elif p.regime_mode == "bull_only":
        long_regime = x["bull_regime"] == 1
        short_regime = pd.Series(False, index=x.index)
    else:
        long_regime = pd.Series(False, index=x.index)
        short_regime = x["bear_regime"] == 1

    long_sig = long_regime & (x["z"] <= p.z_long_entry) & (x["h_rsi"] <= p.h1_rsi_long_max) & (x["dist_fast_atr"] < -0.25)
    short_sig = short_regime & (x["z"] >= p.z_short_entry) & (x["h_rsi"] >= p.h1_rsi_short_min) & (x["dist_fast_atr"] > 0.25)

    if p.use_vwap_filter:
        long_sig &= (x["vwap_dist_atr"] <= -p.vwap_buffer_atr)
        short_sig &= (x["vwap_dist_atr"] >= p.vwap_buffer_atr)

    x["long_signal"] = long_sig.astype(int)
    x["short_signal"] = short_sig.astype(int)
    return x.dropna().reset_index(drop=True)

def run_backtest(df: pd.DataFrame, p: StrategyParams):
    rows = []
    eq_rows = []
    pos = 0
    entry_price = None
    entry_time = None
    stop_price = None
    tp_price = None
    bars_in_trade = 0
    last_exit_idx = -100000
    equity = 0.0

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        current_atr = row["h_atr"]
        if not np.isfinite(current_atr) or current_atr <= 0:
            eq_rows.append({"time": row["time"], "equity_points": equity})
            continue

        if pos != 0:
            bars_in_trade += 1
            exit_reason = None
            exit_price = None

            if pos == 1:
                if nxt["low"] <= stop_price:
                    exit_reason, exit_price = "stop", stop_price
                elif nxt["high"] >= tp_price:
                    exit_reason, exit_price = "target", tp_price
                elif row["short_signal"] == 1:
                    exit_reason, exit_price = "reverse", nxt["open"]
                elif bars_in_trade >= p.max_hold_bars:
                    exit_reason, exit_price = "time", nxt["open"]
                if exit_reason:
                    pnl = exit_price - entry_price
                    equity += pnl
                    rows.append({"entry_time": entry_time, "exit_time": nxt["time"], "side": "LONG", "entry_price": entry_price, "exit_price": exit_price, "bars_held": bars_in_trade, "pnl_points": pnl, "exit_reason": exit_reason})
                    pos = 0
                    entry_price = stop_price = tp_price = None
                    bars_in_trade = 0
                    last_exit_idx = i

            elif pos == -1:
                if nxt["high"] >= stop_price:
                    exit_reason, exit_price = "stop", stop_price
                elif nxt["low"] <= tp_price:
                    exit_reason, exit_price = "target", tp_price
                elif row["long_signal"] == 1:
                    exit_reason, exit_price = "reverse", nxt["open"]
                elif bars_in_trade >= p.max_hold_bars:
                    exit_reason, exit_price = "time", nxt["open"]
                if exit_reason:
                    pnl = entry_price - exit_price
                    equity += pnl
                    rows.append({"entry_time": entry_time, "exit_time": nxt["time"], "side": "SHORT", "entry_price": entry_price, "exit_price": exit_price, "bars_held": bars_in_trade, "pnl_points": pnl, "exit_reason": exit_reason})
                    pos = 0
                    entry_price = stop_price = tp_price = None
                    bars_in_trade = 0
                    last_exit_idx = i

        if pos == 0 and (i - last_exit_idx) >= p.min_reentry_gap_bars:
            if row["long_signal"] == 1 and row["short_signal"] == 0:
                pos = 1
                entry_price = nxt["open"]
                entry_time = nxt["time"]
                stop_price = entry_price - p.stop_atr_mult * current_atr
                tp_price = entry_price + p.tp_atr_mult * current_atr
                bars_in_trade = 0
            elif row["short_signal"] == 1 and row["long_signal"] == 0:
                pos = -1
                entry_price = nxt["open"]
                entry_time = nxt["time"]
                stop_price = entry_price + p.stop_atr_mult * current_atr
                tp_price = entry_price - p.tp_atr_mult * current_atr
                bars_in_trade = 0

        eq_rows.append({"time": row["time"], "equity_points": equity})

    return pd.DataFrame(rows), pd.DataFrame(eq_rows)

def summarize_trades(trades: pd.DataFrame, equity_df: pd.DataFrame, p: StrategyParams) -> dict:
    if trades.empty:
        return {**asdict(p), "n_trades": 0, "win_rate": 0.0, "avg_win_points": 0.0, "avg_loss_points": 0.0, "profit_factor": 0.0, "expectancy_points": 0.0, "median_hold_hours": 0.0, "mean_hold_hours": 0.0, "pct_stop": 0.0, "pct_target": 0.0, "pct_reverse": 0.0, "total_points": 0.0, "max_drawdown_points": 0.0}

    wins = trades.loc[trades["pnl_points"] > 0, "pnl_points"]
    losses = trades.loc[trades["pnl_points"] < 0, "pnl_points"]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    holds = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.total_seconds() / 3600.0
    eq = equity_df["equity_points"] if not equity_df.empty else pd.Series(dtype=float)
    pf = gross_win / gross_loss if gross_loss > 0 else 0.0
    return {
        **asdict(p),
        "n_trades": int(len(trades)),
        "win_rate": float((trades["pnl_points"] > 0).mean()),
        "avg_win_points": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_points": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(pf),
        "expectancy_points": float(trades["pnl_points"].mean()),
        "median_hold_hours": float(holds.median()),
        "mean_hold_hours": float(holds.mean()),
        "pct_stop": float((trades["exit_reason"] == "stop").mean()),
        "pct_target": float((trades["exit_reason"] == "target").mean()),
        "pct_reverse": float((trades["exit_reason"] == "reverse").mean()),
        "total_points": float(trades["pnl_points"].sum()),
        "max_drawdown_points": max_drawdown(eq),
    }

def compare_to_log_profile(summary_df: pd.DataFrame) -> pd.DataFrame:
    target = {"n_trades_per_year": 97.0, "win_rate": 0.474, "median_hold_hours": 23.5, "avg_win_to_avg_loss_abs": 2.08, "pct_stop": 0.32}
    x = summary_df.copy()
    years = max(1.0, (pd.Timestamp.utcnow() - pd.Timestamp("2020-01-01", tz="UTC")).days / 365.25)
    x["n_trades_per_year"] = x["n_trades"] / years
    x["avg_win_to_avg_loss_abs"] = x["avg_win_points"] / x["avg_loss_points"].abs().replace(0, np.nan)
    x["dist_trade_freq"] = (x["n_trades_per_year"] - target["n_trades_per_year"]).abs()
    x["dist_win_rate"] = (x["win_rate"] - target["win_rate"]).abs()
    x["dist_hold"] = (x["median_hold_hours"] - target["median_hold_hours"]).abs()
    x["dist_payoff"] = (x["avg_win_to_avg_loss_abs"] - target["avg_win_to_avg_loss_abs"]).abs()
    x["dist_stop"] = (x["pct_stop"] - target["pct_stop"]).abs()
    x["behavior_similarity_score"] = (
        x["dist_trade_freq"] * 0.15 +
        x["dist_win_rate"] * 100 * 0.20 +
        x["dist_hold"] * 0.10 +
        x["dist_payoff"] * 10 * 0.35 +
        x["dist_stop"] * 100 * 0.20
    )
    return x.sort_values(["profit_factor", "total_points", "behavior_similarity_score"], ascending=[False, False, True])

def parameter_grid():
    for vals in itertools.product(
        ["both"],
        [100, 150],
        [10],
        [52],
        [48],
        [24, 36],
        [-1.25, -1.5],
        [1.25, 1.5],
        [7, 10],
        [38],
        [62],
        [14],
        [2.0, 2.2],
        [2.8, 3.2],
        [24, 36],
        [4, 8],
        [False, True],
        [0.10],
    ):
        yield StrategyParams(*vals)

def save_outputs(summaries, top_n=15):
    if not summaries:
        return
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUTDIR / "all_runs_summary.csv", index=False)
    ranked = compare_to_log_profile(summary_df)
    ranked.to_csv(OUTDIR / "ranked_runs_summary.csv", index=False)
    ranked.head(top_n).to_csv(OUTDIR / "top_candidates.csv", index=False)

def main():
    d1 = pd.read_csv("research_mnq_swing_meanrev/d1_bars.csv")
    h1 = pd.read_csv("research_mnq_swing_meanrev/h1_bars.csv")
    d1["time"] = pd.to_datetime(d1["time"], utc=True)
    h1["time"] = pd.to_datetime(h1["time"], utc=True)

    combos = list(parameter_grid())
    total = len(combos)
    print(f"Total parameter sets: {total}")

    summaries = []

    for k, p in enumerate(combos, start=1):
        dctx = build_daily_context(d1, p)
        hfeat = build_hourly_features(h1, p)
        df = merge_context(hfeat, dctx)
        df = build_signals(df, p)
        trades, equity = run_backtest(df, p)
        s = summarize_trades(trades, equity, p)
        s["run_id"] = k
        summaries.append(s)

        if (k % 5 == 0) or (k == total):
            save_outputs(summaries)
            print(f"Completed {k}/{total} parameter sets...")

        if STOP_REQUESTED:
            break

    save_outputs(summaries)
    ranked = pd.read_csv(OUTDIR / "ranked_runs_summary.csv")
    print("\nTop 10:")
    print(ranked[["run_id", "profit_factor", "total_points", "n_trades", "win_rate", "median_hold_hours", "pct_stop", "behavior_similarity_score"]].head(10).to_string(index=False))

    meta = {
        "n_runs_completed": len(summaries),
        "total_possible": total,
        "d1_rows": int(len(d1)),
        "h1_rows": int(len(h1)),
        "stopped_early": bool(STOP_REQUESTED),
    }
    with open(OUTDIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved results to", OUTDIR)

if __name__ == "__main__":
    main()
