#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from mt5linux import MetaTrader5
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

OUTDIR = Path("research_mnq_swing_meanrev")
OUTDIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Config
# ----------------------------

@dataclass
class StrategyParams:
    regime_mode: str               # "bull_only", "bear_only", "both"
    daily_ma_len: int              # 50, 100, 150
    daily_rsi_len: int             # 10, 14
    daily_rsi_bull_min: float      # e.g. 52
    daily_rsi_bear_max: float      # e.g. 48
    z_window: int                  # 24, 36, 48
    z_long_entry: float            # e.g. -1.25
    z_short_entry: float           # e.g. 1.25
    h1_rsi_len: int                # 7, 10, 14
    h1_rsi_long_max: float         # e.g. 38
    h1_rsi_short_min: float        # e.g. 62
    atr_len: int                   # 14
    stop_atr_mult: float           # e.g. 2.2
    tp_atr_mult: float             # e.g. 3.0
    max_hold_bars: int             # e.g. 36
    min_reentry_gap_bars: int      # e.g. 4
    use_vwap_filter: bool          # True/False
    vwap_buffer_atr: float         # e.g. 0.15


# ----------------------------
# Indicators
# ----------------------------

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
    num = (tp * vol).rolling(window).sum()
    den = vol.rolling(window).sum()
    return num / den.replace(0, np.nan)

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity - peak
    return float(dd.min()) if len(dd) else 0.0


# ----------------------------
# MT5 Data
# ----------------------------

def fetch_rates(mt5, symbol: str, timeframe: int, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)

def choose_symbol(mt5, preferred: str, start_dt: datetime, end_dt: datetime) -> str:
    candidates = [preferred, "MNQ", "@MNQ", "MNQM26", "MNQU26"]
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    for sym in candidates:
        try:
            info = mt5.symbol_info(sym)
            if info is None:
                continue
            mt5.symbol_select(sym, True)
            test = mt5.copy_rates_range(sym, mt5.TIMEFRAME_H1, start_dt, end_dt)
            if test is not None and len(test) > 500:
                return sym
        except Exception:
            pass
    raise RuntimeError("Could not resolve a usable MNQ symbol from MT5.")


# ----------------------------
# Feature Prep
# ----------------------------

def build_daily_context(d1: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    d = d1.copy()
    d["d_ma"] = d["close"].rolling(p.daily_ma_len).mean()
    d["d_rsi"] = rsi(d["close"], p.daily_rsi_len)
    d["d_atr"] = atr(d, 14)
    d["d_ret5"] = d["close"].pct_change(5)
    d["d_ret20"] = d["close"].pct_change(20)

    d["bull_regime"] = (
        (d["close"] > d["d_ma"]) &
        (d["d_rsi"] >= p.daily_rsi_bull_min)
    ).astype(int)

    d["bear_regime"] = (
        (d["close"] < d["d_ma"]) &
        (d["d_rsi"] <= p.daily_rsi_bear_max)
    ).astype(int)

    d["date"] = d["time"].dt.floor("D")
    keep = ["date", "bull_regime", "bear_regime", "d_ma", "d_rsi", "d_atr", "d_ret5", "d_ret20"]
    return d[keep].copy()

def build_hourly_features(h1: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    h = h1.copy()
    h["h_atr"] = atr(h, p.atr_len)
    h["h_rsi"] = rsi(h["close"], p.h1_rsi_len)
    h["z"] = zscore(h["close"], p.z_window)
    h["ret1"] = h["close"].pct_change(1)
    h["ret3"] = h["close"].pct_change(3)
    h["ret6"] = h["close"].pct_change(6)
    h["ma_fast"] = h["close"].rolling(12).mean()
    h["ma_slow"] = h["close"].rolling(36).mean()
    h["dist_fast_atr"] = (h["close"] - h["ma_fast"]) / h["h_atr"].replace(0, np.nan)
    h["dist_slow_atr"] = (h["close"] - h["ma_slow"]) / h["h_atr"].replace(0, np.nan)
    h["vwap24"] = rolling_vwap(h, 24)
    h["vwap_dist_atr"] = (h["close"] - h["vwap24"]) / h["h_atr"].replace(0, np.nan)
    h["hour"] = h["time"].dt.hour
    h["dow"] = h["time"].dt.dayofweek
    h["date"] = h["time"].dt.floor("D")
    return h

def merge_context(h1f: pd.DataFrame, dctx: pd.DataFrame) -> pd.DataFrame:
    df = h1f.merge(dctx, on="date", how="left")
    return df.sort_values("time").reset_index(drop=True)


# ----------------------------
# Signal Logic
# ----------------------------

def build_signals(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    x = df.copy()

    bull_ok = x["bull_regime"] == 1
    bear_ok = x["bear_regime"] == 1

    if p.regime_mode == "bull_only":
        long_regime = bull_ok
        short_regime = pd.Series(False, index=x.index)
    elif p.regime_mode == "bear_only":
        long_regime = pd.Series(False, index=x.index)
        short_regime = bear_ok
    else:
        long_regime = bull_ok
        short_regime = bear_ok

    long_sig = (
        long_regime &
        (x["z"] <= p.z_long_entry) &
        (x["h_rsi"] <= p.h1_rsi_long_max) &
        (x["dist_fast_atr"] < -0.25)
    )

    short_sig = (
        short_regime &
        (x["z"] >= p.z_short_entry) &
        (x["h_rsi"] >= p.h1_rsi_short_min) &
        (x["dist_fast_atr"] > 0.25)
    )

    if p.use_vwap_filter:
        long_sig &= (x["vwap_dist_atr"] <= -p.vwap_buffer_atr)
        short_sig &= (x["vwap_dist_atr"] >= p.vwap_buffer_atr)

    x["long_signal"] = long_sig.astype(int)
    x["short_signal"] = short_sig.astype(int)
    return x


# ----------------------------
# Backtest
# ----------------------------

def run_backtest(df: pd.DataFrame, p: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    equity_rows = []

    pos = 0          # 1 long, -1 short, 0 flat
    entry_price = None
    entry_time = None
    stop_price = None
    tp_price = None
    bars_in_trade = 0
    last_exit_idx = -10_000
    equity = 0.0

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]

        current_atr = row["h_atr"]
        if not np.isfinite(current_atr) or current_atr <= 0:
            equity_rows.append({"time": row["time"], "equity_points": equity})
            continue

        # manage open trade intrabar using next bar range as crude fill model
        if pos != 0:
            bars_in_trade += 1

            exit_reason = None
            exit_price = None

            if pos == 1:
                if nxt["low"] <= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price
                elif nxt["high"] >= tp_price:
                    exit_reason = "target"
                    exit_price = tp_price
                elif row["short_signal"] == 1:
                    exit_reason = "reverse"
                    exit_price = nxt["open"]
                elif bars_in_trade >= p.max_hold_bars:
                    exit_reason = "time"
                    exit_price = nxt["open"]

                if exit_reason is not None:
                    pnl = exit_price - entry_price
                    equity += pnl
                    rows.append({
                        "entry_time": entry_time,
                        "exit_time": nxt["time"],
                        "side": "LONG",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "bars_held": bars_in_trade,
                        "pnl_points": pnl,
                        "exit_reason": exit_reason
                    })
                    pos = 0
                    entry_price = None
                    stop_price = None
                    tp_price = None
                    bars_in_trade = 0
                    last_exit_idx = i

            elif pos == -1:
                if nxt["high"] >= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price
                elif nxt["low"] <= tp_price:
                    exit_reason = "target"
                    exit_price = tp_price
                elif row["long_signal"] == 1:
                    exit_reason = "reverse"
                    exit_price = nxt["open"]
                elif bars_in_trade >= p.max_hold_bars:
                    exit_reason = "time"
                    exit_price = nxt["open"]

                if exit_reason is not None:
                    pnl = entry_price - exit_price
                    equity += pnl
                    rows.append({
                        "entry_time": entry_time,
                        "exit_time": nxt["time"],
                        "side": "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "bars_held": bars_in_trade,
                        "pnl_points": pnl,
                        "exit_reason": exit_reason
                    })
                    pos = 0
                    entry_price = None
                    stop_price = None
                    tp_price = None
                    bars_in_trade = 0
                    last_exit_idx = i

        # enter only if flat and enough bars since last exit
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

        equity_rows.append({"time": row["time"], "equity_points": equity})

    trades = pd.DataFrame(rows)
    equity_df = pd.DataFrame(equity_rows)

    if not trades.empty:
        trades["pnl_usd_1_contract"] = trades["pnl_points"] * 2.0  # MNQ multiplier
    else:
        trades["pnl_usd_1_contract"] = []

    return trades, equity_df


# ----------------------------
# Metrics
# ----------------------------

def summarize_trades(trades: pd.DataFrame, equity_df: pd.DataFrame, p: StrategyParams) -> dict:
    if trades.empty:
        return {
            **asdict(p),
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win_points": 0.0,
            "avg_loss_points": 0.0,
            "profit_factor": 0.0,
            "expectancy_points": 0.0,
            "median_hold_hours": 0.0,
            "mean_hold_hours": 0.0,
            "pct_stop": 0.0,
            "pct_target": 0.0,
            "pct_reverse": 0.0,
            "total_points": 0.0,
            "max_drawdown_points": 0.0,
        }

    wins = trades.loc[trades["pnl_points"] > 0, "pnl_points"]
    losses = trades.loc[trades["pnl_points"] < 0, "pnl_points"]

    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else np.nan

    holds = (pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])).dt.total_seconds() / 3600.0
    eq = equity_df["equity_points"] if not equity_df.empty else pd.Series(dtype=float)

    return {
        **asdict(p),
        "n_trades": int(len(trades)),
        "win_rate": float((trades["pnl_points"] > 0).mean()),
        "avg_win_points": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_points": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(pf) if np.isfinite(pf) else 0.0,
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
    # approximate target profile from your analyzed logs
    target = {
        "n_trades_per_year": 97.0,
        "win_rate": 0.474,
        "median_hold_hours": 23.5,
        "avg_win_to_avg_loss_abs": 2.08,     # ~5288 / 2546 from prior analysis
        "pct_stop": 0.32
    }

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

    return x.sort_values(
        ["profit_factor", "total_points", "behavior_similarity_score"],
        ascending=[False, False, True]
    )


# ----------------------------
# Walk-forward
# ----------------------------

def walk_forward_score(df: pd.DataFrame, p: StrategyParams, splits: int = 5) -> dict:
    x = build_signals(df, p)
    x = x.dropna().reset_index(drop=True)

    if len(x) < 2000:
        trades, equity = run_backtest(x, p)
        s = summarize_trades(trades, equity, p)
        s["wf_profit_factor_mean"] = s["profit_factor"]
        s["wf_expectancy_mean"] = s["expectancy_points"]
        return s

    tscv = TimeSeriesSplit(n_splits=splits)
    fold_rows = []

    for fold, (_, test_idx) in enumerate(tscv.split(x), start=1):
        test_df = x.iloc[test_idx].copy().reset_index(drop=True)
        trades, equity = run_backtest(test_df, p)
        s = summarize_trades(trades, equity, p)
        s["fold"] = fold
        fold_rows.append(s)

    wf = pd.DataFrame(fold_rows)
    out = {**asdict(p)}
    out["wf_profit_factor_mean"] = float(wf["profit_factor"].mean())
    out["wf_expectancy_mean"] = float(wf["expectancy_points"].mean())
    out["wf_win_rate_mean"] = float(wf["win_rate"].mean())
    out["wf_n_trades_mean"] = float(wf["n_trades"].mean())

    # full-period summary too
    trades_all, equity_all = run_backtest(x, p)
    full_s = summarize_trades(trades_all, equity_all, p)
    out.update(full_s)
    return out


# ----------------------------
# Grid
# ----------------------------

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
        [38, 42],
        [58, 62],
        [14],
        [2.0, 2.2],
        [2.8, 3.2],
        [24, 36],
        [4, 8],
        [False, True],
        [0.10, 0.20],
    ):
        yield StrategyParams(*vals)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="MNQ")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18812)
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    start_dt = pd.Timestamp(args.start, tz="UTC").to_pydatetime()
    end_dt = datetime.now(timezone.utc) if args.end is None else pd.Timestamp(args.end, tz="UTC").to_pydatetime()

    mt5 = MetaTrader5(host=args.host, port=args.port)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    try:
        symbol = choose_symbol(mt5, args.symbol, start_dt, end_dt)
        mt5.symbol_select(symbol, True)

        d1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_D1, start_dt, end_dt)
        h1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)

        if d1.empty or h1.empty:
            raise RuntimeError("No D1/H1 data returned from MT5.")

        # save raw bars
        d1.to_csv(OUTDIR / "d1_bars.csv", index=False)
        h1.to_csv(OUTDIR / "h1_bars.csv", index=False)

        # base context for all runs
        summaries = []
        full_trade_store = []

        for k, p in enumerate(parameter_grid(), start=1):
            dctx = build_daily_context(d1, p)
            hfeat = build_hourly_features(h1, p)
            df = merge_context(hfeat, dctx)

            s = walk_forward_score(df, p, splits=5)
            s["run_id"] = k
            s["symbol_used"] = symbol
            summaries.append(s)

            # keep trades for top later by rescoring with same params
            # we'll regenerate for finalists
            if k % 10 == 0:
                print(f"Completed {k} parameter sets...")

        summary_df = pd.DataFrame(summaries)
        scored = compare_to_log_profile(summary_df)

        summary_df.to_csv(OUTDIR / "all_runs_summary.csv", index=False)
        scored.to_csv(OUTDIR / "ranked_runs_summary.csv", index=False)

        top = scored.head(args.topn).copy()

        # regenerate detailed trades for top configs
        detailed_summaries = []
        for _, row in top.iterrows():
            p = StrategyParams(
                regime_mode=row["regime_mode"],
                daily_ma_len=int(row["daily_ma_len"]),
                daily_rsi_len=int(row["daily_rsi_len"]),
                daily_rsi_bull_min=float(row["daily_rsi_bull_min"]),
                daily_rsi_bear_max=float(row["daily_rsi_bear_max"]),
                z_window=int(row["z_window"]),
                z_long_entry=float(row["z_long_entry"]),
                z_short_entry=float(row["z_short_entry"]),
                h1_rsi_len=int(row["h1_rsi_len"]),
                h1_rsi_long_max=float(row["h1_rsi_long_max"]),
                h1_rsi_short_min=float(row["h1_rsi_short_min"]),
                atr_len=int(row["atr_len"]),
                stop_atr_mult=float(row["stop_atr_mult"]),
                tp_atr_mult=float(row["tp_atr_mult"]),
                max_hold_bars=int(row["max_hold_bars"]),
                min_reentry_gap_bars=int(row["min_reentry_gap_bars"]),
                use_vwap_filter=bool(row["use_vwap_filter"]),
                vwap_buffer_atr=float(row["vwap_buffer_atr"]),
            )
            dctx = build_daily_context(d1, p)
            hfeat = build_hourly_features(h1, p)
            df = merge_context(hfeat, dctx)
            df = build_signals(df, p).dropna().reset_index(drop=True)

            trades, equity = run_backtest(df, p)
            tag = f"run_{int(row['run_id'])}"
            trades.to_csv(OUTDIR / f"{tag}_trades.csv", index=False)
            equity.to_csv(OUTDIR / f"{tag}_equity.csv", index=False)

            rec = row.to_dict()
            rec["trade_file"] = f"{tag}_trades.csv"
            rec["equity_file"] = f"{tag}_equity.csv"
            detailed_summaries.append(rec)

        pd.DataFrame(detailed_summaries).to_csv(OUTDIR / "top_runs_with_files.csv", index=False)

        meta = {
            "symbol_used": symbol,
            "start": str(start_dt),
            "end": str(end_dt),
            "n_d1_bars": int(len(d1)),
            "n_h1_bars": int(len(h1)),
            "n_runs": int(len(summary_df)),
            "topn_saved": int(len(top)),
            "notes": [
                "Fresh scratch-built regime-gated swing mean-reversion research script",
                "Walk-forward scoring uses TimeSeriesSplit",
                "Behavior similarity score compares to inferred legacy log profile"
            ]
        }
        with open(OUTDIR / "run_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(json.dumps(meta, indent=2))
        print("\nTop 10 by ranked summary:")
        print(scored.head(10)[[
            "run_id", "profit_factor", "total_points", "n_trades",
            "win_rate", "median_hold_hours", "pct_stop", "behavior_similarity_score"
        ]].to_string(index=False))

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
