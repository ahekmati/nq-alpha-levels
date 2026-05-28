#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mt5linux import MetaTrader5

warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False


OUTPUT_DIR = Path("atr_dip_study_mnq")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class StrategyConfig:
    atr_period: int = 7
    thresholds: tuple = (100, 110, 120, 130, 140, 150, 160)
    sustain_levels: tuple = (120, 130, 140)
    sustain_bars_choices: tuple = (2, 3)
    stop_points: int = 200
    take_profit_points: int = 400
    max_hold_bars: int = 24
    breakeven_trigger: int = 200
    breakeven_offset: int = 20
    slippage_points: int = 2
    hmm_states: int = 3
    daily_fast_ma: int = 20
    daily_slow_ma: int = 50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="@MNQ")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18812)
    p.add_argument("--stop-points", type=int, default=200)
    p.add_argument("--take-profit-points", type=int, default=400)
    p.add_argument("--max-hold-bars", type=int, default=24)
    return p.parse_args()


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


def fetch_rates(mt5, symbol: str, timeframe: int, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def choose_symbol(mt5, preferred: str, start_dt: datetime, end_dt: datetime) -> str:
    candidates = [preferred, "@MNQ", "MNQ", "MNQM26", "MNQU26"]
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    for sym in candidates:
        try:
            info = mt5.symbol_info(sym)
            if info is None:
                continue
            mt5.symbol_select(sym, True)
            test = mt5.copy_rates_range(sym, mt5.TIMEFRAME_H1, start_dt, end_dt)
            if test is not None and len(test) > 100:
                return sym
        except Exception:
            pass
    raise RuntimeError("Could not resolve MNQ symbol in MT5.")


def build_daily_context(d1: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = d1.copy()
    df["ret1"] = df["close"].pct_change(1)
    df["ret5"] = df["close"].pct_change(5)
    df["vol10"] = df["ret1"].rolling(10).std()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]

    df["ma_fast"] = df["close"].rolling(cfg.daily_fast_ma).mean()
    df["ma_slow"] = df["close"].rolling(cfg.daily_slow_ma).mean()
    df["ma_fast_slope"] = df["ma_fast"] - df["ma_fast"].shift(3)
    df["ma_slow_slope"] = df["ma_slow"] - df["ma_slow"].shift(3)

    df["bull_ma_structure"] = (
        (df["close"] > df["ma_fast"]) &
        (df["ma_fast"] > df["ma_slow"]) &
        (df["ma_fast_slope"] > 0) &
        (df["ma_slow_slope"] > 0)
    ).astype(int)

    if HMM_AVAILABLE:
        feat = df[["ret1", "vol10", "range_pct"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        X = feat.values
        hmm = GaussianHMM(n_components=cfg.hmm_states, covariance_type="full", n_iter=300, random_state=42)
        hmm.fit(X)
        states = hmm.predict(X)

        feat = feat.copy()
        feat["state"] = states
        state_perf = feat.groupby("state")["ret1"].mean().sort_values()
        bull_state = state_perf.index[-1]

        df["hmm_state"] = np.nan
        df.loc[feat.index, "hmm_state"] = feat["state"]
        df["hmm_bull"] = (df["hmm_state"] == bull_state).astype(int)
    else:
        df["hmm_state"] = np.nan
        df["hmm_bull"] = df["bull_ma_structure"]

    df["bull_structure_strict"] = ((df["bull_ma_structure"] == 1) & (df["hmm_bull"] == 1)).astype(int)

    keep = [
        "time",
        "bull_ma_structure",
        "hmm_bull",
        "bull_structure_strict",
        "ma_fast",
        "ma_slow",
        "ma_fast_slope",
        "ma_slow_slope",
        "ret1",
        "ret5",
        "vol10",
        "range_pct",
    ]
    out = df[keep].copy()
    out["date"] = out["time"].dt.floor("D")
    out = out.drop(columns=["time"])

    for c in out.columns:
        if c != "date":
            out[c] = out[c].shift(1)

    return out


def build_hourly_features(h1: pd.DataFrame, daily_ctx: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    df = h1.copy()
    df["atr7"] = atr(df, cfg.atr_period)
    df["bar_range"] = df["high"] - df["low"]
    df["bar_body"] = df["close"] - df["open"]
    df["bear_bar"] = (df["close"] < df["open"]).astype(int)
    df["bull_bar"] = (df["close"] > df["open"]).astype(int)
    df["hour"] = df["time"].dt.hour
    df["date"] = df["time"].dt.floor("D")

    df["h_ma12"] = df["close"].rolling(12).mean()
    df["h_ma24"] = df["close"].rolling(24).mean()
    df["h_ma48"] = df["close"].rolling(48).mean()
    df["dist_h_ma12"] = df["close"] - df["h_ma12"]
    df["dist_h_ma24"] = df["close"] - df["h_ma24"]
    df["dist_h_ma48"] = df["close"] - df["h_ma48"]

    df = df.merge(daily_ctx, on="date", how="left")
    return df


def build_signals(df: pd.DataFrame, threshold: int, mode: str, sustain_level: int | None = None, sustain_bars: int | None = None) -> pd.Series:
    atr_now = df["atr7"]
    atr_prev = df["atr7"].shift(1)

    first_cross = (atr_prev < threshold) & (atr_now >= threshold) & (df["bear_bar"] == 1)

    if mode == "first_cross":
        return first_cross.astype(int)

    if mode == "sustain":
        if sustain_level is None or sustain_bars is None:
            return pd.Series(0, index=df.index)
        elevated = (df["atr7"] >= sustain_level).astype(int)
        sustained = elevated.rolling(sustain_bars).sum() >= sustain_bars
        return (first_cross & sustained).astype(int)

    if mode == "reversal_after_cross":
        crossed_recently = first_cross.rolling(3).max().fillna(0) > 0
        reversal = (df["bull_bar"] == 1) & (df["close"] > df["high"].shift(1))
        return (crossed_recently & reversal).astype(int)

    return pd.Series(0, index=df.index)


def simulate_trade(df: pd.DataFrame, idx: int, cfg: StrategyConfig) -> dict | None:
    if idx + 1 >= len(df):
        return None

    entry_bar = df.iloc[idx + 1]
    entry_time = entry_bar["time"]
    entry_price = entry_bar["open"] + cfg.slippage_points

    stop_price = entry_price - cfg.stop_points
    tp_price = entry_price + cfg.take_profit_points

    best_price = entry_price
    moved_to_be = False

    for j in range(idx + 1, min(idx + 1 + cfg.max_hold_bars, len(df))):
        bar = df.iloc[j]
        high_ = bar["high"]
        low_ = bar["low"]

        if not moved_to_be and (high_ - entry_price) >= cfg.breakeven_trigger:
            stop_price = max(stop_price, entry_price + cfg.breakeven_offset)
            moved_to_be = True

        stop_hit = low_ <= stop_price
        tp_hit = high_ >= tp_price

        if stop_hit and tp_hit:
            exit_price = stop_price
            exit_reason = "stop_same_bar"
            exit_time = bar["time"]
            pnl_points = exit_price - entry_price
            return {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_points": pnl_points,
                "bars_held": j - idx,
                "exit_reason": exit_reason,
            }

        if stop_hit:
            exit_price = stop_price
            exit_reason = "stop"
            exit_time = bar["time"]
            pnl_points = exit_price - entry_price
            return {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_points": pnl_points,
                "bars_held": j - idx,
                "exit_reason": exit_reason,
            }

        if tp_hit:
            exit_price = tp_price
            exit_reason = "tp"
            exit_time = bar["time"]
            pnl_points = exit_price - entry_price
            return {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_points": pnl_points,
                "bars_held": j - idx,
                "exit_reason": exit_reason,
            }

        best_price = max(best_price, high_)

    exit_bar = df.iloc[min(idx + cfg.max_hold_bars, len(df) - 1)]
    exit_price = exit_bar["close"]
    exit_time = exit_bar["time"]
    pnl_points = exit_price - entry_price
    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_points": pnl_points,
        "bars_held": min(cfg.max_hold_bars, len(df) - idx - 1),
        "exit_reason": "time_exit",
    }


def run_backtest(df: pd.DataFrame, signal_col: str, cfg: StrategyConfig) -> tuple[pd.DataFrame, dict]:
    trades = []
    i = 0
    while i < len(df) - 1:
        row = df.iloc[i]
        if row[signal_col] != 1:
            i += 1
            continue

        if row["bull_structure_strict"] != 1:
            i += 1
            continue

        tr = simulate_trade(df, i, cfg)
        if tr is not None:
            tr["signal_time"] = row["time"]
            tr["atr7_signal"] = row["atr7"]
            tr["signal_col"] = signal_col
            trades.append(tr)
            i += max(1, int(tr["bars_held"]))
        else:
            i += 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = {
            "signal_col": signal_col,
            "trades": 0,
            "win_rate": np.nan,
            "avg_pnl_points": np.nan,
            "total_pnl_points": 0.0,
            "profit_factor": np.nan,
            "avg_bars_held": np.nan,
            "max_drawdown_points": np.nan,
        }
        return trades_df, summary

    trades_df["cum_pnl_points"] = trades_df["pnl_points"].cumsum()
    trades_df["equity_peak"] = trades_df["cum_pnl_points"].cummax()
    trades_df["drawdown_points"] = trades_df["cum_pnl_points"] - trades_df["equity_peak"]

    gross_profit = trades_df.loc[trades_df["pnl_points"] > 0, "pnl_points"].sum()
    gross_loss = -trades_df.loc[trades_df["pnl_points"] < 0, "pnl_points"].sum()

    summary = {
        "signal_col": signal_col,
        "trades": int(len(trades_df)),
        "win_rate": float((trades_df["pnl_points"] > 0).mean()),
        "avg_pnl_points": float(trades_df["pnl_points"].mean()),
        "median_pnl_points": float(trades_df["pnl_points"].median()),
        "total_pnl_points": float(trades_df["pnl_points"].sum()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
        "avg_bars_held": float(trades_df["bars_held"].mean()),
        "max_drawdown_points": float(trades_df["drawdown_points"].min()),
    }
    return trades_df, summary


def recovery_stats(df: pd.DataFrame, event_col: str, stop_points: int = 200, targets=(100, 150, 200, 300, 400), horizons=(6, 12, 24, 48)) -> pd.DataFrame:
    rows = []
    event_idx = np.where(df[event_col].fillna(0).values == 1)[0]

    for idx in event_idx:
        entry_idx = idx + 1
        if entry_idx >= len(df):
            continue
        entry = df.iloc[entry_idx]["open"]

        for hz in horizons:
            future = df.iloc[entry_idx:min(entry_idx + hz, len(df))]
            if future.empty:
                continue

            min_low = float(future["low"].min())
            max_high = float(future["high"].max())

            adverse = min_low - entry
            favorable = max_high - entry

            row = {
                "event_time": df.iloc[idx]["time"],
                "event_col": event_col,
                "horizon_bars": hz,
                "entry_price": entry,
                "max_favorable": favorable,
                "max_adverse": adverse,
            }

            for t in targets:
                row[f"hit_plus_{t}"] = int(favorable >= t)
                row[f"survive_minus_{stop_points}_and_hit_plus_{t}"] = int((adverse > -stop_points) and (favorable >= t))

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    cfg = StrategyConfig(
        stop_points=args.stop_points,
        take_profit_points=args.take_profit_points,
        max_hold_bars=args.max_hold_bars,
    )

    end_dt = datetime.now(timezone.utc) if args.end is None else pd.Timestamp(args.end, tz="UTC").to_pydatetime()
    start_dt = pd.Timestamp(args.start, tz="UTC").to_pydatetime()

    mt5 = MetaTrader5(host=args.host, port=args.port)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    symbol = choose_symbol(mt5, args.symbol, start_dt, end_dt)
    mt5.symbol_select(symbol, True)

    d1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_D1, start_dt, end_dt)
    h1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)

    mt5.shutdown()

    if d1.empty or h1.empty:
        raise RuntimeError("No D1 or H1 data returned.")

    daily_ctx = build_daily_context(d1, cfg)
    df = build_hourly_features(h1, daily_ctx, cfg).replace([np.inf, -np.inf], np.nan)

    signal_names = []

    for thr in cfg.thresholds:
        col = f"sig_first_cross_{thr}"
        df[col] = build_signals(df, threshold=thr, mode="first_cross")
        signal_names.append(col)

        rev_col = f"sig_reversal_after_cross_{thr}"
        df[rev_col] = build_signals(df, threshold=thr, mode="reversal_after_cross")
        signal_names.append(rev_col)

    for sustain_level in cfg.sustain_levels:
        for sustain_bars in cfg.sustain_bars_choices:
            thr = sustain_level
            col = f"sig_sustain_{sustain_level}_{sustain_bars}"
            df[col] = build_signals(
                df,
                threshold=thr,
                mode="sustain",
                sustain_level=sustain_level,
                sustain_bars=sustain_bars,
            )
            signal_names.append(col)

    summaries = []
    all_trades = []
    all_recovery = []

    for sig in signal_names:
        trades_df, summary = run_backtest(df, sig, cfg)
        summaries.append(summary)

        if not trades_df.empty:
            tmp = trades_df.copy()
            tmp["strategy"] = sig
            all_trades.append(tmp)

        rec = recovery_stats(df, sig, stop_points=cfg.stop_points)
        if not rec.empty:
            agg = rec.drop(columns=["event_time", "entry_price"]).groupby(["event_col", "horizon_bars"]).mean().reset_index()
            all_recovery.append(agg)

    summary_df = pd.DataFrame(summaries).sort_values(
        by=["profit_factor", "avg_pnl_points", "win_rate"],
        ascending=[False, False, False]
    )

    trades_out = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    recovery_out = pd.concat(all_recovery, ignore_index=True) if all_recovery else pd.DataFrame()

    meta = {
        "symbol_used": symbol,
        "start": str(start_dt),
        "end": str(end_dt),
        "n_d1_bars": int(len(d1)),
        "n_h1_bars": int(len(h1)),
        "hmm_available": HMM_AVAILABLE,
        "stop_points": cfg.stop_points,
        "take_profit_points": cfg.take_profit_points,
        "max_hold_bars": cfg.max_hold_bars,
    }

    df.to_csv(OUTPUT_DIR / "study_dataset.csv", index=False)
    summary_df.to_csv(OUTPUT_DIR / "strategy_summary.csv", index=False)
    trades_out.to_csv(OUTPUT_DIR / "all_trades.csv", index=False)
    recovery_out.to_csv(OUTPUT_DIR / "recovery_stats.csv", index=False)

    with open(OUTPUT_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))

    print("\n" + "=" * 100)
    print("TOP ATR DIP STRATEGIES (bullish daily structure + HMM bull)")
    print("=" * 100)
    print(summary_df.head(20).to_string(index=False))

    if not recovery_out.empty:
        print("\n" + "=" * 100)
        print("RECOVERY STATS (mean hit rates by event/horizon)")
        print("=" * 100)
        print(recovery_out.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
