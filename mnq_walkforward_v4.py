#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

OUTDIR = Path("validation_run_441_v4")
OUTDIR.mkdir(parents=True, exist_ok=True)

MNQ_POINT_VALUE = 2.0
HARD_STOP_POINTS = 200.0
HARD_STOP_USD = 400.0
assert HARD_STOP_POINTS * MNQ_POINT_VALUE == HARD_STOP_USD


@dataclass
class StrategyParams:
    regime_mode: str = "both"
    daily_ma_len: int = 100
    daily_rsi_len: int = 10
    daily_rsi_bull_min: float = 52.0
    daily_rsi_bear_max: float = 48.0
    z_window: int = 36
    z_long_entry: float = -1.50
    z_short_entry: float = 1.25
    h1_rsi_len: int = 10
    h1_rsi_long_max: float = 38.0
    h1_rsi_short_min: float = 62.0
    atr_len: int = 14
    stop_atr_mult: float = 2.2
    tp_atr_mult: float = 3.2
    max_hold_bars: int = 24
    min_reentry_gap_bars: int = 4
    use_vwap_filter: bool = False
    vwap_buffer_atr: float = 0.10
    consecutive_loss_cooloff_bars: int = 48
    hard_stop_points: float = HARD_STOP_POINTS
    commission_per_side_usd: float = 0.62
    slippage_points_per_side: float = 0.5
    trail_after_tp: bool = True
    trail_atr_mult_after_tp: float = 1.1
    break_even_plus_points: float = 8.0


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
    long_regime = x["bull_regime"] == 1
    short_regime = x["bear_regime"] == 1

    long_sig = long_regime & (x["z"] <= p.z_long_entry) & (x["h_rsi"] <= p.h1_rsi_long_max) & (x["dist_fast_atr"] < -0.25)
    short_sig = short_regime & (x["z"] >= p.z_short_entry) & (x["h_rsi"] >= p.h1_rsi_short_min) & (x["dist_fast_atr"] > 0.25)

    if p.use_vwap_filter:
        long_sig &= (x["vwap_dist_atr"] <= -p.vwap_buffer_atr)
        short_sig &= (x["vwap_dist_atr"] >= p.vwap_buffer_atr)

    x["long_signal"] = long_sig.astype(int)
    x["short_signal"] = short_sig.astype(int)
    return x.dropna().reset_index(drop=True)


def prepare_dataset(d1: pd.DataFrame, h1: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    dctx = build_daily_context(d1, p)
    hfeat = build_hourly_features(h1, p)
    df = merge_context(hfeat, dctx)
    df = build_signals(df, p)
    return df


def apply_trade_costs(raw_pnl_points: float, p: StrategyParams) -> tuple[float, float]:
    total_slippage_points = 2 * p.slippage_points_per_side
    pnl_usd = (raw_pnl_points - total_slippage_points) * MNQ_POINT_VALUE - 2 * p.commission_per_side_usd
    pnl_points_net = pnl_usd / MNQ_POINT_VALUE
    return pnl_points_net, pnl_usd


def run_backtest(df: pd.DataFrame, p: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eq_rows = []
    pos = 0
    entry_price = entry_time = stop_price = None
    initial_tp = None
    trail_armed = False
    bars_in_trade = 0
    last_exit_idx = -100000
    equity_points = 0.0
    consec_losses = 0
    cooldown_until_idx = -1

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        nxt = df.iloc[i + 1]
        current_atr = row["h_atr"]

        if not np.isfinite(current_atr) or current_atr <= 0:
            eq_rows.append({"time": row["time"], "equity_points": equity_points})
            continue

        if pos != 0:
            bars_in_trade += 1
            exit_reason = None
            exit_price = None

            if pos == 1:
                if (not trail_armed) and nxt["high"] >= initial_tp:
                    trail_armed = True
                    stop_price = max(stop_price, entry_price + p.break_even_plus_points)
                    stop_price = max(stop_price, nxt["close"] - p.trail_atr_mult_after_tp * current_atr)
                elif trail_armed:
                    stop_price = max(stop_price, nxt["close"] - p.trail_atr_mult_after_tp * current_atr)

                if nxt["low"] <= stop_price:
                    exit_reason, exit_price = ("trail_stop" if trail_armed else "stop"), stop_price
                elif bars_in_trade >= p.max_hold_bars:
                    exit_reason, exit_price = "time", nxt["open"]

                if exit_reason is not None:
                    raw_pnl_points = exit_price - entry_price
                    pnl_points, pnl_usd = apply_trade_costs(raw_pnl_points, p)
                    equity_points += pnl_points
                    rows.append({
                        "entry_time": entry_time,
                        "exit_time": nxt["time"],
                        "side": "LONG",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "stop_price_final": stop_price,
                        "initial_tp": initial_tp,
                        "trail_armed": trail_armed,
                        "bars_held": bars_in_trade,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "exit_reason": exit_reason,
                    })
                    consec_losses = consec_losses + 1 if pnl_points < 0 else 0
                    if consec_losses >= 2:
                        cooldown_until_idx = i + p.consecutive_loss_cooloff_bars
                    pos = 0
                    entry_price = entry_time = stop_price = initial_tp = None
                    trail_armed = False
                    bars_in_trade = 0
                    last_exit_idx = i

            elif pos == -1:
                if (not trail_armed) and nxt["low"] <= initial_tp:
                    trail_armed = True
                    stop_price = min(stop_price, entry_price - p.break_even_plus_points)
                    stop_price = min(stop_price, nxt["close"] + p.trail_atr_mult_after_tp * current_atr)
                elif trail_armed:
                    stop_price = min(stop_price, nxt["close"] + p.trail_atr_mult_after_tp * current_atr)

                if nxt["high"] >= stop_price:
                    exit_reason, exit_price = ("trail_stop" if trail_armed else "stop"), stop_price
                elif bars_in_trade >= p.max_hold_bars:
                    exit_reason, exit_price = "time", nxt["open"]

                if exit_reason is not None:
                    raw_pnl_points = entry_price - exit_price
                    pnl_points, pnl_usd = apply_trade_costs(raw_pnl_points, p)
                    equity_points += pnl_points
                    rows.append({
                        "entry_time": entry_time,
                        "exit_time": nxt["time"],
                        "side": "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "stop_price_final": stop_price,
                        "initial_tp": initial_tp,
                        "trail_armed": trail_armed,
                        "bars_held": bars_in_trade,
                        "pnl_points": pnl_points,
                        "pnl_usd": pnl_usd,
                        "exit_reason": exit_reason,
                    })
                    consec_losses = consec_losses + 1 if pnl_points < 0 else 0
                    if consec_losses >= 2:
                        cooldown_until_idx = i + p.consecutive_loss_cooloff_bars
                    pos = 0
                    entry_price = entry_time = stop_price = initial_tp = None
                    trail_armed = False
                    bars_in_trade = 0
                    last_exit_idx = i

        if pos == 0 and i > cooldown_until_idx and (i - last_exit_idx) >= p.min_reentry_gap_bars:
            if row["long_signal"] == 1 and row["short_signal"] == 0:
                atr_stop = p.stop_atr_mult * current_atr
                actual_stop_dist = min(atr_stop, p.hard_stop_points)
                pos = 1
                entry_price = nxt["open"]
                entry_time = nxt["time"]
                stop_price = entry_price - actual_stop_dist
                initial_tp = entry_price + p.tp_atr_mult * current_atr
                trail_armed = False
                bars_in_trade = 0
            elif row["short_signal"] == 1 and row["long_signal"] == 0:
                atr_stop = p.stop_atr_mult * current_atr
                actual_stop_dist = min(atr_stop, p.hard_stop_points)
                pos = -1
                entry_price = nxt["open"]
                entry_time = nxt["time"]
                stop_price = entry_price + actual_stop_dist
                initial_tp = entry_price - p.tp_atr_mult * current_atr
                trail_armed = False
                bars_in_trade = 0

        eq_rows.append({"time": row["time"], "equity_points": equity_points})

    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
        trades["year"] = trades["entry_time"].dt.year
    equity = pd.DataFrame(eq_rows)
    if not equity.empty:
        equity["time"] = pd.to_datetime(equity["time"], utc=True)
        equity["equity_cum"] = equity["equity_points"]
    return trades, equity


def summarize(trades: pd.DataFrame, equity: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "n_trades": 0}
    wins = trades.loc[trades["pnl_points"] > 0, "pnl_points"]
    losses = trades.loc[trades["pnl_points"] < 0, "pnl_points"]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    peak = equity["equity_cum"].cummax()
    dd = equity["equity_cum"] - peak
    return {
        "label": label,
        "n_trades": int(len(trades)),
        "win_rate": float((trades["pnl_points"] > 0).mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else 0.0,
        "expectancy_points": float(trades["pnl_points"].mean()),
        "total_points": float(trades["pnl_points"].sum()),
        "total_usd": float(trades["pnl_usd"].sum()),
        "max_drawdown_points": float(dd.min()) if len(dd) else 0.0,
        "pct_stop_or_trail": float(trades["exit_reason"].isin(["stop", "trail_stop"]).mean()),
        "pct_trail_stop": float((trades["exit_reason"] == "trail_stop").mean()),
        "pct_time": float((trades["exit_reason"] == "time").mean()),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "median_bars_held": float(trades["bars_held"].median()),
    }


def walk_forward_validate(d1: pd.DataFrame, h1: pd.DataFrame, p: StrategyParams, train_years: int = 3, test_months: int = 12) -> pd.DataFrame:
    full_df = prepare_dataset(d1, h1, p)
    times = pd.to_datetime(full_df["time"], utc=True)
    start = times.min().floor("D")
    end = times.max().floor("D")
    folds = []

    train_delta = pd.DateOffset(years=train_years)
    test_delta = pd.DateOffset(months=test_months)
    fold_start = start
    fold_no = 1

    while True:
        train_end = fold_start + train_delta
        test_end = train_end + test_delta
        if test_end > end:
            break

        test_mask = (times >= train_end) & (times < test_end)
        test_df = full_df.loc[test_mask].copy().reset_index(drop=True)
        if len(test_df) < 200:
            break

        trades, equity = run_backtest(test_df, p)
        s = summarize(trades, equity, f"fold_{fold_no}")
        s["fold"] = fold_no
        s["train_start"] = str(fold_start.date())
        s["train_end"] = str(train_end.date())
        s["test_end"] = str(test_end.date())
        folds.append(s)

        fold_start = fold_start + pd.DateOffset(months=test_months)
        fold_no += 1

    return pd.DataFrame(folds)


def neighbor_params() -> list[StrategyParams]:
    base = StrategyParams()
    return [
        base,
        StrategyParams(tp_atr_mult=3.4),
        StrategyParams(tp_atr_mult=3.6),
        StrategyParams(tp_atr_mult=3.8),
        StrategyParams(stop_atr_mult=2.0),
        StrategyParams(stop_atr_mult=2.4),
        StrategyParams(z_window=32),
        StrategyParams(z_window=40),
        StrategyParams(trail_atr_mult_after_tp=1.0),
        StrategyParams(trail_atr_mult_after_tp=1.2),
        StrategyParams(commission_per_side_usd=0.62, slippage_points_per_side=1.0),
        StrategyParams(commission_per_side_usd=0.62, slippage_points_per_side=1.5),
    ]


def main():
    d1 = pd.read_csv("research_mnq_swing_meanrev/d1_bars.csv")
    h1 = pd.read_csv("research_mnq_swing_meanrev/h1_bars.csv")
    d1["time"] = pd.to_datetime(d1["time"], utc=True)
    h1["time"] = pd.to_datetime(h1["time"], utc=True)

    all_summaries = []
    all_wf = []

    for idx, p in enumerate(neighbor_params(), start=1):
        df = prepare_dataset(d1, h1, p)
        trades, equity = run_backtest(df, p)
        s = summarize(trades, equity, f"variant_{idx}")
        s.update({f"param_{k}": v for k, v in asdict(p).items()})
        s["variant_id"] = idx
        all_summaries.append(s)

        trades.to_csv(OUTDIR / f"variant_{idx}_trades.csv", index=False)
        equity.to_csv(OUTDIR / f"variant_{idx}_equity.csv", index=False)

        wf = walk_forward_validate(d1, h1, p, train_years=3, test_months=12)
        if not wf.empty:
            wf["variant_id"] = idx
            all_wf.append(wf)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTDIR / "variant_summary.csv", index=False)

    if all_wf:
        wf_df = pd.concat(all_wf, ignore_index=True)
        wf_df.to_csv(OUTDIR / "walkforward_folds.csv", index=False)
        wf_agg = wf_df.groupby("variant_id").agg(
            wf_folds=("fold", "count"),
            wf_avg_pf=("profit_factor", "mean"),
            wf_avg_expectancy=("expectancy_points", "mean"),
            wf_total_points=("total_points", "sum"),
            wf_avg_win_rate=("win_rate", "mean"),
        ).reset_index()
        wf_agg.to_csv(OUTDIR / "walkforward_summary.csv", index=False)
        merged = summary_df.merge(wf_agg, on="variant_id", how="left")
    else:
        merged = summary_df.copy()

    merged = merged.sort_values(["wf_avg_pf", "profit_factor", "total_points"], ascending=[False, False, False])
    merged.to_csv(OUTDIR / "ranked_variants.csv", index=False)

    report = {
        "hard_stop_points": HARD_STOP_POINTS,
        "hard_stop_usd": HARD_STOP_USD,
        "default_costs": {"commission_per_side_usd": 0.62, "slippage_points_per_side": 0.5},
        "notes": [
            "Each variant rebuilds its own features and signals",
            "Hard stop distance capped at 200 points / $400 for MNQ",
            "Cooldown of 48 H1 bars after 2 consecutive losing trades",
            "Initial target arms trailing logic instead of forced exit",
            "Walk-forward uses 3-year train / 12-month test rolling slices",
            "Ranking prioritizes walk-forward average PF first"
        ]
    }
    with open(OUTDIR / "run_notes.json", "w") as f:
        json.dump(report, f, indent=2)

    cols = [c for c in [
        "variant_id", "wf_avg_pf", "wf_total_points", "profit_factor", "total_points",
        "n_trades", "win_rate", "max_drawdown_points", "pct_trail_stop",
        "param_tp_atr_mult", "param_stop_atr_mult", "param_z_window",
        "param_trail_atr_mult_after_tp", "param_commission_per_side_usd",
        "param_slippage_points_per_side"
    ] if c in merged.columns]

    print("Top variants:")
    print(merged[cols].head(12).to_string(index=False))
    print(f"\nSaved outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
