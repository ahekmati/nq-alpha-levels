#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

OUTDIR = Path("validation_variant8_deepcheck")
OUTDIR.mkdir(parents=True, exist_ok=True)

MNQ_POINT_VALUE = 2.0
RNG = np.random.default_rng(42)

@dataclass
class StrategyParams:
    regime_mode: str = "both"
    daily_ma_len: int = 100
    daily_rsi_len: int = 10
    daily_rsi_bull_min: float = 52.0
    daily_rsi_bear_max: float = 48.0
    z_window: int = 40
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
    hard_stop_points: float = 200.0
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
    return build_signals(df, p)


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
        trades["month"] = trades["entry_time"].dt.strftime("%Y-%m")
        trades["quarter"] = trades["entry_time"].dt.to_period("Q").astype(str)
    equity = pd.DataFrame(eq_rows)
    if not equity.empty:
        equity["time"] = pd.to_datetime(equity["time"], utc=True)
        equity["equity_cum"] = equity["equity_points"]
    return trades, equity


def calc_summary(trades: pd.DataFrame, equity: pd.DataFrame) -> dict:
    wins = trades.loc[trades["pnl_points"] > 0, "pnl_points"]
    losses = trades.loc[trades["pnl_points"] < 0, "pnl_points"]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    peak = equity["equity_cum"].cummax()
    dd = equity["equity_cum"] - peak
    return {
        "n_trades": int(len(trades)),
        "win_rate": float((trades["pnl_points"] > 0).mean()),
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else 0.0,
        "total_points": float(trades["pnl_points"].sum()),
        "total_usd": float(trades["pnl_usd"].sum()),
        "expectancy_points": float(trades["pnl_points"].mean()),
        "max_drawdown_points": float(dd.min()) if len(dd) else 0.0,
        "pct_trail_stop": float((trades["exit_reason"] == "trail_stop").mean()),
        "pct_time_exit": float((trades["exit_reason"] == "time").mean()),
    }


def side_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    out = []
    for side, g in trades.groupby("side"):
        wins = g.loc[g["pnl_points"] > 0, "pnl_points"]
        losses = g.loc[g["pnl_points"] < 0, "pnl_points"]
        gross_win = wins.sum() if len(wins) else 0.0
        gross_loss = abs(losses.sum()) if len(losses) else 0.0
        out.append({
            "side": side,
            "trades": int(len(g)),
            "win_rate": float((g["pnl_points"] > 0).mean()),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else 0.0,
            "total_points": float(g["pnl_points"].sum()),
            "avg_points": float(g["pnl_points"].mean()),
            "median_bars_held": float(g["bars_held"].median()),
            "pct_trail_stop": float((g["exit_reason"] == "trail_stop").mean()),
        })
    return pd.DataFrame(out)


def monthly_stats(trades: pd.DataFrame) -> pd.DataFrame:
    m = trades.groupby("month").agg(
        trades=("pnl_points", "count"),
        total_points=("pnl_points", "sum"),
        total_usd=("pnl_usd", "sum"),
        win_rate=("pnl_points", lambda x: (x > 0).mean()),
        avg_points=("pnl_points", "mean")
    ).reset_index()
    return m


def yearly_stats(trades: pd.DataFrame) -> pd.DataFrame:
    y = trades.groupby("year").agg(
        trades=("pnl_points", "count"),
        total_points=("pnl_points", "sum"),
        total_usd=("pnl_usd", "sum"),
        win_rate=("pnl_points", lambda x: (x > 0).mean()),
        avg_points=("pnl_points", "mean")
    ).reset_index()
    return y


def monte_carlo_trade_shuffle(trades: pd.DataFrame, n_sims: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    pnl = trades["pnl_points"].to_numpy()
    sims = []
    paths = []
    n = len(pnl)

    for sim in range(n_sims):
        sample = RNG.choice(pnl, size=n, replace=True)
        eq = np.cumsum(sample)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        sims.append({
            "sim": sim + 1,
            "final_points": float(eq[-1]),
            "max_drawdown_points": float(dd.min()),
            "win_rate": float(np.mean(sample > 0)),
        })
        if sim < 50:
            path = pd.DataFrame({
                "step": np.arange(1, n + 1),
                "equity_points": eq,
                "sim": sim + 1
            })
            paths.append(path)

    return pd.DataFrame(sims), pd.concat(paths, ignore_index=True)


def monte_carlo_summary(mc: pd.DataFrame) -> pd.DataFrame:
    qs = [0.05, 0.25, 0.50, 0.75, 0.95]
    rows = []
    for col in ["final_points", "max_drawdown_points"]:
        q = mc[col].quantile(qs)
        rows.append({
            "metric": col,
            "p05": float(q.loc[0.05]),
            "p25": float(q.loc[0.25]),
            "p50": float(q.loc[0.50]),
            "p75": float(q.loc[0.75]),
            "p95": float(q.loc[0.95]),
        })
    return pd.DataFrame(rows)


def main():
    p = StrategyParams()

    d1 = pd.read_csv("research_mnq_swing_meanrev/d1_bars.csv")
    h1 = pd.read_csv("research_mnq_swing_meanrev/h1_bars.csv")
    d1["time"] = pd.to_datetime(d1["time"], utc=True)
    h1["time"] = pd.to_datetime(h1["time"], utc=True)

    df = prepare_dataset(d1, h1, p)
    trades, equity = run_backtest(df, p)

    summary = calc_summary(trades, equity)
    summary["params"] = asdict(p)

    side_df = side_breakdown(trades)
    month_df = monthly_stats(trades)
    year_df = yearly_stats(trades)

    month_df["is_profitable"] = (month_df["total_points"] > 0).astype(int)
    summary["profitable_month_rate"] = float(month_df["is_profitable"].mean())
    summary["worst_month_points"] = float(month_df["total_points"].min())
    summary["best_month_points"] = float(month_df["total_points"].max())

    mc_df, mc_paths = monte_carlo_trade_shuffle(trades, n_sims=2000)
    mc_sum = monte_carlo_summary(mc_df)

    summary["mc_prob_final_positive"] = float((mc_df["final_points"] > 0).mean())
    summary["mc_p05_final_points"] = float(mc_df["final_points"].quantile(0.05))
    summary["mc_p50_final_points"] = float(mc_df["final_points"].quantile(0.50))
    summary["mc_p95_final_points"] = float(mc_df["final_points"].quantile(0.95))
    summary["mc_p95_drawdown_worst"] = float(mc_df["max_drawdown_points"].quantile(0.05))

    trades.to_csv(OUTDIR / "variant8_trades.csv", index=False)
    equity.to_csv(OUTDIR / "variant8_equity.csv", index=False)
    side_df.to_csv(OUTDIR / "variant8_side_breakdown.csv", index=False)
    month_df.to_csv(OUTDIR / "variant8_monthly.csv", index=False)
    year_df.to_csv(OUTDIR / "variant8_yearly.csv", index=False)
    mc_df.to_csv(OUTDIR / "variant8_monte_carlo_runs.csv", index=False)
    mc_paths.to_csv(OUTDIR / "variant8_monte_carlo_sample_paths.csv", index=False)
    mc_sum.to_csv(OUTDIR / "variant8_monte_carlo_summary.csv", index=False)

    with open(OUTDIR / "variant8_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n=== VARIANT 8 SUMMARY ===")
    for k, v in summary.items():
        if k != "params":
            print(f"{k}: {v}")

    print("\n=== SIDE BREAKDOWN ===")
    print(side_df.to_string(index=False))

    print("\n=== YEARLY ===")
    print(year_df.to_string(index=False))

    print("\n=== MONTE CARLO SUMMARY ===")
    print(mc_sum.to_string(index=False))

    print(f"\nSaved outputs to {OUTDIR}")


if __name__ == "__main__":
    main()
