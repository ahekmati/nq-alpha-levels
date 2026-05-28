#!/usr/bin/env python3
"""
Detailed validation script for MNQ swing mean-reversion strategy
Analyzes Run 441 winner from parameter sweep
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

OUTPUT = Path("validation_run_441")
OUTPUT.mkdir(parents=True, exist_ok=True)

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
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
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

def run_backtest(df: pd.DataFrame, p: StrategyParams):
    rows = []
    eq_rows = []
    pos = 0
    entry_price = entry_time = stop_price = tp_price = None
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
            exit_reason = exit_price = None

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
                    rows.append({
                        "entry_time": entry_time, "exit_time": nxt["time"], "side": "LONG",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "stop_price": stop_price, "tp_price": tp_price,
                        "bars_held": bars_in_trade, "pnl_points": pnl, "exit_reason": exit_reason,
                    })
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
                    rows.append({
                        "entry_time": entry_time, "exit_time": nxt["time"], "side": "SHORT",
                        "entry_price": entry_price, "exit_price": exit_price,
                        "stop_price": stop_price, "tp_price": tp_price,
                        "bars_held": bars_in_trade, "pnl_points": pnl, "exit_reason": exit_reason,
                    })
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

    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades["pnl_usd"] = trades["pnl_points"] * 2.0  # MNQ multiplier
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
        trades["exit_time"] = pd.to_datetime(trades["exit_time"])
        trades["year"] = trades["entry_time"].dt.year
        trades["month"] = trades["entry_time"].dt.to_period("M")
        trades["quarter"] = trades["entry_time"].dt.to_period("Q")

    equity_df = pd.DataFrame(eq_rows)
    if not equity_df.empty:
        equity_df["time"] = pd.to_datetime(equity_df["time"])

    return trades, equity_df

print("Loading data...")
d1 = pd.read_csv("research_mnq_swing_meanrev/d1_bars.csv")
h1 = pd.read_csv("research_mnq_swing_meanrev/h1_bars.csv")
d1["time"] = pd.to_datetime(d1["time"], utc=True)
h1["time"] = pd.to_datetime(h1["time"], utc=True)

print("Building features and signals...")
p = StrategyParams()
dctx = build_daily_context(d1, p)
hfeat = build_hourly_features(h1, p)
df = merge_context(hfeat, dctx)
df = build_signals(df, p)

print("Running backtest...")
trades, equity = run_backtest(df, p)

print(f"\nGenerated {len(trades)} trades")
trades.to_csv(OUTPUT / "full_trade_blotter.csv", index=False)
equity.to_csv(OUTPUT / "equity_curve.csv", index=False)

# Analysis functions - continuation of validate_run_441.py

# Long vs Short breakdown
long_trades = trades[trades["side"] == "LONG"].copy()
short_trades = trades[trades["side"] == "SHORT"].copy()

def side_metrics(side_df, side_name):
    if side_df.empty:
        return {}
    wins = side_df[side_df["pnl_points"] > 0]
    losses = side_df[side_df["pnl_points"] < 0]
    return {
        f"{side_name}_trades": int(len(side_df)),
        f"{side_name}_win_rate": float((side_df["pnl_points"] > 0).mean()),
        f"{side_name}_total_points": float(side_df["pnl_points"].sum()),
        f"{side_name}_avg_win": float(wins["pnl_points"].mean()) if len(wins) else 0.0,
        f"{side_name}_avg_loss": float(losses["pnl_points"].mean()) if len(losses) else 0.0,
        f"{side_name}_pct_stop": float((side_df["exit_reason"] == "stop").mean()),
        f"{side_name}_pct_target": float((side_df["exit_reason"] == "target").mean()),
        f"{side_name}_median_hold_h": float(side_df["bars_held"].median()),
    }

results = {}
results["config"] = asdict(p)
results["total_trades"] = int(len(trades))
results["total_pnl_points"] = float(trades["pnl_points"].sum())
results["total_pnl_usd"] = float(trades["pnl_usd"].sum())

results.update(side_metrics(long_trades, "long"))
results.update(side_metrics(short_trades, "short"))

wins = trades[trades["pnl_points"] > 0]
losses = trades[trades["pnl_points"] < 0]
results["win_rate"] = float((trades["pnl_points"] > 0).mean())
results["avg_win_points"] = float(wins["pnl_points"].mean()) if len(wins) else 0.0
results["avg_loss_points"] = float(losses["pnl_points"].mean()) if len(losses) else 0.0
results["largest_win"] = float(wins["pnl_points"].max()) if len(wins) else 0.0
results["largest_loss"] = float(losses["pnl_points"].min()) if len(losses) else 0.0

gross_win = wins["pnl_points"].sum() if len(wins) else 0.0
gross_loss = abs(losses["pnl_points"].sum()) if len(losses) else 0.0
results["profit_factor"] = float(gross_win / gross_loss) if gross_loss > 0 else 0.0
results["expectancy_points"] = float(trades["pnl_points"].mean())

results["pct_stop"] = float((trades["exit_reason"] == "stop").mean())
results["pct_target"] = float((trades["exit_reason"] == "target").mean())
results["pct_reverse"] = float((trades["exit_reason"] == "reverse").mean())
results["pct_time"] = float((trades["exit_reason"] == "time").mean())

equity["equity_points_cumsum"] = equity["equity_points"]
peak = equity["equity_points_cumsum"].cummax()
drawdown = equity["equity_points_cumsum"] - peak
results["max_drawdown_points"] = float(drawdown.min())
max_dd_idx = drawdown.idxmin()
if pd.notna(max_dd_idx):
    peak_before_dd = equity.loc[:max_dd_idx, "equity_points_cumsum"].idxmax()
    recovery_idx = equity.loc[max_dd_idx:, "equity_points_cumsum"][
        equity.loc[max_dd_idx:, "equity_points_cumsum"] >= equity.loc[peak_before_dd, "equity_points_cumsum"]
    ]
    if len(recovery_idx) > 0:
        recovery_bars = recovery_idx.index[0] - max_dd_idx
        results["max_dd_recovery_bars"] = int(recovery_bars)
    else:
        results["max_dd_recovery_bars"] = None

trades["is_win"] = (trades["pnl_points"] > 0).astype(int)
trades["streak"] = (trades["is_win"] != trades["is_win"].shift()).cumsum()
streaks = trades.groupby("streak").agg(
    streak_len=("is_win", "count"),
    streak_type=("is_win", "first")
)
max_win_streak = streaks[streaks["streak_type"] == 1]["streak_len"].max() if len(streaks[streaks["streak_type"] == 1]) else 0
max_loss_streak = streaks[streaks["streak_type"] == 0]["streak_len"].max() if len(streaks[streaks["streak_type"] == 0]) else 0
results["max_consecutive_wins"] = int(max_win_streak) if pd.notna(max_win_streak) else 0
results["max_consecutive_losses"] = int(max_loss_streak) if pd.notna(max_loss_streak) else 0

yearly = trades.groupby("year").agg(
    trades=("pnl_points", "count"),
    total_points=("pnl_points", "sum"),
    win_rate=("pnl_points", lambda x: (x > 0).mean()),
    avg_points=("pnl_points", "mean")
).reset_index()
yearly.to_csv(OUTPUT / "yearly_performance.csv", index=False)

monthly = trades.groupby("month").agg(
    trades=("pnl_points", "count"),
    total_points=("pnl_points", "sum"),
    win_rate=("pnl_points", lambda x: (x > 0).mean())
).reset_index()
monthly["month"] = monthly["month"].astype(str)
monthly.to_csv(OUTPUT / "monthly_performance.csv", index=False)

quarterly = trades.groupby("quarter").agg(
    trades=("pnl_points", "count"),
    total_points=("pnl_points", "sum"),
    win_rate=("pnl_points", lambda x: (x > 0).mean())
).reset_index()
quarterly["quarter"] = quarterly["quarter"].astype(str)
quarterly.to_csv(OUTPUT / "quarterly_performance.csv", index=False)

with open(OUTPUT / "validation_metrics.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\n=== VALIDATION SUMMARY ===\n")
print(f"Total Trades: {results['total_trades']}")
print(f"Profit Factor: {results['profit_factor']:.2f}")
print(f"Win Rate: {results['win_rate']:.1%}")
print(f"Expectancy: {results['expectancy_points']:.2f} points/trade")
print(f"Total P&L: {results['total_pnl_points']:.0f} points (${results['total_pnl_usd']:.0f})")
print(f"\nLong: {results['long_trades']} trades | WR {results['long_win_rate']:.1%} | {results['long_total_points']:.0f} pts")
print(f"Short: {results['short_trades']} trades | WR {results['short_win_rate']:.1%} | {results['short_total_points']:.0f} pts")
print(f"\nExit Mix: {results['pct_stop']:.1%} stop | {results['pct_target']:.1%} target | {results['pct_reverse']:.1%} reverse | {results['pct_time']:.1%} time")
print(f"Max DD: {results['max_drawdown_points']:.0f} points")
print(f"Max Consecutive Wins: {results['max_consecutive_wins']}")
print(f"Max Consecutive Losses: {results['max_consecutive_losses']}")

print("\n=== YEARLY PERFORMANCE ===\n")
print(yearly.to_string(index=False))

print("\n=== TOP 10 WINNERS ===\n")
print(trades.nlargest(10, "pnl_points")[["entry_time", "side", "pnl_points", "bars_held", "exit_reason"]].to_string(index=False))

print("\n=== TOP 10 LOSERS ===\n")
print(trades.nsmallest(10, "pnl_points")[["entry_time", "side", "pnl_points", "bars_held", "exit_reason"]].to_string(index=False))

print("\nCreating visualizations...")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(equity["time"], equity["equity_points_cumsum"], linewidth=1.5, color="#2563eb")
ax1.fill_between(equity["time"], 0, equity["equity_points_cumsum"], alpha=0.1, color="#2563eb")
ax1.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)
ax1.set_title("Cumulative P&L (Points)", fontsize=14, fontweight="bold")
ax1.set_ylabel("Points", fontsize=11)
ax1.grid(True, alpha=0.2)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax2 = fig.add_subplot(gs[1, :])
ax2.fill_between(equity["time"], 0, drawdown, alpha=0.3, color="#dc2626")
ax2.plot(equity["time"], drawdown, linewidth=1, color="#dc2626")
ax2.set_title("Drawdown (Points)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Points", fontsize=11)
ax2.grid(True, alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

ax3 = fig.add_subplot(gs[2, 0])
bins = np.linspace(trades["pnl_points"].min(), trades["pnl_points"].max(), 40)
ax3.hist(trades["pnl_points"], bins=bins, edgecolor="white", linewidth=0.5, color="#059669", alpha=0.7)
ax3.axvline(0, color="black", linewidth=1, linestyle="--")
ax3.axvline(trades["pnl_points"].mean(), color="#dc2626", linewidth=2, linestyle="--", label=f"Mean: {trades['pnl_points'].mean():.1f}")
ax3.set_title("Trade P&L Distribution", fontsize=12, fontweight="bold")
ax3.set_xlabel("P&L (Points)", fontsize=10)
ax3.set_ylabel("Frequency", fontsize=10)
ax3.legend()
ax3.grid(True, alpha=0.2, axis="y")

ax4 = fig.add_subplot(gs[2, 1])
bins_hold = np.arange(0, trades["bars_held"].max() + 2, 1)
ax4.hist(trades["bars_held"], bins=bins_hold, edgecolor="white", linewidth=0.5, color="#7c3aed", alpha=0.7)
ax4.axvline(trades["bars_held"].median(), color="#dc2626", linewidth=2, linestyle="--", label=f"Median: {trades['bars_held'].median():.0f}h")
ax4.set_title("Holding Time Distribution", fontsize=12, fontweight="bold")
ax4.set_xlabel("Hours Held", fontsize=10)
ax4.set_ylabel("Frequency", fontsize=10)
ax4.legend()
ax4.grid(True, alpha=0.2, axis="y")

ax5 = fig.add_subplot(gs[3, 0])
yearly_summary = yearly.copy()
ax5.bar(yearly_summary["year"].astype(str), yearly_summary["total_points"],
        color=["#059669" if x > 0 else "#dc2626" for x in yearly_summary["total_points"]], alpha=0.7, edgecolor="white")
ax5.axhline(0, color="black", linewidth=0.8)
ax5.set_title("Yearly P&L", fontsize=12, fontweight="bold")
ax5.set_xlabel("Year", fontsize=10)
ax5.set_ylabel("Points", fontsize=10)
ax5.grid(True, alpha=0.2, axis="y")

ax6 = fig.add_subplot(gs[3, 1])
sides = ["LONG", "SHORT"]
side_pnl = [results["long_total_points"], results["short_total_points"]]
side_colors = ["#2563eb", "#dc2626"]
ax6.bar(sides, side_pnl, color=side_colors, alpha=0.7, edgecolor="white", width=0.6)
ax6.axhline(0, color="black", linewidth=0.8)
ax6.set_title("Long vs Short P&L", fontsize=12, fontweight="bold")
ax6.set_ylabel("Points", fontsize=10)
ax6.grid(True, alpha=0.2, axis="y")

plt.suptitle("Run 441 Validation - MNQ Swing Mean Reversion", fontsize=16, fontweight="bold", y=0.995)
plt.savefig(OUTPUT / "validation_report.png", dpi=150, bbox_inches="tight")

print(f"\nSaved: {OUTPUT / 'validation_report.png'}")
print("\nAll validation outputs saved to:", OUTPUT)
print("\nFiles generated:")
print("  - full_trade_blotter.csv")
print("  - equity_curve.csv")
print("  - validation_metrics.json")
print("  - yearly_performance.csv")
print("  - monthly_performance.csv")
print("  - quarterly_performance.csv")
print("  - validation_report.png")
