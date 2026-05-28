from mt5linux import MetaTrader5
import pandas as pd
import numpy as np
from math import sqrt

SYMBOL_CANDIDATES = ["@MNQ", "MNQ", "MNQM26", "MNQU26", "MNQZ26"]
TIMEFRAME_REGIME = MetaTrader5.TIMEFRAME_D1
TIMEFRAME_EXEC = MetaTrader5.TIMEFRAME_H1
D1_BARS = 5000
H1_BARS = 50000
EMA_PERIOD = 100
SLOPE_LOOKBACK = 10
ATR_PERIOD = 14
RSI_PERIOD = 14
SWING_LOOKBACK = 5
BREAKOUT_LOOKBACK = 5
REGIME_RSI_MAX = 60
ATR_EXPANSION_MIN = 1.2
STOP_ATR_MULT = 1.0
TARGET_ATR_MULT = 1.5
MAX_HOLD_BARS = 24
USE_FALLING_EMA_ONLY = True
USE_RSI_CAP = True

OUTPUT_REGIME_CSV = "ema100_bear_h1_regime_study.csv"
SUMMARY_CSV = "ema100_bear_trigger_summary.csv"
TRADES_CSV = "ema100_bear_trigger_trades.csv"
YEARLY_CSV = "ema100_bear_trigger_yearly_stats.csv"

mt5 = MetaTrader5()


def resolve_symbol(mt5_client, candidates):
    for sym in candidates:
        info = mt5_client.symbol_info(sym)
        if info is None:
            continue
        mt5_client.symbol_select(sym, True)
        rates = mt5_client.copy_rates_from_pos(sym, TIMEFRAME_REGIME, 0, 20)
        if rates is not None and len(rates) > 0:
            return sym
    all_symbols = mt5_client.symbols_get()
    mnq_like = [s.name for s in all_symbols if "MNQ" in s.name.upper()] if all_symbols else []
    raise RuntimeError(f"Could not resolve symbol. MNQ-like symbols visible: {mnq_like[:50]}")


def sharpe_from_returns(rets, annualization=252 * 24):
    rets = pd.Series(rets).dropna()
    if len(rets) < 2:
        return np.nan
    std = rets.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return (rets.mean() / std) * sqrt(annualization)


def max_drawdown(equity_curve):
    eq = pd.Series(equity_curve).dropna()
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def prepare_regime_df(df_d1):
    df = df_d1.copy()
    df["ema100"] = ema(df["close"], EMA_PERIOD)
    df["ema_slope_n"] = df["ema100"] - df["ema100"].shift(SLOPE_LOOKBACK)
    df["rsi14"] = rsi(df["close"], RSI_PERIOD)
    df["below_ema"] = df["close"] < df["ema100"]
    df["below_falling"] = df["below_ema"] & (df["ema_slope_n"] < 0)
    df["bear_permission"] = df["below_falling"] if USE_FALLING_EMA_ONLY else df["below_ema"]
    if USE_RSI_CAP:
        df["bear_permission"] = df["bear_permission"] & (df["rsi14"] <= REGIME_RSI_MAX)
    df["regime_age"] = 0
    age = 0
    for i in range(len(df)):
        if bool(df.loc[i, "bear_permission"]):
            age += 1
        else:
            age = 0
        df.loc[i, "regime_age"] = age
    return df


def prepare_exec_df(df_h1, regime_df):
    df = df_h1.copy()
    df["ema100_h1"] = ema(df["close"], EMA_PERIOD)
    df["atr14"] = atr(df, ATR_PERIOD)
    df["rsi14_h1"] = rsi(df["close"], RSI_PERIOD)
    df["bar_range"] = df["high"] - df["low"]
    df["range_ma20"] = df["bar_range"].rolling(20).mean()
    df["swing_high_prev"] = df["high"].rolling(SWING_LOOKBACK).max().shift(1)
    df["swing_low_prev"] = df["low"].rolling(SWING_LOOKBACK).min().shift(1)
    df["recent_low_prev"] = df["low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    df["recent_high_prev"] = df["high"].rolling(BREAKOUT_LOOKBACK).max().shift(1)

    regime_map = regime_df[["time", "bear_permission", "regime_age", "ema100", "ema_slope_n", "rsi14"]].copy()
    regime_map = regime_map.rename(columns={
        "time": "d1_time",
        "ema100": "d1_ema100",
        "rsi14": "d1_rsi14",
    })
    regime_map = regime_map.sort_values("d1_time")
    df = df.sort_values("time")
    df = pd.merge_asof(df, regime_map, left_on="time", right_on="d1_time", direction="backward")

    in_bear = df["bear_permission"].fillna(False)
    atr_ok = df["bar_range"] > (df["range_ma20"] * ATR_EXPANSION_MIN)
    red_bar = df["close"] < df["open"]

    df["trig_failed_rally"] = (
        in_bear &
        (df["high"] > df["swing_high_prev"]) &
        (df["close"] < df["open"]) &
        (df["close"] < df["high"] - 0.6 * df["bar_range"])
    )

    df["trig_atr_breakdown"] = (
        in_bear & red_bar & atr_ok &
        (df["close"] < df["recent_low_prev"])
    )

    df["trig_ema_rejection"] = (
        in_bear &
        (df["high"] >= df["ema100_h1"]) &
        (df["close"] < df["ema100_h1"]) &
        red_bar
    )

    df["trig_rsi_rollover"] = (
        in_bear &
        (df["rsi14_h1"].shift(1) > 60) &
        (df["rsi14_h1"] < 50) &
        red_bar
    )

    df["trig_bear_divergence"] = (
        in_bear &
        (df["high"] > df["high"].shift(1)) &
        (df["rsi14_h1"] < df["rsi14_h1"].shift(1)) &
        (df["close"] < df["low"].shift(1))
    )

    df["trig_gap_fill_failure"] = (
        in_bear &
        (df["open"] > df["close"].shift(1)) &
        (df["high"] > df["recent_high_prev"].shift(1)) &
        red_bar &
        (df["close"] < df["open"])
    )

    return df


def simulate_trigger(df, trigger_col, trigger_name):
    trades = []
    equity_rets = []
    position = 0
    entry_idx = None
    entry_price = None
    stop_price = None
    target_price = None

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if position == 0:
            if bool(row.get(trigger_col, False)) and pd.notna(row["atr14"]) and row["atr14"] > 0:
                position = -1
                entry_idx = i + 1
                entry_price = next_row["open"]
                stop_price = entry_price + STOP_ATR_MULT * row["atr14"]
                target_price = entry_price - TARGET_ATR_MULT * row["atr14"]
            equity_rets.append(0.0)
            continue

        holding_bars = i - entry_idx + 1
        exit_reason = None
        exit_price = None

        if row["high"] >= stop_price:
            exit_price = stop_price
            exit_reason = "stop"
        elif row["low"] <= target_price:
            exit_price = target_price
            exit_reason = "target"
        elif holding_bars >= MAX_HOLD_BARS:
            exit_price = row["close"]
            exit_reason = "time"
        elif not bool(row.get("bear_permission", True)):
            exit_price = row["close"]
            exit_reason = "regime_off"

        if exit_reason is not None:
            ret = (entry_price / exit_price) - 1.0
            trades.append({
                "strategy": trigger_name,
                "entry_time": df.iloc[entry_idx]["time"],
                "entry_price": entry_price,
                "exit_time": row["time"],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "bars_held": int(holding_bars),
                "atr_at_entry": float(df.iloc[entry_idx - 1]["atr14"]),
                "regime_age": int(df.iloc[entry_idx - 1].get("regime_age", 0) or 0),
                "d1_rsi14": float(df.iloc[entry_idx - 1].get("d1_rsi14", np.nan)),
                "return_pct": ret * 100.0,
                "r_multiple": (entry_price - exit_price) / (STOP_ATR_MULT * df.iloc[entry_idx - 1]["atr14"]),
            })
            equity_rets.append(ret)
            position = 0
            entry_idx = None
            entry_price = None
            stop_price = None
            target_price = None
        else:
            mark_ret = (df.iloc[i - 1]["close"] / row["close"]) - 1.0
            equity_rets.append(mark_ret)

    trades_df = pd.DataFrame(trades)
    eq = pd.Series(equity_rets)
    equity = (1.0 + eq.fillna(0)).cumprod()

    if trades_df.empty:
        return {
            "strategy": trigger_name,
            "trades": 0,
            "win_rate_pct": np.nan,
            "avg_return_pct": np.nan,
            "avg_r": np.nan,
            "total_return_pct": 0.0,
            "profit_factor": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
            "avg_bars_held": np.nan,
        }, trades_df

    gross_profit = trades_df.loc[trades_df["return_pct"] > 0, "return_pct"].sum()
    gross_loss = -trades_df.loc[trades_df["return_pct"] < 0, "return_pct"].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan

    summary = {
        "strategy": trigger_name,
        "trades": int(len(trades_df)),
        "win_rate_pct": float((trades_df["return_pct"] > 0).mean() * 100.0),
        "avg_return_pct": float(trades_df["return_pct"].mean()),
        "avg_r": float(trades_df["r_multiple"].mean()),
        "total_return_pct": float((equity.iloc[-1] - 1.0) * 100.0) if len(equity) else 0.0,
        "profit_factor": float(pf) if pd.notna(pf) else np.nan,
        "sharpe": float(sharpe_from_returns(eq, annualization=252*24)),
        "max_drawdown_pct": float(max_drawdown(equity) * 100.0) if len(equity) else np.nan,
        "avg_bars_held": float(trades_df["bars_held"].mean()),
    }
    return summary, trades_df


def yearly_stats_from_trades(trades_df):
    if trades_df.empty:
        return pd.DataFrame()
    temp = trades_df.copy()
    temp["year"] = pd.to_datetime(temp["entry_time"]).dt.year
    rows = []
    for (strategy, year), grp in temp.groupby(["strategy", "year"]):
        gross_profit = grp.loc[grp["return_pct"] > 0, "return_pct"].sum()
        gross_loss = -grp.loc[grp["return_pct"] < 0, "return_pct"].sum()
        pf = gross_profit / gross_loss if gross_loss > 0 else np.nan
        rows.append({
            "strategy": strategy,
            "year": int(year),
            "trades": int(len(grp)),
            "win_rate_pct": float((grp["return_pct"] > 0).mean() * 100.0),
            "avg_return_pct": float(grp["return_pct"].mean()),
            "avg_r": float(grp["r_multiple"].mean()),
            "profit_factor": float(pf) if pd.notna(pf) else np.nan,
            "avg_bars_held": float(grp["bars_held"].mean()),
        })
    return pd.DataFrame(rows)


if not mt5.initialize():
    raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

symbol = resolve_symbol(mt5, SYMBOL_CANDIDATES)
rates_d1 = mt5.copy_rates_from_pos(symbol, TIMEFRAME_REGIME, 0, D1_BARS)
rates_h1 = mt5.copy_rates_from_pos(symbol, TIMEFRAME_EXEC, 0, H1_BARS)
mt5.shutdown()

if rates_d1 is None or len(rates_d1) == 0:
    raise RuntimeError("No D1 bars returned after resolving symbol.")
if rates_h1 is None or len(rates_h1) == 0:
    raise RuntimeError("No H1 bars returned after resolving symbol.")

df_d1 = pd.DataFrame(rates_d1)
df_d1["time"] = pd.to_datetime(df_d1["time"], unit="s")
df_d1 = df_d1.sort_values("time").reset_index(drop=True)

df_h1 = pd.DataFrame(rates_h1)
df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
df_h1 = df_h1.sort_values("time").reset_index(drop=True)

regime_df = prepare_regime_df(df_d1)
exec_df = prepare_exec_df(df_h1, regime_df)

trigger_specs = [
    ("trig_failed_rally", "failed_rally"),
    ("trig_atr_breakdown", "atr_breakdown"),
    ("trig_ema_rejection", "ema_rejection"),
    ("trig_rsi_rollover", "rsi_rollover"),
    ("trig_bear_divergence", "bear_divergence"),
    ("trig_gap_fill_failure", "gap_fill_failure"),
]

summary_rows = []
trade_frames = []

for col, name in trigger_specs:
    summary, trades_df = simulate_trigger(exec_df, col, name)
    summary_rows.append(summary)
    if not trades_df.empty:
        trade_frames.append(trades_df)

summary_df = pd.DataFrame(summary_rows).sort_values(["avg_r", "profit_factor", "win_rate_pct"], ascending=False, na_position="last")
trades_out = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
yearly_df = yearly_stats_from_trades(trades_out)

print(f"\nResolved symbol: {symbol}")
print(f"D1 bars analyzed: {len(regime_df)}")
print(f"H1 bars analyzed: {len(exec_df)}")
print(f"Date range D1: {regime_df['time'].iloc[0].date()} -> {regime_df['time'].iloc[-1].date()}")
print(f"Date range H1: {exec_df['time'].iloc[0]} -> {exec_df['time'].iloc[-1]}")
print(f"EMA period: {EMA_PERIOD}")
print(f"Slope lookback: {SLOPE_LOOKBACK}")
print(f"ATR period: {ATR_PERIOD}")
print(f"Max hold bars: {MAX_HOLD_BARS}")

print("\n=== Bear Regime Counts (D1) ===")
for c in ["below_ema", "below_falling", "bear_permission"]:
    print(f"{c}: {int(regime_df[c].fillna(False).sum())} ({regime_df[c].fillna(False).mean()*100:.2f}%)")

print("\n=== Trigger Counts (H1 while regime active) ===")
for col, name in trigger_specs:
    count = int(exec_df[col].fillna(False).sum())
    print(f"{name}: {count}")

print("\n=== Trigger Comparison ===")
print(summary_df.to_string(index=False))

if not yearly_df.empty:
    print("\n=== Yearly Trigger Stats ===")
    print(yearly_df.to_string(index=False))

exec_df.to_csv(OUTPUT_REGIME_CSV, index=False)
summary_df.to_csv(SUMMARY_CSV, index=False)
if not trades_out.empty:
    trades_out.to_csv(TRADES_CSV, index=False)
if not yearly_df.empty:
    yearly_df.to_csv(YEARLY_CSV, index=False)

print(f"\nSaved H1 regime study to: {OUTPUT_REGIME_CSV}")
print(f"Saved trigger summary to: {SUMMARY_CSV}")
if not trades_out.empty:
    print(f"Saved trigger trades to: {TRADES_CSV}")
if not yearly_df.empty:
    print(f"Saved yearly trigger stats to: {YEARLY_CSV}")
