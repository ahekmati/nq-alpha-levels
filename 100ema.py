from mt5linux import MetaTrader5
import pandas as pd
import numpy as np
from math import sqrt

# =========================
# CONFIG
# =========================
SYMBOL_CANDIDATES = ["@MNQ", "MNQ", "MNQM26", "MNQU26", "MNQZ26"]
TIMEFRAME = MetaTrader5.TIMEFRAME_D1
BARS = 5000
EMA_PERIOD = 100
SLOPE_LOOKBACK = 10
DIST_BAND_PCT = 1.0

OUTPUT_CSV = "ema100_daily_study.csv"
TRADES_CSV = "ema100_regime_trades.csv"
YEARLY_CSV = "ema100_yearly_stats.csv"
SUMMARY_CSV = "ema100_summary.csv"

mt5 = MetaTrader5()

# =========================
# HELPERS
# =========================
def resolve_symbol(mt5_client, candidates):
    for sym in candidates:
        info = mt5_client.symbol_info(sym)
        if info is None:
            continue
        mt5_client.symbol_select(sym, True)
        rates = mt5_client.copy_rates_from_pos(sym, TIMEFRAME, 0, 20)
        if rates is not None and len(rates) > 0:
            return sym
    all_symbols = mt5_client.symbols_get()
    mnq_like = [s.name for s in all_symbols if "MNQ" in s.name.upper()] if all_symbols else []
    raise RuntimeError(f"Could not resolve symbol. MNQ-like symbols visible: {mnq_like[:50]}")

def sharpe_from_returns(rets):
    rets = pd.Series(rets).dropna()
    if len(rets) < 2:
        return np.nan
    std = rets.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return (rets.mean() / std) * sqrt(252)

def max_drawdown(equity_curve):
    eq = pd.Series(equity_curve).dropna()
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()

def summarize_mask(df, mask, label, horizons):
    subset = df.loc[mask].copy()
    row = {
        "group": label,
        "count": int(mask.sum()),
        "pct_of_sample": float(mask.mean() * 100.0)
    }
    for h in horizons:
        vals = subset[f"fwd_ret_{h}"].dropna()
        row[f"mean_fwd_{h}d"] = float(vals.mean()) if len(vals) else np.nan
        row[f"median_fwd_{h}d"] = float(vals.median()) if len(vals) else np.nan
        row[f"hitrate_fwd_{h}d"] = float((vals > 0).mean() * 100.0) if len(vals) else np.nan
    return row

def compute_strategy_stats(df, position_col, name):
    out = {}
    temp = df.copy()
    temp["position"] = temp[position_col].shift(1).fillna(0)
    temp["strategy_ret"] = temp["position"] * temp["ret_1d"]
    temp["equity"] = (1.0 + temp["strategy_ret"].fillna(0)).cumprod()

    out["strategy"] = name
    out["bars_in_market"] = int((temp["position"] > 0).sum())
    out["pct_time_in_market"] = float((temp["position"] > 0).mean() * 100.0)
    out["total_return_pct"] = float((temp["equity"].iloc[-1] - 1.0) * 100.0)
    out["sharpe"] = float(sharpe_from_returns(temp["strategy_ret"]))
    out["max_drawdown_pct"] = float(max_drawdown(temp["equity"]) * 100.0)
    return out, temp

def extract_trades(df, position_col):
    temp = df.copy()
    temp["position"] = temp[position_col].shift(1).fillna(0)

    trades = []
    in_trade = False
    entry_time = None
    entry_price = None

    for i in range(1, len(temp)):
        prev_pos = temp.loc[i - 1, "position"]
        cur_pos = temp.loc[i, "position"]

        if not in_trade and prev_pos == 0 and cur_pos == 1:
            in_trade = True
            entry_time = temp.loc[i, "time"]
            entry_price = temp.loc[i, "close"]

        elif in_trade and prev_pos == 1 and cur_pos == 0:
            exit_time = temp.loc[i, "time"]
            exit_price = temp.loc[i, "close"]
            ret = exit_price / entry_price - 1.0
            trades.append({
                "strategy": position_col,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "return_pct": ret * 100.0,
                "bars_held": int((exit_time - entry_time).days)
            })
            in_trade = False
            entry_time = None
            entry_price = None

    return pd.DataFrame(trades)

def yearly_stats(df, position_col, name):
    temp = df.copy()
    temp["position"] = temp[position_col].shift(1).fillna(0)
    temp["strategy_ret"] = temp["position"] * temp["ret_1d"]
    temp["year"] = temp["time"].dt.year

    rows = []
    for year, grp in temp.groupby("year"):
        if len(grp) < 10:
            continue
        equity = (1.0 + grp["strategy_ret"].fillna(0)).cumprod()
        rows.append({
            "strategy": name,
            "year": int(year),
            "bars": int(len(grp)),
            "time_in_market_pct": float((grp["position"] > 0).mean() * 100.0),
            "return_pct": float((equity.iloc[-1] - 1.0) * 100.0),
            "sharpe": float(sharpe_from_returns(grp["strategy_ret"])),
            "max_drawdown_pct": float(max_drawdown(equity) * 100.0),
        })
    return pd.DataFrame(rows)

# =========================
# LOAD DATA
# =========================
if not mt5.initialize():
    raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

symbol = resolve_symbol(mt5, SYMBOL_CANDIDATES)
rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, BARS)
mt5.shutdown()

if rates is None or len(rates) == 0:
    raise RuntimeError("No D1 bars returned after resolving symbol.")

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df = df.sort_values("time").reset_index(drop=True)

# =========================
# FEATURES
# =========================
df["ema100"] = df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
df["ema_slope_1"] = df["ema100"].diff()
df["ema_slope_n"] = df["ema100"] - df["ema100"].shift(SLOPE_LOOKBACK)
df["ema_slope_pct_n"] = df["ema_slope_n"] / df["ema100"].shift(SLOPE_LOOKBACK)

df["above_ema"] = df["close"] > df["ema100"]
df["below_ema"] = df["close"] < df["ema100"]
df["dist_pct"] = (df["close"] / df["ema100"] - 1.0) * 100.0
df["ret_1d"] = df["close"].pct_change()

horizons = [1, 3, 5, 10, 20]
for h in horizons:
    df[f"fwd_ret_{h}"] = df["close"].shift(-h) / df["close"] - 1.0

# slope-qualified states using N-bar EMA slope
df["above_ema_rising"] = (df["close"] > df["ema100"]) & (df["ema_slope_n"] > 0)
df["above_ema_falling"] = (df["close"] > df["ema100"]) & (df["ema_slope_n"] <= 0)
df["below_ema_falling"] = (df["close"] < df["ema100"]) & (df["ema_slope_n"] < 0)
df["below_ema_rising"] = (df["close"] < df["ema100"]) & (df["ema_slope_n"] >= 0)

# distance bands
df["above_ema_band"] = df["dist_pct"] > DIST_BAND_PCT
df["below_ema_band"] = df["dist_pct"] < -DIST_BAND_PCT

# combined filters
df["above_rising_band"] = df["above_ema"] & (df["ema_slope_n"] > 0) & (df["dist_pct"] > DIST_BAND_PCT)
df["above_rising"] = df["above_ema"] & (df["ema_slope_n"] > 0)
df["below_falling"] = df["below_ema"] & (df["ema_slope_n"] < 0)

# =========================
# CONDITIONAL STATS
# =========================
summary_rows = []
summary_rows.append(summarize_mask(df, df["above_ema"], "above_ema", horizons))
summary_rows.append(summarize_mask(df, df["below_ema"], "below_ema", horizons))
summary_rows.append(summarize_mask(df, df["above_ema_rising"], "above_ema_rising", horizons))
summary_rows.append(summarize_mask(df, df["above_ema_falling"], "above_ema_falling", horizons))
summary_rows.append(summarize_mask(df, df["below_ema_falling"], "below_ema_falling", horizons))
summary_rows.append(summarize_mask(df, df["below_ema_rising"], "below_ema_rising", horizons))
summary_rows.append(summarize_mask(df, df["above_ema_band"], f"above_ema_{DIST_BAND_PCT:.1f}pct", horizons))
summary_rows.append(summarize_mask(df, df["below_ema_band"], f"below_ema_{DIST_BAND_PCT:.1f}pct", horizons))
summary_rows.append(summarize_mask(df, df["above_rising_band"], f"above_rising_{DIST_BAND_PCT:.1f}pct", horizons))
summary_rows.append(summarize_mask(df, df["above_rising"], "above_rising", horizons))
summary_rows.append(summarize_mask(df, df["below_falling"], "below_falling", horizons))

summary_df = pd.DataFrame(summary_rows)

# =========================
# STRATEGY COMPARISONS
# =========================
df["buyhold_position"] = 1
df["long_above_ema"] = np.where(df["above_ema"], 1, 0)
df["long_above_rising"] = np.where(df["above_rising"], 1, 0)
df["long_above_rising_band"] = np.where(df["above_rising_band"], 1, 0)

strategy_stats = []
buyhold_stats, buyhold_temp = compute_strategy_stats(df, "buyhold_position", "buyhold")
strategy_stats.append(buyhold_stats)

s1_stats, s1_temp = compute_strategy_stats(df, "long_above_ema", "long_above_ema")
strategy_stats.append(s1_stats)

s2_stats, s2_temp = compute_strategy_stats(df, "long_above_rising", "long_above_rising")
strategy_stats.append(s2_stats)

s3_stats, s3_temp = compute_strategy_stats(df, "long_above_rising_band", "long_above_rising_band")
strategy_stats.append(s3_stats)

strategy_df = pd.DataFrame(strategy_stats)

# =========================
# TRADE LISTS
# =========================
trades_frames = []
for col in ["long_above_ema", "long_above_rising", "long_above_rising_band"]:
    tdf = extract_trades(df, col)
    if not tdf.empty:
        trades_frames.append(tdf)

trades_df = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()

# =========================
# YEARLY STATS
# =========================
yearly_frames = []
for col, name in [
    ("buyhold_position", "buyhold"),
    ("long_above_ema", "long_above_ema"),
    ("long_above_rising", "long_above_rising"),
    ("long_above_rising_band", "long_above_rising_band"),
]:
    ydf = yearly_stats(df, col, name)
    if not ydf.empty:
        yearly_frames.append(ydf)

yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()

# =========================
# PRINT RESULTS
# =========================
print(f"\nResolved symbol: {symbol}")
print(f"Bars analyzed: {len(df)}")
print(f"Date range: {df['time'].iloc[0].date()} -> {df['time'].iloc[-1].date()}")
print(f"EMA period: {EMA_PERIOD}")
print(f"Slope lookback: {SLOPE_LOOKBACK}")
print(f"Distance band: {DIST_BAND_PCT:.2f}%")

above_pct = df["above_ema"].mean() * 100.0
below_pct = df["below_ema"].mean() * 100.0

print("\n=== Frequency Above/Below 100 EMA ===")
print(f"Above EMA100: {above_pct:.2f}%")
print(f"Below EMA100: {below_pct:.2f}%")

print("\n=== State Counts ===")
state_cols = [
    "above_ema_rising",
    "above_ema_falling",
    "below_ema_falling",
    "below_ema_rising",
    "above_ema_band",
    "below_ema_band",
    "above_rising_band",
]
for c in state_cols:
    print(f"{c}: {int(df[c].sum())} ({df[c].mean()*100:.2f}%)")

print("\n=== Forward Return Study ===")
cols = ["group", "count", "pct_of_sample"]
for h in horizons:
    cols += [f"mean_fwd_{h}d", f"median_fwd_{h}d", f"hitrate_fwd_{h}d"]
print(summary_df[cols].to_string(index=False))

print("\n=== Strategy Comparison ===")
print(strategy_df.to_string(index=False))

if not yearly_df.empty:
    print("\n=== Yearly Stats ===")
    print(yearly_df.to_string(index=False))

# =========================
# SAVE OUTPUTS
# =========================
df.to_csv(OUTPUT_CSV, index=False)
summary_df.to_csv(SUMMARY_CSV, index=False)

if not trades_df.empty:
    trades_df.to_csv(TRADES_CSV, index=False)

if not yearly_df.empty:
    yearly_df.to_csv(YEARLY_CSV, index=False)

print(f"\nSaved detailed bar study to: {OUTPUT_CSV}")
print(f"Saved summary stats to: {SUMMARY_CSV}")
if not trades_df.empty:
    print(f"Saved regime trade list to: {TRADES_CSV}")
if not yearly_df.empty:
    print(f"Saved yearly stats to: {YEARLY_CSV}")
