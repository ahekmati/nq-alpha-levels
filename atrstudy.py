from datetime import datetime
from mt5linux import MetaTrader5
import pandas as pd
import numpy as np

SYMBOL = "MNQ"
TIMEFRAME = MetaTrader5.TIMEFRAME_D1
BARS = 1500
ATR_PERIOD = 14
SWING_LEFT = 5
SWING_RIGHT = 5
OUTPUT_CSV = "atr_d1_extremes_python.csv"

mt5 = MetaTrader5()

def true_range(df):
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr_wilder(df, period=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def is_swing_high(arr, i, left_bars, right_bars):
    if i - left_bars < 0 or i + right_bars >= len(arr):
        return False
    v = arr[i]
    window = arr[i-left_bars:i+right_bars+1]
    return v == np.max(window) and np.sum(window == v) == 1

def is_swing_low(arr, i, left_bars, right_bars):
    if i - left_bars < 0 or i + right_bars >= len(arr):
        return False
    v = arr[i]
    window = arr[i-left_bars:i+right_bars+1]
    return v == np.min(window) and np.sum(window == v) == 1

if not mt5.initialize():
    raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)

if rates is None or len(rates) == 0:
    mt5.shutdown()
    raise RuntimeError(f"No rates returned. last_error={mt5.last_error()}")

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df = df.sort_values("time").reset_index(drop=True)

df["atr"] = atr_wilder(df, ATR_PERIOD)
df["extreme_type"] = ""

atr_vals = df["atr"].to_numpy()

for i in range(len(df)):
    if is_swing_high(atr_vals, i, SWING_LEFT, SWING_RIGHT):
        df.loc[i, "extreme_type"] = "SWING_HIGH"
    elif is_swing_low(atr_vals, i, SWING_LEFT, SWING_RIGHT):
        df.loc[i, "extreme_type"] = "SWING_LOW"

abs_high_idx = df["atr"].idxmax()
abs_low_idx = df["atr"].idxmin()

df.loc[abs_high_idx, "extreme_type"] = (
    df.loc[abs_high_idx, "extreme_type"] + "|ABSOLUTE_HIGH"
    if df.loc[abs_high_idx, "extreme_type"] else "ABSOLUTE_HIGH"
)
df.loc[abs_low_idx, "extreme_type"] = (
    df.loc[abs_low_idx, "extreme_type"] + "|ABSOLUTE_LOW"
    if df.loc[abs_low_idx, "extreme_type"] else "ABSOLUTE_LOW"
)

out = df[df["extreme_type"] != ""][["time", "open", "high", "low", "close", "atr", "extreme_type"]].copy()
out.to_csv(OUTPUT_CSV, index=False)

print("Done.")
print("Absolute ATR high:")
print(df.loc[abs_high_idx, ["time", "atr", "extreme_type"]])
print("Absolute ATR low:")
print(df.loc[abs_low_idx, ["time", "atr", "extreme_type"]])
print(f"Saved to {OUTPUT_CSV}")

mt5.shutdown()
