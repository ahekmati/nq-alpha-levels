from mt5linux import MetaTrader5
import pandas as pd
import numpy as np

# --- Settings ---
SYMBOL = "MNQM26"      # AMP MNQ June 2026 contract
BARS = 500

EMA_FAST = 26
EMA_SLOW = 150
ATR_PERIOD = 14
RSI_PERIOD = 14


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


mt5 = MetaTrader5()

if not mt5.initialize():
    print("initialize failed:", mt5.last_error())
    raise SystemExit(1)

TIMEFRAME = mt5.TIMEFRAME_H1

# show a tick just to confirm connectivity
tick = mt5.symbol_info_tick(SYMBOL)
print("tick:", tick)

rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, BARS)
if rates is None or len(rates) == 0:
    print("No rates returned:", mt5.last_error())
    mt5.shutdown()
    raise SystemExit(1)

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
df = df.sort_values("time").reset_index(drop=True)

df["ema_fast"] = ema(df["close"], EMA_FAST)
df["ema_slow"] = ema(df["close"], EMA_SLOW)
df["atr_14"] = atr(df, ATR_PERIOD)
df["rsi_14"] = rsi(df["close"], RSI_PERIOD)

last = df.iloc[-1]

print("symbol:", SYMBOL)
print("bars:", len(df))
print("last_time_utc:", last["time"])
print("last_open:", last["open"])
print("last_high:", last["high"])
print("last_low:", last["low"])
print("last_close:", last["close"])
print("last_spread:", last["spread"])
print("last_tick_volume:", last["tick_volume"])
print("ema_fast:", round(float(last["ema_fast"]), 6))
print("ema_slow:", round(float(last["ema_slow"]), 6))
print("atr_14:", round(float(last["atr_14"]), 6))
print("rsi_14:", round(float(last["rsi_14"]), 6))

print("\nLast 5 rows:")
print(
    df[["time", "close", "ema_fast", "ema_slow", "atr_14", "rsi_14"]]
    .tail(5)
    .to_string(index=False)
)

mt5.shutdown()
