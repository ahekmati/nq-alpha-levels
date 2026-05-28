from mt5linux import MetaTrader5
import pandas as pd
import numpy as np

SYMBOL_CANDIDATES = ["@MNQ", "MNQ", "MNQM26", "MNQU26", "MNQZ26"]
D1_TF = MetaTrader5.TIMEFRAME_D1
H1_TF = MetaTrader5.TIMEFRAME_H1

D1_BARS = 5000
H1_BARS = 30000

EMA_PERIOD = 100
ATR_PERIOD = 14
ATR_THRESHOLD = 140.0
HOLD_BARS = 24

OUTPUT_SIGNALS_CSV = "ema100_h1_dip_signals.csv"
OUTPUT_SUMMARY_CSV = "ema100_h1_dip_summary.csv"

mt5 = MetaTrader5()

def resolve_symbol(mt5_client, candidates, tf):
    for sym in candidates:
        info = mt5_client.symbol_info(sym)
        if info is None:
            continue
        mt5_client.symbol_select(sym, True)
        rates = mt5_client.copy_rates_from_pos(sym, tf, 0, 20)
        if rates is not None and len(rates) > 0:
            return sym
    all_symbols = mt5_client.symbols_get()
    mnq_like = [s.name for s in all_symbols if "MNQ" in s.name.upper()] if all_symbols else []
    raise RuntimeError(f"Could not resolve symbol. MNQ-like symbols visible: {mnq_like[:50]}")

def true_range(df):
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr_wilder(df, period=14):
    tr = true_range(df)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def summarize_variant(df, variant_name):
    x = df[df["variant"] == variant_name].copy()
    if x.empty:
        return None
    return {
        "variant": variant_name,
        "signals": int(len(x)),
        "win_rate_pct": float((x["ret"] > 0).mean() * 100.0),
        "mean_ret_pct": float(x["ret"].mean() * 100.0),
        "median_ret_pct": float(x["ret"].median() * 100.0),
        "mean_mae_pct": float(x["mae"].mean() * 100.0),
        "mean_mfe_pct": float(x["mfe"].mean() * 100.0),
    }

if not mt5.initialize():
    raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

symbol = resolve_symbol(mt5, SYMBOL_CANDIDATES, D1_TF)

d1_rates = mt5.copy_rates_from_pos(symbol, D1_TF, 0, D1_BARS)
h1_rates = mt5.copy_rates_from_pos(symbol, H1_TF, 0, H1_BARS)
mt5.shutdown()

if d1_rates is None or len(d1_rates) == 0:
    raise RuntimeError("No D1 bars returned.")
if h1_rates is None or len(h1_rates) == 0:
    raise RuntimeError("No H1 bars returned.")

d1 = pd.DataFrame(d1_rates)
d1["time"] = pd.to_datetime(d1["time"], unit="s")
d1 = d1.sort_values("time").reset_index(drop=True)
d1["date"] = d1["time"].dt.floor("D")
d1["ema100"] = d1["close"].ewm(span=EMA_PERIOD, adjust=False).mean()
d1["daily_bull"] = d1["close"] > d1["ema100"]

h1 = pd.DataFrame(h1_rates)
h1["time"] = pd.to_datetime(h1["time"], unit="s")
h1 = h1.sort_values("time").reset_index(drop=True)
h1["date"] = h1["time"].dt.floor("D")
h1["atr"] = atr_wilder(h1, ATR_PERIOD)
h1["red"] = h1["close"] < h1["open"]

h1 = h1.merge(d1[["date", "daily_bull"]], on="date", how="left")
h1["daily_bull"] = h1["daily_bull"].fillna(False)

signals = []
for i in range(ATR_PERIOD + 5, len(h1) - HOLD_BARS - 3):
    row = h1.iloc[i]

    if not row["daily_bull"]:
        continue
    if not row["red"]:
        continue
    if row["atr"] < ATR_THRESHOLD:
        continue

    signal_time = row["time"]
    signal_close = row["close"]
    signal_high = row["high"]
    signal_mid = (row["high"] + row["low"]) / 2.0

    variants = []

    # 1) Buy next close
    entry_idx = i + 1
    if entry_idx < len(h1):
        entry_price = h1.iloc[entry_idx]["close"]
        exit_idx = min(entry_idx + HOLD_BARS, len(h1) - 1)
        path = h1.iloc[entry_idx:exit_idx + 1]
        variants.append(("next_close", entry_idx, entry_price, path))

    # 2) Wait 1 bar then buy close
    entry_idx = i + 2
    if entry_idx < len(h1):
        entry_price = h1.iloc[entry_idx]["close"]
        exit_idx = min(entry_idx + HOLD_BARS, len(h1) - 1)
        path = h1.iloc[entry_idx:exit_idx + 1]
        variants.append(("wait_1_bar_close", entry_idx, entry_price, path))

    # 3) Wait 2 bars then buy close
    entry_idx = i + 3
    if entry_idx < len(h1):
        entry_price = h1.iloc[entry_idx]["close"]
        exit_idx = min(entry_idx + HOLD_BARS, len(h1) - 1)
        path = h1.iloc[entry_idx:exit_idx + 1]
        variants.append(("wait_2_bar_close", entry_idx, entry_price, path))

    # 4) Buy on break above red candle high
    found = False
    for j in range(i + 1, min(i + 7, len(h1))):
        if h1.iloc[j]["high"] > signal_high:
            entry_idx = j
            entry_price = signal_high
            exit_idx = min(entry_idx + HOLD_BARS, len(h1) - 1)
            path = h1.iloc[entry_idx:exit_idx + 1]
            variants.append(("break_signal_high", entry_idx, entry_price, path))
            found = True
            break

    # 5) Buy on reclaim of midpoint
    for j in range(i + 1, min(i + 7, len(h1))):
        if h1.iloc[j]["high"] > signal_mid:
            entry_idx = j
            entry_price = signal_mid
            exit_idx = min(entry_idx + HOLD_BARS, len(h1) - 1)
            path = h1.iloc[entry_idx:exit_idx + 1]
            variants.append(("reclaim_midpoint", entry_idx, entry_price, path))
            break

    for variant_name, entry_idx, entry_price, path in variants:
        if path.empty:
            continue
        exit_price = path.iloc[-1]["close"]
        ret = exit_price / entry_price - 1.0
        mae = (path["low"].min() / entry_price) - 1.0
        mfe = (path["high"].max() / entry_price) - 1.0

        signals.append({
            "signal_time": signal_time,
            "variant": variant_name,
            "entry_time": h1.iloc[entry_idx]["time"],
            "entry_price": entry_price,
            "exit_time": path.iloc[-1]["time"],
            "exit_price": exit_price,
            "ret": ret,
            "mae": mae,
            "mfe": mfe,
            "signal_atr": row["atr"],
            "signal_high": signal_high,
            "signal_mid": signal_mid,
            "signal_close": signal_close,
        })

signals_df = pd.DataFrame(signals)
if signals_df.empty:
    raise RuntimeError("No qualifying H1 dip signals found.")

summary_rows = []
for variant in sorted(signals_df["variant"].unique()):
    row = summarize_variant(signals_df, variant)
    if row:
        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

print(f"\nResolved symbol: {symbol}")
print(f"D1 bars: {len(d1)} | H1 bars: {len(h1)}")
print(f"EMA period: {EMA_PERIOD}")
print(f"ATR period: {ATR_PERIOD}")
print(f"ATR threshold: {ATR_THRESHOLD}")
print(f"Holding bars: {HOLD_BARS}")

print("\n=== Dip Timing Summary ===")
print(summary_df.to_string(index=False))

signals_df.to_csv(OUTPUT_SIGNALS_CSV, index=False)
summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)

print(f"\nSaved signals to: {OUTPUT_SIGNALS_CSV}")
print(f"Saved summary to: {OUTPUT_SUMMARY_CSV}")
