from mt5linux import MetaTrader5
import pandas as pd

mt5 = MetaTrader5(host="localhost", port=18812)

if not mt5.initialize():
    print("INIT FAIL:", mt5.last_error())
    raise SystemExit

print("TERMINAL OK")
print("LAST ERROR:", mt5.last_error())

symbols = mt5.symbols_get()
print("TOTAL SYMBOLS:", 0 if symbols is None else len(symbols))

names = []
if symbols:
    names = [s.name for s in symbols]

hits = [n for n in names if "MNQ" in n.upper() or "NAS" in n.upper() or "NQ" in n.upper()]
print("MATCHES:")
for h in hits[:100]:
    print(h)

target = "MNQ"
info = mt5.symbol_info(target)
print("SYMBOL_INFO(MNQ):", info)

selected = mt5.symbol_select(target, True)
print("SYMBOL_SELECT(MNQ):", selected, "LAST_ERROR:", mt5.last_error())

for tf_name, tf in [("H1", mt5.TIMEFRAME_H1), ("D1", mt5.TIMEFRAME_D1)]:
    rates = mt5.copy_rates_from_pos(target, tf, 0, 10)
    print(f"{tf_name} RATES TYPE:", type(rates), "LAST_ERROR:", mt5.last_error())
    if rates is not None:
        df = pd.DataFrame(rates)
        print(df.tail())

mt5.shutdown()
