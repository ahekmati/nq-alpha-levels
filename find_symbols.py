from mt5linux import MetaTrader5

mt5 = MetaTrader5()

if not mt5.initialize():
    print("initialize failed:", mt5.last_error())
    raise SystemExit(1)

symbols = mt5.symbols_get()
if symbols is None:
    print("symbols_get failed:", mt5.last_error())
    mt5.shutdown()
    raise SystemExit(1)

for s in symbols:
    name = getattr(s, "name", "")
    if any(x in name.upper() for x in ["NQ", "MNQ", "ES", "MES", "YM", "MYM", "RTY", "M2K"]):
        print(name)

mt5.shutdown()
