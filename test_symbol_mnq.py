from mt5linux import MetaTrader5

SYMBOL = "MNQM26"  # or "@MNQ" if AMP uses that as the tradable root

mt5 = MetaTrader5()
if not mt5.initialize():
    print("init failed:", mt5.last_error())
    raise SystemExit(1)

info = mt5.symbol_info(SYMBOL)
print("symbol_info:", info)

if info is None:
    print("Symbol not found:", SYMBOL)
    print("last_error:", mt5.last_error())
    mt5.shutdown()
    raise SystemExit(1)

if not info.visible:
    print("symbol not visible, selecting...")
    print("symbol_select:", mt5.symbol_select(SYMBOL, True))

tick = mt5.symbol_info_tick(SYMBOL)
print("tick:", tick)

if tick is None:
    print("Could not get tick for symbol", SYMBOL)
    print("last_error:", mt5.last_error())

mt5.shutdown()
