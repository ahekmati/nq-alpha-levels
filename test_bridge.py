from mt5linux import MetaTrader5

mt5 = MetaTrader5()

if not mt5.initialize():
    print("initialize failed:", mt5.last_error())
    raise SystemExit(1)

print("version:", mt5.version())
print("terminal_info:", mt5.terminal_info())
print("account_info:", mt5.account_info())

mt5.shutdown()
