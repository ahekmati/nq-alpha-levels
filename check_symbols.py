from mt5linux import MetaTrader5

mt5 = MetaTrader5()
if not mt5.initialize():
    print(f"Init failed: {mt5.last_error()}")
    exit()

# Find anything with MNQ in the name
symbols = mt5.symbols_get()
mnq_matches = [s.name for s in symbols if "MNQ" in s.name.upper()]
nq_matches  = [s.name for s in symbols if "NQ" in s.name.upper()]

print("MNQ matches:", mnq_matches)
print("NQ matches:", nq_matches)

mt5.shutdown()
