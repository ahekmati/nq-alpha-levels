from mt5linux import MetaTrader5
from datetime import datetime, timezone
import pandas as pd
import os


# ---------------- CONFIG ---------------- #
SYMBOL = "@MNQ"
HOST = "127.0.0.1"
PORT = 18812

# Your periods (inclusive start, inclusive end by date)
PERIODS = [
    ("2020-02-18", "2020-03-23"),
    ("2020-09-01", "2020-09-24"),
    ("2020-10-13", "2020-11-02"),
    ("2021-02-15", "2021-03-05"),
    ("2021-04-26", "2021-05-13"),
    ("2021-12-27", "2022-03-15"),
    ("2022-04-04", "2023-01-10"),
    ("2023-07-30", "2023-08-20"),
    ("2024-03-24", "2024-04-21"),
    ("2024-07-10", "2024-08-08"),
    ("2025-02-17", "2025-04-07"),
    ("2026-01-28", "2026-03-20"),
]

OUTPUT_DIR = "data_mnq_bear_segments"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ---------------------------------------- #


def parse_date_utc(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def fetch_range(symbol: str, start: datetime, end: datetime, timeframe) -> pd.DataFrame:
    mt5 = MetaTrader5(host=HOST, port=PORT)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"No rates returned for {symbol} {timeframe}, error={err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    mt5.shutdown()
    return df


def main():
    all_daily = []
    all_h1 = []

    for idx, (start_str, end_str) in enumerate(PERIODS, 1):
        start_dt = parse_date_utc(start_str)
        # add one day to end so MT5 includes that end date
        end_dt = parse_date_utc(end_str) + pd.Timedelta(days=1)

        print(f"\nSegment {idx}: {start_str} -> {end_str}")

        # Daily
        print("  Fetching D1 ...")
        daily = fetch_range(SYMBOL, start_dt, end_dt, MetaTrader5.TIMEFRAME_D1)
        daily["segment"] = idx
        daily["segment_start"] = start_str
        daily["segment_end"] = end_str
        all_daily.append(daily)

        daily_path = os.path.join(
            OUTPUT_DIR, f"mnq_segment_{idx:02d}_D1_{start_str}_to_{end_str}.csv"
        )
        daily.reset_index().to_csv(daily_path, index=False)
        print(f"  Saved daily to {daily_path} ({len(daily)} rows)")

        # Hourly
        print("  Fetching H1 ...")
        h1 = fetch_range(SYMBOL, start_dt, end_dt, MetaTrader5.TIMEFRAME_H1)
        h1["segment"] = idx
        h1["segment_start"] = start_str
        h1["segment_end"] = end_str
        all_h1.append(h1)

        h1_path = os.path.join(
            OUTPUT_DIR, f"mnq_segment_{idx:02d}_H1_{start_str}_to_{end_str}.csv"
        )
        h1.reset_index().to_csv(h1_path, index=False)
        print(f"  Saved H1 to {h1_path} ({len(h1)} rows)")

    if all_daily:
        daily_all = pd.concat(all_daily).sort_index()
        daily_all_path = os.path.join(OUTPUT_DIR, "mnq_all_segments_D1.csv")
        daily_all.reset_index().to_csv(daily_all_path, index=False)
        print(f"\nSaved combined daily data to {daily_all_path} ({len(daily_all)} rows)")

    if all_h1:
        h1_all = pd.concat(all_h1).sort_index()
        h1_all_path = os.path.join(OUTPUT_DIR, "mnq_all_segments_H1.csv")
        h1_all.reset_index().to_csv(h1_all_path, index=False)
        print(f"Saved combined H1 data to {h1_all_path} ({len(h1_all)} rows)")


if __name__ == "__main__":
    main()
