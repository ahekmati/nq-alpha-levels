from mt5linux import MetaTrader5
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------- CONFIG ---------------- #
SYMBOL = "@MNQ"
HOST = "127.0.0.1"
PORT = 18812
START_DATE = datetime(2020, 1, 1, tzinfo=timezone.utc)

OUTPUT_DIR = Path("research_mnq_bear_model")
OUTPUT_DIR.mkdir(exist_ok=True)

PRE_BEAR_DAYS = 10
RECOVERY_DAYS = 10
EXHAUSTION_RSI_LEVEL = 30.0

RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14

BEAR_WINDOWS = [
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
# ---------------------------------------- #


def parse_date_utc(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def get_mt5():
    mt5 = MetaTrader5(host=HOST, port=PORT)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    return mt5


def fetch_bars(symbol: str, timeframe, start: datetime, end: datetime) -> pd.DataFrame:
    mt5 = get_mt5()
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"No rates returned for {symbol}, timeframe={timeframe}, error={err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    mt5.shutdown()
    return df


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)

    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    tr_smooth = pd.Series(tr, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)

    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_features(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    d = df.copy()

    d[f"{prefix}body"] = d["close"] - d["open"]
    d[f"{prefix}body_abs"] = d[f"{prefix}body"].abs()
    d[f"{prefix}range"] = d["high"] - d["low"]
    d[f"{prefix}upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d[f"{prefix}lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]

    d[f"{prefix}is_red"] = (d["close"] < d["open"]).astype(int)
    d[f"{prefix}is_green"] = (d["close"] > d["open"]).astype(int)

    d[f"{prefix}ret_1"] = d["close"].pct_change()
    d[f"{prefix}ret_3"] = d["close"].pct_change(3)
    d[f"{prefix}ret_5"] = d["close"].pct_change(5)
    d[f"{prefix}ret_10"] = d["close"].pct_change(10)
    d[f"{prefix}log_ret"] = np.log(d["close"]).diff()

    d[f"{prefix}rsi"] = calc_rsi(d["close"], RSI_PERIOD)
    d[f"{prefix}atr"] = calc_atr(d, ATR_PERIOD)
    d[f"{prefix}adx"] = calc_adx(d, ADX_PERIOD)

    for span in [10, 20, 50, 100, 200]:
        d[f"{prefix}ema_{span}"] = d["close"].ewm(span=span, adjust=False).mean()

    d[f"{prefix}ema10_slope_5"] = d[f"{prefix}ema_10"] - d[f"{prefix}ema_10"].shift(5)
    d[f"{prefix}ema20_slope_5"] = d[f"{prefix}ema_20"] - d[f"{prefix}ema_20"].shift(5)
    d[f"{prefix}ema50_slope_5"] = d[f"{prefix}ema_50"] - d[f"{prefix}ema_50"].shift(5)

    d[f"{prefix}dist_ema20_atr"] = (d["close"] - d[f"{prefix}ema_20"]) / d[f"{prefix}atr"]
    d[f"{prefix}dist_ema50_atr"] = (d["close"] - d[f"{prefix}ema_50"]) / d[f"{prefix}atr"]
    d[f"{prefix}dist_ema200_atr"] = (d["close"] - d[f"{prefix}ema_200"]) / d[f"{prefix}atr"]

    d[f"{prefix}rolling_red_ratio_5"] = d[f"{prefix}is_red"].rolling(5).mean()
    d[f"{prefix}rolling_red_ratio_10"] = d[f"{prefix}is_red"].rolling(10).mean()
    d[f"{prefix}rolling_green_ratio_5"] = d[f"{prefix}is_green"].rolling(5).mean()

    d[f"{prefix}rolling_vol_10"] = d[f"{prefix}log_ret"].rolling(10).std()
    d[f"{prefix}downside_vol_10"] = d[f"{prefix}log_ret"].where(d[f"{prefix}log_ret"] < 0).rolling(10).std()

    peak_close = d["close"].cummax()
    d[f"{prefix}drawdown_pct"] = (d["close"] / peak_close) - 1.0

    red_streak_vals = []
    green_streak_vals = []
    red_streak = 0
    green_streak = 0

    for is_red, is_green in zip(d[f"{prefix}is_red"], d[f"{prefix}is_green"]):
        if is_red:
            red_streak += 1
        else:
            red_streak = 0

        if is_green:
            green_streak += 1
        else:
            green_streak = 0

        red_streak_vals.append(red_streak)
        green_streak_vals.append(green_streak)

    d[f"{prefix}red_streak"] = red_streak_vals
    d[f"{prefix}green_streak"] = green_streak_vals

    return d


def label_daily_regimes(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.copy()
    d["date"] = d.index.floor("D")
    d["label"] = "OUTSIDE"
    d["bear_window_id"] = np.nan

    for i, (start_str, end_str) in enumerate(BEAR_WINDOWS, 1):
        start = parse_date_utc(start_str)
        end = parse_date_utc(end_str)

        pre_start = start - timedelta(days=PRE_BEAR_DAYS)
        post_end = end + timedelta(days=RECOVERY_DAYS)

        pre_mask = (d.index >= pre_start) & (d.index < start)
        active_mask = (d.index >= start) & (d.index <= end)
        recovery_mask = (d.index > end) & (d.index <= post_end)

        d.loc[pre_mask & (d["label"] == "OUTSIDE"), "label"] = "PRE_BEAR"
        d.loc[pre_mask, "bear_window_id"] = i

        d.loc[active_mask, "label"] = "ACTIVE_BEAR"
        d.loc[active_mask, "bear_window_id"] = i

        d.loc[recovery_mask & (d["label"] == "OUTSIDE"), "label"] = "RECOVERY"
        d.loc[recovery_mask, "bear_window_id"] = i

    for window_id in sorted(d["bear_window_id"].dropna().unique()):
        mask = d["bear_window_id"] == window_id
        subset = d.loc[mask].copy()

        active = subset[subset["label"] == "ACTIVE_BEAR"].copy()
        if active.empty:
            continue

        exhaustion_candidates = active[active["rsi"] <= EXHAUSTION_RSI_LEVEL]
        if not exhaustion_candidates.empty:
            exhaustion_start = exhaustion_candidates.index[0]
            d.loc[(d.index >= exhaustion_start) & (d.index <= active.index[-1]) & (d["bear_window_id"] == window_id), "label"] = "EXHAUSTION"

    return d


def merge_daily_into_h1(h1: pd.DataFrame, daily_labeled: pd.DataFrame) -> pd.DataFrame:
    d = daily_labeled.copy()
    d["date"] = d.index.floor("D")

    daily_cols = [
        "date",
        "label",
        "bear_window_id",
        "rsi",
        "atr",
        "adx",
        "drawdown_pct",
        "ema_20",
        "ema_50",
        "ema_200",
        "ema20_slope_5",
        "ema50_slope_5",
        "dist_ema20_atr",
        "dist_ema50_atr",
        "dist_ema200_atr",
        "rolling_red_ratio_5",
        "rolling_red_ratio_10",
        "downside_vol_10",
        "red_streak",
        "green_streak",
    ]

    d = d[daily_cols].copy()
    d = d.rename(columns={
        "label": "daily_label",
        "rsi": "daily_rsi",
        "atr": "daily_atr",
        "adx": "daily_adx",
        "drawdown_pct": "daily_drawdown_pct",
        "ema_20": "daily_ema_20",
        "ema_50": "daily_ema_50",
        "ema_200": "daily_ema_200",
        "ema20_slope_5": "daily_ema20_slope_5",
        "ema50_slope_5": "daily_ema50_slope_5",
        "dist_ema20_atr": "daily_dist_ema20_atr",
        "dist_ema50_atr": "daily_dist_ema50_atr",
        "dist_ema200_atr": "daily_dist_ema200_atr",
        "rolling_red_ratio_5": "daily_rolling_red_ratio_5",
        "rolling_red_ratio_10": "daily_rolling_red_ratio_10",
        "downside_vol_10": "daily_downside_vol_10",
        "red_streak": "daily_red_streak",
        "green_streak": "daily_green_streak",
    })

    h = h1.copy()
    h["date"] = h.index.floor("D")

    merged = h.reset_index().merge(d, on="date", how="left")
    merged = merged.set_index("time")
    return merged


def save_label_summary(daily_labeled: pd.DataFrame):
    summary = (
        daily_labeled
        .groupby(["bear_window_id", "label"])
        .size()
        .reset_index(name="bars")
        .sort_values(["bear_window_id", "label"])
    )
    summary.to_csv(OUTPUT_DIR / "bear_label_summary.csv", index=False)


def main():
    end = datetime.now(timezone.utc)

    print(f"Fetching continuous D1 data for {SYMBOL} from {START_DATE.date()} to {end.date()} ...")
    daily = fetch_bars(SYMBOL, MetaTrader5.TIMEFRAME_D1, START_DATE, end)
    print(f"Got {len(daily)} D1 bars.")

    print(f"Fetching continuous H1 data for {SYMBOL} from {START_DATE.date()} to {end.date()} ...")
    h1 = fetch_bars(SYMBOL, MetaTrader5.TIMEFRAME_H1, START_DATE, end)
    print(f"Got {len(h1)} H1 bars.")

    print("Adding daily features ...")
    daily_feat = add_features(daily)

    print("Labeling daily regimes from hand-marked bear windows ...")
    daily_labeled = label_daily_regimes(daily_feat)

    print("Adding H1 features ...")
    h1_feat = add_features(h1, prefix="h1_")

    print("Merging daily context into H1 ...")
    h1_research = merge_daily_into_h1(h1_feat, daily_labeled)

    daily_research = daily_labeled.reset_index().copy()
    h1_research = h1_research.reset_index().copy()

    daily_research.to_csv(OUTPUT_DIR / "mnq_daily_research_dataset.csv", index=False)
    h1_research.to_csv(OUTPUT_DIR / "mnq_h1_research_dataset.csv", index=False)
    save_label_summary(daily_labeled)

    windows_df = pd.DataFrame([
        {"bear_window_id": i + 1, "start": w[0], "end": w[1]}
        for i, w in enumerate(BEAR_WINDOWS)
    ])
    windows_df.to_csv(OUTPUT_DIR / "bear_windows_reference.csv", index=False)

    print("\nSaved files:")
    print(f"- {OUTPUT_DIR / 'mnq_daily_research_dataset.csv'}")
    print(f"- {OUTPUT_DIR / 'mnq_h1_research_dataset.csv'}")
    print(f"- {OUTPUT_DIR / 'bear_label_summary.csv'}")
    print(f"- {OUTPUT_DIR / 'bear_windows_reference.csv'}")

    print("\nDaily label counts:")
    print(daily_research["label"].value_counts(dropna=False).to_string())

    print("\nExample daily columns:")
    print(", ".join(daily_research.columns[:25]))

    print("\nExample H1 columns:")
    print(", ".join(h1_research.columns[:30]))


if __name__ == "__main__":
    main()
