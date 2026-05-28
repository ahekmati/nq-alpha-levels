import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
from mt5linux import MetaTrader5


# =========================
# CONFIG
# =========================
BASE_DIR = Path("research_mnq_bear_model")
PROMO_DIR = BASE_DIR / "promotion_early_short_proto_v1"
OUT_FILE = PROMO_DIR / "h1_execution_dataset_with_proto_v1_live.csv"
MERGED_OUT_FILE = PROMO_DIR / "h1_execution_dataset_with_proto_v1.csv"
LIVE_LOG_FILE = PROMO_DIR / "live_update_log.csv"

MT5_HOST = "localhost"
MT5_PORT = 18812

MT5_SYMBOL = ""
MT5_SYMBOL_ROOT = "MNQ"
AUTO_CONTRACT_ROLLOVER = True

H1_BARS = 2500
D1_BARS = 500

SESSION_START_HOUR_UTC = 14
SESSION_END_HOUR_UTC = 20

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

QUARTER_MONTHS = [3, 6, 9, 12]
MONTH_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}

mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)


def utc_now():
    return datetime.now(timezone.utc)


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def equity_index_roll_date(year: int, month: int) -> date:
    return third_friday(year, month) - timedelta(days=4)


def next_quarter(year: int, month: int):
    for m in QUARTER_MONTHS:
        if m > month:
            return year, m
    return year + 1, 3


def current_or_next_active_quarter(now_utc: datetime):
    y = now_utc.year
    m = now_utc.month

    if m <= 3:
        q_month = 3
    elif m <= 6:
        q_month = 6
    elif m <= 9:
        q_month = 9
    else:
        q_month = 12

    rd = equity_index_roll_date(y, q_month)
    if now_utc.date() >= rd:
        return next_quarter(y, q_month)
    return y, q_month


def resolve_front_month_symbol(now_utc: datetime):
    if MT5_SYMBOL:
        return MT5_SYMBOL
    if not AUTO_CONTRACT_ROLLOVER:
        raise RuntimeError("MT5_SYMBOL is empty and AUTO_CONTRACT_ROLLOVER=False")
    year, month = current_or_next_active_quarter(now_utc)
    code = MONTH_CODE[month]
    yy = str(year)[-2:]
    return f"{MT5_SYMBOL_ROOT}{code}{yy}"


def log_event(event_type, message, extra=None):
    resolved_symbol = resolve_front_month_symbol(utc_now())
    row = {
        "ts_utc": utc_now().isoformat(),
        "event_type": event_type,
        "message": message,
        "symbol_root": MT5_SYMBOL_ROOT,
        "resolved_symbol": resolved_symbol,
        "auto_contract_rollover": AUTO_CONTRACT_ROLLOVER,
    }
    if extra:
        row.update(extra)

    df_new = pd.DataFrame([row])
    if LIVE_LOG_FILE.exists():
        df_old = pd.read_csv(LIVE_LOG_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LIVE_LOG_FILE, index=False)
    print(f"[{row['ts_utc']}] {event_type}: {message}")


def ensure_connection():
    ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")


def ensure_symbol(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info failed for {symbol}: {mt5.last_error()}")

    selected = mt5.symbol_select(symbol, True)
    if not selected:
        raise RuntimeError(f"symbol_select failed for {symbol}: {mt5.last_error()}")

    return info


def fetch_rates(symbol, timeframe, count):
    ensure_symbol(symbol)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        raise RuntimeError(f"copy_rates_from_pos failed for {symbol}, timeframe={timeframe}, error={mt5.last_error()}")
    df = pd.DataFrame(rates)
    if df.empty:
        raise RuntimeError(f"No rates returned for {symbol}, timeframe={timeframe}")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    return df.sort_values("time").reset_index(drop=True)


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def adx(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    tr_smooth = pd.Series(tr, index=df.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di = 100 * (plus_dm_smooth / tr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm_smooth / tr_smooth.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def compute_daily_features(d1):
    d = d1.copy()

    d["daily_model_rsi"] = rsi(d["close"], 14)
    d["daily_model_ema20"] = ema(d["close"], 20)
    d["daily_model_ema50"] = ema(d["close"], 50)
    d["daily_model_atr"] = atr(d, 14)
    d["daily_model_adx"] = adx(d, 14)

    d["daily_model_ema20_slope_5"] = d["daily_model_ema20"] - d["daily_model_ema20"].shift(5)
    d["daily_model_ema50_slope_5"] = d["daily_model_ema50"] - d["daily_model_ema50"].shift(5)

    d["rsi70_streak"] = 0
    streak = 0
    for i, val in enumerate(d["daily_model_rsi"] > 70):
        streak = streak + 1 if bool(val) else 0
        d.loc[d.index[i], "rsi70_streak"] = streak

    d["ob_streak_not_making_higher_high"] = 0
    current_peak = np.nan
    prev_peak_list = []
    for is_ob, h in zip(d["daily_model_rsi"] > 70, d["high"]):
        if is_ob:
            prev_peak_list.append(current_peak)
            current_peak = h if pd.isna(current_peak) else max(current_peak, h)
        else:
            prev_peak_list.append(np.nan)
            current_peak = np.nan

    d["prev_ob_peak_high"] = prev_peak_list
    d["ob_streak_not_making_higher_high"] = (
        (d["rsi70_streak"] >= 2) &
        (d["high"] < d["prev_ob_peak_high"])
    ).astype(int)

    d["rsi_rolling_over"] = (
        (d["daily_model_rsi"] < d["daily_model_rsi"].shift(1)) &
        (d["daily_model_rsi"] > 55)
    ).astype(int)

    d["near_end_of_avg_ob_streak"] = (d["rsi70_streak"] >= 3).astype(int)

    bear_prob = (
        0.35 * (50 - (d["daily_model_rsi"] - 50).abs()) / 50.0 +
        0.30 * (d["daily_model_ema20"] < d["daily_model_ema50"]).astype(float) +
        0.20 * (d["daily_model_ema20_slope_5"] <= 0).astype(float) +
        0.15 * (d["close"] < d["daily_model_ema20"]).astype(float)
    )
    d["bear_prob"] = bear_prob.clip(lower=0.0, upper=1.0)

    exhaust_prob = (
        0.45 * (d["daily_model_rsi"] > 68).astype(float) +
        0.35 * (d["rsi70_streak"] >= 2).astype(float) +
        0.20 * (d["ob_streak_not_making_higher_high"] == 1).astype(float)
    )
    d["exhaust_prob"] = exhaust_prob.clip(lower=0.0, upper=1.0)

    d["date"] = d["time"].dt.floor("D")
    return d


def assign_bear_windows_h1(h1):
    h = h1.copy()
    h["bear_window_id"] = np.nan
    h["hours_from_bear_start"] = np.nan
    h["is_early_window"] = 0

    for i, (s, e) in enumerate(BEAR_WINDOWS, 1):
        start = pd.Timestamp(s, tz="UTC")
        end = pd.Timestamp(e, tz="UTC")
        early_end = min(start + pd.Timedelta(days=8), end)

        in_window = (h["time"] >= start) & (h["time"] <= end)
        early_mask = (h["time"] >= start) & (h["time"] <= early_end)

        h.loc[in_window, "bear_window_id"] = i
        h.loc[in_window, "hours_from_bear_start"] = (
            (h.loc[in_window, "time"] - start) / pd.Timedelta(hours=1)
        )
        h.loc[early_mask, "is_early_window"] = 1

    return h


def compute_h1_features(h1):
    h = h1.copy()
    h["h1_ema_20"] = ema(h["close"], 20)
    h["h1_rsi"] = rsi(h["close"], 14)
    h["hh_8"] = h["high"].rolling(8).max().shift(1)
    h["ll_8"] = h["low"].rolling(8).min().shift(1)
    h["range_pos"] = (h["close"] - h["ll_8"]) / (h["hh_8"] - h["ll_8"]).replace(0, np.nan)
    h["h1_ret_4"] = h["close"].pct_change(4)
    h["hour_utc"] = h["time"].dt.hour
    h["in_session_v3"] = (
        (h["hour_utc"] >= SESSION_START_HOUR_UTC) &
        (h["hour_utc"] <= SESSION_END_HOUR_UTC)
    ).astype(int)
    return h


def merge_daily_to_h1(h1, daily):
    h = h1.copy()
    h["date"] = h["time"].dt.floor("D")

    keep = [
        "date",
        "daily_model_rsi",
        "daily_model_adx",
        "daily_model_atr",
        "daily_model_ema20",
        "daily_model_ema50",
        "daily_model_ema20_slope_5",
        "daily_model_ema50_slope_5",
        "rsi70_streak",
        "ob_streak_not_making_higher_high",
        "rsi_rolling_over",
        "near_end_of_avg_ob_streak",
        "bear_prob",
        "exhaust_prob",
    ]
    return h.merge(daily[keep].copy(), on="date", how="left")


def stamp_proto_signal(h, resolved_symbol):
    out = h.copy()

    out["upper_range_failure_v3"] = (
        (out["range_pos"] >= 0.65) &
        (out["high"] >= out["hh_8"]) &
        (out["close"] < out["open"]) &
        (out["h1_rsi"] >= 42) &
        (out["h1_rsi"] <= 75)
    ).astype(int)

    out["bear_context_soft_v3"] = (
        (out["bear_prob"] >= 0.30) &
        (out["exhaust_prob"] <= 0.65) &
        (out["daily_model_rsi"] >= 35) &
        (out["daily_model_rsi"] <= 65) &
        (out["daily_model_adx"] >= 10)
    ).astype(int)

    out["rsi70_context_tag_v3"] = (
        (out["rsi70_streak"].fillna(0) >= 2) |
        (out["ob_streak_not_making_higher_high"].fillna(0).astype(int) == 1) |
        (out["near_end_of_avg_ob_streak"].fillna(0).astype(int) == 1)
    ).astype(int)

    out["close_back_below_ema20_v3"] = (
        (out["high"] >= out["h1_ema_20"]) &
        (out["close"] < out["h1_ema_20"])
    ).astype(int)

    out["stall_or_reject_tag_v3"] = (
        (out["ob_streak_not_making_higher_high"].fillna(0).astype(int) == 1) |
        (out["close_back_below_ema20_v3"] == 1)
    ).astype(int)

    out["early_short_proto_v1"] = (
        (out["is_early_window"] == 1) &
        (out["in_session_v3"] == 1) &
        (out["upper_range_failure_v3"] == 1) &
        (out["bear_context_soft_v3"] == 1)
    ).astype(int)

    out["proto_enabled"] = 1
    out["proto_stop_points"] = 300.0
    out["proto_time_exit_h"] = 48
    out["proto_r_mult"] = 2.0
    out["proto_mode"] = "stop_time"
    out["proto_has_rsi70_context"] = out["rsi70_context_tag_v3"].fillna(0).astype(int)
    out["proto_has_stall_or_reject"] = out["stall_or_reject_tag_v3"].fillna(0).astype(int)
    out["symbol_root"] = MT5_SYMBOL_ROOT
    out["resolved_symbol"] = resolved_symbol
    out["auto_contract_rollover"] = int(AUTO_CONTRACT_ROLLOVER)

    return out


def main():
    ensure_connection()
    try:
        now_utc = utc_now()
        resolved_symbol = resolve_front_month_symbol(now_utc)
        year, month = current_or_next_active_quarter(now_utc)

        log_event("START", "Starting live proto dataset update", {
            "active_quarter_year": year,
            "active_quarter_month": month,
            "roll_date": str(equity_index_roll_date(year, month)),
        })

        ensure_symbol(resolved_symbol)

        h1 = fetch_rates(resolved_symbol, mt5.TIMEFRAME_H1, H1_BARS)
        d1 = fetch_rates(resolved_symbol, mt5.TIMEFRAME_D1, D1_BARS)

        daily = compute_daily_features(d1)
        h1f = compute_h1_features(h1)
        h1f = assign_bear_windows_h1(h1f)
        merged = merge_daily_to_h1(h1f, daily)
        merged = stamp_proto_signal(merged, resolved_symbol)
        merged = merged.sort_values("time").reset_index(drop=True)

        latest_closed = merged[merged["time"] <= utc_now() - timedelta(hours=1)].tail(1)
        latest_signal = int(latest_closed["early_short_proto_v1"].iloc[0]) if not latest_closed.empty else None
        latest_time = latest_closed["time"].iloc[0] if not latest_closed.empty else None

        merged.to_csv(OUT_FILE, index=False)
        merged.to_csv(MERGED_OUT_FILE, index=False)

        log_event("LIVE_UPDATE_OK", "Updated live proto dataset", {
            "rows": len(merged),
            "latest_bar_utc": str(merged["time"].max()),
            "latest_closed_bar_utc": str(latest_time),
            "latest_closed_signal": latest_signal,
            "output_file": str(MERGED_OUT_FILE),
        })

        print("\nLatest bars:")
        print(merged[["time", "close", "resolved_symbol", "early_short_proto_v1"]].tail(10).to_string(index=False))

    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
