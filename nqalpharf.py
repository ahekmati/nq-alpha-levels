#!/usr/bin/env python3
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# =========================================================
# MT5 BACKEND LOADER
# =========================================================
MT5_BACKEND = None
mt5 = None

MT5_HOST = os.getenv("MT5_HOST", "localhost")
MT5_PORT = int(os.getenv("MT5_PORT", "18812"))

try:
    import MetaTrader5 as mt5
    MT5_BACKEND = "MetaTrader5"
except ImportError:
    try:
        from mt5linux import MetaTrader5
        mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
        MT5_BACKEND = "mt5linux"
    except ImportError:
        try:
            from pymt5linux import MetaTrader5
            mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
            MT5_BACKEND = "pymt5linux"
        except ImportError:
            mt5 = None
            MT5_BACKEND = None


# =========================
# CONFIG
# =========================
SYMBOL = os.getenv("MT5_SYMBOL", "@MNQ")
BARS_H1 = int(os.getenv("BARS_H1", "60000"))
BARS_D1 = int(os.getenv("BARS_D1", "5000"))

FAST_MA = 26
SLOW_MA = 150

DAILY_RSI_PERIOD = 14
H1_RSI_FAST = 7
H1_RSI_STD = 14
ATR_H1 = 14
ATR_D1 = 14
D1_ATR_LONG = 100

DAILY_DIP_ATR_THRESHOLD = float(os.getenv("DAILY_DIP_ATR_THRESHOLD", "100.0"))
DAILY_DIP_REQUIRE_DAILY_RSI_GT = float(os.getenv("DAILY_DIP_REQUIRE_DAILY_RSI_GT", "50.0"))
DAILY_DIP_MAX_STOP_POINTS = float(os.getenv("DAILY_DIP_MAX_STOP_POINTS", "200.0"))

RSI2300_DAILY_RSI_BUY_LEVEL = float(os.getenv("RSI2300_DAILY_RSI_BUY_LEVEL", "70.0"))
RSI2300_LTATR_PERIOD = int(os.getenv("RSI2300_LTATR_PERIOD", "7"))
RSI2300_LTATR_THRESHOLD = float(os.getenv("RSI2300_LTATR_THRESHOLD", "100.0"))
RSI2300_SESSION_START_GMT = int(os.getenv("RSI2300_SESSION_START_GMT", "13"))
RSI2300_SESSION_END_GMT = int(os.getenv("RSI2300_SESSION_END_GMT", "21"))
RSI2300_BUY_GMT_HOUR = int(os.getenv("RSI2300_BUY_GMT_HOUR", "23"))

HOLD_BARS = int(os.getenv("HOLD_BARS", "72"))
TARGET_R = float(os.getenv("TARGET_R", "2.0"))
STOP_R = float(os.getenv("STOP_R", "1.0"))

N_SPLITS = int(os.getenv("N_SPLITS", "3"))
CV_GAP = int(os.getenv("CV_GAP", "24"))
MIN_TEST_SIZE = int(os.getenv("MIN_TEST_SIZE", "10"))

MIN_ROWS_DAILY_DIP = int(os.getenv("MIN_ROWS_DAILY_DIP", "60"))
MIN_ROWS_RSI2300 = int(os.getenv("MIN_ROWS_RSI2300", "40"))
MIN_ROWS_COMBINED = int(os.getenv("MIN_ROWS_COMBINED", "80"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
SERVER_TO_GMT_HOURS = int(os.getenv("SERVER_TO_GMT_HOURS", "0"))
THRESHOLDS = [0.50, 0.55, 0.60, 0.65]


# =========================
# INDICATORS
# =========================
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def smma(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    ma_up = up.ewm(alpha=1 / period, adjust=False).mean()
    ma_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()

def slope_n(s: pd.Series, n: int) -> pd.Series:
    return (s - s.shift(n)) / n

def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def pct_rank(arr):
        a = pd.Series(arr)
        return a.rank(pct=True).iloc[-1]
    return series.rolling(window).apply(pct_rank, raw=False)

def close_location(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["low"]) / rng

def body_ratio(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (df["close"] - df["open"]).abs() / rng

def wick_ratios(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    body_high = pd.concat([df["open"], df["close"]], axis=1).max(axis=1)
    body_low = pd.concat([df["open"], df["close"]], axis=1).min(axis=1)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    upper = (df["high"] - body_high) / rng
    lower = (body_low - df["low"]) / rng
    return upper, lower

def bars_since_last_true(flag: pd.Series) -> pd.Series:
    out = np.full(len(flag), np.nan)
    last_idx = None
    vals = flag.fillna(False).astype(bool).values
    for i, v in enumerate(vals):
        if v:
            last_idx = i
            out[i] = 0
        else:
            out[i] = np.nan if last_idx is None else i - last_idx
    return pd.Series(out, index=flag.index)

def consec_count(mask: pd.Series) -> pd.Series:
    vals = mask.fillna(False).astype(bool).values
    out = np.zeros(len(vals), dtype=int)
    c = 0
    for i, v in enumerate(vals):
        if v:
            c += 1
        else:
            c = 0
        out[i] = c
    return pd.Series(out, index=mask.index)


# =========================
# MT5 LOAD
# =========================
def init_mt5():
    if mt5 is None:
        raise RuntimeError("No MT5 backend found. Install MetaTrader5, mt5linux, or pymt5linux.")
    print(f"[INFO] Using MT5 backend: {MT5_BACKEND}")
    ok = mt5.initialize()
    if not ok:
        err = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
        raise RuntimeError(f"mt5.initialize() failed: {err}")

def shutdown_mt5():
    if mt5 is not None:
        try:
            mt5.shutdown()
        except Exception:
            pass

def load_rates(symbol: str, timeframe, count: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
    if rates is None:
        err = mt5.last_error() if hasattr(mt5, "last_error") else "unknown"
        raise RuntimeError(f"copy_rates_from_pos failed for {symbol}, tf={timeframe}, error={err}")
    df = pd.DataFrame(rates)
    if df.empty:
        raise RuntimeError(f"No rates returned for {symbol}, tf={timeframe}")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)

def add_gmt_columns(df: pd.DataFrame, server_to_gmt_hours: int = 0) -> pd.DataFrame:
    out = df.copy()
    out["time_gmt"] = out["time"] - pd.to_timedelta(server_to_gmt_hours, unit="h")
    out["gmt_hour"] = out["time_gmt"].dt.hour
    out["gmt_date"] = out["time_gmt"].dt.date
    out["weekday"] = out["time_gmt"].dt.weekday
    return out


# =========================
# FEATURE BUILD
# =========================
def build_daily_features(d1: pd.DataFrame) -> pd.DataFrame:
    d = d1.copy()
    d["ema50"] = ema(d["close"], 50)
    d["ema100"] = ema(d["close"], 100)
    d["ema200"] = ema(d["close"], 200)
    d["slow_ma"] = smma(d["close"], SLOW_MA)

    d["rsi14"] = rsi(d["close"], DAILY_RSI_PERIOD)
    d["atr14"] = atr(d, ATR_D1)
    d["atr100"] = atr(d, D1_ATR_LONG)
    d["atr_pct_120"] = rolling_percentile(d["atr14"], 120)

    d["ema50_slope_5"] = slope_n(d["ema50"], 5)
    d["ema100_slope_5"] = slope_n(d["ema100"], 5)

    d["bar_range"] = d["high"] - d["low"]
    d["body_ratio"] = body_ratio(d)
    d["close_pos"] = close_location(d)
    d["trigger_bar_size_atr"] = d["bar_range"] / d["atr14"].replace(0, np.nan)

    d["is_red"] = d["close"] < d["open"]
    d["is_green"] = d["close"] > d["open"]
    d["consec_red"] = consec_count(d["is_red"])
    d["consec_green"] = consec_count(d["is_green"])

    d["price_vs_slow_ma_atr"] = (d["close"] - d["slow_ma"]) / d["atr14"].replace(0, np.nan)
    d["ma_band_width"] = (d["ema50"] - d["ema100"]).abs()
    d["ma_band_width_atr"] = d["ma_band_width"] / d["atr14"].replace(0, np.nan)

    d["atr_ratio"] = d["atr14"] / d["atr100"].replace(0, np.nan)

    d["prev_high"] = d["high"].shift(1)
    d["prev_low"] = d["low"].shift(1)
    d["prev_close"] = d["close"].shift(1)

    return d

def build_h1_features(h1: pd.DataFrame) -> pd.DataFrame:
    h = h1.copy()
    h["fast_ma"] = smma(h["close"], FAST_MA)
    h["slow_ma"] = smma(h["close"], SLOW_MA)

    h["rsi7"] = rsi(h["close"], H1_RSI_FAST)
    h["rsi14"] = rsi(h["close"], H1_RSI_STD)

    h["atr14"] = atr(h, ATR_H1)
    h["atr7"] = atr(h, RSI2300_LTATR_PERIOD)
    h["atr_pct_60"] = rolling_percentile(h["atr14"], 60)

    h["close_loc"] = close_location(h)
    h["body_ratio"] = body_ratio(h)
    h["upper_wick_ratio"], h["lower_wick_ratio"] = wick_ratios(h)

    h["is_red"] = (h["close"] < h["open"]).astype(int)
    h["is_green"] = (h["close"] > h["open"]).astype(int)
    h["down_bars_last_3"] = h["is_red"].rolling(3).sum()
    h["down_bars_last_5"] = h["is_red"].rolling(5).sum()

    h["rolling_10_high"] = h["high"].rolling(10).max().shift(1)
    h["rolling_20_high"] = h["high"].rolling(20).max().shift(1)
    h["pullback_depth_10bar"] = h["rolling_10_high"] - h["close"]
    h["pullback_depth_atr"] = h["pullback_depth_10bar"] / h["atr14"].replace(0, np.nan)

    h["fast_minus_slow"] = h["fast_ma"] - h["slow_ma"]
    h["fast_minus_slow_atr"] = h["fast_minus_slow"] / h["atr14"].replace(0, np.nan)
    h["close_minus_fast"] = h["close"] - h["fast_ma"]
    h["close_minus_slow"] = h["close"] - h["slow_ma"]

    h["bars_since_20bar_high"] = bars_since_last_true(h["high"] >= h["rolling_20_high"])
    return h


# =========================
# MERGE DAILY INTO H1
# =========================
def merge_daily_into_h1(h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    d = d1.copy()
    d["day"] = d["time_gmt"].dt.date

    h = h1.copy()
    h["day"] = h["time_gmt"].dt.date

    daily_cols = [
        "day", "close", "ema50", "ema100", "ema200", "slow_ma",
        "rsi14", "atr14", "atr100", "atr_ratio", "atr_pct_120",
        "ema50_slope_5", "ema100_slope_5",
        "trigger_bar_size_atr", "close_pos", "consec_red", "consec_green",
        "price_vs_slow_ma_atr", "ma_band_width_atr",
        "prev_high", "prev_low", "prev_close", "weekday"
    ]

    d = d[daily_cols].rename(columns={
        "close": "d_close",
        "rsi14": "d_rsi14",
        "atr14": "d_atr14",
        "atr100": "d_atr100",
        "atr_ratio": "d_atr_ratio",
        "atr_pct_120": "d_atr_pct_120",
        "trigger_bar_size_atr": "d_trigger_bar_size",
        "close_pos": "d_trigger_close_pos",
        "consec_red": "d_consec_red",
        "consec_green": "d_consec_green",
        "price_vs_slow_ma_atr": "d_price_vs_slow_ma",
        "ma_band_width_atr": "d_ma_band_width",
        "weekday": "d_day_of_week"
    })

    merged = h.merge(d, on="day", how="left")
    merged["close_above_d_50"] = (merged["close"] > merged["ema50"]).astype(int)
    merged["close_above_d_100"] = (merged["close"] > merged["ema100"]).astype(int)
    merged["close_above_d_200"] = (merged["close"] > merged["ema200"]).astype(int)

    merged["dist_prev_day_high"] = merged["close"] - merged["prev_high"]
    merged["dist_prev_day_low"] = merged["close"] - merged["prev_low"]
    merged["dist_prev_day_close"] = merged["close"] - merged["prev_close"]
    merged["h1_atr_vs_d1"] = merged["atr14"] / merged["d_atr14"].replace(0, np.nan)
    return merged


# =========================
# HARD RULE APPROXIMATION
# =========================
def daily_dip_signal(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    if i < 30:
        return False
    if pd.isna(row["d_rsi14"]) or pd.isna(row["atr14"]):
        return False
    if row["d_rsi14"] <= DAILY_DIP_REQUIRE_DAILY_RSI_GT:
        return False
    if row["atr14"] < DAILY_DIP_ATR_THRESHOLD:
        return False
    if row["is_red"] != 1:
        return False
    if row["fast_minus_slow"] <= 0:
        return False
    if row["close_above_d_100"] != 1 and row["close_above_d_200"] != 1:
        return False
    if pd.isna(row["pullback_depth_atr"]) or row["pullback_depth_atr"] < 0.5:
        return False
    return True

def has_qualified_session_candle(day_df: pd.DataFrame) -> bool:
    scan = day_df[
        (day_df["gmt_hour"] >= RSI2300_SESSION_START_GMT) &
        (day_df["gmt_hour"] < RSI2300_SESSION_END_GMT)
    ]
    if scan.empty:
        return False
    cond = (scan["close"] < scan["open"]) & (scan["atr7"] >= RSI2300_LTATR_THRESHOLD)
    return bool(cond.any())

def rsi2300_signal(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    if pd.isna(row["d_rsi14"]):
        return False
    if row["gmt_hour"] != RSI2300_BUY_GMT_HOUR:
        return False
    if row["d_rsi14"] <= RSI2300_DAILY_RSI_BUY_LEVEL:
        return False
    same_day = df[df["gmt_date"] == row["gmt_date"]]
    same_day = same_day[same_day.index < df.index[i]]
    if not has_qualified_session_candle(same_day):
        return False
    if row["close_above_d_100"] != 1 and row["close_above_d_200"] != 1:
        return False
    return True


# =========================
# LABELING
# =========================
@dataclass
class TradeLabel:
    y: int
    entry: float
    stop: float
    target: float
    mae: float
    mfe: float
    exit_return_r: float
    bars_held: int

def label_trade(df: pd.DataFrame, i: int, trigger_price: Optional[float] = None) -> Optional[TradeLabel]:
    if i + 2 >= len(df):
        return None

    setup = df.iloc[i]
    nxt = df.iloc[i + 1]
    entry = nxt["open"]

    atr_now = setup["atr14"]
    if pd.isna(atr_now) or atr_now <= 0:
        return None

    raw_stop_dist = min(1.5 * atr_now, DAILY_DIP_MAX_STOP_POINTS)
    stop = entry - raw_stop_dist
    risk = entry - stop
    if risk <= 0:
        return None

    target = entry + TARGET_R * risk
    future = df.iloc[i + 1:i + 1 + HOLD_BARS].copy()
    if future.empty:
        return None

    mae = ((future["low"] - entry) / risk).min()
    mfe = ((future["high"] - entry) / risk).max()

    hit_stop_idx = None
    hit_target_idx = None

    lows = future["low"].values
    highs = future["high"].values

    for j in range(len(future)):
        if hit_stop_idx is None and lows[j] <= stop:
            hit_stop_idx = j
        if hit_target_idx is None and highs[j] >= target:
            hit_target_idx = j
        if hit_stop_idx is not None or hit_target_idx is not None:
            break

    if hit_target_idx is not None and (hit_stop_idx is None or hit_target_idx <= hit_stop_idx):
        y = 1
        bars_held = hit_target_idx + 1
        exit_r = TARGET_R
    elif hit_stop_idx is not None:
        y = 0
        bars_held = hit_stop_idx + 1
        exit_r = -STOP_R
    else:
        final_close = future.iloc[-1]["close"]
        exit_r = (final_close - entry) / risk
        y = int(exit_r > 0.75 and mfe >= 1.5 and mae > -0.9)
        bars_held = len(future)

    return TradeLabel(
        y=y,
        entry=float(entry),
        stop=float(stop),
        target=float(target),
        mae=float(mae),
        mfe=float(mfe),
        exit_return_r=float(exit_r),
        bars_held=int(bars_held),
    )


# =========================
# EVENT DATASET
# =========================
def build_event_dataset(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for i in range(len(df) - HOLD_BARS - 2):
        setup_type = None
        trigger_price = None

        if daily_dip_signal(df, i):
            setup_type = "daily_dip"
            trigger_price = df.iloc[i]["close"]
        elif rsi2300_signal(df, i):
            setup_type = "rsi_2300"
            trigger_price = df.iloc[i]["close"]

        if setup_type is None:
            continue

        lab = label_trade(df, i, trigger_price=trigger_price)
        if lab is None:
            continue

        row = df.iloc[i]
        h1_above_trigger = (row["close"] - trigger_price) / row["atr14"] if pd.notna(row["atr14"]) and row["atr14"] > 0 else np.nan
        d_atr14 = row["d_atr14"]

        rows.append({
            "time": row["time"],
            "setup_type": setup_type,

            "d1_rsi_14": row["d_rsi14"],
            "d1_atr_ratio": row["d_atr_ratio"],
            "d1_trigger_bar_size": row["d_trigger_bar_size"],
            "d1_trigger_close_pos": row["d_trigger_close_pos"],
            "d1_consec_red": row["d_consec_red"],
            "d1_consec_green": row["d_consec_green"],
            "d1_price_vs_slow_ma": row["d_price_vs_slow_ma"],
            "d1_ma_band_width": row["d_ma_band_width"],
            "d1_day_of_week": row["d_day_of_week"],

            "h1_body_ratio": row["body_ratio"],
            "h1_atr_vs_d1": row["h1_atr_vs_d1"],
            "h1_hour_gmt": row["gmt_hour"],
            "h1_above_trigger": h1_above_trigger,

            "h1_rsi_7": row["rsi7"],
            "h1_rsi_14": row["rsi14"],
            "h1_atr_14": row["atr14"],
            "h1_atr_pct_60": row["atr_pct_60"],
            "h1_close_location": row["close_loc"],
            "h1_upper_wick_ratio": row["upper_wick_ratio"],
            "h1_lower_wick_ratio": row["lower_wick_ratio"],
            "h1_down_bars_last_3": row["down_bars_last_3"],
            "h1_down_bars_last_5": row["down_bars_last_5"],
            "h1_pullback_depth_atr": row["pullback_depth_atr"],
            "h1_fast_minus_slow_atr": row["fast_minus_slow_atr"],
            "d1_ema50_slope_5": row["ema50_slope_5"],
            "d1_ema100_slope_5": row["ema100_slope_5"],
            "close_above_d_50": row["close_above_d_50"],
            "close_above_d_100": row["close_above_d_100"],
            "close_above_d_200": row["close_above_d_200"],
            "dist_prev_day_high": row["dist_prev_day_high"] / d_atr14 if pd.notna(d_atr14) and d_atr14 > 0 else np.nan,
            "dist_prev_day_low": row["dist_prev_day_low"] / d_atr14 if pd.notna(d_atr14) and d_atr14 > 0 else np.nan,
            "dist_prev_day_close": row["dist_prev_day_close"] / d_atr14 if pd.notna(d_atr14) and d_atr14 > 0 else np.nan,
            "bars_since_20bar_high": row["bars_since_20bar_high"],

            "label": lab.y,
            "entry": lab.entry,
            "stop": lab.stop,
            "target": lab.target,
            "mae_r": lab.mae,
            "mfe_r": lab.mfe,
            "exit_r": lab.exit_return_r,
            "bars_held": lab.bars_held,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("time").reset_index(drop=True)


# =========================
# MODEL FEATURES
# =========================
BASE_FEATURES = [
    "d1_rsi_14",
    "d1_atr_ratio",
    "d1_trigger_bar_size",
    "d1_trigger_close_pos",
    "d1_consec_red",
    "d1_consec_green",
    "d1_price_vs_slow_ma",
    "d1_ma_band_width",
    "d1_day_of_week",
    "h1_body_ratio",
    "h1_atr_vs_d1",
    "h1_hour_gmt",
    "h1_above_trigger",
    "h1_rsi_7",
    "h1_rsi_14",
    "h1_atr_14",
    "h1_atr_pct_60",
    "h1_close_location",
    "h1_upper_wick_ratio",
    "h1_lower_wick_ratio",
    "h1_down_bars_last_3",
    "h1_down_bars_last_5",
    "h1_pullback_depth_atr",
    "h1_fast_minus_slow_atr",
    "d1_ema50_slope_5",
    "d1_ema100_slope_5",
    "close_above_d_50",
    "close_above_d_100",
    "close_above_d_200",
    "dist_prev_day_high",
    "dist_prev_day_low",
    "dist_prev_day_close",
    "bars_since_20bar_high",
]

COMBINED_FEATURES = ["setup_type_code"] + BASE_FEATURES


# =========================
# MODEL HELPERS
# =========================
def fit_rf(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=7,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def compute_safe_cv_params(n_samples: int, desired_splits: int, desired_gap: int, min_test_size: int = 10):
    if n_samples < (2 * min_test_size + 10):
        return None, None, None

    best = None

    for splits in range(desired_splits, 1, -1):
        max_gap = min(desired_gap, max(0, n_samples // 4))

        for gap in range(max_gap, -1, -1):
            max_test_size = (n_samples - gap) // (splits + 1)
            if max_test_size < min_test_size:
                continue

            test_size = max_test_size

            # sklearn feasibility condition:
            # n_samples - gap - (test_size * n_splits) > 0
            if n_samples - gap - (test_size * splits) <= 0:
                continue

            best = (splits, gap, test_size)
            break

        if best is not None:
            break

    return best if best is not None else (None, None, None)

def evaluate_one_model(events: pd.DataFrame, model_name: str, feature_cols: List[str]):
    data = events.dropna(subset=feature_cols + ["label"]).copy()
    if data.empty:
        raise RuntimeError(f"{model_name}: no usable rows after NaN drop.")

    X = data[feature_cols].copy()
    y = data["label"].astype(int)
    n_samples = len(data)

    eff_splits, eff_gap, eff_test_size = compute_safe_cv_params(
        n_samples=n_samples,
        desired_splits=N_SPLITS,
        desired_gap=CV_GAP,
        min_test_size=MIN_TEST_SIZE
    )

    if eff_splits is None or eff_splits < 2 or eff_test_size is None:
        raise RuntimeError(
            f"{model_name}: not enough samples ({n_samples}) for time-series CV with desired gap={CV_GAP}."
        )

    if n_samples < 150:
        print(f"[WARN] {model_name}: only {n_samples} samples; results may be unstable.")

    print(
        f"[INFO] {model_name}: using n_samples={n_samples}, "
        f"n_splits={eff_splits}, gap={eff_gap}, test_size={eff_test_size}"
    )

    tscv = TimeSeriesSplit(
        n_splits=eff_splits,
        gap=eff_gap,
        test_size=eff_test_size
    )

    fold_rows = []
    perm_rows = []

    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[tr], X.iloc[te]
        y_train, y_test = y.iloc[tr], y.iloc[te]
        test_slice = data.iloc[te].copy()

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"[WARN] {model_name} fold {fold}: skipped due to single-class train/test split.")
            continue

        model = fit_rf(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)

        base_count = len(test_slice)
        base_expectancy = test_slice["exit_r"].mean()
        base_winrate = (test_slice["exit_r"] > 0).mean()

        for th in THRESHOLDS:
            preds = (probs >= th).astype(int)
            picked = test_slice.loc[preds == 1].copy()

            if len(picked) == 0:
                fold_rows.append({
                    "model": model_name,
                    "fold": fold,
                    "threshold": th,
                    "auc": auc,
                    "base_count": base_count,
                    "base_expectancy_r": base_expectancy,
                    "base_winrate": base_winrate,
                    "rf_count": 0,
                    "rf_expectancy_r": np.nan,
                    "rf_winrate": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                })
                continue

            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)

            fold_rows.append({
                "model": model_name,
                "fold": fold,
                "threshold": th,
                "auc": auc,
                "base_count": base_count,
                "base_expectancy_r": base_expectancy,
                "base_winrate": base_winrate,
                "rf_count": len(picked),
                "rf_expectancy_r": picked["exit_r"].mean(),
                "rf_winrate": (picked["exit_r"] > 0).mean(),
                "precision": precision,
                "recall": recall,
            })

        try:
            perm = permutation_importance(
                model,
                X_test,
                y_test,
                scoring="roc_auc",
                n_repeats=8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            perm_df = pd.DataFrame({
                "model": model_name,
                "fold": fold,
                "feature": feature_cols,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std
            })
            perm_rows.append(perm_df)
        except Exception as e:
            print(f"[WARN] permutation importance failed for {model_name} fold {fold}: {e}")

    if not fold_rows:
        raise RuntimeError(f"{model_name}: no valid CV folds survived after class-balance checks.")

    results = pd.DataFrame(fold_rows)
    perm_imp = pd.concat(perm_rows, ignore_index=True) if perm_rows else pd.DataFrame()
    return results, perm_imp


# =========================
# MAIN
# =========================
def main():
    if mt5 is None:
        raise RuntimeError("No MT5 backend import succeeded. Install MetaTrader5, mt5linux, or pymt5linux.")

    init_mt5()
    try:
        TF_H1 = mt5.TIMEFRAME_H1
        TF_D1 = mt5.TIMEFRAME_D1

        print(f"[INFO] Loading {SYMBOL} with H1={BARS_H1}, D1={BARS_D1}")
        h1 = load_rates(SYMBOL, TF_H1, BARS_H1)
        d1 = load_rates(SYMBOL, TF_D1, BARS_D1)

        h1 = add_gmt_columns(h1, SERVER_TO_GMT_HOURS)
        d1 = add_gmt_columns(d1, SERVER_TO_GMT_HOURS)

        h1 = build_h1_features(h1)
        d1 = build_daily_features(d1)
        df = merge_daily_into_h1(h1, d1)

        events = build_event_dataset(df)
        if events.empty:
            raise RuntimeError("No event rows built. Relax rules or load more history.")

        events["setup_type_code"] = events["setup_type"].map({"daily_dip": 1, "rsi_2300": 2}).astype(int)

        print(f"[INFO] Total event rows: {len(events)}")
        print(events["setup_type"].value_counts(dropna=False).to_string())

        all_results = []
        all_perm = []

        dd = events[events["setup_type"] == "daily_dip"].copy()
        if len(dd) >= MIN_ROWS_DAILY_DIP:
            try:
                res_dd, perm_dd = evaluate_one_model(dd, "daily_dip", BASE_FEATURES)
                all_results.append(res_dd)
                if not perm_dd.empty:
                    all_perm.append(perm_dd)
            except Exception as e:
                print(f"[WARN] Skipping daily_dip model: {e}")
        else:
            print(f"[WARN] Skipping daily_dip model: only {len(dd)} rows, need >= {MIN_ROWS_DAILY_DIP}")

        r23 = events[events["setup_type"] == "rsi_2300"].copy()
        if len(r23) >= MIN_ROWS_RSI2300:
            try:
                res_r23, perm_r23 = evaluate_one_model(r23, "rsi_2300", BASE_FEATURES)
                all_results.append(res_r23)
                if not perm_r23.empty:
                    all_perm.append(perm_r23)
            except Exception as e:
                print(f"[WARN] Skipping rsi_2300 model: {e}")
        else:
            print(f"[WARN] Skipping rsi_2300 model: only {len(r23)} rows, need >= {MIN_ROWS_RSI2300}")

        comb = events.copy()
        if len(comb) >= MIN_ROWS_COMBINED:
            try:
                res_comb, perm_comb = evaluate_one_model(comb, "combined", COMBINED_FEATURES)
                all_results.append(res_comb)
                if not perm_comb.empty:
                    all_perm.append(perm_comb)
            except Exception as e:
                print(f"[WARN] Skipping combined model: {e}")
        else:
            print(f"[WARN] Skipping combined model: only {len(comb)} rows, need >= {MIN_ROWS_COMBINED}")

        if not all_results:
            raise RuntimeError("No model produced valid results.")

        results = pd.concat(all_results, ignore_index=True)

        print("\n=== AVERAGE RESULTS BY MODEL / THRESHOLD ===")
        summary = (
            results.groupby(["model", "threshold"], as_index=False)[
                ["auc", "base_count", "base_expectancy_r", "base_winrate",
                 "rf_count", "rf_expectancy_r", "rf_winrate", "precision", "recall"]
            ]
            .mean()
            .round(4)
        )
        print(summary.to_string(index=False))

        if all_perm:
            perm_all = pd.concat(all_perm, ignore_index=True)
            perm_summary = (
                perm_all.groupby(["model", "feature"], as_index=False)["importance_mean"]
                .mean()
                .sort_values(["model", "importance_mean"], ascending=[True, False])
            )

            print("\n=== TOP PERMUTATION IMPORTANCE ===")
            for model_name in perm_summary["model"].unique():
                print(f"\n--- {model_name} ---")
                print(
                    perm_summary[perm_summary["model"] == model_name]
                    .head(15)
                    .to_string(index=False)
                )
        else:
            perm_all = pd.DataFrame()
            perm_summary = pd.DataFrame()

        events.to_csv("rf_events_dailydip_rsi2300.csv", index=False)
        results.to_csv("rf_results_dailydip_rsi2300.csv", index=False)
        if not perm_all.empty:
            perm_all.to_csv("rf_permutation_importance_dailydip_rsi2300.csv", index=False)
        if not perm_summary.empty:
            perm_summary.to_csv("rf_permutation_importance_summary_dailydip_rsi2300.csv", index=False)

        print("\nSaved:")
        print(" - rf_events_dailydip_rsi2300.csv")
        print(" - rf_results_dailydip_rsi2300.csv")
        if not perm_all.empty:
            print(" - rf_permutation_importance_dailydip_rsi2300.csv")
            print(" - rf_permutation_importance_summary_dailydip_rsi2300.csv")

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
