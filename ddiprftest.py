import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- CONFIG ---------------- #
BASE_DIR = Path("daily_dip_rf")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

H1_FILE = DATA_DIR / "h1.csv"
D1_FILE = DATA_DIR / "d1.csv"

SIGNAL_TF = "H1"
FAST_MA_PERIOD = 26
SLOW_MA_PERIOD = 150
ATR_PERIOD = 14
DAILY_RSI_PERIOD = 14

DAILY_DIP_ATR_THRESHOLD = 100.0
DAILY_DIP_STOP_BUFFER_POINTS = 50
DAILY_DIP_MAX_STOP_POINTS = 200
DAILY_DIP_LOOKBACK_DAYS = 3
DAILY_DIP_REENTRY_WAIT_DAILY_BARS = 2
DAILY_DIP_REQUIRE_GREEN_CONFIRM = True
DAILY_DIP_REQUIRE_PRICE_ABOVE_TRIGGER_ON_CONFIRM = True
DAILY_DIP_REQUIRE_OUTSIDE_MA_BAND = True
ENABLE_DAILY_DIP_DAILY_RSI_GATE = True
DAILY_DIP_MIN_DAILY_RSI = 50.0

USE_ATR_TAKE_PROFIT = True
TAKE_PROFIT_ATR_MULTIPLIER = 3.0

POINT_VALUE = 1.0
MAX_HOLD_H1_BARS = 72

GOOD_MFE_R = 1.0
MAX_MAE_R = 0.7
MIN_FINAL_R = 0.0
# ---------------------------------------- #


def load_ohlc(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df


def rma(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return rma(tr, period)


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = rma(up, period)
    avg_down = rma(down, period)
    rs = avg_up / avg_down.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def add_indicators_h1(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["fast_ma"] = rma(d["close"], FAST_MA_PERIOD)
    d["slow_ma"] = rma(d["close"], SLOW_MA_PERIOD)
    d["atr"] = atr(d, ATR_PERIOD)

    d["ret_1"] = d["close"].pct_change(1)
    d["ret_3"] = d["close"].pct_change(3)
    d["ret_6"] = d["close"].pct_change(6)
    d["ret_24"] = d["close"].pct_change(24)

    d["body"] = d["close"] - d["open"]
    d["range"] = d["high"] - d["low"]
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]

    d["hour"] = d["time"].dt.hour
    d["dayofweek"] = d["time"].dt.dayofweek
    return d


def add_indicators_d1(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["daily_atr"] = atr(d, ATR_PERIOD)
    d["daily_rsi"] = rsi(d["close"], DAILY_RSI_PERIOD)
    d["daily_fast_ma"] = rma(d["close"], FAST_MA_PERIOD)
    d["daily_slow_ma"] = rma(d["close"], SLOW_MA_PERIOD)

    d["body"] = d["close"] - d["open"]
    d["range"] = d["high"] - d["low"]
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]

    d["rsi_chg_1"] = d["daily_rsi"].diff(1)
    d["rsi_chg_3"] = d["daily_rsi"].diff(3)
    d["ret_1d"] = d["close"].pct_change(1)
    d["ret_3d"] = d["close"].pct_change(3)
    return d


def merge_daily_context(h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    d1_ctx = d1.copy()
    d1_ctx = d1_ctx.rename(columns={"time": "daily_time"})
    d1_ctx["merge_time"] = d1_ctx["daily_time"] + pd.Timedelta(days=1)
    out = pd.merge_asof(
        h1.sort_values("time"),
        d1_ctx.sort_values("merge_time"),
        left_on="time",
        right_on="merge_time",
        direction="backward"
    )
    return out


def price_band_side(price: float, band_low: float, band_high: float) -> int:
    if price < band_low:
        return -1
    if price > band_high:
        return 1
    return 0


def simulate_trade(h1: pd.DataFrame, start_idx: int, entry: float, stop: float, tp: float | None):
    risk = entry - stop
    if risk <= 0:
        return None

    mae = 0.0
    mfe = 0.0
    exit_price = h1.iloc[min(start_idx + MAX_HOLD_H1_BARS, len(h1) - 1)]["close"]
    exit_reason = "timeout"
    bars_held = 0
    hit_stop = 0
    hit_tp = 0

    end_idx = min(start_idx + MAX_HOLD_H1_BARS, len(h1) - 1)

    for j in range(start_idx + 1, end_idx + 1):
        row = h1.iloc[j]
        favorable = row["high"] - entry
        adverse = entry - row["low"]

        mfe = max(mfe, favorable)
        mae = max(mae, adverse)
        bars_held = j - start_idx

        stop_hit = row["low"] <= stop
        tp_hit = (tp is not None) and (row["high"] >= tp)

        if stop_hit and tp_hit:
            exit_price = stop
            exit_reason = "both_hit_same_bar_assume_stop"
            hit_stop = 1
            break
        elif stop_hit:
            exit_price = stop
            exit_reason = "stop"
            hit_stop = 1
            break
        elif tp_hit:
            exit_price = tp
            exit_reason = "tp"
            hit_tp = 1
            break

    final_r = (exit_price - entry) / risk
    mae_r = mae / risk
    mfe_r = mfe / risk

    return {
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "mae_r": mae_r,
        "mfe_r": mfe_r,
        "final_r": final_r,
        "hit_stop": hit_stop,
        "hit_tp": hit_tp,
    }


def build_candidates(h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g_last_daily_dip_exit_time = pd.NaT
    g_daily_dip_anchor_time = pd.NaT
    g_daily_dip_setup_locked = False
    g_daily_dip_signal_armed = False
    g_daily_dip_signal_bar_time = pd.NaT
    g_daily_dip_trigger_price = np.nan
    g_daily_dip_raw_stop_ref = np.nan
    g_daily_dip_band_side = 0

    d1_indexed = d1.set_index("daily_time").copy()

    for i in range(max(SLOW_MA_PERIOD, ATR_PERIOD) + 5, len(h1)):
        row = h1.iloc[i]
        now = row["time"]

        if pd.notna(g_last_daily_dip_exit_time):
            days_since_exit = (now.floor("D") - g_last_daily_dip_exit_time.floor("D")).days
            if days_since_exit < DAILY_DIP_REENTRY_WAIT_DAILY_BARS:
                continue

        if ENABLE_DAILY_DIP_DAILY_RSI_GATE and row["daily_rsi"] <= DAILY_DIP_MIN_DAILY_RSI:
            continue

        valid_daily = d1[d1["daily_time"] < now.floor("D") + pd.Timedelta(days=1)].copy()
        if len(valid_daily) < DAILY_DIP_LOOKBACK_DAYS + 2:
            continue

        found = None
        for lookback in range(1, DAILY_DIP_LOOKBACK_DAYS + 1):
            dbar = valid_daily.iloc[-1 - lookback + 1]
            if dbar["open"] > dbar["close"] and abs(dbar["open"] - dbar["close"]) > 0:
                found = dbar
                found_index = lookback
                break

        if found is None:
            continue

        setup_time = found["daily_time"]

        if pd.isna(g_daily_dip_anchor_time) or setup_time != g_daily_dip_anchor_time:
            g_daily_dip_anchor_time = setup_time
            g_daily_dip_setup_locked = False
            g_daily_dip_signal_armed = False
            g_daily_dip_signal_bar_time = pd.NaT
            g_daily_dip_trigger_price = np.nan
            g_daily_dip_raw_stop_ref = np.nan
            g_daily_dip_band_side = 0

        if g_daily_dip_setup_locked:
            continue

        prev_bar = h1.iloc[i - 1]
        lt_open1 = prev_bar["open"]
        lt_close1 = prev_bar["close"]
        lt_bar1_time = prev_bar["time"]

        lower_tf_bearish = lt_open1 > lt_close1
        lower_tf_green = lt_close1 > lt_open1
        lower_tf_atr_high = prev_bar["atr"] >= DAILY_DIP_ATR_THRESHOLD

        trigger_price = found["close"]
        current_price = row["close"]

        band_low = min(prev_bar["fast_ma"], prev_bar["slow_ma"])
        band_high = max(prev_bar["fast_ma"], prev_bar["slow_ma"])
        band_side = price_band_side(current_price, band_low, band_high)
        outside_band = band_side != 0

        if not g_daily_dip_signal_armed:
            if lower_tf_atr_high and lower_tf_bearish and current_price > trigger_price:
                if DAILY_DIP_REQUIRE_OUTSIDE_MA_BAND and not outside_band:
                    continue

                g_daily_dip_signal_armed = True
                g_daily_dip_signal_bar_time = lt_bar1_time
                g_daily_dip_trigger_price = trigger_price
                g_daily_dip_raw_stop_ref = trigger_price - DAILY_DIP_STOP_BUFFER_POINTS * POINT_VALUE
                g_daily_dip_band_side = band_side
            continue

        cur_band_low = min(prev_bar["fast_ma"], prev_bar["slow_ma"])
        cur_band_high = max(prev_bar["fast_ma"], prev_bar["slow_ma"])
        current_band_side = price_band_side(current_price, cur_band_low, cur_band_high)
        current_outside_band = current_band_side != 0

        if DAILY_DIP_REQUIRE_OUTSIDE_MA_BAND:
            if not current_outside_band:
                continue
            if g_daily_dip_band_side != 0 and current_band_side != g_daily_dip_band_side:
                continue

        if DAILY_DIP_REQUIRE_GREEN_CONFIRM:
            if lt_bar1_time <= g_daily_dip_signal_bar_time:
                continue
            if not lower_tf_green:
                continue

        if DAILY_DIP_REQUIRE_PRICE_ABOVE_TRIGGER_ON_CONFIRM and current_price <= g_daily_dip_trigger_price:
            continue

        entry = current_price
        stop = g_daily_dip_raw_stop_ref if pd.notna(g_daily_dip_raw_stop_ref) else (trigger_price - DAILY_DIP_STOP_BUFFER_POINTS * POINT_VALUE)
        risk_points = entry - stop

        if risk_points <= 0:
            continue

        if DAILY_DIP_MAX_STOP_POINTS > 0 and risk_points > DAILY_DIP_MAX_STOP_POINTS * POINT_VALUE:
            stop = entry - DAILY_DIP_MAX_STOP_POINTS * POINT_VALUE
            risk_points = entry - stop

        tp = None
        if USE_ATR_TAKE_PROFIT and pd.notna(prev_bar["atr"]):
            tp = entry + TAKE_PROFIT_ATR_MULTIPLIER * prev_bar["atr"]

        sim = simulate_trade(h1, i, entry, stop, tp)
        if sim is None:
            continue

        target_good_trade = int(
            (sim["mfe_r"] >= GOOD_MFE_R) and
            (sim["mae_r"] <= MAX_MAE_R) and
            (sim["final_r"] >= MIN_FINAL_R)
        )

        rows.append({
            "time": now,
            "entry_time": now,
            "entry_price": entry,
            "stop_price": stop,
            "tp_price": np.nan if tp is None else tp,
            "risk_points": risk_points,
            "risk_atr": risk_points / prev_bar["atr"] if prev_bar["atr"] and prev_bar["atr"] > 0 else np.nan,

            "anchor_time": found["daily_time"],
            "anchor_lookback_index": found_index,
            "anchor_open": found["open"],
            "anchor_high": found["high"],
            "anchor_low": found["low"],
            "anchor_close": found["close"],
            "anchor_body_abs": abs(found["close"] - found["open"]),
            "anchor_range": found["high"] - found["low"],
            "anchor_upper_wick": found["upper_wick"],
            "anchor_lower_wick": found["lower_wick"],

            "daily_rsi": row["daily_rsi"],
            "daily_rsi_chg_1": row["rsi_chg_1"],
            "daily_rsi_chg_3": row["rsi_chg_3"],
            "daily_atr": row["daily_atr"],
            "daily_fast_ma": row["daily_fast_ma"],
            "daily_slow_ma": row["daily_slow_ma"],
            "daily_ret_1d": row["ret_1d"],
            "daily_ret_3d": row["ret_3d"],

            "arm_bar_time": g_daily_dip_signal_bar_time,
            "arm_open": prev_bar["open"],
            "arm_high": prev_bar["high"],
            "arm_low": prev_bar["low"],
            "arm_close": prev_bar["close"],
            "arm_body": prev_bar["body"],
            "arm_range": prev_bar["range"],
            "arm_upper_wick": prev_bar["upper_wick"],
            "arm_lower_wick": prev_bar["lower_wick"],
            "arm_atr": prev_bar["atr"],

            "confirm_open": prev_bar["open"],
            "confirm_high": prev_bar["high"],
            "confirm_low": prev_bar["low"],
            "confirm_close": prev_bar["close"],
            "confirm_body": prev_bar["body"],
            "confirm_range": prev_bar["range"],

            "price_minus_trigger": entry - g_daily_dip_trigger_price,
            "band_side_armed": g_daily_dip_band_side,
            "band_side_confirm": current_band_side,
            "outside_band_confirm": int(current_outside_band),

            "ret_1": row["ret_1"],
            "ret_3": row["ret_3"],
            "ret_6": row["ret_6"],
            "ret_24": row["ret_24"],
            "hour": row["hour"],
            "dayofweek": row["dayofweek"],

            "mae_r": sim["mae_r"],
            "mfe_r": sim["mfe_r"],
            "final_r": sim["final_r"],
            "bars_held": sim["bars_held"],
            "hit_stop": sim["hit_stop"],
            "hit_tp": sim["hit_tp"],
            "exit_reason": sim["exit_reason"],

            "target_good_trade": target_good_trade
        })

        g_daily_dip_setup_locked = True
        g_daily_dip_signal_armed = False
        g_daily_dip_signal_bar_time = pd.NaT
        g_last_daily_dip_exit_time = now

    return pd.DataFrame(rows)


def main():
    h1 = load_ohlc(H1_FILE)
    d1 = load_ohlc(D1_FILE)

    h1 = add_indicators_h1(h1)
    d1 = add_indicators_d1(d1)

    d1 = d1.rename(columns={"time": "daily_time"})
    merged = merge_daily_context(h1, d1)

    candidates = build_candidates(merged, d1)

    out_file = OUTPUT_DIR / "daily_dip_candidates.csv"
    candidates.to_csv(out_file, index=False)

    print(f"Saved {len(candidates)} DailyDip candidates to: {out_file}")
    if len(candidates):
        print(candidates[[
            "time", "entry_price", "stop_price", "daily_rsi",
            "mae_r", "mfe_r", "final_r", "target_good_trade"
        ]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
