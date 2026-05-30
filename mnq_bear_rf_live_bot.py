import os
import sys
import time
import json
import traceback
import tempfile
from datetime import datetime, timezone

import joblib
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

try:
    import portalocker
except ImportError:
    portalocker = None


def env_str(name, default=None):
    val = os.getenv(name)
    return default if val is None or val == "" else val


def env_int(name, default):
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return int(val)


def env_float(name, default):
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return float(val)


def env_bool(name, default=False):
    val = os.getenv(name)
    if val is None or val == "":
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = env_str("LOG_DIR", os.path.join(BASE_DIR, "logs"))
MODEL_DIR = env_str("MODEL_DIR", os.path.join(BASE_DIR, "models"))
STATE_DIR = env_str("STATE_DIR", os.path.join(BASE_DIR, "state"))
LOCK_DIR = env_str("LOCK_DIR", os.path.join(BASE_DIR, "lock"))
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(LOCK_DIR, exist_ok=True)

CONFIG = {
    "mt5_terminal_path": env_str("MT5_TERMINAL_PATH", ""),
    "mt5_symbol": env_str("LIVE_MT5_SYMBOL", ""),
    "symbol_prefix": env_str("SYMBOL_PREFIX", "MNQ"),
    "enable_session_filter": env_bool("ENABLE_SESSION_FILTER", False),
    "session_start_hour_utc": env_int("SESSION_START_HOUR_UTC", 13),
    "session_end_hour_utc": env_int("SESSION_END_HOUR_UTC", 20),
    "poll_seconds": env_int("POLL_SECONDS", 5),
    "magic": env_int("STRATEGY_MAGIC", 20260522),
    "volume": env_float("TRADE_VOLUME", 1.0),
    "rf_model_file": env_str("RF_MODEL_FILE", os.path.join(MODEL_DIR, "mnq_rf_model.joblib")),
    "rf_features_file": env_str("RF_FEATURES_FILE", os.path.join(MODEL_DIR, "mnq_rf_feature_columns.json")),
    "decision_log_csv": env_str("DECISION_LOG_CSV", os.path.join(LOG_DIR, "mnq_bear_rf_decisions.csv")),
    "trade_log_csv": env_str("TRADE_LOG_CSV", os.path.join(LOG_DIR, "mnq_bear_rf_trades.csv")),
    "state_file": env_str("STATE_FILE", os.path.join(STATE_DIR, "mnq_bear_rf_state.json")),
    "lock_file": env_str("LOCK_FILE", os.path.join(LOCK_DIR, "mnq_bear_rf_live_bot.lock")),
    "d1_bars": env_int("D1_BARS", 1000),
    "h1_bars": env_int("H1_BARS", 4000),
    "ema_period": env_int("EMA_PERIOD", 100),
    "slope_lookback": env_int("SLOPE_LOOKBACK", 10),
    "atr_period": env_int("ATR_PERIOD", 14),
    "rsi_period": env_int("RSI_PERIOD", 14),
    "rsi_tighten_period": env_int("RSI_TIGHTEN_PERIOD", 10),
    "breakout_lookback": env_int("BREAKOUT_LOOKBACK", 5),
    "regime_rsi_max": env_float("REGIME_RSI_MAX", 60),
    "atr_expansion_min": env_float("ATR_EXPANSION_MIN", 1.2),
    "rf_threshold": env_float("RF_THRESHOLD", 0.50),
    "fixed_stop_points": env_float("FIXED_STOP_POINTS", 100.0),
    "fixed_target_points": env_float("FIXED_TARGET_POINTS", 60.0),
    "max_loss_usd": env_float("MAX_LOSS_USD", 400.0),
    "mnq_point_value": env_float("MNQ_POINT_VALUE", 2.0),
    "max_hold_bars": env_int("MAX_HOLD_BARS", 48),
    "tighten_mode": env_str("TIGHTEN_MODE", "either"),
    "tighten_rsi_threshold": env_float("TIGHTEN_RSI_THRESHOLD", 35),
    "tighten_atr_stretch_threshold": env_float("TIGHTEN_ATR_STRETCH_THRESHOLD", 1.10),
    "tighten_factor": env_float("TIGHTEN_FACTOR", 0.50),
    "tighten_buffer_points": env_float("TIGHTEN_BUFFER_POINTS", 8.0),
    "deviation": env_int("DEVIATION", 20),
    "order_comment": env_str("ORDER_COMMENT", "mnq_bear_rf"),
    "allow_only_one_position_in_terminal": env_bool("ALLOW_ONLY_ONE_POSITION_IN_TERMINAL", True),
    "allow_only_one_position_per_symbol": env_bool("ALLOW_ONLY_ONE_POSITION_PER_SYMBOL", True),
    "reconnect_sleep_seconds": env_int("RECONNECT_SLEEP_SECONDS", 10),
    "max_open_shorts": env_int("MAX_OPEN_SHORTS", 1),
    "max_trades_per_day": env_int("MAX_TRADES_PER_DAY", 3),
    "min_bars_between_entries": env_int("MIN_BARS_BETWEEN_ENTRIES", 4),
    "stale_feed_minutes": env_int("STALE_FEED_MINUTES", 120),
    "feed_refresh_retries": env_int("FEED_REFRESH_RETRIES", 5),
    "feed_refresh_sleep_seconds": env_int("FEED_REFRESH_SLEEP_SECONDS", 2),
    "feed_refresh_reinit_on_stale": env_bool("FEED_REFRESH_REINIT_ON_STALE", True),
    "min_history_completion_ratio": env_float("MIN_HISTORY_COMPLETION_RATIO", 0.80),
    "strict_history_check": env_bool("STRICT_HISTORY_CHECK", True),
}

FEATURE_COLS = [
    "regime_age", "d1_rsi14", "d1_rsi10", "d1_range_atr_ratio",
    "d1_close_vs_ema_atr", "d1_ema_slope_atr", "d1_ret_5",
    "h1_atr14", "h1_rsi14", "h1_ret_1h", "h1_ret_6h", "h1_ret_24h",
    "bar_range", "body_size", "atr_ratio", "tickvol_ratio",
    "dist_to_recent_low_atr", "dist_to_recent_high_atr", "dist_to_h1_ema_atr",
    "recent_6h_mean_ret", "recent_6h_std_ret", "recent_24h_mean_ret", "recent_24h_std_ret",
    "recent_24h_low_break_distance", "recent_24h_high_distance",
    "hour", "dayofweek", "is_us_session",
    "trigger_atr_breakdown", "trigger_rsi_rollover",
    "prev_bar_red", "prev_bar_range", "prev_bar_rsi14", "prev_bar_atr_ratio",
]


def utc_now():
    return datetime.now(timezone.utc)


def utc_now_str():
    return utc_now().strftime("%Y-%m-%d %H:%M:%S")


def iso_ts(ts):
    if ts is None:
        return None
    return pd.Timestamp(ts).isoformat()


def log(msg):
    print(f"[{utc_now_str()} UTC] {msg}", flush=True)


def append_csv(path, row):
    df = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=write_header)


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class StateStore:
    def __init__(self, path):
        self.path = path
        self.state = self._load()
        self.dirty = False

    def _default_state(self):
        return {
            "last_processed_h1_bar": None,
            "open_entry_bar_time": None,
            "last_forced_exit_bar_time": None,
            "last_entry_bar_time": None,
            "trades_today_date": None,
            "trades_today_count": 0,
            "active_symbol": None,
        }

    def _load(self):
        state = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    state = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                log(f"WARNING: State file unreadable or corrupt, starting fresh: {e}")
                state = {}

        defaults = self._default_state()
        defaults.update(state)
        return defaults

    def mark_dirty(self):
        self.dirty = True

    def save_if_dirty(self):
        if not self.dirty:
            return

        directory = os.path.dirname(self.path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="state_", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.path)
            self.dirty = False
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


def acquire_process_lock():
    if portalocker is None:
        raise ImportError("portalocker is required for single-instance safety. Install with: pip install portalocker")

    fh = open(CONFIG["lock_file"], "a+")
    try:
        portalocker.lock(fh, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
    except portalocker.exceptions.LockException:
        fh.close()
        raise RuntimeError(f"Another bot instance is already running: {CONFIG['lock_file']}")
    fh.seek(0)
    fh.truncate()
    fh.write(f"pid={os.getpid()} started_utc={utc_now_str()}\n")
    fh.flush()
    return fh


def ensure_utc_timestamp(ts):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize("UTC")
    return t.tz_convert("UTC")


def initialize_mt5():
    kwargs = {}
    if CONFIG["mt5_terminal_path"]:
        kwargs["path"] = CONFIG["mt5_terminal_path"]

    ok = mt5.initialize(**kwargs)
    if not ok:
        raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")

    acct = mt5.account_info()
    if acct is None:
        raise RuntimeError(f"mt5.account_info failed after initialize: {mt5.last_error()}")

    log(f"Connected account={acct.login} server={acct.server}")


def ensure_mt5_connection():
    term = mt5.terminal_info()
    acct = mt5.account_info()
    if term is not None and acct is not None:
        return False

    try:
        mt5.shutdown()
    except Exception:
        pass

    initialize_mt5()
    return True


def ensure_symbol_selected(symbol):
    if not mt5.symbol_select(symbol, True):
        raise RuntimeError(f"symbol_select failed for {symbol}")


def get_symbol_info_or_raise(symbol):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info returned None for {symbol}")
    return info


def get_tick(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick for {symbol}")
    return tick


def get_fresh_tick(symbol, retries=3, sleep_seconds=1):
    last_exc = None
    for _ in range(max(1, retries)):
        try:
            ensure_symbol_selected(symbol)
            tick = get_tick(symbol)
            if getattr(tick, "bid", None) is not None and getattr(tick, "ask", None) is not None:
                return tick
        except Exception as exc:
            last_exc = exc
        time.sleep(max(0, sleep_seconds))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Unable to get fresh tick for {symbol}")


def trigger_symbol_refresh(symbol):
    ensure_symbol_selected(symbol)
    return get_fresh_tick(symbol, retries=2, sleep_seconds=1)


def validate_timeframe_df(df, symbol, timeframe, requested_bars, label):
    if df is None or df.empty:
        raise RuntimeError(f"{label}: empty dataframe for {symbol} timeframe={timeframe}")

    if "time" not in df.columns:
        raise RuntimeError(f"{label}: missing 'time' column for {symbol} timeframe={timeframe}")

    if df["time"].isna().any():
        raise RuntimeError(f"{label}: NaT timestamps found for {symbol} timeframe={timeframe}")

    if not df["time"].is_monotonic_increasing:
        raise RuntimeError(f"{label}: timestamps are not monotonic increasing for {symbol} timeframe={timeframe}")

    dupes = int(df["time"].duplicated().sum())
    if dupes > 0:
        raise RuntimeError(f"{label}: duplicate timestamps found for {symbol} timeframe={timeframe}: {dupes}")

    got = len(df)
    ratio = got / float(requested_bars) if requested_bars > 0 else 1.0
    if ratio < CONFIG["min_history_completion_ratio"]:
        msg = (
            f"{label}: only received {got}/{requested_bars} bars "
            f"({ratio:.2%}) for {symbol} timeframe={timeframe}. "
            f"MT5 local history may be incomplete."
        )
        if CONFIG["strict_history_check"]:
            raise RuntimeError(msg)
        log(f"WARNING: {msg}")

    return {
        "rows": got,
        "requested": requested_bars,
        "completion_ratio": ratio,
        "start": df["time"].iloc[0],
        "end": df["time"].iloc[-1],
    }


def get_rates_df_from_now(symbol, timeframe, bars):
    rates = mt5.copy_rates_from(symbol, timeframe, utc_now(), bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {symbol} timeframe={timeframe}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    return df


def latest_closed_bar_minutes_stale(df):
    if len(df) < 2:
        raise RuntimeError("Not enough bars to compute latest closed bar staleness")
    latest_closed_bar_time = ensure_utc_timestamp(df.iloc[-2]["time"])
    return (utc_now() - latest_closed_bar_time.to_pydatetime()).total_seconds() / 60.0, latest_closed_bar_time


def get_fresh_rates_df(symbol, timeframe, bars, stale_limit_minutes=None):
    retries = max(1, CONFIG["feed_refresh_retries"])
    sleep_seconds = max(0, CONFIG["feed_refresh_sleep_seconds"])
    stale_limit = CONFIG["stale_feed_minutes"] if stale_limit_minutes is None else stale_limit_minutes

    last_df = None
    last_minutes_stale = None
    last_bar_time = None
    best_df = None

    for attempt in range(1, retries + 1):
        trigger_symbol_refresh(symbol)
        df = get_rates_df_from_now(symbol, timeframe, bars)
        last_df = df
        best_df = df

        validate_timeframe_df(df, symbol, timeframe, bars, label=f"fresh_fetch_attempt_{attempt}")

        if len(df) < 2:
            if attempt < retries:
                time.sleep(sleep_seconds)
            continue

        minutes_stale, latest_closed_bar_time = latest_closed_bar_minutes_stale(df)
        last_minutes_stale = minutes_stale
        last_bar_time = latest_closed_bar_time

        if minutes_stale <= stale_limit:
            return df, minutes_stale, latest_closed_bar_time, attempt

        if attempt < retries:
            time.sleep(sleep_seconds)

    if best_df is None:
        raise RuntimeError(f"No rates for {symbol} timeframe={timeframe} after refresh retries={retries}")

    return best_df, last_minutes_stale, last_bar_time, retries


# ─────────────────────────────────────────────────────────────────────────────
# AUTO CONTRACT ROLL — detects front month by highest H1 volume
# ─────────────────────────────────────────────────────────────────────────────

def get_front_month_symbol(prefix="MNQ"):
    """
    Scan all MT5 symbols matching prefix (e.g. MNQ) and return the one
    with the highest recent H1 tick volume — that is the active front month.
    Falls back to LIVE_MT5_SYMBOL env var if no symbols found or all fail.
    """
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        raise RuntimeError("mt5.symbols_get() returned None")

    candidates = [
        s.name for s in all_symbols
        if s.name.startswith(prefix)
        and not s.name.startswith(prefix + "!")   # exclude continuous @MNQ-style aliases
        and len(s.name) > len(prefix)              # exclude bare prefix
    ]

    if not candidates:
        fallback = CONFIG.get("mt5_symbol", "")
        if fallback:
            log(f"AUTO_ROLL: no {prefix}* contracts found — using configured symbol {fallback}")
            return fallback
        raise RuntimeError(f"No symbols found matching prefix '{prefix}' and no fallback configured")

    best_sym = None
    best_vol = -1

    for sym in candidates:
        try:
            if not mt5.symbol_select(sym, True):
                continue
            rates = mt5.copy_rates_from(sym, mt5.TIMEFRAME_H1, utc_now(), 5)
            if rates is None or len(rates) == 0:
                continue
            vol = sum(r[5] for r in rates)   # index 5 = tick_volume
            if vol > best_vol:
                best_vol = vol
                best_sym = sym
        except Exception:
            continue

    if best_sym is None:
        fallback = CONFIG.get("mt5_symbol", "")
        if fallback:
            log(f"AUTO_ROLL: volume scan failed for all candidates — using configured symbol {fallback}")
            return fallback
        raise RuntimeError(f"Could not determine front month for prefix '{prefix}'")

    return best_sym


def resolve_symbol():
    """
    Resolve the active front-month contract.
    Uses AUTO_ROLL: scans MT5 for highest-volume MNQ contract.
    Falls back to LIVE_MT5_SYMBOL env var if scan fails.
    Validates that H1 data is available and fresh before returning.
    """
    prefix = CONFIG.get("symbol_prefix", "MNQ")
    sym = get_front_month_symbol(prefix)

    info = mt5.symbol_info(sym)
    if info is None:
        raise RuntimeError(f"symbol_info returned None for resolved symbol: {sym}")

    ensure_symbol_selected(sym)
    trigger_symbol_refresh(sym)

    h1_df, h1_minutes_stale, _, _ = get_fresh_rates_df(sym, mt5.TIMEFRAME_H1, min(10, CONFIG["h1_bars"]))
    if h1_df is None or len(h1_df) == 0:
        raise RuntimeError(f"No recent H1 rates available for resolved symbol: {sym}")

    if h1_minutes_stale is None:
        raise RuntimeError(f"Unable to validate H1 freshness for resolved symbol: {sym}")

    log(f"AUTO_ROLL: resolved front month = {sym} (volume scan over {prefix}* contracts)")
    return sym


def reset_symbol_state(store, new_symbol):
    store.state["active_symbol"] = new_symbol
    store.state["open_entry_bar_time"] = None
    store.state["last_forced_exit_bar_time"] = None
    store.state["last_entry_bar_time"] = None
    store.mark_dirty()


def prepare_regime_df(df_d1):
    df = df_d1.copy()
    df["ema100"] = ema(df["close"], CONFIG["ema_period"])
    df["ema_slope_n"] = df["ema100"] - df["ema100"].shift(CONFIG["slope_lookback"])
    df["atr14_d1"] = atr(df, CONFIG["atr_period"])
    df["rsi14"] = rsi(df["close"], CONFIG["rsi_period"])
    df["rsi10"] = rsi(df["close"], CONFIG["rsi_tighten_period"])
    df["d1_range"] = df["high"] - df["low"]
    df["d1_range_atr_ratio"] = df["d1_range"] / df["atr14_d1"].replace(0, np.nan)
    df["below_ema"] = df["close"] < df["ema100"]
    df["below_falling"] = df["below_ema"] & (df["ema_slope_n"] < 0)
    df["bear_permission"] = df["below_falling"] & (df["rsi14"] <= CONFIG["regime_rsi_max"])

    ages = []
    age = 0
    for flag in df["bear_permission"].fillna(False):
        age = age + 1 if bool(flag) else 0
        ages.append(age)
    df["regime_age"] = ages

    df["d1_close_vs_ema_atr"] = (df["close"] - df["ema100"]) / df["atr14_d1"].replace(0, np.nan)
    df["d1_ema_slope_atr"] = df["ema_slope_n"] / df["atr14_d1"].replace(0, np.nan)
    df["d1_ret_5"] = df["close"].pct_change(5)
    return df


def prepare_exec_df(df_h1, regime_df):
    df = df_h1.copy()
    df["ema100_h1"] = ema(df["close"], CONFIG["ema_period"])
    df["atr14"] = atr(df, CONFIG["atr_period"])
    df["rsi14_h1"] = rsi(df["close"], CONFIG["rsi_period"])
    df["ret_1h"] = df["close"].pct_change()
    df["ret_6h"] = df["close"].pct_change(6)
    df["ret_24h"] = df["close"].pct_change(24)
    df["bar_range"] = df["high"] - df["low"]
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["range_ma20"] = df["bar_range"].rolling(20).mean()
    df["volume_ma20"] = df["tick_volume"].rolling(20).mean()
    df["atr_ratio"] = df["bar_range"] / df["range_ma20"].replace(0, np.nan)
    df["tickvol_ratio"] = df["tick_volume"] / df["volume_ma20"].replace(0, np.nan)
    df["recent_low_prev"] = df["low"].rolling(CONFIG["breakout_lookback"]).min().shift(1)
    df["recent_high_prev"] = df["high"].rolling(CONFIG["breakout_lookback"]).max().shift(1)
    df["swing_low_24"] = df["low"].rolling(24).min().shift(1)
    df["swing_high_24"] = df["high"].rolling(24).max().shift(1)
    df["dist_to_recent_low_atr"] = (df["close"] - df["recent_low_prev"]) / df["atr14"].replace(0, np.nan)
    df["dist_to_recent_high_atr"] = (df["recent_high_prev"] - df["close"]) / df["atr14"].replace(0, np.nan)
    df["dist_to_h1_ema_atr"] = (df["close"] - df["ema100_h1"]) / df["atr14"].replace(0, np.nan)
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df["is_us_session"] = ((df["hour"] >= CONFIG["session_start_hour_utc"]) &
                           (df["hour"] < CONFIG["session_end_hour_utc"])).astype(int)

    regime_map = regime_df[[
        "time", "bear_permission", "regime_age", "rsi14", "rsi10",
        "d1_close_vs_ema_atr", "d1_ema_slope_atr", "d1_ret_5", "d1_range_atr_ratio"
    ]].copy()
    regime_map = regime_map.rename(
        columns={"time": "d1_time", "rsi14": "d1_rsi14", "rsi10": "d1_rsi10"}
    ).sort_values("d1_time")

    df = pd.merge_asof(
        df.sort_values("time"),
        regime_map,
        left_on="time",
        right_on="d1_time",
        direction="backward"
    ).reset_index(drop=True)

    in_bear = df["bear_permission"].fillna(False)
    red_bar = df["close"] < df["open"]
    atr_ok = df["bar_range"] > (df["range_ma20"] * CONFIG["atr_expansion_min"])
    session_ok = True if not CONFIG["enable_session_filter"] else df["is_us_session"] == 1

    df["trig_atr_breakdown"] = in_bear & red_bar & atr_ok & (df["close"] < df["recent_low_prev"]) & session_ok
    df["trig_rsi_rollover"] = in_bear & (df["rsi14_h1"].shift(1) > 60) & (df["rsi14_h1"] < 50) & red_bar & session_ok
    return df


def build_latest_signal(exec_df):
    if len(exec_df) < 40:
        return None

    signal_idx = len(exec_df) - 2
    entry_idx = signal_idx + 1
    if signal_idx < 0 or entry_idx >= len(exec_df):
        return None

    row = exec_df.iloc[signal_idx]
    entry_row = exec_df.iloc[entry_idx]

    trig_atr = bool(row.get("trig_atr_breakdown", False))
    trig_rsi = bool(row.get("trig_rsi_rollover", False))
    if not trig_atr and not trig_rsi:
        return None

    trigger = "atr_breakdown" if trig_atr else "rsi_rollover"
    atr_now = row.get("atr14", np.nan)
    if pd.isna(atr_now) or atr_now <= 0:
        return None

    prev1 = exec_df.iloc[signal_idx - 1] if signal_idx - 1 >= 0 else row
    slice6 = exec_df.iloc[max(0, signal_idx - 6):signal_idx]
    slice24 = exec_df.iloc[max(0, signal_idx - 24):signal_idx]

    return {
        "signal_idx": int(signal_idx),
        "signal_time": row["time"],
        "entry_idx": int(entry_idx),
        "entry_time": entry_row["time"],
        "trigger": trigger,
        "entry_price": float(entry_row["open"]),
        "regime_age": row.get("regime_age", np.nan),
        "d1_rsi14": row.get("d1_rsi14", np.nan),
        "d1_rsi10": row.get("d1_rsi10", np.nan),
        "d1_range_atr_ratio": row.get("d1_range_atr_ratio", np.nan),
        "d1_close_vs_ema_atr": row.get("d1_close_vs_ema_atr", np.nan),
        "d1_ema_slope_atr": row.get("d1_ema_slope_atr", np.nan),
        "d1_ret_5": row.get("d1_ret_5", np.nan),
        "h1_atr14": row.get("atr14", np.nan),
        "h1_rsi14": row.get("rsi14_h1", np.nan),
        "h1_ret_1h": row.get("ret_1h", np.nan),
        "h1_ret_6h": row.get("ret_6h", np.nan),
        "h1_ret_24h": row.get("ret_24h", np.nan),
        "bar_range": row.get("bar_range", np.nan),
        "body_size": row.get("body_size", np.nan),
        "atr_ratio": row.get("atr_ratio", np.nan),
        "tickvol_ratio": row.get("tickvol_ratio", np.nan),
        "dist_to_recent_low_atr": row.get("dist_to_recent_low_atr", np.nan),
        "dist_to_recent_high_atr": row.get("dist_to_recent_high_atr", np.nan),
        "dist_to_h1_ema_atr": row.get("dist_to_h1_ema_atr", np.nan),
        "recent_6h_mean_ret": slice6["ret_1h"].mean() if len(slice6) else np.nan,
        "recent_6h_std_ret": slice6["ret_1h"].std(ddof=1) if len(slice6) > 1 else np.nan,
        "recent_24h_mean_ret": slice24["ret_1h"].mean() if len(slice24) else np.nan,
        "recent_24h_std_ret": slice24["ret_1h"].std(ddof=1) if len(slice24) > 1 else np.nan,
        "recent_24h_low_break_distance": ((row["close"] - row.get("swing_low_24", np.nan)) / atr_now) if atr_now else np.nan,
        "recent_24h_high_distance": ((row.get("swing_high_24", np.nan) - row["close"]) / atr_now) if atr_now else np.nan,
        "hour": row.get("hour", np.nan),
        "dayofweek": row.get("dayofweek", np.nan),
        "is_us_session": row.get("is_us_session", np.nan),
        "trigger_atr_breakdown": 1 if trigger == "atr_breakdown" else 0,
        "trigger_rsi_rollover": 1 if trigger == "rsi_rollover" else 0,
        "prev_bar_red": 1 if prev1["close"] < prev1["open"] else 0,
        "prev_bar_range": prev1.get("bar_range", np.nan),
        "prev_bar_rsi14": prev1.get("rsi14_h1", np.nan),
        "prev_bar_atr_ratio": prev1.get("atr_ratio", np.nan),
    }


def round_to_tick(symbol, price):
    """
    Round price to the symbol's minimum tick size.
    Uses trade_tick_size (broker minimum increment) rather than just info.digits,
    so prices on futures with non-decimal tick increments (e.g. MNQ = 0.25 pts)
    are always valid.  Falls back to digit-only rounding if tick size is unavailable.
    """
    info = get_symbol_info_or_raise(symbol)
    tick_size = getattr(info, "trade_tick_size", 0.0)
    if tick_size and tick_size > 0:
        rounded = round(round(price / tick_size) * tick_size, info.digits)
    else:
        rounded = round(float(price), info.digits)
    return rounded


def round_price(symbol, price):
    """Kept for backward compatibility — delegates to round_to_tick."""
    return round_to_tick(symbol, price)


def any_open_position_terminal():
    """
    Returns True if any position is open, or if MT5 query fails (fail-safe).
    NEVER returns False on a None result — that would silently allow entries
    during a terminal/API error.
    """
    pos = mt5.positions_get()
    if pos is None:
        log(f"WARNING: positions_get() returned None: {mt5.last_error()} — treating as position open (fail-safe)")
        return True   # block trading on query failure
    return len(pos) > 0


def symbol_open_position(symbol):
    """Fail-safe: returns True on None (query error) to block entries."""
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        log(f"WARNING: positions_get(symbol={symbol}) returned None: {mt5.last_error()} — treating as position open (fail-safe)")
        return True
    return len(pos) > 0


def count_open_shorts(symbol):
    """Fail-safe: raises RuntimeError on None so callers can handle the error explicitly."""
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        raise RuntimeError(f"positions_get(symbol={symbol}) returned None: {mt5.last_error()}")
    return sum(1 for p in pos if p.type == mt5.POSITION_TYPE_SELL)


def current_short_position(symbol):
    """Returns None if no short position, raises RuntimeError on MT5 query failure."""
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        raise RuntimeError(f"positions_get(symbol={symbol}) returned None: {mt5.last_error()}")
    for p in pos:
        if p.type == mt5.POSITION_TYPE_SELL:
            return p
    return None


def close_position(position):
    tick = get_fresh_tick(position.symbol, retries=3, sleep_seconds=1)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": float(position.volume),
        "type": mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": tick.ask,
        "deviation": CONFIG["deviation"],
        "magic": CONFIG["magic"],
        "comment": f"{CONFIG['order_comment']}_time_exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result, request


def send_short(symbol, volume, sl_price, tp_price):
    tick = get_fresh_tick(symbol, retries=3, sleep_seconds=1)

    # ── FINAL GATE: last-line-of-defense position check before order_send ────
    # Closes the race window between the upstream check and the actual send.
    if any_open_position_terminal():
        raise RuntimeError("FINAL GATE: position detected immediately before order_send — aborting")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_SELL,
        "price": tick.bid,
        "sl": round_to_tick(symbol, sl_price),
        "tp": round_to_tick(symbol, tp_price),
        "deviation": CONFIG["deviation"],
        "magic": CONFIG["magic"],
        "comment": CONFIG["order_comment"],
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result, request


def modify_position(position, new_sl=None, new_tp=None):
    # Guard: if existing SL/TP is 0.0 (unset), don't resubmit 0.0 — keep None
    # to avoid accidentally removing a level at the broker.
    existing_sl = position.sl if position.sl and position.sl > 0 else None
    existing_tp = position.tp if position.tp and position.tp > 0 else None

    sl_to_send = round_to_tick(position.symbol, new_sl) if new_sl is not None else \
                 (round_to_tick(position.symbol, existing_sl) if existing_sl is not None else 0.0)
    tp_to_send = round_to_tick(position.symbol, new_tp) if new_tp is not None else \
                 (round_to_tick(position.symbol, existing_tp) if existing_tp is not None else 0.0)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": position.symbol,
        "position": position.ticket,
        "sl": sl_to_send,
        "tp": tp_to_send,
    }
    result = mt5.order_send(request)
    return result, request


def should_tighten(d1_rsi10, d1_range_atr_ratio):
    rsi_hit = pd.notna(d1_rsi10) and float(d1_rsi10) < CONFIG["tighten_rsi_threshold"]
    atr_hit = pd.notna(d1_range_atr_ratio) and float(d1_range_atr_ratio) >= CONFIG["tighten_atr_stretch_threshold"]
    mode = CONFIG["tighten_mode"].lower()

    if mode == "rsi":
        return rsi_hit
    if mode == "atr_exhaust":
        return atr_hit
    return rsi_hit or atr_hit


def compute_trailing_stop_for_short(position, current_price):
    favorable_points = max(0.0, position.price_open - current_price)
    if favorable_points <= 0:
        return None

    lock_points = favorable_points * CONFIG["tighten_factor"]
    locked_stop = position.price_open - lock_points
    min_valid_sl = current_price + CONFIG["tighten_buffer_points"]
    candidate_sl = max(locked_stop, min_valid_sl)

    # Use live position volume for the max-loss cap, not CONFIG volume,
    # in case the position was partially filled or manually adjusted.
    live_volume = float(position.volume) if position.volume and position.volume > 0 else CONFIG["volume"]
    original_cap = position.price_open + min(
        CONFIG["fixed_stop_points"],
        CONFIG["max_loss_usd"] / (CONFIG["mnq_point_value"] * live_volume)
    )
    candidate_sl = min(candidate_sl, original_cap)
    return candidate_sl


def update_daily_trade_limit(store):
    today = utc_now().strftime("%Y-%m-%d")
    if store.state.get("trades_today_date") != today:
        store.state["trades_today_date"] = today
        store.state["trades_today_count"] = 0
        store.mark_dirty()
        store.save_if_dirty()
        return True
    return False


def bars_since_timestamp(exec_df, older_bar_iso, latest_closed_bar_iso):
    if not older_bar_iso:
        return None, "missing_older_timestamp"

    closed_h1 = exec_df.iloc[:-1].copy()

    try:
        older_ts = pd.Timestamp(older_bar_iso)
        if older_ts.tzinfo is None:
            older_ts = older_ts.tz_localize("UTC")
        else:
            older_ts = older_ts.tz_convert("UTC")

        latest_ts = pd.Timestamp(latest_closed_bar_iso)
        if latest_ts.tzinfo is None:
            latest_ts = latest_ts.tz_localize("UTC")
        else:
            latest_ts = latest_ts.tz_convert("UTC")
    except Exception as e:
        return None, f"timestamp_parse_error: {e}"

    # Normalize dataframe timestamps to UTC for comparison
    df_times = closed_h1["time"]
    if hasattr(df_times.dtype, "tz") and df_times.dtype.tz is None:
        df_times = df_times.dt.tz_localize("UTC")
    elif hasattr(df_times.dtype, "tz") and df_times.dtype.tz is not None:
        df_times = df_times.dt.tz_convert("UTC")

    older_matches  = np.where(df_times.values == np.datetime64(older_ts.to_datetime64()))[0]
    latest_matches = np.where(df_times.values == np.datetime64(latest_ts.to_datetime64()))[0]

    if len(older_matches) == 0:
        return None, "older_bar_not_found"
    if len(latest_matches) == 0:
        return None, "latest_bar_not_found"

    older_pos  = int(older_matches[0])
    latest_pos = int(latest_matches[0])

    if latest_pos < older_pos:
        return None, "latest_bar_before_older_bar"

    return int(latest_pos - older_pos), None


def manage_open_trade(symbol, regime_df, exec_df, store):
    pos = current_short_position(symbol)

    # ── State reconciliation: clear stale entry state if no position exists ──
    # Covers SL hit, TP hit, or manual close — any external exit that bypasses
    # the max-hold code path. Without this, open_entry_bar_time persists and
    # distorts bars_since_timestamp() checks for the next entry.
    if pos is None:
        if store.state.get("open_entry_bar_time") is not None:
            log(f"RECONCILE: no open position but open_entry_bar_time={store.state['open_entry_bar_time']} — clearing stale state")
            store.state["open_entry_bar_time"] = None
            store.mark_dirty()
        return {"action": "none", "changed": False, "skip_entry_this_loop": False}

    latest_d1 = regime_df.iloc[-1]
    tick = get_fresh_tick(symbol, retries=3, sleep_seconds=1)
    current_price = tick.ask
    last_closed_bar_time = exec_df.iloc[-2]["time"].isoformat()

    entry_bar_time_str = store.state.get("open_entry_bar_time")
    bars_held = 0
    if entry_bar_time_str:
        bars, err = bars_since_timestamp(exec_df, entry_bar_time_str, last_closed_bar_time)
        if bars is None:
            append_csv(CONFIG["decision_log_csv"], {
                "ts_utc": utc_now_str(),
                "event": "bars_held_calc_warning",
                "symbol": symbol,
                "entry_bar_time": entry_bar_time_str,
                "latest_closed_bar": last_closed_bar_time,
                "reason": err,
            })
        else:
            bars_held = bars + 1

    if CONFIG["max_hold_bars"] > 0 and bars_held >= CONFIG["max_hold_bars"]:
        result, request = close_position(pos)
        append_csv(CONFIG["trade_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "max_hold_exit",
            "symbol": symbol,
            "ticket": pos.ticket,
            "bars_held": bars_held,
            "entry_price": pos.price_open,
            "exit_price_request": request["price"],
            "retcode": getattr(result, "retcode", None),
            "result_comment": getattr(result, "comment", ""),
            "order": getattr(result, "order", None),
            "deal": getattr(result, "deal", None),
        })

        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            log(f"Closed {symbol} short due to max_hold_bars={bars_held}")
            store.state["open_entry_bar_time"] = None
            store.state["last_forced_exit_bar_time"] = last_closed_bar_time
            store.mark_dirty()
            return {"action": "closed_max_hold", "changed": True, "skip_entry_this_loop": True}

        log(f"Max-hold close failed retcode={getattr(result, 'retcode', None)} comment={getattr(result, 'comment', '')}")
        return {"action": "close_failed", "changed": False, "skip_entry_this_loop": True}

    if not should_tighten(latest_d1["rsi10"], latest_d1["d1_range_atr_ratio"]):
        return {"action": "none", "changed": False, "skip_entry_this_loop": False}

    fallback_sl = pos.price_open + min(
        CONFIG["fixed_stop_points"],
        CONFIG["max_loss_usd"] / (CONFIG["mnq_point_value"] * CONFIG["volume"])
    )
    current_sl = pos.sl if pos.sl and pos.sl > 0 else fallback_sl

    candidate_sl = compute_trailing_stop_for_short(pos, current_price)
    if candidate_sl is None:
        return {"action": "none", "changed": False, "skip_entry_this_loop": False}

    candidate_sl = round_price(symbol, candidate_sl)
    if candidate_sl >= current_sl:
        return {"action": "none", "changed": False, "skip_entry_this_loop": False}

    result, request = modify_position(pos, new_sl=candidate_sl, new_tp=pos.tp)
    success = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    append_csv(CONFIG["decision_log_csv"], {
        "ts_utc": utc_now_str(),
        "event": "tighten_stop_success" if success else "tighten_stop_failed",
        "symbol": symbol,
        "ticket": pos.ticket,
        "entry_price": pos.price_open,
        "current_price": current_price,
        "old_sl": current_sl,
        "new_sl": candidate_sl,
        "tp": pos.tp,
        "retcode": getattr(result, "retcode", None),
        "comment": getattr(result, "comment", ""),
    })

    if success:
        log(f"Tighten succeeded for {symbol} ticket={pos.ticket} old_sl={current_sl} new_sl={candidate_sl}")
        return {"action": "tightened", "changed": False, "skip_entry_this_loop": False}

    log(f"Tighten failed for {symbol} ticket={pos.ticket} retcode={getattr(result, 'retcode', None)} comment={getattr(result, 'comment', '')}")
    return {"action": "tighten_failed", "changed": False, "skip_entry_this_loop": False}


def infer_and_trade(symbol, regime_df, exec_df, model, store, block_new_entries=False):
    if len(exec_df) < 2:
        return False

    update_daily_trade_limit(store)

    latest_closed_bar = exec_df.iloc[-2]["time"].isoformat()
    if store.state.get("last_processed_h1_bar") == latest_closed_bar:
        return False

    store.state["last_processed_h1_bar"] = latest_closed_bar
    store.mark_dirty()
    store.save_if_dirty()

    if block_new_entries:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "entry_skipped_same_loop_after_management",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
        })
        return True

    if store.state.get("last_forced_exit_bar_time") == latest_closed_bar:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "entry_skipped_same_bar_after_forced_exit",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
        })
        return True

    if store.state.get("trades_today_count", 0) >= CONFIG["max_trades_per_day"]:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "blocked_daily_trade_limit",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
            "trades_today_count": store.state.get("trades_today_count", 0),
        })
        return True

    bars_since_entry = None
    last_entry_bar = store.state.get("last_entry_bar_time")
    if last_entry_bar:
        bars_since_entry, err = bars_since_timestamp(exec_df, last_entry_bar, latest_closed_bar)
        if bars_since_entry is None:
            append_csv(CONFIG["decision_log_csv"], {
                "ts_utc": utc_now_str(),
                "event": "blocked_min_bars_unknown",
                "symbol": symbol,
                "bar_time": latest_closed_bar,
                "last_entry_bar_time": last_entry_bar,
                "reason": err,
            })
            return True

    if bars_since_entry is not None and bars_since_entry < CONFIG["min_bars_between_entries"]:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "blocked_min_bars_between_entries",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
            "bars_since_entry": bars_since_entry,
            "min_bars_between_entries": CONFIG["min_bars_between_entries"],
        })
        return True

    if count_open_shorts(symbol) >= CONFIG["max_open_shorts"]:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "blocked_max_open_shorts",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
            "max_open_shorts": CONFIG["max_open_shorts"],
        })
        return True

    signal = build_latest_signal(exec_df)
    if signal is None:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "no_signal",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
        })
        return True

    X = pd.DataFrame([signal])[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if list(X.columns) != FEATURE_COLS:
        raise RuntimeError("Feature column ordering mismatch before live inference")

    rf_prob = float(model.predict_proba(X)[0, 1])

    append_csv(CONFIG["decision_log_csv"], {
        "ts_utc": utc_now_str(),
        "event": "signal_detected",
        "symbol": symbol,
        "bar_time": latest_closed_bar,
        "trigger": signal["trigger"],
        "rf_prob": rf_prob,
        "threshold": CONFIG["rf_threshold"],
    })

    if rf_prob < CONFIG["rf_threshold"]:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "signal_rejected_rf",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
            "trigger": signal["trigger"],
            "rf_prob": rf_prob,
        })
        return True

    if CONFIG["allow_only_one_position_in_terminal"] and any_open_position_terminal():
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "blocked_existing_terminal_position",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
            "trigger": signal["trigger"],
            "rf_prob": rf_prob,
        })
        log("Blocked order because an open position already exists in MT5 terminal")
        return True
    elif CONFIG["allow_only_one_position_per_symbol"] and symbol_open_position(symbol):
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "blocked_existing_symbol_position",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
            "trigger": signal["trigger"],
            "rf_prob": rf_prob,
        })
        return True

    if count_open_shorts(symbol) >= CONFIG["max_open_shorts"]:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "blocked_max_open_shorts_pre_send",
            "symbol": symbol,
            "bar_time": latest_closed_bar,
            "max_open_shorts": CONFIG["max_open_shorts"],
        })
        return True

    stop_points = min(
        CONFIG["fixed_stop_points"],
        CONFIG["max_loss_usd"] / (CONFIG["mnq_point_value"] * CONFIG["volume"])
    )
    target_points = CONFIG["fixed_target_points"]

    tick = get_fresh_tick(symbol, retries=3, sleep_seconds=1)
    entry_price = tick.bid
    sl_price = entry_price + stop_points
    tp_price = entry_price - target_points

    result, request = send_short(symbol, CONFIG["volume"], sl_price, tp_price)

    append_csv(CONFIG["trade_log_csv"], {
        "ts_utc": utc_now_str(),
        "event": "new_short_entry",
        "symbol": symbol,
        "trigger": signal["trigger"],
        "signal_time": iso_ts(signal["signal_time"]),
        "entry_bar_time": latest_closed_bar,
        "entry_price_signal": signal["entry_price"],
        "entry_price_request": request["price"],
        "rf_prob": rf_prob,
        "threshold": CONFIG["rf_threshold"],
        "sl": request["sl"],
        "tp": request["tp"],
        "retcode": getattr(result, "retcode", None),
        "result_comment": getattr(result, "comment", ""),
        "order": getattr(result, "order", None),
        "deal": getattr(result, "deal", None),
    })

    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        log(f"Opened SELL {symbol} trigger={signal['trigger']} rf_prob={rf_prob:.3f}")
        store.state["open_entry_bar_time"] = latest_closed_bar
        store.state["last_entry_bar_time"] = latest_closed_bar
        store.state["trades_today_count"] = store.state.get("trades_today_count", 0) + 1
        store.mark_dirty()
        store.save_if_dirty()
        return True

    log(f"Order send failed retcode={getattr(result, 'retcode', None)} comment={getattr(result, 'comment', '')}")
    return True


def validate_model_features():
    features_file = CONFIG["rf_features_file"]
    if not os.path.exists(features_file):
        raise FileNotFoundError(f"RF features file not found: {features_file}")

    with open(features_file, "r", encoding="utf-8") as f:
        trained_cols = json.load(f)

    if trained_cols != FEATURE_COLS:
        raise ValueError(
            f"Feature mismatch between saved model and live bot.\n"
            f"Saved: {trained_cols}\n"
            f"Live:  {FEATURE_COLS}"
        )


def validate_runtime_config():
    if not 0.0 < CONFIG["tighten_factor"] <= 1.0:
        raise ValueError(f"tighten_factor must be in (0, 1]: got {CONFIG['tighten_factor']}")

    if CONFIG["tighten_buffer_points"] >= CONFIG["fixed_stop_points"]:
        log("WARNING: tighten_buffer_points >= fixed_stop_points — tightening may never fire")

    if CONFIG["volume"] <= 0:
        raise ValueError(f"TRADE_VOLUME must be > 0, got {CONFIG['volume']}")

    if CONFIG["mnq_point_value"] <= 0:
        raise ValueError(f"MNQ_POINT_VALUE must be > 0, got {CONFIG['mnq_point_value']}")

    if CONFIG["max_loss_usd"] <= 0:
        raise ValueError(f"MAX_LOSS_USD must be > 0, got {CONFIG['max_loss_usd']}")

    if CONFIG["fixed_stop_points"] <= 0:
        raise ValueError(f"FIXED_STOP_POINTS must be > 0, got {CONFIG['fixed_stop_points']}")

    if CONFIG["fixed_target_points"] <= 0:
        raise ValueError(f"FIXED_TARGET_POINTS must be > 0, got {CONFIG['fixed_target_points']}")

    if CONFIG["max_hold_bars"] < 0:
        raise ValueError(f"MAX_HOLD_BARS must be >= 0, got {CONFIG['max_hold_bars']}")

    if CONFIG["feed_refresh_retries"] < 1:
        raise ValueError(f"FEED_REFRESH_RETRIES must be >= 1, got {CONFIG['feed_refresh_retries']}")

    if CONFIG["stale_feed_minutes"] < 1:
        raise ValueError(f"STALE_FEED_MINUTES must be >= 1, got {CONFIG['stale_feed_minutes']}")

    if not 0 < CONFIG["min_history_completion_ratio"] <= 1:
        raise ValueError(
            f"MIN_HISTORY_COMPLETION_RATIO must be in (0,1], got {CONFIG['min_history_completion_ratio']}"
        )


def run_once(symbol, model, store):
    reconnected = ensure_mt5_connection()
    if reconnected:
        new_symbol = resolve_symbol()
        if new_symbol != symbol:
            log(f"AUTO_ROLL: symbol changed after reconnect: {symbol} -> {new_symbol}")
            reset_symbol_state(store, new_symbol)
        symbol = new_symbol
    else:
        ensure_symbol_selected(symbol)

    if store.state.get("active_symbol") != symbol:
        reset_symbol_state(store, symbol)

    trigger_symbol_refresh(symbol)

    df_d1 = get_rates_df_from_now(symbol, mt5.TIMEFRAME_D1, CONFIG["d1_bars"])
    validate_timeframe_df(df_d1, symbol, mt5.TIMEFRAME_D1, CONFIG["d1_bars"], label="live_d1")

    df_h1, minutes_stale, latest_closed_bar_time, refresh_attempts = get_fresh_rates_df(
        symbol, mt5.TIMEFRAME_H1, CONFIG["h1_bars"], stale_limit_minutes=CONFIG["stale_feed_minutes"]
    )

    if len(df_h1) < 2:
        raise RuntimeError("Not enough H1 bars to evaluate live state")

    stale_feed = minutes_stale is None or minutes_stale > CONFIG["stale_feed_minutes"]

    if stale_feed:
        append_csv(CONFIG["decision_log_csv"], {
            "ts_utc": utc_now_str(),
            "event": "stale_feed_skip_all",
            "symbol": symbol,
            "latest_closed_bar_time": latest_closed_bar_time.isoformat() if latest_closed_bar_time is not None else None,
            "minutes_stale": minutes_stale,
            "stale_feed_minutes": CONFIG["stale_feed_minutes"],
            "refresh_attempts": refresh_attempts,
        })
        stale_txt = "unknown" if minutes_stale is None else f"{minutes_stale:.1f}"
        log(f"Stale feed detected for {symbol}: latest bar is {stale_txt} minutes old after refresh attempts={refresh_attempts}")

        if CONFIG["feed_refresh_reinit_on_stale"]:
            log(f"Forcing MT5 reinitialize on stale feed for {symbol}")
            try:
                mt5.shutdown()
            except Exception:
                pass

        store.save_if_dirty()
        return symbol, False

    regime_df = prepare_regime_df(df_d1)
    exec_df = prepare_exec_df(df_h1, regime_df)

    management = manage_open_trade(symbol, regime_df, exec_df, store)
    changed = infer_and_trade(
        symbol, regime_df, exec_df, model, store,
        block_new_entries=management["skip_entry_this_loop"]
    )
    store.save_if_dirty()
    return symbol, (changed or management["changed"])


# ─────────────────────────────────────────────────────────────────────────────
# PERIODIC ROLL CHECK — re-resolves front month every N loop iterations
# so a roll mid-session is detected without waiting for a reconnect
# ─────────────────────────────────────────────────────────────────────────────
ROLL_CHECK_INTERVAL = 720   # every 720 polls x 5s = 1 hour


def main():
    validate_runtime_config()

    if not os.path.exists(CONFIG["rf_model_file"]):
        raise FileNotFoundError(f"RF model file not found: {CONFIG['rf_model_file']}")

    validate_model_features()

    model = joblib.load(CONFIG["rf_model_file"])
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Loaded model does not support predict_proba: {type(model)}")

    lock_handle = acquire_process_lock()
    store = StateStore(CONFIG["state_file"])

    ensure_mt5_connection()

    acct = mt5.account_info()
    if acct is not None:
        log(f"Account free margin: {getattr(acct, 'margin_free', None)}")

    symbol = resolve_symbol()
    get_symbol_info_or_raise(symbol)

    if store.state.get("active_symbol") != symbol:
        reset_symbol_state(store, symbol)
        store.save_if_dirty()

    log(f"Starting with symbol: {symbol}")

    poll_count = 0

    try:
        while True:
            poll_count += 1

            # ── Periodic roll check — re-resolve front month every hour ──────
            # Catches contract rolls mid-session without needing a reconnect.
            # Skipped if a position is open to avoid disrupting trade management.
            if poll_count % ROLL_CHECK_INTERVAL == 0:
                if not any_open_position_terminal():
                    try:
                        new_symbol = resolve_symbol()
                        if new_symbol != symbol:
                            log(f"AUTO_ROLL: periodic check detected roll: {symbol} -> {new_symbol}")
                            reset_symbol_state(store, new_symbol)
                            store.save_if_dirty()
                            symbol = new_symbol
                        else:
                            log(f"AUTO_ROLL: periodic check — still on {symbol}, no roll needed")
                    except Exception as roll_exc:
                        log(f"AUTO_ROLL: periodic check failed (non-fatal): {roll_exc}")
                else:
                    log(f"AUTO_ROLL: periodic check skipped — position open on {symbol}")

            try:
                symbol, changed = run_once(symbol, model, store)
                if changed:
                    store.save_if_dirty()
            except Exception as exc:
                append_csv(CONFIG["decision_log_csv"], {
                    "ts_utc": utc_now_str(),
                    "event": "loop_exception",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                })
                log(f"Loop exception: {exc}")
                try:
                    mt5.shutdown()
                except Exception:
                    pass
                time.sleep(CONFIG["reconnect_sleep_seconds"])
                continue

            time.sleep(CONFIG["poll_seconds"])
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass
        try:
            if portalocker is not None:
                portalocker.unlock(lock_handle)
            lock_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped by user")
        sys.exit(0)
