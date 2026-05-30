"""
levels_alert.py — Key Level Scanner + Telegram Alert (Windows native MT5 version)
Run via Windows Task Scheduler every 5 minutes during trading hours.

Scans for ML-scored support levels, sends Telegram alert when
actionable levels are found. Logs every signal to build history.

Requirements (Windows):
  pip install MetaTrader5 xgboost scikit-learn pandas numpy joblib requests
"""

import os
import json
import logging
import requests
import traceback
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these
# ─────────────────────────────────────────────────────────────────────────────

# ── CREDENTIALS — load from environment or .env file ─────────────────────────
# On Windows VPS: set as System Environment Variables, or create a .env file
# in the same directory with:
#   TELEGRAM_TOKEN=your_token_here
#   TELEGRAM_CHAT_ID=your_chat_id_here
# Then: pip install python-dotenv  and uncomment the two lines below:
# from dotenv import load_dotenv; load_dotenv()
import os as _os
TELEGRAM_TOKEN   = _os.environ.get("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = _os.environ.get("TELEGRAM_CHAT_ID", "")
# ── FALLBACK: hardcode here ONLY for local testing, never commit to VCS ──────
# TELEGRAM_TOKEN   = "YOUR_TOKEN_HERE"
# TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"

# Symbol
SYMBOL_PREFIX = "MNQ"    # auto-detects front month e.g. MNQM26
TF_LABEL      = "H1"

# Alert thresholds
SCORE_STRONG  = 0.75    # 75%+ → STRONG signal
SCORE_WATCH   = 0.60    # 60-75% → WATCH signal
MAX_DIST_ATR  = 5.0     # only alert levels within this many ATR of price
ALERT_MIN_DIST = 0.0    # minimum distance — 0 means alert even if at level

# Only alert if at least one level is within this distance
CLOSE_LEVEL_ATR = 3.0   # suppress alert if nothing within 3 ATR

# Daily trend filter — include in alert info
DAILY_EMA_PERIOD = 100

# Model + data paths
MODEL_PATH    = "levels_xgb_model.joblib"
ENSEMBLE_PATH = "ensemble_models.joblib"
DATASET_PATH  = "levels_dataset.csv"

# Signal history log — builds up over time
SIGNAL_LOG_PATH  = "logs/signal_history.json"
SCAN_LOG_PATH    = "logs/scan_history.log"
LAST_ALERT_PATH  = "logs/last_alert.json"

# Suppress duplicate alerts — don't re-alert same level within N hours
SUPPRESS_HOURS   = 4

# ── AUTO TRADE CONFIG ─────────────────────────────────────────────────────────
AUTO_TRADE           = True
AUTO_TRADE_MIN_SCORE = 0.75
AUTO_TRADE_VOLUME    = 1.0
AUTO_TRADE_MAGIC     = 20260002          # unique magic number — do not share with other EAs
NY_SESSION_END_UTC   = 21               # Hour (UTC) unfilled orders are cancelled
                                         # 21:00 UTC = 5:00 PM ET = end of NY session
ORDER_EXPIRY_RUNS    = 252              # Hard backstop — 252 runs x 5 min = 21 hours
                                         # catches edge case of order placed just before session end
ORDER_STATE_PATH     = "logs/order_state.json"

# ── LEVEL DETECTION PARAMS ────────────────────────────────────────────────────
SWING_LOOKBACK = 5       # bars each side to confirm a swing low
LEVEL_ZONE_ATR = 0.30    # level zone half-width as ATR multiple

# ── ORDER SIZING ──────────────────────────────────────────────────────────────
SL_ATR = 1.5             # stop loss distance as ATR multiple  [updated per OOS study]
TP_R   = 3.0             # take profit as R multiple              [updated per OOS study]

# ── HTF PROXIMITY THRESHOLDS (must match levels_ml_win.py) ───────────────────
HTF_CONFLUENCE_THRESH     = 0.75   # ATR multiples for support stack count
HTF_ANY_CONFLUENCE_THRESH = 0.50   # ATR multiples for htf_any_confluence flag
HTF_OVERHEAD_TIGHT_THRESH = 1.50   # ATR multiples for overhead resistance flag

# ── TIMEFRAME CONSTANTS ───────────────────────────────────────────────────────
TIMEFRAME_M5  = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1  = 16385
TIMEFRAME_H4  = 16388
TIMEFRAME_D1  = 16408
TIMEFRAME_W1  = 32769
TIMEFRAME_MN1 = 49153

TIMEFRAME_MAP = {
    "M5":  TIMEFRAME_M5,
    "M15": TIMEFRAME_M15,
    "H1":  TIMEFRAME_H1,
    "H4":  TIMEFRAME_H4,
    "D1":  TIMEFRAME_D1,
}
HTF_MAP = {
    TIMEFRAME_H1:  TIMEFRAME_H4,
    TIMEFRAME_M15: TIMEFRAME_H1,
    TIMEFRAME_M5:  TIMEFRAME_M15,
}
# ── INDICATOR PARAMS ─────────────────────────────────────────────────────────
ATR_PERIOD = 14          # ATR period used for all volatility calculations
# ──────────────────────────────────────────────────────────────────────────────

# ── Windows native MT5 ───────────────────────────────────────────────────────
import MetaTrader5 as mt5

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)

# ── Logging — UTF-8 on both handlers so emojis don't crash on Windows ────────
_log_fmt = logging.Formatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_fmt)
# Wrap stdout in utf-8 to handle emoji characters on Windows cp1252 consoles
import io as _io
_stream_handler.stream = _io.TextIOWrapper(
    _stream_handler.stream.buffer, encoding="utf-8", errors="replace", line_buffering=True
)
_file_handler = logging.FileHandler(SCAN_LOG_PATH, encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
log = logging.getLogger("levels_alert")


def get_front_month_symbol(prefix: str = "MNQ") -> str:
    """
    Auto-detect the active front month contract by finding the
    MNQ* symbol with the highest recent tick volume.
    Falls back to prefix if nothing found.
    """
    try:
        symbols = mt5.symbols_get(f"{prefix}*")
        if not symbols:
            log.warning(f"No symbols matching {prefix}* — using {prefix}")
            return prefix
        best_sym = None
        best_vol = -1
        for sym in symbols:
            name = sym.name
            if "@" in name or name == prefix:
                continue
            rates = mt5.copy_rates_from_pos(name, TIMEFRAME_H1, 0, 1)
            if rates is not None and len(rates) > 0:
                vol = rates[0]["tick_volume"]
                if vol > best_vol:
                    best_vol = vol
                    best_sym = name
        if best_sym:
            log.info(f"Front month detected: {best_sym} (vol {best_vol})")
            return best_sym
    except Exception as e:
        log.warning(f"Symbol detection error: {e}")
    log.warning(f"Could not detect front month — using {prefix}")
    return prefix


def get_bars(symbol, tf, n=500):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No bars: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]].copy()


def calc_atr(df, period=ATR_PERIOD):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr_percentile(atr_series, window=100):
    return atr_series.rolling(window).rank(pct=True)



def get_htf_levels(symbol: str) -> dict:
    """
    Pull PDH, PDL, PWH, PWL, PMH, PML directly from MT5.
    Uses the fully completed prior bar on each timeframe (iloc[-2]).
    Called once per scan — 3 x 5 bars, negligible overhead.
    """
    empty = {"pdh": 0.0, "pdl": 0.0,
             "pwh": 0.0, "pwl": 0.0,
             "pmh": 0.0, "pml": 0.0}
    try:
        d1 = get_bars(symbol, TIMEFRAME_D1,  n=5)
        w1 = get_bars(symbol, TIMEFRAME_W1,  n=5)
        mn = get_bars(symbol, TIMEFRAME_MN1, n=5)
        levels = {
            "pdh": float(d1["high"].iloc[-2]),
            "pdl": float(d1["low"].iloc[-2]),
            "pwh": float(w1["high"].iloc[-2]),
            "pwl": float(w1["low"].iloc[-2]),
            "pmh": float(mn["high"].iloc[-2]),
            "pml": float(mn["low"].iloc[-2]),
        }
        log.info(f"HTF levels — PDH:{levels['pdh']:.0f} PDL:{levels['pdl']:.0f} "
                 f"PWH:{levels['pwh']:.0f} PWL:{levels['pwl']:.0f} "
                 f"PMH:{levels['pmh']:.0f} PML:{levels['pml']:.0f}")
        return levels
    except Exception as e:
        log.warning(f"get_htf_levels failed: {e} — using zeros")
        return empty


def htf_features_for_level(level: float, atr_val: float,
                             htf_levels: dict) -> dict:
    """
    Identical to levels_ml_win.py — centralised HTF proximity feature set.
    Must stay in sync with levels_ml_win.py htf_features_for_level.
    """
    if atr_val <= 0:
        return {k: 0 for k in [
            "dist_pdh", "dist_pdl", "dist_pwh", "dist_pwl",
            "dist_pmh", "dist_pml", "htf_support_stack",
            "dist_nearest_htf_support", "dist_nearest_htf_resistance",
            "overhead_resistance_tight", "htf_any_confluence",
        ]}

    pdh = htf_levels.get("pdh", 0.0)
    pdl = htf_levels.get("pdl", 0.0)
    pwh = htf_levels.get("pwh", 0.0)
    pwl = htf_levels.get("pwl", 0.0)
    pmh = htf_levels.get("pmh", 0.0)
    pml = htf_levels.get("pml", 0.0)

    dist_pdh = (pdh - level) / atr_val if pdh > 0 else 99.0
    dist_pdl = (level - pdl) / atr_val if pdl > 0 else 99.0
    dist_pwh = (pwh - level) / atr_val if pwh > 0 else 99.0
    dist_pwl = (level - pwl) / atr_val if pwl > 0 else 99.0
    dist_pmh = (pmh - level) / atr_val if pmh > 0 else 99.0
    dist_pml = (level - pml) / atr_val if pml > 0 else 99.0

    htf_support_stack = sum([
        1 for d in [dist_pdl, dist_pwl, dist_pml]
        if abs(d) <= HTF_CONFLUENCE_THRESH
    ])

    support_dists = [d for d in [dist_pdl, dist_pwl, dist_pml] if 0 <= d]
    dist_nearest_htf_support = min(support_dists) if support_dists else 99.0

    resist_dists = [d for d in [dist_pdh, dist_pwh, dist_pmh] if 0 < d]
    dist_nearest_htf_resistance = min(resist_dists) if resist_dists else 99.0

    overhead_resistance_tight = int(dist_nearest_htf_resistance < HTF_OVERHEAD_TIGHT_THRESH)
    htf_any_confluence = int(any(
        abs(d) <= HTF_ANY_CONFLUENCE_THRESH for d in [dist_pdl, dist_pwl, dist_pml]
    ))

    return {
        "dist_pdh":                    dist_pdh,
        "dist_pdl":                    dist_pdl,
        "dist_pwh":                    dist_pwh,
        "dist_pwl":                    dist_pwl,
        "dist_pmh":                    dist_pmh,
        "dist_pml":                    dist_pml,
        "htf_support_stack":           htf_support_stack,
        "dist_nearest_htf_support":    dist_nearest_htf_support,
        "dist_nearest_htf_resistance": dist_nearest_htf_resistance,
        "overhead_resistance_tight":   overhead_resistance_tight,
        "htf_any_confluence":          htf_any_confluence,
    }


def find_swing_lows(df):
    lows     = df["low"].values
    atr_vals = calc_atr(df).values
    swings   = []
    for i in range(SWING_LOOKBACK, len(df) - SWING_LOOKBACK):
        window = lows[i - SWING_LOOKBACK: i + SWING_LOOKBACK + 1]
        if lows[i] == window.min() and list(window).count(lows[i]) == 1:
            swings.append({
                "bar_index":        i,
                "timestamp":        df.index[i],
                "price":            lows[i],
                "atr_at_formation": atr_vals[i],
            })
    return pd.DataFrame(swings) if swings else pd.DataFrame(
        columns=["bar_index", "timestamp", "price", "atr_at_formation"])


def get_daily_trend(symbol):
    """Returns trend string and values for display."""
    try:
        daily    = get_bars(symbol, TIMEFRAME_D1, n=150)
        ema100   = calc_ema(daily["close"], DAILY_EMA_PERIOD)
        last_c   = daily["close"].iloc[-1]
        last_e   = ema100.iloc[-1]
        bullish  = last_c > last_e
        return {
            "bullish":   bullish,
            "close":     last_c,
            "ema100":    last_e,
            "label":     "BULLISH ✅" if bullish else "BEARISH ❌",
        }
    except Exception as e:
        log.warning(f"Daily trend error: {e}")
        return {"bullish": None, "close": 0, "ema100": 0, "label": "UNKNOWN ⚠️"}


def get_active_levels(bars, current_price, atr_val):
    swings     = find_swing_lows(bars)
    closes_arr = bars["close"].values
    active     = []

    for _, sw in swings.iterrows():
        level    = sw["price"]
        origin_i = int(sw["bar_index"])
        zone_lo  = level - LEVEL_ZONE_ATR * sw["atr_at_formation"]

        if level > current_price + atr_val:
            continue

        dist_atr = (current_price - level) / atr_val
        if dist_atr > MAX_DIST_ATR or dist_atr < ALERT_MIN_DIST:
            continue

        subsequent = closes_arr[origin_i + 1:]
        if len(subsequent) > 0 and np.any(subsequent < zone_lo):
            continue

        touch_count = 0
        in_z = False
        zone_w = LEVEL_ZONE_ATR * sw["atr_at_formation"]
        for k in range(origin_i + 1, len(bars)):
            in_zone_k = abs(bars["low"].iloc[k] - level) <= zone_w * 2
            if in_zone_k and not in_z:
                touch_count += 1
            in_z = in_zone_k

        active.append({
            "price":            level,
            "origin_i":         origin_i,
            "atr_at_formation": sw["atr_at_formation"],
            "dist_atr":         dist_atr,
            "touch_count":      touch_count,
            "age_bars":         len(bars) - 1 - origin_i,
        })

    return sorted(active, key=lambda x: x["dist_atr"])


def build_feature_row(bars, htf_bars, level_info, feat_cols,
                       htf_levels=None):
    if htf_levels is None:
        htf_levels = {"pdh":0.0,"pdl":0.0,"pwh":0.0,"pwl":0.0,"pmh":0.0,"pml":0.0}
    i       = len(bars) - 1
    ts      = bars.index[i]
    close   = bars["close"].iloc[i]
    atr_s   = calc_atr(bars)
    atr_val = atr_s.iloc[i]
    if atr_val == 0:
        return None

    level    = level_info["price"]
    origin_i = level_info["origin_i"]

    body       = abs(bars["close"].iloc[i] - bars["open"].iloc[i])
    bar_low    = bars["low"].iloc[i]
    bar_high   = bars["high"].iloc[i]
    lower_wick = min(bars["open"].iloc[i], close) - bar_low
    upper_wick = bar_high - max(bars["open"].iloc[i], close)
    c_range    = bar_high - bar_low

    consec_red = 0
    for k in range(i - 1, max(0, i - 8), -1):
        if bars["close"].iloc[k] < bars["open"].iloc[k]:
            consec_red += 1
        else:
            break

    vol_ma    = bars["volume"].rolling(20).mean().iloc[i]
    vol_ratio = bars["volume"].iloc[i] / vol_ma if vol_ma > 0 else 1.0

    ref_highs = bars["high"].iloc[origin_i + 1: i]
    max_h     = ref_highs.max() if len(ref_highs) > 0 else level
    dep_h     = (max_h - level) / atr_val

    htf_ema20 = calc_ema(htf_bars["close"], 20)
    htf_ema50 = calc_ema(htf_bars["close"], 50)
    htf_atr   = calc_atr(htf_bars)
    htf_prior = htf_bars[htf_bars.index <= ts]
    if len(htf_prior) < 5:
        return None
    htf_c    = htf_prior["close"].iloc[-1]
    htf_e20  = htf_ema20.reindex(htf_prior.index).iloc[-1]
    htf_e50  = htf_ema50.reindex(htf_prior.index).iloc[-1]
    htf_a    = htf_atr.reindex(htf_prior.index).iloc[-1]
    htf_trend = 1 if htf_c > htf_e20 else -1
    htf_pct   = (htf_c - htf_e20) / htf_e20 if htf_e20 > 0 else 0
    htf_e20d  = abs(level - htf_e20) / htf_a if htf_a > 0 else 99
    htf_e50d  = abs(level - htf_e50) / htf_a if htf_a > 0 else 99
    htf_conf  = int(min(htf_e20d, htf_e50d) < 0.5)

    ema20  = calc_ema(bars["close"], 20).iloc[i]
    ema50  = calc_ema(bars["close"], 50).iloc[i]
    ema200 = calc_ema(bars["close"], 200).iloc[i]
    rsi    = calc_rsi(bars["close"]).iloc[i]
    atr_pv = calc_atr_percentile(atr_s).iloc[i]
    if np.isnan(atr_pv):
        atr_pv = 0.5

    hour    = ts.hour
    session = 0
    if 7  <= hour < 13: session = 1
    if 13 <= hour < 21: session = 2

    round_100  = round(level / 100) * 100
    dist_round = abs(level - round_100) / atr_val
    risk_proxy = abs(close - (level - SL_ATR * atr_val))

    feat = {
        "touch_count":         level_info["touch_count"],
        "level_age_bars":      min(level_info["age_bars"], 500),
        "departure_height":    dep_h,
        "origin_departure":    dep_h,
        "approach_drop_atr":   abs(bars["close"].iloc[max(0,i-5)] - close) / atr_val,
        "approach_consec_red": consec_red,
        "approach_vol_ratio":  vol_ratio,
        "close_above_level":   int(close > level),
        "wick_touched_level":  int(bar_low <= level + LEVEL_ZONE_ATR * atr_val),
        "wick_body_ratio":     lower_wick / body if body > 0 else 0,
        "close_pos_range":     (close - bar_low) / c_range if c_range > 0 else 0,
        "precision":           abs(bar_low - level) / atr_val,
        "body_atr":            body / atr_val,
        "rsi":                 rsi,
        "pct_from_ema20":      (close - ema20) / ema20,
        "pct_from_ema50":      (close - ema50) / ema50,
        "pct_from_ema200":     (close - ema200) / ema200,
        "atr_percentile":      atr_pv,
        "htf_trend":           htf_trend,
        "htf_pct_ema20":       htf_pct,
        "htf_confluence":      htf_conf,
        "session":             session,
        "hour":                hour,
        "day_of_week":         ts.dayofweek,
        "dist_round_number":   dist_round,
        "risk_atr":            risk_proxy / atr_val,
        **htf_features_for_level(level, atr_val, htf_levels),
    }

    return np.array([[feat.get(c, 0) for c in feat_cols]])


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL HISTORY LOG
# ─────────────────────────────────────────────────────────────────────────────

def load_signal_history() -> list:
    if os.path.exists(SIGNAL_LOG_PATH):
        try:
            with open(SIGNAL_LOG_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_signal_history(history: list):
    with open(SIGNAL_LOG_PATH, "w") as f:
        json.dump(history[-200:], f, indent=2)   # keep last 200


def get_last_n_signals(n=2) -> list:
    history = load_signal_history()
    return history[-n:] if history else []


def log_signal(level_price, score, dist_atr, touch_count,
               entry, sl, tp, atr_val, trend_label):
    """Append a new signal to the history log."""
    history = load_signal_history()
    history.append({
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "level":       round(level_price, 2),
        "score":       round(score, 4),
        "dist_atr":    round(dist_atr, 2),
        "touch_count": touch_count,
        "entry":       round(entry, 2),
        "sl":          round(sl, 2),
        "tp":          round(tp, 2),
        "atr":         round(atr_val, 2),
        "trend":       trend_label,
        "outcome":     None,   # filled in later manually or via update script
    })
    save_signal_history(history)


# ─────────────────────────────────────────────────────────────────────────────
# DUPLICATE SUPPRESSION
# ─────────────────────────────────────────────────────────────────────────────

def load_last_alert() -> dict:
    if os.path.exists(LAST_ALERT_PATH):
        try:
            with open(LAST_ALERT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_last_alert(alerted_levels: dict):
    with open(LAST_ALERT_PATH, "w") as f:
        json.dump(alerted_levels, f, indent=2)


def is_duplicate(level_price: float, score: float,
                 last_alert: dict, atr_val: float) -> bool:
    """
    Suppress re-alerting the same level within SUPPRESS_HOURS
    unless score has improved by more than 5%.
    """
    key = str(round(level_price / (atr_val * LEVEL_ZONE_ATR)))
    if key not in last_alert:
        return False
    prev      = last_alert[key]
    prev_time = datetime.fromisoformat(prev["timestamp"])
    now       = datetime.now(timezone.utc)
    hours_ago = (now - prev_time).total_seconds() / 3600
    if hours_ago < SUPPRESS_HOURS:
        if score < prev["score"] + 0.05:
            return True
    return False


def mark_alerted(level_price, score, last_alert, atr_val):
    key = str(round(level_price / (atr_val * LEVEL_ZONE_ATR)))
    last_alert[key] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score":     score,
        "level":     level_price,
    }
    return last_alert


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────────────────────────────────────

def score_level_ensemble(ensemble_data, X_row, min_score):
    """Score a level with all ensemble models. Returns dict of scores + consensus."""
    if ensemble_data is None:
        return None
    scores = {}
    for name, m_info in ensemble_data["models"].items():
        try:
            scores[name] = float(m_info["model"].predict_proba(X_row)[0, 1])
        except Exception:
            scores[name] = 0.0
    n_models  = len(scores)
    n_agree   = sum(1 for s in scores.values() if s >= min_score)
    avg_score = float(np.mean(list(scores.values())))
    return {
        "scores":    scores,
        "n_agree":   n_agree,
        "n_models":  n_models,
        "avg_score": avg_score,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUTO TRADE — Position gate + order placement
# ─────────────────────────────────────────────────────────────────────────────

def any_position_open() -> bool:
    """
    BULLETPROOF GATE: Check ALL open positions in the entire MT5 terminal.
    Returns True if ANY position exists regardless of symbol or magic number.
    This is checked FIRST before any order is placed — no exceptions.
    Called twice: once before scoring, once immediately before order_send.
    """
    try:
        positions = mt5.positions_get()
        if positions is None:
            # None = error querying MT5 — treat conservatively as position exists
            log.warning(f"positions_get() returned None: {mt5.last_error()}")
            return True
        count = len(positions)
        if count > 0:
            syms = [p.symbol for p in positions]
            log.info(f"GATE BLOCKED: {count} open position(s) in terminal: {syms}")
            return True
        return False
    except Exception as e:
        log.warning(f"Position check error: {e} — treating as position open")
        return True  # fail safe — never place order on error


def any_pending_order_exists(symbol: str) -> bool:
    """Check if a bot-placed pending order already exists for this symbol."""
    try:
        orders = mt5.orders_get(symbol=symbol)
        if orders is None:
            # MT5 error — treat conservatively as order may exist
            log.warning(f"orders_get() returned None: {mt5.last_error()} — treating as pending exists")
            return True
        bot_orders = [o for o in orders if o.magic == AUTO_TRADE_MAGIC]
        if bot_orders:
            log.info(f"Pending bot order exists for {symbol} — skipping")
            return True
        return False
    except Exception as e:
        log.warning(f"Pending order check error: {e}")
        return True  # fail safe


def load_order_state() -> dict:
    if os.path.exists(ORDER_STATE_PATH):
        try:
            with open(ORDER_STATE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"pending_ticket": None, "placed_at": None,
            "level": None, "sl": None, "tp": None,
            "runs_since_placed": 0}


def save_order_state(state: dict):
    with open(ORDER_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def clear_order_state():
    save_order_state({"pending_ticket": None, "placed_at": None,
                      "level": None, "sl": None, "tp": None,
                      "runs_since_placed": 0})


def check_and_cleanup_pending(symbol: str) -> bool:
    """
    Check if our previously placed order is still pending.
    If it filled → clear state, return False (no pending).
    If it expired (>3 bars) → cancel it, clear state, return False.
    Returns True if order is still pending and valid.
    """
    state = load_order_state()
    if state.get("pending_ticket") is None:
        return False

    ticket = state["pending_ticket"]

    # Check if order still exists — query MT5 first before modifying state
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        # MT5 query failed — treat conservatively, keep state, try next run
        log.warning(f"orders_get() returned None for {symbol}: {mt5.last_error()} "
                    f"— keeping order state, will retry next run")
        return True

    tickets = [o.ticket for o in orders]

    if ticket not in tickets:
        # Order gone — either filled or externally cancelled
        positions = mt5.positions_get(symbol=symbol)
        filled = [p for p in (positions or []) if p.magic == AUTO_TRADE_MAGIC]
        if filled:
            log.info(f"✅ Order #{ticket} filled → position open")
        else:
            log.info(f"Order #{ticket} no longer pending (cancelled/expired)")
        clear_order_state()
        return False

    # Order confirmed still pending — safe to increment counter
    state["runs_since_placed"] = state.get("runs_since_placed", 0) + 1

    # ── Session-based expiry — cancel at NY close (21:00 UTC) ────────────────
    now_utc    = datetime.now(timezone.utc)
    placed_at  = datetime.fromisoformat(state["placed_at"])
    # Ensure placed_at is timezone-aware
    if placed_at.tzinfo is None:
        placed_at = placed_at.replace(tzinfo=timezone.utc)
    placed_date      = placed_at.date()
    today            = now_utc.date()
    past_ny_close    = now_utc.hour >= NY_SESSION_END_UTC
    session_expired  = (placed_date == today and past_ny_close) or                        (placed_date < today)   # order from a previous day — always cancel
    run_backstop     = state["runs_since_placed"] >= ORDER_EXPIRY_RUNS

    if session_expired or run_backstop:
        reason = "NY session closed (21:00 UTC)" if session_expired else                  f"run backstop ({ORDER_EXPIRY_RUNS} runs)"
        log.info(f"[EXPIRED] Order #{ticket} cancelled — {reason}")
        req = {"action": mt5.TRADE_ACTION_REMOVE, "order": ticket}
        try:
            result = mt5.order_send(req)
        except Exception as _ce:
            log.warning(f"Cancel exception for #{ticket}: {_ce}")
            result = None
        if result is None:
            log.warning(f"Cancel order_send returned None for #{ticket}: {mt5.last_error()}")
        elif result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"Order #{ticket} cancelled successfully")
            send_telegram(f"[EXPIRED] Limit order #{ticket} cancelled — {reason}")
        else:
            log.warning(f"Cancel failed for #{ticket}: retcode={result.retcode}")
        clear_order_state()
        return False

    save_order_state(state)
    now_utc = datetime.now(timezone.utc)
    hrs_to_close = max(0, NY_SESSION_END_UTC - now_utc.hour)
    log.info(f"Pending order #{ticket} still active | "
             f"run {state['runs_since_placed']} | "
             f"NY close in ~{hrs_to_close}h ({NY_SESSION_END_UTC}:00 UTC)")
    return True


def place_limit_order(symbol: str, level: float,
                       sl: float, tp: float) -> int:
    """
    Place a buy limit order via MT5 (Windows native).
    Returns ticket number on success, -1 on failure.

    FINAL GATE: checks for open positions one last time immediately
    before sending the order. If anything is open — aborts.
    """
    # ── FINAL bulletproof position check — last line of defense ──────────────
    if any_position_open():
        log.warning("FINAL GATE BLOCKED: position detected immediately "
                    "before order_send — aborting")
        return -1

    if any_pending_order_exists(symbol):
        log.warning("FINAL GATE BLOCKED: pending order exists — aborting")
        return -1

    # Get symbol info for tick size
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error(f"Symbol info not found for {symbol}")
        return -1

    tick = info.trade_tick_size
    if tick <= 0:
        tick = 0.25  # MNQ default

    # Round to tick size
    def round_tick(price):
        return round(round(price / tick) * tick, info.digits)

    price_r = round_tick(level)
    sl_r    = round_tick(sl)
    tp_r    = round_tick(tp)

    request = {
        "action":        mt5.TRADE_ACTION_PENDING,
        "symbol":        symbol,
        "volume":        float(AUTO_TRADE_VOLUME),
        "type":          mt5.ORDER_TYPE_BUY_LIMIT,
        "price":         price_r,
        "sl":            sl_r,
        "tp":            tp_r,
        "deviation":     10,
        "magic":         AUTO_TRADE_MAGIC,
        "comment":       "nq_alpha_bot",
        "type_time":     mt5.ORDER_TIME_GTC,
        "type_filling":  mt5.ORDER_FILLING_RETURN,  # correct policy for pending limit orders at AMP
    }

    log.info(f"Placing limit order: {symbol} BUY LIMIT @ {price_r} "
             f"SL={sl_r} TP={tp_r} vol={AUTO_TRADE_VOLUME}")

    result = mt5.order_send(request)

    if result is None:
        log.error(f"order_send returned None: {mt5.last_error()}")
        return -1

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(f"Order failed | retcode={result.retcode} | "
                  f"comment={getattr(result, 'comment', 'n/a')}")
        return -1

    ticket = result.order
    log.info(f"✅ ORDER PLACED | #{ticket} | {symbol} BUY LIMIT @ {price_r} | "
             f"SL={sl_r} TP={tp_r}")
    return ticket


def send_telegram(message: str) -> bool:
    if not TELEGRAM_TOKEN or len(TELEGRAM_TOKEN) < 20:
        log.warning("Telegram not configured — printing to console only")
        print("\n" + "=" * 50)
        print(message)
        print("=" * 50)
        return True
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       message,
            "parse_mode": "HTML",
        }, timeout=10)
        if resp.status_code == 200:
            log.info("Telegram alert sent ✅")
            return True
        else:
            log.error(f"Telegram error: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        log.error(f"Telegram send failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# FORMAT ALERT MESSAGE
# ─────────────────────────────────────────────────────────────────────────────

def format_alert(strong_levels, watch_levels, trend_info,
                 current_price, atr_val, last_signals,
                 symbol="@MNQ") -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"🎯 <b>LEVEL ALERT — {symbol} {TF_LABEL}</b>")
    lines.append(f"🕐 {now}")
    lines.append(f"💲 Price: <b>{current_price:,.2f}</b>  |  ATR: {atr_val:.1f} pts")
    lines.append(f"📈 Daily trend: <b>{trend_info['label']}</b> "
                 f"(close {trend_info['close']:,.0f} / EMA100 {trend_info['ema100']:,.0f})")
    lines.append("")

    if strong_levels:
        lines.append("🔥 <b>STRONG (75%+)</b>")
        for lv in strong_levels:
            sl  = lv["price"] - SL_ATR * atr_val
            tp  = lv["price"] + TP_R * abs(lv["price"] - sl)
            rsk = abs(lv["price"] - sl)
            ens = lv.get("ensemble")
            if ens:
                n_a = ens["n_agree"]
                n_m = ens["n_models"]
                bar = "█" * n_a + "░" * (n_m - n_a)
                consensus_str = f" | [{bar}] {n_a}/{n_m}"
                score_detail = "  ".join(
                    f"{k[:3]}:{v:.0%}" for k,v in ens["scores"].items())
            else:
                consensus_str = ""
                score_detail  = f"score {lv['score']:.0%}"
            stack = lv.get("htf_support_stack", 0)
            conf_tag = (" 🏆TRIPLE" if stack >= 3 else
                        " ⭐DOUBLE" if stack == 2 else
                        " ✦CONF" if stack == 1 else "")
            overhead_tag = " ⚠️OHR" if lv.get("overhead_resistance_tight", 0) else ""
            lines.append(
                f"  📍 <b>{lv['price']:,.2f}</b> | avg {lv['score']:.0%}"
                f"{consensus_str}{conf_tag}{overhead_tag} | {lv['dist_atr']:.1f} ATR | "
                f"{lv['touch_count']} touches | age {lv.get('age_bars', '?')}bars"
            )
            if ens:
                lines.append(f"     {score_detail}")
            lines.append(
                f"     Entry: {lv['price']:,.2f}  SL: {sl:,.2f}  TP: {tp:,.2f}"
            )
            lines.append(
                f"     Risk: {rsk:.0f}pts (${rsk*2:.0f})  "
                f"Target: {rsk*TP_R:.0f}pts (${rsk*TP_R*2:.0f})"
            )
        lines.append("")

    if watch_levels:
        lines.append("👀 <b>WATCH (60–75%)</b>")
        for lv in watch_levels:
            sl  = lv["price"] - SL_ATR * atr_val
            tp  = lv["price"] + TP_R * abs(lv["price"] - sl)
            ens = lv.get("ensemble")
            if ens:
                n_a = ens["n_agree"]
                n_m = ens["n_models"]
                bar = "█" * n_a + "░" * (n_m - n_a)
                consensus_str = f" | [{bar}] {n_a}/{n_m}"
            else:
                consensus_str = ""
            w_stack = lv.get("htf_support_stack", 0)
            w_conf = (" ⭐DOUBLE" if w_stack >= 2 else " ✦CONF" if w_stack == 1 else "")
            w_ohr = " ⚠️OHR" if lv.get("overhead_resistance_tight", 0) else ""
            lines.append(
                f"  📍 {lv['price']:,.2f} | avg {lv['score']:.0%}"
                f"{consensus_str}{w_conf}{w_ohr} | {lv['dist_atr']:.1f} ATR | "
                f"{lv['touch_count']} touches"
            )
            lines.append(
                f"     Entry: {lv['price']:,.2f}  SL: {sl:,.2f}  TP: {tp:,.2f}"
            )
        lines.append("")

    # Last N signals from history log
    if last_signals:
        lines.append("📋 <b>Last signals alerted:</b>")
        for s in reversed(last_signals):
            ts      = s["timestamp"][:16].replace("T", " ")
            outcome = s.get("outcome")
            if outcome is None:
                out_str = "⏳ pending"
            elif outcome == "win":
                out_str = "✅ win"
            elif outcome == "loss":
                out_str = "❌ loss"
            else:
                out_str = f"📊 {outcome}"
            lines.append(
                f"  {ts} | {s['level']:,.2f} | {s['score']:.0%} | {out_str}"
            )
        lines.append("")

    lines.append(f"⚙️ Threshold: strong≥75% / watch≥60% | "
                 f"Max dist: {MAX_DIST_ATR} ATR")

    return "\n".join(lines)


def format_no_signal(trend_info, current_price, atr_val, symbol="@MNQ") -> str:
    """Compact heartbeat log when nothing is actionable — not sent to Telegram."""
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    return (f"[{now}] {symbol} @ {current_price:,.2f} | "
            f"ATR {atr_val:.1f} | trend {trend_info['label']} | "
            f"no levels within {CLOSE_LEVEL_ATR} ATR")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCAN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info(f"=== SCAN START | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}")
        send_telegram(f"⚠️ levels_alert.py ERROR\nModel file not found: {MODEL_PATH}")
        return

    saved     = joblib.load(MODEL_PATH)
    model     = saved["model"]
    feat_cols = saved["features"]

    # Load ensemble models if available (path defined in config)
    ensemble_data = None
    if os.path.exists(ENSEMBLE_PATH):
        try:
            ensemble_data = joblib.load(ENSEMBLE_PATH)
            log.info(f"Ensemble models loaded: {ensemble_data['model_names']}")
        except Exception as e:
            log.warning(f"Could not load ensemble models: {e}")

    # ── Connect MT5 ───────────────────────────────────────────────────────────
    try:
        if not mt5.initialize():
            raise RuntimeError(f"MT5 connect failed: {mt5.last_error()}")
        log.info("MT5 connected")
    except Exception as e:
        log.error(f"MT5 connection error: {e}")
        send_telegram(f"⚠️ levels_alert ERROR\nMT5 connection failed: {e}")
        return

    try:
        # Auto-detect front month symbol
        symbol = get_front_month_symbol(SYMBOL_PREFIX)
        log.info(f"Symbol: {symbol}")

        # ── GATE 1: Check for open positions BEFORE doing anything else ───────
        # Outermost guard — skip entire scan if any position is open.
        # Runs regardless of AUTO_TRADE so alerts are never sent into a live position.
        if any_position_open():
            log.info("GATE 1: position open in terminal — skipping scan entirely")
            mt5.shutdown()
            log.info("=== SCAN END — position open, no scan ===\n")
            return

        # ── GATE 2: Check + cleanup pending order state (always runs) ─────────
        # Even when AUTO_TRADE=False we clean up stale order_state.json
        # so it does not block future runs when AUTO_TRADE is re-enabled.
        has_pending = check_and_cleanup_pending(symbol)
        if has_pending and AUTO_TRADE:
            log.info("GATE 2: pending order still active — skipping scan")
            mt5.shutdown()
            log.info("=== SCAN END — pending order exists ===\n")
            return

        tf  = TIMEFRAME_MAP[TF_LABEL]
        htf = HTF_MAP.get(tf, TIMEFRAME_H4)

        # ── Pull bars ─────────────────────────────────────────────────────────
        bars       = get_bars(symbol, tf,  n=600)
        htf_bars   = get_bars(symbol, htf, n=200)
        htf_levels = get_htf_levels(symbol)

        current_price = bars["close"].iloc[-1]
        atr_val       = calc_atr(bars).iloc[-1]

        log.info(f"Price: {current_price:,.2f} | ATR: {atr_val:.2f}")

        # ── Daily trend ───────────────────────────────────────────────────────
        trend_info = get_daily_trend(symbol)
        log.info(f"Daily trend: {trend_info['label']}")

        # ── Find + score levels ───────────────────────────────────────────────
        levels = get_active_levels(bars, current_price, atr_val)
        log.info(f"Active levels in range: {len(levels)}")

        last_alert = load_last_alert()
        strong_levels = []
        watch_levels  = []

        for lv in levels:
            try:
                X = build_feature_row(bars, htf_bars, lv, feat_cols,
                                       htf_levels=htf_levels)
                if X is None:
                    continue
                score = float(model.predict_proba(X)[0, 1])
                lv["score"] = score

                # Store HTF features on lv for alert formatting — single call
                try:
                    _hf = htf_features_for_level(lv["price"], atr_val, htf_levels)
                    lv["htf_support_stack"]        = int(_hf.get("htf_support_stack", 0))
                    lv["overhead_resistance_tight"] = int(_hf.get("overhead_resistance_tight", 0))
                except Exception:
                    lv["htf_support_stack"]        = 0
                    lv["overhead_resistance_tight"] = 0

                log.info(f"  Level {lv['price']:,.2f} | "
                         f"dist={lv['dist_atr']:.2f} ATR | "
                         f"touches={lv['touch_count']} | "
                         f"age={lv['age_bars']} | "
                         f"score={score:.1%}")

                if score < SCORE_WATCH:
                    continue

                # Check duplicate suppression
                if is_duplicate(lv["price"], score, last_alert, atr_val):
                    log.info(f"  → suppressed (alerted recently)")
                    continue

                if score >= SCORE_STRONG:
                    strong_levels.append(lv)
                else:
                    watch_levels.append(lv)

            except Exception as e:
                log.warning(f"Score error for level {lv['price']}: {e}")

        # ── Decide whether to send alert ──────────────────────────────────────
        # Only alert if at least one level is within CLOSE_LEVEL_ATR
        all_levels    = strong_levels + watch_levels
        close_enough  = any(lv["dist_atr"] <= CLOSE_LEVEL_ATR
                            for lv in all_levels)

        if not all_levels or not close_enough:
            log.info(format_no_signal(trend_info, current_price, atr_val, symbol))
            log.info("=== SCAN END — no alert sent ===\n")
            return

        # ── Only alert if trend is bullish (long-only strategy) ───────────────
        if trend_info["bullish"] is False:
            log.info("Daily trend bearish — levels found but not alerting "
                     "(long-only strategy)")
            log.info("=== SCAN END — bearish trend, no alert ===\n")
            return

        # ── Log signals to history — STRONG only (these are the auto-traded signals)
        # WATCH levels are still alerted via Telegram but not logged to history
        for lv in strong_levels:
            sl = lv["price"] - SL_ATR * atr_val
            tp = lv["price"] + TP_R * abs(lv["price"] - sl)
            log_signal(
                level_price  = lv["price"],
                score        = lv["score"],
                dist_atr     = lv["dist_atr"],
                touch_count  = lv["touch_count"],
                entry        = lv["price"],
                sl           = sl,
                tp           = tp,
                atr_val      = atr_val,
                trend_label  = trend_info["label"],
            )
        # Mark all alerted levels (both strong and watch) for duplicate suppression
        for lv in strong_levels + watch_levels:
            last_alert = mark_alerted(lv["price"], lv["score"],
                                      last_alert, atr_val)

        save_last_alert(last_alert)

        # ── Get last 2 signals from history for context ───────────────────────
        last_signals = get_last_n_signals(n=2)

        # ── Format + send alert ───────────────────────────────────────────────
        message = format_alert(
            strong_levels  = strong_levels,
            watch_levels   = watch_levels,
            trend_info     = trend_info,
            current_price  = current_price,
            atr_val        = atr_val,
            last_signals   = last_signals,
            symbol         = symbol,
        )

        send_telegram(message)

        log.info(f"Alert sent | strong={len(strong_levels)} watch={len(watch_levels)}")

        # ── AUTO TRADE: place limit order if enabled ──────────────────────────
        if AUTO_TRADE and strong_levels:
            # Only auto-trade the BEST strong level (highest score)
            best = max(strong_levels, key=lambda x: x["score"])

            if best["score"] >= AUTO_TRADE_MIN_SCORE:
                # GATE 3: Final position check immediately before order
                if any_position_open():
                    log.warning("AUTO_TRADE GATE 3: position opened between "
                                "scan and order — aborting")
                    send_telegram(
                        f"⚠️ Auto-trade aborted\n"
                        f"Position detected before order placement\n"
                        f"Level {best['price']:,.2f} — check manually"
                    )
                else:
                    sl     = best["price"] - SL_ATR * atr_val
                    tp     = best["price"] + TP_R * abs(best["price"] - sl)
                    risk   = abs(best["price"] - sl)

                    ticket = place_limit_order(
                        symbol = symbol,
                        level  = best["price"],
                        sl     = sl,
                        tp     = tp,
                    )

                    if ticket > 0:
                        # Save order state for tracking across cron runs
                        save_order_state({
                            "pending_ticket":    ticket,
                            "placed_at":         datetime.now(timezone.utc).isoformat(),
                            "level":             best["price"],
                            "sl":                sl,
                            "tp":                tp,
                            "runs_since_placed": 0,
                        })

                        order_msg = (
                            f"\n✅ <b>LIMIT ORDER PLACED</b>\n"
                            f"   Ticket: #{ticket}\n"
                            f"   Entry : {best['price']:,.2f}\n"
                            f"   SL    : {sl:,.2f}  ({risk:.0f}pts / ${risk*2:.0f})\n"
                            f"   TP    : {tp:,.2f}  ({risk*TP_R:.0f}pts / ${risk*TP_R*2:.0f})\n"
                            f"   Status: Waiting for fill (expires at NY close 21:00 UTC)"
                        )
                        send_telegram(order_msg)
                        log.info(f"AUTO_TRADE: order placed #{ticket} @ {best['price']:.2f}")

                    else:
                        log.error("AUTO_TRADE: order_send failed")
                        send_telegram(
                            f"❌ Auto-trade order FAILED\n"
                            f"Level {best['price']:,.2f} — place manually"
                        )
            else:
                log.info(f"AUTO_TRADE: best score {best['score']:.1%} < "
                         f"{AUTO_TRADE_MIN_SCORE:.0%} threshold — no order placed")

    except Exception as e:
        log.error(f"Scan error: {e}")
        log.debug(traceback.format_exc())
        send_telegram(f"⚠️ levels_alert ERROR\n{str(e)[:200]}")

    finally:
        mt5.shutdown()
        log.info("=== SCAN END ===\n")


# ─────────────────────────────────────────────────────────────────────────────
# OUTCOME UPDATER — run manually to mark past signals as win/loss
# ─────────────────────────────────────────────────────────────────────────────

def update_outcome(level_price: float, outcome: str):
    """
    Manually mark the outcome of a past signal.
    outcome: 'win', 'loss', or 'cancelled'

    Usage:
      python levels_alert.py --update 29743 win
      python levels_alert.py --update 29743 loss
    """
    history = load_signal_history()
    updated = 0
    for sig in reversed(history):
        if abs(sig["level"] - level_price) < 5 and sig["outcome"] is None:
            sig["outcome"] = outcome
            updated += 1
            print(f"Updated: {sig['timestamp']} | {sig['level']} → {outcome}")
            break
    if updated == 0:
        print(f"No pending signal found near {level_price}")
    else:
        save_signal_history(history)


def show_history(n=10):
    """Print last N signals from history log."""
    history = load_signal_history()
    if not history:
        print("No signal history yet.")
        return

    recent = history[-n:]
    print(f"\n{'='*75}")
    print(f"LAST {len(recent)} SIGNALS FROM HISTORY LOG")
    print(f"{'='*75}")
    print(f"  {'Timestamp':<20} {'Level':>10} {'Score':>7} "
          f"{'Dist':>6} {'Touches':>8} {'Entry':>10} "
          f"{'SL':>10} {'TP':>10} {'Outcome':<12}")
    print("  " + "-" * 95)
    for s in reversed(recent):
        out = s.get("outcome") or "pending"
        mark = "✅" if out == "win" else "❌" if out == "loss" else "⏳"
        print(f"  {s['timestamp'][:16]:<20} {s['level']:>10,.2f} "
              f"{s['score']:>7.1%} {s['dist_atr']:>6.2f} "
              f"{s['touch_count']:>8} {s['entry']:>10,.2f} "
              f"{s['sl']:>10,.2f} {s['tp']:>10,.2f} "
              f"{mark} {out:<10}")
    print()

    wins     = sum(1 for s in history if s.get("outcome") == "win")
    losses   = sum(1 for s in history if s.get("outcome") == "loss")
    pending  = sum(1 for s in history if s.get("outcome") is None)
    total_r  = sum(
        (TP_R if s.get("outcome") == "win" else -1.0)
        for s in history if s.get("outcome") in ("win", "loss")
    )
    resolved = wins + losses
    wr       = wins / resolved if resolved > 0 else 0

    print(f"  Resolved: {resolved}  ({wins}W / {losses}L)  "
          f"WR: {wr:.1%}  Net R: {total_r:+.1f}R  Pending: {pending}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Levels Alert Scanner")
    parser.add_argument("--update", nargs=2, metavar=("LEVEL", "OUTCOME"),
                        help="Mark outcome of a past signal: --update 29743 win")
    parser.add_argument("--history", type=int, default=0,
                        help="Show last N signals from history: --history 10")
    parser.add_argument("--test", action="store_true",
                        help="Test Telegram connection only")
    args = parser.parse_args()

    if args.test:
        msg = (f"✅ levels_alert.py test message\n"
               f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
               f"Telegram connection working!")
        success = send_telegram(msg)
        print("Telegram test:", "OK" if success else "FAILED")

    elif args.update:
        level_price = float(args.update[0])
        outcome     = args.update[1].lower()
        if outcome not in ("win", "loss", "cancelled"):
            print("Outcome must be: win, loss, or cancelled")
        else:
            update_outcome(level_price, outcome)

    elif args.history:
        show_history(args.history)

    else:
        main()
