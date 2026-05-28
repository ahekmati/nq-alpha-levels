# =============================================================================
# supertrend_ml.py
# MNQ Daily EMA100 + RSI10 Regime Gate → H1 SuperTrend Entry
# Full ML Research Pipeline with Walk-Forward OOS Validation
# + Last 5 trades replay
# + Live execution engine with open-position guard
# + Full debug/status prints
# =============================================================================
#
# STRATEGY LOGIC (plain English):
# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DAILY REGIME GATE
#   • Is today's Daily close ABOVE the 100-period EMA?
#     → If NO  : no long trades allowed (market is in a downtrend)
#     → If YES : proceed to Step 2
#
# STEP 2 — DAILY RSI MOMENTUM GATE
#   • Is the 10-period Daily RSI inside the optimal band? (model finds best)
#     Best bands historically: (50-60) or (55-65)
#     → This ensures we only trade when momentum is bullish but NOT overbought
#     → If NO  : wait for RSI to pull back into range
#     → If YES : proceed to Step 3
#
# STEP 3 — H1 SUPERTREND FLIP ENTRY
#   • On the H1 chart, did SuperTrend just flip from RED → GREEN?
#     (i.e. previous bar direction=-1, current bar direction=+1)
#     → This is the actual entry trigger
#     → We buy the CLOSE of the flip bar (+ slippage)
#
# STEP 4 — ML CONSENSUS FILTER
#   • 4 classifiers (LogReg, RF, ExtraTrees, GradientBoosting) vote
#   • Average probability must be >= 0.60 to take the trade
#   • 3 regressors also estimate expected net_usd for confirmation
#
# STOP LOSS:
#   • Placed at min(entry - 1.0*ATR14, recent swing low, SuperTrend line)
#   • At minimum 0.25 points below entry
#
# TAKE PROFIT:
#   • trailing_st mode: TP trails with the SuperTrend line as it rises
#   • fixed_rr mode: fixed R:R multiple (1.5x, 2.0x, 2.5x)
#   • Max hold: 72 H1 bars (~3 trading days)
#
# EXIT CONDITIONS (whichever hits first):
#   1. Price touches stop loss → exit at SL - slippage
#   2. Price touches target    → exit at TP - slippage
#   3. Trailing ST flips back  → stop trails up, exits on reversal
#   4. Max hold bars reached   → exit at close
# ─────────────────────────────────────────────────────────────────────────────
#
# GOING LIVE:
#   Run in two modes:
#   MODE 1 — RESEARCH (default): full walk-forward train + OOS validation
#   MODE 2 — LIVE:  python supertrend_ml.py --live
#     • Fetches latest bars, rebuilds features, checks signal on last bar
#     • If signal fires AND ML confirms AND no open position → sends order
#     • Saves state to mnq_live_state.json for monitoring
#     • Run via cron every hour: 0 * * * * cd ~/projects/mt5-python && .venv/bin/python supertrend_ml.py --live
# =============================================================================

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import traceback
import urllib.request
import urllib.parse
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from mt5linux import MetaTrader5

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# DEBUG HELPERS
# =============================================================================

DEBUG = True   # set False to suppress verbose prints

def dbg(msg: str) -> None:
    if DEBUG:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  [DBG {ts}] {msg}")

def status(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"\n{'='*60}\n[STATUS {ts}]\n  {msg}\n{'='*60}")

def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")

def err(msg: str) -> None:
    print(f"  [ERROR] {msg}", file=sys.stderr)


# =============================================================================
# TELEGRAM ALERTS
# =============================================================================

def tg(msg: str) -> None:
    """
    Send a Telegram message. Silent no-op if TELEGRAM_ENABLED is False
    or if token/chat_id are not configured. Never crashes the main program.
    Uses only stdlib urllib — no extra pip install needed.
    """
    if not TELEGRAM_ENABLED:
        return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        warn("Telegram enabled but TOKEN or CHAT_ID not set — skipping alert")
        return
    try:
        url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id":    TELEGRAM_CHAT_ID,
            "text":       msg,
            "parse_mode": "HTML",
        }).encode()
        req  = urllib.request.Request(url, data=payload, method="POST")
        resp = urllib.request.urlopen(req, timeout=10)
        code = resp.getcode()
        if code != 200:
            warn(f"Telegram returned HTTP {code}")
        else:
            dbg(f"Telegram alert sent: {msg[:60]}...")
    except Exception as e:
        warn(f"Telegram alert failed (non-fatal): {e}")


def tg_signal_fired(sig: dict) -> None:
    """Alert when rule signal fires — before ML filter."""
    tg(
        f"📡 <b>ST-ML Signal</b>\n"
        f"Time: {sig['time']}\n"
        f"Direction: {sig['direction']}\n"
        f"Close: {sig['close']:.2f}\n"
        f"RSI10: {sig['rsi10']:.1f} band={sig['rsi_band']}\n"
        f"ATR14: {sig['atr14']:.2f}\n"
        f"Entry est: {sig['entry_estimate']:.2f}\n"
        f"SL est: {sig['sl_estimate']:.2f}\n"
        f"⏳ Checking ML filter..."
    )


def tg_ml_filtered(sig: dict) -> None:
    """Alert when ML rejects a rule signal."""
    tg(
        f"🚫 <b>ST-ML Signal FILTERED</b>\n"
        f"Time: {sig['time']}\n"
        f"ML consensus proba: {sig.get('consensus_proba', 0):.4f} "
        f"(threshold={CONSENSUS_PROBA_THRESHOLD})\n"
        f"Est net USD: ${sig.get('consensus_net_usd', 0):.0f}\n"
        f"No order sent."
    )


def tg_order_sent(sig: dict, order_result: dict) -> None:
    """Alert when an order is successfully filled."""
    tg(
        f"✅ <b>ST-ML ORDER FILLED</b>\n"
        f"Direction: {sig['direction']}\n"
        f"Symbol: {SYMBOL}\n"
        f"Ticket: {order_result.get('ticket', 'N/A')}\n"
        f"Fill price: {order_result.get('price', 0):.2f}\n"
        f"Volume: {order_result.get('volume', 0)}\n"
        f"SL: {order_result.get('sl', 0):.2f}\n"
        f"TP: {order_result.get('tp', 0):.2f if order_result.get('tp') else 'TRAIL'}\n"
        f"ML proba: {sig.get('consensus_proba', 0):.4f}\n"
        f"Est net USD: ${sig.get('consensus_net_usd', 0):.0f}"
    )


def tg_order_failed(sig: dict, order_result: dict) -> None:
    """Alert when an order fails to fill."""
    tg(
        f"❌ <b>ST-ML ORDER FAILED</b>\n"
        f"Direction: {sig['direction']}\n"
        f"Error: {order_result.get('error', 'unknown')}\n"
        f"Retcode: {order_result.get('retcode', 'N/A')}\n"
        f"Comment: {order_result.get('comment', '')}"
    )


def tg_blocked_position(positions: list) -> None:
    """Alert when blocked by an existing open position."""
    pos_str = "\n".join(
        f"  ticket={p.ticket} magic={p.magic} "
        f"{'BUY' if p.type==0 else 'SELL'} "
        f"profit={p.profit:.2f}"
        for p in positions
    )
    tg(
        f"🔒 <b>ST-ML BLOCKED</b> — open position exists\n"
        f"{pos_str}\n"
        f"No order sent."
    )


def tg_retrain_complete(metrics: dict, best_params: dict) -> None:
    """Alert when model retraining completes."""
    tg(
        f"🔄 <b>ST-ML Model Retrained</b>\n"
        f"AUC: {metrics.get('consensus_clf_auc', 0):.3f}\n"
        f"Precision: {metrics.get('consensus_precision', 0):.3f}\n"
        f"Recall: {metrics.get('consensus_recall', 0):.3f}\n"
        f"Trades: {metrics.get('n_trades', 0)}\n"
        f"Best params: rsi=({best_params.get('rsi_lo')},{best_params.get('rsi_hi')}) "
        f"ST({int(best_params.get('st_period',0))},{best_params.get('st_mult',0)}) "
        f"stop={best_params.get('stop_mult')} tp={best_params.get('tp_mode')}"
    )


def tg_error(context: str, error: str) -> None:
    """Alert on critical errors."""
    tg(f"🚨 <b>ST-ML ERROR</b>\n{context}\n{error}")


# =============================================================================
# USER CONFIG
# =============================================================================
SYMBOL = "@MNQ"
DAILY_TF_MIN = 1440
H1_TF_MIN    = 60

UTC_FROM = datetime(2018, 1, 1, tzinfo=timezone.utc)
UTC_TO   = datetime.now(timezone.utc)

DAILY_EMA_LEN = 100
DAILY_RSI_LEN = 10

RSI_BANDS = [
    (35, 45), (40, 50), (45, 55),
    (50, 60), (55, 65), (60, 70),
]

SUPER_PERIODS  = [10, 14]
SUPER_MULTS    = [2.5, 3.0]
STOP_ATR_MULTS = [1.0, 1.5, 2.0]
TP_RR_GRID     = [1.5, 2.0, 2.5]
TP_ATR_MULTS   = [2.0, 3.0]

MAX_HOLD_BARS             = 72
MIN_TRAIN_TRADES          = 25
CONSENSUS_PROBA_THRESHOLD = 0.60
QUALITY_USD_THRESHOLD     = 15.0
SLIPPAGE_POINTS           = 0.50
COMMISSION_PER_SIDE_USD   = 0.82
MNQ_DOLLARS_PER_POINT     = 2.0

# Live execution config
LIVE_LOT_SIZE        = 1          # number of MNQ contracts
LIVE_MAGIC           = 20240101   # unique EA magic number — change if running multiple EAs
LIVE_ORDER_COMMENT   = "ST_ML_v2"
LIVE_MAX_SPREAD_PTS  = 5.0        # refuse to trade if spread > this many points
LAST_N_TRADES        = 5          # how many recent historical trades to replay/print

# =============================================================================
# TELEGRAM CONFIG
# =============================================================================
# To set up:
#   1. Message @BotFather on Telegram → /newbot → copy the token
#   2. Message your bot once, then visit:
#      https://api.telegram.org/bot<TOKEN>/getUpdates
#      to find your chat_id
#   3. Fill in both values below
# Set TELEGRAM_ENABLED = False to disable all Telegram alerts
# =============================================================================
TELEGRAM_ENABLED  = True
TELEGRAM_TOKEN    = "8602513961:AAFzTS_2lSxza7soWiF3REUA6GewIgc8Grw"  # ← REVOKE AND REPLACE WITH NEW TOKEN FROM @BotFather
TELEGRAM_CHAT_ID  = "7902956948"

WF_TRAIN_BARS = 5000
WF_OOS_BARS   = 1000
WF_MIN_FOLDS  = 3

OUT_DIR        = Path(".")
LIVE_STATE_FILE  = OUT_DIR / "mnq_live_state.json"

# =============================================================================
# MODEL PERSISTENCE CONFIG
# =============================================================================
MODEL_FILE       = OUT_DIR / "mnq_model.pkl"   # saved model bundle
PARAMS_FILE      = OUT_DIR / "mnq_best_params.pkl"  # saved best params
MODEL_MAX_AGE_H  = 24    # hours before model is considered stale and retrains
                         # set to 0 to always retrain (same as old behaviour)


# =============================================================================
# MT5 HELPERS
# =============================================================================

def initialize_mt5() -> MetaTrader5:
    dbg("Initializing MT5 connection...")
    mt5 = MetaTrader5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    info = mt5.terminal_info()
    dbg(f"MT5 connected | build={info.build} connected={info.connected}")
    return mt5


def _tf_map(mt5: MetaTrader5, mins: int) -> int:
    return {
        1:    mt5.TIMEFRAME_M1,
        5:    mt5.TIMEFRAME_M5,
        15:   mt5.TIMEFRAME_M15,
        30:   mt5.TIMEFRAME_M30,
        60:   mt5.TIMEFRAME_H1,
        240:  mt5.TIMEFRAME_H4,
        1440: mt5.TIMEFRAME_D1,
    }[mins]


def fetch_rates(mt5: MetaTrader5, symbol: str, tf_mins: int,
                date_from: datetime, date_to: datetime) -> pd.DataFrame:
    dbg(f"Fetching {symbol} tf={tf_mins}min from {date_from.date()} to {date_to.date()}")
    rates = mt5.copy_rates_range(symbol, _tf_map(mt5, tf_mins), date_from, date_to)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {symbol} tf={tf_mins}. Error: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns=str.lower).sort_values("time").reset_index(drop=True)
    dbg(f"Fetched {len(df)} bars | first={df['time'].iloc[0]} last={df['time'].iloc[-1]}")
    return df


# =============================================================================
# OPEN POSITION GUARD
# =============================================================================

def get_open_positions(mt5: MetaTrader5, symbol: str, magic: int) -> list:
    """
    Returns ALL open positions for this symbol regardless of magic number.
    This ensures System 1 and System 2 cannot both be in a trade simultaneously.
    magic param kept for logging only — NOT used to filter.
    """
    dbg(f"Checking ALL open positions for {symbol} (any magic)...")
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        dbg(f"positions_get returned None (no positions or error): {mt5.last_error()}")
        return []
    # CRITICAL: return ALL positions on this symbol — not just our magic
    # This prevents System 2 opening while System 1 is in a trade and vice versa
    filtered = list(positions)
    own      = [p for p in positions if p.magic == magic]
    other    = [p for p in positions if p.magic != magic]
    dbg(f"Found {len(filtered)} total positions on {symbol} "
        f"({len(own)} ours magic={magic}, {len(other)} other systems)")
    return filtered


def has_open_position(mt5: MetaTrader5, symbol: str, magic: int) -> bool:
    """
    Returns True if there is ANY open position on this symbol.
    Blocks regardless of which system opened it — System 1 or System 2.
    This is the PRIMARY guard — no order is ever sent if this returns True.
    """
    positions = get_open_positions(mt5, symbol, magic)
    if positions:
        for p in positions:
            own_flag = " [OURS]" if p.magic == magic else " [OTHER SYSTEM]"
            warn(f"Open position exists{own_flag}: ticket={p.ticket} "
                 f"magic={p.magic} "
                 f"type={'BUY' if p.type==0 else 'SELL'} "
                 f"vol={p.volume} price={p.price_open:.2f} "
                 f"profit={p.profit:.2f}")
        return True
    dbg("No open positions on symbol — clear to trade")
    return False


def check_spread(mt5: MetaTrader5, symbol: str, max_spread: float) -> bool:
    """Returns True if spread is acceptable, False if too wide."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        warn(f"Could not get tick for {symbol}: {mt5.last_error()}")
        return False
    spread = tick.ask - tick.bid
    dbg(f"Current spread: {spread:.2f} pts (max allowed: {max_spread:.2f})")
    if spread > max_spread:
        warn(f"Spread {spread:.2f} exceeds max {max_spread:.2f} — skipping trade")
        return False
    return True


# =============================================================================
# ORDER EXECUTION
# =============================================================================

def send_market_order(mt5: MetaTrader5, symbol: str, direction: int,
                      lot: float, sl: float, tp: float,
                      magic: int, comment: str) -> dict:
    """
    Send a market order with SL and TP.
    direction: +1 = BUY, -1 = SELL
    Returns result dict with success flag and details.
    """
    dbg(f"Preparing order: {'BUY' if direction==1 else 'SELL'} {lot} {symbol} "
        f"SL={sl:.2f} TP={tp if not np.isnan(tp) else 'NONE'}")

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": f"Cannot get tick: {mt5.last_error()}"}

    info = mt5.symbol_info(symbol)
    if info is None:
        return {"success": False, "error": f"Cannot get symbol info: {mt5.last_error()}"}

    # Round SL/TP to tick size
    tick_size = info.trade_tick_size if info.trade_tick_size > 0 else 0.25
    def round_price(p):
        if p is None or np.isnan(p):
            return 0.0
        return round(round(p / tick_size) * tick_size, 2)

    price    = tick.ask if direction == 1 else tick.bid
    sl_clean = round_price(sl)
    tp_clean = round_price(tp) if not np.isnan(tp) else 0.0

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

    request = {
        "action":     mt5.TRADE_ACTION_DEAL,
        "symbol":     symbol,
        "volume":     float(lot),
        "type":       order_type,
        "price":      price,
        "sl":         sl_clean,
        "tp":         tp_clean,
        "deviation":  20,
        "magic":      magic,
        "comment":    comment,
        "type_time":  mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    dbg(f"Order request: price={price} sl={sl_clean} tp={tp_clean} "
        f"deviation=20 filling=IOC")

    result = mt5.order_send(request)

    if result is None:
        err_msg = f"order_send returned None: {mt5.last_error()}"
        err(err_msg)
        return {"success": False, "error": err_msg}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        err_msg = (f"Order failed: retcode={result.retcode} "
                   f"comment={result.comment}")
        err(err_msg)
        return {
            "success":  False,
            "retcode":  result.retcode,
            "error":    err_msg,
            "comment":  result.comment,
        }

    dbg(f"Order FILLED: ticket={result.order} price={result.price} "
        f"volume={result.volume}")
    return {
        "success": True,
        "ticket":  result.order,
        "price":   result.price,
        "volume":  result.volume,
        "sl":      sl_clean,
        "tp":      tp_clean,
    }


# =============================================================================
# INDICATORS
# =============================================================================

def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    up    = delta.clip(lower=0)
    down  = (-delta).clip(lower=0)
    rs    = (up.ewm(alpha=1/period, adjust=False).mean() /
             down.ewm(alpha=1/period, adjust=False).mean().replace(0, np.nan))
    return 100 - (100 / (1 + rs))


def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - pc).abs(),
        (df["low"]  - pc).abs(),
    ], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1/period, adjust=False).mean()


def _adx(df: pd.DataFrame, n: int = 14):
    up_move  = df["high"].diff()
    dn_move  = -df["low"].diff()
    plus_dm  = pd.Series(
        np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0),
        index=df.index)
    minus_dm = pd.Series(
        np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0),
        index=df.index)
    atrv     = _true_range(df).ewm(alpha=1/n, adjust=False).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.ewm(alpha=1/n,  adjust=False).mean() / atrv
    minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / atrv
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False).mean(), plus_di, minus_di


def _rolling_pctile(s: pd.Series, window: int = 252) -> pd.Series:
    return s.rolling(window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def _stochastic(df: pd.DataFrame, k: int = 14, d: int = 3):
    lo    = df["low"].rolling(k).min()
    hi    = df["high"].rolling(k).max()
    pct_k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return pct_k, pct_k.rolling(d).mean()


def _bollinger(df: pd.DataFrame, n: int = 20, std: float = 2.0):
    mid   = df["close"].rolling(n).mean()
    sigma = df["close"].rolling(n).std()
    upper = mid + std * sigma
    lower = mid - std * sigma
    bw    = (upper - lower) / mid.replace(0, np.nan) * 100
    pct_b = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, bw, pct_b


def _cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def _mfi(df: pd.DataFrame, n: int = 14) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["tick_volume"]
    pos = rmf.where(tp > tp.shift(1), 0.0)
    neg = rmf.where(tp < tp.shift(1), 0.0)
    mfr = pos.rolling(n).sum() / neg.rolling(n).sum().replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def _vwap_session(df: pd.DataFrame) -> pd.Series:
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["tick_volume"].replace(0, np.nan)
    td  = df["time"].dt.floor("D")
    return (tp * vol).groupby(td).cumsum() / vol.groupby(td).cumsum()


# =============================================================================
# VECTORIZED SUPERTREND (Numba JIT)
# =============================================================================

@njit
def _supertrend_numba(close, basic_upper, basic_lower):
    n           = len(close)
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st          = np.empty(n)
    direction   = np.empty(n, dtype=np.int32)

    direction[0] = 1
    st[0]        = final_lower[0]

    for i in range(1, n):
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]

        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]

        if st[i-1] == final_upper[i-1]:
            if close[i] <= final_upper[i]:
                direction[i] = -1
                st[i]        = final_upper[i]
            else:
                direction[i] = 1
                st[i]        = final_lower[i]
        else:
            if close[i] >= final_lower[i]:
                direction[i] = 1
                st[i]        = final_lower[i]
            else:
                direction[i] = -1
                st[i]        = final_upper[i]

    return st, direction, final_upper, final_lower


def add_supertrend(df: pd.DataFrame, period: int, mult: float) -> pd.DataFrame:
    atrv        = _atr(df, period).values
    hl2         = ((df["high"] + df["low"]) / 2.0).values
    basic_upper = hl2 + mult * atrv
    basic_lower = hl2 - mult * atrv

    st, direction, _, _ = _supertrend_numba(
        df["close"].values, basic_upper, basic_lower)

    tag   = f"st_{period}_{mult}"
    dir_s = pd.Series(direction, index=df.index)
    df[tag]              = st
    df[f"dir_{tag}"]     = direction
    df[f"flip_up_{tag}"] = ((dir_s.shift(1) == -1) & (dir_s == 1)).astype(int)
    df[f"flip_dn_{tag}"] = ((dir_s.shift(1) == 1)  & (dir_s == -1)).astype(int)
    return df


def add_all_supertrends(df: pd.DataFrame) -> pd.DataFrame:
    dbg(f"Adding {len(SUPER_PERIODS)*len(SUPER_MULTS)} SuperTrend variants...")
    for p, m in itertools.product(SUPER_PERIODS, SUPER_MULTS):
        df = add_supertrend(df, p, m)
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_daily_features(d: pd.DataFrame) -> pd.DataFrame:
    dbg("Computing daily EMA100, RSI10, ATR, ADX, Bollinger, CCI...")
    d = d.copy()
    d["ema100"]       = _ema(d["close"], DAILY_EMA_LEN)
    d["rsi10"]        = _rsi(d["close"], DAILY_RSI_LEN)
    d["atr14_d"]      = _atr(d, 14)
    d["above_ema100"] = (d["close"] > d["ema100"]).astype(int)
    d["ema_gap_pct"]  = (d["close"] / d["ema100"] - 1.0) * 100
    d["rsi_slope3"]   = d["rsi10"] - d["rsi10"].shift(3)
    d["rsi_slope5"]   = d["rsi10"] - d["rsi10"].shift(5)
    d["d_ret1"]       = d["close"].pct_change(1)
    d["d_ret5"]       = d["close"].pct_change(5)
    d["d_ret10"]      = d["close"].pct_change(10)
    d["d_range_pct"]  = (d["high"] - d["low"]) / d["close"] * 100
    d["d_atr_pct"]    = d["atr14_d"] / d["close"] * 100
    d["atr_pctile"]   = _rolling_pctile(d["d_atr_pct"], 252)
    adx, pdi, mdi     = _adx(d, 14)
    d["adx14_d"]      = adx
    d["pdi14_d"]      = pdi
    d["mdi14_d"]      = mdi
    _, _, _, bw, pb   = _bollinger(d, 20)
    d["bb_bw_d"]      = bw
    d["bb_pctb_d"]    = pb
    d["cci20_d"]      = _cci(d, 20)
    d["trade_date"]   = d["time"].dt.floor("D")
    keep = [
        "trade_date", "close", "ema100", "rsi10", "atr14_d",
        "above_ema100", "ema_gap_pct", "rsi_slope3", "rsi_slope5",
        "d_ret1", "d_ret5", "d_ret10", "d_range_pct", "d_atr_pct",
        "atr_pctile", "adx14_d", "pdi14_d", "mdi14_d",
        "bb_bw_d", "bb_pctb_d", "cci20_d",
    ]
    dbg(f"Daily features built: {len(d)} rows")
    return d[keep].rename(columns={"close": "daily_close"})


def build_h1_features(h: pd.DataFrame) -> pd.DataFrame:
    dbg("Computing H1 features: EMA, RSI, ATR, ADX, Stoch, BB, CCI, MFI, VWAP...")
    h = h.copy()
    h["trade_date"]    = h["time"].dt.floor("D")
    h["ret1"]          = h["close"].pct_change(1)
    h["ret3"]          = h["close"].pct_change(3)
    h["ret6"]          = h["close"].pct_change(6)
    h["ret12"]         = h["close"].pct_change(12)
    h["range_pct"]     = (h["high"] - h["low"]) / h["close"] * 100
    h["atr14"]         = _atr(h, 14)
    h["atr14_pct"]     = h["atr14"] / h["close"] * 100
    h["vol_ma20"]      = h["tick_volume"].rolling(20).mean()
    h["vol_ratio"]     = h["tick_volume"] / h["vol_ma20"].replace(0, np.nan)
    h["vol_ratio_6"]   = h["tick_volume"].rolling(6).mean() / h["vol_ma20"].replace(0, np.nan)
    h["ema20"]         = _ema(h["close"], 20)
    h["ema50"]         = _ema(h["close"], 50)
    h["ema9"]          = _ema(h["close"], 9)
    h["ema_slope20"]   = h["ema20"] - h["ema20"].shift(3)
    h["ema_slope50"]   = h["ema50"] - h["ema50"].shift(6)
    h["vs_ema20_pct"]  = (h["close"] / h["ema20"] - 1.0) * 100
    h["vs_ema50_pct"]  = (h["close"] / h["ema50"] - 1.0) * 100
    h["ema9_20_cross"] = (h["ema9"] > h["ema20"]).astype(int)
    h["rsi14"]         = _rsi(h["close"], 14)
    h["rsi6"]          = _rsi(h["close"], 6)
    h["rsi14_slope3"]  = h["rsi14"] - h["rsi14"].shift(3)
    adx, pdi, mdi      = _adx(h, 14)
    h["adx14"]         = adx
    h["pdi14"]         = pdi
    h["mdi14"]         = mdi
    h["stoch_k"], h["stoch_d"] = _stochastic(h, 14, 3)
    _, _, _, h["bb_bw"], h["bb_pctb"] = _bollinger(h, 20)
    h["cci20"]         = _cci(h, 20)
    h["mfi14"]         = _mfi(h, 14)
    h["vwap"]          = _vwap_session(h)
    h["vs_vwap_pct"]   = (h["close"] / h["vwap"] - 1.0) * 100
    h["swing_low5"]    = h["low"].rolling(5).min().shift(1)
    h["swing_high5"]   = h["high"].rolling(5).max().shift(1)
    cu = (h["close"] > h["close"].shift(1)).astype(int)
    h["consec_up"]     = cu.groupby(
        (cu != cu.shift()).cumsum()).cumcount().add(1) * cu
    h["hour"]          = h["time"].dt.hour
    h["dow"]           = h["time"].dt.dayofweek
    h["is_us_open"]    = h["hour"].isin([13, 14, 15, 16]).astype(int)
    h["is_us_core"]    = h["hour"].isin([14, 15, 16, 17, 18, 19, 20]).astype(int)
    h["is_london"]     = h["hour"].isin([7, 8, 9, 10, 11, 12]).astype(int)
    h["bar_of_day"]    = h.groupby("trade_date").cumcount()
    dbg(f"H1 features built: {len(h)} rows")
    return h


def merge_mtf(h1: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    dbg("Merging H1 and Daily frames on trade_date...")
    merged = h1.merge(daily, on="trade_date", how="left")
    na_count = merged["rsi10"].isna().sum()
    dbg(f"Merged: {len(merged)} rows | {na_count} rows with NaN rsi10 (will drop)")
    return merged


# =============================================================================
# CANDIDATE GENERATION
# =============================================================================

def _rsi_in_band(val, lo, hi) -> bool:
    """Returns True if val is within [lo, hi] inclusive. NaN-safe."""
    return False if (val is None or pd.isna(val)) else (lo <= val <= hi)


def get_candidates(df: pd.DataFrame, rsi_band: tuple,
                   st_period: int, st_mult: float,
                   direction: int) -> list[int]:
    """
    Returns positional (iloc) indices of rows satisfying all 3 entry conditions:
      1. Daily close above EMA100 (for longs) or below (for shorts)
      2. Daily RSI10 inside the optimal band
      3. H1 SuperTrend just flipped in the trade direction
    df must have clean 0-based RangeIndex (always reset_index before calling).
    """
    tag      = f"st_{st_period}_{st_mult}"
    flip_col = f"flip_up_{tag}" if direction == 1 else f"flip_dn_{tag}"
    lo, hi   = rsi_band

    if flip_col not in df.columns:
        dbg(f"Flip column {flip_col} not found — skipping")
        return []

    regime_ok = (df["above_ema100"] == 1) if direction == 1 else (df["above_ema100"] == 0)
    rsi_ok    = df["rsi10"].apply(lambda x: _rsi_in_band(x, lo, hi))
    flip_ok   = df[flip_col] == 1
    mask      = regime_ok & rsi_ok & flip_ok
    positions = list(np.where(mask.values)[0])
    return positions


# =============================================================================
# TRADE SIMULATION  — positional index only, no label index
# =============================================================================

def _simulate_trade(df: pd.DataFrame, pos: int, direction: int,
                    st_col: str, stop_atr_mult: float,
                    tp_mode: str, tp_param: float) -> dict | None:
    """
    Simulate one trade starting at positional index pos.
    Returns trade dict or None if invalid setup.
    df must have clean 0-based RangeIndex.
    """
    # Guard: positional index must be valid
    if pos < 0 or pos >= len(df):
        dbg(f"pos={pos} out of bounds for df len={len(df)}")
        return None

    row   = df.iloc[pos]
    entry = row["close"] + direction * SLIPPAGE_POINTS
    atrv  = row["atr14"]

    # Guard: ATR must be valid and positive
    if pd.isna(atrv) or atrv <= 0:
        dbg(f"Invalid ATR={atrv} at pos={pos}")
        return None

    # --- Compute stop loss ---
    if direction == 1:
        swing_sl   = row.get("swing_low5", np.nan)
        st_sl      = row[st_col] if (
            st_col in df.columns and pd.notna(row.get(st_col))) else np.nan
        candidates = [entry - stop_atr_mult * atrv]
        if pd.notna(swing_sl): candidates.append(float(swing_sl))
        if pd.notna(st_sl):    candidates.append(float(st_sl))
        stop = min(candidates)
        stop = min(stop, entry - 0.25)   # enforce minimum risk
    else:
        swing_sl   = row.get("swing_high5", np.nan)
        st_sl      = row[st_col] if (
            st_col in df.columns and pd.notna(row.get(st_col))) else np.nan
        candidates = [entry + stop_atr_mult * atrv]
        if pd.notna(swing_sl): candidates.append(float(swing_sl))
        if pd.notna(st_sl):    candidates.append(float(st_sl))
        stop = max(candidates)
        stop = max(stop, entry + 0.25)

    risk = abs(entry - stop)
    if risk <= 0:
        dbg(f"Zero risk at pos={pos} entry={entry} stop={stop}")
        return None

    # --- Compute initial take profit ---
    if tp_mode == "fixed_rr":
        target = entry + direction * tp_param * risk
    elif tp_mode == "atr_multiple":
        target = entry + direction * tp_param * atrv
    else:
        target = np.nan   # trailing_st: no fixed target, managed bar-by-bar

    # --- Simulate bar-by-bar ---
    trailing_stop = stop
    end_pos       = min(pos + MAX_HOLD_BARS, len(df) - 1)
    exit_px = exit_time = bars_held = None

    for j in range(pos + 1, end_pos + 1):
        r = df.iloc[j]

        # Update trailing stop for trailing_st mode
        if (tp_mode == "trailing_st"
                and st_col in df.columns
                and pd.notna(r.get(st_col))):
            if direction == 1:
                trailing_stop = max(trailing_stop, float(r[st_col]))
            else:
                trailing_stop = min(trailing_stop, float(r[st_col]))

        sl_hit = (r["low"]  <= trailing_stop) if direction == 1 \
                 else (r["high"] >= trailing_stop)
        tp_hit = False if (pd.isna(target) or target == 0) else (
            (r["high"] >= target) if direction == 1 else (r["low"] <= target))

        if sl_hit and tp_hit:
            # Both hit same bar — conservative: take SL
            exit_px = trailing_stop
        elif sl_hit:
            exit_px = trailing_stop - direction * SLIPPAGE_POINTS
        elif tp_hit:
            exit_px = target - direction * SLIPPAGE_POINTS

        if exit_px is not None:
            exit_time = r["time"]
            bars_held = j - pos
            break

    # Time-exit if no SL/TP hit
    if exit_px is None:
        lr        = df.iloc[end_pos]
        exit_px   = lr["close"] - direction * SLIPPAGE_POINTS
        exit_time = lr["time"]
        bars_held = end_pos - pos

    gross_points = (exit_px - entry) * direction
    net_usd      = gross_points * MNQ_DOLLARS_PER_POINT - 2 * COMMISSION_PER_SIDE_USD

    # FIX: win = profitable, regardless of exit type
    win         = int(net_usd > 0)
    rr_realized = gross_points / risk if risk > 0 else np.nan

    return {
        "entry_time":   row["time"],
        "exit_time":    exit_time,
        "direction":    direction,
        "entry_px":     entry,
        "exit_px":      exit_px,
        "stop_px":      stop,
        "target_px":    target,
        "bars_held":    bars_held,
        "win":          win,
        "gross_points": gross_points,
        "net_usd":      net_usd,
        "risk_points":  risk,
        "rr_realized":  rr_realized,
    }


# =============================================================================
# PARAMETER SEARCH  (train slice only — never called on OOS data)
# =============================================================================

def parameter_search_on_slice(df: pd.DataFrame) -> pd.DataFrame:
    df      = df.reset_index(drop=True)
    results = []
    combos  = 0

    for st_period, st_mult in itertools.product(SUPER_PERIODS, SUPER_MULTS):
        st_col = f"st_{st_period}_{st_mult}"
        if st_col not in df.columns:
            warn(f"ST column {st_col} missing — skipping")
            continue

        for rsi_band in RSI_BANDS:
            for direction in [1, -1]:
                positions = get_candidates(
                    df, rsi_band, st_period, st_mult, direction)
                if len(positions) < 30:
                    continue

                for stop_mult in STOP_ATR_MULTS:
                    for tp_mode, tp_grid in [
                        ("fixed_rr",     TP_RR_GRID),
                        ("atr_multiple", TP_ATR_MULTS),
                        ("trailing_st",  [0.0]),
                    ]:
                        for tp_param in tp_grid:
                            combos += 1
                            trades = []
                            for pos in positions:
                                tr = _simulate_trade(
                                    df, pos, direction, st_col,
                                    stop_mult, tp_mode, tp_param)
                                if tr:
                                    trades.append(tr)

                            if len(trades) < 30:
                                continue

                            tdf     = pd.DataFrame(trades)
                            pos_sum = tdf.loc[tdf["net_usd"] > 0,  "net_usd"].sum()
                            neg_sum = abs(
                                tdf.loc[tdf["net_usd"] <= 0, "net_usd"].sum()) + 1e-9

                            results.append({
                                "rsi_lo":        rsi_band[0],
                                "rsi_hi":        rsi_band[1],
                                "st_period":     st_period,
                                "st_mult":       st_mult,
                                "direction":     direction,
                                "stop_mult":     stop_mult,
                                "tp_mode":       tp_mode,
                                "tp_param":      tp_param,
                                "n_trades":      len(tdf),
                                "win_rate":      tdf["win"].mean(),
                                "expectancy":    tdf["net_usd"].mean(),
                                "profit_factor": pos_sum / neg_sum,
                                "median_rr":     tdf["rr_realized"].median(),
                                "total_usd":     tdf["net_usd"].sum(),
                            })

    dbg(f"Parameter search: {combos} combinations evaluated, "
        f"{len(results)} with >=30 trades")

    if not results:
        return pd.DataFrame()

    return (pd.DataFrame(results)
            .sort_values(["expectancy", "profit_factor", "win_rate"],
                         ascending=False)
            .reset_index(drop=True))


# =============================================================================
# FEATURE EXTRACTION  (single shared function — no duplication)
# =============================================================================

FEATURE_COLS = [
    "rsi10", "ema_gap_pct", "rsi_slope3", "rsi_slope5",
    "d_ret1", "d_ret5", "d_ret10", "d_range_pct", "d_atr_pct",
    "atr_pctile", "adx14_d", "pdi14_d", "mdi14_d",
    "bb_bw_d", "bb_pctb_d", "cci20_d",
    "ret1", "ret3", "ret6", "ret12",
    "rsi14", "rsi6", "rsi14_slope3",
    "stoch_k", "stoch_d",
    "bb_pctb", "bb_bw", "cci20", "mfi14",
    "vs_ema20_pct", "vs_ema50_pct", "ema_slope20", "ema_slope50",
    "ema9_20_cross",
    "adx14", "pdi14", "mdi14",
    "vs_vwap_pct",
    "atr14_pct", "vol_ratio", "vol_ratio_6", "range_pct",
    "dist_to_st_pct",
    "hour", "dow", "bar_of_day",
    "is_us_open", "is_us_core", "is_london",
    "consec_up",
]


def _extract_features(r, st_col: str) -> dict:
    """Extract ML feature dict from a merged DataFrame row."""
    # Distance from close to SuperTrend line as % of close
    dist = np.nan
    if (st_col in r.index
            and pd.notna(r.get(st_col))
            and r["close"] != 0):
        dist = (r["close"] - r[st_col]) / r["close"] * 100

    return {
        "rsi10":         r.get("rsi10"),
        "ema_gap_pct":   r.get("ema_gap_pct"),
        "rsi_slope3":    r.get("rsi_slope3"),
        "rsi_slope5":    r.get("rsi_slope5"),
        "d_ret1":        r.get("d_ret1"),
        "d_ret5":        r.get("d_ret5"),
        "d_ret10":       r.get("d_ret10"),
        "d_range_pct":   r.get("d_range_pct"),
        "d_atr_pct":     r.get("d_atr_pct"),
        "atr_pctile":    r.get("atr_pctile"),
        "adx14_d":       r.get("adx14_d"),
        "pdi14_d":       r.get("pdi14_d"),
        "mdi14_d":       r.get("mdi14_d"),
        "bb_bw_d":       r.get("bb_bw_d"),
        "bb_pctb_d":     r.get("bb_pctb_d"),
        "cci20_d":       r.get("cci20_d"),
        "ret1":          r.get("ret1"),
        "ret3":          r.get("ret3"),
        "ret6":          r.get("ret6"),
        "ret12":         r.get("ret12"),
        "rsi14":         r.get("rsi14"),
        "rsi6":          r.get("rsi6"),
        "rsi14_slope3":  r.get("rsi14_slope3"),
        "stoch_k":       r.get("stoch_k"),
        "stoch_d":       r.get("stoch_d"),
        "bb_pctb":       r.get("bb_pctb"),
        "bb_bw":         r.get("bb_bw"),
        "cci20":         r.get("cci20"),
        "mfi14":         r.get("mfi14"),
        "vs_ema20_pct":  r.get("vs_ema20_pct"),
        "vs_ema50_pct":  r.get("vs_ema50_pct"),
        "ema_slope20":   r.get("ema_slope20"),
        "ema_slope50":   r.get("ema_slope50"),
        "ema9_20_cross": r.get("ema9_20_cross"),
        "adx14":         r.get("adx14"),
        "pdi14":         r.get("pdi14"),
        "mdi14":         r.get("mdi14"),
        "vs_vwap_pct":   r.get("vs_vwap_pct"),
        "atr14_pct":     r.get("atr14_pct"),
        "vol_ratio":     r.get("vol_ratio"),
        "vol_ratio_6":   r.get("vol_ratio_6"),
        "range_pct":     r.get("range_pct"),
        "dist_to_st_pct": dist,
        "hour":          r.get("hour"),
        "dow":           r.get("dow"),
        "bar_of_day":    r.get("bar_of_day"),
        "is_us_open":    r.get("is_us_open"),
        "is_us_core":    r.get("is_us_core"),
        "is_london":     r.get("is_london"),
        "consec_up":     r.get("consec_up"),
    }


def build_ml_dataset(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    df        = df.reset_index(drop=True)
    st_period = int(params["st_period"])
    st_mult   = float(params["st_mult"])
    direction = int(params["direction"])
    rsi_band  = (params["rsi_lo"], params["rsi_hi"])
    st_col    = f"st_{st_period}_{st_mult}"
    tp_param  = float(params["tp_param"]) if not pd.isna(
        params["tp_param"]) else 0.0

    positions = get_candidates(df, rsi_band, st_period, st_mult, direction)
    rows = []

    for pos in positions:
        tr = _simulate_trade(df, pos, direction, st_col,
                             params["stop_mult"], params["tp_mode"], tp_param)
        if tr is None:
            continue
        r    = df.iloc[pos]
        feat = _extract_features(r, st_col)
        feat.update({
            "time":         r["time"],
            "win":          tr["win"],
            "net_usd":      tr["net_usd"],
            "quality_win":  int(tr["net_usd"] > QUALITY_USD_THRESHOLD),
            "bars_held":    tr["bars_held"],
            "gross_points": tr["gross_points"],
            "rr_realized":  tr["rr_realized"],
        })
        rows.append(feat)

    dbg(f"ML dataset: {len(rows)} trades from {len(positions)} candidates")
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


# =============================================================================
# CONSENSUS ML MODELS
# =============================================================================

def _make_clf(name: str) -> Pipeline:
    if name == "logreg":
        return Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=2000, C=0.5))])
    if name == "rf":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=6,
                min_samples_leaf=8, random_state=42, n_jobs=-1))])
    if name == "et":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", ExtraTreesClassifier(
                n_estimators=500, max_depth=7,
                min_samples_leaf=6, random_state=42, n_jobs=-1))])
    if name == "gb":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4,
                subsample=0.8, random_state=42))])
    raise ValueError(f"Unknown classifier: {name}")


def _make_reg(name: str) -> Pipeline:
    if name == "ridge":
        return Pipeline([
            ("imp",    SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("reg",    Ridge(alpha=1.0))])
    if name == "rfr":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(
                n_estimators=300, max_depth=6,
                min_samples_leaf=8, random_state=42, n_jobs=-1))])
    if name == "gbr":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("reg", GradientBoostingRegressor(
                n_estimators=200, max_depth=4,
                subsample=0.8, random_state=42))])
    raise ValueError(f"Unknown regressor: {name}")


def train_consensus_models(mdf: pd.DataFrame) -> dict | None:
    if len(mdf) < MIN_TRAIN_TRADES:
        dbg(f"Too few trades ({len(mdf)} < {MIN_TRAIN_TRADES}) — skipping ML")
        return None

    avail_fc = [c for c in FEATURE_COLS if c in mdf.columns]
    X   = mdf[avail_fc]
    y_c = mdf["quality_win"].astype(int)
    y_r = mdf["net_usd"]

    # Need at least 2 classes to train binary classifiers
    if y_c.nunique() < 2:
        dbg("Only one class in quality_win — skipping ML")
        return None

    clf_names = ["logreg", "rf", "et", "gb"]
    reg_names = ["ridge", "rfr", "gbr"]
    tscv      = TimeSeriesSplit(n_splits=5)

    clf_models = {n: _make_clf(n) for n in clf_names}
    clf_oof    = pd.DataFrame(index=mdf.index)
    clf_scores = {}

    for name, model in clf_models.items():
        pred = pd.Series(np.nan, index=mdf.index)
        for tr_i, te_i in tscv.split(X):
            # Skip fold if only one class in training split
            if y_c.iloc[tr_i].nunique() < 2:
                continue
            model.fit(X.iloc[tr_i], y_c.iloc[tr_i])
            pred.iloc[te_i] = model.predict_proba(X.iloc[te_i])[:, 1]
        valid = pred.dropna().index
        if len(valid) == 0 or y_c.loc[valid].nunique() < 2:
            dbg(f"Classifier {name}: no valid OOF predictions — skipping")
            continue
        clf_scores[name] = roc_auc_score(y_c.loc[valid], pred.loc[valid])
        clf_oof[name]    = pred
        model.fit(X, y_c)   # refit on full training data

    if not clf_scores:
        dbg("No classifiers trained successfully")
        return None

    scored_clfs = list(clf_scores.keys())
    clf_oof["consensus"] = clf_oof[scored_clfs].mean(axis=1)
    valid_idx = clf_oof["consensus"].dropna().index

    if len(valid_idx) == 0 or y_c.loc[valid_idx].nunique() < 2:
        dbg("Consensus has no valid predictions")
        return None

    preds_bin = (clf_oof.loc[valid_idx, "consensus"]
                 >= CONSENSUS_PROBA_THRESHOLD).astype(int)

    # Train regressors
    reg_models = {n: _make_reg(n) for n in reg_names}
    reg_scores = {}

    for name, model in reg_models.items():
        pred = pd.Series(np.nan, index=mdf.index)
        for tr_i, te_i in tscv.split(X):
            model.fit(X.iloc[tr_i], y_r.iloc[tr_i])
            pred.iloc[te_i] = model.predict(X.iloc[te_i])
        valid = pred.dropna().index
        reg_scores[name] = mean_absolute_error(y_r.loc[valid], pred.loc[valid])
        model.fit(X, y_r)

    # Feature importances from tree models
    importances = {}
    for name in ["rf", "et", "gb"]:
        if name not in clf_scores:
            continue
        clf = clf_models[name].named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances[name] = pd.Series(
                clf.feature_importances_, index=avail_fc)

    feat_rank = None
    if importances:
        feat_rank = (pd.concat(importances, axis=1)
                     .fillna(0).mean(axis=1)
                     .sort_values(ascending=False))

    return {
        "clf_models":   clf_models,
        "reg_models":   reg_models,
        "feature_cols": avail_fc,
        "metrics": {
            "feature_cols":        avail_fc,
            "clf_auc":             clf_scores,
            "consensus_clf_auc":   roc_auc_score(
                y_c.loc[valid_idx],
                clf_oof.loc[valid_idx, "consensus"]),
            "consensus_precision": precision_score(
                y_c.loc[valid_idx], preds_bin, zero_division=0),
            "consensus_recall":    recall_score(
                y_c.loc[valid_idx], preds_bin, zero_division=0),
            "reg_mae":             reg_scores,
            "feature_rank":        feat_rank,
            "n_trades":            len(mdf),
        },
    }


# =============================================================================
# SCORE A SINGLE ROW
# =============================================================================

def score_row(row, st_col: str, clf_models: dict, reg_models: dict,
              feature_cols: list, direction: int) -> dict:
    """Score one bar using the trained consensus models."""
    feat = _extract_features(row, st_col)
    # Align to exact columns the models were trained on
    X    = pd.DataFrame([feat]).reindex(columns=feature_cols)

    clf_probs = {n: float(m.predict_proba(X)[0, 1])
                 for n, m in clf_models.items()}
    reg_preds = {n: float(m.predict(X)[0])
                 for n, m in reg_models.items()}

    consensus_proba   = float(np.mean(list(clf_probs.values())))
    consensus_net_usd = float(np.mean(list(reg_preds.values())))

    return {
        "clf_probs":         clf_probs,
        "reg_preds":         reg_preds,
        "consensus_proba":   consensus_proba,
        "consensus_net_usd": consensus_net_usd,
        "take_trade":        consensus_proba >= CONSENSUS_PROBA_THRESHOLD,
    }


# =============================================================================
# WALK-FORWARD OOS VALIDATION
# =============================================================================

def walk_forward_eval(full_df: pd.DataFrame) -> pd.DataFrame:
    n        = len(full_df)
    fold_num = 0
    all_oos  = []
    start    = 0

    while (start + WF_TRAIN_BARS + WF_OOS_BARS) <= n:
        fold_num += 1
        train_df = full_df.iloc[
            start : start + WF_TRAIN_BARS].reset_index(drop=True)
        oos_df   = full_df.iloc[
            start + WF_TRAIN_BARS :
            start + WF_TRAIN_BARS + WF_OOS_BARS].reset_index(drop=True)

        print(f"\n--- WF Fold {fold_num} | "
              f"train {train_df['time'].iloc[0].date()} → "
              f"{train_df['time'].iloc[-1].date()} | "
              f"OOS  {oos_df['time'].iloc[0].date()} → "
              f"{oos_df['time'].iloc[-1].date()}")

        param_results = parameter_search_on_slice(train_df)
        if param_results.empty:
            print("  No valid params — skipping fold.")
            start += WF_OOS_BARS
            continue

        best = param_results.iloc[0].to_dict()
        print(f"  Best: rsi=({best['rsi_lo']},{best['rsi_hi']}) "
              f"ST({int(best['st_period'])},{best['st_mult']}) "
              f"dir={'LONG' if best['direction']==1 else 'SHORT'} "
              f"stop={best['stop_mult']} tp={best['tp_mode']} "
              f"exp=${best['expectancy']:.2f}")

        mdf_train = build_ml_dataset(train_df, best)
        result    = train_consensus_models(mdf_train)

        if result:
            m = result["metrics"]
            print(f"  ML | AUC={m['consensus_clf_auc']:.3f} "
                  f"prec={m['consensus_precision']:.3f} "
                  f"recall={m['consensus_recall']:.3f} "
                  f"n={m['n_trades']}")
            if m["feature_rank"] is not None:
                print(f"  Top features: "
                      f"{m['feature_rank'].head(5).index.tolist()}")
        else:
            print(f"  Not enough trades or single class "
                  f"({len(mdf_train)}) — rule-based only.")

        # OOS evaluation: use context window so indicators are warm
        context_df = full_df.iloc[
            start : start + WF_TRAIN_BARS + WF_OOS_BARS
        ].reset_index(drop=True)
        oos_start = WF_TRAIN_BARS

        st_period = int(best["st_period"])
        st_mult_v = float(best["st_mult"])
        direction = int(best["direction"])
        rsi_band  = (best["rsi_lo"], best["rsi_hi"])
        st_col    = f"st_{st_period}_{st_mult_v}"
        tp_param  = float(best["tp_param"]) if not pd.isna(
            best["tp_param"]) else 0.0

        all_positions = get_candidates(
            context_df, rsi_band, st_period, st_mult_v, direction)
        oos_positions = [p for p in all_positions if p >= oos_start]
        dbg(f"OOS candidates: {len(oos_positions)}")

        for pos in oos_positions:
            tr = _simulate_trade(
                context_df, pos, direction, st_col,
                best["stop_mult"], best["tp_mode"], tp_param)
            if tr is None:
                continue

            row = context_df.iloc[pos]
            tr.update({
                "fold":              fold_num,
                "rsi_lo":            best["rsi_lo"],
                "rsi_hi":            best["rsi_hi"],
                "st_period":         st_period,
                "st_mult":           st_mult_v,
                "consensus_proba":   np.nan,
                "consensus_net_usd": np.nan,
                "take_trade_ml":     np.nan,
            })

            if result:
                try:
                    scored = score_row(
                        row, st_col,
                        result["clf_models"], result["reg_models"],
                        result["feature_cols"], direction)
                    tr["consensus_proba"]   = scored["consensus_proba"]
                    tr["consensus_net_usd"] = scored["consensus_net_usd"]
                    tr["take_trade_ml"]     = int(scored["take_trade"])
                except Exception as e:
                    dbg(f"Score error at pos={pos}: {e}")

            all_oos.append(tr)

        start += WF_OOS_BARS

    if not all_oos:
        print("\nNo OOS trades generated.")
        return pd.DataFrame()

    return (pd.DataFrame(all_oos)
            .sort_values("entry_time")
            .reset_index(drop=True))


# =============================================================================
# LAST N TRADES REPLAY
# =============================================================================

def print_last_n_trades(full_df: pd.DataFrame, best_params: dict,
                        result: dict | None, n: int = LAST_N_TRADES) -> None:
    """
    Replay the last N historical signal occurrences and print
    a detailed trade-by-trade breakdown showing exactly what the
    strategy would have done and why.
    """
    status(f"LAST {n} HISTORICAL TRADES (most recent signals)")

    df        = full_df.reset_index(drop=True)
    st_period = int(best_params["st_period"])
    st_mult_v = float(best_params["st_mult"])
    direction = int(best_params["direction"])
    rsi_band  = (best_params["rsi_lo"], best_params["rsi_hi"])
    st_col    = f"st_{st_period}_{st_mult_v}"
    tp_param  = float(best_params["tp_param"]) if not pd.isna(
        best_params["tp_param"]) else 0.0

    positions = get_candidates(df, rsi_band, st_period, st_mult_v, direction)

    if not positions:
        print("  No historical signals found with current best params.")
        return

    # Take last N positions
    last_positions = positions[-n:]

    for i, pos in enumerate(last_positions):
        tr = _simulate_trade(
            df, pos, direction, st_col,
            best_params["stop_mult"], best_params["tp_mode"], tp_param)

        if tr is None:
            print(f"\n  Trade {i+1}: INVALID SETUP at pos={pos}")
            continue

        row = df.iloc[pos]

        # ML scoring
        ml_proba   = np.nan
        ml_net_usd = np.nan
        ml_take    = None
        if result:
            try:
                scored     = score_row(row, st_col,
                                       result["clf_models"],
                                       result["reg_models"],
                                       result["feature_cols"], direction)
                ml_proba   = scored["consensus_proba"]
                ml_net_usd = scored["consensus_net_usd"]
                ml_take    = scored["take_trade"]
            except Exception as e:
                dbg(f"ML score error: {e}")

        outcome_str = "WIN  ✓" if tr["win"] else "LOSS ✗"
        dir_str     = "LONG" if direction == 1 else "SHORT"
        ml_str      = (f"proba={ml_proba:.3f} "
                       f"est=${ml_net_usd:.0f} "
                       f"take={'YES' if ml_take else 'NO'}"
                       if not np.isnan(ml_proba) else "ML=N/A")

        print(f"\n  ── Trade {i+1} of {len(last_positions)} ──────────────────────")
        print(f"  Entry time  : {tr['entry_time']}")
        print(f"  Direction   : {dir_str}")
        print(f"  Entry price : {tr['entry_px']:.2f}")
        print(f"  Stop loss   : {tr['stop_px']:.2f}  "
              f"(risk = {tr['risk_points']:.2f} pts)")
        print(f"  Target      : "
              f"{tr['target_px']:.2f}" if not pd.isna(tr['target_px'])
              else f"  Target      : trailing SuperTrend")
        print(f"  Exit time   : {tr['exit_time']}")
        print(f"  Exit price  : {tr['exit_px']:.2f}")
        print(f"  Bars held   : {tr['bars_held']}  "
              f"(~{tr['bars_held']}h)")
        print(f"  Gross pts   : {tr['gross_points']:+.2f}")
        print(f"  Net USD     : ${tr['net_usd']:+.2f}")
        print(f"  R:R realized: {tr['rr_realized']:.2f}R")
        print(f"  Outcome     : {outcome_str}")
        print(f"  ML filter   : {ml_str}")
        print(f"  ── Why entered? ─────────────────────────────────────")
        print(f"  Daily above EMA100 : "
              f"{'YES' if row.get('above_ema100')==1 else 'NO'} "
              f"(close={row.get('daily_close', row.get('close')):.2f} "
              f"ema={row.get('ema100', 0):.2f})")
        print(f"  Daily RSI10        : {row.get('rsi10', 0):.1f} "
              f"in band ({rsi_band[0]},{rsi_band[1]})")
        print(f"  H1 ST flip         : "
              f"ST({st_period},{st_mult_v}) flipped "
              f"{'RED→GREEN' if direction==1 else 'GREEN→RED'}")
        print(f"  ST line value      : {row.get(st_col, 0):.2f}")
        print(f"  H1 ATR14           : {row.get('atr14', 0):.2f}")
        print(f"  VWAP deviation     : {row.get('vs_vwap_pct', 0):+.2f}%")
        print(f"  ADX14              : {row.get('adx14', 0):.1f}")
        print(f"  Vol ratio          : {row.get('vol_ratio', 0):.2f}x")


# =============================================================================
# REPORTING
# =============================================================================

def _perf(tdf: pd.DataFrame, label: str = "") -> dict:
    if tdf.empty:
        return {"label": label, "n_trades": 0}
    pos = tdf.loc[tdf["net_usd"] > 0,  "net_usd"].sum()
    neg = abs(tdf.loc[tdf["net_usd"] <= 0, "net_usd"].sum()) + 1e-9
    return {
        "label":         label,
        "n_trades":      len(tdf),
        "win_rate":      round(float(tdf["win"].mean()), 4),
        "expectancy":    round(float(tdf["net_usd"].mean()), 2),
        "profit_factor": round(float(pos / neg), 3),
        "total_usd":     round(float(tdf["net_usd"].sum()), 2),
        "median_rr":     round(float(tdf["rr_realized"].median()), 3),
        "avg_bars":      round(float(tdf["bars_held"].mean()), 1),
    }


def print_wf_report(oos: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("WALK-FORWARD OOS RESULTS")
    print("="*70)
    print(json.dumps(_perf(oos, "ALL OOS"), indent=2))

    ml_f = oos.loc[oos["take_trade_ml"] == 1]
    if not ml_f.empty:
        print(json.dumps(_perf(ml_f, "ML FILTERED"), indent=2))
        print(f"  ML filter kept {len(ml_f)}/{len(oos)} trades "
              f"({len(ml_f)/len(oos)*100:.1f}%)")

    print("\nPer-fold:")
    for fid, grp in oos.groupby("fold"):
        s = _perf(grp)
        print(f"  Fold {fid}: n={s['n_trades']} wr={s['win_rate']:.2%} "
              f"exp=${s['expectancy']:.2f} pf={s['profit_factor']:.2f} "
              f"total=${s['total_usd']:.2f}")

    print("\nLong vs Short:")
    for d, lbl in [(1, "LONG"), (-1, "SHORT")]:
        sub = oos.loc[oos["direction"] == d]
        if not sub.empty:
            s = _perf(sub)
            print(f"  {lbl}: n={s['n_trades']} wr={s['win_rate']:.2%} "
                  f"exp=${s['expectancy']:.2f} pf={s['profit_factor']:.2f}")


# =============================================================================
# LIVE SIGNAL SCANNER
# =============================================================================

def scan_live_signal(full_df: pd.DataFrame, best_params: dict,
                     result: dict | None) -> dict:
    """
    Check the most recent H1 bar for a live entry signal.
    Returns full signal dict for logging and execution.
    """
    st_period = int(best_params["st_period"])
    st_mult_v = float(best_params["st_mult"])
    direction = int(best_params["direction"])
    rsi_band  = (best_params["rsi_lo"], best_params["rsi_hi"])
    st_col    = f"st_{st_period}_{st_mult_v}"
    flip_col  = (f"flip_up_{st_col}" if direction == 1
                 else f"flip_dn_{st_col}")

    latest    = full_df.iloc[-1]
    regime_ok = bool(latest.get("above_ema100") == 1) \
                if direction == 1 \
                else bool(latest.get("above_ema100") == 0)
    rsi_ok    = bool(_rsi_in_band(latest.get("rsi10"), *rsi_band))
    flip_ok   = bool(latest.get(flip_col, 0) == 1)

    dbg(f"Live bar: {latest['time']} close={latest['close']:.2f}")
    dbg(f"  regime_ok={regime_ok} (above_ema100="
        f"{latest.get('above_ema100')})")
    dbg(f"  rsi_ok={rsi_ok} (rsi10={latest.get('rsi10', 0):.1f} "
        f"band={rsi_band})")
    dbg(f"  flip_ok={flip_ok} ({flip_col}="
        f"{latest.get(flip_col, 0)})")

    # Compute SL for signal output
    atrv      = float(latest.get("atr14", 0))
    entry_est = float(latest["close"]) + direction * SLIPPAGE_POINTS
    sl_est    = entry_est - direction * best_params["stop_mult"] * atrv
    tp_param  = float(best_params["tp_param"]) if not pd.isna(
        best_params["tp_param"]) else 0.0
    risk_est  = abs(entry_est - sl_est)
    tp_est    = (entry_est + direction * tp_param * risk_est
                 if best_params["tp_mode"] != "trailing_st"
                 else float("nan"))

    sig = {
        "time":              str(latest["time"]),
        "close":             float(latest["close"]),
        "direction":         "LONG" if direction == 1 else "SHORT",
        "regime_ok":         regime_ok,
        "rsi10":             float(latest.get("rsi10", np.nan)),
        "rsi_band":          list(rsi_band),
        "rsi_ok":            rsi_ok,
        "st_flip":           flip_ok,
        "rule_signal":       bool(regime_ok and rsi_ok and flip_ok),
        "entry_estimate":    round(entry_est, 2),
        "sl_estimate":       round(sl_est, 2),
        "tp_estimate":       round(tp_est, 2) if not np.isnan(tp_est) else None,
        "atr14":             round(atrv, 2),
        "consensus_proba":   None,
        "consensus_net_usd": None,
        "take_trade":        False,
    }

    if sig["rule_signal"] and result:
        try:
            scored = score_row(
                latest, st_col,
                result["clf_models"], result["reg_models"],
                result["feature_cols"], direction)
            sig["consensus_proba"]   = round(scored["consensus_proba"], 4)
            sig["consensus_net_usd"] = round(scored["consensus_net_usd"], 2)
            sig["take_trade"]        = bool(scored["take_trade"])
            sig["clf_probs"]         = {
                k: round(v, 4) for k, v in scored["clf_probs"].items()}
            sig["reg_preds"]         = {
                k: round(v, 2)  for k, v in scored["reg_preds"].items()}
            dbg(f"  ML consensus_proba={sig['consensus_proba']:.4f} "
                f"take_trade={sig['take_trade']}")
        except Exception as e:
            sig["scoring_error"] = str(e)
            err(f"Signal scoring error: {e}")
    else:
        dbg(f"  ML skipped: rule_signal={sig['rule_signal']} "
            f"result={'loaded' if result else 'None'}")

    return sig


# =============================================================================
# LIVE EXECUTION ENGINE
# =============================================================================

def run_live(full_df: pd.DataFrame, best_params: dict,
             result: dict | None) -> None:
    """
    Live mode: check signal, guard against open positions, send order.
    This is the production execution function.
    """
    status("LIVE MODE — checking for trade signal")

    sig = scan_live_signal(full_df, best_params, result)

    print(f"\n  Bar time  : {sig['time']}")
    print(f"  Close     : {sig['close']:.2f}")
    print(f"  Direction : {sig['direction']}")
    print(f"  regime_ok : {sig['regime_ok']}  "
          f"(daily above EMA100)")
    print(f"  rsi_ok    : {sig['rsi_ok']}  "
          f"(RSI10={sig['rsi10']:.1f} in {sig['rsi_band']})")
    print(f"  st_flip   : {sig['st_flip']}  "
          f"(SuperTrend flipped)")
    print(f"  rule_sig  : {sig['rule_signal']}")
    if sig.get("consensus_proba") is not None:
        print(f"  ML proba  : {sig['consensus_proba']:.4f}  "
              f"(threshold={CONSENSUS_PROBA_THRESHOLD})")
        print(f"  ML net$   : ${sig['consensus_net_usd']:.2f}")
    print(f"  take_trade: {sig['take_trade']}")

    # Save signal state regardless of outcome
    with open(LIVE_STATE_FILE, "w") as f:
        json.dump(sig, f, indent=2, default=str)
    dbg(f"Signal state saved to {LIVE_STATE_FILE}")

    # ── GATE 1: Rule signal must fire ──────────────────────────────
    if not sig["rule_signal"]:
        status("NO SIGNAL — rule conditions not met. No order sent.")
        return

    # Rule signal fired — alert Telegram
    tg_signal_fired(sig)

    # ── GATE 2: ML must confirm ────────────────────────────────────
    if not sig["take_trade"]:
        status("SIGNAL FILTERED BY ML — consensus proba below threshold. "
               "No order sent.")
        tg_ml_filtered(sig)
        return

    # ── GATE 3: Open position guard ───────────────────────────────
    # Reconnect MT5 for order checks (keep connection scoped)
    status("Signal confirmed — connecting MT5 for position check...")
    mt5 = initialize_mt5()
    try:
        # PRIMARY GUARD: check for any open position on this symbol
        if has_open_position(mt5, SYMBOL, LIVE_MAGIC):
            status("BLOCKED — open position already exists. "
                   "No order sent. Will check again next bar.")
            positions = mt5.positions_get(symbol=SYMBOL) or []
            tg_blocked_position(list(positions))
            return

        # ── GATE 4: Spread check ───────────────────────────────────
        if not check_spread(mt5, SYMBOL, LIVE_MAX_SPREAD_PTS):
            status("BLOCKED — spread too wide. No order sent.")
            tg(f"⚠️ <b>ST-ML BLOCKED</b> — spread too wide on {SYMBOL}. "
               f"No order sent.")
            return

        # ── EXECUTE ────────────────────────────────────────────────
        direction = 1 if sig["direction"] == "LONG" else -1
        sl        = sig["sl_estimate"]
        tp        = sig["tp_estimate"] if sig["tp_estimate"] else float("nan")

        status(f"SENDING ORDER: {sig['direction']} {LIVE_LOT_SIZE} {SYMBOL} "
               f"SL={sl:.2f} TP={tp if not np.isnan(tp) else 'TRAIL'}")

        order_result = send_market_order(
            mt5       = mt5,
            symbol    = SYMBOL,
            direction = direction,
            lot       = LIVE_LOT_SIZE,
            sl        = sl,
            tp        = tp,
            magic     = LIVE_MAGIC,
            comment   = LIVE_ORDER_COMMENT,
        )

        if order_result["success"]:
            status(f"ORDER FILLED ✓ | ticket={order_result['ticket']} "
                   f"price={order_result['price']:.2f} "
                   f"vol={order_result['volume']}")
            sig["order_result"] = order_result
            tg_order_sent(sig, order_result)
        else:
            err(f"ORDER FAILED: {order_result['error']}")
            sig["order_result"] = order_result
            tg_order_failed(sig, order_result)

        # Update state file with order result
        with open(LIVE_STATE_FILE, "w") as f:
            json.dump(sig, f, indent=2, default=str)

    except Exception as e:
        err(f"Live execution exception: {e}")
        traceback.print_exc()
        tg_error("run_live execution", str(e))
    finally:
        mt5.shutdown()
        dbg("MT5 connection closed")


# =============================================================================
# MODEL PERSISTENCE  — save / load / age check
# =============================================================================

def save_model(result: dict, best_params: dict) -> None:
    """
    Persist the trained model bundle and best params to disk.
    Saves a timestamp so we can check model age on next run.
    """
    bundle = {
        "trained_at":  datetime.now(timezone.utc).isoformat(),
        "best_params": best_params,
        "result":      result,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PARAMS_FILE, "wb") as f:
        pickle.dump(best_params, f, protocol=pickle.HIGHEST_PROTOCOL)
    status(f"Model saved → {MODEL_FILE}  "
           f"(trained at {bundle['trained_at']})")


def load_model() -> tuple[dict | None, dict | None, bool]:
    """
    Load model bundle from disk if it exists and is not stale.
    Returns (result, best_params, is_fresh).
    is_fresh=True  → model loaded successfully and within age limit
    is_fresh=False → model missing, corrupt, or too old → caller must retrain
    """
    if not MODEL_FILE.exists() or not PARAMS_FILE.exists():
        dbg("No saved model found — will retrain")
        return None, None, False

    try:
        with open(MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
        with open(PARAMS_FILE, "rb") as f:
            best_params = pickle.load(f)
    except Exception as e:
        warn(f"Failed to load saved model ({e}) — will retrain")
        return None, None, False

    trained_at = datetime.fromisoformat(bundle["trained_at"])
    age_hours  = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600

    if MODEL_MAX_AGE_H > 0 and age_hours > MODEL_MAX_AGE_H:
        warn(f"Model is {age_hours:.1f}h old (max {MODEL_MAX_AGE_H}h) — retraining")
        return None, None, False

    status(f"Loaded saved model | trained {age_hours:.1f}h ago "
           f"({bundle['trained_at']})")

    # Print key metrics from saved model so we know what we loaded
    if bundle["result"] and "metrics" in bundle["result"]:
        m = bundle["result"]["metrics"]
        print(f"  Saved model metrics | "
              f"AUC={m.get('consensus_clf_auc', 0):.3f} "
              f"prec={m.get('consensus_precision', 0):.3f} "
              f"recall={m.get('consensus_recall', 0):.3f} "
              f"n={m.get('n_trades', 0)}")

    # Print best params so we know what strategy is loaded
    bp = best_params
    print(f"  Loaded params | rsi=({bp.get('rsi_lo')},{bp.get('rsi_hi')}) "
          f"ST({int(bp.get('st_period',0))},{bp.get('st_mult',0)}) "
          f"dir={'LONG' if bp.get('direction')==1 else 'SHORT'} "
          f"stop={bp.get('stop_mult')} tp={bp.get('tp_mode')}")

    return bundle["result"], best_params, True


def model_needs_retrain(force: bool = False) -> bool:
    """
    Returns True if we need to retrain.
    Checks: --retrain flag, missing file, or model age exceeded.
    """
    if force:
        dbg("--retrain flag set — forcing retrain")
        return True
    if not MODEL_FILE.exists():
        dbg("No model file found — retraining")
        return True
    if MODEL_MAX_AGE_H == 0:
        dbg("MODEL_MAX_AGE_H=0 — always retraining")
        return True
    try:
        with open(MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
        trained_at = datetime.fromisoformat(bundle["trained_at"])
        age_hours  = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600
        if age_hours > MODEL_MAX_AGE_H:
            dbg(f"Model age {age_hours:.1f}h > {MODEL_MAX_AGE_H}h — retraining")
            return True
        dbg(f"Model age {age_hours:.1f}h — within limit, loading from disk")
        return False
    except Exception as e:
        warn(f"Model age check failed ({e}) — retraining")
        return True


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MNQ SuperTrend ML — Research + Live Execution")
    parser.add_argument(
        "--live", action="store_true",
        help="Run in live signal/execution mode (skip full WF training)")
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retrain even if a fresh saved model exists")
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Suppress debug prints")
    args = parser.parse_args()

    global DEBUG
    if args.no_debug:
        DEBUG = False

    # ── DATA FETCH ──────────────────────────────────────────────────
    status("Connecting to MT5 and fetching data...")
    mt5 = initialize_mt5()
    try:
        d1 = fetch_rates(mt5, SYMBOL, DAILY_TF_MIN, UTC_FROM, UTC_TO)
        h1 = fetch_rates(mt5, SYMBOL, H1_TF_MIN,    UTC_FROM, UTC_TO)
    finally:
        mt5.shutdown()

    print(f"  D1 bars: {len(d1)}  |  H1 bars: {len(h1)}")

    # ── FEATURE ENGINEERING ─────────────────────────────────────────
    status("Building features...")
    d1f  = build_daily_features(d1)
    h1f  = build_h1_features(h1)

    status("Adding SuperTrend variants (Numba JIT — first call compiles)...")
    h1f  = add_all_supertrends(h1f)

    status("Merging timeframes...")
    full = (merge_mtf(h1f, d1f)
            .dropna(subset=["rsi10", "ema100"])
            .reset_index(drop=True))
    print(f"  Merged rows: {len(full)}")

    # ── LIVE MODE: load or retrain, then execute ───────────────────
    if args.live:
        status("LIVE MODE")

        if model_needs_retrain(force=args.retrain):
            # ── Retrain on last training window ───────────────────
            status("LIVE MODE: retraining model on last window...")
            last_start      = max(0, len(full) - WF_TRAIN_BARS - WF_OOS_BARS)
            last_train      = full.iloc[
                last_start : last_start + WF_TRAIN_BARS
            ].reset_index(drop=True)

            final_params_df = parameter_search_on_slice(last_train)
            if final_params_df.empty:
                err("No valid params on last training window — cannot go live.")
                sys.exit(1)

            best_params  = final_params_df.iloc[0].to_dict()
            mdf_final    = build_ml_dataset(last_train, best_params)
            final_result = train_consensus_models(mdf_final)

            if final_result is None:
                err("Model training failed — cannot go live.")
                sys.exit(1)

            m = final_result["metrics"]
            print(f"  New model | AUC={m['consensus_clf_auc']:.3f} "
                  f"prec={m['consensus_precision']:.3f} "
                  f"recall={m['consensus_recall']:.3f} "
                  f"n={m['n_trades']}")

            # Save to disk for future runs
            save_model(final_result, best_params)

            # Telegram alert — retrain complete
            tg_retrain_complete(m, best_params)

        else:
            # ── Load from disk — fast path (~1 second) ─────────────
            final_result, best_params, fresh = load_model()
            if not fresh:
                # Shouldn't happen given model_needs_retrain check, but guard anyway
                err("Model load failed after age check passed — aborting.")
                sys.exit(1)

        # Print last N trades and execute
        print_last_n_trades(full, best_params, final_result, LAST_N_TRADES)
        run_live(full, best_params, final_result)
        return

    # ── RESEARCH MODE: full walk-forward ────────────────────────────
    n_folds = (len(full) - WF_TRAIN_BARS) // WF_OOS_BARS
    print(f"  Estimated folds: {n_folds}")
    if n_folds < WF_MIN_FOLDS:
        warn(f"Only {n_folds} folds possible — consider reducing WF_TRAIN_BARS.")

    status("Starting walk-forward evaluation...")
    oos_trades = walk_forward_eval(full)

    if oos_trades.empty:
        err("No OOS trades generated.")
        return

    print_wf_report(oos_trades)

    # ── FINAL MODEL ─────────────────────────────────────────────────
    status("Training final model on last window...")
    last_start      = max(0, len(full) - WF_TRAIN_BARS - WF_OOS_BARS)
    last_train      = full.iloc[
        last_start : last_start + WF_TRAIN_BARS
    ].reset_index(drop=True)
    final_params_df = parameter_search_on_slice(last_train)

    if final_params_df.empty:
        err("No valid params on last training window.")
        return

    best_params  = final_params_df.iloc[0].to_dict()
    mdf_final    = build_ml_dataset(last_train, best_params)
    final_result = train_consensus_models(mdf_final)

    if final_result:
        m = final_result["metrics"]
        print(f"\n  Final model | AUC={m['consensus_clf_auc']:.3f} "
              f"prec={m['consensus_precision']:.3f} "
              f"recall={m['consensus_recall']:.3f}")
        if m["feature_rank"] is not None:
            print("\n  Top 15 features:")
            print(m["feature_rank"].head(15).to_string())

        # Save model so --live mode can load it without retraining
        save_model(final_result, best_params)

    # ── LAST N TRADES ────────────────────────────────────────────────
    print_last_n_trades(full, best_params, final_result, LAST_N_TRADES)

    # ── LIVE SIGNAL CHECK ────────────────────────────────────────────
    sig = scan_live_signal(full, best_params, final_result)
    status("CURRENT LIVE SIGNAL")
    print(json.dumps(sig, indent=2, default=str))

    # ── SAVE OUTPUTS ─────────────────────────────────────────────────
    oos_trades.to_csv(OUT_DIR / "mnq_oos_trades.csv", index=False)
    final_params_df.to_csv(OUT_DIR / "mnq_param_search.csv", index=False)
    with open(OUT_DIR / "mnq_live_signal.json", "w") as f:
        json.dump(sig, f, indent=2, default=str)

    status(f"Outputs saved to {OUT_DIR.resolve()}")
    print("  mnq_oos_trades.csv    — all OOS trades with ML scores")
    print("  mnq_param_search.csv  — ranked parameter combinations")
    print("  mnq_live_signal.json  — current bar signal")
    print("  mnq_model.pkl         — saved model (auto-loaded by --live)")
    print("  mnq_best_params.pkl   — saved best params")
    print()
    print("  LIVE MODE commands:")
    print("  python supertrend_ml.py --live            "
          "# load saved model, check signal, execute")
    print("  python supertrend_ml.py --live --retrain  "
          "# force retrain then execute")
    print()
    print("  Recommended cron setup:")
    print("  # Check signal every hour (uses saved model — fast):")
    print("  0 * * * * cd ~/projects/mt5-python && "
          ".venv/bin/python supertrend_ml.py --live --no-debug "
          ">> live_log.txt 2>&1")
    print()
    print("  # Force retrain every Sunday at 18:00 UTC (weekly refresh):")
    print("  0 18 * * 0 cd ~/projects/mt5-python && "
          ".venv/bin/python supertrend_ml.py --live --retrain --no-debug "
          ">> live_log.txt 2>&1")
    print()
    print("  # Full research re-validation every month (first Sunday):")
    print("  0 20 1-7 * 0 cd ~/projects/mt5-python && "
          ".venv/bin/python supertrend_ml.py --no-debug "
          ">> research_log.txt 2>&1")


if __name__ == "__main__":
    main()
