# =============================================================================
# supertrend_ml_win.py  (Windows native MT5 version)
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
#   MODE 2 — LIVE:  python supertrend_ml_win.py --live
#     • Fetches latest bars, rebuilds features, checks signal on last bar
#     • If signal fires AND ML confirms AND no open position → sends order
#     • Saves state to mnq_live_state.json for monitoring
#     • Run via Task Scheduler every hour
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

# ── Windows native MT5 ───────────────────────────────────────────────────────
import MetaTrader5 as _mt5_module
import sys, io

# Guard stdout/stderr re-wrap — Task Scheduler / redirected output may not
# expose .buffer, and wrapping a non-buffered stream raises AttributeError.
try:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        if hasattr(sys.stdout, "buffer"):
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass
try:
    if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
        if hasattr(sys.stderr, "buffer"):
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

class MetaTrader5:
    """
    Thin shim that wraps the Windows native MetaTrader5 module so the rest of
    the script can keep using the mt5linux instance-based calling style.
    """
    def __getattr__(self, name):
        return getattr(_mt5_module, name)

    def initialize(self, **kwargs):
        return _mt5_module.initialize(**kwargs)

    def shutdown(self):
        return _mt5_module.shutdown()

    def last_error(self):
        return _mt5_module.last_error()

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

# Suppress only convergence/deprecation warnings — not all warnings.
# Global suppression hides data leakage and sklearn warnings that matter.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# DEBUG HELPERS
# =============================================================================

DEBUG = True

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
    """Send a Telegram message. Silent no-op if not configured."""
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
            # Retry without HTML parse_mode — dynamic text may contain < > &
            payload2 = urllib.parse.urlencode({
                "chat_id": TELEGRAM_CHAT_ID,
                "text":    msg,
            }).encode()
            req2  = urllib.request.Request(url, data=payload2, method="POST")
            urllib.request.urlopen(req2, timeout=10)
        else:
            dbg(f"Telegram alert sent: {msg[:60]}...")
    except Exception as e:
        warn(f"Telegram alert failed (non-fatal): {e}")


def tg_signal_fired(sig: dict) -> None:
    tg(
        f"📡 <b>ST-ML Signal</b>\n"
        f"Time: {sig['time']}\n"
        f"Direction: {sig['direction']}\n"
        f"Close: {sig['close']:.2f}\n"
        f"RSI10: {sig['rsi10']:.1f} band={sig['rsi_band']}\n"
        f"ATR14: {sig['atr14']:.2f}\n"
        f"Entry est: {sig['entry_estimate']:.2f}\n"
        f"SL est: {sig['sl_estimate']:.2f}\n"
        f"Checking ML filter..."
    )


def tg_ml_filtered(sig: dict) -> None:
    tg(
        f"🚫 <b>ST-ML Signal FILTERED</b>\n"
        f"Time: {sig['time']}\n"
        f"ML consensus proba: {sig.get('consensus_proba', 0):.4f} "
        f"(threshold={CONSENSUS_PROBA_THRESHOLD})\n"
        f"Est net USD: ${sig.get('consensus_net_usd', 0):.0f}\n"
        f"No order sent."
    )


def tg_order_sent(sig: dict, order_result: dict) -> None:
    tp_val  = order_result.get('tp', 0)
    atr14   = sig.get('atr14', 0)
    tp_str  = f"{tp_val:.2f} (2 ATR = {2.0*atr14:.0f} pts)" if tp_val else 'NONE'
    tg(
        f"✅ <b>ST-ML ORDER FILLED</b>\n"
        f"Direction: {sig['direction']}\n"
        f"Symbol: {SYMBOL}\n"
        f"Ticket: {order_result.get('ticket', 'N/A')}\n"
        f"Fill price: {order_result.get('price', 0):.2f}\n"
        f"Volume: {order_result.get('volume', 0)}\n"
        f"SL: {order_result.get('sl', 0):.2f}\n"
        f"TP: {tp_str}\n"
        f"ML proba: {sig.get('consensus_proba', 0):.4f}\n"
        f"Est net USD: ${sig.get('consensus_net_usd', 0):.0f}"
    )


def tg_order_failed(sig: dict, order_result: dict) -> None:
    tg(
        f"❌ <b>ST-ML ORDER FAILED</b>\n"
        f"Direction: {sig['direction']}\n"
        f"Error: {order_result.get('error', 'unknown')}\n"
        f"Retcode: {order_result.get('retcode', 'N/A')}\n"
        f"Comment: {order_result.get('comment', '')}"
    )


def tg_blocked_position(positions: list) -> None:
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
    tg(f"🚨 <b>ST-ML ERROR</b>\n{context}\n{error}")


# =============================================================================
# USER CONFIG
# =============================================================================
SYMBOL       = "@MNQ"   # continuous contract — used for data fetching only
SYMBOL_ROOT  = "MNQ"    # root used to resolve front month for live trading
DAILY_TF_MIN = 1440
H1_TF_MIN    = 60

UTC_FROM = datetime(2018, 1, 1, tzinfo=timezone.utc)

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

# =============================================================================
# LIVE RSI BAND OVERRIDE
# =============================================================================
LIVE_RSI_BAND_OVERRIDE = (55, 65)   # set to None to use model's band
COMMISSION_PER_SIDE_USD   = 0.82
MNQ_DOLLARS_PER_POINT     = 2.0

# Live execution config
LIVE_LOT_SIZE        = 1
LIVE_MAGIC           = 20240101
LIVE_ORDER_COMMENT   = "ST_ML_v2"
LIVE_MAX_SPREAD_PTS  = 5.0
LAST_N_TRADES        = 5

# =============================================================================
# SESSION GATE CONFIG
# =============================================================================
ENFORCE_SESSION_GATE = True
SESSION_START_UTC    = 8
SESSION_END_UTC      = 21

# =============================================================================
# TELEGRAM CONFIG — load from environment variables
# =============================================================================
import os as _os
TELEGRAM_ENABLED  = True
TELEGRAM_TOKEN    = _os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID  = _os.environ.get("TELEGRAM_CHAT_ID", "")

WF_TRAIN_BARS = 5000
WF_OOS_BARS   = 1000
WF_MIN_FOLDS  = 3

OUT_DIR          = Path(".")
LIVE_STATE_FILE  = OUT_DIR / "mnq_live_state.json"

# =============================================================================
# MODEL PERSISTENCE CONFIG
# =============================================================================
MODEL_FILE      = OUT_DIR / "mnq_model.pkl"
PARAMS_FILE     = OUT_DIR / "mnq_best_params.pkl"
MODEL_MAX_AGE_H = 24


# =============================================================================
# MT5 HELPERS
# =============================================================================

def get_live_symbol(mt5: MetaTrader5, root: str = SYMBOL_ROOT) -> str:
    """
    Resolve the active front month contract by highest H1 tick_volume
    (not symbol_info_tick.volume which is last-trade size).
    Raises RuntimeError if no tradable front-month found — never falls
    back to @MNQ since that is data-only and not a tradable symbol.
    """
    try:
        symbols = mt5.symbols_get(f"{root}*")
        if not symbols:
            raise RuntimeError(
                f"No symbols found matching '{root}*' in MT5 — "
                f"cannot resolve front month for live trading")

        candidates = []
        for s in symbols:
            name = s.name
            if name.startswith("@") or len(name) > 8:
                continue
            # Use H1 tick_volume from recent bars — not symbol_info_tick.volume
            # which is just last-trade size and unreliable for contract selection
            try:
                rates = mt5.copy_rates_from(
                    name, _mt5_module.TIMEFRAME_H1,
                    datetime.now(timezone.utc), 5)
                if rates is None or len(rates) == 0:
                    continue
                vol = sum(r[5] for r in rates)   # index 5 = tick_volume
                candidates.append((vol, name))
            except Exception:
                continue

        if not candidates:
            raise RuntimeError(
                f"All '{root}*' contracts returned no H1 data — "
                f"cannot resolve front month. Check MT5 symbol subscription.")

        candidates.sort(reverse=True)
        front = candidates[0][1]
        dbg(f"Front month resolved: {front} (H1 tick_vol={candidates[0][0]})")
        return front

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"get_live_symbol failed: {e}") from e


def initialize_mt5() -> MetaTrader5:
    dbg("Initializing MT5 connection...")
    mt5 = MetaTrader5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    # Guard terminal_info — can return None on partial init
    info = mt5.terminal_info()
    if info is None:
        raise RuntimeError(
            f"MT5 initialized but terminal_info() returned None: "
            f"{mt5.last_error()}")
    dbg(f"MT5 connected | build={info.build} connected={info.connected}")
    return mt5


def _tf_map(mt5: MetaTrader5, mins: int) -> int:
    mapping = {
        1:    mt5.TIMEFRAME_M1,
        5:    mt5.TIMEFRAME_M5,
        15:   mt5.TIMEFRAME_M15,
        30:   mt5.TIMEFRAME_M30,
        60:   mt5.TIMEFRAME_H1,
        240:  mt5.TIMEFRAME_H4,
        1440: mt5.TIMEFRAME_D1,
    }
    if mins not in mapping:
        raise ValueError(
            f"Unsupported timeframe: {mins} minutes. "
            f"Valid values: {sorted(mapping.keys())}")
    return mapping[mins]


def fetch_rates(mt5: MetaTrader5, symbol: str, tf_mins: int,
                date_from: datetime, date_to: datetime) -> pd.DataFrame:
    dbg(f"Fetching {symbol} tf={tf_mins}min from {date_from.date()} to {date_to.date()}")
    rates = mt5.copy_rates_range(symbol, _tf_map(mt5, tf_mins), date_from, date_to)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates for {symbol} tf={tf_mins}. Error: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    # Validate required columns exist before feature engineering
    required = {"time", "open", "high", "low", "close", "tick_volume"}
    missing  = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"MT5 returned data missing required columns for {symbol}: {missing}")
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
    Raises RuntimeError if positions_get() returns None (MT5 query error) —
    never treats a query failure as "no positions" (fail-safe).
    """
    dbg(f"Checking ALL open positions for {symbol} (any magic)...")
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        raise RuntimeError(
            f"positions_get(symbol={symbol}) returned None — MT5 query error: "
            f"{mt5.last_error()}. Cannot confirm position state — aborting.")
    own   = [p for p in positions if p.magic == magic]
    other = [p for p in positions if p.magic != magic]
    dbg(f"Found {len(positions)} total positions on {symbol} "
        f"({len(own)} ours magic={magic}, {len(other)} other systems)")
    return list(positions)


def has_open_position(mt5: MetaTrader5, symbol: str, magic: int) -> bool:
    """
    Returns True if ANY open position exists on this symbol.
    Raises RuntimeError on MT5 query failure (fail-safe — never allows
    an order when position state is unknown).
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


def check_spread(mt5: MetaTrader5, symbol: str, max_spread_pts: float) -> bool:
    """
    Returns True if spread is acceptable.
    Compares spread in instrument points (ask - bid), not dollars.
    For MNQ with 0.25pt tick this is correct — 5.0pt max = 20 ticks.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        warn(f"Could not get tick for {symbol}: {mt5.last_error()}")
        return False
    spread = tick.ask - tick.bid
    dbg(f"Current spread: {spread:.2f} pts (max allowed: {max_spread_pts:.2f})")
    if spread > max_spread_pts:
        warn(f"Spread {spread:.2f} exceeds max {max_spread_pts:.2f} — skipping trade")
        return False
    return True


# =============================================================================
# ORDER EXECUTION
# =============================================================================

def _round_to_tick(price: float, tick_size: float, digits: int) -> float:
    """
    Round price to broker tick size, then to instrument digit precision.
    Uses tick_size (minimum price increment) not just decimal digits —
    on futures like MNQ (tick=0.25) digit-only rounding can produce
    invalid prices like 21450.10 instead of 21450.25.
    """
    if tick_size <= 0:
        tick_size = 0.25   # MNQ default
    return round(round(price / tick_size) * tick_size, digits)


def send_market_order(mt5: MetaTrader5, symbol: str, direction: int,
                      lot: float, sl: float, tp: float,
                      magic: int, comment: str) -> dict:
    """
    Send a market order with SL and TP.
    direction: +1 = BUY, -1 = SELL
    Includes a final position check immediately before order_send to
    close the race window between the upstream guard and execution.
    """
    dbg(f"Preparing order: {'BUY' if direction==1 else 'SELL'} {lot} {symbol} "
        f"SL={sl:.2f} TP={tp if not np.isnan(tp) else 'NONE'}")

    info = mt5.symbol_info(symbol)
    if info is None:
        return {"success": False, "error": f"Cannot get symbol info: {mt5.last_error()}"}

    tick_size = info.trade_tick_size if info.trade_tick_size > 0 else 0.25
    digits    = info.digits

    def rnd(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return 0.0
        return _round_to_tick(float(p), tick_size, digits)

    # ── FINAL GATE: re-check position immediately before order_send ──
    # Closes the race window between the upstream has_open_position check
    # and the actual send — another bot or manual trade could open here.
    final_check = mt5.positions_get(symbol=symbol)
    if final_check is None:
        return {"success": False,
                "error": f"FINAL GATE: positions_get returned None before send: "
                         f"{mt5.last_error()}"}
    if len(final_check) > 0:
        return {"success": False,
                "error": f"FINAL GATE: position appeared between check and send "
                         f"— aborting to prevent double entry"}

    # Fresh tick for order price
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return {"success": False, "error": f"Cannot get tick: {mt5.last_error()}"}

    price    = tick.ask if direction == 1 else tick.bid
    sl_clean = rnd(sl)
    tp_clean = rnd(tp) if not (isinstance(tp, float) and np.isnan(tp)) else 0.0

    order_type = mt5.ORDER_TYPE_BUY if direction == 1 else mt5.ORDER_TYPE_SELL

    # Determine best filling mode
    filling = mt5.ORDER_FILLING_RETURN
    sym_fm  = getattr(info, "filling_mode", 0)
    if sym_fm == getattr(mt5, "SYMBOL_FILLING_IOC", 2):
        filling = mt5.ORDER_FILLING_IOC
    elif sym_fm == getattr(mt5, "SYMBOL_FILLING_FOK", 1):
        filling = mt5.ORDER_FILLING_FOK

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       float(lot),
        "type":         order_type,
        "price":        price,
        "sl":           sl_clean,
        "tp":           tp_clean,
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": filling,
    }

    dbg(f"Order request: price={price} sl={sl_clean} tp={tp_clean} filling={filling}")

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        # Retry with alternate filling modes — re-fetch tick for fresh price
        for alt_filling in [mt5.ORDER_FILLING_IOC,
                            mt5.ORDER_FILLING_FOK,
                            mt5.ORDER_FILLING_RETURN]:
            if alt_filling == filling:
                continue
            # Re-fetch tick to get current price for retry
            retry_tick = mt5.symbol_info_tick(symbol)
            if retry_tick is None:
                continue
            request["price"]        = retry_tick.ask if direction == 1 else retry_tick.bid
            request["type_filling"] = alt_filling
            dbg(f"Retrying with filling={alt_filling} price={request['price']:.2f}...")
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                dbg(f"Order succeeded with filling mode {alt_filling}")
                break

    if result is None:
        err_msg = f"order_send returned None: {mt5.last_error()}"
        err(err_msg)
        return {"success": False, "error": err_msg}

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        err_msg = f"Order failed: retcode={result.retcode} comment={result.comment}"
        err(err_msg)
        return {"success": False, "retcode": result.retcode,
                "error": err_msg, "comment": result.comment}

    dbg(f"Order FILLED: ticket={result.order} price={result.price} volume={result.volume}")
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
    avg_up   = up.ewm(alpha=1/period, adjust=False).mean()
    avg_down = down.ewm(alpha=1/period, adjust=False).mean()
    # Replace 0 avg_down with a tiny epsilon instead of NaN
    # so uninterrupted up-moves produce RSI near 100 rather than NaN,
    # which would silently remove valid bullish-regime rows.
    avg_down = avg_down.replace(0, 1e-10)
    rs = avg_up / avg_down
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

    # Initialise direction based on price vs bands rather than hardcoding bullish.
    # Hardcoding direction[0]=1 can distort the first flip sequence on short samples.
    direction[0] = 1 if close[0] >= basic_lower[0] else -1
    st[0]        = final_lower[0] if direction[0] == 1 else final_upper[0]

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
    # Shift daily features by 1 so each H1 bar sees the PRIOR completed D1 bar
    # not the still-forming current daily candle — prevents look-ahead bias.
    d["trade_date"]   = d["time"].dt.floor("D") + pd.Timedelta(days=1)
    keep = [
        "trade_date", "close", "ema100", "rsi10", "atr14_d",
        "above_ema100", "ema_gap_pct", "rsi_slope3", "rsi_slope5",
        "d_ret1", "d_ret5", "d_ret10", "d_range_pct", "d_atr_pct",
        "atr_pctile", "adx14_d", "pdi14_d", "mdi14_d",
        "bb_bw_d", "bb_pctb_d", "cci20_d",
    ]
    dbg(f"Daily features built: {len(d)} rows (shifted +1 day to prevent look-ahead)")
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
    """
    Merge H1 and Daily frames on trade_date.
    daily.trade_date is already shifted +1 day in build_daily_features()
    so each H1 bar sees only the prior completed D1 bar (no look-ahead).
    """
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
    Returns positional (iloc) indices of rows satisfying all 3 entry conditions.
    For shorts, uses strictly_below_ema (above_ema100 == 0 AND close != ema100)
    to match the stated rule precisely.
    """
    tag      = f"st_{st_period}_{st_mult}"
    flip_col = f"flip_up_{tag}" if direction == 1 else f"flip_dn_{tag}"
    lo, hi   = rsi_band

    if flip_col not in df.columns:
        dbg(f"Flip column {flip_col} not found — skipping")
        return []

    if direction == 1:
        regime_ok = (df["above_ema100"] == 1)
    else:
        # Strictly below EMA100 — exclude equality (close == ema100)
        regime_ok = (df["above_ema100"] == 0) & (df.get("daily_close", df["close"]) != df.get("ema100", df["close"]))

    rsi_ok  = df["rsi10"].apply(lambda x: _rsi_in_band(x, lo, hi))
    flip_ok = df[flip_col] == 1
    mask    = regime_ok & rsi_ok & flip_ok
    return list(np.where(mask.values)[0])


# =============================================================================
# TRADE SIMULATION
# =============================================================================

def _simulate_trade(df: pd.DataFrame, pos: int, direction: int,
                    st_col: str, stop_atr_mult: float,
                    tp_mode: str, tp_param: float) -> dict | None:
    if pos < 0 or pos >= len(df):
        dbg(f"pos={pos} out of bounds for df len={len(df)}")
        return None

    row   = df.iloc[pos]
    entry = row["close"] + direction * SLIPPAGE_POINTS
    atrv  = row["atr14"]

    if pd.isna(atrv) or atrv <= 0:
        dbg(f"Invalid ATR={atrv} at pos={pos}")
        return None

    if direction == 1:
        swing_sl   = row.get("swing_low5", np.nan)
        st_sl      = row[st_col] if (
            st_col in df.columns and pd.notna(row.get(st_col))) else np.nan
        candidates = [entry - stop_atr_mult * atrv]
        if pd.notna(swing_sl): candidates.append(float(swing_sl))
        if pd.notna(st_sl):    candidates.append(float(st_sl))
        stop = min(candidates)
        stop = min(stop, entry - 0.25)
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

    if tp_mode == "fixed_rr":
        target = entry + direction * tp_param * risk
    elif tp_mode == "atr_multiple":
        target = entry + direction * tp_param * atrv
    else:
        target = np.nan

    trailing_stop = stop
    end_pos       = min(pos + MAX_HOLD_BARS, len(df) - 1)
    exit_px = exit_time = bars_held = None

    for j in range(pos + 1, end_pos + 1):
        r = df.iloc[j]

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
            # Note: this creates a small systematic bias vs live fills
            # where order of execution is unknown. Acknowledged — conservative
            # assumption is preferred for parameter ranking.
            exit_px = trailing_stop
        elif sl_hit:
            exit_px = trailing_stop - direction * SLIPPAGE_POINTS
        elif tp_hit:
            exit_px = target - direction * SLIPPAGE_POINTS

        if exit_px is not None:
            exit_time = r["time"]
            bars_held = j - pos
            break

    if exit_px is None:
        lr        = df.iloc[end_pos]
        exit_px   = lr["close"] - direction * SLIPPAGE_POINTS
        exit_time = lr["time"]
        bars_held = end_pos - pos

    gross_points = (exit_px - entry) * direction
    net_usd      = gross_points * MNQ_DOLLARS_PER_POINT - 2 * COMMISSION_PER_SIDE_USD
    win          = int(net_usd > 0)
    rr_realized  = gross_points / risk if risk > 0 else np.nan

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
# PARAMETER SEARCH
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
                            # Track open trade end position to prevent overlaps
                            # in parameter search — overlapping trades overstate
                            # count and corrupt parameter ranking.
                            next_valid_pos = 0
                            for pos in positions:
                                if pos < next_valid_pos:
                                    continue
                                tr = _simulate_trade(
                                    df, pos, direction, st_col,
                                    stop_mult, tp_mode, tp_param)
                                if tr:
                                    trades.append(tr)
                                    next_valid_pos = pos + tr["bars_held"] + 1

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
# FEATURE EXTRACTION
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
    dist = np.nan
    if (st_col in r.index
            and pd.notna(r.get(st_col))
            and r["close"] != 0):
        dist = (r["close"] - r[st_col]) / r["close"] * 100

    return {col: r.get(col) for col in FEATURE_COLS
            if col != "dist_to_st_pct"} | {"dist_to_st_pct": dist}


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

    # Guard: return empty DataFrame with correct columns if no trades
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS + ["time", "win", "net_usd",
                                                     "quality_win", "bars_held",
                                                     "gross_points", "rr_realized"])

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

    if y_c.nunique() < 2:
        dbg("Only one class in quality_win — skipping ML")
        return None

    # Scale n_splits to sample size — hardcoded 5 splits can fail on small datasets
    # even if they pass MIN_TRAIN_TRADES.
    n_splits = min(5, max(2, len(mdf) // (MIN_TRAIN_TRADES * 2)))
    tscv     = TimeSeriesSplit(n_splits=n_splits)
    dbg(f"TimeSeriesSplit n_splits={n_splits} (dataset size={len(mdf)})")

    clf_names  = ["logreg", "rf", "et", "gb"]
    reg_names  = ["ridge", "rfr", "gbr"]
    clf_models = {n: _make_clf(n) for n in clf_names}
    clf_oof    = pd.DataFrame(index=mdf.index)
    clf_scores = {}

    for name, model in clf_models.items():
        pred = pd.Series(np.nan, index=mdf.index)
        for tr_i, te_i in tscv.split(X):
            if y_c.iloc[tr_i].nunique() < 2:
                continue
            # Guard: skip fold if too few samples for reliable fitting
            if len(tr_i) < MIN_TRAIN_TRADES:
                continue
            try:
                model.fit(X.iloc[tr_i], y_c.iloc[tr_i])
                pred.iloc[te_i] = model.predict_proba(X.iloc[te_i])[:, 1]
            except Exception as e:
                dbg(f"Classifier {name} fold failed: {e}")
                continue
        valid = pred.dropna().index
        if len(valid) == 0 or y_c.loc[valid].nunique() < 2:
            dbg(f"Classifier {name}: no valid OOF predictions — skipping")
            continue
        try:
            clf_scores[name] = roc_auc_score(y_c.loc[valid], pred.loc[valid])
        except Exception as e:
            dbg(f"AUC computation failed for {name}: {e}")
            continue
        clf_oof[name] = pred
        model.fit(X, y_c)

    if not clf_scores:
        dbg("No classifiers trained successfully")
        return None

    scored_clfs          = list(clf_scores.keys())
    clf_oof["consensus"] = clf_oof[scored_clfs].mean(axis=1)
    valid_idx            = clf_oof["consensus"].dropna().index

    if len(valid_idx) == 0 or y_c.loc[valid_idx].nunique() < 2:
        dbg("Consensus has no valid predictions")
        return None

    preds_bin = (clf_oof.loc[valid_idx, "consensus"]
                 >= CONSENSUS_PROBA_THRESHOLD).astype(int)

    reg_models = {n: _make_reg(n) for n in reg_names}
    reg_scores = {}

    for name, model in reg_models.items():
        pred = pd.Series(np.nan, index=mdf.index)
        for tr_i, te_i in tscv.split(X):
            if len(tr_i) < MIN_TRAIN_TRADES:
                continue
            try:
                model.fit(X.iloc[tr_i], y_r.iloc[tr_i])
                pred.iloc[te_i] = model.predict(X.iloc[te_i])
            except Exception as e:
                dbg(f"Regressor {name} fold failed: {e}")
                continue
        valid = pred.dropna().index
        if len(valid) == 0:
            continue
        reg_scores[name] = mean_absolute_error(y_r.loc[valid], pred.loc[valid])
        model.fit(X, y_r)

    importances = {}
    for name in ["rf", "et", "gb"]:
        if name not in clf_scores:
            continue
        clf = clf_models[name].named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances[name] = pd.Series(clf.feature_importances_, index=avail_fc)

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
    feat = _extract_features(row, st_col)
    X    = pd.DataFrame([feat]).reindex(columns=feature_cols)

    # Warn if feature row is entirely NaN — signals a column mismatch
    if X.isnull().all(axis=1).iloc[0]:
        warn(f"score_row: all features are NaN for st_col={st_col} — "
             f"possible feature column mismatch. Scores may be unreliable.")

    clf_probs = {}
    for n, m in clf_models.items():
        try:
            clf_probs[n] = float(m.predict_proba(X)[0, 1])
        except Exception as e:
            dbg(f"score_row clf {n} failed: {e}")
            clf_probs[n] = 0.5   # neutral fallback

    reg_preds = {}
    for n, m in reg_models.items():
        try:
            reg_preds[n] = float(m.predict(X)[0])
        except Exception as e:
            dbg(f"score_row reg {n} failed: {e}")
            reg_preds[n] = 0.0

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
                "rsi10":             float(row.get("rsi10", np.nan)),
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

    last_positions = positions[-n:]

    for i, pos in enumerate(last_positions):
        tr = _simulate_trade(
            df, pos, direction, st_col,
            best_params["stop_mult"], best_params["tp_mode"], tp_param)

        if tr is None:
            print(f"\n  Trade {i+1}: INVALID SETUP at pos={pos}")
            continue

        row = df.iloc[pos]

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
        if not pd.isna(tr['target_px']):
            print(f"  Target      : {tr['target_px']:.2f}")
        else:
            print(f"  Target      : trailing SuperTrend")
        print(f"  Exit time   : {tr['exit_time']}")
        print(f"  Exit price  : {tr['exit_px']:.2f}")
        print(f"  Bars held   : {tr['bars_held']}")
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
    st_period = int(best_params["st_period"])
    st_mult_v = float(best_params["st_mult"])
    direction = int(best_params["direction"])

    if LIVE_RSI_BAND_OVERRIDE is not None:
        rsi_band = LIVE_RSI_BAND_OVERRIDE
        dbg(f"RSI band override active: {rsi_band} "
            f"(model selected ({best_params['rsi_lo']},{best_params['rsi_hi']}))")
    else:
        rsi_band = (best_params["rsi_lo"], best_params["rsi_hi"])

    st_col   = f"st_{st_period}_{st_mult_v}"
    flip_col = (f"flip_up_{st_col}" if direction == 1
                else f"flip_dn_{st_col}")

    latest    = full_df.iloc[-1]
    regime_ok = bool(latest.get("above_ema100") == 1) \
                if direction == 1 \
                else bool(latest.get("above_ema100") == 0)
    rsi_ok    = bool(_rsi_in_band(latest.get("rsi10"), *rsi_band))
    flip_ok   = bool(latest.get(flip_col, 0) == 1)

    dbg(f"Live bar: {latest['time']} close={latest['close']:.2f}")
    dbg(f"  regime_ok={regime_ok} rsi_ok={rsi_ok} flip_ok={flip_ok}")

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
        except Exception as e:
            sig["scoring_error"] = str(e)
            err(f"Signal scoring error: {e}")

    return sig


# =============================================================================
# SESSION GATE
# =============================================================================

def is_in_session(force: bool = False) -> bool:
    if not ENFORCE_SESSION_GATE or force:
        dbg("Session gate bypassed")
        return True
    now_hour   = datetime.now(timezone.utc).hour
    in_session = SESSION_START_UTC <= now_hour < SESSION_END_UTC
    dbg(f"Session gate: UTC hour={now_hour} "
        f"window={SESSION_START_UTC}:00-{SESSION_END_UTC}:00 "
        f"in_session={in_session}")
    return in_session


# =============================================================================
# LIVE EXECUTION ENGINE
# =============================================================================

def run_live(full_df: pd.DataFrame, best_params: dict,
             result: dict | None, force_session: bool = False) -> None:
    status("LIVE MODE — checking for trade signal")

    if not is_in_session(force=force_session):
        now_hour = datetime.now(timezone.utc).hour
        status(f"OUTSIDE SESSION HOURS — UTC hour={now_hour} "
               f"(window={SESSION_START_UTC}:00-{SESSION_END_UTC}:00). "
               f"No order sent.")
        return

    sig = scan_live_signal(full_df, best_params, result)

    print(f"\n  Bar time  : {sig['time']}")
    print(f"  Close     : {sig['close']:.2f}")
    print(f"  Direction : {sig['direction']}")
    print(f"  regime_ok : {sig['regime_ok']}")
    print(f"  rsi_ok    : {sig['rsi_ok']}  "
          f"(RSI10={sig['rsi10']:.1f} in {sig['rsi_band']})")
    print(f"  st_flip   : {sig['st_flip']}")
    print(f"  rule_sig  : {sig['rule_signal']}")
    if sig.get("consensus_proba") is not None:
        print(f"  ML proba  : {sig['consensus_proba']:.4f}  "
              f"(threshold={CONSENSUS_PROBA_THRESHOLD})")
        print(f"  ML net$   : ${sig['consensus_net_usd']:.2f}")
    print(f"  take_trade: {sig['take_trade']}")

    with open(LIVE_STATE_FILE, "w") as f:
        json.dump(sig, f, indent=2, default=str)

    if not sig["rule_signal"]:
        status("NO SIGNAL — rule conditions not met. No order sent.")
        return

    tg_signal_fired(sig)

    if not sig["take_trade"]:
        status("SIGNAL FILTERED BY ML — consensus proba below threshold.")
        tg_ml_filtered(sig)
        return

    status("Signal confirmed — connecting MT5 for position check...")
    mt5 = initialize_mt5()
    try:
        live_symbol = get_live_symbol(mt5, SYMBOL_ROOT)
        dbg(f"Live trading symbol: {live_symbol}")

        if has_open_position(mt5, live_symbol, LIVE_MAGIC):
            status("BLOCKED — open position already exists. No order sent.")
            positions = mt5.positions_get(symbol=live_symbol) or []
            tg_blocked_position(list(positions))
            return

        if not check_spread(mt5, live_symbol, LIVE_MAX_SPREAD_PTS):
            status("BLOCKED — spread too wide. No order sent.")
            tg(f"⚠️ ST-ML BLOCKED — spread too wide on {live_symbol}.")
            return

        direction = 1 if sig["direction"] == "LONG" else -1
        sl        = sig["sl_estimate"]
        atr14     = sig["atr14"]
        entry_est = sig["entry_estimate"]
        tp        = round(entry_est + direction * 2.0 * atr14, 2)

        status(f"SENDING ORDER: {sig['direction']} {LIVE_LOT_SIZE} {live_symbol} "
               f"SL={sl:.2f} TP={tp:.2f} (2 ATR = {2.0*atr14:.0f} pts)")

        order_result = send_market_order(
            mt5       = mt5,
            symbol    = live_symbol,
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
# MODEL PERSISTENCE
# =============================================================================

def save_model(result: dict, best_params: dict) -> None:
    bundle = {
        "trained_at":  datetime.now(timezone.utc).isoformat(),
        "best_params": best_params,
        "result":      result,
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PARAMS_FILE, "wb") as f:
        pickle.dump(best_params, f, protocol=pickle.HIGHEST_PROTOCOL)
    status(f"Model saved → {MODEL_FILE}  (trained at {bundle['trained_at']})")


def load_model() -> tuple[dict | None, dict | None, bool]:
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

    if bundle["result"] and "metrics" in bundle["result"]:
        m = bundle["result"]["metrics"]
        print(f"  Saved model metrics | "
              f"AUC={m.get('consensus_clf_auc', 0):.3f} "
              f"prec={m.get('consensus_precision', 0):.3f} "
              f"recall={m.get('consensus_recall', 0):.3f} "
              f"n={m.get('n_trades', 0)}")

    bp = best_params
    print(f"  Loaded params | rsi=({bp.get('rsi_lo')},{bp.get('rsi_hi')}) "
          f"ST({int(bp.get('st_period',0))},{bp.get('st_mult',0)}) "
          f"dir={'LONG' if bp.get('direction')==1 else 'SHORT'} "
          f"stop={bp.get('stop_mult')} tp={bp.get('tp_mode')}")

    return bundle["result"], best_params, True


def model_needs_retrain(force: bool = False) -> bool:
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
# RSI ZONE BREAKDOWN ANALYSIS
# =============================================================================

def print_rsi_zone_breakdown(oos_trades: pd.DataFrame) -> None:
    if oos_trades.empty or "rsi10" not in oos_trades.columns:
        print("\n[RSI ZONE] rsi10 column not found in OOS trades — skipping")
        return

    zones = [
        ("40-50",  40,  50),
        ("50-55",  50,  55),
        ("55-60",  55,  60),
        ("60-65",  60,  65),
        ("65-70",  65,  70),
        ("70-75",  70,  75),
        ("75-80",  75,  80),
        ("80+",    80, 100),
    ]

    sep = "=" * 75
    print(f"\n{sep}")
    print("RSI10 ZONE BREAKDOWN — OOS Trade Performance by Daily RSI at Entry")
    print(sep)
    print(f"  {'Zone':<8} {'N':>5} {'WR':>7} {'ExpectUSD':>10} {'PF':>7} "
          f"{'TotalUSD':>10} {'ML_N':>6} {'ML_WR':>7} {'ML_Exp':>9}")
    print("  " + "-" * 71)

    for label, lo, hi in zones:
        mask = (oos_trades["rsi10"] >= lo) & (oos_trades["rsi10"] < hi)
        sub  = oos_trades[mask]
        if len(sub) == 0:
            continue

        n    = len(sub)
        wins = (sub["net_usd"] > 0).sum()
        wr   = wins / n
        exp  = sub["net_usd"].mean()
        tot  = sub["net_usd"].sum()
        gl   = sub.loc[sub["net_usd"] < 0, "net_usd"].abs().sum()
        gp   = sub.loc[sub["net_usd"] > 0, "net_usd"].sum()
        pf   = gp / gl if gl > 0 else float("inf")

        ml_col = "take_trade_ml" if "take_trade_ml" in sub.columns else None
        if ml_col and (sub[ml_col] == 1).any():
            ml     = sub[sub[ml_col] == 1]
            ml_n   = len(ml)
            ml_w   = (ml["net_usd"] > 0).sum()
            ml_wr  = ml_w / ml_n if ml_n > 0 else 0
            ml_exp = ml["net_usd"].mean() if ml_n > 0 else 0
            ml_str = f"{ml_n:>6} {ml_wr:>7.1%} {ml_exp:>+9.0f}"
        else:
            ml_str = f"{'—':>6} {'—':>7} {'—':>9}"

        flag = " ◀ strong" if wr >= 0.65 and n >= 5 else ""
        print(f"  {label:<8} {n:>5} {wr:>7.1%} {exp:>+10.0f} {pf:>7.2f} "
              f"{tot:>+10.0f} {ml_str}{flag}")

    print(sep)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MNQ SuperTrend ML — Research + Live Execution")
    parser.add_argument("--live", action="store_true",
        help="Run in live signal/execution mode")
    parser.add_argument("--retrain", action="store_true",
        help="Force retrain even if a fresh saved model exists")
    parser.add_argument("--no-debug", action="store_true",
        help="Suppress debug prints")
    parser.add_argument("--force-session", action="store_true",
        help="Bypass session gate")
    args = parser.parse_args()

    global DEBUG
    if args.no_debug:
        DEBUG = False

    # UTC_TO evaluated at runtime, not import time
    utc_to = datetime.now(timezone.utc)

    # ── DATA FETCH ──────────────────────────────────────────────────
    status("Connecting to MT5 and fetching data...")
    mt5 = initialize_mt5()
    try:
        d1 = fetch_rates(mt5, SYMBOL, DAILY_TF_MIN, UTC_FROM, utc_to)
        h1 = fetch_rates(mt5, SYMBOL, H1_TF_MIN,    UTC_FROM, utc_to)
    finally:
        mt5.shutdown()

    print(f"  D1 bars: {len(d1)}  |  H1 bars: {len(h1)}")

    # ── FEATURE ENGINEERING ─────────────────────────────────────────
    status("Building features...")
    d1f = build_daily_features(d1)
    h1f = build_h1_features(h1)

    status("Adding SuperTrend variants (Numba JIT — first call compiles)...")
    h1f = add_all_supertrends(h1f)

    status("Merging timeframes...")
    full = (merge_mtf(h1f, d1f)
            .dropna(subset=["rsi10", "ema100"])
            .reset_index(drop=True))
    print(f"  Merged rows: {len(full)}")

    # ── WALK-FORWARD FOLD COUNT DIAGNOSTIC ──────────────────────────
    n_folds = max(0, (len(full) - WF_TRAIN_BARS) // WF_OOS_BARS)
    if n_folds < WF_MIN_FOLDS:
        warn(f"Only {n_folds} folds possible (min={WF_MIN_FOLDS}) — "
             f"consider reducing WF_TRAIN_BARS or fetching more history.")

    # ── LIVE MODE ───────────────────────────────────────────────────
    if args.live:
        status("LIVE MODE")

        if model_needs_retrain(force=args.retrain):
            status("LIVE MODE: retraining model on last window...")
            # Use the most recent WF_TRAIN_BARS — not offset by WF_OOS_BARS
            last_start      = max(0, len(full) - WF_TRAIN_BARS)
            last_train      = full.iloc[last_start:].reset_index(drop=True)

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

            save_model(final_result, best_params)
            tg_retrain_complete(m, best_params)

        else:
            final_result, best_params, fresh = load_model()
            if not fresh:
                err("Model load failed after age check passed — aborting.")
                sys.exit(1)

        print_last_n_trades(full, best_params, final_result, LAST_N_TRADES)
        run_live(full, best_params, final_result,
                 force_session=args.force_session)
        return

    # ── RESEARCH MODE ───────────────────────────────────────────────
    print(f"  Estimated folds: {n_folds}")

    status("Starting walk-forward evaluation...")
    oos_trades = walk_forward_eval(full)

    if oos_trades.empty:
        err("No OOS trades generated.")
        return

    print_wf_report(oos_trades)
    print_rsi_zone_breakdown(oos_trades)

    # ── FINAL MODEL — use most recent WF_TRAIN_BARS ─────────────────
    status("Training final model on last window...")
    last_start      = max(0, len(full) - WF_TRAIN_BARS)
    last_train      = full.iloc[last_start:].reset_index(drop=True)
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
        save_model(final_result, best_params)
    else:
        warn("Final model training failed — model not saved.")

    print_last_n_trades(full, best_params, final_result, LAST_N_TRADES)

    sig = scan_live_signal(full, best_params, final_result)
    status("CURRENT LIVE SIGNAL")
    print(json.dumps(sig, indent=2, default=str))

    oos_trades.to_csv(OUT_DIR / "mnq_oos_trades.csv", index=False)
    final_params_df.to_csv(OUT_DIR / "mnq_param_search.csv", index=False)
    with open(OUT_DIR / "mnq_live_signal.json", "w") as f:
        json.dump(sig, f, indent=2, default=str)

    # Only print saved-model commands if model was actually saved
    if final_result:
        status(f"Outputs saved to {OUT_DIR.resolve()}")
        print("  mnq_oos_trades.csv    — all OOS trades with ML scores")
        print("  mnq_param_search.csv  — ranked parameter combinations")
        print("  mnq_live_signal.json  — current bar signal")
        print("  mnq_model.pkl         — saved model (auto-loaded by --live)")
        print("  mnq_best_params.pkl   — saved best params")
        print()
        print("  LIVE MODE commands:")
        print("  python supertrend_ml_win.py --live")
        print("  python supertrend_ml_win.py --live --retrain")


if __name__ == "__main__":
    main()
