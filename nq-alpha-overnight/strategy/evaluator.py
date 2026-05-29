"""
overnight_evaluator.py
=======================
Runs once at 4PM ET (or shortly after RTH close).
Reads today's H1 + D1 bars from MT5, evaluates Study 1 and Study 2
conditions, and writes tonight_setup.json for the watcher.

Study 1 — Strong Up Day Overnight Dip
  Conditions (ALL required):
    - Daily RSI(10) >= 60
    - H1 close > H1 EMA(100)
    - H1 EMA(20) > H1 EMA(100)
    - RTH gain >= 0.8%
  Entry : RTH_close - 0.75x ATR  (buy limit)
  Stop  : entry - 1.0x ATR
  Target: entry + 2.0R

Study 2 — Overnight Rally → RTH Selloff → Overnight Dip
  Conditions (ALL required):
    - Prior overnight rally range >= 2.5x ATR
    - RTH session retraced >= 0.5% from overnight high
  Entry : RTH_close - 2.0x ATR  (buy limit)
  Stop  : entry - 1.0x ATR
  Target: entry + 2.5R

Output : tonight_setup.json  (read by watcher.py)

Run via Windows Task Scheduler at 16:05 ET daily.
"""

import json
import logging
import ssl
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import URLError
from zoneinfo import ZoneInfo       # Python 3.9+ (Windows 10+)

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SYMBOL       = "@MNQ"
# Use integer literals — same as levels_ml.py.
# Do NOT use mt5.TIMEFRAME_H1 etc at module level before initialize().
TIMEFRAME_H1 = 16385
TIMEFRAME_D1 = 16408
N_BARS_H1    = 500        # enough for EMA(100) warmup
N_BARS_D1    = 60         # enough for RSI(10) warmup (min 30, use 60 for safety)

ET           = ZoneInfo("America/New_York")

# Paths relative to repo root (two levels up from strategy/)
_ROOT        = Path(__file__).resolve().parent.parent
OUTPUT_FILE  = _ROOT / "tonight_setup.json"
LOG_FILE     = _ROOT / "logs" / "evaluator.log"

# ── Study 1 filters ───────────────────────────────────────────
DAILY_RSI_PERIOD   = 10
DAILY_RSI_MIN      = 60.0
H1_EMA_FAST        = 20
H1_EMA_SLOW        = 100
RTH_GAIN_MIN       = 0.008        # 0.8%

# ── Study 1 entry params ──────────────────────────────────────
S1_DIP_ATR_MULT    = 0.75
S1_STOP_ATR_MULT   = 1.0
S1_RR              = 2.0

# ── Study 2 filters ───────────────────────────────────────────
S2_RALLY_ATR_MIN   = 2.5          # overnight rally must be >= this * ATR
S2_SELLOFF_PCT_MIN = 0.005        # RTH must retrace >= 0.5% from overnight high

# ── Study 2 entry params ──────────────────────────────────────
S2_DIP_ATR_MULT    = 2.0
S2_STOP_ATR_MULT   = 1.0
S2_RR              = 2.5

ATR_PERIOD         = 14

# ── Telegram config ───────────────────────────────────────────
# Create telegram_config.json in the repo root (gitignored):
# {
#   "bot_token": "YOUR_BOT_TOKEN",
#   "chat_id":   "YOUR_CHAT_ID"
# }
TELEGRAM_CONFIG = _ROOT / "telegram_config.json"


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TELEGRAM ALERTS
# ─────────────────────────────────────────────

def load_telegram_config() -> tuple:
    """
    Load bot_token and chat_id from telegram_config.json.
    Returns (token, chat_id) or (None, None) if file missing or invalid.
    Never raises — Telegram failure must not block the strategy.
    """
    if not TELEGRAM_CONFIG.exists():
        return None, None
    try:
        with open(TELEGRAM_CONFIG, encoding="utf-8") as f:
            cfg = json.load(f)
        token   = cfg.get("bot_token", "").strip()
        chat_id = cfg.get("chat_id", "").strip()
        if not token or not chat_id:
            log.warning("Telegram config missing bot_token or chat_id.")
            return None, None
        return token, chat_id
    except Exception as e:
        log.warning(f"Telegram config load error: {e}")
        return None, None


def send_telegram(message: str) -> bool:
    """
    Send a message via Telegram Bot API using only stdlib urllib.
    Returns True if sent successfully.
    Never raises — failure is logged as warning only.
    """
    token, chat_id = load_telegram_config()
    if token is None:
        log.debug("Telegram not configured — skipping alert.")
        return False

    try:
        params  = urlencode({
            "chat_id"    : chat_id,
            "text"       : message,
            "parse_mode" : "HTML",
        }).encode("utf-8")
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        ctx     = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        req     = urlopen(url, data=params, timeout=10, context=ctx)
        resp    = json.loads(req.read().decode("utf-8"))
        if resp.get("ok"):
            log.info("Telegram alert sent.")
            return True
        else:
            log.warning(f"Telegram API error: {resp}")
            return False
    except URLError as e:
        log.warning(f"Telegram network error: {e}")
        return False
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False


# ─────────────────────────────────────────────
# MT5 HELPERS
# ─────────────────────────────────────────────

def connect():
    """Initialize MT5. Raises RuntimeError on failure."""
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    acct = mt5.account_info()
    if acct is None:
        raise RuntimeError("MT5 account_info() returned None after init.")
    log.info(f"MT5 connected | account={acct.login} server={acct.server}")


def get_bars(symbol: str, tf: int, n: int) -> pd.DataFrame:
    """
    Pull n bars for symbol/timeframe.
    Returns DataFrame with OHLCV + time_et columns.
    Raises ValueError if no data returned.
    """
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise ValueError(
            f"No data for {symbol} tf={tf}: {mt5.last_error()}"
        )
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open", "high": "High",
        "low": "Low",   "close": "Close",
        "tick_volume": "Volume",
    }, inplace=True)
    df["time_et"] = df.index.tz_convert(ET)
    return df[["Open", "High", "Low", "Close", "Volume", "time_et"]].copy()


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_h1_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR, EMA20, EMA100 to H1 dataframe. Returns a new copy."""
    df = df.copy()
    df["atr"]    = calc_atr(df)
    df["ema20"]  = calc_ema(df["Close"], H1_EMA_FAST)
    df["ema100"] = calc_ema(df["Close"], H1_EMA_SLOW)
    return df


# ─────────────────────────────────────────────
# SESSION HELPERS
# ─────────────────────────────────────────────

def last_trading_day(date) -> "datetime.date":
    """
    Return the most recent trading day before 'date'.
    Skips Saturday (5) and Sunday (6).
    Works correctly for Monday → returns Friday.
    """
    prev = date - timedelta(days=1)
    while prev.weekday() >= 5:          # 5=Sat, 6=Sun
        prev -= timedelta(days=1)
    return prev


def rth_bars_today(df: pd.DataFrame, date) -> pd.DataFrame:
    """H1 bars for today's RTH session: 09:30–16:00 ET."""
    start = datetime(date.year, date.month, date.day, 9, 30, tzinfo=ET)
    end   = datetime(date.year, date.month, date.day, 16, 0, tzinfo=ET)
    mask  = (df["time_et"] >= start) & (df["time_et"] < end)
    return df[mask].copy()


def overnight_bars_prior(df: pd.DataFrame, date) -> pd.DataFrame:
    """
    H1 bars for the overnight session BEFORE today's RTH open:
    16:00 ET last trading day → 09:30 ET today.
    Handles Monday (prev = Friday) correctly.
    """
    prev  = last_trading_day(date)
    start = datetime(prev.year,  prev.month,  prev.day,  16, 0,  tzinfo=ET)
    end   = datetime(date.year,  date.month,  date.day,   9, 30, tzinfo=ET)
    mask  = (df["time_et"] >= start) & (df["time_et"] < end)
    return df[mask].copy()


# ─────────────────────────────────────────────
# STUDY 1 EVALUATION
# ─────────────────────────────────────────────

def evaluate_study1(df_h1: pd.DataFrame,
                    df_d1: pd.DataFrame,
                    today) -> dict:
    """
    Evaluate Study 1 conditions for today.
    Returns dict with 'armed' bool and all computed levels.
    """
    result = {"armed": False, "reasons": []}

    # ── RTH bars ──────────────────────────────────────────────
    rth = rth_bars_today(df_h1, today)
    if len(rth) < 2:
        result["reasons"].append("S1: insufficient RTH bars today")
        return result

    last      = rth.iloc[-1]
    first     = rth.iloc[0]
    rth_open  = float(first["Open"])
    rth_close = float(last["Close"])
    atr_val   = float(last["atr"])
    ema20     = float(last["ema20"])
    ema100    = float(last["ema100"])

    # Guard against bad indicator values
    if not np.isfinite(atr_val) or atr_val <= 0:
        result["reasons"].append(f"S1: ATR invalid ({atr_val})")
        return result
    if not np.isfinite(ema20) or not np.isfinite(ema100):
        result["reasons"].append("S1: EMA values not yet warmed up")
        return result
    if rth_open <= 0:
        result["reasons"].append("S1: RTH open price is zero")
        return result

    # ── Filter 1: Daily RSI ───────────────────────────────────
    rsi_series = calc_rsi(df_d1["Close"], DAILY_RSI_PERIOD)
    # Drop NaN from warmup period
    rsi_series = rsi_series.dropna()
    if len(rsi_series) == 0:
        result["reasons"].append("S1: daily RSI has no valid values")
        return result
    daily_rsi = float(rsi_series.iloc[-1])
    if not np.isfinite(daily_rsi):
        result["reasons"].append(f"S1: daily RSI is NaN/Inf")
        return result
    rsi_ok = daily_rsi >= DAILY_RSI_MIN

    # ── Filter 2: Price above EMA(100) ───────────────────────
    ema_ok = rth_close > ema100

    # ── Filter 3: EMA(20) > EMA(100) — trend aligned ─────────
    align_ok = ema20 > ema100

    # ── Filter 4: RTH gain >= 0.8% ────────────────────────────
    gain_pct = (rth_close - rth_open) / rth_open
    gain_ok  = gain_pct >= RTH_GAIN_MIN

    # Always write diagnostics to result regardless of pass/fail
    result.update({
        "daily_rsi"  : round(daily_rsi, 2),
        "ema20"      : round(ema20, 2),
        "ema100"     : round(ema100, 2),
        "gain_pct"   : round(gain_pct * 100, 3),
        "rth_open"   : round(rth_open, 2),
        "rth_close"  : round(rth_close, 2),
        "atr_val"    : round(atr_val, 2),
    })

    failures = []
    if not rsi_ok:   failures.append(
        f"RSI {daily_rsi:.1f} < {DAILY_RSI_MIN}")
    if not ema_ok:   failures.append(
        f"close {rth_close:.2f} <= EMA100 {ema100:.2f}")
    if not align_ok: failures.append(
        f"EMA20 {ema20:.2f} <= EMA100 {ema100:.2f}")
    if not gain_ok:  failures.append(
        f"RTH gain {gain_pct*100:.2f}% < {RTH_GAIN_MIN*100:.1f}%")

    if failures:
        result["reasons"] = [f"S1 not armed: {', '.join(failures)}"]
        return result

    # ── All filters passed — compute entry levels ──────────────
    entry  = round(rth_close - S1_DIP_ATR_MULT * atr_val, 2)
    stop   = round(entry     - S1_STOP_ATR_MULT * atr_val, 2)
    target = round(entry     + S1_RR * S1_STOP_ATR_MULT * atr_val, 2)
    risk   = round(entry - stop, 2)

    # Sanity check: entry must be below close, stop below entry,
    # target above entry
    if not (stop < entry < rth_close):
        result["reasons"].append(
            f"S1: level sanity check failed stop={stop} entry={entry} "
            f"close={rth_close}")
        return result
    if target <= entry:
        result["reasons"].append(
            f"S1: target {target} <= entry {entry}")
        return result

    result.update({
        "armed"        : True,
        "entry_price"  : entry,
        "stop_price"   : stop,
        "target_price" : target,
        "risk_pts"     : risk,
        "rr"           : S1_RR,
        "reasons"      : [
            f"S1 ARMED | RSI={daily_rsi:.1f} gain={gain_pct*100:.2f}% "
            f"EMA20={ema20:.0f} EMA100={ema100:.0f} | "
            f"entry={entry} stop={stop} target={target} "
            f"risk={risk}pts"
        ],
    })
    return result


# ─────────────────────────────────────────────
# STUDY 2 EVALUATION
# ─────────────────────────────────────────────

def evaluate_study2(df_h1: pd.DataFrame, today) -> dict:
    """
    Evaluate Study 2 conditions for today.
    Returns dict with 'armed' bool and all computed levels.
    """
    result = {"armed": False, "reasons": []}

    # ── Prior overnight bars ───────────────────────────────────
    prior_on = overnight_bars_prior(df_h1, today)
    if len(prior_on) < 3:
        result["reasons"].append(
            f"S2: only {len(prior_on)} prior overnight bars (need >= 3)")
        return result

    on_low   = float(prior_on["Low"].min())
    on_high  = float(prior_on["High"].max())
    on_range = on_high - on_low

    # Use ATR from the overnight bars (already computed on full df)
    avg_atr = float(prior_on["atr"].mean())

    if not np.isfinite(avg_atr) or avg_atr <= 0:
        result["reasons"].append(f"S2: overnight ATR invalid ({avg_atr:.4f})")
        return result
    if on_range <= 0:
        result["reasons"].append("S2: overnight range is zero")
        return result

    rally_mult = on_range / avg_atr

    # ── Filter 1: overnight rally >= 2.5x ATR ─────────────────
    rally_ok = rally_mult >= S2_RALLY_ATR_MIN

    # ── Today's RTH bars ──────────────────────────────────────
    rth = rth_bars_today(df_h1, today)
    if len(rth) < 2:
        result["reasons"].append("S2: insufficient RTH bars today")
        return result

    rth_close = float(rth.iloc[-1]["Close"])
    rth_atr   = float(rth.iloc[-1]["atr"])

    if not np.isfinite(rth_atr) or rth_atr <= 0:
        # Fall back to overnight ATR if RTH ATR is bad
        log.warning(f"S2: RTH ATR invalid ({rth_atr:.4f}), using overnight ATR")
        rth_atr = avg_atr

    # Selloff: how much did RTH retrace from overnight high
    if on_high <= 0:
        result["reasons"].append("S2: overnight high is zero")
        return result

    selloff_pts = on_high - rth_close
    selloff_pct = selloff_pts / on_high

    # ── Filter 2: RTH retraced >= 0.5% from overnight high ────
    selloff_ok = selloff_pct >= S2_SELLOFF_PCT_MIN

    # Always write diagnostics
    result.update({
        "rally_mult"   : round(rally_mult, 2),
        "on_high"      : round(on_high, 2),
        "on_low"       : round(on_low, 2),
        "on_range_pts" : round(on_range, 2),
        "rth_close"    : round(rth_close, 2),
        "selloff_pct"  : round(selloff_pct * 100, 3),
        "atr_val"      : round(rth_atr, 2),
    })

    failures = []
    if not rally_ok:   failures.append(
        f"overnight rally {rally_mult:.2f}x ATR < {S2_RALLY_ATR_MIN}x")
    if not selloff_ok: failures.append(
        f"RTH selloff {selloff_pct*100:.2f}% < {S2_SELLOFF_PCT_MIN*100:.1f}%")

    if failures:
        result["reasons"] = [f"S2 not armed: {', '.join(failures)}"]
        return result

    # ── All filters passed — compute entry levels ──────────────
    entry  = round(rth_close - S2_DIP_ATR_MULT * rth_atr, 2)
    stop   = round(entry     - S2_STOP_ATR_MULT * rth_atr, 2)
    target = round(entry     + S2_RR * S2_STOP_ATR_MULT * rth_atr, 2)
    risk   = round(entry - stop, 2)

    # Sanity checks
    if not (stop < entry < rth_close):
        result["reasons"].append(
            f"S2: level sanity check failed stop={stop} entry={entry} "
            f"close={rth_close}")
        return result
    if target <= entry:
        result["reasons"].append(
            f"S2: target {target} <= entry {entry}")
        return result

    result.update({
        "armed"        : True,
        "entry_price"  : entry,
        "stop_price"   : stop,
        "target_price" : target,
        "risk_pts"     : risk,
        "rr"           : S2_RR,
        "reasons"      : [
            f"S2 ARMED | rally={rally_mult:.2f}x ATR "
            f"selloff={selloff_pct*100:.2f}% | "
            f"entry={entry} stop={stop} target={target} "
            f"risk={risk}pts"
        ],
    })
    return result


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("OVERNIGHT EVALUATOR  starting")

    try:
        connect()
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    now_et = datetime.now(ET)
    today  = now_et.date()
    log.info(f"Date: {today}  ET time: {now_et.strftime('%H:%M:%S')}")

    # ── Fetch H1 data ─────────────────────────────────────────
    try:
        df_h1 = get_bars(SYMBOL, TIMEFRAME_H1, N_BARS_H1)
    except ValueError as e:
        log.error(f"H1 data fetch failed: {e}")
        mt5.shutdown()
        sys.exit(1)

    df_h1 = add_h1_indicators(df_h1)
    log.info(f"H1 bars: {len(df_h1)} | "
             f"{df_h1['time_et'].min().strftime('%Y-%m-%d')} → "
             f"{df_h1['time_et'].max().strftime('%Y-%m-%d %H:%M')}")

    # ── Fetch D1 data ─────────────────────────────────────────
    try:
        df_d1 = get_bars(SYMBOL, TIMEFRAME_D1, N_BARS_D1)
    except ValueError as e:
        log.error(f"D1 data fetch failed: {e}")
        mt5.shutdown()
        sys.exit(1)

    log.info(f"D1 bars: {len(df_d1)}")

    # ── Evaluate studies ──────────────────────────────────────
    s1 = evaluate_study1(df_h1, df_d1, today)
    s2 = evaluate_study2(df_h1, today)

    for msg in s1.get("reasons", []):
        log.info(msg)
    for msg in s2.get("reasons", []):
        log.info(msg)

    any_armed = s1["armed"] or s2["armed"]

    # ── Write setup file ──────────────────────────────────────
    setup = {
        "date"         : str(today),
        "evaluated_at" : now_et.isoformat(),
        "symbol"       : SYMBOL,
        "study1"       : s1,
        "study2"       : s2,
        "any_armed"    : any_armed,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=2)

    log.info(f"Setup written → {OUTPUT_FILE}")
    log.info(f"  Study 1: {'✅ ARMED' if s1['armed'] else '— not armed'}")
    log.info(f"  Study 2: {'✅ ARMED' if s2['armed'] else '— not armed'}")

    # ── Telegram alert ────────────────────────────────────────
    if any_armed:
        lines = [f"🎯 <b>NQ Alpha Overnight — {today}</b>"]

        if s1["armed"]:
            lines.append(
                f"\n📗 <b>Study 1 ARMED</b> (Trend Day Dip)\n"
                f"  Entry : {s1['entry_price']}\n"
                f"  Stop  : {s1['stop_price']}\n"
                f"  Target: {s1['target_price']}\n"
                f"  Risk  : {s1['risk_pts']} pts  |  RR {s1['rr']}:1\n"
                f"  RSI={s1.get('daily_rsi','?')}  "
                f"Gain={s1.get('gain_pct','?')}%  "
                f"ATR={s1.get('atr_val','?')}"
            )

        if s2["armed"]:
            lines.append(
                f"\n📘 <b>Study 2 ARMED</b> (Rally Reversal)\n"
                f"  Entry : {s2['entry_price']}\n"
                f"  Stop  : {s2['stop_price']}\n"
                f"  Target: {s2['target_price']}\n"
                f"  Risk  : {s2['risk_pts']} pts  |  RR {s2['rr']}:1\n"
                f"  Rally={s2.get('rally_mult','?')}x ATR  "
                f"Selloff={s2.get('selloff_pct','?')}%  "
                f"ATR={s2.get('atr_val','?')}"
            )

        lines.append("\n⏳ Watcher monitoring overnight...")
        send_telegram("\n".join(lines))
    else:
        log.info("No setups armed tonight — watcher will monitor but not trade.")

    mt5.shutdown()
    log.info("MT5 disconnected. Evaluator complete.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
