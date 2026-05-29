"""
overnight_watcher.py
=====================
Runs overnight 16:00 ET → 09:30 ET next day.
Reads tonight_setup.json, monitors @MNQ price every 60s,
places buy limit orders when dip conditions are met,
and force-closes any open positions at 09:15 ET.

Safety guarantees:
  1. Never places an order if ANY position is open on @MNQ
     in the terminal (any magic, any strategy).
  2. Never places an order if ANY pending order already exists
     on @MNQ with this strategy's magic numbers.
  3. Maximum one filled position per study per night.
  4. All pending orders cancelled + positions closed at 09:15 ET.
  5. Stale setup file (wrong date) → logs warning, does NOT trade.
  6. MT5 disconnection mid-loop → retries, exits cleanly if unrecoverable.

Magic numbers:
  Study 1: 70260001
  Study 2: 70260002
  Starts with 7 — distinct from existing EAs which use 2xxxxxxx range.
  Watcher prints all active magics on startup for conflict verification.

Run via run_overnight.bat (started automatically after evaluator.py).
"""

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import MetaTrader5 as mt5

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SYMBOL         = "@MNQ"
ET             = ZoneInfo("America/New_York")

_ROOT          = Path(__file__).resolve().parent.parent
SETUP_FILE     = _ROOT / "tonight_setup.json"
LOG_FILE       = _ROOT / "logs" / "watcher.log"
TRADE_LOG_FILE = _ROOT / "logs" / "trades.log"

# ── Unique magic numbers for this strategy ────────────────────
# Starts with 7 — distinct from all your existing EAs which use 2xxxxxxx.
# Study 1 = 70260001, Study 2 = 70260002
# Verify these are unused before deploying: watcher prints all active
# magics on startup via print_active_magics().
S1_MAGIC       = 70260001
S2_MAGIC       = 70260002
ALL_MAGICS     = {S1_MAGIC, S2_MAGIC}

# ── Timing ────────────────────────────────────────────────────
POLL_SECS      = 60       # price check interval
EOD_HOUR       = 9        # force-close hour (ET)
EOD_MIN        = 15       # force-close minute (ET)

# ── Order execution ───────────────────────────────────────────
SLIPPAGE       = 5        # max deviation in points
# AMP/Rithmic requires ORDER_FILLING_RETURN (integer 2) for pending orders.
# Do NOT use mt5.ORDER_FILLING_RETURN at module level before initialize().
FILLING_MODE   = 2        # mt5.ORDER_FILLING_RETURN

# MT5 reconnect
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_WAIT_SECS    = 30


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

# Separate trade log — one line per order/fill/close event
_trade_handler = logging.FileHandler(TRADE_LOG_FILE, encoding="utf-8")
_trade_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
trade_log = logging.getLogger("trades")
trade_log.setLevel(logging.INFO)
trade_log.addHandler(_trade_handler)
trade_log.propagate = False   # don't double-print to console


# ─────────────────────────────────────────────
# MT5 CONNECTION
# ─────────────────────────────────────────────

def connect() -> bool:
    """
    Initialize MT5 connection.
    Returns True on success, False on failure.
    Does NOT raise — caller decides whether to exit.
    """
    if not mt5.initialize():
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    acct = mt5.account_info()
    if acct is None:
        log.error("MT5 account_info() returned None after initialize().")
        mt5.shutdown()
        return False
    log.info(f"MT5 connected | account={acct.login} server={acct.server}")
    return True


def reconnect() -> bool:
    """
    Attempt to reconnect MT5 up to MAX_RECONNECT_ATTEMPTS times.
    Returns True if successful.
    """
    for attempt in range(1, MAX_RECONNECT_ATTEMPTS + 1):
        log.warning(f"Reconnect attempt {attempt}/{MAX_RECONNECT_ATTEMPTS}...")
        mt5.shutdown()
        time.sleep(RECONNECT_WAIT_SECS)
        if connect():
            log.info("Reconnect successful.")
            return True
    log.error("All reconnect attempts failed. Exiting watcher.")
    return False


# ─────────────────────────────────────────────
# PRICE HELPERS
# ─────────────────────────────────────────────

def get_bid_ask(symbol: str):
    """
    Returns (bid, ask) floats or (None, None) if tick unavailable.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None, None
    if tick.bid <= 0 or tick.ask <= 0:
        return None, None
    return float(tick.bid), float(tick.ask)


def round_to_tick(price: float, symbol: str) -> float:
    """
    Round price to the instrument's tick size.
    Returns original price if symbol info unavailable.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return price
    tick_size = info.trade_tick_size
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)


def current_session_label(now_et: datetime) -> str:
    """Return a human-readable session label for logging."""
    t = now_et.hour * 60 + now_et.minute
    if t >= 16 * 60:    return "Asian"       # 16:00–00:00
    if t < 7 * 60:      return "European"    # 00:00–07:00
    if t < 9 * 60 + 30: return "Pre-market"  # 07:00–09:30
    return "RTH"


def print_active_magics():
    """
    Scan all open positions and pending orders in the terminal and
    print every unique magic number currently in use.
    Called once on startup so you can verify no conflicts.
    """
    active_magics = {}

    positions = mt5.positions_get()
    if positions:
        for p in positions:
            key = p.magic
            active_magics.setdefault(key, []).append(
                f"POS symbol={p.symbol} ticket={p.ticket}"
            )

    orders = mt5.orders_get()
    if orders:
        for o in orders:
            key = o.magic
            active_magics.setdefault(key, []).append(
                f"ORD symbol={o.symbol} ticket={o.ticket}"
            )

    if not active_magics:
        log.info("MAGIC SCAN: no open positions or pending orders in terminal.")
        return

    log.info(f"MAGIC SCAN: {len(active_magics)} magic number(s) currently active:")
    for magic, entries in sorted(active_magics.items()):
        conflict = " ⚠ CONFLICT WITH THIS STRATEGY" if magic in ALL_MAGICS else ""
        log.info(f"  magic={magic}{conflict}")
        for e in entries:
            log.info(f"    {e}")

    # Hard check: if our own magics are already in use, something is wrong
    conflicts = ALL_MAGICS & set(active_magics.keys())
    if conflicts:
        log.error(
            f"CONFLICT DETECTED: magic(s) {conflicts} are already in use "
            f"by another position or order. "
            f"Change S1_MAGIC / S2_MAGIC before trading."
        )


# ─────────────────────────────────────────────
# POSITION / ORDER GUARDS
# ─────────────────────────────────────────────

def any_mnq_position_open(symbol: str) -> bool:
    """
    Returns True if ANY open position exists on symbol,
    regardless of magic number or which EA placed it.
    This is the primary safety guard against double-entry.
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        # None means error from MT5 — treat conservatively as "position exists"
        log.warning("positions_get() returned None — treating as position open.")
        return True
    return len(positions) > 0


def any_mnq_pending_order(symbol: str, magic: int) -> bool:
    """
    Returns True if a pending order already exists for this
    symbol + magic number combination.
    Prevents duplicate order placement on loop restart.
    """
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        log.warning("orders_get() returned None — treating as order exists.")
        return True
    return any(o.magic == magic for o in orders)


def get_position_by_magic(symbol: str, magic: int):
    """
    Returns the open position for symbol+magic, or None.
    """
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return None
    for p in positions:
        if p.magic == magic:
            return p
    return None


# ─────────────────────────────────────────────
# ORDER EXECUTION
# ─────────────────────────────────────────────

def place_buy_limit(symbol: str,
                    entry: float,
                    stop: float,
                    target: float,
                    magic: int,
                    comment: str) -> bool:
    """
    Place a GTC buy limit order with SL and TP.

    Pre-flight checks (all must pass before sending):
      1. Symbol info available
      2. No open positions on symbol (ANY magic)
      3. No existing pending order for this magic
      4. Entry price is valid (above stop, below market)
      5. SL and TP are on correct sides of entry

    Returns True if order accepted by broker.
    """
    # ── Pre-flight: symbol info ───────────────────────────────
    sym_info = mt5.symbol_info(symbol)
    if sym_info is None:
        log.error(f"BLOCKED: symbol_info({symbol}) returned None")
        return False

    # ── Pre-flight: no open positions on this symbol ──────────
    if any_mnq_position_open(symbol):
        log.warning(
            f"BLOCKED order magic={magic}: open position already exists "
            f"on {symbol} — will not add another position."
        )
        return False

    # ── Pre-flight: no duplicate pending order ────────────────
    if any_mnq_pending_order(symbol, magic):
        log.warning(
            f"BLOCKED order magic={magic}: pending order already exists "
            f"— skipping to avoid duplicate."
        )
        return False

    # ── Pre-flight: get current price for sanity check ────────
    bid, ask = get_bid_ask(symbol)
    if bid is None:
        log.error("BLOCKED: cannot get current price for pre-flight check.")
        return False

    mid = (bid + ask) / 2.0

    # ── Round all levels to tick ──────────────────────────────
    entry  = round_to_tick(entry,  symbol)
    stop   = round_to_tick(stop,   symbol)
    target = round_to_tick(target, symbol)

    # ── Pre-flight: level sanity checks ──────────────────────
    if entry <= 0 or stop <= 0 or target <= 0:
        log.error(f"BLOCKED: non-positive price level "
                  f"entry={entry} stop={stop} target={target}")
        return False

    if stop >= entry:
        log.error(f"BLOCKED: stop {stop} >= entry {entry} — invalid SL")
        return False

    if target <= entry:
        log.error(f"BLOCKED: target {target} <= entry {entry} — invalid TP")
        return False

    if entry >= mid:
        log.warning(
            f"BLOCKED: entry {entry} >= current mid {mid:.2f} — "
            f"price already below limit level, dip passed."
        )
        return False

    min_stop_distance = sym_info.trade_stops_level * sym_info.point
    if (entry - stop) < min_stop_distance:
        log.error(
            f"BLOCKED: SL distance {entry-stop:.2f} < broker minimum "
            f"{min_stop_distance:.2f}"
        )
        return False

    # ── Build and send order ──────────────────────────────────
    request = {
        "action"      : mt5.TRADE_ACTION_PENDING,
        "symbol"      : symbol,
        "volume"      : 1.0,
        "type"        : mt5.ORDER_TYPE_BUY_LIMIT,
        "price"       : entry,
        "sl"          : stop,
        "tp"          : target,
        "deviation"   : SLIPPAGE,
        "magic"       : magic,
        "comment"     : comment[:31],   # MT5 comment max 31 chars
        "type_time"   : mt5.ORDER_TIME_GTC,       # GTC — we cancel manually at 09:15
        "type_filling": FILLING_MODE,
    }

    result = mt5.order_send(request)

    if result is None:
        log.error(f"order_send() returned None: {mt5.last_error()}")
        return False

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error(
            f"Order rejected | retcode={result.retcode} "
            f"comment='{result.comment}' | "
            f"entry={entry} stop={stop} target={target} magic={magic}"
        )
        return False

    msg = (
        f"ORDER PLACED | magic={magic} comment={comment} | "
        f"entry={entry} sl={stop} tp={target} | "
        f"ticket={result.order}"
    )
    log.info(msg)
    trade_log.info(msg)
    return True


def cancel_pending_by_magic(symbol: str, magic: int):
    """Cancel all pending orders for symbol + magic."""
    orders = mt5.orders_get(symbol=symbol)
    if orders is None:
        return
    for o in orders:
        if o.magic != magic:
            continue
        req = {"action": mt5.TRADE_ACTION_REMOVE, "order": o.ticket}
        r   = mt5.order_send(req)
        if r and r.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"Cancelled pending order ticket={o.ticket} magic={magic}")
        else:
            err = mt5.last_error()
            log.warning(
                f"Failed to cancel order ticket={o.ticket}: "
                f"retcode={r.retcode if r else 'None'} err={err}"
            )


def market_close_position(pos, symbol: str) -> bool:
    """
    Market-close a specific open position.
    Uses the position's own ticket and magic for the close request.
    Returns True if close succeeded.
    """
    bid, ask = get_bid_ask(symbol)
    if bid is None:
        log.error(f"Cannot close ticket={pos.ticket} — no tick data.")
        return False

    # For a long position we sell at bid; for short we buy at ask
    if pos.type == mt5.POSITION_TYPE_BUY:
        close_type  = mt5.ORDER_TYPE_SELL
        close_price = bid
    else:
        close_type  = mt5.ORDER_TYPE_BUY
        close_price = ask

    request = {
        "action"      : mt5.TRADE_ACTION_DEAL,
        "symbol"      : symbol,
        "volume"      : pos.volume,
        "type"        : close_type,
        "position"    : pos.ticket,
        "price"       : close_price,
        "deviation"   : SLIPPAGE,
        "magic"       : pos.magic,    # use position's own magic, not hardcoded
        "comment"     : "ON_EOD_CLOSE",
        "type_filling": FILLING_MODE,
    }

    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        err = mt5.last_error()
        retcode = result.retcode if result else "None"
        log.error(
            f"Close failed ticket={pos.ticket} "
            f"retcode={retcode} err={err}"
        )
        return False

    msg = (
        f"POSITION CLOSED | ticket={pos.ticket} magic={pos.magic} "
        f"close_price={close_price} float_pnl={pos.profit:.2f}"
    )
    log.info(msg)
    trade_log.info(msg)
    return True


# ─────────────────────────────────────────────
# SETUP FILE HANDLING
# ─────────────────────────────────────────────

def load_setup() -> dict:
    """Load tonight_setup.json. Raises FileNotFoundError if missing."""
    if not SETUP_FILE.exists():
        raise FileNotFoundError(
            f"Setup file not found: {SETUP_FILE}\n"
            f"Run evaluator.py first."
        )
    with open(SETUP_FILE, encoding="utf-8") as f:
        return json.load(f)


def validate_setup_date(setup: dict) -> bool:
    """
    Returns True only if setup file is dated today.
    A stale file from a prior day is a hard block — we refuse to trade
    on yesterday's conditions.
    """
    setup_date = setup.get("date", "")
    today_str  = datetime.now(ET).strftime("%Y-%m-%d")
    if setup_date != today_str:
        log.error(
            f"STALE SETUP FILE: dated {setup_date}, today is {today_str}. "
            f"Evaluator must run before watcher. Refusing to trade."
        )
        return False
    return True


# ─────────────────────────────────────────────
# SESSION TIMING
# ─────────────────────────────────────────────

def is_overnight_window(now_et: datetime) -> bool:
    """
    True if we are in the overnight trading window:
    16:00 ET → 09:30 ET next day.
    """
    t = now_et.hour * 60 + now_et.minute
    return t >= 16 * 60 or t < 9 * 60 + 30


def is_eod_close_time(now_et: datetime) -> bool:
    """
    True at 09:15 ET — time to cancel pending orders
    and close any open positions before RTH open.
    Checked once; caller sets a flag to prevent re-triggering.
    """
    return (now_et.hour == EOD_HOUR and
            now_et.minute >= EOD_MIN and
            now_et.minute < 30)     # window: 09:15-09:29 ET


# ─────────────────────────────────────────────
# MAIN WATCHER LOOP
# ─────────────────────────────────────────────

def main():
    log.info("=" * 65)
    log.info("OVERNIGHT WATCHER  starting")
    log.info(f"  Strategy magic numbers: S1={S1_MAGIC}  S2={S2_MAGIC}")
    log.info(f"  Setup file: {SETUP_FILE}")

    # ── Connect ───────────────────────────────────────────────
    if not connect():
        log.error("Initial MT5 connection failed. Exiting.")
        sys.exit(1)

    # ── Load and validate setup ───────────────────────────────
    try:
        setup = load_setup()
    except FileNotFoundError as e:
        log.error(str(e))
        mt5.shutdown()
        sys.exit(1)
    except json.JSONDecodeError as e:
        log.error(f"Setup file JSON parse error: {e}")
        mt5.shutdown()
        sys.exit(1)

    # Hard stop on stale setup — do not trade yesterday's conditions
    if not validate_setup_date(setup):
        mt5.shutdown()
        sys.exit(1)

    s1 = setup.get("study1", {})
    s2 = setup.get("study2", {})

    log.info(f"  Study 1: {'ARMED  entry=' + str(s1.get('entry_price', '?')) if s1.get('armed') else 'not armed'}")
    log.info(f"  Study 2: {'ARMED  entry=' + str(s2.get('entry_price', '?')) if s2.get('armed') else 'not armed'}")

    if not setup.get("any_armed"):
        log.info("No setups armed tonight — watcher running but will not trade.")

    # ── Per-night state flags ─────────────────────────────────
    # These prevent duplicate orders and track fill status
    s1_order_placed  = False   # True once BUY LIMIT sent to broker
    s2_order_placed  = False
    s1_filled        = False   # True once limit order becomes a position
    s2_filled        = False
    eod_close_done   = False   # True once EOD cancel/close has run

    # On startup: check if a pending order already exists from a prior
    # watcher run that crashed/restarted — prevents duplicate placement
    if s1.get("armed") and any_mnq_pending_order(SYMBOL, S1_MAGIC):
        log.warning("S1: pending order already found on startup — "
                    "marking as placed to avoid duplicate.")
        s1_order_placed = True

    if s2.get("armed") and any_mnq_pending_order(SYMBOL, S2_MAGIC):
        log.warning("S2: pending order already found on startup — "
                    "marking as placed to avoid duplicate.")
        s2_order_placed = True

    # ── Print all active magics for conflict verification ─────
    print_active_magics()

    log.info(f"Entering poll loop ({POLL_SECS}s interval)...")
    consecutive_errors = 0

    # ── Main loop ─────────────────────────────────────────────
    while True:
        try:
            now_et = datetime.now(ET)

            # ── EOD: cancel pending + close positions ──────────
            if is_eod_close_time(now_et) and not eod_close_done:
                log.info(f"EOD close triggered at {now_et.strftime('%H:%M')} ET")

                for magic in [S1_MAGIC, S2_MAGIC]:
                    cancel_pending_by_magic(SYMBOL, magic)

                # Close any open positions from this strategy
                for magic in [S1_MAGIC, S2_MAGIC]:
                    pos = get_position_by_magic(SYMBOL, magic)
                    if pos:
                        log.info(f"Force-closing position magic={magic} "
                                 f"ticket={pos.ticket}")
                        market_close_position(pos, SYMBOL)

                eod_close_done = True
                log.info("EOD close complete. Watcher exiting.")
                break

            # ── Outside overnight window — sleep longer ────────
            if not is_overnight_window(now_et):
                log.debug(f"Outside overnight window "
                          f"({now_et.strftime('%H:%M')} ET) — waiting...")
                time.sleep(300)
                consecutive_errors = 0
                continue

            # ── Get current price ──────────────────────────────
            bid, ask = get_bid_ask(SYMBOL)
            if bid is None:
                consecutive_errors += 1
                log.warning(
                    f"No tick data (error #{consecutive_errors}) — "
                    f"retrying in 30s"
                )
                if consecutive_errors >= 10:
                    log.error("10 consecutive tick failures — attempting reconnect")
                    if not reconnect():
                        sys.exit(1)
                    consecutive_errors = 0
                time.sleep(30)
                continue

            consecutive_errors = 0
            mid  = (bid + ask) / 2.0
            sess = current_session_label(now_et)

            # ── Check fill status for placed orders ───────────
            if s1_order_placed and not s1_filled:
                pos = get_position_by_magic(SYMBOL, S1_MAGIC)
                if pos:
                    s1_filled = True
                    msg = (f"S1 FILLED | ticket={pos.ticket} "
                           f"open_price={pos.price_open:.2f} "
                           f"sl={pos.sl:.2f} tp={pos.tp:.2f} "
                           f"session={sess}")
                    log.info(msg)
                    trade_log.info(msg)

            if s2_order_placed and not s2_filled:
                pos = get_position_by_magic(SYMBOL, S2_MAGIC)
                if pos:
                    s2_filled = True
                    msg = (f"S2 FILLED | ticket={pos.ticket} "
                           f"open_price={pos.price_open:.2f} "
                           f"sl={pos.sl:.2f} tp={pos.tp:.2f} "
                           f"session={sess}")
                    log.info(msg)
                    trade_log.info(msg)

            # ── Study 1: arm and place ─────────────────────────
            if s1.get("armed") and not s1_order_placed:
                entry  = float(s1["entry_price"])
                stop   = float(s1["stop_price"])
                target = float(s1["target_price"])

                if mid > entry:
                    # Check position BEFORE attempting to place — log clearly
                    if any_mnq_position_open(SYMBOL):
                        # A position is open right now (could be from another EA
                        # or a previously filled S2). Do NOT disarm — just skip
                        # this poll cycle and check again next iteration.
                        log.info(
                            f"S1: position open on {SYMBOL} — holding order "
                            f"placement, will retry next cycle."
                        )
                    else:
                        placed = place_buy_limit(
                            SYMBOL, entry, stop, target,
                            S1_MAGIC,
                            f"S1_{setup['date']}"
                        )
                        if placed:
                            s1_order_placed = True
                            log.info(
                                f"S1 order live | waiting for fill at {entry} | "
                                f"current mid={mid:.2f} "
                                f"({mid - entry:.1f} pts above) | {sess}"
                            )
                        else:
                            # Broker rejected for a non-position reason
                            # (bad levels, connection issue, etc.)
                            # Disarm to prevent spamming rejected orders.
                            log.warning(
                                "S1: broker rejected order — disarming S1 tonight. "
                                "Check logs for rejection reason."
                            )
                            s1["armed"] = False
                else:
                    # Price already at or through the entry level —
                    # dip already happened before we got there.
                    log.info(
                        f"S1: mid {mid:.2f} already <= entry {entry} — "
                        f"dip passed, disarming S1 tonight."
                    )
                    s1["armed"] = False

            # ── Study 2: arm and place ─────────────────────────
            if s2.get("armed") and not s2_order_placed:
                entry  = float(s2["entry_price"])
                stop   = float(s2["stop_price"])
                target = float(s2["target_price"])

                if mid > entry:
                    if any_mnq_position_open(SYMBOL):
                        log.info(
                            f"S2: position open on {SYMBOL} — holding order "
                            f"placement, will retry next cycle."
                        )
                    else:
                        placed = place_buy_limit(
                            SYMBOL, entry, stop, target,
                            S2_MAGIC,
                            f"S2_{setup['date']}"
                        )
                        if placed:
                            s2_order_placed = True
                            log.info(
                                f"S2 order live | waiting for fill at {entry} | "
                                f"current mid={mid:.2f} "
                                f"({mid - entry:.1f} pts above) | {sess}"
                            )
                        else:
                            log.warning(
                                "S2: broker rejected order — disarming S2 tonight. "
                                "Check logs for rejection reason."
                            )
                            s2["armed"] = False
                else:
                    log.info(
                        f"S2: mid {mid:.2f} already <= entry {entry} — "
                        f"dip passed, disarming S2 tonight."
                    )
                    s2["armed"] = False

            # ── Heartbeat every 15 minutes ─────────────────────
            # Fires once per 15-min mark (sleep(60) ensures one hit per mark)
            if now_et.minute % 15 == 0:
                s1_status = (
                    "filled"  if s1_filled        else
                    "pending" if s1_order_placed   else
                    "armed"   if s1.get("armed")   else
                    "inactive"
                )
                s2_status = (
                    "filled"  if s2_filled        else
                    "pending" if s2_order_placed   else
                    "armed"   if s2.get("armed")   else
                    "inactive"
                )
                log.info(
                    f"♥ {now_et.strftime('%H:%M')} ET | "
                    f"mid={mid:.2f} | "
                    f"S1={s1_status} S2={s2_status} | "
                    f"session={sess}"
                )

            time.sleep(POLL_SECS)

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt received — running EOD cleanup...")
            for magic in [S1_MAGIC, S2_MAGIC]:
                cancel_pending_by_magic(SYMBOL, magic)
            break

        except Exception as e:
            consecutive_errors += 1
            log.exception(f"Unexpected error in main loop: {e}")
            if consecutive_errors >= 5:
                log.error("5 consecutive loop errors — attempting reconnect")
                if not reconnect():
                    sys.exit(1)
                consecutive_errors = 0
            time.sleep(30)

    mt5.shutdown()
    log.info("MT5 disconnected. WATCHER complete.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
