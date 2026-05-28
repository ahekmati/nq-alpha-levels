from __future__ import annotations

import atexit
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import MetaTrader5 as mt5


RULE_NAME = os.getenv("RULE_NAME", "early_short_proto_v1").strip()
DATA_FILE = Path(os.getenv("DATA_FILE", r"C:\mt5_proto\h1_execution_dataset_with_proto_v1.csv"))
LOG_DIR = Path(os.getenv("LOG_DIR", r"C:\mt5_proto\logs"))
STATE_FILE = LOG_DIR / "mt5_proto_live_state.json"
DECISION_LOG = LOG_DIR / "mt5_proto_live_decisions.jsonl"
RUN_LOG = LOG_DIR / "mt5_proto_live_terminal.log"
LOCK_FILE = LOG_DIR / "mt5_proto_live.lock"

MT5_PATH = os.getenv("MT5_PATH", "").strip()
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0")) if os.getenv("MT5_LOGIN", "").strip() else None
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "").strip()
MT5_SERVER = os.getenv("MT5_SERVER", "").strip()

MT5_SYMBOL = os.getenv("MT5_SYMBOL", "").strip()
MT5_SYMBOL_ROOT = os.getenv("MT5_SYMBOL_ROOT", "MNQ").strip().upper()
AUTO_CONTRACT_ROLLOVER = os.getenv("AUTO_CONTRACT_ROLLOVER", "1").strip() == "1"
TIMEFRAME_NAME = os.getenv("MT5_TIMEFRAME", "H1").strip().upper()

EXECUTION_ENABLED = os.getenv("EXECUTION_ENABLED", "0").strip() == "1"
DEBUG_MODE = os.getenv("DEBUG", "1").strip() == "1"
NO_COLOR = os.getenv("NO_COLOR", "0").strip() == "1"
DEVIATION = int(os.getenv("DEVIATION", "30"))
MAGIC = int(os.getenv("MAGIC", "9001001"))
LOT_SIZE = float(os.getenv("LOT_SIZE", "1.0"))
STRICT_NO_ENTRY_ON_ANY_OPEN_POSITION = os.getenv("STRICT_NO_ENTRY_ON_ANY_OPEN_POSITION", "1").strip() == "1"
STRICT_NO_FLIP = os.getenv("STRICT_NO_FLIP", "1").strip() == "1"
STRICT_ONE_POSITION_ONLY = os.getenv("STRICT_ONE_POSITION_ONLY", "1").strip() == "1"
LOG_ALL_TERMINAL_POSITIONS = os.getenv("LOG_ALL_TERMINAL_POSITIONS", "1").strip() == "1"

STALE_BAR_MAX_HOURS = float(os.getenv("STALE_BAR_MAX_HOURS", "8"))
ORDER_COMMENT_TAG = os.getenv("ORDER_COMMENT_TAG", "PROTO_LIVE").strip()
DEFAULT_STOP_POINTS = float(os.getenv("DEFAULT_STOP_POINTS", "300"))
DEFAULT_R_MULT = float(os.getenv("DEFAULT_R_MULT", "0"))
LOCK_MAX_AGE_HOURS = float(os.getenv("LOCK_MAX_AGE_HOURS", "12"))
REQUIRE_MARKET_OPEN_FOR_ENTRY = os.getenv("REQUIRE_MARKET_OPEN_FOR_ENTRY", "1").strip() == "1"
SELF_TEST_ONLY = os.getenv("SELF_TEST_ONLY", "0").strip() == "1"
ADVANCE_STATE_ON_PLACED = os.getenv("ADVANCE_STATE_ON_PLACED", "1").strip() == "1"

TIMEFRAME_TO_BAR_HOURS = {
    "M1": 1 / 60,
    "M5": 5 / 60,
    "M15": 15 / 60,
    "M30": 30 / 60,
    "H1": 1,
    "H4": 4,
    "D1": 24,
}
BAR_DURATION_HOURS = TIMEFRAME_TO_BAR_HOURS.get(TIMEFRAME_NAME, 1)

QUARTER_MONTHS = [3, 6, 9, 12]
MONTH_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}

LOG_DIR.mkdir(parents=True, exist_ok=True)


class C:
    RESET = "" if NO_COLOR else "\033[0m"
    BOLD = "" if NO_COLOR else "\033[1m"
    RED = "" if NO_COLOR else "\033[31m"
    GREEN = "" if NO_COLOR else "\033[32m"
    YELLOW = "" if NO_COLOR else "\033[33m"
    BLUE = "" if NO_COLOR else "\033[34m"
    MAGENTA = "" if NO_COLOR else "\033[35m"
    CYAN = "" if NO_COLOR else "\033[36m"
    BRIGHT_RED = "" if NO_COLOR else "\033[91m"
    BRIGHT_GREEN = "" if NO_COLOR else "\033[92m"
    BRIGHT_YELLOW = "" if NO_COLOR else "\033[93m"
    BRIGHT_CYAN = "" if NO_COLOR else "\033[96m"


class ColorFormatter(logging.Formatter):
    BASE_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LEVEL_COLORS = {
        logging.DEBUG: C.BLUE,
        logging.INFO: C.GREEN,
        logging.WARNING: C.BRIGHT_YELLOW,
        logging.ERROR: C.BRIGHT_RED,
        logging.CRITICAL: C.BOLD + C.RED,
    }

    def format(self, record):
        formatter = logging.Formatter(self.BASE_FORMAT, self.DATE_FORMAT)
        msg = formatter.format(record)
        color = self.LEVEL_COLORS.get(record.levelno, "")
        return f"{color}{msg}{C.RESET}"


logger = logging.getLogger("mt5_proto_live_windows_v5")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
logger.handlers.clear()

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
_console.setFormatter(ColorFormatter())
_file = logging.FileHandler(RUN_LOG, encoding="utf-8")
_file.setLevel(logging.DEBUG)
_file.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
logger.addHandler(_console)
logger.addHandler(_file)

_LOCK_HANDLE = None


@dataclass
class LiveState:
    last_processed_signal_time: str = ""
    last_resolved_symbol: str = ""
    last_action: str = ""
    last_note: str = ""

    def save(self, path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LiveState":
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


@dataclass
class Decision:
    timestamp_utc: str
    resolved_symbol: str
    signal_time: str
    latest_dataset_time: str
    latest_signal_value: int
    any_open_position: bool
    open_position_count: int
    terminal_open_position_count: int
    stale_data_blocked: bool
    action: str
    note: str
    execution_enabled: bool


def acquire_lock() -> None:
    global _LOCK_HANDLE
    now_utc = datetime.now(timezone.utc)
    if LOCK_FILE.exists():
        try:
            raw = LOCK_FILE.read_text(encoding="utf-8").strip()
            payload = json.loads(raw)
            lock_ts_raw = payload.get("timestamp_utc")
            lock_ts = datetime.fromisoformat(lock_ts_raw)
            if lock_ts.tzinfo is None:
                lock_ts = lock_ts.replace(tzinfo=timezone.utc)
            age_h = (now_utc - lock_ts).total_seconds() / 3600.0
            if age_h > LOCK_MAX_AGE_HOURS:
                logger.warning(f"Removing stale lock file age={age_h:.2f}h path={LOCK_FILE}")
                LOCK_FILE.unlink(missing_ok=True)
        except Exception:
            logger.warning(f"Lock file exists but is unreadable; removing stale lock: {LOCK_FILE}")
            try:
                LOCK_FILE.unlink(missing_ok=True)
            except Exception:
                pass
    try:
        _LOCK_HANDLE = open(LOCK_FILE, "x", encoding="utf-8")
    except FileExistsError:
        payload = ""
        try:
            payload = LOCK_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        raise RuntimeError(f"Another instance may already be running; lock file exists: {LOCK_FILE} payload={payload}")
    _LOCK_HANDLE.write(json.dumps({"pid": os.getpid(), "timestamp_utc": now_utc.isoformat()}))
    _LOCK_HANDLE.flush()


def release_lock() -> None:
    global _LOCK_HANDLE
    try:
        if _LOCK_HANDLE is not None:
            _LOCK_HANDLE.close()
    except Exception:
        pass
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception:
        pass
    _LOCK_HANDLE = None


atexit.register(release_lock)


def banner(title: str) -> None:
    line = "=" * 108
    logger.info(line)
    logger.info(title)
    logger.info(line)


def section(title: str) -> None:
    logger.info("")
    logger.info(f"{'-' * 36} {title} {'-' * 36}")


def kv(key: str, value: Any) -> None:
    logger.info(f"{key:<34}: {value}")


def dump_json(title: str, payload: Any) -> None:
    logger.debug(f"{title}: {json.dumps(payload, ensure_ascii=False, indent=2, default=str)}")


def append_decision(d: Decision) -> None:
    with DECISION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d


def equity_index_roll_date(year: int, month: int) -> date:
    return third_friday(year, month) - timedelta(days=4)


def next_quarter(year: int, month: int) -> Tuple[int, int]:
    for m in QUARTER_MONTHS:
        if m > month:
            return year, m
    return year + 1, 3


def current_or_next_active_quarter(now_utc: datetime) -> Tuple[int, int]:
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


def resolve_front_month_symbol(now_utc: datetime) -> str:
    if MT5_SYMBOL:
        return MT5_SYMBOL
    if not AUTO_CONTRACT_ROLLOVER:
        raise RuntimeError("MT5_SYMBOL is empty and AUTO_CONTRACT_ROLLOVER=0")
    year, month = current_or_next_active_quarter(now_utc)
    return f"{MT5_SYMBOL_ROOT}{MONTH_CODE[month]}{str(year)[-2:]}"


def init_mt5() -> Any:
    kwargs: Dict[str, Any] = {}
    if MT5_PATH:
        kwargs["path"] = MT5_PATH
    if MT5_LOGIN:
        kwargs["login"] = MT5_LOGIN
    if MT5_PASSWORD:
        kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER:
        kwargs["server"] = MT5_SERVER
    ok = mt5.initialize(**kwargs) if kwargs else mt5.initialize()
    kv("MT5 initialize()", ok)
    kv("MT5 last_error()", mt5.last_error())
    if not ok:
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")
    ti = mt5.terminal_info()
    if ti is None:
        raise RuntimeError("MT5 terminal_info() returned None after successful initialize")
    kv("MT5 terminal connected", getattr(ti, "connected", None))
    kv("MT5 trade_allowed", getattr(ti, "trade_allowed", None))
    kv("MT5 company", getattr(ti, "company", None))
    return ti


def ensure_symbol_ready(symbol: str) -> Any:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol not found in terminal: {symbol}")
    if not getattr(info, "visible", False):
        ok = mt5.symbol_select(symbol, True)
        kv("symbol_select()", ok)
        if not ok:
            raise RuntimeError(f"symbol_select failed for {symbol}; last_error={mt5.last_error()}")
        info = mt5.symbol_info(symbol)
        if info is None or not getattr(info, "visible", False):
            raise RuntimeError(f"Symbol still not visible after select: {symbol}")
    kv("Resolved symbol", symbol)
    kv("Symbol visible", getattr(info, "visible", None))
    kv("Symbol path", getattr(info, "path", None))
    return info


def get_symbol_trade_rules(symbol: str) -> Dict[str, Any]:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"No symbol_info for {symbol}")
    point = float(getattr(info, "point", 0.0) or 0.0)
    digits = int(getattr(info, "digits", 2) or 2)
    tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
    if tick_size <= 0:
        tick_size = point if point > 0 else 0.01
    stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
    freeze_level_points = int(getattr(info, "trade_freeze_level", 0) or 0)
    volume_min = float(getattr(info, "volume_min", 0.0) or 0.0)
    volume_max = float(getattr(info, "volume_max", 0.0) or 0.0)
    volume_step = float(getattr(info, "volume_step", 0.0) or 0.0)
    trade_mode = getattr(info, "trade_mode", None)
    min_stop_price = max(stops_level_points * point, tick_size)
    filling_mode = getattr(info, "filling_mode", None)
    if volume_max == 0:
        logger.warning(
            f"Broker returned volume_max=0 for {symbol}; treating as uncapped/unknown. "
            f"If this broker uses 0 to mean trading disabled, order_send/order_check should reject it downstream."
        )
    rules = {
        "point": point,
        "digits": digits,
        "tick_size": tick_size,
        "stops_level_points": stops_level_points,
        "freeze_level_points": freeze_level_points,
        "min_stop_price": min_stop_price,
        "filling_mode": filling_mode,
        "volume_min": volume_min,
        "volume_max": volume_max,
        "volume_step": volume_step,
        "trade_mode": trade_mode,
    }
    dump_json("Symbol trade rules", rules)
    return rules


def validate_startup() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")
    if LOT_SIZE <= 0:
        raise RuntimeError(f"LOT_SIZE must be > 0, got {LOT_SIZE}")
    if BAR_DURATION_HOURS <= 0:
        raise RuntimeError(f"Unsupported MT5_TIMEFRAME={TIMEFRAME_NAME}")


def load_dataset() -> pd.DataFrame:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    if "time" not in df.columns:
        raise RuntimeError("Dataset missing 'time' column")
    if RULE_NAME not in df.columns:
        raise RuntimeError(f"Dataset missing strategy column: {RULE_NAME}")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)
    kv("Dataset rows", len(df))
    kv("Dataset last bar", df["time"].iloc[-1].isoformat())
    return df


def get_latest_closed_bar(df: pd.DataFrame) -> pd.Series:
    now_utc = datetime.now(timezone.utc)
    bar_close_times = df["time"] + pd.to_timedelta(BAR_DURATION_HOURS, unit="h")
    closed = df[bar_close_times <= now_utc].copy()
    if closed.empty:
        raise RuntimeError("No closed bar available in dataset")
    row = closed.iloc[-1]
    kv("Latest closed signal bar", row["time"].isoformat())
    kv("Latest signal value", int(row[RULE_NAME]))
    return row


def is_stale_bar(bar_time: pd.Timestamp) -> Tuple[bool, float]:
    bar_close_time = bar_time.to_pydatetime() + timedelta(hours=BAR_DURATION_HOURS)
    age_hours = (datetime.now(timezone.utc) - bar_close_time).total_seconds() / 3600.0
    return age_hours > STALE_BAR_MAX_HOURS, age_hours


def get_positions_on_symbol(symbol: str):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        logger.warning(f"positions_get(symbol={symbol}) returned None; last_error={mt5.last_error()}")
        return []
    positions = list(positions)
    kv("Open positions on resolved symbol", len(positions))
    for p in positions:
        logger.info(
            f"SYMBOL_POS ticket={getattr(p,'ticket',None)} symbol={getattr(p,'symbol',None)} type={getattr(p,'type',None)} "
            f"volume={getattr(p,'volume',None)} price_open={getattr(p,'price_open',None)} magic={getattr(p,'magic',None)} "
            f"comment={getattr(p,'comment',None)}"
        )
    return positions


def get_all_terminal_positions():
    positions = mt5.positions_get()
    if positions is None:
        logger.warning(f"positions_get() returned None; last_error={mt5.last_error()}")
        return []
    positions = list(positions)
    kv("All terminal open positions", len(positions))
    if LOG_ALL_TERMINAL_POSITIONS:
        for p in positions:
            logger.info(
                f"TERMINAL_POS ticket={getattr(p,'ticket',None)} symbol={getattr(p,'symbol',None)} type={getattr(p,'type',None)} "
                f"volume={getattr(p,'volume',None)} price_open={getattr(p,'price_open',None)} magic={getattr(p,'magic',None)} "
                f"comment={getattr(p,'comment',None)}"
            )
    return positions


def position_type_name(pos_type: Any) -> str:
    _buy = getattr(mt5, "POSITION_TYPE_BUY", None)
    _sell = getattr(mt5, "POSITION_TYPE_SELL", None)
    if _buy is not None and pos_type == _buy:
        return "buy"
    if _sell is not None and pos_type == _sell:
        return "sell"
    return f"unknown:{pos_type}"


def terminal_has_opposite_position(terminal_positions: List[Any], desired_side: str) -> bool:
    if desired_side not in {"buy", "sell"}:
        return False
    opposite = "sell" if desired_side == "buy" else "buy"
    return any(position_type_name(getattr(p, "type", None)) == opposite for p in terminal_positions)


def terminal_has_same_direction_position(terminal_positions: List[Any], desired_side: str) -> bool:
    return any(position_type_name(getattr(p, "type", None)) == desired_side for p in terminal_positions)


def normalize_price(price: float, tick_size: float, digits: int) -> float:
    if tick_size <= 0:
        return round(float(price), digits)
    return round(round(float(price) / tick_size) * tick_size, digits)


def floor_to_tick(price: float, tick_size: float, digits: int) -> float:
    if tick_size <= 0:
        return round(float(price), digits)
    return round(math.floor(float(price) / tick_size) * tick_size, digits)


def ceil_to_tick(price: float, tick_size: float, digits: int) -> float:
    if tick_size <= 0:
        return round(float(price), digits)
    return round(math.ceil(float(price) / tick_size) * tick_size, digits)


def normalize_volume(volume: float, volume_min: float, volume_max: float, volume_step: float) -> float:
    v = float(volume)
    if volume_step > 0:
        v = round(round(v / volume_step) * volume_step, 8)
    if volume_min > 0:
        v = max(v, volume_min)
    if volume_max > 0:
        v = min(v, volume_max)
    return v


def market_points_to_price_distance(points_value: float, point_size: float, tick_size: float, digits: int) -> float:
    raw_distance = float(points_value) * float(point_size)
    if raw_distance <= 0:
        return 0.0
    ticks = max(1, math.ceil(raw_distance / tick_size))
    return round(ticks * tick_size, digits)


def candidate_filling_modes(symbol: str) -> List[int]:
    info = mt5.symbol_info(symbol)
    if info is None:
        return []

    _sym_fok = getattr(mt5, "SYMBOL_FILLING_FOK", None)
    _sym_ioc = getattr(mt5, "SYMBOL_FILLING_IOC", None)
    _sym_ret = getattr(mt5, "SYMBOL_FILLING_RETURN", None)
    _ord_fok = getattr(mt5, "ORDER_FILLING_FOK", None)
    _ord_ioc = getattr(mt5, "ORDER_FILLING_IOC", None)
    _ord_ret = getattr(mt5, "ORDER_FILLING_RETURN", None)

    symbol_to_order: Dict[Any, Any] = {}
    if _sym_fok is not None and _ord_fok is not None:
        symbol_to_order[_sym_fok] = _ord_fok
    if _sym_ioc is not None and _ord_ioc is not None:
        symbol_to_order[_sym_ioc] = _ord_ioc
    if _sym_ret is not None and _ord_ret is not None:
        symbol_to_order[_sym_ret] = _ord_ret

    candidates: List[int] = []
    symbol_mode = getattr(info, "filling_mode", None)
    if symbol_mode is not None and symbol_mode in symbol_to_order:
        preferred = symbol_to_order[symbol_mode]
        if preferred is not None:
            candidates.append(preferred)

    for value in [_ord_ret, _ord_ioc, _ord_fok]:
        if value is not None and value not in candidates:
            candidates.append(value)

    return candidates


def success_retcodes() -> set:
    return {
        x for x in [
            0,
            getattr(mt5, "TRADE_RETCODE_DONE", None),
            getattr(mt5, "TRADE_RETCODE_PLACED", None),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None),
            getattr(mt5, "TRADE_RETCODE_NO_CHANGES", None),
        ] if x is not None
    }


def filled_retcodes() -> set:
    return {
        x for x in [
            getattr(mt5, "TRADE_RETCODE_DONE", None),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None),
        ] if x is not None
    }


def placed_retcodes() -> set:
    return {
        x for x in [
            getattr(mt5, "TRADE_RETCODE_PLACED", None),
        ] if x is not None
    }


def choose_filling_mode_by_check(base_request: Dict[str, Any]) -> Tuple[Optional[int], Any]:
    last_check = None
    for filling_mode in candidate_filling_modes(base_request["symbol"]):
        req = dict(base_request)
        req["type_filling"] = filling_mode
        check = mt5.order_check(req)
        payload = check._asdict() if hasattr(check, "_asdict") else str(check)
        dump_json(f"order_check candidate filling={filling_mode}", payload)
        last_check = check
        if check is None:
            continue
        retcode = getattr(check, "retcode", None)
        if retcode in success_retcodes():
            return filling_mode, check
    return None, last_check


def is_market_open_for_symbol(symbol: str) -> Tuple[bool, str]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, "No tick available for symbol"
    bid = float(getattr(tick, "bid", 0.0) or 0.0)
    ask = float(getattr(tick, "ask", 0.0) or 0.0)
    tick_time = getattr(tick, "time", None)
    if bid <= 0 or ask <= 0:
        return False, f"Invalid bid/ask bid={bid} ask={ask}"
    if tick_time is None:
        return False, "Tick timestamp missing"
    tick_dt = datetime.fromtimestamp(int(tick_time), tz=timezone.utc)
    age_sec = (datetime.now(timezone.utc) - tick_dt).total_seconds()
    if age_sec > 300:
        return False, f"Latest tick is stale age_sec={age_sec:.0f}"
    return True, f"Market looks open bid={bid} ask={ask} tick_age_sec={age_sec:.0f}"


def run_startup_self_test(symbol: str, market_open_check: Optional[Tuple[bool, str]] = None) -> Dict[str, Any]:
    results: Dict[str, Any] = {"checks": [], "passed": True}

    def add_check(name: str, ok: bool, detail: str) -> None:
        results["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            results["passed"] = False

    add_check("data_file_exists", DATA_FILE.exists(), str(DATA_FILE))
    add_check("log_dir_exists", LOG_DIR.exists(), str(LOG_DIR))
    add_check("lot_size_positive", LOT_SIZE > 0, f"LOT_SIZE={LOT_SIZE}")
    add_check("timeframe_supported", TIMEFRAME_NAME in TIMEFRAME_TO_BAR_HOURS, f"TIMEFRAME_NAME={TIMEFRAME_NAME}")

    info = mt5.symbol_info(symbol)
    add_check("symbol_info_present", info is not None, f"symbol={symbol}")
    if info is not None:
        point = float(getattr(info, "point", 0.0) or 0.0)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        volume_step = float(getattr(info, "volume_step", 0.0) or 0.0)
        add_check("point_positive", point > 0, f"point={point}")
        add_check("tick_size_positive", tick_size > 0 or point > 0, f"tick_size={tick_size} point={point}")
        add_check("volume_step_available", volume_step > 0, f"volume_step={volume_step}")

    if market_open_check is None:
        market_open_check = is_market_open_for_symbol(symbol)
    market_ok, market_msg = market_open_check
    add_check("market_open_check", (market_ok or not REQUIRE_MARKET_OPEN_FOR_ENTRY), market_msg)

    dump_json("Startup self-test", results)
    return results


def build_short_request(symbol: str, lot: float, stop_points: float, tp_points: float, signal_time: str) -> Dict[str, Any]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick for symbol {symbol}")
    rules = get_symbol_trade_rules(symbol)
    point = float(rules["point"])
    digits = int(rules["digits"])
    tick_size = float(rules["tick_size"])
    min_stop_price = float(rules["min_stop_price"])
    volume_min = float(rules["volume_min"])
    volume_max = float(rules["volume_max"])
    volume_step = float(rules["volume_step"])

    final_lot = normalize_volume(lot, volume_min, volume_max, volume_step)
    if abs(final_lot - float(lot)) > 1e-9:
        logger.warning(f"Adjusted LOT_SIZE from {lot} to broker-valid volume {final_lot}")
    if final_lot <= 0:
        raise RuntimeError(f"Normalized lot size is invalid: {final_lot}")

    entry_price = normalize_price(float(tick.bid), tick_size, digits)
    stop_distance = max(market_points_to_price_distance(stop_points, point, tick_size, digits), min_stop_price)

    tp_distance = 0.0
    if tp_points > 0:
        raw_tp_distance = market_points_to_price_distance(tp_points, point, tick_size, digits)
        if raw_tp_distance <= 0.0:
            logger.warning(
                f"TP distance rounded to zero for tp_points={tp_points} point={point} tick_size={tick_size}; "
                f"TP will be omitted (set to 0). Check tp_points / point / tick_size configuration."
            )
        tp_distance = raw_tp_distance

    sl = ceil_to_tick(entry_price + stop_distance, tick_size, digits)
    tp = floor_to_tick(entry_price - tp_distance, tick_size, digits) if tp_distance > 0 else 0.0

    if sl - entry_price < min_stop_price:
        sl = ceil_to_tick(entry_price + min_stop_price, tick_size, digits)
    if tp and entry_price - tp < min_stop_price:
        tp = floor_to_tick(entry_price - min_stop_price, tick_size, digits)

    kv("Entry price", entry_price)
    kv("Raw stop points", stop_points)
    kv("Converted stop distance", stop_distance)
    kv("Raw TP points", tp_points)
    kv("Converted TP distance", tp_distance)
    kv("Final SL", sl)
    kv("Final TP", tp if tp else "none (market exit / stop-time mode)")
    kv("Final lot", final_lot)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(final_lot),
        "type": mt5.ORDER_TYPE_SELL,
        "price": float(entry_price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": int(DEVIATION),
        "magic": int(MAGIC),
        "comment": f"{ORDER_COMMENT_TAG}|{RULE_NAME}|{signal_time}",
        "type_time": mt5.ORDER_TIME_GTC,
    }
    dump_json("Order request draft", request)
    return request


def count_positions_after_send(symbol: str) -> int:
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return len(list(positions))


def order_send_checked(base_request: Dict[str, Any]) -> Dict[str, Any]:
    chosen_fill, check = choose_filling_mode_by_check(base_request)
    if check is None:
        return {"success": False, "stage": "order_check", "retcode": None, "payload": None}
    check_payload = check._asdict() if hasattr(check, "_asdict") else str(check)
    check_retcode = getattr(check, "retcode", None)
    if check_retcode not in success_retcodes():
        return {"success": False, "stage": "order_check", "retcode": check_retcode, "payload": check_payload}

    if chosen_fill is None:
        fallback_candidates = candidate_filling_modes(base_request["symbol"])
        if not fallback_candidates:
            return {
                "success": False,
                "stage": "order_check",
                "retcode": check_retcode,
                "payload": check_payload,
                "error": "No valid filling mode candidates found for symbol; cannot send order",
            }
        chosen_fill = fallback_candidates[0]
        logger.warning(f"No filling mode passed order_check; falling back to first candidate: {chosen_fill}")

    request = dict(base_request)
    request["type_filling"] = chosen_fill
    dump_json("Order request final", request)

    if not EXECUTION_ENABLED:
        return {
            "success": True,
            "filled": False,
            "placed": False,
            "stage": "dry_run",
            "retcode": check_retcode,
            "payload": check_payload,
            "request": request,
        }

    before_count = count_positions_after_send(base_request["symbol"])
    result = mt5.order_send(request)
    if result is None:
        logger.error(f"mt5.order_send() returned None; last_error={mt5.last_error()}")
        return {
            "success": False,
            "filled": False,
            "placed": False,
            "opened_position": False,
            "stage": "order_send",
            "retcode": None,
            "payload": None,
            "request": request,
            "error": f"mt5.order_send() returned None; last_error={mt5.last_error()}",
        }
    result_payload = result._asdict() if hasattr(result, "_asdict") else str(result)
    dump_json("order_send payload", result_payload)
    retcode = getattr(result, "retcode", None)
    success = retcode in success_retcodes()
    filled = retcode in filled_retcodes()
    placed = retcode in placed_retcodes()
    after_count = count_positions_after_send(base_request["symbol"])
    opened_position = after_count > before_count

    return {
        "success": success,
        "filled": filled,
        "placed": placed,
        "opened_position": opened_position,
        "stage": "order_send",
        "retcode": retcode,
        "payload": result_payload,
        "request": request,
    }


def should_advance_state_after_entry(res: Dict[str, Any]) -> bool:
    if not EXECUTION_ENABLED:
        return True
    if res.get("opened_position", False) or res.get("filled", False):
        return True
    if ADVANCE_STATE_ON_PLACED and res.get("placed", False):
        return True
    return False


def main() -> None:
    acquire_lock()
    validate_startup()
    state = LiveState.load(STATE_FILE)
    now_utc = datetime.now(timezone.utc)
    resolved_symbol = resolve_front_month_symbol(now_utc)

    banner(f"START mt5_proto_live_windows_v5 now_utc={now_utc.isoformat()} symbol={resolved_symbol} execution_enabled={EXECUTION_ENABLED}")

    section("CONFIG")
    kv("RULE_NAME", RULE_NAME)
    kv("DATA_FILE", DATA_FILE)
    kv("LOG_DIR", LOG_DIR)
    kv("MT5_SYMBOL_ROOT", MT5_SYMBOL_ROOT)
    kv("AUTO_CONTRACT_ROLLOVER", AUTO_CONTRACT_ROLLOVER)
    kv("TIMEFRAME_NAME", TIMEFRAME_NAME)
    kv("BAR_DURATION_HOURS", BAR_DURATION_HOURS)
    kv("Resolved symbol at startup", resolved_symbol)
    kv("STRICT_NO_ENTRY_ON_ANY_OPEN_POSITION", STRICT_NO_ENTRY_ON_ANY_OPEN_POSITION)
    kv("STRICT_NO_FLIP", STRICT_NO_FLIP)
    kv("STRICT_ONE_POSITION_ONLY", STRICT_ONE_POSITION_ONLY)
    kv("REQUIRE_MARKET_OPEN_FOR_ENTRY", REQUIRE_MARKET_OPEN_FOR_ENTRY)
    kv("SELF_TEST_ONLY", SELF_TEST_ONLY)
    kv("ADVANCE_STATE_ON_PLACED", ADVANCE_STATE_ON_PLACED)
    kv("LOG_ALL_TERMINAL_POSITIONS", LOG_ALL_TERMINAL_POSITIONS)
    kv("STALE_BAR_MAX_HOURS", STALE_BAR_MAX_HOURS)
    kv("EXECUTION_ENABLED", EXECUTION_ENABLED)

    section("STATE")
    dump_json("Loaded state", asdict(state))

    section("MT5 INIT")
    ti = init_mt5()
    terminal_trade_allowed = getattr(ti, "trade_allowed", None)
    ensure_symbol_ready(resolved_symbol)

    initial_market_open_check = is_market_open_for_symbol(resolved_symbol)

    section("SELF TEST")
    self_test = run_startup_self_test(resolved_symbol, market_open_check=initial_market_open_check)
    for c in self_test["checks"]:
        kv(f"SELFTEST {c['name']}", f"ok={c['ok']} detail={c['detail']}")
    if not self_test["passed"]:
        raise RuntimeError("Startup self-test failed; refusing to continue")
    if SELF_TEST_ONLY:
        logger.info("SELF_TEST_ONLY=1 set; exiting after successful self-test")
        mt5.shutdown()
        return

    section("DATASET")
    df = load_dataset()
    latest = get_latest_closed_bar(df)
    stale, age_hours = is_stale_bar(latest["time"])
    kv("Latest closed bar age since close hours", round(age_hours, 3))
    kv("Stale data blocked", stale)

    section("POSITIONS")
    symbol_positions = get_positions_on_symbol(resolved_symbol)
    terminal_positions = get_all_terminal_positions()
    any_open_position = len(symbol_positions) > 0
    any_terminal_position = len(terminal_positions) > 0
    has_opposite_position = terminal_has_opposite_position(terminal_positions, "sell")
    has_same_direction_position = terminal_has_same_direction_position(terminal_positions, "sell")
    kv("Any open position on symbol", any_open_position)
    kv("Any open position in terminal", any_terminal_position)
    kv("Terminal has opposite position", has_opposite_position)
    kv("Terminal has same-direction position", has_same_direction_position)

    section("SESSION")
    initial_market_open, initial_market_open_msg = initial_market_open_check
    kv("Initial market open check", initial_market_open)
    kv("Initial market open detail", initial_market_open_msg)
    market_open, market_open_msg = is_market_open_for_symbol(resolved_symbol)
    kv("Decision-time market open check", market_open)
    kv("Decision-time market open detail", market_open_msg)

    section("DECISION")
    action = "hold"
    note = "No action taken"
    signal_time = latest["time"].isoformat()
    signal_value = int(latest[RULE_NAME])

    logger.info(
        f"DECISION DEBUG | resolved_symbol={resolved_symbol} | signal_time={signal_time} | signal_value={signal_value} "
        f"symbol_open_positions={len(symbol_positions)} | terminal_open_positions={len(terminal_positions)} | terminal_trade_allowed={terminal_trade_allowed}"
    )

    if stale:
        action = "blocked_stale_data"
        note = f"Latest closed bar is stale by {age_hours:.2f}h after bar close; refusing to trade"
        logger.warning(note)
    elif REQUIRE_MARKET_OPEN_FOR_ENTRY and not market_open:
        action = "blocked_market_closed_or_stale_tick"
        note = f"Market-open check failed: {market_open_msg}"
        logger.warning(note)
    elif any_open_position and STRICT_NO_ENTRY_ON_ANY_OPEN_POSITION:
        action = "blocked_existing_position"
        note = (
            "STRICT_NO_ENTRY_ON_ANY_OPEN_POSITION active: open position exists on resolved symbol; no new entry allowed. "
            "Note: STRICT_ONE_POSITION_ONLY would also block if enabled."
        )
        logger.warning(note)
    elif STRICT_ONE_POSITION_ONLY and any_terminal_position:
        action = "blocked_terminal_existing_position"
        note = "STRICT_ONE_POSITION_ONLY active: at least one terminal position exists; no new entry allowed"
        logger.warning(note)
    elif STRICT_NO_FLIP and has_opposite_position:
        action = "blocked_flip_risk"
        note = (
            "STRICT_NO_FLIP active: opposite-direction terminal position exists; refusing short entry. "
            "This branch is only reachable when STRICT_ONE_POSITION_ONLY is disabled or no terminal position blocks first."
        )
        logger.warning(note)
    elif signal_value != 1:
        action = "no_signal"
        note = "Latest closed bar does not contain a live short signal"
        logger.info(note)
    elif state.last_processed_signal_time == signal_time and state.last_resolved_symbol == resolved_symbol:
        action = "duplicate_signal_bar"
        note = "This signal bar was already processed previously; no repeat entry allowed"
        logger.warning(note)
    elif terminal_trade_allowed is False:
        action = "blocked_terminal_trade_not_allowed"
        note = "MT5 terminal reports trade_allowed=False; refusing entry"
        logger.warning(note)
    else:
        stop_points = float(latest.get("proto_stop_points", DEFAULT_STOP_POINTS) or DEFAULT_STOP_POINTS)
        tp_mode = str(latest.get("proto_mode", "stop_time") or "stop_time")
        r_mult = float(latest.get("proto_r_mult", DEFAULT_R_MULT) or DEFAULT_R_MULT)
        tp_points = stop_points * r_mult if tp_mode != "stop_time" and r_mult > 0 else 0.0
        logger.info(
            f"ENTRY DEBUG | resolved_symbol={resolved_symbol} | stop_points={stop_points} | tp_mode={tp_mode} | "
            f"r_mult={r_mult} | tp_points={tp_points} | lot={LOT_SIZE}"
        )
        req = build_short_request(resolved_symbol, LOT_SIZE, stop_points, tp_points, signal_time)
        res = order_send_checked(req)
        dump_json("Entry response", res)
        if res.get("success", False):
            action = "enter_short" if EXECUTION_ENABLED else "dry_run_entry_short"
            note = f"Entry request accepted at stage={res.get('stage')} retcode={res.get('retcode')}"
            if should_advance_state_after_entry(res):
                state.last_processed_signal_time = signal_time
                state.last_resolved_symbol = resolved_symbol
                state.last_action = action
                state.last_note = note
                state.save(STATE_FILE)
            else:
                logger.warning(
                    "Order accepted but no confirmed opened/filled/placed position state detected; state not advanced"
                )
            logger.info(note)
        else:
            action = "entry_failed"
            note = f"Entry failed at stage={res.get('stage')} retcode={res.get('retcode')}"
            logger.error(note)

    decision = Decision(
        timestamp_utc=now_utc.isoformat(),
        resolved_symbol=resolved_symbol,
        signal_time=signal_time,
        latest_dataset_time=df["time"].iloc[-1].isoformat(),
        latest_signal_value=signal_value,
        any_open_position=any_open_position,
        open_position_count=len(symbol_positions),
        terminal_open_position_count=len(terminal_positions),
        stale_data_blocked=stale,
        action=action,
        note=note,
        execution_enabled=EXECUTION_ENABLED,
    )
    append_decision(decision)
    dump_json("Decision", asdict(decision))

    if action not in {"enter_short", "dry_run_entry_short"}:
        state.last_action = action
        state.last_note = note
        state.last_resolved_symbol = resolved_symbol
        state.save(STATE_FILE)

    section("FINAL")
    logger.info(
        f"SUMMARY | SYMBOL={resolved_symbol} | SIGNAL={signal_value} | SYMBOL_OPEN_POSITIONS={len(symbol_positions)} | "
        f"TERMINAL_OPEN_POSITIONS={len(terminal_positions)} | ACTION={action} | NOTE={note}"
    )

    section("SHUTDOWN")
    logger.info("MT5 shutting down terminal connection")
    mt5.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("fatal error")
        raise
