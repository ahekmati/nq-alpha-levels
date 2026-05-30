#!/usr/bin/env python3
from __future__ import annotations

import atexit
import json
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
import MetaTrader5 as MetaTrader5


AMP_URL = os.getenv("AMP_URL", "https://ampfutures.isystems.com/Systems/TopStrategies")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
TOP_N = int(os.getenv("TOP_N", "10"))
CONSENSUS_MIN = int(os.getenv("CONSENSUS_MIN", "3"))
STRONG_CONSENSUS_MIN = int(os.getenv("STRONG_CONSENSUS_MIN", "4"))

MT5_SYMBOL = os.getenv("MT5_SYMBOL", "").strip()
MT5_SYMBOL_ROOT = os.getenv("MT5_SYMBOL_ROOT", "MNQ").strip().upper()
AUTO_CONTRACT_ROLLOVER = os.getenv("AUTO_CONTRACT_ROLLOVER", "1").strip() == "1"

MT5_PATH = os.getenv("MT5_PATH", "").strip()
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0")) if os.getenv("MT5_LOGIN", "").strip() else None
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "").strip()
MT5_SERVER = os.getenv("MT5_SERVER", "").strip()

TIMEFRAME_NAME = os.getenv("MT5_TIMEFRAME", "H1").strip().upper()
DEVIATION = int(os.getenv("DEVIATION", "20"))
MAGIC = int(os.getenv("MAGIC", "26050902"))
BASE_LOT = float(os.getenv("BASE_LOT", "1.0"))
DOUBLE_LOT = float(os.getenv("DOUBLE_LOT", "2.0"))
ALLOW_DOUBLE_SIZE = os.getenv("ALLOW_DOUBLE_SIZE", "1").strip() == "1"
EXECUTION_ENABLED = True
ALLOW_DIRECT_FLIP = os.getenv("ALLOW_DIRECT_FLIP", "1").strip() == "1"
ENFORCE_ONE_POSITION = True

# Note: all positions on the symbol are checked before entry (raw_positions)
# so any open position from any system blocks new entries automatically.
STRATEGY_COMMENT_TAG = os.getenv("STRATEGY_COMMENT_TAG", "AMP").strip()

STOP_MODE = os.getenv("STOP_MODE", "dynamic").strip().lower()
STOP_POINTS = float(os.getenv("STOP_POINTS", "200"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
DYNAMIC_STOP_ATR_MULT = 2.0
MIN_DYNAMIC_STOP_POINTS = 0.0
MAX_DYNAMIC_STOP_POINTS = 20000.0

MAX_SL_PRICE_DISTANCE = float(os.getenv("MAX_SL_PRICE_DISTANCE", "200"))
SL_CAP_RETRIES = int(os.getenv("SL_CAP_RETRIES", "5"))
SL_CAP_RETRY_DELAY = float(os.getenv("SL_CAP_RETRY_DELAY", "1.0"))

ENABLE_TAKE_PROFIT = True
TP_MODE = "atr"
TAKE_PROFIT_POINTS = float(os.getenv("TAKE_PROFIT_POINTS", "400"))
TAKE_PROFIT_ATR_MULT = 12.0
MIN_TAKE_PROFIT_POINTS = 0.0
MAX_TAKE_PROFIT_POINTS = 120000.0

ENABLE_BREAKEVEN = False
BREAKEVEN_R_MULT = 999999.0
ENABLE_TRAILING_AFTER_BE = False
TRAIL_ATR_MULT = 0.0

ENABLE_SESSION_FILTER = True
ALLOWED_UTC_WINDOWS = os.getenv("ALLOWED_UTC_WINDOWS", "15-18")
BLOCK_DATES = {d.strip() for d in os.getenv("BLOCK_DATES", "").split(",") if d.strip()}

ONE_LOSS_PER_DAY = os.getenv("ONE_LOSS_PER_DAY", "1").strip() == "1"

DEBUG_MODE = os.getenv("DEBUG", "1").strip() == "1"
NO_COLOR = os.getenv("NO_COLOR", "0").strip() == "1"

LOG_DIR = Path(os.getenv("LOG_DIR", "./history"))
STATE_FILE = LOG_DIR / "amp_mt5_state.json"
DECISION_LOG = LOG_DIR / "amp_mt5_decisions.jsonl"
RUN_LOG = LOG_DIR / "amp_mt5_terminal.log"
LOCK_FILE = LOG_DIR / "amp_mt5.lock"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DAILY EMA100 FILTER CONFIG
# =============================================================================
# When enabled, short trades are blocked if the daily close is ABOVE the
# 100-period EMA on the daily chart. This prevents shorting into an uptrend.
# Long trades are never blocked by this filter.
# Set ENABLE_DAILY_EMA_FILTER = False to disable entirely.
# =============================================================================
ENABLE_DAILY_EMA_FILTER = True
DAILY_EMA_PERIOD        = 100    # EMA period on daily chart
DAILY_EMA_BARS          = 150    # bars to fetch (must be > DAILY_EMA_PERIOD)


def parse_allowed_windows(spec: str) -> List[Tuple[int, int]]:
    windows = []
    for raw in spec.split(","):
        part = raw.strip()
        if not part:
            continue
        try:
            start_s, end_s = part.split("-")
            start_h = int(start_s.strip())
            end_h = int(end_s.strip())
        except Exception:
            raise ValueError(f"Invalid ALLOWED_UTC_WINDOWS segment: '{part}'")
        if not (0 <= start_h <= 23 and 0 <= end_h <= 24 and start_h < end_h):
            raise ValueError(f"Invalid ALLOWED_UTC_WINDOWS hour range: '{part}'")
        windows.append((start_h, end_h))
    if not windows:
        raise ValueError("ALLOWED_UTC_WINDOWS produced no valid windows")
    return windows


PARSED_UTC_WINDOWS: List[Tuple[int, int]] = parse_allowed_windows(ALLOWED_UTC_WINDOWS)


class C:
    RESET = "" if NO_COLOR else "\033[0m"
    BOLD = "" if NO_COLOR else "\033[1m"
    DIM = "" if NO_COLOR else "\033[2m"
    RED = "" if NO_COLOR else "\033[31m"
    GREEN = "" if NO_COLOR else "\033[32m"
    YELLOW = "" if NO_COLOR else "\033[33m"
    BLUE = "" if NO_COLOR else "\033[34m"
    MAGENTA = "" if NO_COLOR else "\033[35m"
    CYAN = "" if NO_COLOR else "\033[36m"
    WHITE = "" if NO_COLOR else "\033[37m"
    BRIGHT_RED = "" if NO_COLOR else "\033[91m"
    BRIGHT_GREEN = "" if NO_COLOR else "\033[92m"
    BRIGHT_YELLOW = "" if NO_COLOR else "\033[93m"
    BRIGHT_BLUE = "" if NO_COLOR else "\033[94m"
    BRIGHT_MAGENTA = "" if NO_COLOR else "\033[95m"
    BRIGHT_CYAN = "" if NO_COLOR else "\033[96m"


class ColorFormatter(logging.Formatter):
    BASE_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    LEVEL_COLORS = {
        logging.DEBUG: C.DIM + C.BLUE,
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


logger = logging.getLogger("amp_mt5windows")
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


def acquire_lock() -> None:
    global _LOCK_HANDLE
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        _LOCK_HANDLE = open(LOCK_FILE, "x", encoding="utf-8")
    except FileExistsError:
        existing_pid = ""
        try:
            existing_pid = LOCK_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        raise RuntimeError(f"Another instance may already be running; lock file exists: {LOCK_FILE} pid={existing_pid}")
    _LOCK_HANDLE.write(str(os.getpid()))
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


def color_text(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}" if color and not NO_COLOR else text


def banner(title: str) -> None:
    line = "=" * 110
    logger.info(color_text(line, C.BRIGHT_CYAN))
    logger.info(color_text(title, C.BOLD + C.BRIGHT_CYAN))
    logger.info(color_text(line, C.BRIGHT_CYAN))


def section(title: str) -> None:
    line = "-" * 36
    logger.info("")
    logger.info(color_text(f"{line} {title} {line}", C.BOLD + C.BRIGHT_MAGENTA))


def kv(key: str, value: Any, level: str = "info") -> None:
    getattr(logger, level)(f"{key:<34}: {value}")


def dump_json(title: str, obj: Any, level: str = "debug") -> None:
    getattr(logger, level)(f"{title}: {json.dumps(obj, ensure_ascii=False, indent=2, default=str)}")


def side_color(side: str) -> str:
    side = (side or "").lower()
    if side in ("long", "enter", "flip"):
        return C.BRIGHT_GREEN
    if side in ("short",):
        return C.BRIGHT_RED
    if side in ("flat", "neutral", "idle", "hold", "close", "wait_new_amp_signal"):
        return C.BRIGHT_YELLOW
    if "failed" in side or "blocked" in side or "rejected" in side:
        return C.BRIGHT_RED
    return C.WHITE


def bool_color(flag: bool) -> str:
    return C.BRIGHT_GREEN if flag else C.BRIGHT_RED


def summary_line(
    amp_side: str,
    final_side: str,
    entry_allowed: bool,
    trade_allowed: Optional[bool],
    action: str,
    symbol: str,
) -> str:
    parts = [
        color_text("SUMMARY", C.BOLD + C.BRIGHT_CYAN),
        f"SYMBOL={symbol}",
        f"AMP={color_text(amp_side, side_color(amp_side))}",
        f"FINAL={color_text(final_side, side_color(final_side))}",
        f"ENTRY_ALLOWED={color_text(str(entry_allowed), bool_color(entry_allowed))}",
        f"TRADE_ALLOWED={color_text(str(trade_allowed), bool_color(bool(trade_allowed)) if trade_allowed is not None else C.BRIGHT_YELLOW)}",
        f"ACTION={color_text(action, side_color(action))}",
    ]
    return " | ".join(parts)


@dataclass
class ScrapedRow:
    rank: int
    system: str
    product: str
    pnl: float
    current_position: str
    nearest_order: str
    developer: str = ""


@dataclass
class Decision:
    timestamp_utc: str
    symbol: str
    amp_side: str
    amp_consensus_long: int
    amp_consensus_short: int
    amp_consensus_strength: int
    final_side: str
    final_lot: float
    strong_consensus: bool
    action: str
    note: str
    execution_enabled: bool
    duplicate_bar_blocked: bool
    daily_ema100: float = 0.0
    daily_close: float = 0.0
    daily_above_ema100: bool = False
    short_blocked_by_ema: bool = False


@dataclass
class LiveState:
    last_bar_time: str = ""
    last_signal_side: str = "flat"
    last_trade_day: str = ""
    loss_block_day: str = ""
    last_closed_ticket: int = 0
    consumed_amp_signal_id: str = ""

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LiveState":
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


def canonical_header(x: str) -> str:
    x = normalize_text(x).upper()
    x = x.replace("#", "RANK")
    x = x.replace("P/L", "PL")
    x = re.sub(r"[^A-Z0-9 ]+", "", x)
    return x.strip()


def header_matches(header: str, candidates: List[str]) -> bool:
    h = canonical_header(header)
    return any(canonical_header(c) == h for c in candidates)


def find_col(header_map: Dict[str, int], candidates: List[str]) -> Optional[int]:
    for h, idx in header_map.items():
        if header_matches(h, candidates):
            return idx
    return None


def money_to_float(s: str) -> Optional[float]:
    if s is None:
        return None
    s = s.replace("$", "").replace(",", "").strip()
    if s.startswith("(") and s.endswith(")"):
        s = f"-{s[1:-1]}"
    try:
        return float(s)
    except Exception:
        return None


def make_mt5() -> MetaTrader5:
    return MetaTrader5


QUARTER_MONTHS = [3, 6, 9, 12]
MONTH_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}


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


def infer_root_from_symbol(symbol: str) -> str:
    s = (symbol or "").upper()
    for root in ("MNQ", "NQ", "MES", "ES", "MYM", "YM"):
        if s.startswith(root):
            return root
    return MT5_SYMBOL_ROOT


def resolve_symbol(now_utc: datetime) -> str:
    if MT5_SYMBOL:
        return MT5_SYMBOL
    if not AUTO_CONTRACT_ROLLOVER:
        raise RuntimeError("MT5_SYMBOL is empty and AUTO_CONTRACT_ROLLOVER=0, cannot resolve trading symbol")
    year, month = current_or_next_active_quarter(now_utc)
    code = MONTH_CODE[month]
    yy = str(year)[-2:]
    return f"{MT5_SYMBOL_ROOT}{code}{yy}"


def fetch_amp_html() -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    kv("AMP URL", AMP_URL)
    r = requests.get(AMP_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    kv("AMP HTTP status", r.status_code)
    kv("AMP HTML bytes", len(r.text))
    r.raise_for_status()
    return r.text


def locate_amp_table(soup: BeautifulSoup):
    table = soup.find("table", id="tableCurrentSession")
    if table:
        logger.info("[AMP] Found tableCurrentSession directly")
        return table

    logger.warning("[AMP] tableCurrentSession not found, using header-based fallback")
    candidates = soup.find_all("table")
    best_score = -1
    best_table = None

    for idx, t in enumerate(candidates):
        headers = [normalize_text(th.get_text(" ", strip=True)).upper() for th in t.find_all("th")]
        header_set = set(h for h in headers if h)

        score = 0
        if "RANK" in header_set or "#" in header_set:
            score += 2
        if "SYSTEM" in header_set or "STRATEGY" in header_set:
            score += 2
        if "PRODUCT" in header_set or "MARKET" in header_set:
            score += 2
        if "P/L" in header_set or "PL" in header_set or "NET RESULT" in header_set:
            score += 1
        if "CURRENT POSITION" in header_set:
            score += 3
        if "NEAREST ORDER" in header_set:
            score += 3
        if "DEVELOPER" in header_set:
            score += 1

        logger.debug(f"[AMP] table[{idx}] score={score} headers={headers}")
        if score > best_score:
            best_score = score
            best_table = t

    if best_table is not None and best_score >= 6:
        logger.info(f"[AMP] Selected fallback table with score={best_score}")
        return best_table

    raise RuntimeError("Could not locate AMP current-session table in HTML")


def extract_header_map(table) -> Dict[str, int]:
    thead = table.find("thead")
    header_row = None

    if thead:
        for tr in thead.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if cells:
                header_row = tr
                break

    if header_row is None:
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if cells:
                header_row = tr
                break

    if header_row is None:
        raise RuntimeError("AMP table header row not found")

    headers = [normalize_text(cell.get_text(" ", strip=True)) for cell in header_row.find_all(["th", "td"])]
    header_map = {h: i for i, h in enumerate(headers) if h}
    dump_json("AMP header_map", header_map, "info")
    return header_map


def parse_current_session(html: str) -> List[ScrapedRow]:
    soup = BeautifulSoup(html, "html.parser")
    table = locate_amp_table(soup)
    header_map = extract_header_map(table)

    rank_idx = find_col(header_map, ["#", "Rank"])
    system_idx = find_col(header_map, ["System", "Strategy"])
    product_idx = find_col(header_map, ["Product", "Market"])
    developer_idx = find_col(header_map, ["Developer"])
    pnl_idx = find_col(header_map, ["P/L", "PL", "Net Result"])
    current_position_idx = find_col(header_map, ["Current Position"])
    nearest_order_idx = find_col(header_map, ["Nearest Order"])

    required = {
        "rank": rank_idx,
        "system": system_idx,
        "product": product_idx,
        "pnl": pnl_idx,
        "current_position": current_position_idx,
        "nearest_order": nearest_order_idx,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(f"AMP table missing required columns: {missing}")

    body = table.find("tbody") or table
    rows: List[ScrapedRow] = []

    for tr_idx, tr in enumerate(body.find_all("tr")):
        tds = tr.find_all("td")
        if not tds:
            continue

        text_cells = [normalize_text(td.get_text(" ", strip=True)) for td in tds]
        logger.debug(f"[AMP] row[{tr_idx}] cells={text_cells}")

        def cell(idx: Optional[int]) -> str:
            if idx is None:
                return ""
            return text_cells[idx] if idx < len(text_cells) else ""

        rank_text = cell(rank_idx)
        m = re.search(r"#?(\d+)", rank_text)
        if not m:
            logger.debug(f"[AMP] row[{tr_idx}] skipped: no rank in '{rank_text}'")
            continue

        rank = int(m.group(1))
        if rank > TOP_N:
            continue

        system = cell(system_idx)
        product = cell(product_idx)
        developer = cell(developer_idx)
        pnl_text = cell(pnl_idx)
        current_position = cell(current_position_idx)
        nearest_order = cell(nearest_order_idx)

        pnl = money_to_float(pnl_text)
        if pnl is None:
            logger.warning(f"[AMP] rank={rank} skipped: pnl parse failed from '{pnl_text}'")
            continue

        rows.append(
            ScrapedRow(
                rank=rank,
                system=system,
                product=product.upper(),
                pnl=pnl,
                current_position=current_position,
                nearest_order=nearest_order,
                developer=developer,
            )
        )

    kv("AMP parsed rows", len(rows))
    for r in rows:
        logger.info(
            f"[AMP] rank={r.rank} product={r.product} developer='{r.developer}' "
            f"pnl={r.pnl} pos='{r.current_position}' nearest='{r.nearest_order}' system='{r.system}'"
        )
    return rows


def product_root(product: str, system: str = "") -> Optional[str]:
    p = normalize_text(product).upper()
    s = normalize_text(system).upper()
    exact_map = {
        "NQ": "NQ", "MNQ": "MNQ",
        "ES": "ES", "MES": "MES",
        "YM": "YM", "MYM": "MYM",
    }
    if p in exact_map:
        return exact_map[p]

    combined = f"{p} {s}"
    patterns = [
        (r"\bMICRO\s+NASDAQ\b", "MNQ"),
        (r"\bE-?MINI\s+NASDAQ\b", "NQ"),
        (r"\bNASDAQ\b", "NQ"),
        (r"\bMICRO\s+E-?MINI\s+S&P\b", "MES"),
        (r"\bE-?MINI\s+S&P\b", "ES"),
        (r"\bS&P\s*500\b", "ES"),
        (r"\bMICRO\s+DOW\s*JONES\b", "MYM"),
        (r"\bDOW\s*JONES\b", "YM"),
    ]
    for pattern, root in patterns:
        if re.search(pattern, combined):
            return root
    return None


def consensus_roots() -> Set[str]:
    return {"NQ", "MNQ", "ES", "MES", "YM", "MYM"}


def parse_position_text(pos: str) -> str:
    p = normalize_text(pos).upper()
    if not p:
        return "unknown"
    if any(x in p for x in ["LONG", "BUY"]):
        return "long"
    if any(x in p for x in ["SHORT", "SELL"]):
        return "short"
    if any(x in p for x in ["FLAT", "NONE", "EXIT", "CLOSE", "NO POSITION", "SQUARE"]):
        return "flat"
    return "unknown"


def count_directional_consensus(rows: List[ScrapedRow], symbol_root: str) -> Dict[str, int]:
    long_count, short_count = 0, 0
    eligible_roots = consensus_roots()
    kv("Consensus mode", "full_us_index_cluster")
    kv("Consensus roots", sorted(eligible_roots))

    for r in rows:
        root = product_root(r.product, r.system)
        pos = parse_position_text(r.current_position)
        logger.debug(f"[AMP] consensus row rank={r.rank} root={root} pos={pos} product={r.product} system={r.system}")
        if root in eligible_roots and pos in ("long", "short"):
            if pos == "long":
                long_count += 1
            else:
                short_count += 1

    return {"long_count": long_count, "short_count": short_count}


def decide_amp_side(rows: List[ScrapedRow], symbol_root: str) -> Tuple[str, Dict[str, int], bool]:
    c = count_directional_consensus(rows, symbol_root)
    strength = max(c["long_count"], c["short_count"])
    strong = strength >= STRONG_CONSENSUS_MIN
    logger.info(
        f"[AMP] consensus long={c['long_count']} short={c['short_count']} "
        f"strength={strength} strong={strong}"
    )
    if c["long_count"] >= CONSENSUS_MIN and c["long_count"] > c["short_count"]:
        return "long", c, strong
    if c["short_count"] >= CONSENSUS_MIN and c["short_count"] > c["long_count"]:
        return "short", c, strong
    return "flat", c, False


def build_amp_signal_id(rows: List[ScrapedRow], amp_side: str) -> str:
    parts = [f"amp_side={amp_side}"]
    for r in rows[:TOP_N]:
        parts.append(
            f"{r.rank}|{normalize_text(r.system)}|{normalize_text(r.product)}|"
            f"{normalize_text(r.current_position)}|{normalize_text(r.nearest_order)}|"
            f"{normalize_text(r.developer)}"
        )
    return "||".join(parts)


def init_mt5(mt5: MetaTrader5):
    kwargs = {}
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
    kv("MT5 terminal connected", getattr(ti, "connected", None))
    kv("MT5 trade_allowed", getattr(ti, "trade_allowed", None), "warning" if not getattr(ti, "trade_allowed", False) else "info")
    kv("MT5 tradeapi_disabled", getattr(ti, "tradeapi_disabled", None))
    kv("MT5 broker", getattr(ti, "company", None))
    kv("MT5 terminal", getattr(ti, "name", None))
    logger.debug(f"MT5 terminal_info full={ti}")
    return ti


def ensure_symbol_ready(mt5: MetaTrader5, symbol: str) -> Any:
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        raise RuntimeError(f"Symbol not found in terminal: {symbol}")

    kv("Symbol found", symbol)
    kv("Symbol visible", getattr(symbol_info, "visible", None))
    kv("Symbol select", getattr(symbol_info, "select", None))
    kv("Symbol path", getattr(symbol_info, "path", None))

    if not getattr(symbol_info, "visible", False):
        ok = mt5.symbol_select(symbol, True)
        kv("symbol_select()", ok, "warning" if not ok else "info")
        if not ok:
            raise RuntimeError(f"symbol_select failed for {symbol}; last_error={mt5.last_error()}")
        symbol_info = mt5.symbol_info(symbol)
        kv("Symbol visible after select", getattr(symbol_info, "visible", None))
        if symbol_info is None or not getattr(symbol_info, "visible", False):
            raise RuntimeError(f"Symbol is still not visible after symbol_select for {symbol}")

    return symbol_info


def get_timeframe(mt5: MetaTrader5):
    mapping = {
        "M1":  mt5.TIMEFRAME_M1,
        "M2":  mt5.TIMEFRAME_M2,
        "M3":  mt5.TIMEFRAME_M3,
        "M4":  mt5.TIMEFRAME_M4,
        "M5":  mt5.TIMEFRAME_M5,
        "M6":  mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,
        "H2":  mt5.TIMEFRAME_H2,
        "H3":  mt5.TIMEFRAME_H3,
        "H4":  mt5.TIMEFRAME_H4,
        "H6":  mt5.TIMEFRAME_H6,
        "H8":  mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1":  mt5.TIMEFRAME_D1,
        "W1":  mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    if TIMEFRAME_NAME not in mapping:
        raise ValueError(f"Unsupported timeframe: {TIMEFRAME_NAME}")
    return mapping[TIMEFRAME_NAME]


def get_bars(mt5: MetaTrader5, symbol: str, n: int) -> pd.DataFrame:
    kv("Bar fetch symbol", symbol)
    kv("Bar fetch timeframe", TIMEFRAME_NAME)
    kv("Bar fetch count requested", n)
    rates = mt5.copy_rates_from_pos(symbol, get_timeframe(mt5), 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No MT5 bars returned for {symbol}; last_error={mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"}).set_index("time")
    kv("Bars received", len(df))
    kv("First bar", df.index[0])
    kv("Last bar", df.index[-1])
    kv("Last close", float(df["close"].iloc[-1]))
    return df


def get_daily_bars(mt5: MetaTrader5, symbol: str, n: int) -> pd.DataFrame:
    """
    Fetch D1 bars specifically for the daily EMA100 filter.
    Uses TIMEFRAME_D1 regardless of the strategy's main TIMEFRAME_NAME.
    """
    kv("Daily bar fetch symbol", symbol)
    kv("Daily bar fetch count", n)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, n)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"No D1 bars returned for {symbol}; last_error={mt5.last_error()}"
        )
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"}).set_index("time")
    kv("Daily bars received", len(df))
    kv("Daily last bar", df.index[-1])
    kv("Daily last close", float(df["close"].iloc[-1]))
    return df


def check_daily_ema_filter(mt5: MetaTrader5, symbol: str) -> Tuple[bool, float, float, bool]:
    """
    Check if the daily close is above or below the EMA100 on the daily chart.

    Returns:
        above_ema  : True if daily close > EMA100
        daily_close: last daily close price
        ema100     : current EMA100 value
        short_blocked: True if a short trade should be blocked

    Logic:
        - Daily close ABOVE EMA100 → uptrend → block shorts, allow longs
        - Daily close BELOW EMA100 → downtrend → allow shorts, allow longs
        - If filter disabled → never block (short_blocked=False always)
    """
    if not ENABLE_DAILY_EMA_FILTER:
        kv("Daily EMA filter", "DISABLED")
        return True, 0.0, 0.0, False

    section("DAILY EMA100 FILTER")
    try:
        daily = get_daily_bars(mt5, symbol, DAILY_EMA_BARS)
        ema100 = float(
            daily["close"].ewm(span=DAILY_EMA_PERIOD, adjust=False).mean().iloc[-1]
        )
        daily_close = float(daily["close"].iloc[-1])
        above_ema = daily_close > ema100
        short_blocked = above_ema  # block shorts when above EMA100

        kv("Daily EMA100", round(ema100, 2))
        kv("Daily close", round(daily_close, 2))
        kv("Daily above EMA100", above_ema,
           "info" if above_ema else "warning")
        kv("Short trade blocked by EMA filter", short_blocked,
           "warning" if short_blocked else "info")

        if short_blocked:
            logger.warning(
                f"[EMA FILTER] Daily close {daily_close:.2f} is ABOVE EMA100 {ema100:.2f} "
                f"— SHORT trades are blocked. Market is in an uptrend."
            )
        else:
            logger.info(
                f"[EMA FILTER] Daily close {daily_close:.2f} is BELOW EMA100 {ema100:.2f} "
                f"— SHORT trades are allowed. Market is in a downtrend."
            )

        return above_ema, daily_close, ema100, short_blocked

    except Exception as e:
        logger.warning(
            f"[EMA FILTER] Failed to compute daily EMA100 ({e}); "
            "defaulting to NOT blocking shorts (fail open)"
        )
        return True, 0.0, 0.0, False


def get_positions(mt5: MetaTrader5, symbol: str):
    pos = mt5.positions_get(symbol=symbol)
    if pos is None:
        logger.warning(f"[POS] positions_get returned None; last_error={mt5.last_error()}")
        return []
    pos = list(pos)
    kv("Open positions", len(pos))
    for p in pos:
        logger.info(
            f"[POS] ticket={p.ticket} type={p.type} volume={p.volume} "
            f"price_open={p.price_open} sl={p.sl} tp={p.tp} "
            f"magic={getattr(p, 'magic', None)} comment={getattr(p, 'comment', None)}"
        )
    return pos


def get_symbol_trade_rules(mt5: MetaTrader5, symbol: str) -> Dict[str, Any]:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info failed for {symbol}; last_error={mt5.last_error()}")

    point = float(getattr(info, "point", 0.0) or 0.0)
    digits = int(getattr(info, "digits", 2) or 2)
    tick_size = float(getattr(info, "trade_tick_size", 0.0) or point or 0.01)
    stops_level_points = int(getattr(info, "trade_stops_level", 0) or 0)
    freeze_level_points = int(getattr(info, "trade_freeze_level", 0) or 0)
    filling_mode = int(getattr(info, "filling_mode", 0) or 0)
    min_stop_price = stops_level_points * point

    kv("Symbol digits", digits)
    kv("Symbol point", point)
    kv("Symbol tick_size", tick_size)
    kv("Symbol stops_level points", stops_level_points)
    kv("Symbol freeze_level points", freeze_level_points)
    kv("Symbol min_stop_price", min_stop_price)
    kv("Symbol filling_mode", filling_mode)

    return {
        "point": point,
        "digits": digits,
        "tick_size": tick_size,
        "stops_level_points": stops_level_points,
        "freeze_level_points": freeze_level_points,
        "min_stop_price": min_stop_price,
        "filling_mode": filling_mode,
    }


def normalize_price(price: float, tick_size: float, digits: int) -> float:
    eps = tick_size * 1e-9 if tick_size > 0 else 1e-12
    if tick_size <= 0:
        return round(price, digits)
    return round(round((price + eps) / tick_size) * tick_size, digits)


def floor_to_tick(price: float, tick_size: float, digits: int) -> float:
    eps = tick_size * 1e-9 if tick_size > 0 else 1e-12
    if tick_size <= 0:
        return round(price, digits)
    return round(math.floor((price + eps) / tick_size) * tick_size, digits)


def ceil_to_tick(price: float, tick_size: float, digits: int) -> float:
    eps = tick_size * 1e-9 if tick_size > 0 else 1e-12
    if tick_size <= 0:
        return round(price, digits)
    return round(math.ceil((price - eps) / tick_size) * tick_size, digits)


def candidate_filling_modes(mt5: MetaTrader5, symbol: str) -> List[int]:
    candidates = []
    info = mt5.symbol_info(symbol)
    symbol_mode = getattr(info, "filling_mode", None) if info is not None else None

    symbol_to_order = {
        getattr(mt5, "SYMBOL_FILLING_FOK", object()): getattr(mt5, "ORDER_FILLING_FOK", None),
        getattr(mt5, "SYMBOL_FILLING_IOC", object()): getattr(mt5, "ORDER_FILLING_IOC", None),
        getattr(mt5, "SYMBOL_FILLING_RETURN", object()): getattr(mt5, "ORDER_FILLING_RETURN", None),
    }

    if symbol_mode in symbol_to_order and symbol_to_order[symbol_mode] is not None:
        candidates.append(symbol_to_order[symbol_mode])

    for name in ("ORDER_FILLING_RETURN", "ORDER_FILLING_IOC", "ORDER_FILLING_FOK"):
        value = getattr(mt5, name, None)
        if value is not None and value not in candidates:
            candidates.append(value)

    return candidates


def retcode_done_set(mt5: MetaTrader5) -> set:
    return {
        x for x in {
            0,
            getattr(mt5, "TRADE_RETCODE_DONE", None),
            getattr(mt5, "TRADE_RETCODE_PLACED", None),
            getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", None),
            getattr(mt5, "TRADE_RETCODE_NO_CHANGES", None),
        } if x is not None
    }


def retcode_name_map(mt5: MetaTrader5) -> Dict[int, str]:
    out = {}
    for name in dir(mt5):
        if name.startswith("TRADE_RETCODE_"):
            val = getattr(mt5, name, None)
            if isinstance(val, int):
                out[val] = name
    return out


def describe_retcode(mt5: MetaTrader5, retcode: Optional[int]) -> str:
    if retcode is None:
        return "None"
    return retcode_name_map(mt5).get(retcode, str(retcode))


def choose_filling_mode_by_check(mt5: MetaTrader5, request: Dict[str, Any]) -> Tuple[Optional[int], Any, Optional[Tuple[int, str]]]:
    last_check = None
    last_error = None

    for filling_mode in candidate_filling_modes(mt5, request["symbol"]):
        req = dict(request)
        req["type_filling"] = filling_mode

        check = mt5.order_check(req)
        err = mt5.last_error()
        last_check = check
        last_error = err

        kv("Try filling mode", filling_mode)

        if check is None:
            kv("order_check retcode", "None", "warning")
            kv("order_check comment", "", "warning")
            kv("order_check last_error", err, "warning")
            continue

        retcode = getattr(check, "retcode", None)
        comment = getattr(check, "comment", "")
        kv("order_check retcode", f"{retcode} ({describe_retcode(mt5, retcode)})")
        kv("order_check comment", comment)

        if retcode in retcode_done_set(mt5):
            logger.info(f"[ORDER] accepted filling mode via order_check: {filling_mode}")
            return filling_mode, check, err

    return None, last_check, last_error


def calc_atr_points(bars: pd.DataFrame) -> float:
    d = bars.copy()
    prev_close = d["close"].shift(1)
    tr = pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.rolling(ATR_PERIOD).mean().iloc[-1])
    kv("ATR", round(atr, 5))
    kv("ATR market points", round(atr, 2))
    return atr


def choose_stop_points(bars: pd.DataFrame) -> float:
    if STOP_MODE == "fixed":
        kv("Stop mode", f"fixed ({STOP_POINTS} market points)")
        return STOP_POINTS
    atr_points = calc_atr_points(bars)
    sl_points = atr_points * DYNAMIC_STOP_ATR_MULT
    sl_points = max(MIN_DYNAMIC_STOP_POINTS, min(MAX_DYNAMIC_STOP_POINTS, sl_points))
    kv("Stop mode", "dynamic")
    kv("Dynamic stop market points", round(sl_points, 2))
    return float(sl_points)


def choose_take_profit_points(bars: pd.DataFrame) -> float:
    if not ENABLE_TAKE_PROFIT:
        return 0.0
    if TP_MODE == "fixed":
        kv("TP mode", f"fixed ({TAKE_PROFIT_POINTS} market points)")
        return TAKE_PROFIT_POINTS
    atr_points = calc_atr_points(bars)
    tp_points = atr_points * TAKE_PROFIT_ATR_MULT
    tp_points = max(MIN_TAKE_PROFIT_POINTS, min(MAX_TAKE_PROFIT_POINTS, tp_points))
    kv("TP mode", "atr")
    kv("Dynamic TP market points", round(tp_points, 2))
    return float(tp_points)


def decide_lot(strong_consensus: bool) -> float:
    lot = BASE_LOT
    kv("Sizing strong_consensus", strong_consensus)
    if ALLOW_DOUBLE_SIZE and strong_consensus:
        lot = DOUBLE_LOT
    kv("Chosen lot", lot)
    return lot


def side_from_positions(mt5: MetaTrader5, positions) -> str:
    if not positions:
        return "flat"
    buy_count = sum(1 for p in positions if p.type == mt5.POSITION_TYPE_BUY)
    sell_count = sum(1 for p in positions if p.type == mt5.POSITION_TYPE_SELL)
    if buy_count > 0 and sell_count == 0:
        return "long"
    if sell_count > 0 and buy_count == 0:
        return "short"
    return "mixed"


def is_our_position(pos) -> bool:
    pos_magic = getattr(pos, "magic", None)
    pos_comment = str(getattr(pos, "comment", "") or "")
    magic_match = pos_magic == MAGIC
    comment_match = STRATEGY_COMMENT_TAG.lower() in pos_comment.lower() if STRATEGY_COMMENT_TAG else True
    return magic_match and comment_match


def filter_positions_for_strategy(positions: List[Any]) -> List[Any]:
    ours = [p for p in positions if is_our_position(p)]
    logger.info(
        f"[POS] strict ownership filter: ours={len(ours)} "
        f"other_symbol_positions={len(positions) - len(ours)} total_symbol_positions={len(positions)}"
    )
    return ours


def is_our_deal(deal) -> bool:
    deal_magic = getattr(deal, "magic", None)
    deal_comment = str(getattr(deal, "comment", "") or "")
    magic_match = deal_magic == MAGIC
    comment_match = STRATEGY_COMMENT_TAG.lower() in deal_comment.lower() if STRATEGY_COMMENT_TAG else True
    return magic_match and comment_match


def can_trade_now(now_utc: datetime) -> Tuple[bool, str]:
    dstr = now_utc.date().isoformat()
    if dstr in BLOCK_DATES:
        return False, f"blocked date {dstr}"
    if ENABLE_SESSION_FILTER:
        hour = now_utc.hour
        allowed = any(start_h <= hour < end_h for start_h, end_h in PARSED_UTC_WINDOWS)
        if not allowed:
            return False, f"hour {hour} UTC outside allowed windows {ALLOWED_UTC_WINDOWS}"
    return True, "ok"


def build_entry_request(mt5: MetaTrader5, symbol: str, side: str, lot: float, stop_points: float, tp_points: float) -> Dict[str, Any]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Could not get symbol tick for {symbol}")

    rules = get_symbol_trade_rules(mt5, symbol)
    digits = int(rules["digits"])
    tick_size = float(rules["tick_size"])
    min_stop_price = float(rules["min_stop_price"])

    if side == "long":
        order_type = mt5.ORDER_TYPE_BUY
        price = normalize_price(float(tick.ask), tick_size, digits)
        raw_sl = price - float(stop_points)
        sl = floor_to_tick(raw_sl, tick_size, digits)
        if (price - sl) < max(min_stop_price, tick_size):
            sl = floor_to_tick(price - max(min_stop_price + tick_size, tick_size), tick_size, digits)
        if tp_points > 0:
            raw_tp = price + float(tp_points)
            tp = ceil_to_tick(raw_tp, tick_size, digits)
            if (tp - price) < max(min_stop_price, tick_size):
                tp = ceil_to_tick(price + max(min_stop_price + tick_size, tick_size), tick_size, digits)
        else:
            raw_tp = None
            tp = 0.0
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = normalize_price(float(tick.bid), tick_size, digits)
        raw_sl = price + float(stop_points)
        sl = ceil_to_tick(raw_sl, tick_size, digits)
        if (sl - price) < max(min_stop_price, tick_size):
            sl = ceil_to_tick(price + max(min_stop_price + tick_size, tick_size), tick_size, digits)
        if tp_points > 0:
            raw_tp = price - float(tp_points)
            tp = floor_to_tick(raw_tp, tick_size, digits)
            if (price - tp) < max(min_stop_price, tick_size):
                tp = floor_to_tick(price - max(min_stop_price + tick_size, tick_size), tick_size, digits)
        else:
            raw_tp = None
            tp = 0.0

    kv("Entry price", price)
    kv("Requested stop points", stop_points)
    kv("Raw SL", raw_sl)
    kv("Final SL", sl)
    kv("Actual stop distance", round(abs(price - sl), 5))
    kv("Requested TP points", tp_points)
    kv("Raw TP", raw_tp if raw_tp is not None else "disabled")
    kv("Final TP", tp)
    kv("Actual TP distance", round(abs(tp - price), 5) if tp else 0.0)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot),
        "type": order_type,
        "price": float(price),
        "sl": float(sl),
        "tp": float(tp),
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": f"{STRATEGY_COMMENT_TAG} mt5win".strip(),
        "type_time": mt5.ORDER_TIME_GTC,
    }
    dump_json("Order request draft", request, "info")
    return request


def build_close_request(mt5: MetaTrader5, symbol: str, pos) -> Dict[str, Any]:
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"symbol_info_tick failed for {symbol}")

    rules = get_symbol_trade_rules(mt5, symbol)
    tick_size = float(rules["tick_size"])
    digits = int(rules["digits"])

    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    close_price = normalize_price(float(tick.bid if is_buy else tick.ask), tick_size, digits)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos.ticket,
        "symbol": symbol,
        "volume": float(pos.volume),
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "price": float(close_price),
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": f"{STRATEGY_COMMENT_TAG} close mt5win".strip(),
        "type_time": mt5.ORDER_TIME_GTC,
    }
    dump_json(f"Close request draft ticket={pos.ticket}", request, "info")
    return request


def is_successful_trade_result(mt5: MetaTrader5, result: Any, payload: Dict[str, Any]) -> bool:
    if result is None:
        return False
    retcode = getattr(result, "retcode", None)
    if retcode is None:
        retcode = payload.get("retcode")
    if retcode in retcode_done_set(mt5):
        return True
    comment = str(payload.get("comment", "") or "").lower()
    if "done" in comment or "placed" in comment:
        return True
    return False


def send_trade_request(mt5: MetaTrader5, request: Dict[str, Any], label: str) -> Dict[str, Any]:
    logger.info(f"[ORDER] {label} requested")
    req = dict(request)

    chosen_fill, check, check_error = choose_filling_mode_by_check(mt5, req)

    if chosen_fill is not None:
        req["type_filling"] = chosen_fill
    else:
        fills = candidate_filling_modes(mt5, req["symbol"])
        if fills:
            req["type_filling"] = fills[0]
            logger.warning(f"[ORDER] falling back to first candidate filling mode without check: {fills[0]}")

    if check is None:
        logger.warning(f"[ORDER] {label} order_check returned None; last_error={check_error}")
        if not EXECUTION_ENABLED:
            return {
                "execution_enabled": False, "request": req, "check": None,
                "check_ok": False, "check_retcode": None, "check_retcode_name": "NONE",
                "check_last_error": check_error, "success": False, "simulated": True,
                "sent": False, "check_only": True, "validation_unavailable": True,
            }
        return {
            "execution_enabled": True, "request": req, "check": None,
            "check_ok": False, "check_retcode": None, "check_retcode_name": "NONE",
            "check_last_error": check_error, "success": False, "simulated": False,
            "sent": False, "validation_unavailable": True,
        }

    check_payload = check._asdict() if hasattr(check, "_asdict") else {"check": str(check)}
    dump_json(f"{label} order_check payload", check_payload, "info")
    check_retcode = getattr(check, "retcode", None)
    check_ok = check_retcode in retcode_done_set(mt5)

    if not check_ok:
        logger.warning(f"[ORDER] {label} rejected by order_check retcode={check_retcode} ({describe_retcode(mt5, check_retcode)})")
        return {
            "execution_enabled": EXECUTION_ENABLED, "request": req, "check": check_payload,
            "check_ok": False, "check_retcode": check_retcode,
            "check_retcode_name": describe_retcode(mt5, check_retcode),
            "check_last_error": check_error, "success": False,
            "simulated": not EXECUTION_ENABLED, "sent": False,
        }

    if not EXECUTION_ENABLED:
        logger.warning(f"[ORDER] EXECUTION_ENABLED=0; {label} passed order_check but was not sent")
        return {
            "execution_enabled": False, "request": req, "check": check_payload,
            "check_ok": True, "check_retcode": check_retcode,
            "check_retcode_name": describe_retcode(mt5, check_retcode),
            "check_last_error": check_error, "success": True,
            "simulated": True, "sent": False, "check_only": True,
        }

    result = mt5.order_send(req)
    send_error = mt5.last_error()
    payload = result._asdict() if hasattr(result, "_asdict") else {"result": str(result)}
    dump_json(f"{label} order_send payload", payload, "info")
    kv(f"{label} order_send last_error", send_error)

    return {
        "execution_enabled": True, "check": check_payload, "check_ok": True,
        "check_retcode": check_retcode,
        "check_retcode_name": describe_retcode(mt5, check_retcode),
        "check_last_error": check_error, "result": payload,
        "send_last_error": send_error, "request": req,
        "success": is_successful_trade_result(mt5, result, payload),
        "simulated": False, "sent": True,
    }


def send_sltp_request(mt5: MetaTrader5, request: Dict[str, Any], label: str) -> Dict[str, Any]:
    logger.info(f"[SLTP] {label} requested")
    dump_json(f"{label} request", request, "info")

    if not EXECUTION_ENABLED:
        logger.warning(f"[SLTP] EXECUTION_ENABLED=0; {label} not sent")
        return {
            "execution_enabled": False, "request": request, "success": False,
            "simulated": True, "sent": False, "check_only": True,
        }

    result = mt5.order_send(request)
    send_error = mt5.last_error()
    payload = result._asdict() if hasattr(result, "_asdict") else {"result": str(result)}
    dump_json(f"{label} payload", payload, "info")
    kv(f"{label} last_error", send_error)

    return {
        "execution_enabled": True, "request": request, "result": payload,
        "send_last_error": send_error,
        "success": is_successful_trade_result(mt5, result, payload),
        "simulated": False, "sent": True,
    }


def close_position(mt5: MetaTrader5, symbol: str, pos) -> Dict[str, Any]:
    request = build_close_request(mt5, symbol, pos)
    return send_trade_request(mt5, request, f"close ticket {pos.ticket}")


def enforce_sl_price_cap(mt5: MetaTrader5, symbol: str) -> None:
    if MAX_SL_PRICE_DISTANCE <= 0:
        logger.info("[SLCAP] MAX_SL_PRICE_DISTANCE=0; SL cap disabled")
        return

    section("SL PRICE CAP")
    kv("SL cap max distance", MAX_SL_PRICE_DISTANCE)
    kv("SL cap retries", SL_CAP_RETRIES)
    kv("SL cap retry delay (s)", SL_CAP_RETRY_DELAY)

    managed_positions: List[Any] = []
    for attempt in range(1, SL_CAP_RETRIES + 1):
        raw = get_positions(mt5, symbol)
        managed_positions = filter_positions_for_strategy(raw)
        if managed_positions:
            logger.info(f"[SLCAP] found {len(managed_positions)} managed position(s) on attempt {attempt}")
            break
        logger.warning(
            f"[SLCAP] no managed positions visible yet (attempt {attempt}/{SL_CAP_RETRIES}); "
            f"retrying in {SL_CAP_RETRY_DELAY}s"
        )
        time.sleep(SL_CAP_RETRY_DELAY)

    if not managed_positions:
        logger.error(
            f"[SLCAP] position still not visible after {SL_CAP_RETRIES} attempts; "
            "SL cap could not be applied — review position manually"
        )
        return

    rules = get_symbol_trade_rules(mt5, symbol)
    tick_size = float(rules["tick_size"])
    digits = int(rules["digits"])
    min_stop_price = float(rules["min_stop_price"])

    for pos in managed_positions:
        ticket = pos.ticket
        entry = float(pos.price_open)
        current_sl = float(pos.sl)
        side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"

        if current_sl == 0:
            logger.warning(f"[SLCAP] ticket={ticket} has no SL set; skipping cap check")
            continue

        current_distance = abs(entry - current_sl)
        kv(f"[SLCAP] ticket={ticket} side", side)
        kv(f"[SLCAP] ticket={ticket} entry", entry)
        kv(f"[SLCAP] ticket={ticket} current SL", current_sl)
        kv(f"[SLCAP] ticket={ticket} SL distance", round(current_distance, 5))

        if current_distance <= MAX_SL_PRICE_DISTANCE:
            logger.info(
                f"[SLCAP] ticket={ticket} SL distance {round(current_distance, 5)} "
                f"is within cap {MAX_SL_PRICE_DISTANCE}; no adjustment needed"
            )
            continue

        if side == "long":
            new_sl = floor_to_tick(entry - MAX_SL_PRICE_DISTANCE, tick_size, digits)
            distance_ok = (entry - new_sl) >= max(min_stop_price, tick_size)
        else:
            new_sl = ceil_to_tick(entry + MAX_SL_PRICE_DISTANCE, tick_size, digits)
            distance_ok = (new_sl - entry) >= max(min_stop_price, tick_size)

        if not distance_ok:
            logger.warning(
                f"[SLCAP] ticket={ticket} capped SL {new_sl} violates broker min_stop_price "
                f"({min_stop_price}); skipping adjustment to avoid rejection"
            )
            continue

        logger.warning(
            f"[SLCAP] ticket={ticket} SL distance {round(current_distance, 5)} exceeds cap "
            f"{MAX_SL_PRICE_DISTANCE}; adjusting SL {current_sl} -> {new_sl}"
        )

        req = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": ticket,
            "sl": float(new_sl),
            "tp": float(pos.tp),
        }
        res = send_sltp_request(mt5, req, f"SL cap ticket={ticket}")

        if res.get("success"):
            logger.info(f"[SLCAP] ticket={ticket} SL successfully moved to {new_sl}")
        else:
            logger.error(
                f"[SLCAP] ticket={ticket} SL cap adjustment FAILED; "
                "position is live with oversized stop — review manually"
            )


def move_stop_if_needed(mt5: MetaTrader5, symbol: str, positions, bars: pd.DataFrame) -> None:
    if not ENABLE_BREAKEVEN or not positions:
        logger.info("[MANAGE] breakeven/trailing skipped")
        return

    rules = get_symbol_trade_rules(mt5, symbol)
    tick_size = float(rules["tick_size"])
    digits = int(rules["digits"])
    min_stop_price = float(rules["min_stop_price"])
    atr_points = calc_atr_points(bars)

    for pos in positions:
        side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
        tick = mt5.symbol_info_tick(symbol)
        current = normalize_price(float(tick.bid if side == "long" else tick.ask), tick_size, digits)
        risk_points = abs(float(pos.price_open) - float(pos.sl)) if pos.sl else 0.0
        if risk_points <= 0:
            logger.warning(f"[MANAGE] ticket={pos.ticket} no valid risk_points; skipping")
            continue

        move_points = (current - float(pos.price_open)) if side == "long" else (float(pos.price_open) - current)
        logger.info(
            f"[MANAGE] ticket={pos.ticket} side={side} entry={pos.price_open} current={current} "
            f"sl={pos.sl} tp={pos.tp} move_points={move_points:.2f} risk_points={risk_points:.2f}"
        )

        if move_points >= risk_points * BREAKEVEN_R_MULT:
            new_sl = normalize_price(float(pos.price_open), tick_size, digits)
            if side == "long" and (current - new_sl) >= max(min_stop_price, tick_size):
                improve = (pos.sl == 0 or new_sl > float(pos.sl))
            elif side == "short" and (new_sl - current) >= max(min_stop_price, tick_size):
                improve = (pos.sl == 0 or new_sl < float(pos.sl))
            else:
                improve = False

            if improve:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": pos.ticket,
                    "sl": float(new_sl),
                    "tp": float(pos.tp),
                }
                send_sltp_request(mt5, req, f"breakeven ticket={pos.ticket}")

        if ENABLE_TRAILING_AFTER_BE:
            trail_points = atr_points * TRAIL_ATR_MULT
            if side == "long":
                trail_sl = floor_to_tick(current - trail_points, tick_size, digits)
                if trail_sl > max(float(pos.sl), float(pos.price_open)) and (current - trail_sl) >= max(min_stop_price, tick_size):
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": pos.ticket,
                        "sl": float(trail_sl),
                        "tp": float(pos.tp),
                    }
                    send_sltp_request(mt5, req, f"trail ticket={pos.ticket}")
            else:
                trail_sl = ceil_to_tick(current + trail_points, tick_size, digits)
                if (pos.sl == 0 or trail_sl < min(float(pos.sl), float(pos.price_open))) and (trail_sl - current) >= max(min_stop_price, tick_size):
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": symbol,
                        "position": pos.ticket,
                        "sl": float(trail_sl),
                        "tp": float(pos.tp),
                    }
                    send_sltp_request(mt5, req, f"trail ticket={pos.ticket}")


def append_decision(d: Decision) -> None:
    with DECISION_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")


def check_one_loss_rule(state: LiveState, now_utc: datetime) -> Tuple[bool, str]:
    if not ONE_LOSS_PER_DAY:
        return True, "disabled"
    today = now_utc.date().isoformat()
    if state.loss_block_day == today:
        return False, f"one loss already recorded today ({today})"
    return True, "ok"


def detect_closed_loss_and_update_state(mt5: MetaTrader5, state: LiveState, now_utc: datetime) -> None:
    if not EXECUTION_ENABLED:
        logger.info("[RISK] EXECUTION_ENABLED=0; closed-loss scan skipped")
        return

    today = now_utc.date()
    start_dt = datetime.combine(today, dtime.min).replace(tzinfo=timezone.utc)
    deals = mt5.history_deals_get(start_dt, now_utc)
    if deals is None:
        logger.warning(f"[RISK] history_deals_get returned None last_error={mt5.last_error()}")
        return

    newest_seen_ticket = state.last_closed_ticket
    loss_found = False

    for d in deals:
        if not is_our_deal(d):
            continue
        if getattr(d, "entry", None) != mt5.DEAL_ENTRY_OUT:
            continue

        ticket = int(getattr(d, "ticket", 0) or 0)
        pnl = float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0))

        if ticket <= state.last_closed_ticket:
            continue

        logger.info(
            f"[RISK] closed managed deal ticket={ticket} pnl={pnl} "
            f"magic={getattr(d, 'magic', None)} comment={getattr(d, 'comment', None)}"
        )

        if ticket > newest_seen_ticket:
            newest_seen_ticket = ticket

        if pnl < 0:
            loss_found = True

    if newest_seen_ticket > state.last_closed_ticket:
        state.last_closed_ticket = newest_seen_ticket

    if loss_found:
        state.loss_block_day = now_utc.date().isoformat()
        logger.warning(f"[RISK] loss detected; blocking entries for rest of day {state.loss_block_day}")


def main() -> None:
    acquire_lock()
    now_utc = datetime.now(timezone.utc)
    symbol = resolve_symbol(now_utc)
    symbol_root = infer_root_from_symbol(symbol)

    banner(f"START amp_mt5windows | now_utc={now_utc.isoformat()} | symbol={symbol} | execution_enabled={EXECUTION_ENABLED}")

    section("CONFIG")
    kv("DEBUG_MODE", DEBUG_MODE)
    kv("NO_COLOR", NO_COLOR)
    kv("Manual MT5_SYMBOL", MT5_SYMBOL or "<auto>")
    kv("MT5_SYMBOL_ROOT", MT5_SYMBOL_ROOT)
    kv("Resolved symbol root", symbol_root)
    kv("AUTO_CONTRACT_ROLLOVER", AUTO_CONTRACT_ROLLOVER)
    kv("Resolved symbol", symbol)
    kv("MT5 path", MT5_PATH or "<default>")
    kv("Timeframe", TIMEFRAME_NAME)
    kv("Consensus min", CONSENSUS_MIN)
    kv("Strong consensus min", STRONG_CONSENSUS_MIN)
    kv("Base lot", BASE_LOT)
    kv("Double lot", DOUBLE_LOT)
    kv("Execution enabled", EXECUTION_ENABLED)
    kv("Stop mode", STOP_MODE)
    kv("Fixed stop points", STOP_POINTS)
    kv("Dynamic stop ATR mult", DYNAMIC_STOP_ATR_MULT)
    kv("Max SL price distance", MAX_SL_PRICE_DISTANCE)
    kv("SL cap retries", SL_CAP_RETRIES)
    kv("SL cap retry delay (s)", SL_CAP_RETRY_DELAY)
    kv("Enable take profit", ENABLE_TAKE_PROFIT)
    kv("TP mode", TP_MODE)
    kv("Take profit points", TAKE_PROFIT_POINTS)
    kv("Take profit ATR mult", TAKE_PROFIT_ATR_MULT)
    kv("Block on any symbol position", "True (always — raw_positions checked)")
    kv("Strategy comment tag", STRATEGY_COMMENT_TAG)
    kv("Session filter", ENABLE_SESSION_FILTER)
    kv("Allowed UTC windows", ALLOWED_UTC_WINDOWS)
    kv("Parsed UTC windows", PARSED_UTC_WINDOWS)
    kv("Daily EMA filter enabled", ENABLE_DAILY_EMA_FILTER)
    kv("Daily EMA period", DAILY_EMA_PERIOD)
    kv("Lock file", LOCK_FILE)
    kv("Log file", RUN_LOG)

    section("STATE")
    state = LiveState.load(STATE_FILE)
    dump_json("Loaded state", asdict(state), "info")

    section("FILTERS")
    allowed_time, time_reason = can_trade_now(now_utc)
    kv("Time/date allowed", allowed_time, "warning" if not allowed_time else "info")
    kv("Time/date reason", time_reason, "warning" if not allowed_time else "info")

    section("CONTRACT")
    if not MT5_SYMBOL:
        year, month = current_or_next_active_quarter(now_utc)
        kv("Active quarter year", year)
        kv("Active quarter month", month)
        kv("CME roll date", equity_index_roll_date(year, month))
    kv("Trading symbol", symbol)

    section("MT5 INIT")
    mt5 = make_mt5()
    terminal_info = None
    terminal_trade_allowed = None
    hard_trade_gate_ok = False

    # EMA filter state — initialise before try block so available in Decision
    daily_above_ema  = True
    daily_close_val  = 0.0
    ema100_val       = 0.0
    short_blocked_by_ema = False

    try:
        terminal_info = init_mt5(mt5)
        terminal_trade_allowed = getattr(terminal_info, "trade_allowed", None)
        hard_trade_gate_ok = terminal_trade_allowed is not False
        kv("Terminal hard gate", hard_trade_gate_ok, "warning" if not hard_trade_gate_ok else "info")

        section("SYMBOL READY")
        ensure_symbol_ready(mt5, symbol)

        section("MARKET DATA")
        bars = get_bars(mt5, symbol, 500)
        latest_bar = bars.index[-1].isoformat()
        kv("Latest closed bar", latest_bar)

        duplicate_bar_blocked = state.last_bar_time == latest_bar and state.last_bar_time != ""
        kv("Previous bar in state", state.last_bar_time or "<empty>")
        kv("Duplicate bar blocked", duplicate_bar_blocked, "warning" if duplicate_bar_blocked else "info")

        # ── DAILY EMA100 FILTER ────────────────────────────────────────────────
        # Use continuous @MNQ contract for daily bars — front-month contracts
        # like MNQM26 only have a few months of data which is not enough for
        # a meaningful 100-period EMA. @MNQ has full history back to 2019.
        daily_symbol = "@MNQ"
        daily_above_ema, daily_close_val, ema100_val, short_blocked_by_ema = \
            check_daily_ema_filter(mt5, daily_symbol)

        section("RISK RULES")
        detect_closed_loss_and_update_state(mt5, state, now_utc)
        can_trade_loss, loss_reason = check_one_loss_rule(state, now_utc)
        kv("One-loss rule allowed", can_trade_loss, "warning" if not can_trade_loss else "info")
        kv("One-loss reason", loss_reason, "warning" if not can_trade_loss else "info")

        section("AMP")
        html = fetch_amp_html()
        rows = parse_current_session(html)
        if not rows:
            logger.warning("[AMP] No numeric P/L rows found; AMP side=flat")
            amp_side, consensus, strong_consensus = "flat", {"long_count": 0, "short_count": 0}, False
            amp_signal_id = "amp_side=flat||no_rows"
        else:
            amp_side, consensus, strong_consensus = decide_amp_side(rows, symbol_root)
            amp_signal_id = build_amp_signal_id(rows, amp_side)
        kv("AMP side", color_text(amp_side, side_color(amp_side)))
        kv("AMP long count", consensus["long_count"])
        kv("AMP short count", consensus["short_count"])
        kv("AMP strong consensus", strong_consensus)
        kv("AMP signal id", amp_signal_id[:240] + ("..." if len(amp_signal_id) > 240 else ""))

        if state.consumed_amp_signal_id and amp_signal_id != state.consumed_amp_signal_id:
            logger.info("[AMP] New scraped AMP signal detected; clearing consumed signal lock")
            state.consumed_amp_signal_id = ""

        signal_already_consumed = (
            bool(state.consumed_amp_signal_id) and
            amp_signal_id == state.consumed_amp_signal_id
        )
        kv("Signal already consumed", signal_already_consumed, "warning" if signal_already_consumed else "info")

        section("DECISION")
        if amp_side == "flat":
            final_side = "flat"
            final_lot = 0.0
            note = "AMP consensus not strong enough or no systems available"
        else:
            final_side = amp_side
            final_lot = decide_lot(strong_consensus)
            note = "Signal approved from AMP consensus"

        # ── APPLY DAILY EMA100 SHORT BLOCK ────────────────────────────────────
        if final_side == "short" and short_blocked_by_ema:
            logger.warning(
                f"[EMA FILTER] AMP wants SHORT but daily close {daily_close_val:.2f} "
                f"is ABOVE EMA100 {ema100_val:.2f} — overriding to FLAT. "
                "Will not short into an uptrend."
            )
            final_side = "flat"
            final_lot  = 0.0
            note = (
                f"SHORT blocked by daily EMA100 filter: "
                f"close={daily_close_val:.2f} > ema100={ema100_val:.2f}"
            )

        kv("Final side", color_text(final_side, side_color(final_side)))
        kv("Final lot", final_lot)
        kv("Decision note", note)

        if not allowed_time:
            logger.warning("[FILTER] outside allowed trading time/date -> no new entry allowed")
        if not can_trade_loss:
            logger.warning("[FILTER] one-loss-per-day active -> no new entry allowed")
        if not hard_trade_gate_ok:
            logger.warning("[FILTER] terminal trade_allowed=False -> hard blocking new entries/flips")
        if duplicate_bar_blocked:
            logger.warning("[FILTER] duplicate closed bar detected")
            logger.warning("[FILTER] duplicate bar will NOT block a new entry if strategy-controlled symbol exposure is flat")

        section("POSITIONS")
        raw_positions = get_positions(mt5, symbol)
        managed_positions = filter_positions_for_strategy(raw_positions)
        existing_side = side_from_positions(mt5, managed_positions)
        kv("Existing side", color_text(existing_side, side_color(existing_side)))
        kv("Raw symbol positions", len(raw_positions))
        kv("Managed strategy positions", len(managed_positions))

        unrelated_positions = [p for p in raw_positions if not is_our_position(p)]
        kv("Unrelated (external) positions", len(unrelated_positions))
        if unrelated_positions:
            logger.warning(
                f"[POS] {len(unrelated_positions)} unrelated position(s) detected on {symbol}; "
                "this script will NOT open new orders or close these positions under any circumstances"
            )

        section("POSITION MANAGEMENT")
        move_stop_if_needed(mt5, symbol, managed_positions, bars)

        section("EXECUTION")
        action = "hold"
        mark_bar_consumed = False
        entry_allowed = allowed_time and can_trade_loss and hard_trade_gate_ok

        if ENFORCE_ONE_POSITION and len(managed_positions) > 1:
            logger.error("[SAFETY] more than 1 managed position detected; forcing no new orders")
            final_side = "flat"
            note = "safety stop: more than one managed position detected"
            entry_allowed = False

        kv("Entry allowed", entry_allowed, "warning" if not entry_allowed else "info")

        if existing_side == "flat":
            if len(raw_positions) > 0:
                logger.warning(
                    f"[ENTRY] blocked because there is already an open position on {symbol}; "
                    f"raw_symbol_positions={len(raw_positions)} managed_positions={len(managed_positions)} "
                    f"unrelated_positions={len(unrelated_positions)}"
                )
                action = "blocked_existing_position"
                note = "Existing symbol position already open; no new order sent"

            elif final_side in ("long", "short") and not entry_allowed:
                logger.info("[ENTRY] no open managed position, but entry gate is blocked")
                action = "idle"
                note = "No entry because entry gate is blocked"

            elif final_side in ("long", "short") and signal_already_consumed:
                logger.warning(
                    "[ENTRY] blocked because current AMP signal was already consumed earlier; "
                    "waiting for a new scraped AMP position state before re-entering"
                )
                action = "wait_new_amp_signal"
                note = "Manual close safe-lock active; waiting for new AMP scraped position state"

            elif final_side in ("long", "short"):
                stop_points = choose_stop_points(bars)
                tp_points = choose_take_profit_points(bars)
                req = build_entry_request(mt5, symbol, final_side, final_lot, stop_points, tp_points)
                res = send_trade_request(mt5, req, "entry")
                dump_json("Entry response", res, "info")

                if res.get("success", False):
                    logger.info("[ENTRY] order send confirmed successful")
                    action = "enter"
                    note = "Entry order accepted"
                    mark_bar_consumed = True
                    state.consumed_amp_signal_id = amp_signal_id
                    enforce_sl_price_cap(mt5, symbol)
                else:
                    logger.warning("[ENTRY] request failed validation or send")
                    action = "entry_failed"
                    note = "Entry failed validation or send"
            else:
                logger.info("[ENTRY] no open managed position and no entry sent")
                action = "idle"
                if final_side not in ("long", "short"):
                    note = "No entry because final signal is flat"
                elif not entry_allowed:
                    note = "No entry because entry gate is blocked"
                else:
                    note = "No entry sent"
        else:
            if existing_side == "mixed":
                logger.error("[SAFETY] mixed managed exposure detected; no new orders")
                action = "hold"
                note = "safety stop: mixed managed exposure detected"
                entry_allowed = False

            elif final_side == "flat":
                logger.info(
                    f"[HOLD] managed position exists and AMP consensus is flat on {symbol} - "
                    "no close, no new order"
                )
                action = "hold"
                note = "Flat consensus detected; existing managed position left open and no new order sent"

            elif existing_side != final_side and ALLOW_DIRECT_FLIP:
                if not entry_allowed:
                    logger.warning("[FLIP] blocked by entry hard gate")
                    action = "flip_blocked"
                    note = "Flip blocked by time/risk/terminal gate"
                elif duplicate_bar_blocked:
                    logger.warning("[FLIP] duplicate bar detected; flip skipped to avoid repeated same-bar reversal")
                    action = "hold"
                    note = "Flip skipped due to duplicate bar"
                else:
                    logger.info(f"[FLIP] existing_side={existing_side} final_side={final_side} -> close then reopen")
                    close_ok = True
                    for p in managed_positions:
                        res = close_position(mt5, symbol, p)
                        dump_json(f"Flip close response ticket={p.ticket}", res, "info")
                        if not res.get("success", False):
                            close_ok = False

                    if close_ok:
                        refreshed_raw_positions = get_positions(mt5, symbol)
                        refreshed_managed_positions = filter_positions_for_strategy(refreshed_raw_positions)
                        refreshed_unrelated_positions = [p for p in refreshed_raw_positions if not is_our_position(p)]

                        logger.info(
                            f"[FLIP] post-close verification on {symbol}: "
                            f"raw_symbol_positions={len(refreshed_raw_positions)} "
                            f"managed_positions={len(refreshed_managed_positions)} "
                            f"unrelated_positions={len(refreshed_unrelated_positions)}"
                        )

                        if len(refreshed_raw_positions) != 0:
                            logger.warning(
                                f"[FLIP] reopen blocked because symbol is not flat after close on {symbol}; "
                                f"raw_symbol_positions={len(refreshed_raw_positions)} "
                                "strict mode requires exactly zero open positions before reopening"
                            )
                            action = "flip_blocked_existing_position"
                            note = "Flip close succeeded but reopen blocked because symbol was not fully flat after close"
                        else:
                            stop_points = choose_stop_points(bars)
                            tp_points = choose_take_profit_points(bars)
                            req = build_entry_request(mt5, symbol, final_side, final_lot, stop_points, tp_points)
                            res = send_trade_request(mt5, req, "flip entry")
                            dump_json("Flip entry response", res, "info")
                            if res.get("success", False):
                                action = "flip"
                                note = "Flip close and reopen succeeded"
                                mark_bar_consumed = True
                                state.consumed_amp_signal_id = amp_signal_id
                                enforce_sl_price_cap(mt5, symbol)
                            else:
                                action = "flip_entry_failed"
                                note = "Flip close succeeded but reopen failed"
                    else:
                        action = "flip_close_failed"
                        note = "Flip close failed"

            else:
                logger.info("[HOLD] managed position remains aligned")
                action = "hold"
                note = "Existing position remains aligned"

        section("SAVE STATE")
        if mark_bar_consumed:
            state.last_bar_time = latest_bar
        state.last_signal_side = final_side
        state.save(STATE_FILE)
        dump_json("Saved state", asdict(state), "info")

        d = Decision(
            timestamp_utc=now_utc.isoformat(),
            symbol=symbol,
            amp_side=amp_side,
            amp_consensus_long=consensus["long_count"],
            amp_consensus_short=consensus["short_count"],
            amp_consensus_strength=max(consensus["long_count"], consensus["short_count"]),
            final_side=final_side,
            final_lot=final_lot,
            strong_consensus=strong_consensus,
            action=action,
            note=note,
            execution_enabled=EXECUTION_ENABLED,
            duplicate_bar_blocked=duplicate_bar_blocked,
            daily_ema100=round(ema100_val, 2),
            daily_close=round(daily_close_val, 2),
            daily_above_ema100=daily_above_ema,
            short_blocked_by_ema=short_blocked_by_ema,
        )
        append_decision(d)

        section("FINAL")
        dump_json("Decision", asdict(d), "info")
        logger.info("")
        logger.info(summary_line(
            amp_side=amp_side,
            final_side=final_side,
            entry_allowed=entry_allowed,
            trade_allowed=terminal_trade_allowed,
            action=action,
            symbol=symbol,
        ))

    finally:
        section("SHUTDOWN")
        logger.info("[MT5] shutting down terminal connection")
        try:
            mt5.shutdown()
        except Exception as e:
            logger.warning(f"[MT5] shutdown exception ignored: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("fatal error")
        raise