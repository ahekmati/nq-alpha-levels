#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from hmmlearn.hmm import GaussianHMM
from mt5linux import MetaTrader5

AMP_URL = os.getenv("AMP_URL", "https://ampfutures.isystems.com/Systems/TopStrategies")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
TOP_N = int(os.getenv("TOP_N", "10"))
CONSENSUS_MIN = int(os.getenv("CONSENSUS_MIN", "3"))
STRONG_CONSENSUS_MIN = int(os.getenv("STRONG_CONSENSUS_MIN", "4"))

MT5_HOST = os.getenv("MT5_HOST", "127.0.0.1")
MT5_PORT = int(os.getenv("MT5_PORT", "18812"))
MT5_SYMBOL = os.getenv("MT5_SYMBOL", "MNQM26")
TIMEFRAME_NAME = os.getenv("MT5_TIMEFRAME", "H1")
DEVIATION = int(os.getenv("DEVIATION", "20"))
MAGIC = int(os.getenv("MAGIC", "26050902"))
BASE_LOT = float(os.getenv("BASE_LOT", "1.0"))
DOUBLE_LOT = float(os.getenv("DOUBLE_LOT", "2.0"))
ALLOW_DOUBLE_SIZE = os.getenv("ALLOW_DOUBLE_SIZE", "1").strip() == "1"
DRY_RUN = os.getenv("DRY_RUN", "1").strip() == "1"
ALLOW_DIRECT_FLIP = os.getenv("ALLOW_DIRECT_FLIP", "1").strip() == "1"
ENFORCE_ONE_POSITION = True

HMM_ENABLED = os.getenv("HMM_ENABLED", "1").strip() == "1"
HMM_STATES = int(os.getenv("HMM_STATES", "2"))
HMM_TRAIN_BARS = int(os.getenv("HMM_TRAIN_BARS", str(252 * 24)))
HMM_BULL_BLOCK = float(os.getenv("HMM_BULL_BLOCK", "0.60"))
HMM_BEAR_BLOCK = float(os.getenv("HMM_BEAR_BLOCK", "0.40"))
HMM_UNCERTAIN_LOW = float(os.getenv("HMM_UNCERTAIN_LOW", "0.45"))
HMM_UNCERTAIN_HIGH = float(os.getenv("HMM_UNCERTAIN_HIGH", "0.55"))

KALMAN_ENABLED = os.getenv("KALMAN_ENABLED", "1").strip() == "1"
KALMAN_Q_LEVEL = float(os.getenv("KALMAN_Q_LEVEL", "1e-4"))
KALMAN_Q_VEL = float(os.getenv("KALMAN_Q_VEL", "1e-5"))
KALMAN_R_OBS = float(os.getenv("KALMAN_R_OBS", "1e-3"))

STOP_MODE = os.getenv("STOP_MODE", "dynamic").strip().lower()
FIXED_SL_POINTS = float(os.getenv("FIXED_SL_POINTS", "180"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
DYNAMIC_STOP_ATR_MULT = float(os.getenv("DYNAMIC_STOP_ATR_MULT", "2.2"))
MIN_DYNAMIC_SL_POINTS = float(os.getenv("MIN_DYNAMIC_SL_POINTS", "140"))
MAX_DYNAMIC_SL_POINTS = float(os.getenv("MAX_DYNAMIC_SL_POINTS", "240"))

ENABLE_BREAKEVEN = os.getenv("ENABLE_BREAKEVEN", "1").strip() == "1"
BREAKEVEN_R_MULT = float(os.getenv("BREAKEVEN_R_MULT", "1.0"))
ENABLE_TRAILING_AFTER_BE = os.getenv("ENABLE_TRAILING_AFTER_BE", "0").strip() == "1"
TRAIL_ATR_MULT = float(os.getenv("TRAIL_ATR_MULT", "1.5"))

ENABLE_SESSION_FILTER = os.getenv("ENABLE_SESSION_FILTER", "1").strip() == "1"
ALLOWED_UTC_WINDOWS = os.getenv("ALLOWED_UTC_WINDOWS", "6-20")
BLOCK_DATES = {d.strip() for d in os.getenv("BLOCK_DATES", "").split(",") if d.strip()}

ONE_LOSS_PER_DAY = os.getenv("ONE_LOSS_PER_DAY", "1").strip() == "1"

DEBUG_MODE = os.getenv("DEBUG", "1").strip() == "1"
NO_COLOR = os.getenv("NO_COLOR", "0").strip() == "1"

LOG_DIR = Path(os.getenv("LOG_DIR", "./history"))
STATE_FILE = LOG_DIR / "amp_hmm_mt5_state.json"
DECISION_LOG = LOG_DIR / "amp_hmm_mt5_decisions.jsonl"
RUN_LOG = LOG_DIR / "amp_hmm_mt5_terminal.log"

LOG_DIR.mkdir(parents=True, exist_ok=True)


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


logger = logging.getLogger("amp_hmm_mt5_v2_mt5linux")
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
    msg = f"{key:<28}: {value}"
    getattr(logger, level)(msg)


def dump_json(title: str, obj: Any, level: str = "debug") -> None:
    getattr(logger, level)(f"{title}: {json.dumps(obj, ensure_ascii=False, indent=2, default=str)}")


def side_color(side: str) -> str:
    side = (side or "").lower()
    if side in ("long", "bull", "up"):
        return C.BRIGHT_GREEN
    if side in ("short", "bear", "down"):
        return C.BRIGHT_RED
    if side in ("flat", "neutral", "idle"):
        return C.BRIGHT_YELLOW
    return C.WHITE


def bool_color(flag: bool) -> str:
    return C.BRIGHT_GREEN if flag else C.BRIGHT_RED


def summary_line(
    amp_side: str,
    hmm_side: str,
    kalman_side_: str,
    final_side: str,
    entry_allowed: bool,
    trade_allowed: Optional[bool],
    action: str,
) -> str:
    parts = [
        color_text("SUMMARY", C.BOLD + C.BRIGHT_CYAN),
        f"AMP={color_text(amp_side, side_color(amp_side))}",
        f"HMM={color_text(hmm_side, side_color(hmm_side))}",
        f"KALMAN={color_text(kalman_side_, side_color(kalman_side_))}",
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
    amp_side: str
    amp_consensus_long: int
    amp_consensus_short: int
    amp_consensus_strength: int
    hmm_side: str
    hmm_prob_bull: Optional[float]
    kalman_side: str
    final_side: str
    final_lot: float
    strong_consensus: bool
    action: str
    note: str
    mt5_symbol: str
    dry_run: bool
    duplicate_bar_blocked: bool


@dataclass
class LiveState:
    last_bar_time: str = ""
    last_signal_side: str = "flat"
    last_trade_day: str = ""
    loss_block_day: str = ""
    last_closed_ticket: int = 0

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
        return cls(**data)


def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", (x or "").strip())


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
    kv("mt5linux host", MT5_HOST)
    kv("mt5linux port", MT5_PORT)
    return MetaTrader5(host=MT5_HOST, port=MT5_PORT)


def fetch_amp_html() -> str:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    }
    kv("AMP URL", AMP_URL)
    r = requests.get(AMP_URL, headers=headers, timeout=REQUEST_TIMEOUT)
    kv("AMP HTTP status", r.status_code)
    kv("AMP HTML bytes", len(r.text))
    r.raise_for_status()
    return r.text


def parse_current_session(html: str) -> List[ScrapedRow]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="tableCurrentSession")

    if table:
        logger.info("[AMP] Found tableCurrentSession directly")
    else:
        logger.warning("[AMP] tableCurrentSession not found, using header-based fallback")
        candidates = soup.find_all("table")
        best_score = -1
        best_table = None

        for idx, t in enumerate(candidates):
            headers = [normalize_text(th.get_text(" ", strip=True)).upper() for th in t.find_all("th")]
            header_set = set(h for h in headers if h)

            score = 0
            if "#" in header_set or "RANK" in header_set:
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
            table = best_table
            logger.info(f"[AMP] Selected fallback table with score={best_score}")
        else:
            raise RuntimeError("Could not locate AMP current-session table in HTML")

    body = table.find("tbody") or table
    rows: List[ScrapedRow] = []

    for tr_idx, tr in enumerate(body.find_all("tr")):
        tds = tr.find_all("td")
        if not tds:
            continue

        text_cells = [normalize_text(td.get_text(" ", strip=True)) for td in tds]
        logger.debug(f"[AMP] row[{tr_idx}] cells={text_cells}")

        rank = None
        for i in range(min(3, len(text_cells))):
            m = re.search(r"#?(\d+)", text_cells[i])
            if m:
                rank = int(m.group(1))
                break

        if rank is None:
            logger.debug(f"[AMP] row[{tr_idx}] skipped: no rank found")
            continue
        if rank > TOP_N:
            logger.debug(f"[AMP] row[{tr_idx}] skipped: rank {rank} > TOP_N {TOP_N}")
            continue

        system = text_cells[2] if len(text_cells) > 2 else ""
        product = text_cells[3] if len(text_cells) > 3 else ""
        developer = text_cells[5] if len(text_cells) > 5 else ""
        pnl_text = text_cells[6] if len(text_cells) > 6 else ""
        current_position = text_cells[7] if len(text_cells) > 7 else ""
        nearest_order = text_cells[8] if len(text_cells) > 8 else ""

        pnl = money_to_float(pnl_text)
        if pnl is None:
            logger.warning(f"[AMP] rank={rank} skipped: pnl parse failed from '{pnl_text}'")
            continue

        row = ScrapedRow(
            rank=rank,
            system=system,
            product=product.upper(),
            pnl=pnl,
            current_position=current_position,
            nearest_order=nearest_order,
            developer=developer,
        )
        rows.append(row)

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
    exact_map = {"NQ": "NQ", "MNQ": "MNQ", "ES": "ES", "MES": "MES", "YM": "YM", "MYM": "MYM"}
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


def count_directional_consensus(rows: List[ScrapedRow]) -> Dict[str, int]:
    long_count, short_count = 0, 0
    for r in rows:
        root = product_root(r.product, r.system)
        pos = parse_position_text(r.current_position)
        logger.debug(f"[AMP] consensus row rank={r.rank} root={root} pos={pos} product={r.product} system={r.system}")
        if root in ("NQ", "MNQ", "ES", "MES", "YM", "MYM") and pos in ("long", "short"):
            if pos == "long":
                long_count += 1
            else:
                short_count += 1
    return {"long_count": long_count, "short_count": short_count}


def decide_amp_side(rows: List[ScrapedRow]) -> Tuple[str, Dict[str, int], bool]:
    c = count_directional_consensus(rows)
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


def init_mt5(mt5: MetaTrader5):
    ok = mt5.initialize()
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


def get_timeframe(mt5: MetaTrader5):
    mapping = {"H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}
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


def get_positions(mt5: MetaTrader5, symbol: str):
    pos = mt5.positions_get(symbol=symbol)
    pos = list(pos) if pos else []
    kv("Open positions", len(pos))
    for p in pos:
        logger.info(
            f"[POS] ticket={p.ticket} type={p.type} volume={p.volume} "
            f"price_open={p.price_open} sl={p.sl} tp={p.tp}"
        )
    return pos


def run_kalman(close: pd.Series) -> pd.Series:
    x = np.zeros(2)
    P = np.eye(2)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[KALMAN_Q_LEVEL, 0.0], [0.0, KALMAN_Q_VEL]])
    R = np.array([[KALMAN_R_OBS]])
    out = []
    for z in close.astype(float).values:
        x = F @ x
        P = F @ P @ F.T + Q
        y = np.array([[z]]) - H @ x.reshape(-1, 1)
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = (x.reshape(-1, 1) + K @ y).ravel()
        P = (np.eye(2) - K @ H) @ P
        out.append(x[1])
    return pd.Series(out, index=close.index, name="kalman_velocity")


def hmm_filter_side(bars: pd.DataFrame) -> Tuple[str, Optional[float]]:
    if not HMM_ENABLED:
        logger.info("[HMM] disabled")
        return "neutral", None

    d = bars.copy()
    d["log_ret"] = np.log(d["close"] / d["close"].shift(1))
    d["vol20"] = d["log_ret"].rolling(20).std()
    d = d.dropna()
    kv("HMM rows after prep", len(d))

    if len(d) < 120:
        logger.warning(f"[HMM] not enough rows ({len(d)})")
        return "neutral", None

    X = d[["log_ret", "vol20"]].replace([np.inf, -np.inf], np.nan).dropna().values
    model = GaussianHMM(n_components=HMM_STATES, covariance_type="diag", n_iter=200, random_state=42)
    model.fit(X)
    bull_state = int(np.argmax(model.means_[:, 0]))
    post = model.predict_proba(X)
    p_bull = float(post[-1, bull_state])

    if HMM_UNCERTAIN_LOW <= p_bull <= HMM_UNCERTAIN_HIGH:
        side = "neutral"
    elif p_bull > 0.5:
        side = "bull"
    else:
        side = "bear"

    kv("HMM side", color_text(side, side_color(side)))
    kv("HMM p_bull", round(p_bull, 6))
    logger.debug(f"[HMM] bull_state={bull_state} means={model.means_.tolist()}")
    return side, p_bull


def allowed_by_hmm(amp_side: str, hmm_side: str, p_bull: Optional[float]) -> bool:
    if not HMM_ENABLED or p_bull is None:
        logger.info("[HMM] no veto applied")
        return True
    if amp_side == "long" and p_bull < HMM_BEAR_BLOCK:
        logger.warning(f"[HMM] blocking LONG because p_bull={p_bull:.4f} < {HMM_BEAR_BLOCK}")
        return False
    if amp_side == "short" and p_bull > HMM_BULL_BLOCK:
        logger.warning(f"[HMM] blocking SHORT because p_bull={p_bull:.4f} > {HMM_BULL_BLOCK}")
        return False
    logger.info("[HMM] passed filter")
    return True


def kalman_side(bars: pd.DataFrame) -> str:
    if not KALMAN_ENABLED:
        logger.info("[KALMAN] disabled")
        return "neutral"
    kv_series = run_kalman(bars["close"])
    v = float(kv_series.iloc[-1])
    side = "up" if v > 0 else "down" if v < 0 else "neutral"
    kv("Kalman velocity", round(v, 8))
    kv("Kalman side", color_text(side, side_color(side)))
    return side


def calc_atr_points(mt5: MetaTrader5, bars: pd.DataFrame) -> float:
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
    atr = tr.rolling(ATR_PERIOD).mean().iloc[-1]
    info = mt5.symbol_info(MT5_SYMBOL)
    points = atr / info.point if info and info.point else FIXED_SL_POINTS
    kv("ATR", round(float(atr), 5))
    kv("ATR points", round(float(points), 2))
    return float(points)


def choose_stop_points(mt5: MetaTrader5, bars: pd.DataFrame) -> float:
    if STOP_MODE == "fixed":
        kv("Stop mode", f"fixed ({FIXED_SL_POINTS} points)")
        return FIXED_SL_POINTS
    atr_pts = calc_atr_points(mt5, bars)
    sl = atr_pts * DYNAMIC_STOP_ATR_MULT
    sl = max(MIN_DYNAMIC_SL_POINTS, min(MAX_DYNAMIC_SL_POINTS, sl))
    kv("Stop mode", "dynamic")
    kv("Dynamic stop points", round(sl, 2))
    return sl


def decide_lot(amp_side: str, strong_consensus: bool, hmm_side: str, p_bull: Optional[float], k_side: str) -> float:
    lot = BASE_LOT
    kalman_agree = (amp_side == "long" and k_side == "up") or (amp_side == "short" and k_side == "down")
    hmm_agree = (amp_side == "long" and hmm_side == "bull") or (amp_side == "short" and hmm_side == "bear")
    kv("Sizing amp_side", amp_side)
    kv("Sizing strong_consensus", strong_consensus)
    kv("Sizing kalman_agree", kalman_agree)
    kv("Sizing hmm_agree", hmm_agree)
    if ALLOW_DOUBLE_SIZE and strong_consensus and kalman_agree and hmm_agree:
        lot = DOUBLE_LOT
    kv("Chosen lot", lot)
    return lot


def side_from_positions(mt5: MetaTrader5, positions) -> str:
    if not positions:
        return "flat"
    return "long" if positions[0].type == mt5.POSITION_TYPE_BUY else "short"


def can_trade_now(now_utc: datetime) -> Tuple[bool, str]:
    dstr = now_utc.date().isoformat()
    if dstr in BLOCK_DATES:
        return False, f"blocked date {dstr}"
    if ENABLE_SESSION_FILTER:
        hour = now_utc.hour
        allowed = False
        for part in ALLOWED_UTC_WINDOWS.split(","):
            start_s, end_s = part.strip().split("-")
            if int(start_s) <= hour < int(end_s):
                allowed = True
                break
        if not allowed:
            return False, f"hour {hour} UTC outside allowed windows {ALLOWED_UTC_WINDOWS}"
    return True, "ok"


def build_entry_request(mt5: MetaTrader5, side: str, lot: float, sl_points: float) -> Dict[str, Any]:
    info = mt5.symbol_info(MT5_SYMBOL)
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    if info is None or tick is None:
        raise RuntimeError(f"Could not get symbol info/tick for {MT5_SYMBOL}")

    point = info.point
    if side == "long":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - sl_points * point
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + sl_points * point

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": MT5_SYMBOL,
        "volume": float(lot),
        "type": order_type,
        "price": float(price),
        "sl": float(sl),
        "tp": 0.0,
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "AMP+HMM v2 mt5linux",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    dump_json("Order request", request, "info")
    return request


def send_request(mt5: MetaTrader5, request: Dict[str, Any], label: str) -> Dict[str, Any]:
    logger.info(f"[ORDER] {label} send requested")
    if DRY_RUN:
        logger.warning(f"[ORDER] DRY_RUN=1; {label} not sent")
        return {"dry_run": True, "request": request}

    check = mt5.order_check(request)
    kv(f"{label} order_check retcode", getattr(check, "retcode", None))
    kv(f"{label} order_check comment", getattr(check, "comment", ""))

    result = mt5.order_send(request)
    payload = result._asdict() if hasattr(result, "_asdict") else {"result": str(result)}
    dump_json(f"{label} order_send payload", payload, "info")
    return {
        "dry_run": False,
        "check": getattr(check, "_asdict", lambda: str(check))(),
        "result": payload,
        "request": request,
    }


def close_position(mt5: MetaTrader5, pos) -> Dict[str, Any]:
    tick = mt5.symbol_info_tick(MT5_SYMBOL)
    is_buy = pos.type == mt5.POSITION_TYPE_BUY
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": pos.ticket,
        "symbol": MT5_SYMBOL,
        "volume": float(pos.volume),
        "type": mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
        "price": float(tick.bid if is_buy else tick.ask),
        "deviation": DEVIATION,
        "magic": MAGIC,
        "comment": "AMP+HMM v2 close mt5linux",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    dump_json(f"Close request ticket={pos.ticket}", request, "info")
    return send_request(mt5, request, f"close ticket {pos.ticket}")


def move_stop_if_needed(mt5: MetaTrader5, positions, bars: pd.DataFrame) -> None:
    if not ENABLE_BREAKEVEN or not positions:
        logger.info("[MANAGE] breakeven/trailing skipped")
        return

    info = mt5.symbol_info(MT5_SYMBOL)
    point = info.point if info else 0.25
    atr_pts = calc_atr_points(mt5, bars)

    for pos in positions:
        side = "long" if pos.type == mt5.POSITION_TYPE_BUY else "short"
        tick = mt5.symbol_info_tick(MT5_SYMBOL)
        current = tick.bid if side == "long" else tick.ask
        risk_pts = abs(pos.price_open - pos.sl) / point if pos.sl else 0.0
        if risk_pts <= 0:
            logger.warning(f"[MANAGE] ticket={pos.ticket} no valid risk_pts; skipping")
            continue

        move_pts = (current - pos.price_open) / point if side == "long" else (pos.price_open - current) / point
        logger.info(
            f"[MANAGE] ticket={pos.ticket} side={side} entry={pos.price_open} current={current} "
            f"sl={pos.sl} move_pts={move_pts:.2f} risk_pts={risk_pts:.2f}"
        )

        if move_pts >= risk_pts * BREAKEVEN_R_MULT:
            new_sl = pos.price_open
            improve = (side == "long" and (pos.sl == 0 or new_sl > pos.sl)) or (
                side == "short" and (pos.sl == 0 or new_sl < pos.sl)
            )
            if improve:
                req = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": MT5_SYMBOL,
                    "position": pos.ticket,
                    "sl": float(new_sl),
                    "tp": float(pos.tp),
                }
                dump_json(f"Breakeven request ticket={pos.ticket}", req, "info")
                send_request(mt5, req, f"breakeven ticket {pos.ticket}")

        if ENABLE_TRAILING_AFTER_BE:
            trail_pts = atr_pts * TRAIL_ATR_MULT
            if side == "long":
                trail_sl = current - trail_pts * point
                if trail_sl > max(pos.sl, pos.price_open):
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": MT5_SYMBOL,
                        "position": pos.ticket,
                        "sl": float(trail_sl),
                        "tp": float(pos.tp),
                    }
                    dump_json(f"Trail request ticket={pos.ticket}", req, "info")
                    send_request(mt5, req, f"trail ticket {pos.ticket}")
            else:
                trail_sl = current + trail_pts * point
                if pos.sl == 0 or trail_sl < min(pos.sl, pos.price_open):
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": MT5_SYMBOL,
                        "position": pos.ticket,
                        "sl": float(trail_sl),
                        "tp": float(pos.tp),
                    }
                    dump_json(f"Trail request ticket={pos.ticket}", req, "info")
                    send_request(mt5, req, f"trail ticket {pos.ticket}")


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
    if DRY_RUN:
        logger.info("[RISK] DRY_RUN active; closed-loss scan skipped")
        return

    today = now_utc.date()
    deals = mt5.history_deals_get(datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc), now_utc)
    if deals is None:
        logger.warning(f"[RISK] history_deals_get returned None last_error={mt5.last_error()}")
        return

    worst = None
    for d in deals:
        if getattr(d, "magic", None) != MAGIC:
            continue
        if getattr(d, "entry", None) == mt5.DEAL_ENTRY_OUT:
            pnl = float(getattr(d, "profit", 0.0)) + float(getattr(d, "commission", 0.0)) + float(getattr(d, "swap", 0.0))
            logger.info(f"[RISK] closed deal ticket={d.ticket} pnl={pnl}")
            if state.last_closed_ticket and d.ticket <= state.last_closed_ticket:
                continue
            if worst is None or d.ticket > worst.ticket:
                worst = d

    if worst is not None:
        pnl = float(getattr(worst, "profit", 0.0)) + float(getattr(worst, "commission", 0.0)) + float(getattr(worst, "swap", 0.0))
        state.last_closed_ticket = int(worst.ticket)
        if pnl < 0:
            state.loss_block_day = now_utc.date().isoformat()
            logger.warning(f"[RISK] loss detected; blocking entries for rest of day {state.loss_block_day}")


def main() -> None:
    now_utc = datetime.now(timezone.utc)

    banner(f"START amp_hmm_mt5_v2_mt5linux | now_utc={now_utc.isoformat()} | symbol={MT5_SYMBOL} | dry_run={DRY_RUN}")

    section("CONFIG")
    kv("DEBUG_MODE", DEBUG_MODE)
    kv("NO_COLOR", NO_COLOR)
    kv("Symbol", MT5_SYMBOL)
    kv("Timeframe", TIMEFRAME_NAME)
    kv("Consensus min", CONSENSUS_MIN)
    kv("Strong consensus min", STRONG_CONSENSUS_MIN)
    kv("Base lot", BASE_LOT)
    kv("Double lot", DOUBLE_LOT)
    kv("Stop mode", STOP_MODE)
    kv("Session filter", ENABLE_SESSION_FILTER)
    kv("Allowed UTC windows", ALLOWED_UTC_WINDOWS)
    kv("Dry run", DRY_RUN)
    kv("Log file", RUN_LOG)

    section("STATE")
    state = LiveState.load(STATE_FILE)
    dump_json("Loaded state", asdict(state), "info")

    section("FILTERS")
    allowed_time, time_reason = can_trade_now(now_utc)
    kv("Time/date allowed", allowed_time, "warning" if not allowed_time else "info")
    kv("Time/date reason", time_reason, "warning" if not allowed_time else "info")

    section("MT5 INIT")
    mt5 = make_mt5()
    terminal_info = None

    try:
        terminal_info = init_mt5(mt5)

        section("MARKET DATA")
        bars = get_bars(mt5, MT5_SYMBOL, max(HMM_TRAIN_BARS, 500))
        latest_bar = bars.index[-1].isoformat()
        kv("Latest closed bar", latest_bar)

        duplicate_bar_blocked = state.last_bar_time == latest_bar and state.last_bar_time != ""
        kv("Previous bar in state", state.last_bar_time or "<empty>")
        kv("Duplicate bar blocked", duplicate_bar_blocked, "warning" if duplicate_bar_blocked else "info")

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
        else:
            amp_side, consensus, strong_consensus = decide_amp_side(rows)

        kv("AMP side", color_text(amp_side, side_color(amp_side)))
        kv("AMP long count", consensus["long_count"])
        kv("AMP short count", consensus["short_count"])
        kv("AMP strong consensus", strong_consensus)

        section("MODELS")
        hmm_side, p_bull = hmm_filter_side(bars)
        kalman_side_ = kalman_side(bars)

        section("DECISION")
        if amp_side == "flat":
            final_side = "flat"
            final_lot = 0.0
            note = "AMP consensus not strong enough or no systems available"
        elif not allowed_by_hmm(amp_side, hmm_side, p_bull):
            final_side = "flat"
            final_lot = 0.0
            note = "Blocked by HMM"
        else:
            final_side = amp_side
            final_lot = decide_lot(amp_side, strong_consensus, hmm_side, p_bull, kalman_side_)
            note = "Signal approved"

        kv("Final side", color_text(final_side, side_color(final_side)))
        kv("Final lot", final_lot)
        kv("Decision note", note)

        if not allowed_time:
            logger.warning("[FILTER] outside allowed trading time/date -> no new entry allowed")
        if not can_trade_loss:
            logger.warning("[FILTER] one-loss-per-day active -> no new entry allowed")
        if duplicate_bar_blocked:
            logger.warning("[FILTER] duplicate closed bar -> signal execution blocked this run")

        section("POSITIONS")
        positions = get_positions(mt5, MT5_SYMBOL)
        existing_side = side_from_positions(mt5, positions)
        kv("Existing side", color_text(existing_side, side_color(existing_side)))

        section("POSITION MANAGEMENT")
        move_stop_if_needed(mt5, positions, bars)

        section("EXECUTION")
        action = "hold"

        if ENFORCE_ONE_POSITION and len(positions) > 1:
            logger.error("[SAFETY] more than 1 position detected; forcing no new orders")
            final_side = "flat"
            note = "safety stop: more than one position detected"

        entry_allowed = allowed_time and can_trade_loss and (not duplicate_bar_blocked)
        kv("Entry allowed", entry_allowed, "warning" if not entry_allowed else "info")

        if existing_side == "flat":
            if final_side in ("long", "short") and entry_allowed:
                sl_points = choose_stop_points(mt5, bars)
                req = build_entry_request(mt5, final_side, final_lot, sl_points)
                res = send_request(mt5, req, "entry")
                action = "enter"
                dump_json("Entry response", res, "info")
            else:
                logger.info("[ENTRY] no open position and no entry sent")
                action = "idle"
        else:
            if duplicate_bar_blocked:
                logger.warning("[HOLD] duplicate bar protection active; no close/flip")
                action = "hold"
            elif final_side == "flat":
                logger.info("[EXIT] existing position but combined signal is flat -> close")
                for p in positions:
                    res = close_position(mt5, p)
                    dump_json(f"Exit response ticket={p.ticket}", res, "info")
                action = "close"
            elif existing_side != final_side and ALLOW_DIRECT_FLIP:
                logger.info(f"[FLIP] existing_side={existing_side} final_side={final_side} -> close then reopen")
                for p in positions:
                    res = close_position(mt5, p)
                    dump_json(f"Flip close response ticket={p.ticket}", res, "info")
                if entry_allowed:
                    sl_points = choose_stop_points(mt5, bars)
                    req = build_entry_request(mt5, final_side, final_lot, sl_points)
                    res = send_request(mt5, req, "flip entry")
                    dump_json("Flip entry response", res, "info")
                action = "flip"
            else:
                logger.info("[HOLD] existing position remains aligned")
                action = "hold"

        section("SAVE STATE")
        if not duplicate_bar_blocked:
            state.last_bar_time = latest_bar
        state.last_signal_side = final_side
        state.save(STATE_FILE)
        dump_json("Saved state", asdict(state), "info")

        d = Decision(
            timestamp_utc=now_utc.isoformat(),
            amp_side=amp_side,
            amp_consensus_long=consensus["long_count"],
            amp_consensus_short=consensus["short_count"],
            amp_consensus_strength=max(consensus["long_count"], consensus["short_count"]),
            hmm_side=hmm_side,
            hmm_prob_bull=p_bull,
            kalman_side=kalman_side_,
            final_side=final_side,
            final_lot=final_lot,
            strong_consensus=strong_consensus,
            action=action,
            note=note,
            mt5_symbol=MT5_SYMBOL,
            dry_run=DRY_RUN,
            duplicate_bar_blocked=duplicate_bar_blocked,
        )
        append_decision(d)

        section("FINAL")
        dump_json("Decision", asdict(d), "info")

        trade_allowed = getattr(terminal_info, "trade_allowed", None) if terminal_info is not None else None
        logger.info("")
        logger.info(summary_line(
            amp_side=amp_side,
            hmm_side=hmm_side,
            kalman_side_=kalman_side_,
            final_side=final_side,
            entry_allowed=entry_allowed,
            trade_allowed=trade_allowed,
            action=action,
        ))

    finally:
        section("SHUTDOWN")
        logger.info("[MT5] shutting down bridge connection")
        try:
            mt5.shutdown()
        except Exception as e:
            logger.warning(f"[MT5] shutdown exception ignored: {e}")

    banner("END amp_hmm_mt5_v2_mt5linux")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception:
        logger.exception("fatal error")
        raise
