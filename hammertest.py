#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import pytz

try:
    from mt5linux import MetaTrader5
except ImportError:
    MetaTrader5 = None

UTC = pytz.UTC
POINT_VALUE_USD = 2.0
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18812

mt5 = None
TIMEFRAME_MAP: Dict[str, int] = {}


@dataclass
class HammerSpec:
    name: str
    min_lower_to_body: float
    max_upper_to_body: float
    max_body_to_range: float
    min_close_location: float
    bullish_only: bool = True


HAMMER_SPECS = {
    "loose": HammerSpec("loose", 1.5, 1.5, 0.45, 0.55, True),
    "standard": HammerSpec("standard", 2.0, 1.0, 0.35, 0.60, True),
    "strict": HammerSpec("strict", 2.5, 0.75, 0.30, 0.65, True),
    "extreme": HammerSpec("extreme", 3.0, 0.50, 0.25, 0.70, True),
}


def parse_csv_list(s: str, cast=str) -> List:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    return [cast(v) for v in vals]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNQ hammer optimizer via mt5linux")
    p.add_argument("--symbol", default="@MNQ")
    p.add_argument("--timeframes", default="M15,M30,H1,H4")
    p.add_argument("--from", dest="date_from", default="2018-01-01")
    p.add_argument("--to", dest="date_to", default=None)
    p.add_argument("--outdir", default="output/mnq_hammer_optimizer_mt5linux")
    p.add_argument("--session-tz", default="America/New_York")
    p.add_argument("--cash-start", default="09:30")
    p.add_argument("--cash-end", default="16:00")
    p.add_argument("--hammer-specs", default="standard,strict,extreme")
    p.add_argument("--entry-modes", default="next_open,break_high")
    p.add_argument("--stop-modes", default="hammer_low_buffer,atr_multiple")
    p.add_argument("--target-modes", default="r_multiple,time_exit_only")
    p.add_argument("--stop-buffer-points-list", default="4,8,12")
    p.add_argument("--atr-stop-mults", default="0.75,1.0,1.25")
    p.add_argument("--r-multiples", default="1.0,1.5,2.0")
    p.add_argument("--atr-target-mults", default="1.0,1.5,2.0")
    p.add_argument("--max-hold-bars-list", default="6,12,24")
    p.add_argument("--cooldown-bars-list", default="0,4")
    p.add_argument("--contracts", type=int, default=1)
    p.add_argument("--starting-capital", type=float, default=5000.0)
    p.add_argument("--commission-per-side", type=float, default=0.0)
    p.add_argument("--slippage-points", type=float, default=0.0)
    p.add_argument("--be-trigger-list", default="0,150,300")
    p.add_argument("--be-plus-list", default="0,8")
    p.add_argument("--trail-atr-mults", default="0,1.0")
    p.add_argument("--require-above-ema50-options", default="0,1")
    p.add_argument("--require-above-ema100-options", default="0,1")
    p.add_argument("--require-prior-5bar-down-options", default="0,1")
    p.add_argument("--require-prior-10bar-down-options", default="0,1")
    p.add_argument("--rsi-below-options", default="none,35,40")
    p.add_argument("--volume-z-options", default="none,0.5,1.0")
    p.add_argument("--cash-session-only-options", default="0,1")
    p.add_argument("--min-trades", type=int, default=40)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--mt5-host", default=DEFAULT_HOST)
    p.add_argument("--mt5-port", type=int, default=DEFAULT_PORT)
    return p.parse_args()


def ensure_mt5(host: str, port: int):
    global mt5, TIMEFRAME_MAP
    if MetaTrader5 is None:
        raise RuntimeError("mt5linux package not installed. Install with: pip install mt5linux")
    mt5 = MetaTrader5(host=host, port=port)
    if not mt5.initialize():
        raise RuntimeError(
            f"mt5.initialize() failed via mt5linux at {host}:{port}. "
            f"Make sure the mt5linux server is running and MT5 is open. "
            f"last_error={mt5.last_error()}"
        )
    TIMEFRAME_MAP = {
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }


def shutdown_mt5():
    global mt5
    if mt5 is not None:
        try:
            mt5.shutdown()
        except Exception:
            pass


def ensure_symbol(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info failed for {symbol}: {mt5.last_error()}")
    if not info.visible:
        ok = mt5.symbol_select(symbol, True)
        if not ok:
            raise RuntimeError(f"symbol_select failed for {symbol}: {mt5.last_error()}")


def fetch_bars(symbol: str, timeframe: str, date_from: str, date_to: Optional[str]) -> pd.DataFrame:
    tf = TIMEFRAME_MAP.get(timeframe)
    if tf is None:
        raise RuntimeError(f"Unsupported timeframe: {timeframe}")

    ensure_symbol(symbol)

    utc_from = UTC.localize(datetime.fromisoformat(date_from))
    utc_to = UTC.localize(datetime.fromisoformat(date_to)) if date_to else datetime.now(UTC)

    rates = None
    range_error = None

    try:
        rates = mt5.copy_rates_range(symbol, tf, utc_from, utc_to)
        if rates is None:
            range_error = mt5.last_error()
    except Exception as e:
        range_error = f"copy_rates_range exception: {e}"

    if rates is None or len(rates) == 0:
        try:
            bars_guess = {
                "M5": 120000,
                "M15": 120000,
                "M30": 100000,
                "H1": 60000,
                "H4": 30000,
                "D1": 12000,
            }.get(timeframe, 50000)

            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars_guess)
            if rates is None:
                raise RuntimeError(f"copy_rates_from_pos failed: {mt5.last_error()} | prior={range_error}")
        except Exception as e:
            raise RuntimeError(f"All bar retrieval methods failed for {symbol} {timeframe}: {e} | prior={range_error}")

    if len(rates) == 0:
        raise RuntimeError(
            f"No bars returned for {symbol} {timeframe}. "
            f"Open the chart in MT5, scroll back, and increase Max bars in chart."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)

    if date_from:
        df = df[df["time"] >= utc_from]
    if date_to:
        df = df[df["time"] <= utc_to]

    if df.empty:
        raise RuntimeError(
            f"Bars were fetched for {symbol} {timeframe}, but none remained after date filtering. "
            f"Check available MT5 history for the requested date range."
        )

    return df.reset_index(drop=True)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_features(df: pd.DataFrame, session_tz: str, cash_start: str, cash_end: str) -> pd.DataFrame:
    df = df.copy()
    local = df["time"].dt.tz_convert(session_tz)
    df["local_time"] = local
    mins = local.dt.hour * 60 + local.dt.minute
    sh, sm = map(int, cash_start.split(":"))
    eh, em = map(int, cash_end.split(":"))
    start_mins = sh * 60 + sm
    end_mins = eh * 60 + em
    df["is_cash"] = (mins >= start_mins) & (mins <= end_mins)
    df["year"] = local.dt.year

    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"] = body
    df["range"] = rng
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
    df["body_to_range"] = body / rng
    df["lower_to_body"] = df["lower_wick"] / body.replace(0, np.nan)
    df["upper_to_body"] = df["upper_wick"] / body.replace(0, np.nan)
    df["close_location"] = ((df["close"] - df["low"]) / rng).clip(0, 1)
    df["bullish_body"] = df["close"] > df["open"]

    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["ema100"] = df["close"].ewm(span=100, adjust=False).mean()
    df["above_ema50"] = df["close"] > df["ema50"]
    df["above_ema100"] = df["close"] > df["ema100"]
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)

    vol_mean = df["tick_volume"].rolling(20).mean()
    vol_std = df["tick_volume"].rolling(20).std()
    df["volume_z20"] = (df["tick_volume"] - vol_mean) / vol_std

    df["ret_5bar"] = df["close"].pct_change(5)
    df["ret_10bar"] = df["close"].pct_change(10)
    return df


def mark_hammer(df: pd.DataFrame, spec: HammerSpec) -> pd.Series:
    cond = (
        (df["lower_to_body"] >= spec.min_lower_to_body)
        & (df["upper_to_body"] <= spec.max_upper_to_body)
        & (df["body_to_range"] <= spec.max_body_to_range)
        & (df["close_location"] >= spec.min_close_location)
    )
    if spec.bullish_only:
        cond &= df["bullish_body"]
    return cond.fillna(False)


def points_to_usd(points: float, contracts: int) -> float:
    return points * POINT_VALUE_USD * contracts


def compute_entry(i: int, df: pd.DataFrame, entry_mode: str, slippage_points: float) -> Optional[Tuple[int, float]]:
    if i + 1 >= len(df):
        return None

    next_bar = df.iloc[i + 1]

    if entry_mode == "next_open":
        return i + 1, float(next_bar["open"] + slippage_points)

    trigger = float(df.iloc[i]["high"])
    if float(next_bar["high"]) >= trigger:
        entry = max(float(next_bar["open"]), trigger) + slippage_points
        return i + 1, entry

    return None


def pass_filters(row: pd.Series, cfg: Dict) -> bool:
    if cfg["require_above_ema50"] and not bool(row["above_ema50"]):
        return False
    if cfg["require_above_ema100"] and not bool(row["above_ema100"]):
        return False
    if cfg["require_prior_5bar_down"] and not (pd.notna(row["ret_5bar"]) and row["ret_5bar"] < 0):
        return False
    if cfg["require_prior_10bar_down"] and not (pd.notna(row["ret_10bar"]) and row["ret_10bar"] < 0):
        return False
    if cfg["require_rsi_below"] is not None and not (pd.notna(row["rsi14"]) and row["rsi14"] < cfg["require_rsi_below"]):
        return False
    if cfg["require_volume_z_above"] is not None and not (
        pd.notna(row["volume_z20"]) and row["volume_z20"] > cfg["require_volume_z_above"]
    ):
        return False
    if cfg["cash_session_only"] and not bool(row["is_cash"]):
        return False
    return True


def compute_initial_stop_target(signal_row: pd.Series, entry_price: float, cfg: Dict):
    if cfg["stop_mode"] == "hammer_low_buffer":
        stop_price = float(signal_row["low"] - cfg["stop_buffer_points"])
    elif cfg["stop_mode"] == "fixed_points":
        stop_price = entry_price - cfg["stop_points"]
    else:
        stop_price = entry_price - cfg["atr_stop_mult"] * float(signal_row["atr14"])

    initial_risk_points = entry_price - stop_price

    if cfg["target_mode"] == "time_exit_only":
        target_price = None
    elif cfg["target_mode"] == "fixed_points":
        target_price = entry_price + cfg["target_points"]
    elif cfg["target_mode"] == "atr_multiple":
        target_price = entry_price + cfg["atr_target_mult"] * float(signal_row["atr14"])
    else:
        target_price = entry_price + cfg["r_multiple"] * initial_risk_points

    return stop_price, target_price, initial_risk_points


def simulate_trade(signal_idx: int, entry_idx: int, entry_price: float, df: pd.DataFrame, cfg: Dict):
    signal_row = df.iloc[signal_idx]
    stop_price, target_price, initial_risk_points = compute_initial_stop_target(signal_row, entry_price, cfg)
    if initial_risk_points <= 0:
        return None

    stop_live = stop_price
    target_live = target_price
    highest_high = entry_price
    mfe = -np.inf
    mae = np.inf

    max_exit_idx = min(entry_idx + cfg["max_hold_bars"] - 1, len(df) - 1)
    if max_exit_idx < entry_idx:
        return None

    exit_idx = None
    exit_price = None
    exit_reason = None

    for j in range(entry_idx, max_exit_idx + 1):
        bar = df.iloc[j]
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])

        highest_high = max(highest_high, h)
        mfe = max(mfe, h - entry_price)
        mae = min(mae, l - entry_price)

        if cfg["breakeven_trigger_points"] > 0 and highest_high - entry_price >= cfg["breakeven_trigger_points"]:
            stop_live = max(stop_live, entry_price + cfg["breakeven_plus_points"])

        if cfg["trail_atr_mult"] > 0 and pd.notna(bar["atr14"]):
            trail_stop = h - cfg["trail_atr_mult"] * float(bar["atr14"])
            stop_live = max(stop_live, trail_stop)

        stop_hit = l <= stop_live
        target_hit = target_live is not None and h >= target_live

        if stop_hit and target_hit:
            exit_idx = j
            if o <= stop_live:
                exit_price = o
                exit_reason = "gap_stop"
            elif o >= target_live:
                exit_price = o
                exit_reason = "gap_target"
            else:
                exit_price = stop_live
                exit_reason = "stop_and_target_same_bar_stop_first"
            break
        elif stop_hit:
            exit_idx = j
            exit_price = o if o <= stop_live else stop_live
            exit_reason = "stop"
            break
        elif target_hit:
            exit_idx = j
            exit_price = o if o >= target_live else target_live
            exit_reason = "target"
            break
        elif j == max_exit_idx:
            exit_idx = j
            exit_price = c
            exit_reason = "time_exit"
            break

    if exit_idx is None:
        return None

    gross_points = exit_price - entry_price
    net_points = gross_points - 2.0 * cfg["slippage_points"]
    gross_pnl = points_to_usd(gross_points, cfg["contracts"])
    commission = 2.0 * cfg["commission_per_side"] * cfg["contracts"]
    net_pnl = points_to_usd(net_points, cfg["contracts"]) - commission
    r_mult = gross_points / initial_risk_points if initial_risk_points > 0 else np.nan

    return {
        "entry_time": df.iloc[entry_idx]["time"],
        "exit_time": df.iloc[exit_idx]["time"],
        "exit_idx": exit_idx,
        "year": signal_row["year"],
        "net_pnl": net_pnl,
        "net_points": net_points,
        "r_multiple": r_mult,
        "mfe_points": mfe,
        "mae_points": mae,
        "exit_reason": exit_reason,
    }


def build_equity_curve(trades: pd.DataFrame, starting_capital: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["exit_time", "equity", "drawdown"])
    eq = trades[["exit_time", "net_pnl"]].copy().sort_values("exit_time")
    eq["equity"] = starting_capital + eq["net_pnl"].cumsum()
    eq["peak"] = eq["equity"].cummax()
    eq["drawdown"] = eq["equity"] - eq["peak"]
    return eq


def summarize_trades(trades: pd.DataFrame, cfg: Dict) -> Dict:
    if trades.empty:
        return {
            **cfg,
            "trades": 0,
            "win_rate": np.nan,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "profit_factor": np.nan,
            "avg_trade": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "expectancy_r": np.nan,
            "avg_mfe_points": np.nan,
            "avg_mae_points": np.nan,
            "max_drawdown": 0.0,
            "final_capital": cfg["starting_capital"],
            "net_profit": 0.0,
            "sharpe_like": np.nan,
            "dd_pct_of_start": 0.0,
        }

    wins = trades["net_pnl"] > 0
    gross_profit = trades.loc[trades["net_pnl"] > 0, "net_pnl"].sum()
    gross_loss = -trades.loc[trades["net_pnl"] < 0, "net_pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan
    avg_trade = trades["net_pnl"].mean()
    avg_win = trades.loc[wins, "net_pnl"].mean() if wins.any() else np.nan
    avg_loss = trades.loc[~wins, "net_pnl"].mean() if (~wins).any() else np.nan
    expectancy_r = trades["r_multiple"].mean()
    final_capital = cfg["starting_capital"] + trades["net_pnl"].sum()
    equity = build_equity_curve(trades, cfg["starting_capital"])
    max_dd = equity["drawdown"].min() if not equity.empty else 0.0
    ret_std = trades["net_pnl"].std(ddof=0)
    sharpe_like = avg_trade / ret_std if ret_std and not np.isnan(ret_std) else np.nan
    dd_pct = abs(max_dd) / cfg["starting_capital"] if cfg["starting_capital"] else np.nan

    return {
        **cfg,
        "trades": len(trades),
        "win_rate": wins.mean(),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy_r": expectancy_r,
        "avg_mfe_points": trades["mfe_points"].mean(),
        "avg_mae_points": trades["mae_points"].mean(),
        "max_drawdown": max_dd,
        "final_capital": final_capital,
        "net_profit": trades["net_pnl"].sum(),
        "sharpe_like": sharpe_like,
        "dd_pct_of_start": dd_pct,
    }


def valid_combo(cfg: Dict) -> bool:
    if cfg["stop_mode"] == "hammer_low_buffer" and cfg["stop_buffer_points"] is None:
        return False
    if cfg["stop_mode"] == "atr_multiple" and cfg["atr_stop_mult"] is None:
        return False
    if cfg["target_mode"] == "r_multiple" and cfg["r_multiple"] is None:
        return False
    if cfg["target_mode"] == "atr_multiple" and cfg["atr_target_mult"] is None:
        return False
    if cfg["breakeven_trigger_points"] == 0 and cfg["breakeven_plus_points"] != 0:
        return False
    if cfg["require_above_ema50"] and cfg["require_above_ema100"]:
        return False
    return True


def build_grid(args: argparse.Namespace) -> List[Dict]:
    timeframes = parse_csv_list(args.timeframes, str)
    hammer_specs = parse_csv_list(args.hammer_specs, str)
    entry_modes = parse_csv_list(args.entry_modes, str)
    stop_modes = parse_csv_list(args.stop_modes, str)
    target_modes = parse_csv_list(args.target_modes, str)
    stop_buffer_points_list = parse_csv_list(args.stop_buffer_points_list, float)
    atr_stop_mults = parse_csv_list(args.atr_stop_mults, float)
    r_multiples = parse_csv_list(args.r_multiples, float)
    atr_target_mults = parse_csv_list(args.atr_target_mults, float)
    max_hold_bars_list = parse_csv_list(args.max_hold_bars_list, int)
    cooldown_bars_list = parse_csv_list(args.cooldown_bars_list, int)
    be_trigger_list = parse_csv_list(args.be_trigger_list, float)
    be_plus_list = parse_csv_list(args.be_plus_list, float)
    trail_atr_mults = parse_csv_list(args.trail_atr_mults, float)
    ema50_opts = [bool(int(x)) for x in parse_csv_list(args.require_above_ema50_options, str)]
    ema100_opts = [bool(int(x)) for x in parse_csv_list(args.require_above_ema100_options, str)]
    prior5_opts = [bool(int(x)) for x in parse_csv_list(args.require_prior_5bar_down_options, str)]
    prior10_opts = [bool(int(x)) for x in parse_csv_list(args.require_prior_10bar_down_options, str)]
    cash_opts = [bool(int(x)) for x in parse_csv_list(args.cash_session_only_options, str)]

    def parse_opt_float_list(s: str):
        out = []
        for x in parse_csv_list(s, str):
            out.append(None if x.lower() == "none" else float(x))
        return out

    rsi_opts = parse_opt_float_list(args.rsi_below_options)
    vol_opts = parse_opt_float_list(args.volume_z_options)

    grid = []
    for vals in itertools.product(
        timeframes,
        hammer_specs,
        entry_modes,
        stop_modes,
        target_modes,
        stop_buffer_points_list,
        atr_stop_mults,
        r_multiples,
        atr_target_mults,
        max_hold_bars_list,
        cooldown_bars_list,
        be_trigger_list,
        be_plus_list,
        trail_atr_mults,
        ema50_opts,
        ema100_opts,
        prior5_opts,
        prior10_opts,
        rsi_opts,
        vol_opts,
        cash_opts,
    ):
        cfg = {
            "symbol": args.symbol,
            "timeframe": vals[0],
            "hammer_spec": vals[1],
            "entry_mode": vals[2],
            "stop_mode": vals[3],
            "target_mode": vals[4],
            "stop_buffer_points": vals[5],
            "stop_points": None,
            "atr_stop_mult": vals[6],
            "r_multiple": vals[7],
            "atr_target_mult": vals[8],
            "target_points": None,
            "max_hold_bars": vals[9],
            "cooldown_bars": vals[10],
            "breakeven_trigger_points": vals[11],
            "breakeven_plus_points": vals[12],
            "trail_atr_mult": vals[13],
            "require_above_ema50": vals[14],
            "require_above_ema100": vals[15],
            "require_prior_5bar_down": vals[16],
            "require_prior_10bar_down": vals[17],
            "require_rsi_below": vals[18],
            "require_volume_z_above": vals[19],
            "cash_session_only": vals[20],
            "one_position_only": True,
            "contracts": args.contracts,
            "starting_capital": args.starting_capital,
            "commission_per_side": args.commission_per_side,
            "slippage_points": args.slippage_points,
        }
        if valid_combo(cfg):
            grid.append(cfg)

    return grid


def score_results(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[out["trades"] > 0].copy()
    out["score"] = (
        out["profit_factor"].fillna(0) * 100.0
        + out["expectancy_r"].fillna(0) * 50.0
        + out["net_profit"].fillna(0) / 100.0
        - out["dd_pct_of_start"].fillna(0) * 40.0
    )
    return out.sort_values(
        ["score", "profit_factor", "expectancy_r", "net_profit"],
        ascending=[False, False, False, False],
    )


def write_readme(outdir: Path, total_runs: int, kept_runs: int):
    txt = f"""MNQ Hammer Optimizer via mt5linux

This run tested {total_runs} parameter combinations and retained {kept_runs} rows in the summary output.

Files:
- optimizer_results_all.csv: all tested configs with metrics
- optimizer_results_filtered.csv: configs with min trade count and basic sanity filtering
- optimizer_top_ranked.csv: top-ranked configurations

Ranking notes:
- Higher profit factor is better.
- Higher expectancy_r is better.
- Higher net_profit is better.
- Lower drawdown is better.
- Use out-of-sample validation before trusting the top rows.
"""
    (outdir / "README.txt").write_text(txt)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ensure_mt5(args.mt5_host, args.mt5_port)

    try:
        print(f"[INFO] Connected to mt5linux server at {args.mt5_host}:{args.mt5_port}")

        ensure_symbol(args.symbol)

        timeframes = parse_csv_list(args.timeframes, str)
        data_map = {}

        for tf in timeframes:
            print(f"[INFO] Loading {args.symbol} {tf}")
            df = fetch_bars(args.symbol, tf, args.date_from, args.date_to)
            data_map[tf] = add_features(df, args.session_tz, args.cash_start, args.cash_end)
            print(f"[INFO] Loaded {len(data_map[tf])} bars for {args.symbol} {tf}")

        grid = build_grid(args)
        print(f"[INFO] Testing {len(grid)} combinations")
        results = []

        for idx, cfg in enumerate(grid, start=1):
            df = data_map[cfg["timeframe"]]
            trades = run_backtest(df, cfg)
            summary = summarize_trades(trades, cfg)
            results.append(summary)

            if idx % 100 == 0 or idx == len(grid):
                print(f"[INFO] Completed {idx}/{len(grid)}")

        all_results = pd.DataFrame(results)
        all_results.to_csv(outdir / "optimizer_results_all.csv", index=False)

        filtered = all_results.copy()
        filtered = filtered[
            (filtered["trades"] >= args.min_trades)
            & (filtered["profit_factor"].fillna(0) >= 1.0)
            & (filtered["dd_pct_of_start"].fillna(np.inf) <= 1.5)
        ].copy()

        filtered = score_results(filtered)
        filtered.to_csv(outdir / "optimizer_results_filtered.csv", index=False)

        top = filtered.head(args.top_n).copy()
        top.to_csv(outdir / "optimizer_top_ranked.csv", index=False)

        write_readme(outdir, len(grid), len(filtered))

        print("[INFO] Optimization complete")
        if not top.empty:
            cols = [
                "timeframe", "hammer_spec", "entry_mode", "stop_mode", "target_mode",
                "profit_factor", "expectancy_r", "net_profit", "max_drawdown", "trades", "score"
            ]
            print(top[cols].head(20).to_string(index=False))
        else:
            print("[INFO] No configurations passed the filter thresholds.")

        print(f"[INFO] Output directory: {outdir.resolve()}")

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
