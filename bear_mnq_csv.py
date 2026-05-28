from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# ======================================
# SETTINGS
# ======================================
CSV_PATH = "mnq_h1.csv"          # export from MT5: H1 MNQ, columns time, open, high, low, close
INITIAL_CAPITAL = 10000.0
RISK_PER_TRADE = 0.005           # 0.5% equity risk
POINT_VALUE = 2.0                # MNQ = $2 / point
SLIPPAGE_POINTS = 4.0
COMMISSION_PER_CONTRACT = 1.50
MAX_CONTRACTS = 10

DAILY_FAST_MA = 20
DAILY_SLOW_MA = 50
H1_FAST_MA = 20
H1_SLOW_MA = 50
ATR_LEN = 14

PULLBACK_LOOKBACK = 10
BREAKDOWN_LOOKBACK = 20
STOP_ATR_MULT = 1.3
PARTIAL_R = 2.0
TRAIL_MA_LEN = 20
TIME_STOP_BARS = 72
MIN_PULLBACK_BARS = 2

OUTPUT_DIR = "output"


# ======================================
# DATA
# ======================================
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    required = ["time", "open", "high", "low", "close"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    out = pd.DataFrame()
    out["time"] = pd.to_datetime(df[cols["time"]], utc=True)
    for c in ["open", "high", "low", "close"]:
        out[c] = pd.to_numeric(df[cols[c]], errors="coerce")

    out = out.dropna().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return out


# ======================================
# INDICATORS
# ======================================
def atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ma_fast"] = df["close"].rolling(H1_FAST_MA).mean()
    df["ma_slow"] = df["close"].rolling(H1_SLOW_MA).mean()
    df["ma_fast_slope"] = df["ma_fast"] - df["ma_fast"].shift(3)
    df["ma_slow_slope"] = df["ma_slow"] - df["ma_slow"].shift(3)
    df["atr"] = atr(df, ATR_LEN)
    df["range"] = df["high"] - df["low"]
    df["trail_ma"] = df["close"].rolling(TRAIL_MA_LEN).mean()

    df["recent_high"] = df["high"].shift(1).rolling(PULLBACK_LOOKBACK).max()
    df["recent_low"] = df["low"].shift(1).rolling(BREAKDOWN_LOOKBACK).min()
    df["up_bar"] = (df["close"] > df["open"]).astype(int)
    df["recent_up_bars"] = df["up_bar"].shift(1).rolling(MIN_PULLBACK_BARS).sum()

    daily = (
        df.set_index("time")[["open", "high", "low", "close"]]
        .resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )

    daily["d_ma_fast"] = daily["close"].rolling(DAILY_FAST_MA).mean()
    daily["d_ma_slow"] = daily["close"].rolling(DAILY_SLOW_MA).mean()
    daily["d_fast_slope"] = daily["d_ma_fast"] - daily["d_ma_fast"].shift(3)
    daily["d_slow_slope"] = daily["d_ma_slow"] - daily["d_ma_slow"].shift(3)

    daily["bear_regime"] = (
        (daily["close"] < daily["d_ma_fast"]) &
        (daily["d_ma_fast"] < daily["d_ma_slow"]) &
        (daily["d_fast_slope"] < 0) &
        (daily["d_slow_slope"] < 0)
    ).astype(int)

    daily["date"] = daily["time"].dt.floor("D")
    df["date"] = df["time"].dt.floor("D")
    df = df.merge(daily[["date", "bear_regime"]], on="date", how="left")

    df["h1_bear"] = (
        (df["close"] < df["ma_fast"]) &
        (df["ma_fast"] < df["ma_slow"]) &
        (df["ma_fast_slope"] < 0)
    )

    df["short_signal"] = (
        (df["bear_regime"] == 1) &
        (df["h1_bear"]) &
        (df["recent_up_bars"] >= MIN_PULLBACK_BARS) &
        (df["close"] < df["recent_low"]) &
        (df["range"] > 0.8 * df["atr"])
    )

    return df


# ======================================
# BACKTEST
# ======================================
@dataclass
class Position:
    side: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    qty: int
    bars_held: int = 0
    partial_taken: bool = False


def contract_qty(equity: float, entry_price: float, stop_price: float) -> int:
    risk_dollars = equity * RISK_PER_TRADE
    risk_points = abs(stop_price - entry_price)
    if risk_points <= 0:
        return 0
    dollars_per_contract = risk_points * POINT_VALUE
    qty = int(risk_dollars // dollars_per_contract)
    return max(0, min(qty, MAX_CONTRACTS))


def pnl_dollars(side: int, entry: float, exit_: float, qty: int) -> float:
    gross = (exit_ - entry) * side * POINT_VALUE * qty
    costs = (SLIPPAGE_POINTS * POINT_VALUE * qty) + (2 * COMMISSION_PER_CONTRACT * qty)
    return gross - costs


def run_backtest(df: pd.DataFrame):
    trades: List[Dict] = []
    equity_curve: List[Dict] = []
    equity = INITIAL_CAPITAL
    pos: Optional[Position] = None

    start_idx = max(DAILY_SLOW_MA * 24, 200)

    for i in range(start_idx, len(df) - 1):
        row = df.iloc[i]
        next_bar = df.iloc[i + 1]

        equity_curve.append({"time": row["time"], "equity": equity})

        if pos is not None:
            pos.bars_held += 1
            exit_reason = None
            exit_price = None

            target_price = pos.entry_price - PARTIAL_R * (pos.stop_price - pos.entry_price)

            if row["high"] >= pos.stop_price:
                exit_price = pos.stop_price
                exit_reason = "stop"

            elif (not pos.partial_taken) and row["low"] <= target_price:
                partial_qty = max(1, pos.qty // 2)
                realized = pnl_dollars(pos.side, pos.entry_price, target_price, partial_qty)
                equity += realized

                trades.append({
                    "side": "SHORT_PARTIAL",
                    "entry_time": pos.entry_time,
                    "exit_time": row["time"],
                    "entry_price": pos.entry_price,
                    "exit_price": target_price,
                    "qty": partial_qty,
                    "reason": "partial_2R",
                    "pnl": realized
                })

                pos.qty -= partial_qty
                pos.partial_taken = True
                pos.stop_price = pos.entry_price

                if pos.qty <= 0:
                    pos = None

            elif row["close"] > row["trail_ma"]:
                exit_price = row["close"]
                exit_reason = "trail_ma"

            elif pos.bars_held >= TIME_STOP_BARS:
                exit_price = row["close"]
                exit_reason = "time_stop"

            if pos is not None and exit_reason is not None and pos.qty > 0:
                realized = pnl_dollars(pos.side, pos.entry_price, exit_price, pos.qty)
                equity += realized

                trades.append({
                    "side": "SHORT",
                    "entry_time": pos.entry_time,
                    "exit_time": row["time"],
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "qty": pos.qty,
                    "reason": exit_reason,
                    "pnl": realized
                })

                pos = None

        if pos is None and bool(row["short_signal"]):
            entry = next_bar["open"]
            stop = max(row["recent_high"], entry + STOP_ATR_MULT * row["atr"])
            qty = contract_qty(equity, entry, stop)

            if qty > 0 and stop > entry:
                pos = Position(
                    side=-1,
                    entry_time=next_bar["time"],
                    entry_price=entry,
                    stop_price=stop,
                    qty=qty,
                )

    if pos is not None and len(df) > 0 and pos.qty > 0:
        last_row = df.iloc[-1]
        realized = pnl_dollars(pos.side, pos.entry_price, last_row["close"], pos.qty)
        equity += realized

        trades.append({
            "side": "SHORT",
            "entry_time": pos.entry_time,
            "exit_time": last_row["time"],
            "entry_price": pos.entry_price,
            "exit_price": last_row["close"],
            "qty": pos.qty,
            "reason": "final_bar_close",
            "pnl": realized
        })

        equity_curve.append({"time": last_row["time"], "equity": equity})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    return trades_df, equity_df


# ======================================
# REPORTING
# ======================================
def summarize(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame([{
            "initial_capital": INITIAL_CAPITAL,
            "final_equity": INITIAL_CAPITAL,
            "net_pnl": 0.0,
            "trades": 0,
            "win_rate": np.nan,
            "avg_pnl": np.nan,
            "profit_factor": np.nan,
            "max_drawdown": np.nan,
        }])

    closed = trades_df[trades_df["side"] == "SHORT"]

    wins = closed.loc[closed["pnl"] > 0, "pnl"].sum()
    losses = -closed.loc[closed["pnl"] < 0, "pnl"].sum()
    profit_factor = wins / losses if losses > 0 else np.nan
    win_rate = (closed["pnl"] > 0).mean() if len(closed) else np.nan

    eq = equity_df.copy()
    if not eq.empty:
        eq["peak"] = eq["equity"].cummax()
        eq["dd"] = eq["equity"] - eq["peak"]
        max_dd = eq["dd"].min()
        final_equity = eq["equity"].iloc[-1]
    else:
        max_dd = np.nan
        final_equity = INITIAL_CAPITAL

    return pd.DataFrame([{
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": final_equity,
        "net_pnl": final_equity - INITIAL_CAPITAL,
        "trades": int(len(closed)),
        "win_rate": win_rate,
        "avg_pnl": closed["pnl"].mean() if len(closed) else np.nan,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
    }])


def print_trade_log(trades_df: pd.DataFrame):
    if trades_df.empty:
        print("\nNo trades were taken.")
        return

    t = trades_df.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
    t["exit_time"] = pd.to_datetime(t["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
    t["entry_price"] = t["entry_price"].map(lambda x: f"{x:.2f}")
    t["exit_price"] = t["exit_price"].map(lambda x: f"{x:.2f}")
    t["pnl"] = t["pnl"].map(lambda x: f"{x:.2f}")

    cols = ["side", "entry_time", "exit_time", "entry_price", "exit_price", "qty", "reason", "pnl"]

    print("\nTRADE LOG")
    print(t[cols].to_string(index=False))


def main():
    df = load_csv(CSV_PATH)
    df = add_features(df)
    trades_df, equity_df = run_backtest(df)
    summary_df = summarize(trades_df, equity_df)

    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    trades_df.to_csv(f"{OUTPUT_DIR}/bear_market_csv_trades.csv", index=False)
    equity_df.to_csv(f"{OUTPUT_DIR}/bear_market_csv_equity.csv", index=False)
    summary_df.to_csv(f"{OUTPUT_DIR}/bear_market_csv_summary.csv", index=False)

    print("\nBEAR MARKET CSV STRATEGY SUMMARY\n")
    print(summary_df.to_string(index=False))
    print_trade_log(trades_df)

    print(f"\nTrades saved to {OUTPUT_DIR}/bear_market_csv_trades.csv")
    print(f"Equity saved to {OUTPUT_DIR}/bear_market_csv_equity.csv")
    print(f"Summary saved to {OUTPUT_DIR}/bear_market_csv_summary.csv")


if __name__ == "__main__":
    main()
