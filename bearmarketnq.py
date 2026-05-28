from mt5linux import MetaTrader5
from datetime import datetime, timezone
import pandas as pd
import numpy as np


# ---------------- CONFIG ---------------- #
SYMBOL = "@MNQ"
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)

STARTING_CAPITAL_USD = 5000
USD_PER_POINT = 2.0
CONTRACTS = 1
RISK_FREE_RATE = 0.0

DAILY_RSI_PERIOD = 14
H1_RSI_PERIOD = 14

DAILY_ANCHOR_RSI = 70.0
H1_ENTRY_CROSS_BELOW = 50.0
DAILY_EXIT_RSI = 30.0

STOP_BUFFER_POINTS = 5.0

USE_DYNAMIC_POSITION_SIZING = False
RISK_PER_TRADE_PCT = 0.01
MAX_CONTRACTS = 10
# ---------------------------------------- #


def get_timeframe(mt5, mode: str):
    if mode == "D1":
        return mt5.TIMEFRAME_D1
    elif mode == "H1":
        return mt5.TIMEFRAME_H1
    else:
        raise ValueError("mode must be 'D1' or 'H1'")


def get_annualization_factor():
    return 252


def fetch_bars(symbol: str, start: datetime, mode: str) -> pd.DataFrame:
    mt5 = MetaTrader5(host="127.0.0.1", port=18812)

    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    timeframe = get_timeframe(mt5, mode)
    end = datetime.now(timezone.utc)

    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"No rates returned for {symbol}, error={err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    mt5.shutdown()
    return df


def calc_rsi(prices: pd.Series, period: int) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rsi_d"] = calc_rsi(d["close"], DAILY_RSI_PERIOD)
    d["anchor_day"] = d["rsi_d"] > DAILY_ANCHOR_RSI
    d["date"] = d.index.floor("D")
    return d


def build_h1_features(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["rsi_h1"] = calc_rsi(h["close"], H1_RSI_PERIOD)
    h["rsi_h1_prev"] = h["rsi_h1"].shift(1)
    h["cross_below_50"] = (
        (h["rsi_h1_prev"] >= H1_ENTRY_CROSS_BELOW) &
        (h["rsi_h1"] < H1_ENTRY_CROSS_BELOW)
    )
    h["date"] = h.index.floor("D")
    return h


def close_trade(position, exit_time, exit_price, exit_reason):
    position["exit_time"] = exit_time
    position["exit_price"] = float(exit_price)
    position["exit_reason"] = exit_reason
    return position


def calc_contracts(equity_usd: float, entry_price: float, stop_price: float) -> int:
    if not USE_DYNAMIC_POSITION_SIZING:
        return CONTRACTS

    risk_budget = equity_usd * RISK_PER_TRADE_PCT
    risk_points = abs(stop_price - entry_price)

    if risk_points <= 0:
        return 0

    risk_usd_per_contract = risk_points * USD_PER_POINT
    contracts = int(risk_budget // risk_usd_per_contract)
    contracts = max(1, min(contracts, MAX_CONTRACTS))
    return contracts


def merge_timeframes(h1: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    d = daily[["date", "rsi_d", "anchor_day", "high"]].copy()
    d = d.rename(columns={"high": "daily_anchor_high"})

    merged = h1.merge(d, on="date", how="left")
    merged.index = h1.index
    return merged


def backtest_same_day_rsi_break_short(data: pd.DataFrame) -> pd.DataFrame:
    d = data.copy()
    d = d.dropna(subset=["rsi_h1", "rsi_d"])

    trades = []
    position = None

    for t, row in d.iterrows():
        open_price = float(row["open"])
        high_price = float(row["high"])
        close_price = float(row["close"])
        rsi_d = float(row["rsi_d"])
        anchor_day = bool(row["anchor_day"])
        cross_below_50 = bool(row["cross_below_50"])
        anchor_high = float(row["daily_anchor_high"]) if not pd.isna(row["daily_anchor_high"]) else np.nan

        # manage open position
        if position is not None:
            if high_price >= position["stop_price"]:
                trades.append(close_trade(position, t, position["stop_price"], "stop_loss"))
                position = None
            elif rsi_d < DAILY_EXIT_RSI:
                trades.append(close_trade(position, t, close_price, "daily_rsi_below_30"))
                position = None

        # same-day entry:
        # if the daily candle is an anchor day (daily RSI > 70),
        # and within that same calendar day H1 RSI crosses below 50, short it
        if position is None:
            if anchor_day and cross_below_50 and not np.isnan(anchor_high):
                stop_price = anchor_high + STOP_BUFFER_POINTS

                if stop_price > open_price:
                    qty = calc_contracts(STARTING_CAPITAL_USD, open_price, stop_price)

                    position = {
                        "side": "short",
                        "entry_time": t,
                        "entry_price": open_price,
                        "stop_price": stop_price,
                        "anchor_date": row["date"],
                        "anchor_rsi_d": rsi_d,
                        "qty": qty,
                    }

    if position is not None:
        last_time = d.index[-1]
        last_close = float(d["close"].iloc[-1])
        trades.append(close_trade(position, last_time, last_close, "final_close"))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    direction = trades_df["side"].map({"short": -1})
    trades_df["pnl_points"] = (trades_df["exit_price"] - trades_df["entry_price"]) * direction
    trades_df["pnl_usd"] = trades_df["pnl_points"] * USD_PER_POINT * trades_df["qty"]
    trades_df["cum_pnl_usd"] = trades_df["pnl_usd"].cumsum()
    trades_df["equity_usd"] = STARTING_CAPITAL_USD + trades_df["cum_pnl_usd"]
    trades_df["trade_return"] = trades_df["pnl_usd"] / STARTING_CAPITAL_USD
    trades_df["duration_hours"] = (
        (trades_df["exit_time"] - trades_df["entry_time"]).dt.total_seconds() / 3600.0
    )

    return trades_df


def max_consecutive(condition_series):
    max_streak = 0
    current = 0
    for val in condition_series:
        if val:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if trades_df.empty:
        return {}

    annual_factor = get_annualization_factor()

    wins = trades_df[trades_df["pnl_usd"] > 0]
    losses = trades_df[trades_df["pnl_usd"] < 0]
    shorts = trades_df[trades_df["side"] == "short"]

    gross_profit = wins["pnl_usd"].sum()
    gross_loss = losses["pnl_usd"].sum()
    net_profit = trades_df["pnl_usd"].sum()

    profit_factor = np.nan
    if gross_loss != 0:
        profit_factor = abs(gross_profit / gross_loss)

    win_rate = (trades_df["pnl_usd"] > 0).mean() * 100
    short_win_rate = (shorts["pnl_usd"] > 0).mean() * 100 if len(shorts) > 0 else np.nan

    avg_win_usd = wins["pnl_usd"].mean() if len(wins) > 0 else 0.0
    avg_loss_usd = losses["pnl_usd"].mean() if len(losses) > 0 else 0.0
    avg_win_points = wins["pnl_points"].mean() if len(wins) > 0 else 0.0
    avg_loss_points = losses["pnl_points"].mean() if len(losses) > 0 else 0.0

    avg_win_loss_ratio = np.nan
    if avg_loss_usd != 0:
        avg_win_loss_ratio = abs(avg_win_usd / avg_loss_usd)

    expectancy_usd = trades_df["pnl_usd"].mean()
    expectancy_points = trades_df["pnl_points"].mean()

    equity = trades_df["equity_usd"]
    running_max = equity.cummax()
    drawdown_usd = equity - running_max
    drawdown_pct = drawdown_usd / running_max
    max_drawdown_usd = drawdown_usd.min()
    max_drawdown_pct = drawdown_pct.min() * 100

    recovery_factor = np.nan
    if max_drawdown_usd != 0:
        recovery_factor = abs(net_profit / max_drawdown_usd)

    returns = trades_df["trade_return"]
    ret_mean = returns.mean()
    ret_std = returns.std(ddof=1)

    sharpe_ratio = np.nan
    if ret_std and ret_std > 0:
        sharpe_ratio = np.sqrt(annual_factor) * ((ret_mean - RISK_FREE_RATE / annual_factor) / ret_std)

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan

    sortino_ratio = np.nan
    if downside_std is not None and not np.isnan(downside_std) and downside_std > 0:
        sortino_ratio = np.sqrt(annual_factor) * ((ret_mean - RISK_FREE_RATE / annual_factor) / downside_std)

    volatility_pct = ret_std * np.sqrt(annual_factor) * 100 if ret_std and ret_std > 0 else np.nan

    consec_wins = max_consecutive(trades_df["pnl_usd"] > 0)
    consec_losses = max_consecutive(trades_df["pnl_usd"] < 0)

    best_trade_usd = trades_df["pnl_usd"].max()
    worst_trade_usd = trades_df["pnl_usd"].min()
    best_trade_points = trades_df["pnl_points"].max()
    worst_trade_points = trades_df["pnl_points"].min()

    total_return_pct = ((equity.iloc[-1] / STARTING_CAPITAL_USD) - 1.0) * 100
    avg_duration_hours = trades_df["duration_hours"].mean()

    stopped_out = (trades_df["exit_reason"] == "stop_loss").sum()
    rsi_exits = (trades_df["exit_reason"] == "daily_rsi_below_30").sum()

    return {
        "starting_capital_usd": STARTING_CAPITAL_USD,
        "final_equity_usd": equity.iloc[-1],
        "net_profit_usd": net_profit,
        "net_profit_points": trades_df["pnl_points"].sum(),
        "total_return_pct": total_return_pct,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "profit_factor": profit_factor,
        "win_rate_pct": win_rate,
        "short_win_rate_pct": short_win_rate,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "avg_win_points": avg_win_points,
        "avg_loss_points": avg_loss_points,
        "avg_win_loss_ratio": avg_win_loss_ratio,
        "expectancy_usd": expectancy_usd,
        "expectancy_points": expectancy_points,
        "max_drawdown_usd": max_drawdown_usd,
        "max_drawdown_pct": max_drawdown_pct,
        "recovery_factor": recovery_factor,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "volatility_pct": volatility_pct,
        "consecutive_wins": consec_wins,
        "consecutive_losses": consec_losses,
        "best_trade_usd": best_trade_usd,
        "worst_trade_usd": worst_trade_usd,
        "best_trade_points": best_trade_points,
        "worst_trade_points": worst_trade_points,
        "trades": len(trades_df),
        "short_trades": len(shorts),
        "stopped_out_trades": stopped_out,
        "rsi_exits": rsi_exits,
        "avg_duration_hours": avg_duration_hours,
    }


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def print_trades(trades_df: pd.DataFrame):
    if trades_df.empty:
        print("No trades generated.")
        return

    print(f"\n--- Daily Anchor + H1 RSI Breakdown Shorts for {SYMBOL} ---")
    print(f"Starting capital: {STARTING_CAPITAL_USD:.2f} USD")
    print(f"Contracts: {'dynamic' if USE_DYNAMIC_POSITION_SIZING else CONTRACTS}")
    print(f"USD per point: {USD_PER_POINT:.2f}")
    print(f"Daily anchor RSI: {DAILY_ANCHOR_RSI:.2f}")
    print(f"H1 entry cross below: {H1_ENTRY_CROSS_BELOW:.2f}")
    print(f"Daily exit RSI: {DAILY_EXIT_RSI:.2f}\n")

    for _, tr in trades_df.iterrows():
        pnl_pts = tr["pnl_points"]
        pnl_usd = tr["pnl_usd"]
        equity_usd = tr["equity_usd"]

        color_start = "\033[92m" if pnl_usd >= 0 else "\033[91m"
        color_end = "\033[0m"

        print(
            f"{color_start}"
            f"{tr['side'].upper():5} "
            f"{tr['entry_time']} -> {tr['exit_time']} "
            f"entry={tr['entry_price']:.2f} "
            f"exit={tr['exit_price']:.2f} "
            f"stop={tr['stop_price']:.2f} "
            f"anchor_date={tr['anchor_date']} "
            f"anchor_rsi_d={tr['anchor_rsi_d']:.2f} "
            f"qty={tr['qty']} "
            f"reason={tr['exit_reason']} "
            f"pnl={pnl_pts:.2f} pts / {pnl_usd:.2f} USD "
            f"equity={equity_usd:.2f} USD"
            f"{color_end}"
        )


def print_metrics(metrics: dict):
    if not metrics:
        print("No metrics available.")
        return

    g = "\033[92m"
    e = "\033[0m"

    print(f"\n{g}--- Performance Metrics ---{e}")
    print(f"{g}Starting capital: {fmt(metrics['starting_capital_usd'])} USD{e}")
    print(f"{g}Final equity: {fmt(metrics['final_equity_usd'])} USD{e}")
    print(f"{g}Net profit: {fmt(metrics['net_profit_points'])} pts / {fmt(metrics['net_profit_usd'])} USD{e}")
    print(f"{g}Total return: {fmt(metrics['total_return_pct'])} %{e}")
    print(f"{g}Gross profit: {fmt(metrics['gross_profit_usd'])} USD{e}")
    print(f"{g}Gross loss: {fmt(metrics['gross_loss_usd'])} USD{e}")
    print(f"{g}Profit factor: {fmt(metrics['profit_factor'], 3)}{e}")
    print(f"{g}Sharpe ratio: {fmt(metrics['sharpe_ratio'], 3)}{e}")
    print(f"{g}Sortino ratio: {fmt(metrics['sortino_ratio'], 3)}{e}")
    print(f"{g}Annualized volatility: {fmt(metrics['volatility_pct'])} %{e}")
    print(f"{g}Max drawdown: {fmt(metrics['max_drawdown_usd'])} USD / {fmt(metrics['max_drawdown_pct'])} %{e}")
    print(f"{g}Recovery factor: {fmt(metrics['recovery_factor'], 3)}{e}")
    print(f"{g}Trades: {metrics['trades']}{e}")
    print(f"{g}Short trades: {metrics['short_trades']}{e}")
    print(f"{g}Stopped out trades: {metrics['stopped_out_trades']}{e}")
    print(f"{g}Daily RSI exits: {metrics['rsi_exits']}{e}")
    print(f"{g}Win rate: {fmt(metrics['win_rate_pct'])} %{e}")
    print(f"{g}Short win rate: {fmt(metrics['short_win_rate_pct'])} %{e}")
    print(f"{g}Average win: {fmt(metrics['avg_win_points'])} pts / {fmt(metrics['avg_win_usd'])} USD{e}")
    print(f"{g}Average loss: {fmt(metrics['avg_loss_points'])} pts / {fmt(metrics['avg_loss_usd'])} USD{e}")
    print(f"{g}Avg win/loss ratio: {fmt(metrics['avg_win_loss_ratio'], 3)}{e}")
    print(f"{g}Expectancy: {fmt(metrics['expectancy_points'])} pts / {fmt(metrics['expectancy_usd'])} USD per trade{e}")
    print(f"{g}Max consecutive wins: {metrics['consecutive_wins']}{e}")
    print(f"{g}Max consecutive losses: {metrics['consecutive_losses']}{e}")
    print(f"{g}Best trade: {fmt(metrics['best_trade_points'])} pts / {fmt(metrics['best_trade_usd'])} USD{e}")
    print(f"{g}Worst trade: {fmt(metrics['worst_trade_points'])} pts / {fmt(metrics['worst_trade_usd'])} USD{e}")
    print(f"{g}Average trade duration: {fmt(metrics['avg_duration_hours'])} hours{e}")


def main():
    print(f"Fetching D1 bars for {SYMBOL} from {START_DATE.date()} ...")
    daily_bars = fetch_bars(SYMBOL, START_DATE, "D1")
    print(f"Got {len(daily_bars)} D1 bars.")

    print(f"Fetching H1 bars for {SYMBOL} from {START_DATE.date()} ...")
    h1_bars = fetch_bars(SYMBOL, START_DATE, "H1")
    print(f"Got {len(h1_bars)} H1 bars.")

    print("Building daily features ...")
    daily = build_daily_features(daily_bars)

    print("Building H1 features ...")
    h1 = build_h1_features(h1_bars)

    print("Merging daily anchor context into H1 bars ...")
    data = merge_timeframes(h1, daily)

    trades_df = backtest_same_day_rsi_break_short(data)
    print_trades(trades_df)

    metrics = compute_metrics(trades_df)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
