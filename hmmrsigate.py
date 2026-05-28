from mt5linux import MetaTrader5
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timezone
import pandas as pd
import numpy as np


# ---------------- CONFIG ---------------- #
SYMBOL = "MNQM26"  # adjust if needed
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)

# choose "D1" or "H1"
MODE = "H1"  # "D1" for daily, "H1" for hourly

ROLL_VOL_PERIOD = 10
N_COMPONENTS = 2

STOP_LOSS_POINTS = 60
USD_PER_POINT = 2.0
CONTRACTS = 1
STARTING_CAPITAL_USD = 5000
RISK_FREE_RATE = 0.0

RSI_PERIOD = 10
RSI_LONG_THRESHOLD = 30
RSI_SHORT_THRESHOLD = 60.0
# ---------------------------------------- #


def get_timeframe(mt5, mode: str):
    if mode == "D1":
        return mt5.TIMEFRAME_D1
    elif mode == "H1":
        return mt5.TIMEFRAME_H1
    else:
        raise ValueError("MODE must be 'D1' or 'H1'")


def get_annualization_factor(mode: str):
    if mode == "D1":
        return 252
    elif mode == "H1":
        return 252 * 24
    else:
        return 252


def fetch_bars(symbol: str, start: datetime, mode: str) -> pd.DataFrame:
    mt5 = MetaTrader5()
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

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Wilder smoothing
    avg_gain = avg_gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["log_ret"] = np.log(f["close"]).diff()
    f["rv_10"] = f["log_ret"].rolling(ROLL_VOL_PERIOD).std()
    f["rsi"] = calc_rsi(f["close"], RSI_PERIOD)
    return f.dropna(subset=["log_ret", "rv_10", "rsi"])


def fit_hmm(features: pd.DataFrame) -> pd.Series:
    X = features[["log_ret", "rv_10"]].values
    model = GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)
    hidden_states = model.predict(X)

    features = features.copy()
    features["state"] = hidden_states

    state_means = features.groupby("state")["log_ret"].mean()
    bull_state = state_means.idxmax()
    bear_state = state_means.idxmin()

    features["regime"] = np.where(
        features["state"] == bull_state, "bull", "bear"
    )
    return features["regime"]


def close_trade(position, exit_time, exit_price, exit_reason):
    position["exit_time"] = exit_time
    position["exit_price"] = float(exit_price)
    position["exit_reason"] = exit_reason
    return position


def backtest_long_short_with_stop(data: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    d = data.copy()
    d["regime"] = regime
    d["regime"] = d["regime"].ffill()
    d = d.dropna(subset=["regime", "rsi"])

    signals = d["regime"]
    rsi_series = d["rsi"]
    regime_shift = signals.ne(signals.shift(1))

    trades = []
    position = None

    for t, row in d.iterrows():
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        reg = signals.loc[t]
        rsi = rsi_series.loc[t]

        # manage existing position: stop loss
        if position is not None:
            if position["side"] == "long":
                stop_price = position["stop_price"]
                if low_price <= stop_price:
                    trades.append(close_trade(position, t, stop_price, "stop_loss"))
                    position = None

            elif position["side"] == "short":
                stop_price = position["stop_price"]
                if high_price >= stop_price:
                    trades.append(close_trade(position, t, stop_price, "stop_loss"))
                    position = None

        # regime flip logic + RSI gate for new entries
        if regime_shift.loc[t]:
            if reg == "bull":
                # close short if any
                if position is not None and position["side"] == "short":
                    trades.append(close_trade(position, t, open_price, "regime_flip"))
                    position = None

                # open long only if RSI > 50
                if position is None and rsi > RSI_LONG_THRESHOLD:
                    position = {
                        "side": "long",
                        "entry_time": t,
                        "entry_price": open_price,
                        "stop_price": open_price - STOP_LOSS_POINTS,
                    }

            elif reg == "bear":
                # close long if any
                if position is not None and position["side"] == "long":
                    trades.append(close_trade(position, t, open_price, "regime_flip"))
                    position = None

                # open short only if RSI < 50
                if position is None and rsi < RSI_SHORT_THRESHOLD:
                    position = {
                        "side": "short",
                        "entry_time": t,
                        "entry_price": open_price,
                        "stop_price": open_price + STOP_LOSS_POINTS,
                    }

    # close any open position at the end
    if position is not None:
        last_time = d.index[-1]
        last_close = float(d["close"].iloc[-1])
        trades.append(close_trade(position, last_time, last_close, "final_close"))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    direction = trades_df["side"].map({"long": 1, "short": -1})
    trades_df["pnl_points"] = (
        trades_df["exit_price"] - trades_df["entry_price"]
    ) * direction

    trades_df["pnl_usd"] = trades_df["pnl_points"] * USD_PER_POINT * CONTRACTS
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


def compute_metrics(trades_df: pd.DataFrame, mode: str) -> dict:
    if trades_df.empty:
        return {}

    annual_factor = get_annualization_factor(mode)

    wins = trades_df[trades_df["pnl_usd"] > 0]
    losses = trades_df[trades_df["pnl_usd"] < 0]
    longs = trades_df[trades_df["side"] == "long"]
    shorts = trades_df[trades_df["side"] == "short"]

    gross_profit = wins["pnl_usd"].sum()
    gross_loss = losses["pnl_usd"].sum()
    net_profit = trades_df["pnl_usd"].sum()

    profit_factor = np.nan
    if gross_loss != 0:
        profit_factor = abs(gross_profit / gross_loss)

    win_rate = (trades_df["pnl_usd"] > 0).mean() * 100
    long_win_rate = (longs["pnl_usd"] > 0).mean() * 100 if len(longs) > 0 else np.nan
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
    regime_flip_exits = (trades_df["exit_reason"] == "regime_flip").sum()

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
        "long_win_rate_pct": long_win_rate,
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
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "stopped_out_trades": stopped_out,
        "regime_flip_exits": regime_flip_exits,
        "avg_duration_hours": avg_duration_hours,
    }


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def print_trades(trades_df: pd.DataFrame, mode: str):
    if trades_df.empty:
        print("No trades generated.")
        return

    tf_label = "Daily" if mode == "D1" else "H1"
    print(f"\n--- {tf_label} Regime Trades for {SYMBOL} ---")
    print(f"Starting capital: {STARTING_CAPITAL_USD:.2f} USD")
    print(f"Contracts: {CONTRACTS}")
    print(f"USD per point: {USD_PER_POINT:.2f}")
    print(f"Stop loss: {STOP_LOSS_POINTS:.2f} points / {STOP_LOSS_POINTS * USD_PER_POINT * CONTRACTS:.2f} USD\n")

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
    print(f"{g}Long trades: {metrics['long_trades']}{e}")
    print(f"{g}Short trades: {metrics['short_trades']}{e}")
    print(f"{g}Stopped out trades: {metrics['stopped_out_trades']}{e}")
    print(f"{g}Regime-flip exits: {metrics['regime_flip_exits']}{e}")
    print(f"{g}Win rate: {fmt(metrics['win_rate_pct'])} %{e}")
    print(f"{g}Long win rate: {fmt(metrics['long_win_rate_pct'])} %{e}")
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
    print(f"Fetching {MODE} bars for {SYMBOL} from {START_DATE.date()} ...")
    bars = fetch_bars(SYMBOL, START_DATE, MODE)

    print(f"Got {len(bars)} bars.")
    features = build_features(bars)

    print("Fitting HMM for regime detection ...")
    regime_series = fit_hmm(features)

    data = bars.join(features[["log_ret", "rv_10", "rsi"]], how="left")
    data = data.join(regime_series.rename("regime"), how="left")

    trades_df = backtest_long_short_with_stop(data, data["regime"])
    print_trades(trades_df, MODE)

    metrics = compute_metrics(trades_df, MODE)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
