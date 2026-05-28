from mt5linux import MetaTrader5
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timezone
import pandas as pd
import numpy as np


# ---------------- CONFIG ---------------- #
SYMBOL = "@MNQ"
START_DATE = datetime(2019, 5, 3, tzinfo=timezone.utc)

ROLL_VOL_PERIOD = 10
N_COMPONENTS = 2

# Live-style fixed risk assumptions
STOP_LOSS_POINTS = 200.0
USD_PER_POINT = 2.0
CONTRACTS = 1
STARTING_CAPITAL_USD = 10000.0
RISK_FREE_RATE = 0.0

RSI_PERIOD = 7

TIMEFRAMES = [
    ("M5", 252 * 24 * 12),
    ("M30", 252 * 24 * 2),
    ("H1", 252 * 24),
    ("H2", 252 * 12),
    ("H4", 252 * 6),
]

RSI_GATES = [
    (55, 45),
    (60, 40),
    (65, 35),
    (70, 30),
]
# ---------------------------------------- #


def get_timeframe(mt5, label: str):
    label = label.upper()
    if label == "M5":
        return mt5.TIMEFRAME_M5
    elif label == "M30":
        return mt5.TIMEFRAME_M30
    elif label == "H1":
        return mt5.TIMEFRAME_H1
    elif label == "H2":
        return mt5.TIMEFRAME_H2
    elif label == "H4":
        return mt5.TIMEFRAME_H4
    elif label == "D1":
        return mt5.TIMEFRAME_D1
    else:
        raise ValueError(f"Unsupported timeframe: {label}")


def fetch_bars(symbol: str, start: datetime, tf_label: str) -> pd.DataFrame:
    mt5 = MetaTrader5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    timeframe = get_timeframe(mt5, tf_label)
    end = datetime.now(timezone.utc)

    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"No rates returned for {symbol} on {tf_label}, error={err}")

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

    avg_gain = avg_gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1 / period, adjust=False).mean()

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

    tmp = features.copy()
    tmp["state"] = hidden_states

    state_means = tmp.groupby("state")["log_ret"].mean()
    bull_state = state_means.idxmax()
    bear_state = state_means.idxmin()

    tmp["regime"] = np.where(tmp["state"] == bull_state, "bull", "bear")
    return tmp["regime"]


def close_trade(position, exit_time, exit_price, exit_reason):
    position["exit_time"] = exit_time
    position["exit_price"] = float(exit_price)
    position["exit_reason"] = exit_reason
    return position


def backtest_long_short_with_stop(
    data: pd.DataFrame,
    regime: pd.Series,
    rsi_long_threshold: float,
    rsi_short_threshold: float,
) -> pd.DataFrame:
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

        # Manage fixed 200-point stop
        if position is not None:
            if position["side"] == "long":
                if low_price <= position["stop_price"]:
                    trades.append(close_trade(position, t, position["stop_price"], "stop_loss"))
                    position = None
            elif position["side"] == "short":
                if high_price >= position["stop_price"]:
                    trades.append(close_trade(position, t, position["stop_price"], "stop_loss"))
                    position = None

        # Regime-flip entries/exits with RSI filter
        if regime_shift.loc[t]:
            if reg == "bull":
                if position is not None and position["side"] == "short":
                    trades.append(close_trade(position, t, open_price, "regime_flip"))
                    position = None

                if position is None and rsi > rsi_long_threshold:
                    position = {
                        "side": "long",
                        "entry_time": t,
                        "entry_price": open_price,
                        "stop_price": open_price - STOP_LOSS_POINTS,
                    }

            elif reg == "bear":
                if position is not None and position["side"] == "long":
                    trades.append(close_trade(position, t, open_price, "regime_flip"))
                    position = None

                if position is None and rsi < rsi_short_threshold:
                    position = {
                        "side": "short",
                        "entry_time": t,
                        "entry_price": open_price,
                        "stop_price": open_price + STOP_LOSS_POINTS,
                    }

    if position is not None:
        last_time = d.index[-1]
        last_close = float(d["close"].iloc[-1])
        trades.append(close_trade(position, last_time, last_close, "final_close"))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df

    direction = trades_df["side"].map({"long": 1, "short": -1})
    trades_df["pnl_points"] = (trades_df["exit_price"] - trades_df["entry_price"]) * direction
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


def compute_metrics(trades_df: pd.DataFrame, annual_factor: float) -> dict:
    if trades_df.empty:
        return {}

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

    total_return_pct = ((equity.iloc[-1] / STARTING_CAPITAL_USD) - 1.0) * 100
    ending_equity_usd = equity.iloc[-1]

    return {
        "starting_capital_usd": STARTING_CAPITAL_USD,
        "ending_equity_usd": ending_equity_usd,
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
        "trades": len(trades_df),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "stopped_out_trades": (trades_df["exit_reason"] == "stop_loss").sum(),
        "regime_flip_exits": (trades_df["exit_reason"] == "regime_flip").sum(),
        "avg_duration_hours": trades_df["duration_hours"].mean(),
    }


def main():
    all_results = []

    for tf_label, annual_factor in TIMEFRAMES:
        print(f"\n=== Timeframe: {tf_label} ===")
        try:
            bars = fetch_bars(SYMBOL, START_DATE, tf_label)
        except Exception as e:
            print(f"Failed to fetch {tf_label}: {e}")
            continue

        print(f"Got {len(bars)} bars for {tf_label}. Building features...")
        features = build_features(bars)
        if features.empty:
            print(f"No usable features for {tf_label}.")
            continue

        print("Fitting HMM...")
        regime_series = fit_hmm(features)

        data = bars.join(features[["log_ret", "rv_10", "rsi"]], how="left")
        data = data.join(regime_series.rename("regime"), how="left")

        for rsi_long, rsi_short in RSI_GATES:
            print(f"Testing TF={tf_label}, RSI long>{rsi_long}, short<{rsi_short}")

            trades_df = backtest_long_short_with_stop(
                data=data,
                regime=data["regime"],
                rsi_long_threshold=rsi_long,
                rsi_short_threshold=rsi_short,
            )

            metrics = compute_metrics(trades_df, annual_factor)
            if not metrics:
                print("No trades for this combo.")
                continue

            metrics["tf"] = tf_label
            metrics["rsi_long"] = rsi_long
            metrics["rsi_short"] = rsi_short
            all_results.append(metrics)

            print(
                f"Start={metrics['starting_capital_usd']:.2f}, "
                f"End={metrics['ending_equity_usd']:.2f}, "
                f"Return={metrics['total_return_pct']:.2f}%, "
                f"PF={metrics['profit_factor']:.3f}, "
                f"DD={metrics['max_drawdown_pct']:.2f}%"
            )

    if not all_results:
        print("\nNo results generated.")
        return

    df_res = pd.DataFrame(all_results)

    df_sorted = df_res.sort_values(
        by=["profit_factor", "recovery_factor", "total_return_pct"],
        ascending=False
    )

    print("\n=== Top configurations by PF / Recovery / Return ===")
    print(
        df_sorted[
            [
                "tf",
                "rsi_long",
                "rsi_short",
                "starting_capital_usd",
                "ending_equity_usd",
                "net_profit_usd",
                "total_return_pct",
                "trades",
                "profit_factor",
                "recovery_factor",
                "max_drawdown_pct",
                "expectancy_usd",
            ]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
