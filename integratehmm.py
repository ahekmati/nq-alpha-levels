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

STOP_LOSS_POINTS = 200.0
USD_PER_POINT = 2.0
CONTRACTS = 1          # always 1 contract; code never opens if position != None
STARTING_CAPITAL_USD = 10000.0
RISK_FREE_RATE = 0.0

RSI_PERIOD = 7         # H1 RSI period

# Feature toggles (0 = off, 1 = on)
USE_REGIME_PERSISTENCE = 1     # 1) require regime to persist N bars before we act
USE_HIGHER_TF_RSI_GATE = 0     # 2) use higher TF RSI (D1/H4) as an extra gate
USE_ENTRY_COOLDOWN = 1         # 3) wait X bars after a closed trade before new entry

# Regime persistence
REGIME_PERSIST_BARS = 2        # number of H1 bars new regime must persist

# Higher timeframe RSI gate
HTF_RSI_PERIOD = 7             # RSI period on higher TF
HTF_RSI_TF = "H4"              # "H4" or "D1"
HTF_RSI_LONG_THRESHOLD = 55    # higher TF RSI must be > this to allow long
HTF_RSI_SHORT_THRESHOLD = 45   # higher TF RSI must be < this to allow short

# Entry cooldown (H1 bars)
ENTRY_COOLDOWN_BARS = 2

# RSI gates to test (on H1)
RSI_GATES = [
    (55, 45),
    (60, 40),
    (65, 35),
    (70, 30),
]
# ---------------------------------------- #


def get_timeframe_label_to_mt5(mt5, label: str):
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

    timeframe = get_timeframe_label_to_mt5(mt5, tf_label)
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


def build_persistent_regime(regime_series: pd.Series, persist_bars: int) -> pd.Series:
    """
    Only switch regime after 'persist_bars' consecutive bars
    of the new regime. This reduces chatter.
    """
    if persist_bars <= 1:
        return regime_series.copy()

    reg_raw = regime_series.copy()
    reg_persistent = reg_raw.copy()

    current = reg_raw.iloc[0]
    count = 0
    for i, val in enumerate(reg_raw):
        if val == current:
            count = 0
        else:
            count += 1
            if count >= persist_bars:
                current = val
                count = 0
        reg_persistent.iloc[i] = current

    return reg_persistent


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
    htf_rsi_aligned: pd.DataFrame = None,
    entry_cooldown_bars: int = 0,
) -> pd.DataFrame:
    d = data.copy()
    d["regime"] = regime
    d["regime"] = d["regime"].ffill()
    d = d.dropna(subset=["regime", "rsi"])

    if htf_rsi_aligned is not None:
        d["htf_rsi"] = htf_rsi_aligned["htf_rsi"]
    else:
        d["htf_rsi"] = np.nan

    signals = d["regime"]
    rsi_series = d["rsi"]
    htf_rsi_series = d["htf_rsi"]
    regime_shift = signals.ne(signals.shift(1))

    trades = []
    position = None
    last_exit_idx = -10**9

    for idx, (t, row) in enumerate(d.iterrows()):
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        reg = signals.loc[t]
        rsi = rsi_series.loc[t]
        htf_rsi = htf_rsi_series.loc[t]

        # manage existing position: fixed 200-point stop
        if position is not None:
            if position["side"] == "long":
                if low_price <= position["stop_price"]:
                    trades.append(close_trade(position, t, position["stop_price"], "stop_loss"))
                    position = None
                    last_exit_idx = idx
            elif position["side"] == "short":
                if high_price >= position["stop_price"]:
                    trades.append(close_trade(position, t, position["stop_price"], "stop_loss"))
                    position = None
                    last_exit_idx = idx

        # regime flip logic with RSI + optional HTF gate + cooldown
        if regime_shift.loc[t]:
            # enforce cooldown before any new entry
            can_open_new = True
            if USE_ENTRY_COOLDOWN and entry_cooldown_bars > 0:
                if (idx - last_exit_idx) < entry_cooldown_bars:
                    can_open_new = False

            if reg == "bull":
                # close short on flip
                if position is not None and position["side"] == "short":
                    trades.append(close_trade(position, t, open_price, "regime_flip"))
                    position = None
                    last_exit_idx = idx

                # only open if flat, cooldown satisfied
                if position is None and can_open_new:
                    # local RSI gate
                    if rsi > rsi_long_threshold:
                        if USE_HIGHER_TF_RSI_GATE and htf_rsi_aligned is not None:
                            # require higher TF RSI confirmation
                            if not np.isnan(htf_rsi) and htf_rsi > HTF_RSI_LONG_THRESHOLD:
                                position = {
                                    "side": "long",
                                    "entry_time": t,
                                    "entry_price": open_price,
                                    "stop_price": open_price - STOP_LOSS_POINTS,
                                }
                        else:
                            # no HTF gate
                            position = {
                                "side": "long",
                                "entry_time": t,
                                "entry_price": open_price,
                                "stop_price": open_price - STOP_LOSS_POINTS,
                            }

            elif reg == "bear":
                # close long on flip
                if position is not None and position["side"] == "long":
                    trades.append(close_trade(position, t, open_price, "regime_flip"))
                    position = None
                    last_exit_idx = idx

                if position is None and can_open_new:
                    if rsi < rsi_short_threshold:
                        if USE_HIGHER_TF_RSI_GATE and htf_rsi_aligned is not None:
                            if not np.isnan(htf_rsi) and htf_rsi < HTF_RSI_SHORT_THRESHOLD:
                                position = {
                                    "side": "short",
                                    "entry_time": t,
                                    "entry_price": open_price,
                                    "stop_price": open_price + STOP_LOSS_POINTS,
                                }
                        else:
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
        sharpe_ratio = np.sqrt(annual_factor) * (
            (ret_mean - RISK_FREE_RATE / annual_factor) / ret_std
        )

    downside = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan

    sortino_ratio = np.nan
    if downside_std is not None and not np.isnan(downside_std) and downside_std > 0:
        sortino_ratio = np.sqrt(annual_factor) * (
            (ret_mean - RISK_FREE_RATE / annual_factor) / downside_std
        )

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


def fmt(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def main():
    print(f"Fetching H1 bars for {SYMBOL} from {START_DATE.date()} ...")
    bars = fetch_bars(SYMBOL, START_DATE, "H1")
    print(f"Got {len(bars)} bars.")

    print("Building features ...")
    features = build_features(bars)
    if features.empty:
        print("No usable features on H1.")
        return

    print("Fitting HMM for regime detection ...")
    regime_series_raw = fit_hmm(features)

    if USE_REGIME_PERSISTENCE:
        print(f"Applying regime persistence: {REGIME_PERSIST_BARS} bars")
        regime_series = build_persistent_regime(regime_series_raw, REGIME_PERSIST_BARS)
    else:
        regime_series = regime_series_raw

    # Optional higher TF RSI gate
    htf_rsi_aligned = None
    if USE_HIGHER_TF_RSI_GATE:
        print(f"Fetching {HTF_RSI_TF} bars for higher TF RSI gate ...")
        mt5 = MetaTrader5()
        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
        htf_tf = get_timeframe_label_to_mt5(mt5, HTF_RSI_TF)
        end = datetime.now(timezone.utc)
        htf_rates = mt5.copy_rates_range(SYMBOL, htf_tf, START_DATE, end)
        mt5.shutdown()

        if htf_rates is not None and len(htf_rates) > 0:
            hdf = pd.DataFrame(htf_rates)
            hdf["time"] = pd.to_datetime(hdf["time"], unit="s", utc=True)
            hdf.set_index("time", inplace=True)
            hdf.sort_index(inplace=True)
            hdf["htf_rsi"] = calc_rsi(hdf["close"], HTF_RSI_PERIOD)
            hdf = hdf[["htf_rsi"]].dropna()
            htf_rsi_aligned = hdf.reindex(bars.index, method="ffill")
        else:
            print(f"No {HTF_RSI_TF} data for higher TF RSI gate.")

    data = bars.join(features[["log_ret", "rv_10", "rsi"]], how="left")
    data = data.join(regime_series.rename("regime"), how="left")

    all_results = []
    annual_factor = 252 * 24  # H1 bars per year approx.

    for rsi_long, rsi_short in RSI_GATES:
        print(f"\nTF=H1, RSI long>{rsi_long}, short<{rsi_short}")
        trades_df = backtest_long_short_with_stop(
            data=data,
            regime=data["regime"],
            rsi_long_threshold=rsi_long,
            rsi_short_threshold=rsi_short,
            htf_rsi_aligned=htf_rsi_aligned if USE_HIGHER_TF_RSI_GATE else None,
            entry_cooldown_bars=ENTRY_COOLDOWN_BARS if USE_ENTRY_COOLDOWN else 0,
        )

        metrics = compute_metrics(trades_df, annual_factor)
        if not metrics:
            print("No trades for this combo.")
            continue

        metrics["tf"] = "H1"
        metrics["rsi_long"] = rsi_long
        metrics["rsi_short"] = rsi_short
        all_results.append(metrics)

        print(
            f"Start={metrics['starting_capital_usd']:.2f}, "
            f"End={metrics['ending_equity_usd']:.2f}, "
            f"Ret={metrics['total_return_pct']:.2f}%, "
            f"Trades={metrics['trades']}, "
            f"PF={fmt(metrics['profit_factor'],3)}, "
            f"DD={fmt(metrics['max_drawdown_pct'])}%, "
            f"RF={fmt(metrics['recovery_factor'],3)}"
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
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
