from mt5linux import MetaTrader5
from datetime import datetime, timezone
from itertools import product
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import random

# ---------------- GLOBAL CONFIG ---------------- #
SYMBOL = "MNQM26"
START_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
TIMEFRAME = "H1"

INITIAL_TRAIN_BARS = 1200
RETRAIN_EVERY = 168          # retrain once per week on H1 bars
TEST_HORIZON_BARS = 1

STARTING_CAPITAL_USD = 5000.0
USD_PER_POINT = 2.0
CONTRACTS = 1
RISK_FREE_RATE = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
SAVE_ALL_TRADES = True

# -------- SMALL / FAST TEST GRID -------- #
PARAM_GRID = {
    "seq_len": [24, 48],
    "hidden_size": [32],
    "num_layers": [1],
    "dropout": [0.0],
    "learning_rate": [0.001],
    "epochs": [4],
    "batch_size": [32],
    "feature_set": ["trend", "trend_rsi"],
    "prob_long": [0.55],
    "prob_short": [0.45],
    "stop_loss_points": [200.0],
    "take_profit_points": [2000.0],
}
# ---------------------------------------- #

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
END = "\033[0m"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_timeframe(mt5, mode: str):
    mapping = {
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "M30": mt5.TIMEFRAME_M30,
        "M15": mt5.TIMEFRAME_M15,
    }
    if mode not in mapping:
        raise ValueError(f"Unsupported timeframe: {mode}")
    return mapping[mode]


def get_annualization_factor(mode: str):
    mapping = {
        "M15": 252 * 24 * 4,
        "M30": 252 * 24 * 2,
        "H1": 252 * 24,
        "H4": 252 * 6,
        "D1": 252,
    }
    return mapping.get(mode, 252)


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
        raise RuntimeError(f"No rates returned for {symbol}, timeframe={mode}, error={err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    mt5.shutdown()
    return df


def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()

    f["log_ret_1"] = np.log(f["close"]).diff()
    f["rv_10"] = f["log_ret_1"].rolling(10).std()

    f["ema_10"] = f["close"].ewm(span=10, adjust=False).mean()
    f["ema_20"] = f["close"].ewm(span=20, adjust=False).mean()
    f["ema_50"] = f["close"].ewm(span=50, adjust=False).mean()

    f["trend_10_20"] = (f["ema_10"] - f["ema_20"]) / f["close"]
    f["trend_20_50"] = (f["ema_20"] - f["ema_50"]) / f["close"]
    f["rsi_14"] = calc_rsi(f["close"], 14)

    f["target_up"] = (f["close"].shift(-TEST_HORIZON_BARS) > f["close"]).astype(int)

    needed = [
        "log_ret_1", "rv_10", "trend_10_20", "trend_20_50", "rsi_14", "target_up"
    ]
    return f.dropna(subset=needed).copy()


def get_feature_columns(feature_set: str):
    if feature_set == "trend":
        return ["log_ret_1", "rv_10", "trend_10_20", "trend_20_50"]
    elif feature_set == "trend_rsi":
        return ["log_ret_1", "rv_10", "trend_10_20", "trend_20_50", "rsi_14"]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


def standardize_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame, cols):
    mu = train_df[cols].mean()
    sd = train_df[cols].std(ddof=0).replace(0, np.nan)

    train_z = ((train_df[cols] - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    test_z = ((test_df[cols] - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_z, test_z, mu, sd


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_len]
        y_val = self.y[idx + self.seq_len]
        return (
            torch.tensor(x_seq, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, dropout=0.0):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        logits = self.fc(last_out).squeeze(-1)
        return logits


def train_lstm_model(train_df: pd.DataFrame, cfg: dict):
    feature_cols = get_feature_columns(cfg["feature_set"])
    train_z, _, mu, sd = standardize_train_test(train_df, train_df, feature_cols)

    X_train = train_z.values
    y_train = train_df["target_up"].values.astype(np.float32)

    dataset = SequenceDataset(X_train, y_train, cfg["seq_len"])
    if len(dataset) < 100:
        raise RuntimeError("Not enough sequence samples to train.")

    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = LSTMClassifier(
        input_size=len(feature_cols),
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"]
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(cfg["epochs"]):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

    scaler = {"mu": mu, "sd": sd, "cols": feature_cols}
    return model, scaler


def predict_prob_up(model, seq_df: pd.DataFrame, scaler: dict):
    cols = scaler["cols"]
    mu = scaler["mu"]
    sd = scaler["sd"].replace(0, np.nan)

    z = ((seq_df[cols] - mu) / sd).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = torch.tensor(z.values, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(X)
        prob = torch.sigmoid(logits).item()

    return prob


def walk_forward_predict(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seq_len = cfg["seq_len"]
    out = pd.DataFrame(index=features.index, columns=["prob_up", "signal"], dtype=float)

    model = None
    scaler = None

    for i in range(len(features)):
        if i < INITIAL_TRAIN_BARS:
            continue
        if i < seq_len + 5:
            continue

        need_refit = model is None or ((i - INITIAL_TRAIN_BARS) % RETRAIN_EVERY == 0)

        if need_refit:
            train_df = features.iloc[:i].copy()
            if len(train_df) < INITIAL_TRAIN_BARS:
                continue
            try:
                model, scaler = train_lstm_model(train_df, cfg)
            except Exception:
                continue

        seq_df = features.iloc[i - seq_len:i].copy()
        if len(seq_df) < seq_len:
            continue

        prob_up = predict_prob_up(model, seq_df, scaler)

        signal = 0
        if prob_up >= cfg["prob_long"]:
            signal = 1
        elif prob_up <= cfg["prob_short"]:
            signal = -1

        out.iloc[i] = [prob_up, signal]

    return out


def close_trade(position, exit_time, exit_price, exit_reason):
    position["exit_time"] = exit_time
    position["exit_price"] = float(exit_price)
    position["exit_reason"] = exit_reason
    return position


def backtest_next_bar_execution(data: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = data.copy().dropna(subset=["signal", "prob_up"]).copy()
    if len(d) < 2:
        return pd.DataFrame()

    trades = []
    position = None
    prev_signal = 0

    for i in range(len(d) - 1):
        t = d.index[i]
        next_t = d.index[i + 1]

        row = d.iloc[i]
        next_row = d.iloc[i + 1]

        high_price = float(row["high"])
        low_price = float(row["low"])
        signal = int(row["signal"])
        prob_up = float(row["prob_up"])
        next_open = float(next_row["open"])

        if position is not None:
            if position["side"] == "long":
                if low_price <= position["stop_price"]:
                    trades.append(close_trade(position, t, position["stop_price"], "stop_loss"))
                    position = None
                elif high_price >= position["take_profit"]:
                    trades.append(close_trade(position, t, position["take_profit"], "take_profit"))
                    position = None
                elif signal <= 0:
                    trades.append(close_trade(position, next_t, next_open, "signal_flip"))
                    position = None

            elif position["side"] == "short":
                if high_price >= position["stop_price"]:
                    trades.append(close_trade(position, t, position["stop_price"], "stop_loss"))
                    position = None
                elif low_price <= position["take_profit"]:
                    trades.append(close_trade(position, t, position["take_profit"], "take_profit"))
                    position = None
                elif signal >= 0:
                    trades.append(close_trade(position, next_t, next_open, "signal_flip"))
                    position = None

        signal_shift = signal != prev_signal

        if position is None and signal_shift:
            if signal == 1:
                position = {
                    "side": "long",
                    "entry_time": next_t,
                    "entry_price": next_open,
                    "stop_price": next_open - cfg["stop_loss_points"],
                    "take_profit": next_open + cfg["take_profit_points"],
                    "signal_time": t,
                    "entry_prob_up": prob_up,
                }
            elif signal == -1:
                position = {
                    "side": "short",
                    "entry_time": next_t,
                    "entry_price": next_open,
                    "stop_price": next_open + cfg["stop_loss_points"],
                    "take_profit": next_open - cfg["take_profit_points"],
                    "signal_time": t,
                    "entry_prob_up": prob_up,
                }

        prev_signal = signal

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
    drawdown_pct = drawdown_usd / running_max.replace(0, np.nan)
    max_drawdown_usd = drawdown_usd.min()
    max_drawdown_pct = drawdown_pct.min() * 100

    recovery_factor = np.nan
    if max_drawdown_usd != 0:
        recovery_factor = abs(net_profit / max_drawdown_usd)

    returns = trades_df["trade_return"]
    ret_mean = returns.mean()
    ret_std = returns.std(ddof=1)

    sharpe_ratio = np.nan
    if ret_std is not None and not np.isnan(ret_std) and ret_std > 0:
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

    volatility_pct = (
        ret_std * np.sqrt(annual_factor) * 100
        if ret_std is not None and not np.isnan(ret_std) and ret_std > 0
        else np.nan
    )

    consec_wins = max_consecutive(trades_df["pnl_usd"] > 0)
    consec_losses = max_consecutive(trades_df["pnl_usd"] < 0)

    best_trade_usd = trades_df["pnl_usd"].max()
    worst_trade_usd = trades_df["pnl_usd"].min()
    best_trade_points = trades_df["pnl_points"].max()
    worst_trade_points = trades_df["pnl_points"].min()

    total_return_pct = ((equity.iloc[-1] / STARTING_CAPITAL_USD) - 1.0) * 100
    avg_duration_hours = trades_df["duration_hours"].mean()

    stopped_out = (trades_df["exit_reason"] == "stop_loss").sum()
    signal_flip_exits = (trades_df["exit_reason"] == "signal_flip").sum()
    take_profit_exits = (trades_df["exit_reason"] == "take_profit").sum()

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
        "signal_flip_exits": signal_flip_exits,
        "take_profit_exits": take_profit_exits,
        "avg_duration_hours": avg_duration_hours,
    }


def make_param_grid(param_grid: dict):
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo))


def score_result(row: pd.Series) -> float:
    sharpe = row.get("sharpe_ratio", np.nan)
    profit_factor = row.get("profit_factor", np.nan)
    total_return = row.get("total_return_pct", np.nan)
    max_dd = abs(row.get("max_drawdown_pct", np.nan))
    trades = row.get("trades", 0)

    if np.isnan(sharpe):
        sharpe = -5.0
    if np.isnan(profit_factor):
        profit_factor = 0.0
    if np.isnan(total_return):
        total_return = -999.0
    if np.isnan(max_dd):
        max_dd = 999.0

    trade_penalty = 0.0
    if trades < 10:
        trade_penalty = 2.0
    elif trades < 20:
        trade_penalty = 1.0

    return (
        2.0 * sharpe +
        1.2 * profit_factor +
        0.03 * total_return -
        0.05 * max_dd -
        trade_penalty
    )


def run_experiment(bars: pd.DataFrame, cfg: dict, experiment_id: int):
    features = build_features(bars)
    if len(features) < INITIAL_TRAIN_BARS + cfg["seq_len"] + 50:
        return None, None

    preds = walk_forward_predict(features, cfg)

    data = bars.join(
        features[
            [
                "log_ret_1", "rv_10",
                "trend_10_20", "trend_20_50",
                "rsi_14", "target_up"
            ]
        ],
        how="left"
    )
    data = data.join(preds, how="left")

    trades_df = backtest_next_bar_execution(data, cfg)
    metrics = compute_metrics(trades_df, TIMEFRAME)

    result = {
        "experiment_id": experiment_id,
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        **cfg,
    }

    if metrics:
        result.update(metrics)
        result["score"] = score_result(pd.Series(result))
    else:
        result["score"] = -9999.0
        result["trades"] = 0

    if trades_df is not None and not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["experiment_id"] = experiment_id
        trades_df["timeframe"] = TIMEFRAME
        trades_df["symbol"] = SYMBOL
        trades_df["config_json"] = json.dumps(cfg, sort_keys=True)

    return result, trades_df


def print_top_results(results_df: pd.DataFrame, top_n: int = 10):
    if results_df.empty:
        print(f"{YELLOW}No results to display.{END}")
        return

    cols = [
        "experiment_id", "feature_set", "seq_len", "hidden_size", "epochs",
        "total_return_pct", "net_profit_usd", "profit_factor",
        "sharpe_ratio", "max_drawdown_pct", "trades", "score"
    ]
    view = results_df.sort_values("score", ascending=False).head(top_n)
    print(f"\n{BOLD}{MAGENTA}--- TOP {top_n} FAST LSTM H1 RESULTS ---{END}")
    print(view[cols].to_string(index=False))


def main():
    set_seed(SEED)

    print(f"{BOLD}Using device: {DEVICE}{END}")
    print(f"{BOLD}Fetching {TIMEFRAME} data for {SYMBOL}...{END}")
    bars = fetch_bars(SYMBOL, START_DATE, TIMEFRAME)
    print(f"{GREEN}Got {len(bars)} bars.{END}")

    results = []
    all_trades = []

    experiment_id = 1
    for cfg in make_param_grid(PARAM_GRID):
        print(f"{CYAN}Running fast LSTM experiment {experiment_id} | cfg={cfg}{END}")
        try:
            result, trades_df = run_experiment(bars, cfg, experiment_id)
            if result is not None:
                results.append(result)
            if SAVE_ALL_TRADES and trades_df is not None and not trades_df.empty:
                all_trades.append(trades_df)
        except Exception as e:
            results.append({
                "experiment_id": experiment_id,
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                **cfg,
                "score": -9999.0,
                "error": str(e),
            })
        experiment_id += 1

    results_df = pd.DataFrame(results)
    if results_df.empty:
        print(f"{YELLOW}No results generated.{END}")
        return

    metric_cols = [
        "score", "net_profit_usd", "total_return_pct", "profit_factor",
        "sharpe_ratio", "sortino_ratio", "max_drawdown_pct", "trades"
    ]
    for col in metric_cols:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    results_df = results_df.sort_values(["score", "net_profit_usd"], ascending=[False, False])

    if all_trades:
        all_trades_df = pd.concat(all_trades, ignore_index=True)
    else:
        all_trades_df = pd.DataFrame()

    top10_df = results_df.head(10).copy()

    results_df.to_csv(f"{SYMBOL}_fast_lstm_H1_results.csv", index=False)
    top10_df.to_csv(f"{SYMBOL}_fast_lstm_H1_top10.csv", index=False)

    if not all_trades_df.empty:
        all_trades_df.to_csv(f"{SYMBOL}_fast_lstm_H1_all_trades.csv", index=False)

    print_top_results(results_df, top_n=10)

    print(f"\n{YELLOW}Saved:{END} {SYMBOL}_fast_lstm_H1_results.csv")
    print(f"{YELLOW}Saved:{END} {SYMBOL}_fast_lstm_H1_top10.csv")
    if not all_trades_df.empty:
        print(f"{YELLOW}Saved:{END} {SYMBOL}_fast_lstm_H1_all_trades.csv")


if __name__ == "__main__":
    main()
