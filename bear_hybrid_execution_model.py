import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ---------------- CONFIG ---------------- #
BASE_DIR = Path("research_mnq_bear_model")
DAILY_FILE = BASE_DIR / "mnq_daily_research_dataset.csv"
H1_FILE = BASE_DIR / "mnq_h1_research_dataset.csv"
OUTPUT_DIR = BASE_DIR / "hybrid_backtest_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# Contract / execution assumptions
POINT_VALUE = 2.0
STARTING_EQUITY = 100000.0
RISK_PCT = 0.005
MAX_CONTRACTS = 10
COMMISSION_PER_CONTRACT_RT = 1.50
SLIPPAGE_POINTS = 2.0

# Entry / exit logic
ENTRY_LOOKBACK_HOURS = 20
PULLBACK_LOOKBACK_HOURS = 8
STOP_ATR_MULT = 1.6
TARGET_R_MULT = 2.0
TIME_STOP_HOURS = 48
TRAIL_STOP_ATR_MULT = 2.2

# Daily regime thresholds
BEAR_PROB_THRESHOLD = 0.55
EXHAUST_PROB_THRESHOLD = 0.45

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_MAGENTA = "\033[95m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"
# ---------------------------------------- #


def color(text, c):
    return f"{c}{text}{ANSI_RESET}"


def section(title):
    print()
    print(color(f"{'=' * 12} {title} {'=' * 12}", ANSI_BOLD + ANSI_CYAN))


def fmt(x, digits=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def load_data():
    if not DAILY_FILE.exists():
        raise FileNotFoundError(f"Missing {DAILY_FILE}")
    if not H1_FILE.exists():
        raise FileNotFoundError(f"Missing {H1_FILE}")

    daily = pd.read_csv(DAILY_FILE)
    h1 = pd.read_csv(H1_FILE)

    daily["time"] = pd.to_datetime(daily["time"], utc=True)
    h1["time"] = pd.to_datetime(h1["time"], utc=True)

    daily = daily.sort_values("time").reset_index(drop=True)
    h1 = h1.sort_values("time").reset_index(drop=True)

    return daily, h1


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["target_active_bear"] = (d["label"] == "ACTIVE_BEAR").astype(int)

    d["target_exhaustion_vs_active"] = np.where(
        d["label"].isin(["ACTIVE_BEAR", "EXHAUSTION"]),
        (d["label"] == "EXHAUSTION").astype(int),
        np.nan
    )
    return d


def choose_daily_features(df: pd.DataFrame):
    preferred = [
        "rsi",
        "dist_ema20_atr",
        "dist_ema50_atr",
        "dist_ema200_atr",
        "ema10_slope_5",
        "ema20_slope_5",
        "ema50_slope_5",
        "drawdown_pct",
        "ret_10",
        "ret_5",
        "atr",
        "adx",
        "rolling_red_ratio_10",
        "rolling_red_ratio_5",
        "rolling_green_ratio_5",
        "rolling_vol_10",
        "downside_vol_10",
        "range",
        "body",
    ]
    return [c for c in preferred if c in df.columns]


def fit_daily_models(daily: pd.DataFrame):
    daily = build_targets(daily)
    feature_cols = choose_daily_features(daily)

    active_df = daily.dropna(subset=["target_active_bear"]).copy()
    X_active = active_df[feature_cols]
    y_active = active_df["target_active_bear"].astype(int)

    active_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    active_model.fit(X_active, y_active)

    exh_df = daily.dropna(subset=["target_exhaustion_vs_active"]).copy()
    X_exh = exh_df[feature_cols]
    y_exh = exh_df["target_exhaustion_vs_active"].astype(int)

    exhaustion_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=5,
            min_samples_leaf=15,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE + 1,
            n_jobs=-1
        ))
    ])
    exhaustion_model.fit(X_exh, y_exh)

    daily["bear_prob"] = active_model.predict_proba(daily[feature_cols])[:, 1]

    active_subset = daily["label"].isin(["ACTIVE_BEAR", "EXHAUSTION"])
    daily["exhaust_prob"] = np.nan
    if active_subset.sum() > 0:
        daily.loc[active_subset, "exhaust_prob"] = exhaustion_model.predict_proba(
            daily.loc[active_subset, feature_cols]
        )[:, 1]

    daily["exhaust_prob"] = daily["exhaust_prob"].fillna(0.0)

    return daily, active_model, exhaustion_model, feature_cols


def merge_daily_probs_into_h1(h1: pd.DataFrame, daily: pd.DataFrame):
    h = h1.copy()
    h["date"] = h["time"].dt.floor("D")

    d = daily.copy()
    d["date"] = d["time"].dt.floor("D")

    keep = [
        "date",
        "bear_prob",
        "exhaust_prob",
        "label",
        "rsi",
        "adx",
        "atr",
        "dist_ema20_atr",
        "dist_ema50_atr",
        "dist_ema200_atr",
        "ema20_slope_5",
        "ema50_slope_5",
        "drawdown_pct",
        "rolling_red_ratio_10",
        "rolling_green_ratio_5",
    ]
    d = d[keep].copy()

    d = d.rename(columns={
        "label": "daily_model_label",
        "rsi": "daily_model_rsi",
        "adx": "daily_model_adx",
        "atr": "daily_model_atr",
        "dist_ema20_atr": "daily_model_dist_ema20_atr",
        "dist_ema50_atr": "daily_model_dist_ema50_atr",
        "dist_ema200_atr": "daily_model_dist_ema200_atr",
        "ema20_slope_5": "daily_model_ema20_slope_5",
        "ema50_slope_5": "daily_model_ema50_slope_5",
        "drawdown_pct": "daily_model_drawdown_pct",
        "rolling_red_ratio_10": "daily_model_rolling_red_ratio_10",
        "rolling_green_ratio_5": "daily_model_rolling_green_ratio_5",
    })

    dupes = [c for c in d.columns if c in h.columns and c != "date"]
    if dupes:
        raise ValueError(f"Duplicate columns would be created in merge: {dupes}")

    merged = h.merge(d, on="date", how="left")
    merged = merged.sort_values("time").reset_index(drop=True)

    return merged


def add_h1_execution_features(h1: pd.DataFrame):
    h = h1.copy()

    required = [
        "h1_atr", "h1_ema_20", "h1_ema_50", "h1_ema_200",
        "h1_rsi", "open", "high", "low", "close"
    ]
    missing = [c for c in required if c not in h.columns]
    if missing:
        raise ValueError(f"Missing required H1 columns: {missing}")

    h["hh_entry_lb"] = h["high"].rolling(ENTRY_LOOKBACK_HOURS).max().shift(1)
    h["ll_entry_lb"] = h["low"].rolling(ENTRY_LOOKBACK_HOURS).min().shift(1)

    h["pullback_high_8"] = h["high"].rolling(PULLBACK_LOOKBACK_HOURS).max().shift(1)
    h["pullback_low_8"] = h["low"].rolling(PULLBACK_LOOKBACK_HOURS).min().shift(1)

    h["h1_below_ema20"] = (h["close"] < h["h1_ema_20"]).astype(int)
    h["h1_below_ema50"] = (h["close"] < h["h1_ema_50"]).astype(int)
    h["h1_ema_bear_stack"] = (
        (h["h1_ema_20"] < h["h1_ema_50"]) &
        (h["h1_ema_50"] < h["h1_ema_200"])
    ).astype(int)

    h["h1_ret_2"] = h["close"].pct_change(2)
    h["h1_ret_6"] = h["close"].pct_change(6)

    h["breakdown_signal"] = (
        (h["close"] < h["ll_entry_lb"]) &
        (h["close"] < h["open"]) &
        (h["h1_below_ema20"] == 1) &
        (h["h1_ema_bear_stack"] == 1)
    )

    h["rebound_failure_signal"] = (
        (h["high"] >= h["pullback_high_8"]) &
        (h["close"] < h["open"]) &
        (h["close"] < h["h1_ema_20"]) &
        (h["h1_rsi"] < 55)
    )

    h["entry_signal"] = h["breakdown_signal"] | h["rebound_failure_signal"]

    return h


def regime_allows_short(row):
    bear_prob = row.get("bear_prob", np.nan)
    exhaust_prob = row.get("exhaust_prob", np.nan)

    daily_dist_ema50_atr = row.get("daily_model_dist_ema50_atr", np.nan)
    daily_dist_ema200_atr = row.get("daily_model_dist_ema200_atr", np.nan)
    daily_ema20_slope_5 = row.get("daily_model_ema20_slope_5", np.nan)
    daily_ema50_slope_5 = row.get("daily_model_ema50_slope_5", np.nan)
    daily_rsi = row.get("daily_model_rsi", np.nan)
    daily_adx = row.get("daily_model_adx", np.nan)

    if pd.isna(bear_prob) or pd.isna(exhaust_prob):
        return False

    strong_trend_structure = (
        pd.notna(daily_dist_ema50_atr) and
        pd.notna(daily_dist_ema200_atr) and
        pd.notna(daily_ema20_slope_5) and
        pd.notna(daily_ema50_slope_5) and
        (daily_dist_ema50_atr < -0.75) and
        (daily_dist_ema200_atr < -0.50) and
        (daily_ema20_slope_5 < 0) and
        (daily_ema50_slope_5 < 0)
    )

    trend_quality = (
        pd.notna(daily_rsi) and
        pd.notna(daily_adx) and
        (daily_rsi < 45) and
        (daily_adx > 20)
    )

    return (
        (bear_prob >= BEAR_PROB_THRESHOLD) and
        (exhaust_prob <= EXHAUST_PROB_THRESHOLD) and
        strong_trend_structure and
        trend_quality
    )


def calc_contracts(equity, entry, stop):
    risk_points = abs(stop - entry)
    if risk_points <= 0:
        return 0

    risk_dollars_per_contract = risk_points * POINT_VALUE
    if risk_dollars_per_contract <= 0:
        return 0

    risk_budget = equity * RISK_PCT
    contracts = int(np.floor(risk_budget / risk_dollars_per_contract))
    return max(0, min(MAX_CONTRACTS, contracts))


def backtest_short_strategy(h1: pd.DataFrame):
    equity = STARTING_EQUITY
    equity_curve = []
    trades = []

    in_position = False
    entry_price = None
    stop_price = None
    stop_price_init = None
    target_price = None
    entry_time = None
    contracts = 0
    bars_in_trade = 0
    mfe = 0.0
    mae = 0.0

    start_idx = max(ENTRY_LOOKBACK_HOURS, PULLBACK_LOOKBACK_HOURS) + 5

    for i in range(start_idx, len(h1)):
        row = h1.iloc[i]

        if not in_position:
            if regime_allows_short(row) and bool(row.get("entry_signal", False)):
                raw_entry = row["close"] - SLIPPAGE_POINTS

                stop_candidate_1 = row["high"] + STOP_ATR_MULT * row["h1_atr"]
                if pd.notna(row.get("pullback_high_8", np.nan)):
                    stop_candidate_2 = row["pullback_high_8"] + 0.5 * row["h1_atr"]
                else:
                    stop_candidate_2 = stop_candidate_1

                raw_stop = max(stop_candidate_1, stop_candidate_2)

                qty = calc_contracts(equity, raw_entry, raw_stop)
                if qty > 0 and raw_stop > raw_entry:
                    in_position = True
                    entry_price = raw_entry
                    stop_price = raw_stop
                    stop_price_init = raw_stop
                    risk_per_contract = stop_price - entry_price
                    target_price = entry_price - TARGET_R_MULT * risk_per_contract
                    entry_time = row["time"]
                    contracts = qty
                    bars_in_trade = 0
                    mfe = 0.0
                    mae = 0.0

        else:
            bars_in_trade += 1

            favorable = entry_price - row["low"]
            adverse = row["high"] - entry_price
            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            exit_reason = None
            exit_price = None

            trail_price = row["close"] + TRAIL_STOP_ATR_MULT * row["h1_atr"]
            stop_price = min(stop_price, trail_price)

            if row["high"] >= stop_price:
                exit_price = stop_price + SLIPPAGE_POINTS
                exit_reason = "stop"

            elif row["low"] <= target_price:
                exit_price = target_price - SLIPPAGE_POINTS
                exit_reason = "target"

            elif row.get("exhaust_prob", 0.0) > (EXHAUST_PROB_THRESHOLD + 0.10):
                exit_price = row["close"] + SLIPPAGE_POINTS
                exit_reason = "daily_exhaustion"

            elif row.get("daily_model_rsi", np.nan) > 50 and row["close"] > row["h1_ema_20"]:
                exit_price = row["close"] + SLIPPAGE_POINTS
                exit_reason = "trend_break"

            elif bars_in_trade >= TIME_STOP_HOURS:
                exit_price = row["close"] + SLIPPAGE_POINTS
                exit_reason = "time_stop"

            if exit_reason is not None:
                gross_pnl_points = (entry_price - exit_price)
                gross_pnl_dollars = gross_pnl_points * POINT_VALUE * contracts
                costs = COMMISSION_PER_CONTRACT_RT * contracts
                net_pnl_dollars = gross_pnl_dollars - costs

                risk_budget_at_entry = STARTING_EQUITY * RISK_PCT if STARTING_EQUITY > 0 else np.nan
                if stop_price_init is not None and entry_price is not None:
                    initial_risk_dollars = (stop_price_init - entry_price) * POINT_VALUE * contracts
                else:
                    initial_risk_dollars = np.nan

                net_r = net_pnl_dollars / initial_risk_dollars if pd.notna(initial_risk_dollars) and initial_risk_dollars > 0 else np.nan

                equity += net_pnl_dollars

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": row["time"],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_price_init": stop_price_init,
                    "stop_price_final": stop_price,
                    "target_price": target_price,
                    "contracts": contracts,
                    "bars_in_trade": bars_in_trade,
                    "gross_pnl_points": gross_pnl_points,
                    "gross_pnl_dollars": gross_pnl_dollars,
                    "net_pnl_dollars": net_pnl_dollars,
                    "net_r": net_r,
                    "mfe_points": mfe,
                    "mae_points": mae,
                    "exit_reason": exit_reason,
                    "entry_bear_prob": row.get("bear_prob", np.nan),
                    "entry_exhaust_prob": row.get("exhaust_prob", np.nan),
                    "entry_daily_rsi": row.get("daily_model_rsi", np.nan),
                    "entry_daily_adx": row.get("daily_model_adx", np.nan),
                    "entry_daily_dist_ema50_atr": row.get("daily_model_dist_ema50_atr", np.nan),
                    "entry_daily_dist_ema200_atr": row.get("daily_model_dist_ema200_atr", np.nan),
                })

                in_position = False
                entry_price = None
                stop_price = None
                stop_price_init = None
                target_price = None
                entry_time = None
                contracts = 0
                bars_in_trade = 0
                mfe = 0.0
                mae = 0.0

        equity_curve.append({
            "time": row["time"],
            "equity": equity,
            "in_position": int(in_position)
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    return trades_df, equity_df


def summarize_performance(trades_df: pd.DataFrame, equity_df: pd.DataFrame):
    if trades_df.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_net_pnl": np.nan,
            "total_net_pnl": 0.0,
            "profit_factor": np.nan,
            "avg_bars": np.nan,
            "avg_r": np.nan,
            "max_drawdown": np.nan,
            "ending_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else STARTING_EQUITY,
        }

    wins = trades_df[trades_df["net_pnl_dollars"] > 0]
    losses = trades_df[trades_df["net_pnl_dollars"] < 0]

    gross_profit = wins["net_pnl_dollars"].sum() if not wins.empty else 0.0
    gross_loss = abs(losses["net_pnl_dollars"].sum()) if not losses.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.nan

    eq = equity_df["equity"].copy()
    running_max = eq.cummax()
    dd = eq - running_max
    max_dd = dd.min() if not dd.empty else np.nan

    return {
        "trades": len(trades_df),
        "win_rate": (trades_df["net_pnl_dollars"] > 0).mean(),
        "avg_net_pnl": trades_df["net_pnl_dollars"].mean(),
        "total_net_pnl": trades_df["net_pnl_dollars"].sum(),
        "profit_factor": profit_factor,
        "avg_bars": trades_df["bars_in_trade"].mean(),
        "avg_r": trades_df["net_r"].mean(),
        "max_drawdown": max_dd,
        "ending_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else STARTING_EQUITY,
    }


def print_summary(summary: dict, trades_df: pd.DataFrame):
    section("Hybrid Backtest Summary")

    print(color(f"Trades: {summary['trades']}", ANSI_YELLOW))
    print(color(
        f"Win rate: {fmt(100 * summary['win_rate'], 2)}%",
        ANSI_GREEN if pd.notna(summary["win_rate"]) and summary["win_rate"] >= 0.5 else ANSI_RED
    ))
    print(color(f"Average net PnL/trade: ${fmt(summary['avg_net_pnl'], 2)}", ANSI_CYAN))
    print(color(
        f"Total net PnL: ${fmt(summary['total_net_pnl'], 2)}",
        ANSI_GREEN if pd.notna(summary["total_net_pnl"]) and summary["total_net_pnl"] > 0 else ANSI_RED
    ))
    print(color(f"Profit factor: {fmt(summary['profit_factor'], 3)}", ANSI_MAGENTA))
    print(color(f"Average bars in trade: {fmt(summary['avg_bars'], 2)}", ANSI_YELLOW))
    print(color(f"Average R per trade: {fmt(summary['avg_r'], 3)}", ANSI_CYAN))
    print(color(f"Max drawdown: ${fmt(summary['max_drawdown'], 2)}", ANSI_RED))
    print(color(
        f"Ending equity: ${fmt(summary['ending_equity'], 2)}",
        ANSI_GREEN if pd.notna(summary["ending_equity"]) and summary["ending_equity"] > STARTING_EQUITY else ANSI_RED
    ))

    if not trades_df.empty:
        section("Exit Reasons")
        exit_counts = trades_df["exit_reason"].value_counts()
        for k, v in exit_counts.items():
            print(color(f"{k:>18}: {v}", ANSI_MAGENTA))

        section("Trade Diagnostics")
        print(color(f"Average entry bear_prob: {fmt(trades_df['entry_bear_prob'].mean(), 3)}", ANSI_CYAN))
        print(color(f"Average entry exhaust_prob: {fmt(trades_df['entry_exhaust_prob'].mean(), 3)}", ANSI_CYAN))
        print(color(f"Average entry daily RSI: {fmt(trades_df['entry_daily_rsi'].mean(), 2)}", ANSI_CYAN))
        print(color(f"Average entry daily ADX: {fmt(trades_df['entry_daily_adx'].mean(), 2)}", ANSI_CYAN))
        print(color(f"Average entry daily dist_ema50_atr: {fmt(trades_df['entry_daily_dist_ema50_atr'].mean(), 3)}", ANSI_CYAN))
        print(color(f"Average entry daily dist_ema200_atr: {fmt(trades_df['entry_daily_dist_ema200_atr'].mean(), 3)}", ANSI_CYAN))


def main():
    daily, h1 = load_data()

    section("Fit Daily Regime Models")
    daily_scored, active_model, exhaustion_model, feature_cols = fit_daily_models(daily)
    print(color(f"Daily feature set: {', '.join(feature_cols)}", ANSI_YELLOW))

    daily_scored_path = OUTPUT_DIR / "daily_scored_with_probs.csv"
    daily_scored.to_csv(daily_scored_path, index=False)
    print(color(f"Saved {daily_scored_path}", ANSI_CYAN))

    section("Prepare H1 Execution Dataset")
    h1_merged = merge_daily_probs_into_h1(h1, daily_scored)
    h1_exec = add_h1_execution_features(h1_merged)

    h1_exec_path = OUTPUT_DIR / "h1_execution_dataset.csv"
    h1_exec.to_csv(h1_exec_path, index=False)
    print(color(f"Saved {h1_exec_path}", ANSI_CYAN))

    section("Run Hybrid Backtest")
    trades_df, equity_df = backtest_short_strategy(h1_exec)

    trades_path = OUTPUT_DIR / "hybrid_short_trades.csv"
    equity_path = OUTPUT_DIR / "hybrid_short_equity.csv"
    summary_path = OUTPUT_DIR / "hybrid_short_summary.csv"

    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)

    summary = summarize_performance(trades_df, equity_df)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print_summary(summary, trades_df)

    print()
    print(color("Saved outputs:", ANSI_BOLD + ANSI_GREEN))
    print(f"- {trades_path}")
    print(f"- {equity_path}")
    print(f"- {summary_path}")

    section("Next Iteration Ideas")
    print("- Tune bear_prob and exhaust_prob thresholds on walk-forward segments.")
    print("- Add session filters so entries only trigger during liquid index futures hours.")
    print("- Replace the rules-based H1 trigger with a trained H1 execution classifier.")
    print("- Add partial exits and better trailing-stop memory.")
    print("- Evaluate performance by labeled regime and by calendar subperiod.")

if __name__ == "__main__":
    main()
