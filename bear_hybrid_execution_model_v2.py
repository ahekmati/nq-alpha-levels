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
OUTPUT_DIR = BASE_DIR / "hybrid_backtest_outputs_v2"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

POINT_VALUE = 2.0
STARTING_EQUITY = 100000.0
RISK_PCT = 0.005
MAX_CONTRACTS = 10
COMMISSION_PER_CONTRACT_RT = 1.50
SLIPPAGE_POINTS = 2.0

TIME_STOP_HOURS = 72

ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_CYAN = "\033[96m"
ANSI_MAGENTA = "\033[95m"
ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"
# ---------------------------------------- #


CONFIGS = [
    {
        "name": "rebound_primary",
        "bear_prob_threshold": 0.55,
        "exhaust_prob_threshold": 0.42,
        "max_daily_ext_50": -2.50,
        "max_daily_ext_200": -3.25,
        "min_daily_rsi": 30,
        "max_daily_rsi": 47,
        "min_daily_adx": 18,
        "entry_mode": "rebound_only",
        "allow_early_range_short": False,
        "us_session_only": True,
        "session_start_hour_utc": 14,
        "session_end_hour_utc": 20,
        "stop_atr_mult": 1.6,
        "trail_atr_mult": 2.4,
        "target_r": 2.0,
        "early_size_mult": 1.0,
    },
    {
        "name": "hybrid_early_plus_rebound",
        "bear_prob_threshold": 0.50,
        "exhaust_prob_threshold": 0.40,
        "max_daily_ext_50": -2.25,
        "max_daily_ext_200": -3.00,
        "min_daily_rsi": 32,
        "max_daily_rsi": 50,
        "min_daily_adx": 17,
        "entry_mode": "hybrid",
        "allow_early_range_short": True,
        "us_session_only": True,
        "session_start_hour_utc": 14,
        "session_end_hour_utc": 20,
        "stop_atr_mult": 1.7,
        "trail_atr_mult": 2.5,
        "target_r": 2.2,
        "early_size_mult": 0.50,
    },
    {
        "name": "early_focus",
        "bear_prob_threshold": 0.48,
        "exhaust_prob_threshold": 0.35,
        "max_daily_ext_50": -2.00,
        "max_daily_ext_200": -2.75,
        "min_daily_rsi": 35,
        "max_daily_rsi": 55,
        "min_daily_adx": 15,
        "entry_mode": "early_and_rebound",
        "allow_early_range_short": True,
        "us_session_only": True,
        "session_start_hour_utc": 14,
        "session_end_hour_utc": 20,
        "stop_atr_mult": 1.8,
        "trail_atr_mult": 2.6,
        "target_r": 2.5,
        "early_size_mult": 0.40,
    },
]


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

    return daily, feature_cols


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
        "ret_5",
        "ret_10",
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
        "ret_5": "daily_model_ret_5",
        "ret_10": "daily_model_ret_10",
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
        "h1_rsi", "open", "high", "low", "close", "time"
    ]
    missing = [c for c in required if c not in h.columns]
    if missing:
        raise ValueError(f"Missing required H1 columns: {missing}")

    h["hour_utc"] = h["time"].dt.hour

    h["hh_20"] = h["high"].rolling(20).max().shift(1)
    h["ll_20"] = h["low"].rolling(20).min().shift(1)

    h["hh_8"] = h["high"].rolling(8).max().shift(1)
    h["ll_8"] = h["low"].rolling(8).min().shift(1)

    h["range_20"] = h["hh_20"] - h["ll_20"]
    h["range_pos"] = (h["close"] - h["ll_20"]) / h["range_20"]

    h["h1_below_ema20"] = (h["close"] < h["h1_ema_20"]).astype(int)
    h["h1_below_ema50"] = (h["close"] < h["h1_ema_50"]).astype(int)
    h["h1_ema_bear_stack"] = (
        (h["h1_ema_20"] < h["h1_ema_50"]) &
        (h["h1_ema_50"] < h["h1_ema_200"])
    ).astype(int)

    h["h1_ret_2"] = h["close"].pct_change(2)
    h["h1_ret_4"] = h["close"].pct_change(4)
    h["h1_ret_6"] = h["close"].pct_change(6)

    h["h1_pullback_to_ema20_atr"] = (h["high"] - h["h1_ema_20"]) / h["h1_atr"]
    h["h1_pullback_to_ema50_atr"] = (h["high"] - h["h1_ema_50"]) / h["h1_atr"]

    h["breakdown_signal"] = (
        (h["close"] < h["ll_20"]) &
        (h["close"] < h["open"]) &
        (h["h1_ema_bear_stack"] == 1) &
        (h["h1_rsi"] < 45)
    )

    h["rebound_failure_signal"] = (
        (h["high"] >= h["hh_8"]) &
        (h["close"] < h["open"]) &
        (h["close"] < h["h1_ema_20"]) &
        (h["h1_rsi"] < 55) &
        (h["h1_ret_4"] > 0)
    )

    h["early_range_short_signal"] = (
        (h["range_pos"] >= 0.75) &
        (h["close"] < h["open"]) &
        (h["high"] >= h["hh_8"]) &
        (h["h1_rsi"] >= 50) &
        (h["h1_rsi"] <= 68)
    )

    return h


def in_session(row, cfg):
    if not cfg["us_session_only"]:
        return True
    hour = row.get("hour_utc", np.nan)
    if pd.isna(hour):
        return False
    return cfg["session_start_hour_utc"] <= hour <= cfg["session_end_hour_utc"]


def base_regime_ok(row, cfg):
    bear_prob = row.get("bear_prob", np.nan)
    exhaust_prob = row.get("exhaust_prob", np.nan)
    daily_rsi = row.get("daily_model_rsi", np.nan)
    daily_adx = row.get("daily_model_adx", np.nan)
    d50 = row.get("daily_model_dist_ema50_atr", np.nan)
    d200 = row.get("daily_model_dist_ema200_atr", np.nan)
    s20 = row.get("daily_model_ema20_slope_5", np.nan)
    s50 = row.get("daily_model_ema50_slope_5", np.nan)

    if any(pd.isna(v) for v in [bear_prob, exhaust_prob, daily_rsi, daily_adx, d50, d200, s20, s50]):
        return False

    if bear_prob < cfg["bear_prob_threshold"]:
        return False
    if exhaust_prob > cfg["exhaust_prob_threshold"]:
        return False
    if daily_rsi < cfg["min_daily_rsi"] or daily_rsi > cfg["max_daily_rsi"]:
        return False
    if daily_adx < cfg["min_daily_adx"]:
        return False
    if d50 < cfg["max_daily_ext_50"]:
        return False
    if d200 < cfg["max_daily_ext_200"]:
        return False
    if s20 >= 0 or s50 >= 0:
        return False

    return True


def early_range_regime_ok(row, cfg):
    if not base_regime_ok(row, cfg):
        return False

    bear_prob = row.get("bear_prob", np.nan)
    daily_ret_5 = row.get("daily_model_ret_5", np.nan)
    daily_ret_10 = row.get("daily_model_ret_10", np.nan)
    daily_red = row.get("daily_model_rolling_red_ratio_10", np.nan)

    conds = [
        pd.notna(bear_prob) and bear_prob >= max(0.45, cfg["bear_prob_threshold"] - 0.05),
        pd.notna(daily_ret_5) and daily_ret_5 <= 0.01,
        pd.notna(daily_ret_10) and daily_ret_10 <= 0.02,
        pd.notna(daily_red) and daily_red >= 0.45,
    ]
    return all(conds)


def rebound_regime_ok(row, cfg):
    return base_regime_ok(row, cfg)


def calc_contracts(equity, entry, stop, size_mult=1.0):
    risk_points = abs(stop - entry)
    if risk_points <= 0:
        return 0

    risk_dollars_per_contract = risk_points * POINT_VALUE
    if risk_dollars_per_contract <= 0:
        return 0

    risk_budget = equity * RISK_PCT * size_mult
    contracts = int(np.floor(risk_budget / risk_dollars_per_contract))
    return max(0, min(MAX_CONTRACTS, contracts))


def choose_entry_type(row, cfg):
    if not in_session(row, cfg):
        return None

    if cfg["entry_mode"] in ["early_and_rebound", "hybrid"] and cfg["allow_early_range_short"]:
        if early_range_regime_ok(row, cfg) and bool(row.get("early_range_short_signal", False)):
            return "early_range_short"

    if cfg["entry_mode"] in ["rebound_only", "hybrid", "early_and_rebound"]:
        if rebound_regime_ok(row, cfg) and bool(row.get("rebound_failure_signal", False)):
            return "rebound_failure"

    if cfg["entry_mode"] == "hybrid":
        if rebound_regime_ok(row, cfg) and bool(row.get("breakdown_signal", False)):
            return "breakdown"

    return None


def backtest_config(h1: pd.DataFrame, cfg: dict):
    equity = STARTING_EQUITY
    equity_curve = []
    trades = []

    in_position = False
    entry_price = None
    stop_price = None
    stop_price_init = None
    target_price = None
    entry_time = None
    entry_type = None
    contracts = 0
    bars_in_trade = 0
    mfe = 0.0
    mae = 0.0

    start_idx = 30

    for i in range(start_idx, len(h1)):
        row = h1.iloc[i]

        if not in_position:
            signal_type = choose_entry_type(row, cfg)

            if signal_type is not None:
                raw_entry = row["close"] - SLIPPAGE_POINTS

                if signal_type == "early_range_short":
                    stop_ref = max(
                        row["high"],
                        row["hh_8"] if pd.notna(row.get("hh_8", np.nan)) else row["high"]
                    )
                    raw_stop = stop_ref + cfg["stop_atr_mult"] * row["h1_atr"]
                    size_mult = cfg["early_size_mult"]

                elif signal_type == "rebound_failure":
                    stop_ref = max(
                        row["high"],
                        row["hh_8"] if pd.notna(row.get("hh_8", np.nan)) else row["high"]
                    )
                    raw_stop = stop_ref + cfg["stop_atr_mult"] * row["h1_atr"]
                    size_mult = 1.0

                else:  # breakdown
                    stop_ref = row["high"]
                    raw_stop = stop_ref + cfg["stop_atr_mult"] * row["h1_atr"]
                    size_mult = 0.75

                qty = calc_contracts(equity, raw_entry, raw_stop, size_mult=size_mult)

                if qty > 0 and raw_stop > raw_entry:
                    in_position = True
                    entry_price = raw_entry
                    stop_price = raw_stop
                    stop_price_init = raw_stop
                    risk_per_contract = stop_price - entry_price
                    target_price = entry_price - cfg["target_r"] * risk_per_contract
                    entry_time = row["time"]
                    entry_type = signal_type
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

            trail_price = row["close"] + cfg["trail_atr_mult"] * row["h1_atr"]
            stop_price = min(stop_price, trail_price)

            if row["high"] >= stop_price:
                exit_price = stop_price + SLIPPAGE_POINTS
                exit_reason = "stop"

            elif row["low"] <= target_price:
                exit_price = target_price - SLIPPAGE_POINTS
                exit_reason = "target"

            elif row.get("exhaust_prob", 0.0) > (cfg["exhaust_prob_threshold"] + 0.10):
                exit_price = row["close"] + SLIPPAGE_POINTS
                exit_reason = "daily_exhaustion"

            elif row.get("daily_model_rsi", np.nan) > 52 and row["close"] > row["h1_ema_20"]:
                exit_price = row["close"] + SLIPPAGE_POINTS
                exit_reason = "trend_break"

            elif bars_in_trade >= TIME_STOP_HOURS:
                exit_price = row["close"] + SLIPPAGE_POINTS
                exit_reason = "time_stop"

            if exit_reason is not None:
                gross_pnl_points = entry_price - exit_price
                gross_pnl_dollars = gross_pnl_points * POINT_VALUE * contracts
                costs = COMMISSION_PER_CONTRACT_RT * contracts
                net_pnl_dollars = gross_pnl_dollars - costs

                initial_risk_dollars = (stop_price_init - entry_price) * POINT_VALUE * contracts
                net_r = net_pnl_dollars / initial_risk_dollars if initial_risk_dollars > 0 else np.nan

                equity += net_pnl_dollars

                trades.append({
                    "config": cfg["name"],
                    "entry_time": entry_time,
                    "exit_time": row["time"],
                    "entry_type": entry_type,
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
                    "entry_range_pos": row.get("range_pos", np.nan),
                })

                in_position = False
                entry_price = None
                stop_price = None
                stop_price_init = None
                target_price = None
                entry_time = None
                entry_type = None
                contracts = 0
                bars_in_trade = 0
                mfe = 0.0
                mae = 0.0

        equity_curve.append({
            "config": cfg["name"],
            "time": row["time"],
            "equity": equity,
            "in_position": int(in_position)
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_curve)
    return trades_df, equity_df


def summarize_performance(trades_df: pd.DataFrame, equity_df: pd.DataFrame, cfg_name: str):
    if trades_df.empty:
        return {
            "config": cfg_name,
            "trades": 0,
            "win_rate": np.nan,
            "avg_net_pnl": np.nan,
            "total_net_pnl": 0.0,
            "profit_factor": np.nan,
            "avg_bars": np.nan,
            "avg_r": np.nan,
            "max_drawdown": np.nan,
            "ending_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else STARTING_EQUITY,
            "early_trades": 0,
            "rebound_trades": 0,
            "breakdown_trades": 0,
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
        "config": cfg_name,
        "trades": len(trades_df),
        "win_rate": (trades_df["net_pnl_dollars"] > 0).mean(),
        "avg_net_pnl": trades_df["net_pnl_dollars"].mean(),
        "total_net_pnl": trades_df["net_pnl_dollars"].sum(),
        "profit_factor": profit_factor,
        "avg_bars": trades_df["bars_in_trade"].mean(),
        "avg_r": trades_df["net_r"].mean(),
        "max_drawdown": max_dd,
        "ending_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else STARTING_EQUITY,
        "early_trades": (trades_df["entry_type"] == "early_range_short").sum(),
        "rebound_trades": (trades_df["entry_type"] == "rebound_failure").sum(),
        "breakdown_trades": (trades_df["entry_type"] == "breakdown").sum(),
    }


def print_config_summary(summary: dict):
    title_color = ANSI_GREEN if pd.notna(summary["total_net_pnl"]) and summary["total_net_pnl"] > 0 else ANSI_RED
    print(color(f"\n[{summary['config']}]", ANSI_BOLD + title_color))
    print(f"Trades={summary['trades']} | WinRate={fmt(100 * summary['win_rate'], 2)}% | "
          f"PF={fmt(summary['profit_factor'], 3)} | AvgR={fmt(summary['avg_r'], 3)} | "
          f"NetPnL=${fmt(summary['total_net_pnl'], 2)} | MaxDD=${fmt(summary['max_drawdown'], 2)}")
    print(f"Entry mix -> Early:{summary['early_trades']} | Rebound:{summary['rebound_trades']} | Breakdown:{summary['breakdown_trades']}")


def main():
    daily, h1 = load_data()

    section("Fit Daily Regime Models")
    daily_scored, feature_cols = fit_daily_models(daily)
    print(color(f"Daily feature set: {', '.join(feature_cols)}", ANSI_YELLOW))

    daily_scored_path = OUTPUT_DIR / "daily_scored_with_probs.csv"
    daily_scored.to_csv(daily_scored_path, index=False)
    print(color(f"Saved {daily_scored_path}", ANSI_CYAN))

    section("Prepare H1 Execution Dataset")
    h1_merged = merge_daily_probs_into_h1(h1, daily_scored)
    h1_exec = add_h1_execution_features(h1_merged)

    h1_exec_path = OUTPUT_DIR / "h1_execution_dataset_v2.csv"
    h1_exec.to_csv(h1_exec_path, index=False)
    print(color(f"Saved {h1_exec_path}", ANSI_CYAN))

    section("Run Config Sweep")

    all_trades = []
    all_equity = []
    summaries = []

    for cfg in CONFIGS:
        trades_df, equity_df = backtest_config(h1_exec, cfg)
        summary = summarize_performance(trades_df, equity_df, cfg["name"])

        all_trades.append(trades_df)
        all_equity.append(equity_df)
        summaries.append(summary)

        print_config_summary(summary)

    summary_df = pd.DataFrame(summaries).sort_values(
        ["total_net_pnl", "profit_factor", "avg_r"],
        ascending=False
    )

    trades_all_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity_all_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()

    summary_path = OUTPUT_DIR / "config_sweep_summary.csv"
    trades_path = OUTPUT_DIR / "config_sweep_trades.csv"
    equity_path = OUTPUT_DIR / "config_sweep_equity.csv"

    summary_df.to_csv(summary_path, index=False)
    trades_all_df.to_csv(trades_path, index=False)
    equity_all_df.to_csv(equity_path, index=False)

    section("Best Configuration")
    if not summary_df.empty:
        best = summary_df.iloc[0].to_dict()
        print_config_summary(best)

    print()
    print(color("Saved outputs:", ANSI_BOLD + ANSI_GREEN))
    print(f"- {summary_path}")
    print(f"- {trades_path}")
    print(f"- {equity_path}")

    section("Interpretation")
    print("- Early range shorts are intentionally smaller because they are anticipatory and can be early.")
    print("- Rebound-failure shorts are the higher-confidence continuation entries.")
    print("- Breakdown shorts are retained only in the hybrid config and should usually be the minority.")
    print("- If early entries help, they should improve entry price and lower extension risk, not necessarily raise hit rate dramatically.")

if __name__ == "__main__":
    main()
