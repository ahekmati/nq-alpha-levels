from mt5linux import MetaTrader5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from math import sqrt
from itertools import product

SYMBOL_CANDIDATES = ["@MNQ", "MNQ", "MNQM26", "MNQU26", "MNQZ26"]
TIMEFRAME_REGIME = MetaTrader5.TIMEFRAME_D1
TIMEFRAME_EXEC = MetaTrader5.TIMEFRAME_H1

D1_BARS = 5000
H1_BARS = 50000

EMA_PERIOD = 100
SLOPE_LOOKBACK = 10
ATR_PERIOD = 14
RSI_PERIOD = 14
RSI_TIGHTEN_PERIOD = 10
BREAKOUT_LOOKBACK = 5

REGIME_RSI_MAX = 60
ATR_EXPANSION_MIN = 1.2
USE_FALLING_EMA_ONLY = True
USE_RSI_CAP = True

WF_START_DATE = "2020-01-01"
TRAIN_START_DATE = "2020-01-01"
INITIAL_TRAIN_MONTHS = 12
TEST_MONTHS = 2
EMBARGO_BARS = 24 * 3
MIN_TRAIN_ROWS = 40
MIN_TEST_ROWS = 8

RF_ESTIMATORS = 300
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 10
RF_RANDOM_STATE = 42
RF_THRESHOLD = 0.50

INITIAL_CAPITAL = 5000.0
CONTRACTS = 1
MNQ_POINT_VALUE = 2.0
MAX_LOSS_PER_TRADE_USD = 400.0
MAX_STOP_POINTS = MAX_LOSS_PER_TRADE_USD / (MNQ_POINT_VALUE * CONTRACTS)
MAX_HOLD_BARS = 48

BASE_STOP_TYPE = "fixed"
BASE_STOP_VALUE = 100.0
BASE_TARGET_TYPE = "fixed"
BASE_TARGET_VALUE = 60.0

RESULTS_CSV = "ema100_tightening_optimizer_results.csv"
BEST_TRADES_CSV = "ema100_tightening_optimizer_best_trades.csv"
WF_SUMMARY_CSV = "ema100_tightening_optimizer_walkforward_summary.csv"

RSI_THRESHOLDS = [25, 28, 30, 32, 35]
ATR_STRETCH_THRESHOLDS = [1.00, 1.10, 1.25, 1.40, 1.50]
TIGHTEN_FACTORS = [0.25, 0.50, 0.75, 1.00]
TIGHTEN_MODES = [
    {"tighten_mode": "rsi", "force_exit": False},
    {"tighten_mode": "atr_exhaust", "force_exit": False},
    {"tighten_mode": "either", "force_exit": False},
    {"tighten_mode": "either", "force_exit": True},
]

mt5 = MetaTrader5()


def resolve_symbol(mt5_client, candidates):
    for sym in candidates:
        info = mt5_client.symbol_info(sym)
        if info is None:
            continue
        mt5_client.symbol_select(sym, True)
        rates = mt5_client.copy_rates_from_pos(sym, TIMEFRAME_REGIME, 0, 20)
        if rates is not None and len(rates) > 0:
            return sym
    all_symbols = mt5_client.symbols_get()
    mnq_like = [s.name for s in all_symbols if "MNQ" in s.name.upper()] if all_symbols else []
    raise RuntimeError(f"Could not resolve symbol. MNQ-like symbols visible: {mnq_like[:50]}")


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def atr(df, period=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def sharpe_from_returns(rets, annualization=252):
    rets = pd.Series(rets).dropna()
    if len(rets) < 2:
        return np.nan
    std = rets.std(ddof=1)
    if std == 0 or np.isnan(std):
        return np.nan
    return (rets.mean() / std) * sqrt(annualization)


def max_drawdown_from_equity(equity_curve):
    eq = pd.Series(equity_curve).dropna()
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = eq - peak
    return dd.min()


def profit_factor_from_pnl(pnl_series):
    s = pd.Series(pnl_series).dropna()
    gross_profit = s[s > 0].sum()
    gross_loss = -s[s < 0].sum()
    if gross_loss <= 0:
        return np.nan
    return gross_profit / gross_loss


def prepare_regime_df(df_d1):
    df = df_d1.copy()
    df["ema100"] = ema(df["close"], EMA_PERIOD)
    df["ema_slope_n"] = df["ema100"] - df["ema100"].shift(SLOPE_LOOKBACK)
    df["atr14_d1"] = atr(df, ATR_PERIOD)
    df["rsi14"] = rsi(df["close"], RSI_PERIOD)
    df["rsi10"] = rsi(df["close"], RSI_TIGHTEN_PERIOD)
    df["d1_range"] = df["high"] - df["low"]
    df["d1_range_atr_ratio"] = df["d1_range"] / df["atr14_d1"].replace(0, np.nan)

    df["below_ema"] = df["close"] < df["ema100"]
    df["below_falling"] = df["below_ema"] & (df["ema_slope_n"] < 0)
    df["bear_permission"] = df["below_falling"] if USE_FALLING_EMA_ONLY else df["below_ema"]

    if USE_RSI_CAP:
        df["bear_permission"] = df["bear_permission"] & (df["rsi14"] <= REGIME_RSI_MAX)

    df["regime_age"] = 0
    age = 0
    for i in range(len(df)):
        if bool(df.loc[i, "bear_permission"]):
            age += 1
        else:
            age = 0
        df.loc[i, "regime_age"] = age

    df["d1_close_vs_ema_atr"] = (df["close"] - df["ema100"]) / df["atr14_d1"].replace(0, np.nan)
    df["d1_ema_slope_atr"] = df["ema_slope_n"] / df["atr14_d1"].replace(0, np.nan)
    df["d1_ret_5"] = df["close"].pct_change(5)
    return df


def prepare_exec_df(df_h1, regime_df):
    df = df_h1.copy()
    df["ema100_h1"] = ema(df["close"], EMA_PERIOD)
    df["atr14"] = atr(df, ATR_PERIOD)
    df["rsi14_h1"] = rsi(df["close"], RSI_PERIOD)
    df["ret_1h"] = df["close"].pct_change()
    df["ret_6h"] = df["close"].pct_change(6)
    df["ret_24h"] = df["close"].pct_change(24)
    df["bar_range"] = df["high"] - df["low"]
    df["body_size"] = (df["close"] - df["open"]).abs()
    df["range_ma20"] = df["bar_range"].rolling(20).mean()
    df["volume_ma20"] = df["tick_volume"].rolling(20).mean()
    df["atr_ratio"] = df["bar_range"] / df["range_ma20"].replace(0, np.nan)
    df["tickvol_ratio"] = df["tick_volume"] / df["volume_ma20"].replace(0, np.nan)
    df["recent_low_prev"] = df["low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    df["recent_high_prev"] = df["high"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    df["swing_low_24"] = df["low"].rolling(24).min().shift(1)
    df["swing_high_24"] = df["high"].rolling(24).max().shift(1)
    df["dist_to_recent_low_atr"] = (df["close"] - df["recent_low_prev"]) / df["atr14"].replace(0, np.nan)
    df["dist_to_recent_high_atr"] = (df["recent_high_prev"] - df["close"]) / df["atr14"].replace(0, np.nan)
    df["dist_to_h1_ema_atr"] = (df["close"] - df["ema100_h1"]) / df["atr14"].replace(0, np.nan)
    df["hour"] = df["time"].dt.hour
    df["dayofweek"] = df["time"].dt.dayofweek
    df["is_us_session"] = df["hour"].between(13, 20).astype(int)

    regime_map = regime_df[[
        "time", "bear_permission", "regime_age", "ema100", "ema_slope_n", "rsi14", "rsi10",
        "atr14_d1", "d1_close_vs_ema_atr", "d1_ema_slope_atr", "d1_ret_5", "d1_range_atr_ratio"
    ]].copy()

    regime_map = regime_map.rename(columns={
        "time": "d1_time",
        "ema100": "d1_ema100",
        "rsi14": "d1_rsi14",
        "rsi10": "d1_rsi10",
    }).sort_values("d1_time")

    df = pd.merge_asof(
        df.sort_values("time"),
        regime_map,
        left_on="time",
        right_on="d1_time",
        direction="backward"
    )

    in_bear = df["bear_permission"].fillna(False)
    red_bar = df["close"] < df["open"]
    atr_ok = df["bar_range"] > (df["range_ma20"] * ATR_EXPANSION_MIN)

    df["trig_atr_breakdown"] = (
        in_bear &
        red_bar &
        atr_ok &
        (df["close"] < df["recent_low_prev"])
    )

    df["trig_rsi_rollover"] = (
        in_bear &
        (df["rsi14_h1"].shift(1) > 60) &
        (df["rsi14_h1"] < 50) &
        red_bar
    )

    return df


def build_signal_rows(df):
    rows = []
    trigger_specs = [
        ("trig_atr_breakdown", "atr_breakdown"),
        ("trig_rsi_rollover", "rsi_rollover"),
    ]

    for trig_col, trig_name in trigger_specs:
        signal_idx = np.where(df[trig_col].fillna(False).values)[0]

        for i in signal_idx:
            if i + 1 >= len(df):
                continue

            row = df.iloc[i]
            entry_row = df.iloc[i + 1]
            atr_now = row.get("atr14", np.nan)
            if pd.isna(atr_now) or atr_now <= 0:
                continue

            prev1 = df.iloc[i - 1] if i - 1 >= 0 else row
            prev6_start = max(0, i - 6)
            prev24_start = max(0, i - 24)
            slice6 = df.iloc[prev6_start:i]
            slice24 = df.iloc[prev24_start:i]

            rows.append({
                "signal_idx": i,
                "signal_time": row["time"],
                "trigger": trig_name,
                "entry_idx": i + 1,
                "entry_time": entry_row["time"],
                "entry_price": entry_row["open"],

                "regime_age": row.get("regime_age", np.nan),
                "d1_rsi14": row.get("d1_rsi14", np.nan),
                "d1_rsi10": row.get("d1_rsi10", np.nan),
                "d1_range_atr_ratio": row.get("d1_range_atr_ratio", np.nan),
                "d1_close_vs_ema_atr": row.get("d1_close_vs_ema_atr", np.nan),
                "d1_ema_slope_atr": row.get("d1_ema_slope_atr", np.nan),
                "d1_ret_5": row.get("d1_ret_5", np.nan),

                "h1_atr14": row.get("atr14", np.nan),
                "h1_rsi14": row.get("rsi14_h1", np.nan),
                "h1_ret_1h": row.get("ret_1h", np.nan),
                "h1_ret_6h": row.get("ret_6h", np.nan),
                "h1_ret_24h": row.get("ret_24h", np.nan),

                "bar_range": row.get("bar_range", np.nan),
                "body_size": row.get("body_size", np.nan),
                "atr_ratio": row.get("atr_ratio", np.nan),
                "tickvol_ratio": row.get("tickvol_ratio", np.nan),

                "dist_to_recent_low_atr": row.get("dist_to_recent_low_atr", np.nan),
                "dist_to_recent_high_atr": row.get("dist_to_recent_high_atr", np.nan),
                "dist_to_h1_ema_atr": row.get("dist_to_h1_ema_atr", np.nan),

                "recent_6h_mean_ret": slice6["ret_1h"].mean() if len(slice6) else np.nan,
                "recent_6h_std_ret": slice6["ret_1h"].std(ddof=1) if len(slice6) > 1 else np.nan,
                "recent_24h_mean_ret": slice24["ret_1h"].mean() if len(slice24) else np.nan,
                "recent_24h_std_ret": slice24["ret_1h"].std(ddof=1) if len(slice24) > 1 else np.nan,

                "recent_24h_low_break_distance": (
                    (row["close"] - row.get("swing_low_24", np.nan)) / atr_now
                    if pd.notna(atr_now) and atr_now != 0 else np.nan
                ),
                "recent_24h_high_distance": (
                    (row.get("swing_high_24", np.nan) - row["close"]) / atr_now
                    if pd.notna(atr_now) and atr_now != 0 else np.nan
                ),

                "hour": row.get("hour", np.nan),
                "dayofweek": row.get("dayofweek", np.nan),
                "is_us_session": row.get("is_us_session", np.nan),

                "trigger_atr_breakdown": 1 if trig_name == "atr_breakdown" else 0,
                "trigger_rsi_rollover": 1 if trig_name == "rsi_rollover" else 0,

                "prev_bar_red": 1 if prev1["close"] < prev1["open"] else 0,
                "prev_bar_range": prev1.get("bar_range", np.nan),
                "prev_bar_rsi14": prev1.get("rsi14_h1", np.nan),
                "prev_bar_atr_ratio": prev1.get("atr_ratio", np.nan),
            })

    return pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)


def apply_proxy_label(exec_df, entry_idx, max_hold=24, stop_mult=1.0, target_mult=1.5):
    if entry_idx >= len(exec_df):
        return np.nan
    signal_idx = entry_idx - 1
    if signal_idx < 0:
        return np.nan

    signal_row = exec_df.iloc[signal_idx]
    entry_row = exec_df.iloc[entry_idx]
    atr_val = signal_row["atr14"]
    if pd.isna(atr_val) or atr_val <= 0:
        return np.nan

    entry_price = entry_row["open"]
    stop_price = entry_price + stop_mult * atr_val
    target_price = entry_price - target_mult * atr_val
    end_idx = min(entry_idx + max_hold, len(exec_df) - 1)

    for j in range(entry_idx, end_idx + 1):
        row = exec_df.iloc[j]
        if row["high"] >= stop_price:
            return 0
        if row["low"] <= target_price:
            return 1

    exit_price = exec_df.iloc[end_idx]["close"]
    return int(exit_price < entry_price)


def walk_forward_rf_expanding(signals_df, feature_cols):
    df = signals_df.copy().sort_values("signal_time").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df["signal_time"] = pd.to_datetime(df["signal_time"])
    df = df[df["signal_time"] >= pd.Timestamp(WF_START_DATE)].reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    all_preds = []
    wf_rows = []

    split_start = pd.Timestamp(TRAIN_START_DATE)
    test_start = split_start + pd.DateOffset(months=INITIAL_TRAIN_MONTHS)
    end_date = df["signal_time"].max()

    while test_start <= end_date:
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)

        train_mask = (df["signal_time"] >= split_start) & (df["signal_time"] < test_start)
        test_mask = (df["signal_time"] >= test_start) & (df["signal_time"] < test_end)

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        if len(test_idx) == 0:
            test_start = test_start + pd.DateOffset(months=TEST_MONTHS)
            continue

        purged_train_idx = train_idx[train_idx <= (test_idx.min() - EMBARGO_BARS)]
        if len(purged_train_idx) < MIN_TRAIN_ROWS or len(test_idx) < MIN_TEST_ROWS:
            test_start = test_start + pd.DateOffset(months=TEST_MONTHS)
            continue

        train_df = df.loc[purged_train_idx].copy()
        test_df = df.loc[test_idx].copy()

        y_train = train_df["proxy_label"].astype(int)
        y_test = test_df["proxy_label"].astype(int)
        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            test_start = test_start + pd.DateOffset(months=TEST_MONTHS)
            continue

        clf = RandomForestClassifier(
            n_estimators=RF_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=RF_RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_auc = roc_auc_score(y_test, proba)
        fold_acc = accuracy_score(y_test, pred)

        fold_pred = test_df.copy()
        fold_pred["rf_prob"] = proba
        fold_pred["pred_label"] = pred
        all_preds.append(fold_pred)

        wf_rows.append({
            "train_start": split_start,
            "train_end": test_start,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_pos_rate": float(y_test.mean()),
            "auc": float(fold_auc),
            "accuracy": float(fold_acc),
        })

        test_start = test_start + pd.DateOffset(months=TEST_MONTHS)

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    wf_df = pd.DataFrame(wf_rows)
    return pred_df, wf_df


def should_tighten(row, tighten_mode, rsi_threshold, atr_stretch_threshold):
    rsi_hit = pd.notna(row.get("d1_rsi10", np.nan)) and row.get("d1_rsi10", np.nan) < rsi_threshold
    atr_hit = pd.notna(row.get("d1_range_atr_ratio", np.nan)) and row.get("d1_range_atr_ratio", np.nan) >= atr_stretch_threshold

    if tighten_mode == "rsi":
        return rsi_hit
    if tighten_mode == "atr_exhaust":
        return atr_hit
    if tighten_mode == "either":
        return rsi_hit or atr_hit
    return False


def simulate_trade(exec_df, sig, tighten_mode, rsi_threshold, atr_stretch_threshold, tighten_factor, force_exit):
    signal_idx = int(sig["signal_idx"])
    entry_idx = int(sig["entry_idx"])

    if entry_idx >= len(exec_df):
        return None

    signal_bar = exec_df.iloc[signal_idx]
    entry_bar = exec_df.iloc[entry_idx]
    entry_price = float(entry_bar["open"])

    stop_pts = min(BASE_STOP_VALUE, MAX_STOP_POINTS)
    risk_usd = stop_pts * MNQ_POINT_VALUE * CONTRACTS
    if risk_usd > MAX_LOSS_PER_TRADE_USD + 1e-9:
        return None

    target_pts = BASE_TARGET_VALUE
    initial_stop = entry_price + stop_pts
    current_stop = initial_stop
    target_price = entry_price - target_pts

    best_low = entry_price
    end_idx = min(entry_idx + MAX_HOLD_BARS, len(exec_df) - 1)
    exit_price = float(exec_df.iloc[end_idx]["close"])
    exit_time = exec_df.iloc[end_idx]["time"]
    exit_reason = "time"
    bars_held = end_idx - entry_idx + 1

    for j in range(entry_idx, end_idx + 1):
        row = exec_df.iloc[j]
        low_j = float(row["low"])
        high_j = float(row["high"])

        best_low = min(best_low, low_j)
        favorable_pts = entry_price - best_low

        if should_tighten(row, tighten_mode, rsi_threshold, atr_stretch_threshold):
            if force_exit:
                exit_price = float(row["close"])
                exit_time = row["time"]
                exit_reason = "tighten_force_exit"
                bars_held = j - entry_idx + 1
                break
            tightened_stop = entry_price - favorable_pts * tighten_factor
            current_stop = min(current_stop, tightened_stop)

        if high_j >= current_stop:
            exit_price = current_stop
            exit_time = row["time"]
            exit_reason = "stop" if current_stop == initial_stop else "tightened_stop"
            bars_held = j - entry_idx + 1
            break

        if low_j <= target_price:
            exit_price = target_price
            exit_time = row["time"]
            exit_reason = "target"
            bars_held = j - entry_idx + 1
            break

    pnl_usd = (entry_price - exit_price) * MNQ_POINT_VALUE * CONTRACTS
    ret_pct = ((entry_price / exit_price) - 1.0) * 100.0 if exit_price != 0 else np.nan
    r_multiple = (entry_price - exit_price) / stop_pts if stop_pts > 0 else np.nan

    return {
        "signal_time": sig["signal_time"],
        "trigger": sig["trigger"],
        "entry_time": sig["entry_time"],
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "pnl_usd": pnl_usd,
        "return_pct": ret_pct,
        "r_multiple": r_multiple,
        "rf_prob": sig["rf_prob"],
        "risk_usd": risk_usd,
        "tighten_mode": tighten_mode,
        "rsi_threshold": rsi_threshold,
        "atr_stretch_threshold": atr_stretch_threshold,
        "tighten_factor": tighten_factor,
        "force_exit": force_exit,
    }


def summarize_combo(trades_df, combo_name):
    if trades_df.empty:
        return None, None

    bt = trades_df.sort_values("entry_time").copy().reset_index(drop=True)
    bt["capital_before"] = 0.0
    bt["capital_after"] = 0.0

    capital = INITIAL_CAPITAL
    for i in range(len(bt)):
        bt.loc[i, "capital_before"] = capital
        capital += bt.loc[i, "pnl_usd"]
        bt.loc[i, "capital_after"] = capital

    winners = int((bt["pnl_usd"] > 0).sum())
    losers = int((bt["pnl_usd"] < 0).sum())
    total_pnl = float(bt["pnl_usd"].sum())
    equity_curve = [INITIAL_CAPITAL] + bt["capital_after"].tolist()
    max_dd_usd = float(max_drawdown_from_equity(equity_curve))
    max_dd_pct = max_dd_usd / INITIAL_CAPITAL * 100.0

    post = bt[bt["entry_time"] >= pd.Timestamp("2024-01-01")].copy()

    summary = {
        "combo": combo_name,
        "trades": int(len(bt)),
        "final_capital_usd": INITIAL_CAPITAL + total_pnl,
        "total_pnl_usd": total_pnl,
        "total_return_pct": total_pnl / INITIAL_CAPITAL * 100.0,
        "win_rate_pct": winners / len(bt) * 100.0 if len(bt) else np.nan,
        "profit_factor": float(profit_factor_from_pnl(bt["pnl_usd"])),
        "sharpe": float(sharpe_from_returns(bt["pnl_usd"] / INITIAL_CAPITAL)),
        "max_drawdown_usd": max_dd_usd,
        "max_drawdown_pct": max_dd_pct,
        "avg_winner_usd": float(bt.loc[bt["pnl_usd"] > 0, "pnl_usd"].mean()) if winners else np.nan,
        "avg_loser_usd": float(bt.loc[bt["pnl_usd"] < 0, "pnl_usd"].mean()) if losers else np.nan,
        "avg_pnl_per_trade_usd": float(bt["pnl_usd"].mean()),
        "avg_bars_held": float(bt["bars_held"].mean()),
        "post_2024_trades": int(len(post)),
        "post_2024_total_pnl_usd": float(post["pnl_usd"].sum()) if len(post) else np.nan,
        "post_2024_return_pct": float(post["pnl_usd"].sum() / INITIAL_CAPITAL * 100.0) if len(post) else np.nan,
        "post_2024_win_rate_pct": float((post["pnl_usd"] > 0).mean() * 100.0) if len(post) else np.nan,
        "post_2024_profit_factor": float(profit_factor_from_pnl(post["pnl_usd"])) if len(post) else np.nan,
        "post_2024_sharpe": float(sharpe_from_returns(post["pnl_usd"] / INITIAL_CAPITAL)) if len(post) else np.nan,
        "post_2024_max_drawdown_usd": float(max_drawdown_from_equity([INITIAL_CAPITAL] + post["capital_after"].tolist())) if len(post) else np.nan,
    }
    return summary, bt


def score_row(row):
    if row["trades"] < 30:
        return -1e9
    vals = [
        row["sharpe"] * 25 if pd.notna(row["sharpe"]) else -50,
        row["profit_factor"] * 20 if pd.notna(row["profit_factor"]) else -50,
        row["total_return_pct"] * 0.25 if pd.notna(row["total_return_pct"]) else -50,
        row["post_2024_return_pct"] * 0.20 if pd.notna(row["post_2024_return_pct"]) else -20,
        row["post_2024_sharpe"] * 20 if pd.notna(row["post_2024_sharpe"]) else -20,
        row["max_drawdown_pct"] * 0.35 if pd.notna(row["max_drawdown_pct"]) else -20,
    ]
    return float(sum(vals))


if not mt5.initialize():
    raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

symbol = resolve_symbol(mt5, SYMBOL_CANDIDATES)
rates_d1 = mt5.copy_rates_from_pos(symbol, TIMEFRAME_REGIME, 0, D1_BARS)
rates_h1 = mt5.copy_rates_from_pos(symbol, TIMEFRAME_EXEC, 0, H1_BARS)
mt5.shutdown()

if rates_d1 is None or len(rates_d1) == 0:
    raise RuntimeError("No D1 bars returned after resolving symbol.")
if rates_h1 is None or len(rates_h1) == 0:
    raise RuntimeError("No H1 bars returned after resolving symbol.")

df_d1 = pd.DataFrame(rates_d1)
df_d1["time"] = pd.to_datetime(df_d1["time"], unit="s")
df_d1 = df_d1.sort_values("time").reset_index(drop=True)

df_h1 = pd.DataFrame(rates_h1)
df_h1["time"] = pd.to_datetime(df_h1["time"], unit="s")
df_h1 = df_h1.sort_values("time").reset_index(drop=True)

regime_df = prepare_regime_df(df_d1)
exec_df = prepare_exec_df(df_h1, regime_df)
signals_df = build_signal_rows(exec_df)

signals_df["proxy_label"] = signals_df["entry_idx"].apply(lambda idx: apply_proxy_label(exec_df, int(idx)))
signals_df = signals_df.dropna(subset=["proxy_label"]).copy()
signals_df["proxy_label"] = signals_df["proxy_label"].astype(int)

feature_cols = [
    "regime_age", "d1_rsi14", "d1_rsi10", "d1_range_atr_ratio",
    "d1_close_vs_ema_atr", "d1_ema_slope_atr", "d1_ret_5",
    "h1_atr14", "h1_rsi14", "h1_ret_1h", "h1_ret_6h", "h1_ret_24h",
    "bar_range", "body_size", "atr_ratio", "tickvol_ratio",
    "dist_to_recent_low_atr", "dist_to_recent_high_atr", "dist_to_h1_ema_atr",
    "recent_6h_mean_ret", "recent_6h_std_ret", "recent_24h_mean_ret", "recent_24h_std_ret",
    "recent_24h_low_break_distance", "recent_24h_high_distance",
    "hour", "dayofweek", "is_us_session",
    "trigger_atr_breakdown", "trigger_rsi_rollover",
    "prev_bar_red", "prev_bar_range", "prev_bar_rsi14", "prev_bar_atr_ratio",
]

pred_df, wf_df = walk_forward_rf_expanding(signals_df, feature_cols)
wf_df.to_csv(WF_SUMMARY_CSV, index=False)

if pred_df.empty:
    print("No OOS predictions generated.")
    raise SystemExit

filtered = pred_df[pred_df["rf_prob"] >= RF_THRESHOLD].copy()
filtered = filtered.sort_values("entry_time").reset_index(drop=True)

print("\n" + "=" * 110)
print("TIGHTENING / EXHAUSTION OPTIMIZER")
print("=" * 110)
print(f"Symbol: {symbol}")
print(f"Filtered signals: {len(filtered)}")
print(f"Base exit: stop={BASE_STOP_VALUE} fixed points, target={BASE_TARGET_VALUE} fixed points")
print(f"Max loss per trade: ${MAX_LOSS_PER_TRADE_USD:.2f}")
print(f"Walk-forward folds: {len(wf_df)}")
if not wf_df.empty:
    print(f"Mean AUC: {wf_df['auc'].mean():.4f}")
    print(f"Mean Accuracy: {wf_df['accuracy'].mean():.4f}")

results = []
best_summary = None
best_trades = None
best_score = -1e18

for mode_cfg, rsi_thr, atr_thr, tighten_factor in product(TIGHTEN_MODES, RSI_THRESHOLDS, ATR_STRETCH_THRESHOLDS, TIGHTEN_FACTORS):
    combo_name = (
        f"{mode_cfg['tighten_mode']}_"
        f"rsi{rsi_thr}_"
        f"atr{atr_thr}_"
        f"tf{tighten_factor}_"
        f"force{int(mode_cfg['force_exit'])}"
    )

    sim_rows = []
    for _, sig in filtered.iterrows():
        tr = simulate_trade(
            exec_df,
            sig,
            tighten_mode=mode_cfg["tighten_mode"],
            rsi_threshold=rsi_thr,
            atr_stretch_threshold=atr_thr,
            tighten_factor=tighten_factor,
            force_exit=mode_cfg["force_exit"],
        )
        if tr is not None:
            sim_rows.append(tr)

    if not sim_rows:
        continue

    sim_df = pd.DataFrame(sim_rows)
    summary, bt = summarize_combo(sim_df, combo_name)
    if summary is None:
        continue

    summary.update({
        "tighten_mode": mode_cfg["tighten_mode"],
        "rsi_threshold": rsi_thr,
        "atr_stretch_threshold": atr_thr,
        "tighten_factor": tighten_factor,
        "force_exit": mode_cfg["force_exit"],
    })
    summary["score"] = score_row(summary)
    results.append(summary)

    if summary["score"] > best_score:
        best_score = summary["score"]
        best_summary = summary.copy()
        best_trades = bt.copy()

results_df = pd.DataFrame(results).sort_values(
    ["score", "post_2024_return_pct", "sharpe", "profit_factor"],
    ascending=[False, False, False, False]
).reset_index(drop=True)

results_df.to_csv(RESULTS_CSV, index=False)
if best_trades is not None:
    best_trades.to_csv(BEST_TRADES_CSV, index=False)

print("\n" + "-" * 110)
print("TOP 20 TIGHTENING CONFIGURATIONS")
print("-" * 110)
show_cols = [
    "score",
    "tighten_mode", "rsi_threshold", "atr_stretch_threshold", "tighten_factor", "force_exit",
    "trades", "total_return_pct", "profit_factor", "sharpe", "max_drawdown_pct",
    "post_2024_trades", "post_2024_return_pct", "post_2024_profit_factor", "post_2024_sharpe"
]
print(results_df[show_cols].head(20).to_string(index=False))

if best_summary is not None:
    print("\n" + "-" * 110)
    print("BEST TIGHTENING CONFIGURATION")
    print("-" * 110)
    for k, v in best_summary.items():
        print(f"{k}: {v}")

if best_trades is not None and not best_trades.empty:
    print("\n" + "-" * 110)
    print("LAST 10 TRADES FOR BEST TIGHTENING CONFIG")
    print("-" * 110)
    last10 = best_trades.sort_values("entry_time").tail(10).iloc[::-1]
    for _, row in last10.iterrows():
        outcome = "WIN" if row["pnl_usd"] > 0 else "LOSS"
        print(f"\n{row['entry_time']} | {row['trigger']} | {outcome}")
        print(f"  Entry: {row['entry_price']:.2f}")
        print(f"  Exit:  {row['exit_price']:.2f} ({row['exit_reason']}) at {row['exit_time']}")
        print(f"  P&L:   ${row['pnl_usd']:,.2f} | R: {row['r_multiple']:.2f} | Bars: {int(row['bars_held'])}")
        print(f"  RF:    {row['rf_prob']:.3f} | Risk: ${row['risk_usd']:.2f}")

print("\nSaved tightening optimizer results to:", RESULTS_CSV)
print("Saved best trades to:", BEST_TRADES_CSV)
print("Saved walk-forward diagnostics to:", WF_SUMMARY_CSV)
print("=" * 110 + "\n")
