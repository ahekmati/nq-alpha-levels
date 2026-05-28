from mt5linux import MetaTrader5
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from math import sqrt

SYMBOL_CANDIDATES = ["@MNQ", "MNQ", "MNQM26", "MNQU26", "MNQZ26"]
TIMEFRAME_REGIME = MetaTrader5.TIMEFRAME_D1
TIMEFRAME_EXEC = MetaTrader5.TIMEFRAME_H1

D1_BARS = 5000
H1_BARS = 50000

EMA_PERIOD = 100
SLOPE_LOOKBACK = 10
ATR_PERIOD = 14
RSI_PERIOD = 14
BREAKOUT_LOOKBACK = 5

REGIME_RSI_MAX = 60
ATR_EXPANSION_MIN = 1.2
USE_FALLING_EMA_ONLY = True
USE_RSI_CAP = True

LABEL_HORIZON_BARS = 24
STOP_ATR_MULT = 1.0
TARGET_ATR_MULT = 1.5

WF_START_DATE = "2020-01-01"
TRAIN_MONTHS = 6
TEST_MONTHS = 2
EMBARGO_BARS = 24 * 3
MIN_TRAIN_ROWS = 40
MIN_TEST_ROWS = 12

RF_ESTIMATORS = 300
RF_MAX_DEPTH = 5
RF_MIN_SAMPLES_LEAF = 10
RF_RANDOM_STATE = 42

THRESHOLDS = [0.50, 0.55, 0.60]

FEATURES_CSV = "ema100_bear_rf_features.csv"
PREDICTIONS_CSV = "ema100_bear_rf_walkforward_predictions.csv"
IMPORTANCE_CSV = "ema100_bear_rf_feature_importance.csv"
WF_SUMMARY_CSV = "ema100_bear_rf_walkforward_summary.csv"
STRATEGY_SUMMARY_CSV = "ema100_bear_rf_strategy_comparison.csv"
YEARLY_SUMMARY_CSV = "ema100_bear_rf_strategy_yearly_stats.csv"
STRATEGY_TRADES_CSV = "ema100_bear_rf_strategy_trades.csv"

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


def max_drawdown(equity_curve):
    eq = pd.Series(equity_curve).dropna()
    if eq.empty:
        return np.nan
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return dd.min()


def profit_factor_from_returns(ret_series_pct):
    s = pd.Series(ret_series_pct).dropna()
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
        "time", "bear_permission", "regime_age", "ema100", "ema_slope_n", "rsi14",
        "atr14_d1", "d1_close_vs_ema_atr", "d1_ema_slope_atr", "d1_ret_5"
    ]].copy()

    regime_map = regime_map.rename(columns={
        "time": "d1_time",
        "ema100": "d1_ema100",
        "rsi14": "d1_rsi14",
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


def apply_triple_barrier_short(df, signal_idx, stop_mult=1.0, target_mult=1.5, max_hold=24):
    if signal_idx + 1 >= len(df):
        return None

    signal_row = df.iloc[signal_idx]
    entry_row = df.iloc[signal_idx + 1]
    atr_val = signal_row["atr14"]

    if pd.isna(atr_val) or atr_val <= 0:
        return None

    entry_price = entry_row["open"]
    stop_price = entry_price + stop_mult * atr_val
    target_price = entry_price - target_mult * atr_val

    end_idx = min(signal_idx + 1 + max_hold, len(df) - 1)
    outcome = 0
    exit_price = df.iloc[end_idx]["close"]
    exit_time = df.iloc[end_idx]["time"]
    exit_reason = "time"
    bars_held = end_idx - (signal_idx + 1) + 1

    for j in range(signal_idx + 1, end_idx + 1):
        row = df.iloc[j]
        if row["high"] >= stop_price:
            outcome = 0
            exit_price = stop_price
            exit_time = row["time"]
            exit_reason = "stop"
            bars_held = j - (signal_idx + 1) + 1
            break
        if row["low"] <= target_price:
            outcome = 1
            exit_price = target_price
            exit_time = row["time"]
            exit_reason = "target"
            bars_held = j - (signal_idx + 1) + 1
            break

    ret_pct = (entry_price / exit_price) - 1.0
    r_mult = (entry_price - exit_price) / (stop_mult * atr_val)

    return {
        "entry_time": entry_row["time"],
        "entry_price": entry_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "label": outcome,
        "return_pct": ret_pct * 100.0,
        "r_multiple": r_mult,
    }


def build_feature_rows(df):
    rows = []
    trigger_specs = [
        ("trig_atr_breakdown", "atr_breakdown"),
        ("trig_rsi_rollover", "rsi_rollover"),
    ]

    for trig_col, trig_name in trigger_specs:
        signal_idx = np.where(df[trig_col].fillna(False).values)[0]

        for i in signal_idx:
            labeled = apply_triple_barrier_short(df, i, STOP_ATR_MULT, TARGET_ATR_MULT, LABEL_HORIZON_BARS)
            if labeled is None:
                continue

            row = df.iloc[i]
            prev1 = df.iloc[i - 1] if i - 1 >= 0 else row
            prev6_start = max(0, i - 6)
            prev24_start = max(0, i - 24)
            slice6 = df.iloc[prev6_start:i]
            slice24 = df.iloc[prev24_start:i]
            atr_now = row.get("atr14", np.nan)

            feature_row = {
                "signal_time": row["time"],
                "trigger": trig_name,
                "label": labeled["label"],
                "entry_time": labeled["entry_time"],
                "exit_time": labeled["exit_time"],
                "exit_reason": labeled["exit_reason"],
                "bars_held": labeled["bars_held"],
                "return_pct": labeled["return_pct"],
                "r_multiple": labeled["r_multiple"],

                "regime_age": row.get("regime_age", np.nan),
                "d1_rsi14": row.get("d1_rsi14", np.nan),
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
            }
            rows.append(feature_row)

    out = pd.DataFrame(rows).sort_values("signal_time").reset_index(drop=True)
    return out


def walk_forward_rf(features_df, feature_cols):
    df = features_df.copy().sort_values("signal_time").reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["signal_time"] = pd.to_datetime(df["signal_time"])
    df = df[df["signal_time"] >= pd.Timestamp(WF_START_DATE)].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_preds = []
    importances = []
    summary_rows = []

    split_start = pd.Timestamp(WF_START_DATE)
    end_date = df["signal_time"].max()

    while split_start < end_date:
        train_end = split_start + pd.DateOffset(months=TRAIN_MONTHS)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=TEST_MONTHS)

        train_mask = (df["signal_time"] >= split_start) & (df["signal_time"] < train_end)
        test_mask = (df["signal_time"] >= test_start) & (df["signal_time"] < test_end)

        train_idx = df.index[train_mask]
        test_idx = df.index[test_mask]

        if len(test_idx) == 0:
            split_start = split_start + pd.DateOffset(months=TEST_MONTHS)
            continue

        if len(train_idx) < MIN_TRAIN_ROWS or len(test_idx) < MIN_TEST_ROWS:
            split_start = split_start + pd.DateOffset(months=TEST_MONTHS)
            continue

        purged_train_idx = train_idx[train_idx <= (test_idx.min() - EMBARGO_BARS)]
        if len(purged_train_idx) < MIN_TRAIN_ROWS:
            split_start = split_start + pd.DateOffset(months=TEST_MONTHS)
            continue

        train_df = df.loc[purged_train_idx].copy()
        test_df = df.loc[test_idx].copy()

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_train = train_df["label"].astype(int)
        X_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        y_test = test_df["label"].astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            split_start = split_start + pd.DateOffset(months=TEST_MONTHS)
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

        fold_pred = test_df[
            [
                "signal_time",
                "trigger",
                "label",
                "return_pct",
                "r_multiple",
                "bars_held",
                "entry_time",
                "exit_time",
                "exit_reason",
            ]
        ].copy()
        fold_pred["prob_target_hit_first"] = proba
        fold_pred["pred_label"] = pred
        fold_pred["fold_train_start"] = split_start
        fold_pred["fold_train_end"] = train_end
        fold_pred["fold_test_start"] = test_start
        fold_pred["fold_test_end"] = test_end
        all_preds.append(fold_pred)

        fold_imp = pd.DataFrame({
            "feature": feature_cols,
            "importance": clf.feature_importances_,
            "fold_test_start": test_start,
            "fold_test_end": test_end,
        })
        importances.append(fold_imp)

        summary_rows.append({
            "train_start": split_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "test_pos_rate": float(y_test.mean()),
            "auc": float(fold_auc),
            "accuracy": float(fold_acc),
        })

        split_start = split_start + pd.DateOffset(months=TEST_MONTHS)

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    imp_df = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return pred_df, imp_df, summary_df


def summarize_strategy(trades_df, name):
    if trades_df.empty:
        return {
            "strategy": name,
            "trades": 0,
            "win_rate_pct": np.nan,
            "avg_return_pct": np.nan,
            "avg_r": np.nan,
            "profit_factor": np.nan,
            "total_return_pct": np.nan,
            "sharpe": np.nan,
            "max_drawdown_pct": np.nan,
            "avg_bars_held": np.nan,
            "trades_per_year": np.nan,
        }

    temp = trades_df.sort_values("entry_time").copy()
    eq = (1.0 + temp["return_pct"].fillna(0) / 100.0).cumprod()

    years = temp["entry_time"].dt.year.nunique()
    years = years if years > 0 else np.nan

    return {
        "strategy": name,
        "trades": int(len(temp)),
        "win_rate_pct": float((temp["return_pct"] > 0).mean() * 100.0),
        "avg_return_pct": float(temp["return_pct"].mean()),
        "avg_r": float(temp["r_multiple"].mean()),
        "profit_factor": float(profit_factor_from_returns(temp["return_pct"])),
        "total_return_pct": float((eq.iloc[-1] - 1.0) * 100.0),
        "sharpe": float(sharpe_from_returns(temp["return_pct"] / 100.0)),
        "max_drawdown_pct": float(max_drawdown(eq) * 100.0),
        "avg_bars_held": float(temp["bars_held"].mean()),
        "trades_per_year": float(len(temp) / years) if pd.notna(years) and years > 0 else np.nan,
    }


def yearly_stats(trades_df, name):
    if trades_df.empty:
        return pd.DataFrame()

    temp = trades_df.sort_values("entry_time").copy()
    temp["year"] = temp["entry_time"].dt.year

    rows = []
    for year, grp in temp.groupby("year"):
        eq = (1.0 + grp["return_pct"].fillna(0) / 100.0).cumprod()
        rows.append({
            "strategy": name,
            "year": int(year),
            "trades": int(len(grp)),
            "win_rate_pct": float((grp["return_pct"] > 0).mean() * 100.0),
            "avg_return_pct": float(grp["return_pct"].mean()),
            "avg_r": float(grp["r_multiple"].mean()),
            "profit_factor": float(profit_factor_from_returns(grp["return_pct"])),
            "total_return_pct": float((eq.iloc[-1] - 1.0) * 100.0),
            "sharpe": float(sharpe_from_returns(grp["return_pct"] / 100.0)),
            "max_drawdown_pct": float(max_drawdown(eq) * 100.0),
            "avg_bars_held": float(grp["bars_held"].mean()),
        })
    return pd.DataFrame(rows)


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
features_df = build_feature_rows(exec_df)

feature_cols = [
    "regime_age",
    "d1_rsi14",
    "d1_close_vs_ema_atr",
    "d1_ema_slope_atr",
    "d1_ret_5",
    "h1_atr14",
    "h1_rsi14",
    "h1_ret_1h",
    "h1_ret_6h",
    "h1_ret_24h",
    "bar_range",
    "body_size",
    "atr_ratio",
    "tickvol_ratio",
    "dist_to_recent_low_atr",
    "dist_to_recent_high_atr",
    "dist_to_h1_ema_atr",
    "recent_6h_mean_ret",
    "recent_6h_std_ret",
    "recent_24h_mean_ret",
    "recent_24h_std_ret",
    "recent_24h_low_break_distance",
    "recent_24h_high_distance",
    "hour",
    "dayofweek",
    "is_us_session",
    "trigger_atr_breakdown",
    "trigger_rsi_rollover",
    "prev_bar_red",
    "prev_bar_range",
    "prev_bar_rsi14",
    "prev_bar_atr_ratio",
]

pred_df, imp_df, wf_summary_df = walk_forward_rf(features_df, feature_cols)

features_df.to_csv(FEATURES_CSV, index=False)

if not pred_df.empty:
    pred_df["entry_time"] = pd.to_datetime(pred_df["entry_time"])
    pred_df["exit_time"] = pd.to_datetime(pred_df["exit_time"])
    pred_df = pred_df.sort_values("entry_time").reset_index(drop=True)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

if not imp_df.empty:
    avg_imp = imp_df.groupby("feature", as_index=False)["importance"].mean().sort_values("importance", ascending=False)
    avg_imp.to_csv(IMPORTANCE_CSV, index=False)
else:
    avg_imp = pd.DataFrame()

wf_summary_df.to_csv(WF_SUMMARY_CSV, index=False)

strategy_trade_frames = []
strategy_summary_rows = []
yearly_frames = []

if not pred_df.empty:
    raw_trades = pred_df.copy()
    raw_trades["strategy"] = "raw_oos_all"
    strategy_trade_frames.append(raw_trades)
    strategy_summary_rows.append(summarize_strategy(raw_trades, "raw_oos_all"))
    yearly_frames.append(yearly_stats(raw_trades, "raw_oos_all"))

    for thr in THRESHOLDS:
        subset = pred_df[pred_df["prob_target_hit_first"] >= thr].copy()
        name = f"rf_ge_{thr:.2f}"
        subset["strategy"] = name
        strategy_trade_frames.append(subset)
        strategy_summary_rows.append(summarize_strategy(subset, name))
        yearly_frames.append(yearly_stats(subset, name))

strategy_trades_df = pd.concat(strategy_trade_frames, ignore_index=True) if strategy_trade_frames else pd.DataFrame()
strategy_summary_df = pd.DataFrame(strategy_summary_rows)
yearly_summary_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()

strategy_summary_df.to_csv(STRATEGY_SUMMARY_CSV, index=False)
if not strategy_trades_df.empty:
    strategy_trades_df.to_csv(STRATEGY_TRADES_CSV, index=False)
if not yearly_summary_df.empty:
    yearly_summary_df.to_csv(YEARLY_SUMMARY_CSV, index=False)

print(f"\nResolved symbol: {symbol}")
print(f"D1 bars analyzed: {len(regime_df)}")
print(f"H1 bars analyzed: {len(exec_df)}")
print(f"Feature rows built: {len(features_df)}")

if not features_df.empty:
    print(f"Label positive rate: {features_df['label'].mean() * 100:.2f}%")
    print("\n=== Trigger Mix ===")
    print(features_df["trigger"].value_counts().to_string())

if wf_summary_df.empty:
    print("\nNo valid walk-forward folds were created with current settings.")
else:
    print("\n=== Walk-Forward Summary ===")
    print(wf_summary_df.to_string(index=False))
    print(f"\nMean AUC: {wf_summary_df['auc'].mean():.4f}")
    print(f"Mean Accuracy: {wf_summary_df['accuracy'].mean():.4f}")

if not avg_imp.empty:
    print("\n=== Average Feature Importance ===")
    print(avg_imp.head(15).to_string(index=False))

if not strategy_summary_df.empty:
    print("\n=== Strategy Comparison ===")
    print(strategy_summary_df.to_string(index=False))

if not yearly_summary_df.empty:
    print("\n=== Yearly Strategy Stats ===")
    print(yearly_summary_df.to_string(index=False))

print(f"\nSaved features to: {FEATURES_CSV}")
if not pred_df.empty:
    print(f"Saved walk-forward predictions to: {PREDICTIONS_CSV}")
if not avg_imp.empty:
    print(f"Saved feature importance to: {IMPORTANCE_CSV}")
print(f"Saved walk-forward summary to: {WF_SUMMARY_CSV}")
print(f"Saved strategy comparison to: {STRATEGY_SUMMARY_CSV}")
if not strategy_trades_df.empty:
    print(f"Saved strategy trades to: {STRATEGY_TRADES_CSV}")
if not yearly_summary_df.empty:
    print(f"Saved yearly strategy stats to: {YEARLY_SUMMARY_CSV}")
