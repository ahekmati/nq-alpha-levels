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

RF_THRESHOLD = 0.50

INITIAL_CAPITAL = 5000.0
MNQ_POINT_VALUE = 2.0
CONTRACTS = 1

BACKTEST_TRADES_CSV = "ema100_bear_backtest_trades.csv"
BACKTEST_SUMMARY_CSV = "ema100_bear_backtest_summary.csv"

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
    pnl_usd = (entry_price - exit_price) * MNQ_POINT_VALUE * CONTRACTS

    return {
        "entry_time": entry_row["time"],
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "bars_held": bars_held,
        "label": outcome,
        "return_pct": ret_pct * 100.0,
        "r_multiple": r_mult,
        "pnl_usd": pnl_usd,
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
                "entry_price": labeled["entry_price"],
                "stop_price": labeled["stop_price"],
                "target_price": labeled["target_price"],
                "exit_time": labeled["exit_time"],
                "exit_price": labeled["exit_price"],
                "exit_reason": labeled["exit_reason"],
                "bars_held": labeled["bars_held"],
                "return_pct": labeled["return_pct"],
                "r_multiple": labeled["r_multiple"],
                "pnl_usd": labeled["pnl_usd"],

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
        return pd.DataFrame()

    df["signal_time"] = pd.to_datetime(df["signal_time"])
    df = df[df["signal_time"] >= pd.Timestamp(WF_START_DATE)].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    all_preds = []
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

        fold_pred = test_df[[
            "signal_time", "trigger", "label", "entry_time", "entry_price", "stop_price",
            "target_price", "exit_time", "exit_price", "exit_reason", "bars_held",
            "return_pct", "r_multiple", "pnl_usd"
        ]].copy()
        fold_pred["rf_prob"] = proba
        all_preds.append(fold_pred)

        split_start = split_start + pd.DateOffset(months=TEST_MONTHS)

    pred_df = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return pred_df


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
    "regime_age", "d1_rsi14", "d1_close_vs_ema_atr", "d1_ema_slope_atr", "d1_ret_5",
    "h1_atr14", "h1_rsi14", "h1_ret_1h", "h1_ret_6h", "h1_ret_24h",
    "bar_range", "body_size", "atr_ratio", "tickvol_ratio",
    "dist_to_recent_low_atr", "dist_to_recent_high_atr", "dist_to_h1_ema_atr",
    "recent_6h_mean_ret", "recent_6h_std_ret", "recent_24h_mean_ret", "recent_24h_std_ret",
    "recent_24h_low_break_distance", "recent_24h_high_distance",
    "hour", "dayofweek", "is_us_session",
    "trigger_atr_breakdown", "trigger_rsi_rollover",
    "prev_bar_red", "prev_bar_range", "prev_bar_rsi14", "prev_bar_atr_ratio",
]

pred_df = walk_forward_rf(features_df, feature_cols)

if pred_df.empty:
    print("No OOS predictions generated. Cannot run backtest.")
else:
    pred_df["entry_time"] = pd.to_datetime(pred_df["entry_time"])
    pred_df["exit_time"] = pd.to_datetime(pred_df["exit_time"])
    pred_df = pred_df.sort_values("entry_time").reset_index(drop=True)

    filtered_trades = pred_df[pred_df["rf_prob"] >= RF_THRESHOLD].copy()

    if filtered_trades.empty:
        print(f"No trades met RF threshold >= {RF_THRESHOLD}. Cannot run backtest.")
    else:
        filtered_trades = filtered_trades.sort_values("entry_time").reset_index(drop=True)
        filtered_trades["capital_before"] = 0.0
        filtered_trades["capital_after"] = 0.0

        capital = INITIAL_CAPITAL
        for i in range(len(filtered_trades)):
            filtered_trades.loc[i, "capital_before"] = capital
            pnl = filtered_trades.loc[i, "pnl_usd"]
            capital += pnl
            filtered_trades.loc[i, "capital_after"] = capital

        filtered_trades.to_csv(BACKTEST_TRADES_CSV, index=False)

        total_trades = len(filtered_trades)
        winners = (filtered_trades["pnl_usd"] > 0).sum()
        losers = (filtered_trades["pnl_usd"] < 0).sum()
        win_rate = (winners / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = filtered_trades[filtered_trades["pnl_usd"] > 0]["pnl_usd"].sum()
        gross_loss = -filtered_trades[filtered_trades["pnl_usd"] < 0]["pnl_usd"].sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else np.nan

        total_pnl = filtered_trades["pnl_usd"].sum()
        final_capital = INITIAL_CAPITAL + total_pnl
        total_return_pct = (total_pnl / INITIAL_CAPITAL) * 100.0

        equity_curve = [INITIAL_CAPITAL] + filtered_trades["capital_after"].tolist()
        max_dd = max_drawdown_from_equity(equity_curve)
        max_dd_pct = (max_dd / INITIAL_CAPITAL) * 100.0

        sharpe = sharpe_from_returns(filtered_trades["pnl_usd"] / INITIAL_CAPITAL)

        avg_winner = filtered_trades[filtered_trades["pnl_usd"] > 0]["pnl_usd"].mean() if winners > 0 else 0.0
        avg_loser = filtered_trades[filtered_trades["pnl_usd"] < 0]["pnl_usd"].mean() if losers > 0 else 0.0

        summary = {
            "initial_capital_usd": INITIAL_CAPITAL,
            "final_capital_usd": final_capital,
            "total_pnl_usd": total_pnl,
            "total_return_pct": total_return_pct,
            "total_trades": total_trades,
            "winners": int(winners),
            "losers": int(losers),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "sharpe": sharpe,
            "max_drawdown_usd": max_dd,
            "max_drawdown_pct": max_dd_pct,
            "avg_winner_usd": avg_winner,
            "avg_loser_usd": avg_loser,
            "avg_pnl_per_trade_usd": total_pnl / total_trades if total_trades > 0 else 0.0,
        }

        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(BACKTEST_SUMMARY_CSV, index=False)

        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        print(f"Symbol: {symbol}")
        print(f"Strategy: RF Meta-Labeling ≥ {RF_THRESHOLD:.2f} | 1.0 ATR Stop / 1.5 ATR Target")
        print(f"Position Size: {CONTRACTS} MNQ contract ($2/point)")
        print(f"Backtest Period: {filtered_trades['entry_time'].min()} to {filtered_trades['exit_time'].max()}")
        print("\n" + "-"*80)
        print(f"Initial Capital:      ${INITIAL_CAPITAL:,.2f}")
        print(f"Final Capital:        ${final_capital:,.2f}")
        print(f"Total P&L:            ${total_pnl:,.2f}")
        print(f"Total Return:         {total_return_pct:.2f}%")
        print(f"\nTotal Trades:         {total_trades}")
        print(f"Winners:              {winners} ({win_rate:.2f}%)")
        print(f"Losers:               {losers}")
        print(f"Profit Factor:        {profit_factor:.2f}")
        print(f"Sharpe Ratio:         {sharpe:.2f}")
        print(f"\nMax Drawdown:         ${max_dd:,.2f} ({max_dd_pct:.2f}%)")
        print(f"Avg Winner:           ${avg_winner:,.2f}")
        print(f"Avg Loser:            ${avg_loser:,.2f}")
        print(f"Avg P&L per Trade:    ${total_pnl / total_trades:,.2f}")
        print("\n" + "="*80)
        print("LAST 10 TRADES (Most Recent First)")
        print("="*80)

        last_10 = filtered_trades.tail(10).iloc[::-1]
        for idx, row in last_10.iterrows():
            outcome = "WIN" if row["pnl_usd"] > 0 else "LOSS"
            print(f"\nTrade #{idx + 1} | {row['trigger']} | {outcome}")
            print(f"  Entry:  {row['entry_time']} @ {row['entry_price']:.2f}")
            print(f"  Exit:   {row['exit_time']} @ {row['exit_price']:.2f} ({row['exit_reason']})")
            print(f"  Stop:   {row['stop_price']:.2f} | Target: {row['target_price']:.2f}")
            print(f"  P&L:    ${row['pnl_usd']:,.2f} | R: {row['r_multiple']:.2f} | Bars: {int(row['bars_held'])}")
            print(f"  RF Prob: {row['rf_prob']:.3f} | Capital: ${row['capital_before']:,.2f} → ${row['capital_after']:,.2f}")

        print("\n" + "="*80)
        print(f"Saved backtest trades to: {BACKTEST_TRADES_CSV}")
        print(f"Saved backtest summary to: {BACKTEST_SUMMARY_CSV}")
        print("="*80 + "\n")
