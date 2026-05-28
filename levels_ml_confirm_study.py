"""
Key Levels ML Pipeline — Support Detection + Bounce Classifier
Extended with confirmation study mode.

Stages:
1. Rule-based swing low detection → candidate support levels
2. Retest detection → each time price returns to a level
3. Feature extraction → context at each retest
4. XGBoost classifier → did this retest produce a tradeable bounce?

Modes:
collect — pull bars, find levels, label retests, save dataset
train — train XGBoost on labeled retests
validate — walk-forward OOS comparison (filtered vs unfiltered)
scan — show live levels being approached RIGHT NOW + ML score
study — SL/TP optimisation via walk-forward OOS bar simulation
confirm_study — compare base entry vs confirmation entry variants

Requirements:
pip install mt5linux xgboost scikit-learn pandas numpy joblib
"""

import argparse
import json
from datetime import datetime, timezone
import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from mt5linux import MetaTrader5

MT5_HOST = "127.0.0.1"
MT5_PORT = 18812
mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)

DATASET_PATH = "levels_dataset.csv"
MODEL_PATH = "levels_xgb_model.joblib"
SCORE_THRESH = 0.60
R_TARGET = 1.5
ATR_PERIOD = 14

SWING_LOOKBACK = 5
LEVEL_ZONE_ATR = 0.30
MIN_DEPARTURE_BARS = 3
MIN_BOUNCE_BARS = 20
MIN_DEPARTURE_ATR = 1.0

TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16408

HTF_MAP = {
    TIMEFRAME_H1: TIMEFRAME_H4,
    TIMEFRAME_M15: TIMEFRAME_H1,
    TIMEFRAME_M5: TIMEFRAME_M15,
}

TF_LABEL_MAP = {
    "M5": TIMEFRAME_M5,
    "M15": TIMEFRAME_M15,
    "H1": TIMEFRAME_H1,
    "H4": TIMEFRAME_H4,
    "D1": TIMEFRAME_D1,
}


def connect():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 connection failed: {mt5.last_error()}")
    print("[MT5] Connected")


def get_bars(symbol: str, tf: int, n: int = 5000) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data for {symbol} tf={tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]].copy()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr_percentile(atr_series: pd.Series, window: int = 100) -> pd.Series:
    return atr_series.rolling(window).rank(pct=True)


def find_swing_lows(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    lows = df["low"].values
    atr_vals = calc_atr(df, ATR_PERIOD).values
    swings = []
    for i in range(lookback, len(df) - lookback):
        window = lows[i - lookback: i + lookback + 1]
        if lows[i] == window.min() and list(window).count(lows[i]) == 1:
            swings.append({
                "bar_index": i,
                "timestamp": df.index[i],
                "price": lows[i],
                "atr_at_formation": atr_vals[i],
            })
    return pd.DataFrame(swings)


def find_retests(df: pd.DataFrame, swings: pd.DataFrame) -> list[dict]:
    atr_vals = calc_atr(df, ATR_PERIOD).values
    closes = df["close"].values
    lows = df["low"].values
    highs = df["high"].values
    opens = df["open"].values
    volumes = df["volume"].values
    times = df.index
    retests = []

    for _, swing in swings.iterrows():
        origin_i = int(swing["bar_index"])
        level = swing["price"]
        atr_f = swing["atr_at_formation"]
        zone_width = LEVEL_ZONE_ATR * atr_f
        zone_lo = level - zone_width
        zone_hi = level + zone_width
        in_zone = False
        bars_outside = 0
        max_high_since_departure = level
        touch_count = 0

        for i in range(origin_i + 1, len(df)):
            price_low = lows[i]
            price_close = closes[i]
            if price_close < zone_lo:
                break
            currently_in_zone = (price_low <= zone_hi) and (price_low >= zone_lo - zone_width)
            if not in_zone:
                if currently_in_zone:
                    if (
                        bars_outside >= MIN_DEPARTURE_BARS
                        and max_high_since_departure >= level + MIN_DEPARTURE_ATR * atr_f
                    ):
                        retests.append({
                            "level_price": level,
                            "level_origin_i": origin_i,
                            "level_origin_ts": swing["timestamp"],
                            "retest_bar_i": i,
                            "retest_ts": times[i],
                            "touch_count": touch_count,
                            "bars_since_origin": i - origin_i,
                            "bars_outside": bars_outside,
                            "max_high_since_dep": max_high_since_departure,
                            "atr_at_retest": atr_vals[i],
                            "atr_at_formation": atr_f,
                            "open": opens[i],
                            "high": highs[i],
                            "low": price_low,
                            "close": price_close,
                            "volume": volumes[i],
                        })
                        touch_count += 1
                    in_zone = True
                    bars_outside = 0
                    max_high_since_departure = level
                else:
                    bars_outside += 1
                    if highs[i] > max_high_since_departure:
                        max_high_since_departure = highs[i]
            else:
                if not currently_in_zone:
                    in_zone = False
                    bars_outside = 0
                    max_high_since_departure = closes[i]

    return retests


def extract_features(df: pd.DataFrame, htf_df: pd.DataFrame, retests: list[dict], live_mode: bool = False) -> pd.DataFrame:
    atr_s = calc_atr(df, ATR_PERIOD)
    rsi_s = calc_rsi(df["close"])
    ema20_s = calc_ema(df["close"], 20)
    ema50_s = calc_ema(df["close"], 50)
    ema200_s = calc_ema(df["close"], 200)
    atr_pct_s = calc_atr_percentile(atr_s)
    vol_ma20 = df["volume"].rolling(20).mean()
    htf_ema20 = calc_ema(htf_df["close"], 20)
    htf_ema50 = calc_ema(htf_df["close"], 50)
    htf_atr = calc_atr(htf_df, ATR_PERIOD)
    records = []

    for r in retests:
        i = r["retest_bar_i"]
        level = r["level_price"]
        origin_i = r["level_origin_i"]
        if i < 50:
            continue
        if not live_mode and i >= len(df) - MIN_BOUNCE_BARS:
            continue

        atr_val = atr_s.iloc[i]
        if atr_val == 0 or np.isnan(atr_val):
            continue

        departure_height = (r["max_high_since_dep"] - level) / atr_val
        level_age_bars = r["bars_since_origin"]
        touch_count = r["touch_count"]

        depart_window = min(5, i - origin_i - 1)
        if depart_window > 0:
            post_origin_bars = df.iloc[origin_i + 1: origin_i + 1 + depart_window]
            origin_departure = (post_origin_bars["close"].max() - level) / atr_val
        else:
            origin_departure = 0.0

        approach = df.iloc[max(0, i - 5): i]
        approach_drop = (approach["close"].iloc[0] - df["close"].iloc[i]) / atr_val if len(approach) > 0 else 0
        approach_consec_red = 0
        for j in range(i - 1, max(0, i - 8), -1):
            if df["close"].iloc[j] < df["open"].iloc[j]:
                approach_consec_red += 1
            else:
                break
        approach_vol_ratio = df["volume"].iloc[i] / vol_ma20.iloc[i] if vol_ma20.iloc[i] > 0 else 1.0

        o, h, lw, c = r["open"], r["high"], r["low"], r["close"]
        body = abs(c - o)
        lower_wick = min(o, c) - lw
        candle_range = h - lw
        close_above_level = int(c > level)
        wick_touched_level = int(lw <= level + LEVEL_ZONE_ATR * atr_val)
        wick_body_ratio = lower_wick / body if body > 0 else 0
        close_pos_range = (c - lw) / candle_range if candle_range > 0 else 0
        precision = abs(lw - level) / atr_val

        rsi_val = rsi_s.iloc[i]
        pct_ema20 = (c - ema20_s.iloc[i]) / ema20_s.iloc[i]
        pct_ema50 = (c - ema50_s.iloc[i]) / ema50_s.iloc[i]
        pct_ema200 = (c - ema200_s.iloc[i]) / ema200_s.iloc[i]
        atr_pct = atr_pct_s.iloc[i] if not np.isnan(atr_pct_s.iloc[i]) else 0.5

        ts = df.index[i]
        htf_prior = htf_df[htf_df.index <= ts]
        if len(htf_prior) < 5:
            continue
        htf_c = htf_prior["close"].iloc[-1]
        htf_ema20_v = htf_ema20.reindex(htf_prior.index).iloc[-1]
        htf_ema50_v = htf_ema50.reindex(htf_prior.index).iloc[-1]
        htf_atr_v = htf_atr.reindex(htf_prior.index).iloc[-1]
        htf_trend = 1 if htf_c > htf_ema20_v else -1
        htf_pct_ema20 = (htf_c - htf_ema20_v) / htf_ema20_v if htf_ema20_v > 0 else 0
        htf_ema20_dist = abs(level - htf_ema20_v) / htf_atr_v if htf_atr_v > 0 else 99
        htf_ema50_dist = abs(level - htf_ema50_v) / htf_atr_v if htf_atr_v > 0 else 99
        htf_confluence = int(min(htf_ema20_dist, htf_ema50_dist) < 0.5)

        hour = ts.hour
        session = 0
        if 7 <= hour < 13:
            session = 1
        elif 13 <= hour < 21:
            session = 2
        day_of_week = ts.dayofweek
        round_100 = round(level / 100) * 100
        dist_round = abs(level - round_100) / atr_val

        if live_mode or i + 1 >= len(df):
            entry_price = df["close"].iloc[i]
        else:
            entry_price = df["open"].iloc[i + 1]

        stop_price = level - (0.5 * atr_val)
        risk = abs(entry_price - stop_price)
        target_price = entry_price + R_TARGET * risk

        outcome = np.nan
        if not live_mode and risk > 0:
            for k in range(i + 1, min(i + MIN_BOUNCE_BARS + 1, len(df))):
                future = df.iloc[k]
                if future["low"] <= stop_price:
                    outcome = 0
                    break
                if future["high"] >= target_price:
                    outcome = 1
                    break

        records.append({
            "timestamp": ts,
            "atr_at_retest": atr_val,
            "level_price": level,
            "retest_bar_i": i,
            "touch_count": touch_count,
            "level_age_bars": min(level_age_bars, 500),
            "departure_height": departure_height,
            "origin_departure": origin_departure,
            "approach_drop_atr": approach_drop,
            "approach_consec_red": approach_consec_red,
            "approach_vol_ratio": approach_vol_ratio,
            "close_above_level": close_above_level,
            "wick_touched_level": wick_touched_level,
            "wick_body_ratio": wick_body_ratio,
            "close_pos_range": close_pos_range,
            "precision": precision,
            "body_atr": body / atr_val,
            "rsi": rsi_val,
            "pct_from_ema20": pct_ema20,
            "pct_from_ema50": pct_ema50,
            "pct_from_ema200": pct_ema200,
            "atr_percentile": atr_pct,
            "htf_trend": htf_trend,
            "htf_pct_ema20": htf_pct_ema20,
            "htf_confluence": htf_confluence,
            "session": session,
            "hour": hour,
            "day_of_week": day_of_week,
            "dist_round_number": dist_round,
            "risk_atr": risk / atr_val if atr_val > 0 else 0,
            "outcome": outcome,
        })

    return pd.DataFrame(records)


def collect(symbol: str, tf_label: str, n_bars: int = 10000):
    tf = TF_LABEL_MAP[tf_label]
    htf = HTF_MAP.get(tf, TIMEFRAME_H4)
    connect()
    print(f"[COLLECT] Pulling {n_bars} bars for {symbol} {tf_label} ...")
    df = get_bars(symbol, tf, n=n_bars)
    htf_df = get_bars(symbol, htf, n=max(1000, n_bars // 4))
    mt5.shutdown()
    swings = find_swing_lows(df)
    retests = find_retests(df, swings)
    features = extract_features(df, htf_df, retests)
    features.to_csv(DATASET_PATH, index=False)
    print(f"[COLLECT] Found {len(swings)} swing lows | {len(retests)} retests | saved -> {DATASET_PATH}")


def train(dataset_path: str = DATASET_PATH):
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df.dropna(subset=["outcome"], inplace=True)
    df["outcome"] = df["outcome"].astype(int)
    df.sort_values("timestamp", inplace=True)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "outcome")]
    X = df[feature_cols]
    y = df["outcome"]

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"[TRAIN] 5-fold CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    model.fit(X, y)
    joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
    print(f"[TRAIN] Model saved -> {MODEL_PATH}")


def validate(dataset_path: str = DATASET_PATH, n_splits: int = 5, threshold: float = SCORE_THRESH):
    from xgboost import XGBClassifier

    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df.dropna(subset=["outcome"], inplace=True)
    df["outcome"] = df["outcome"].astype(int)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    feature_cols = [c for c in df.columns if c not in ("timestamp", "outcome")]
    X = df[feature_cols].values
    y = df["outcome"].values
    n = len(df)
    fold_size = n // n_splits
    oos_probs = np.full(n, np.nan)

    print(f"[VALIDATE] {n} labeled retests — walk-forward {n_splits}-fold OOS")
    for fold in range(1, n_splits):
        train_end = fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break
        X_train, y_train = X[:train_end], y[:train_end]
        X_test = X[test_start:test_end]
        if len(np.unique(y_train)) < 2:
            continue
        model = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=5,
            scale_pos_weight=(y_train == 0).sum() / max(y_train.sum(), 1),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train)
        oos_probs[test_start:test_end] = model.predict_proba(X_test)[:, 1]

    df_oos = df[~np.isnan(oos_probs)].copy()
    df_oos["prob"] = oos_probs[~np.isnan(oos_probs)]
    print(f"[VALIDATE] OOS retests: {len(df_oos)} | filtered @ {threshold:.0%}: {len(df_oos[df_oos['prob'] >= threshold])}")


def scan(symbol: str, tf_label: str, n_bars: int = 3000, approach_atr: float = 2.0, top: int = 20, min_score: float = 0.0):
    print("[SCAN] Use your existing scan implementation or merge it from the prior script.")


def walkforward_oos_scores(df_feat: pd.DataFrame, feature_cols: list[str], n_splits: int = 5) -> pd.DataFrame:
    from xgboost import XGBClassifier

    df_feat = df_feat.copy().sort_values("timestamp").reset_index(drop=True)
    n = len(df_feat)
    fold_size = n // n_splits
    oos_probs = np.full(n, np.nan)

    for fold in range(1, n_splits):
        train_end = fold * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        if test_end <= test_start:
            break
        X_tr = df_feat[feature_cols].values[:train_end]
        y_tr = df_feat["outcome"].values[:train_end]
        X_te = df_feat[feature_cols].values[test_start:test_end]
        if len(np.unique(y_tr)) < 2:
            continue
        m = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=5,
            scale_pos_weight=(y_tr == 0).sum() / max(y_tr.sum(), 1),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        m.fit(X_tr, y_tr)
        oos_probs[test_start:test_end] = m.predict_proba(X_te)[:, 1]

    df_feat["oos_score"] = oos_probs
    return df_feat


def simulate_trade_variant(
    bars: pd.DataFrame,
    pos: int,
    atr_val: float,
    level: float,
    variant: str,
    sl_atr: float = 1.0,
    tp_r: float = 2.0,
    max_fwd_bars: int = 48,
    wick_ratio_min: float = 1.5,
):
    if pos >= len(bars) - 2:
        return None

    sig = bars.iloc[pos]
    nxt = bars.iloc[pos + 1]
    o, h, l, c = sig["open"], sig["high"], sig["low"], sig["close"]
    body = abs(c - o)
    lower_wick = min(o, c) - l
    bullish = c > o
    wick_ratio = lower_wick / body if body > 0 else 0.0

    triggered = False
    entry_price = None
    entry_bar_i = None

    if variant == "base":
        triggered, entry_bar_i = True, pos + 1
        entry_price = bars["open"].iloc[entry_bar_i]
    elif variant == "close_above_level":
        if c > level:
            triggered, entry_bar_i = True, pos + 1
            entry_price = bars["open"].iloc[entry_bar_i]
    elif variant == "break_retest_high":
        if pos + 2 >= len(bars):
            return None
        if nxt["high"] > h:
            triggered, entry_bar_i = True, pos + 2
            entry_price = bars["open"].iloc[entry_bar_i]
    elif variant == "bullish_wick":
        if bullish and wick_ratio >= wick_ratio_min:
            triggered, entry_bar_i = True, pos + 1
            entry_price = bars["open"].iloc[entry_bar_i]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if not triggered or entry_bar_i is None or entry_bar_i >= len(bars):
        return None

    stop_px = level - sl_atr * atr_val
    risk = entry_price - stop_px
    if risk <= 0:
        return None
    target_px = entry_price + tp_r * risk

    fwd = bars.iloc[entry_bar_i: min(entry_bar_i + max_fwd_bars, len(bars))]
    if len(fwd) < 1:
        return None

    trade_mfe = 0.0
    trade_mae = 0.0
    outcome_r = None
    bars_held = 0

    for _, fb in fwd.iterrows():
        bars_held += 1
        up = fb["high"] - entry_price
        down = entry_price - fb["low"]
        trade_mfe = max(trade_mfe, up)
        trade_mae = max(trade_mae, down)
        if fb["low"] <= stop_px:
            outcome_r = -1.0
            break
        if fb["high"] >= target_px:
            outcome_r = float(tp_r)
            break

    if outcome_r is None:
        outcome_r = (fwd["close"].iloc[-1] - entry_price) / risk

    return {
        "entry_bar_i": entry_bar_i,
        "entry_price": entry_price,
        "stop": stop_px,
        "target": target_px,
        "risk": risk,
        "bars_held": bars_held,
        "mfe_atr": trade_mfe / atr_val if atr_val > 0 else np.nan,
        "mae_atr": trade_mae / atr_val if atr_val > 0 else np.nan,
        "outcome_r": outcome_r,
        "win": int(outcome_r > 0),
        "wick_ratio": wick_ratio,
        "bullish": int(bullish),
    }


def summarise_variant(df: pd.DataFrame, name: str) -> dict:
    if df.empty:
        return {
            "variant": name,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "net_r": 0.0,
            "expectancy_r": 0.0,
            "avg_mfe_atr": np.nan,
            "avg_mae_atr": np.nan,
        }

    wins = int((df["outcome_r"] > 0).sum())
    losses = int((df["outcome_r"] <= 0).sum())
    total = len(df)
    gp = float(df.loc[df["outcome_r"] > 0, "outcome_r"].sum())
    gl = float(abs(df.loc[df["outcome_r"] <= 0, "outcome_r"].sum()))

    return {
        "variant": name,
        "trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total if total else 0.0,
        "profit_factor": gp / gl if gl > 0 else float("inf"),
        "net_r": gp - gl,
        "expectancy_r": (gp - gl) / total if total else 0.0,
        "avg_mfe_atr": float(df["mfe_atr"].mean()),
        "avg_mae_atr": float(df["mae_atr"].mean()),
    }


def confirm_study(
    symbol="@MNQ",
    tf_label="H1",
    dataset_path=DATASET_PATH,
    min_score=0.70,
    n_bars=20000,
    max_fwd_bars=48,
    sl_atr=1.0,
    tp_r=2.0,
    wick_ratio_min=1.5,
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Run --mode collect first.")

    df_feat = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df_feat.dropna(subset=["outcome"], inplace=True)
    df_feat["outcome"] = df_feat["outcome"].astype(int)
    df_feat.sort_values("timestamp", inplace=True)
    df_feat.reset_index(drop=True, inplace=True)
    feature_cols = [c for c in df_feat.columns if c not in ("timestamp", "outcome")]

    print(f"[CONFIRM] Walk-forward scoring {len(df_feat)} retests ...")
    df_feat = walkforward_oos_scores(df_feat, feature_cols, n_splits=5)
    signals = df_feat[~df_feat["oos_score"].isna() & (df_feat["oos_score"] >= min_score)].copy()
    print(f"[CONFIRM] {len(signals)} OOS signals at score >= {min_score:.0%}")

    if len(signals) < 10:
        print("[CONFIRM] Too few signals. Lower threshold or collect more data.")
        return

    tf = TF_LABEL_MAP[tf_label]
    connect()
    print(f"[CONFIRM] Pulling {n_bars} raw bars from MT5 ...")
    bars = get_bars(symbol, tf, n=n_bars)
    mt5.shutdown()

    atr_series = calc_atr(bars, ATR_PERIOD)

    variants = [
        ("base", "Base entry"),
        ("close_above_level", "Confirm: close above level"),
        ("break_retest_high", "Confirm: next bar breaks retest high"),
        ("bullish_wick", f"Confirm: bullish retest + wick ratio >= {wick_ratio_min:.2f}"),
    ]

    variant_trades = {k: [] for k, _ in variants}
    skipped = 0

    for _, sig in signals.iterrows():
        ts = sig["timestamp"]
        pos = bars.index.searchsorted(ts)
        if pos >= len(bars) - max_fwd_bars - 2:
            skipped += 1
            continue
        if abs((bars.index[pos] - ts).total_seconds()) > 7200:
            skipped += 1
            continue

        atr_val = sig["atr_at_retest"] if "atr_at_retest" in sig and sig["atr_at_retest"] > 0 else atr_series.iloc[pos]
        if atr_val <= 0 or np.isnan(atr_val):
            skipped += 1
            continue

        level = sig["level_price"] if "level_price" in sig else np.nan
        if np.isnan(level):
            skipped += 1
            continue

        for key, _ in variants:
            out = simulate_trade_variant(
                bars=bars,
                pos=pos,
                atr_val=atr_val,
                level=level,
                variant=key,
                sl_atr=sl_atr,
                tp_r=tp_r,
                max_fwd_bars=max_fwd_bars,
                wick_ratio_min=wick_ratio_min,
            )
            if out is None:
                continue
            out.update({
                "signal_ts": ts,
                "signal_score": sig["oos_score"],
                "level": level,
            })
            variant_trades[key].append(out)

    rows = []
    for key, label in variants:
        rows.append(summarise_variant(pd.DataFrame(variant_trades[key]), label))
    res = pd.DataFrame(rows).sort_values("expectancy_r", ascending=False)

    SEP = "=" * 88
    print(f"\n{SEP}")
    print(f"CONFIRMATION STUDY — {symbol} {tf_label}")
    print(SEP)
    print(f" OOS score threshold : {min_score:.0%}")
    print(f" Stop loss          : {sl_atr:.2f} ATR")
    print(f" Target             : {tp_r:.1f}R")
    print(f" Max bars held      : {max_fwd_bars}")
    print(f" Signals skipped    : {skipped}")
    print()
    print(f" {'Variant':<42} {'Trades':>8} {'WinRate':>9} {'PF':>8} {'Expect':>10} {'Net R':>10}")
    print(" " + "-" * 91)
    for _, r in res.iterrows():
        pf_str = "inf" if r["profit_factor"] == float("inf") else f"{r['profit_factor']:.3f}"
        print(f" {r['variant']:<42} {int(r['trades']):>8} {r['win_rate']:>9.1%} {pf_str:>8} {r['expectancy_r']:>+10.4f} {r['net_r']:>+10.2f}")

    out_csv = "confirm_study_results.csv"
    res.to_csv(out_csv, index=False)
    print(f"\n[CONFIRM] Results saved -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Key Levels ML Pipeline")
    parser.add_argument("--mode", choices=["collect", "train", "validate", "scan", "confirm_study"], required=True)
    parser.add_argument("--symbol", default="@MNQ")
    parser.add_argument("--tf", default="H1", choices=list(TF_LABEL_MAP.keys()))
    parser.add_argument("--bars", type=int, default=10000)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESH)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--tp-r", type=float, default=2.0)
    parser.add_argument("--max-fwd-bars", type=int, default=48)
    parser.add_argument("--wick-ratio", type=float, default=1.5)
    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.symbol, args.tf, args.bars)
    elif args.mode == "train":
        train()
    elif args.mode == "validate":
        validate(threshold=args.threshold)
    elif args.mode == "scan":
        scan(args.symbol, args.tf, n_bars=args.bars, top=args.top, min_score=args.min_score)
    elif args.mode == "confirm_study":
        confirm_study(
            symbol=args.symbol,
            tf_label=args.tf,
            min_score=args.threshold,
            n_bars=args.bars,
            max_fwd_bars=args.max_fwd_bars,
            sl_atr=args.sl_atr,
            tp_r=args.tp_r,
            wick_ratio_min=args.wick_ratio,
        )
