"""
Hammer ML Pipeline — Feature Extraction + XGBoost Classifier
Uses mt5linux to pull OHLCV data for feature engineering.

Requirements:
    pip install mt5linux xgboost scikit-learn pandas numpy joblib

Usage:
    1. Set MT5_HOST/MT5_PORT to match your mt5linux server
    2. Run collect mode first to build the dataset:
           python hammer_ml_pipeline.py --mode collect --symbol @MNQ --tf H1
    3. Then train the model:
           python hammer_ml_pipeline.py --mode train
    4. Then score live hammers:
           python hammer_ml_pipeline.py --mode score --symbol @MNQ --tf H1
"""

import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# ── mt5linux ──────────────────────────────────────────────────────────────────
from mt5linux import MetaTrader5

MT5_HOST = "127.0.0.1"
MT5_PORT = 18812

# Instantiate once — mt5linux requires an instance, not static calls
mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH   = "hammer_dataset.csv"
MODEL_PATH     = "hammer_xgb_model.joblib"
SCORE_THRESH   = 0.60       # minimum probability to consider a hammer tradeable
R_TARGET       = 1.5        # must match your backtest target
ATR_PERIOD     = 14
EMA_PERIODS    = [20, 50, 200]

# Timeframe constants (mt5linux instance exposes these)
TIMEFRAME_M5  = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1  = 16385
TIMEFRAME_H4  = 16388
TIMEFRAME_D1  = 16408

HTF_MAP = {
    TIMEFRAME_H1:  TIMEFRAME_H4,
    TIMEFRAME_M15: TIMEFRAME_H1,
    TIMEFRAME_M5:  TIMEFRAME_M15,
}
TF_LABEL_MAP = {
    "M5":  TIMEFRAME_M5,
    "M15": TIMEFRAME_M15,
    "H1":  TIMEFRAME_H1,
    "H4":  TIMEFRAME_H4,
    "D1":  TIMEFRAME_D1,
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. MT5 helpers
# ─────────────────────────────────────────────────────────────────────────────

def connect():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 connection failed: {mt5.last_error()}")
    print("[MT5] Connected")


def get_bars(symbol: str, tf: int, n: int = 1000) -> pd.DataFrame:
    """Pull last n bars and return as DataFrame with standard columns."""
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data for {symbol} tf={tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hammer detection
# ─────────────────────────────────────────────────────────────────────────────

def is_hammer(row: pd.Series, atr_val: float,
              min_wick_ratio: float = 2.0,
              max_upper_wick_ratio: float = 0.3) -> bool:
    """
    Standard hammer rules:
      - Lower wick >= min_wick_ratio * body
      - Upper wick <= max_upper_wick_ratio * body (or very small)
      - Body > 0 (not a doji)
      - Lower wick > 0.5 * ATR  (avoid tiny candles)
    """
    body      = abs(row["close"] - row["open"])
    lower_wick = min(row["open"], row["close"]) - row["low"]
    upper_wick = row["high"] - max(row["open"], row["close"])

    if body == 0:
        return False
    if lower_wick < 0.5 * atr_val:
        return False
    if lower_wick < min_wick_ratio * body:
        return False
    if upper_wick > max_upper_wick_ratio * lower_wick:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. Feature extraction  (one row per hammer)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame, htf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for every hammer candle found in df.
    htf_df  — higher-timeframe bars aligned to df index.
    """
    atr_s  = atr(df, ATR_PERIOD)
    rsi_s  = rsi(df["close"])
    emas   = {p: ema(df["close"], p) for p in EMA_PERIODS}

    # HTF trend: slope of 20 EMA on higher TF
    htf_ema20    = ema(htf_df["close"], 20)
    htf_ema_slope = htf_ema20.diff(3)  # 3-bar slope

    records = []

    for i in range(max(EMA_PERIODS) + 5, len(df) - 2):
        row     = df.iloc[i]
        atr_val = atr_s.iloc[i]

        if not is_hammer(row, atr_val):
            continue

        body       = abs(row["close"] - row["open"])
        lower_wick = min(row["open"], row["close"]) - row["low"]
        upper_wick = row["high"] - max(row["open"], row["close"])
        candle_range = row["high"] - row["low"]

        # ── price vs EMAs ────────────────────────────────────────────────────
        close = row["close"]
        ema20_v  = emas[20].iloc[i]
        ema50_v  = emas[50].iloc[i]
        ema200_v = emas[200].iloc[i]

        # ── prior trend: count consecutive red bars before hammer ────────────
        consec_red = 0
        for j in range(i - 1, max(0, i - 10), -1):
            if df["close"].iloc[j] < df["open"].iloc[j]:
                consec_red += 1
            else:
                break

        # ── volume vs 20-bar average ─────────────────────────────────────────
        vol_avg = df["volume"].iloc[i - 20:i].mean()
        vol_ratio = row["volume"] / vol_avg if vol_avg > 0 else 1.0

        # ── HTF context (align to nearest prior bar) ─────────────────────────
        ts = df.index[i]
        htf_prior = htf_df[htf_df.index <= ts]
        if len(htf_prior) < 2:
            continue
        htf_row   = htf_prior.iloc[-1]
        htf_close = htf_row["close"]
        htf_ema20_v = htf_ema20.reindex(htf_prior.index).iloc[-1]
        htf_slope_v = htf_ema_slope.reindex(htf_prior.index).iloc[-1]
        htf_trend   = 1 if htf_close > htf_ema20_v else -1

        # ── session hour (UTC) ────────────────────────────────────────────────
        hour = ts.hour
        # NY session 13–20, London 07–12, Asia 00–06
        session = 0
        if 7 <= hour < 13:
            session = 1   # London
        elif 13 <= hour < 21:
            session = 2   # NY

        day_of_week = ts.dayofweek  # 0=Mon … 4=Fri

        # ── distance to round number (MNQ ticks of 0.25) ─────────────────────
        round_100 = round(close / 100) * 100
        dist_round = abs(close - round_100) / atr_val

        # ── outcome label (look-forward 1.5R) ────────────────────────────────
        stop_price   = row["low"] - (0.25 * atr_val)   # buffer below wick low
        entry_price  = df["open"].iloc[i + 1]           # next bar open
        risk         = abs(entry_price - stop_price)
        target_price = entry_price + R_TARGET * risk

        outcome = np.nan
        for k in range(i + 2, min(i + 30, len(df))):
            future = df.iloc[k]
            if future["low"] <= stop_price:
                outcome = 0
                break
            if future["high"] >= target_price:
                outcome = 1
                break

        feat = {
            "timestamp": ts,
            # hammer shape
            "wick_body_ratio":    lower_wick / body if body > 0 else 0,
            "upper_wick_ratio":   upper_wick / candle_range if candle_range > 0 else 0,
            "body_atr_ratio":     body / atr_val if atr_val > 0 else 0,
            "lower_wick_atr":     lower_wick / atr_val if atr_val > 0 else 0,
            "close_pos_in_range": (close - row["low"]) / candle_range if candle_range > 0 else 0,
            # trend
            "pct_from_ema20":     (close - ema20_v) / ema20_v,
            "pct_from_ema50":     (close - ema50_v) / ema50_v,
            "pct_from_ema200":    (close - ema200_v) / ema200_v,
            "ema20_slope":        (emas[20].iloc[i] - emas[20].iloc[i - 5]) / atr_val,
            "consec_red_bars":    consec_red,
            # momentum
            "rsi":                rsi_s.iloc[i],
            # volume
            "vol_ratio":          vol_ratio,
            # session/time
            "hour":               hour,
            "session":            session,
            "day_of_week":        day_of_week,
            # HTF
            "htf_trend":          htf_trend,
            "htf_ema_slope":      htf_slope_v / atr_val if atr_val > 0 else 0,
            # structure
            "dist_round_number":  dist_round,
            # label
            "outcome":            outcome,
        }
        records.append(feat)

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH):
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report

    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df.dropna(subset=["outcome"], inplace=True)
    df["outcome"] = df["outcome"].astype(int)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "outcome")]
    X = df[feature_cols]
    y = df["outcome"]

    print(f"[TRAIN] Dataset: {len(df)} hammers  |  wins: {y.sum()}  losses: {(y==0).sum()}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / y.sum(),  # handle class imbalance
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"[TRAIN] 5-fold CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X, y)

    # feature importance
    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n[TRAIN] Top 10 features:")
    print(imp.head(10).to_string())

    # in-sample report (just indicative)
    preds = model.predict(X)
    print("\n[TRAIN] In-sample classification report:")
    print(classification_report(y, preds, target_names=["loss", "win"]))

    joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
    print(f"\n[TRAIN] Model saved → {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Collect dataset
# ─────────────────────────────────────────────────────────────────────────────

def collect(symbol: str, tf_label: str, n_bars: int = 5000):
    tf     = TF_LABEL_MAP[tf_label]
    htf    = HTF_MAP.get(tf, TIMEFRAME_H4)

    connect()
    print(f"[COLLECT] Pulling {n_bars} bars for {symbol} {tf_label} ...")
    df     = get_bars(symbol, tf,  n=n_bars)
    htf_df = get_bars(symbol, htf, n=n_bars // 4)
    mt5.shutdown()  # no-op on instance, safe to call

    print("[COLLECT] Extracting features ...")
    features = extract_features(df, htf_df)
    labeled  = features.dropna(subset=["outcome"])

    print(f"[COLLECT] Found {len(features)} hammers, {len(labeled)} fully labeled")
    features.to_csv(DATASET_PATH, index=False)
    print(f"[COLLECT] Saved → {DATASET_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Score live hammers
# ─────────────────────────────────────────────────────────────────────────────

def score(symbol: str, tf_label: str, n_bars: int = 600):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Run --mode train first.")

    saved      = joblib.load(MODEL_PATH)
    model      = saved["model"]
    feat_cols  = saved["features"]

    tf  = TF_LABEL_MAP[tf_label]
    htf = HTF_MAP.get(tf, TIMEFRAME_H4)

    connect()
    df     = get_bars(symbol, tf,  n=n_bars)
    htf_df = get_bars(symbol, htf, n=n_bars // 4)
    mt5.shutdown()  # no-op on instance, safe to call

    features = extract_features(df, htf_df)
    if features.empty:
        print("[SCORE] No hammers found in recent bars.")
        return

    X      = features[feat_cols].fillna(0)
    probs  = model.predict_proba(X)[:, 1]
    features = features.copy()
    features["win_probability"] = probs
    features["signal"] = np.where(probs >= SCORE_THRESH, "✅ TAKE", "⛔ SKIP")

    print(f"\n[SCORE] Recent hammers on {symbol} {tf_label}:")
    print(features[["timestamp", "win_probability", "signal"]].tail(20).to_string(index=False))

    tradeable = features[features["win_probability"] >= SCORE_THRESH]
    print(f"\n[SCORE] Tradeable hammers (>={SCORE_THRESH*100:.0f}% confidence): {len(tradeable)}")
    if not tradeable.empty:
        print(tradeable[["timestamp", "win_probability"]].to_string(index=False))



# ─────────────────────────────────────────────────────────────────────────────
# 8. Validate — walk-forward OOS comparison
# ─────────────────────────────────────────────────────────────────────────────

def validate(dataset_path=DATASET_PATH, n_splits=5, threshold=SCORE_THRESH):
    """
    Walk-forward (time-series) validation.
    Splits labeled dataset chronologically into n_splits folds.
    For each fold: train on all prior folds, predict on current fold (OOS).
    Compares unfiltered vs ML-filtered metrics side by side.
    """
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

    print(f"\n[VALIDATE] {n} labeled hammers — walk-forward {n_splits}-fold OOS validation")
    print(f"[VALIDATE] Score threshold: {threshold:.0%}  |  R target: {R_TARGET}R\n")

    oos_probs = np.full(n, np.nan)
    fold_size = n // n_splits

    for fold in range(1, n_splits):
        train_end  = fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)
        if test_end <= test_start:
            break
        X_train, y_train = X[:train_end], y[:train_end]
        X_test = X[test_start:test_end]
        if len(np.unique(y_train)) < 2:
            continue

        model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / max(y_train.sum(), 1),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train)
        oos_probs[test_start:test_end] = model.predict_proba(X_test)[:, 1]

        wins_in_fold   = y[test_start:test_end].sum()
        losses_in_fold = (y[test_start:test_end] == 0).sum()
        print(f"  Fold {fold}: trained on {train_end} samples | "
              f"tested on {test_end - test_start} ({wins_in_fold}W / {losses_in_fold}L)")

    oos_mask = ~np.isnan(oos_probs)
    df_oos   = df[oos_mask].copy()
    df_oos["prob"] = oos_probs[oos_mask]

    def calc_metrics(subset, label):
        if len(subset) == 0:
            print(f"\n  {label}: no trades")
            return
        w  = (subset["outcome"] == 1).sum()
        l  = (subset["outcome"] == 0).sum()
        total = len(subset)
        wr = w / total
        gp = w * R_TARGET
        gl = l * 1.0
        pf = gp / gl if gl > 0 else float("inf")
        net_r = gp - gl
        exp_r = net_r / total
        print(f"\n  ── {label} ──")
        print(f"     Trades        : {total}")
        print(f"     Win Rate      : {wr:.1%}")
        print(f"     Profit Factor : {pf:.3f}")
        print(f"     Net R         : {net_r:+.2f}R")
        print(f"     Expectancy    : {exp_r:+.4f}R/trade")

    calc_metrics(df_oos, "UNFILTERED  (all OOS hammers)")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
        calc_metrics(df_oos[df_oos["prob"] >= t], f"ML FILTERED (prob >= {t:.0%})")

    print("\n\n[VALIDATE] Threshold sensitivity table (OOS):")
    print(f"  {'Threshold':>10} {'Trades':>8} {'WinRate':>9} {'ProfitFactor':>14} {'ExpectancyR':>13}")
    print("  " + "-" * 58)
    for t in np.arange(0.40, 0.85, 0.05):
        sub = df_oos[df_oos["prob"] >= t]
        if len(sub) == 0:
            continue
        w = (sub["outcome"] == 1).sum()
        l = (sub["outcome"] == 0).sum()
        wr = w / len(sub)
        pf = (w * R_TARGET) / l if l > 0 else float("inf")
        ex = (w * R_TARGET - l) / len(sub)
        print(f"  {t:>10.0%} {len(sub):>8} {wr:>9.1%} {pf:>14.3f} {ex:>+13.4f}R")

    print(f"\n[VALIDATE] Done — {len(df_oos)} OOS hammers evaluated.")
    print("[VALIDATE] Note: with <500 samples, treat results as directional, not definitive.")
    return df_oos


if __name__ == "__main__":
    import sys
    # rebuild parser with validate included
    parser = argparse.ArgumentParser(description="Hammer ML Pipeline")
    parser.add_argument("--mode", choices=["collect", "train", "score", "validate"], required=True)
    parser.add_argument("--symbol", default="@MNQ")
    parser.add_argument("--tf", default="H1", choices=list(TF_LABEL_MAP.keys()))
    parser.add_argument("--bars", type=int, default=5000)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESH)
    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.symbol, args.tf, args.bars)
    elif args.mode == "train":
        train()
    elif args.mode == "score":
        score(args.symbol, args.tf)
    elif args.mode == "validate":
        validate(threshold=args.threshold)
