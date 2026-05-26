"""
Key Levels ML Pipeline — Support Detection + Bounce Classifier
Uses mt5linux to pull OHLCV data.

Stages:
  1. Rule-based swing low detection  → candidate support levels
  2. Retest detection                → each time price returns to a level
  3. Feature extraction              → context at each retest
  4. XGBoost classifier              → did this retest produce a tradeable bounce?

Modes:
  collect  — pull bars, find levels, label retests, save dataset
  train    — train XGBoost on labeled retests
  validate — walk-forward OOS comparison (filtered vs unfiltered)
  scan     — show live levels being approached RIGHT NOW + ML score

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

# ── mt5linux ──────────────────────────────────────────────────────────────────
from mt5linux import MetaTrader5

MT5_HOST = "127.0.0.1"
MT5_PORT = 18812
mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_PATH  = "levels_dataset.csv"
MODEL_PATH    = "levels_xgb_model.joblib"
SCORE_THRESH  = 0.60
R_TARGET      = 1.5
ATR_PERIOD    = 14

# Level detection params
SWING_LOOKBACK     = 5      # bars each side to confirm a swing low
LEVEL_ZONE_ATR     = 0.30   # level zone width = 0.30 * ATR
MIN_DEPARTURE_BARS = 3      # price must leave level for this many bars before retest counts
MIN_BOUNCE_BARS    = 20     # look forward up to N bars to resolve outcome
MIN_DEPARTURE_ATR  = 1.0    # price must move away at least 1 ATR before retest

# Timeframe constants
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


def get_bars(symbol: str, tf: int, n: int = 5000) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data for {symbol} tf={tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]].copy()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Indicator helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    """Rolling percentile rank of ATR — measures if volatility is expanding."""
    return atr_series.rolling(window).rank(pct=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Stage 1 — Swing low detection
# ─────────────────────────────────────────────────────────────────────────────

def find_swing_lows(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    A swing low at index i: df['low'][i] is the minimum of
    the window [i-lookback : i+lookback+1].
    Returns DataFrame of swing lows with columns:
        bar_index, timestamp, price, atr_at_formation
    """
    lows = df["low"].values
    atr_vals = calc_atr(df, ATR_PERIOD).values
    swings = []

    for i in range(lookback, len(df) - lookback):
        window = lows[i - lookback: i + lookback + 1]
        if lows[i] == window.min() and list(window).count(lows[i]) == 1:
            swings.append({
                "bar_index":        i,
                "timestamp":        df.index[i],
                "price":            lows[i],
                "atr_at_formation": atr_vals[i],
            })

    return pd.DataFrame(swings)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stage 2 — Retest detection
# ─────────────────────────────────────────────────────────────────────────────

def find_retests(df: pd.DataFrame, swings: pd.DataFrame) -> list[dict]:
    """
    For each swing low, find every subsequent bar where price
    returns into the level zone without having broken below it first.

    Level zone: [swing_price - zone_width, swing_price + zone_width]
    where zone_width = LEVEL_ZONE_ATR * atr_at_formation

    A retest is valid when:
      - Price left the zone for at least MIN_DEPARTURE_BARS bars
      - Price moved at least MIN_DEPARTURE_ATR * ATR above the level
      - Price has not closed below the level (level not broken)
      - This is the first bar back inside the zone after departure
    """
    atr_vals = calc_atr(df, ATR_PERIOD).values
    closes   = df["close"].values
    lows     = df["low"].values
    highs    = df["high"].values
    opens    = df["open"].values
    volumes  = df["volume"].values
    times    = df.index

    retests = []

    for _, swing in swings.iterrows():
        origin_i   = int(swing["bar_index"])
        level      = swing["price"]
        atr_f      = swing["atr_at_formation"]
        zone_width = LEVEL_ZONE_ATR * atr_f
        zone_lo    = level - zone_width
        zone_hi    = level + zone_width

        in_zone      = False
        bars_outside = 0
        max_high_since_departure = level
        touch_count  = 0        # how many prior touches before this retest
        broken       = False

        for i in range(origin_i + 1, len(df)):
            price_low   = lows[i]
            price_close = closes[i]

            # Level broken if close goes below zone low
            if price_close < zone_lo:
                broken = True
                break

            currently_in_zone = (price_low <= zone_hi) and (price_low >= zone_lo - zone_width)

            if not in_zone:
                if currently_in_zone:
                    # entering zone — check if departure was valid
                    if (bars_outside >= MIN_DEPARTURE_BARS and
                            max_high_since_departure >= level + MIN_DEPARTURE_ATR * atr_f):
                        retests.append({
                            "level_price":       level,
                            "level_origin_i":    origin_i,
                            "level_origin_ts":   swing["timestamp"],
                            "retest_bar_i":      i,
                            "retest_ts":         times[i],
                            "touch_count":       touch_count,
                            "bars_since_origin": i - origin_i,
                            "bars_outside":      bars_outside,
                            "max_high_since_dep":max_high_since_departure,
                            "atr_at_retest":     atr_vals[i],
                            "atr_at_formation":  atr_f,
                            # raw bar data at retest
                            "open":   opens[i],
                            "high":   highs[i],
                            "low":    price_low,
                            "close":  price_close,
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
                # already in zone — wait until we leave again
                if not currently_in_zone:
                    in_zone = False
                    bars_outside = 0
                    max_high_since_departure = closes[i]

    return retests


# ─────────────────────────────────────────────────────────────────────────────
# 5. Feature extraction per retest
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame, htf_df: pd.DataFrame,
                     retests: list[dict], live_mode: bool = False) -> pd.DataFrame:
    """
    Build one feature row per retest event.
    Features capture: level quality, approach characteristics,
    retest candle, momentum, HTF context, session.
    """
    atr_s      = calc_atr(df, ATR_PERIOD)
    rsi_s      = calc_rsi(df["close"])
    ema20_s    = calc_ema(df["close"], 20)
    ema50_s    = calc_ema(df["close"], 50)
    ema200_s   = calc_ema(df["close"], 200)
    atr_pct_s  = calc_atr_percentile(atr_s)
    vol_ma20   = df["volume"].rolling(20).mean()

    htf_ema20  = calc_ema(htf_df["close"], 20)
    htf_ema50  = calc_ema(htf_df["close"], 50)
    htf_atr    = calc_atr(htf_df, ATR_PERIOD)

    records = []

    for r in retests:
        i          = r["retest_bar_i"]
        level      = r["level_price"]
        origin_i   = r["level_origin_i"]

        if i < 50:
            continue
        if not live_mode and i >= len(df) - MIN_BOUNCE_BARS:
            continue

        atr_val    = atr_s.iloc[i]
        if atr_val == 0:
            continue

        # ── Level quality features ────────────────────────────────────────────
        departure_height = (r["max_high_since_dep"] - level) / atr_val
        level_age_bars   = r["bars_since_origin"]
        touch_count      = r["touch_count"]

        # How clean was the original swing? (sharpness of departure)
        depart_window = min(5, i - origin_i - 1)
        if depart_window > 0:
            post_origin_bars = df.iloc[origin_i + 1: origin_i + 1 + depart_window]
            origin_departure = (post_origin_bars["close"].max() - level) / atr_val
        else:
            origin_departure = 0.0

        # ── Approach features (last 5 bars coming into level) ─────────────────
        approach = df.iloc[max(0, i - 5): i]
        approach_bars        = len(approach)
        approach_drop        = (approach["close"].iloc[0] - df["close"].iloc[i]) / atr_val if approach_bars > 0 else 0
        approach_consec_red  = 0
        for j in range(i - 1, max(0, i - 8), -1):
            if df["close"].iloc[j] < df["open"].iloc[j]:
                approach_consec_red += 1
            else:
                break
        approach_vol_ratio = (df["volume"].iloc[i] / vol_ma20.iloc[i]
                              if vol_ma20.iloc[i] > 0 else 1.0)

        # ── Retest candle features ────────────────────────────────────────────
        o, h, lw, c = r["open"], r["high"], r["low"], r["close"]
        body         = abs(c - o)
        lower_wick   = min(o, c) - lw
        upper_wick   = h - max(o, c)
        candle_range = h - lw

        close_above_level  = int(c > level)
        wick_touched_level = int(lw <= level + LEVEL_ZONE_ATR * atr_val)
        wick_body_ratio    = lower_wick / body if body > 0 else 0
        close_pos_range    = (c - lw) / candle_range if candle_range > 0 else 0

        # How precisely did price hit the level?
        precision = abs(lw - level) / atr_val   # 0 = exact touch

        # ── Momentum ─────────────────────────────────────────────────────────
        rsi_val     = rsi_s.iloc[i]
        pct_ema20   = (c - ema20_s.iloc[i])  / ema20_s.iloc[i]
        pct_ema50   = (c - ema50_s.iloc[i])  / ema50_s.iloc[i]
        pct_ema200  = (c - ema200_s.iloc[i]) / ema200_s.iloc[i]
        atr_pct     = atr_pct_s.iloc[i] if not np.isnan(atr_pct_s.iloc[i]) else 0.5

        # ── HTF context ──────────────────────────────────────────────────────
        ts = df.index[i]
        htf_prior = htf_df[htf_df.index <= ts]
        if len(htf_prior) < 5:
            continue
        htf_c        = htf_prior["close"].iloc[-1]
        htf_ema20_v  = htf_ema20.reindex(htf_prior.index).iloc[-1]
        htf_ema50_v  = htf_ema50.reindex(htf_prior.index).iloc[-1]
        htf_atr_v    = htf_atr.reindex(htf_prior.index).iloc[-1]
        htf_trend    = 1 if htf_c > htf_ema20_v else -1
        htf_pct_ema20 = (htf_c - htf_ema20_v) / htf_ema20_v if htf_ema20_v > 0 else 0

        # Is the level also near an HTF EMA? (confluence)
        htf_ema20_dist = abs(level - htf_ema20_v) / htf_atr_v if htf_atr_v > 0 else 99
        htf_ema50_dist = abs(level - htf_ema50_v) / htf_atr_v if htf_atr_v > 0 else 99
        htf_confluence = int(min(htf_ema20_dist, htf_ema50_dist) < 0.5)

        # ── Session / time ────────────────────────────────────────────────────
        hour        = ts.hour
        session     = 0
        if 7 <= hour < 13:   session = 1   # London
        elif 13 <= hour < 21: session = 2  # NY
        day_of_week = ts.dayofweek

        # ── Distance to round number ──────────────────────────────────────────
        round_100   = round(level / 100) * 100
        dist_round  = abs(level - round_100) / atr_val

        # ── Outcome labeling (look forward) ──────────────────────────────────
        # In live mode the next bar doesn't exist yet — use close as entry proxy
        if live_mode or i + 1 >= len(df):
            entry_price = df["close"].iloc[i]
        else:
            entry_price = df["open"].iloc[i + 1]
        stop_price   = level - (0.5 * atr_val)   # below the level zone
        risk         = abs(entry_price - stop_price)
        target_price = entry_price + R_TARGET * risk

        outcome = np.nan
        if not live_mode and risk > 0:
            for k in range(i + 1, min(i + MIN_BOUNCE_BARS + 1, len(df))):
                future = df.iloc[k]
                if future["low"] <= stop_price:
                    outcome = 0   # stop hit
                    break
                if future["high"] >= target_price:
                    outcome = 1   # target hit
                    break

        records.append({
            "timestamp":            ts,
            # Level quality
            "touch_count":          touch_count,
            "level_age_bars":       min(level_age_bars, 500),
            "departure_height":     departure_height,
            "origin_departure":     origin_departure,
            # Approach
            "approach_drop_atr":    approach_drop,
            "approach_consec_red":  approach_consec_red,
            "approach_vol_ratio":   approach_vol_ratio,
            # Retest candle
            "close_above_level":    close_above_level,
            "wick_touched_level":   wick_touched_level,
            "wick_body_ratio":      wick_body_ratio,
            "close_pos_range":      close_pos_range,
            "precision":            precision,
            "body_atr":             body / atr_val,
            # Momentum
            "rsi":                  rsi_val,
            "pct_from_ema20":       pct_ema20,
            "pct_from_ema50":       pct_ema50,
            "pct_from_ema200":      pct_ema200,
            "atr_percentile":       atr_pct,
            # HTF
            "htf_trend":            htf_trend,
            "htf_pct_ema20":        htf_pct_ema20,
            "htf_confluence":       htf_confluence,
            # Session
            "session":              session,
            "hour":                 hour,
            "day_of_week":          day_of_week,
            # Structure
            "dist_round_number":    dist_round,
            # Entry geometry
            "risk_atr":             risk / atr_val if atr_val > 0 else 0,
            # Label
            "outcome":              outcome,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Collect
# ─────────────────────────────────────────────────────────────────────────────

def collect(symbol: str, tf_label: str, n_bars: int = 10000):
    tf  = TF_LABEL_MAP[tf_label]
    htf = HTF_MAP.get(tf, TIMEFRAME_H4)

    connect()
    print(f"[COLLECT] Pulling {n_bars} bars for {symbol} {tf_label} ...")
    df     = get_bars(symbol, tf,  n=n_bars)
    htf_df = get_bars(symbol, htf, n=n_bars // 4)
    mt5.shutdown()

    print("[COLLECT] Finding swing lows ...")
    swings = find_swing_lows(df)
    print(f"[COLLECT] Found {len(swings)} swing lows")

    print("[COLLECT] Detecting retests ...")
    retests = find_retests(df, swings)
    print(f"[COLLECT] Found {len(retests)} level retests")

    print("[COLLECT] Extracting features ...")
    features = extract_features(df, htf_df, retests)
    labeled  = features.dropna(subset=["outcome"])

    print(f"[COLLECT] Total retests: {len(features)} | Labeled: {len(labeled)}")
    print(f"[COLLECT] Wins: {int(labeled['outcome'].sum())} | "
          f"Losses: {int((labeled['outcome']==0).sum())}")

    features.to_csv(DATASET_PATH, index=False)
    print(f"[COLLECT] Saved → {DATASET_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Train
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset_path: str = DATASET_PATH):
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import classification_report

    df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df.dropna(subset=["outcome"], inplace=True)
    df["outcome"] = df["outcome"].astype(int)
    df.sort_values("timestamp", inplace=True)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "outcome")]
    X = df[feature_cols]
    y = df["outcome"]

    print(f"[TRAIN] Dataset: {len(df)} retests  |  wins: {y.sum()}  losses: {(y==0).sum()}")
    print(f"[TRAIN] Base win rate: {y.mean():.1%}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,       # prevents overfitting on small samples
        scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"[TRAIN] 5-fold CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X, y)

    imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n[TRAIN] Top 15 features:")
    print(imp.head(15).to_string())

    preds = model.predict(X)
    print("\n[TRAIN] In-sample classification report:")
    print(classification_report(y, preds, target_names=["loss", "win"]))

    joblib.dump({"model": model, "features": feature_cols}, MODEL_PATH)
    print(f"\n[TRAIN] Model saved → {MODEL_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Validate — walk-forward OOS
# ─────────────────────────────────────────────────────────────────────────────

def validate(dataset_path: str = DATASET_PATH, n_splits: int = 5,
             threshold: float = SCORE_THRESH):
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

    print(f"\n[VALIDATE] {n} labeled retests — walk-forward {n_splits}-fold OOS")
    print(f"[VALIDATE] Threshold: {threshold:.0%}  |  R target: {R_TARGET}R\n")

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
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
            scale_pos_weight=(y_train == 0).sum() / max(y_train.sum(), 1),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        model.fit(X_train, y_train)
        oos_probs[test_start:test_end] = model.predict_proba(X_test)[:, 1]

        w = y[test_start:test_end].sum()
        l = (y[test_start:test_end] == 0).sum()
        print(f"  Fold {fold}: train={train_end} | test={test_end-test_start} "
              f"({w}W/{l}L)")

    oos_mask = ~np.isnan(oos_probs)
    df_oos   = df[oos_mask].copy()
    df_oos["prob"] = oos_probs[oos_mask]

    def metrics(subset, label):
        if len(subset) == 0:
            print(f"\n  {label}: no trades"); return
        w   = (subset["outcome"] == 1).sum()
        l   = (subset["outcome"] == 0).sum()
        tot = len(subset)
        wr  = w / tot
        gp  = w * R_TARGET
        gl  = float(l)
        pf  = gp / gl if gl > 0 else float("inf")
        exp = (gp - gl) / tot
        print(f"\n  ── {label} ──")
        print(f"     Trades        : {tot}")
        print(f"     Win Rate      : {wr:.1%}")
        print(f"     Profit Factor : {pf:.3f}")
        print(f"     Net R         : {gp-gl:+.2f}R")
        print(f"     Expectancy    : {exp:+.4f}R/trade")

    metrics(df_oos, "UNFILTERED  (all OOS retests)")
    for t in [0.50, 0.55, 0.60, 0.65, 0.70]:
        metrics(df_oos[df_oos["prob"] >= t], f"ML FILTERED (prob >= {t:.0%})")

    print("\n\n[VALIDATE] Threshold sensitivity table (OOS):")
    print(f"  {'Threshold':>10} {'Trades':>8} {'WinRate':>9} "
          f"{'ProfitFactor':>14} {'ExpectancyR':>13}")
    print("  " + "-" * 58)
    for t in np.arange(0.40, 0.85, 0.05):
        sub = df_oos[df_oos["prob"] >= t]
        if len(sub) == 0:
            continue
        w  = (sub["outcome"] == 1).sum()
        l  = (sub["outcome"] == 0).sum()
        wr = w / len(sub)
        pf = (w * R_TARGET) / l if l > 0 else float("inf")
        ex = (w * R_TARGET - l) / len(sub)
        print(f"  {t:>10.0%} {len(sub):>8} {wr:>9.1%} {pf:>14.3f} {ex:>+13.4f}R")

    print(f"\n[VALIDATE] Done — {len(df_oos)} OOS retests evaluated.")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Scan — live levels being approached right now
# ─────────────────────────────────────────────────────────────────────────────

def scan(symbol: str, tf_label: str, n_bars: int = 3000,
         approach_atr: float = 2.0, top: int = 20, min_score: float = 0.0):
    """
    Finds all active support levels and checks if current price
    is within approach_atr * ATR of any level.
    Scores each approaching level with the ML model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run --mode train first.")

    saved      = joblib.load(MODEL_PATH)
    model      = saved["model"]
    feat_cols  = saved["features"]

    tf  = TF_LABEL_MAP[tf_label]
    htf = HTF_MAP.get(tf, TIMEFRAME_H4)

    connect()
    df     = get_bars(symbol, tf,  n=n_bars)
    htf_df = get_bars(symbol, htf, n=n_bars // 4)
    mt5.shutdown()

    current_price = df["close"].iloc[-1]
    current_atr   = calc_atr(df, ATR_PERIOD).iloc[-1]

    print(f"\n[SCAN] {symbol} {tf_label} | "
          f"Price: {current_price:.2f} | ATR: {current_atr:.2f}")

    # Find all swing lows
    swings = find_swing_lows(df)

    # Filter: level must still be active (not broken by a close below it)
    # and within scanning range of current price
    active_levels = []
    closes = df["close"].values

    for _, sw in swings.iterrows():
        level = sw["price"]

        # Skip levels too far from current price
        if abs(current_price - level) > 20 * current_atr:
            continue

        # Skip levels above current price (those are resistance, not support)
        if level > current_price + current_atr:
            continue

        # Check if level has been broken (any close below level - zone)
        origin_i   = int(sw["bar_index"])
        zone_lo    = level - LEVEL_ZONE_ATR * sw["atr_at_formation"]
        subsequent = closes[origin_i + 1:]
        if len(subsequent) > 0 and np.any(subsequent < zone_lo):
            continue

        active_levels.append(sw)

    if not active_levels:
        print("[SCAN] No active levels found near current price.")
        return

    print(f"[SCAN] {len(active_levels)} active support levels found\n")

    # For each active level, simulate a retest feature row at current bar
    retests_now = []
    last_i = len(df) - 1

    for sw in active_levels:
        level    = sw["price"]
        origin_i = int(sw["bar_index"])

        dist_atr = (current_price - level) / current_atr
        zone_width = LEVEL_ZONE_ATR * sw["atr_at_formation"]

        # count prior touches
        touch_count = 0
        in_zone = False
        for i in range(origin_i + 1, last_i):
            in_z = abs(df["low"].iloc[i] - level) <= zone_width * 2
            if in_z and not in_zone:
                touch_count += 1
            in_zone = in_z

        retests_now.append({
            "level_price":        level,
            "level_origin_i":     origin_i,
            "level_origin_ts":    sw["timestamp"],
            "retest_bar_i":       last_i,
            "retest_ts":          df.index[last_i],
            "touch_count":        touch_count,
            "bars_since_origin":  last_i - origin_i,
            "bars_outside":       5,
            "max_high_since_dep": current_price + current_atr,
            "atr_at_retest":      current_atr,
            "atr_at_formation":   sw["atr_at_formation"],
            "open":   df["open"].iloc[last_i],
            "high":   df["high"].iloc[last_i],
            "low":    df["low"].iloc[last_i],
            "close":  df["close"].iloc[last_i],
            "volume": df["volume"].iloc[last_i],
            "dist_atr": dist_atr,
        })

    features = extract_features(df, htf_df, retests_now, live_mode=True)
    if features.empty:
        print("[SCAN] Could not extract features for current levels.")
        return

    # Fill any missing cols
    for col in feat_cols:
        if col not in features.columns:
            features[col] = 0

    X     = features[feat_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    results = []
    for idx, r in enumerate(retests_now[:len(probs)]):
        results.append({
            "Level":       f"{r['level_price']:.2f}",
            "Distance":    f"{r['dist_atr']:.2f} ATR",
            "Age (bars)":  last_i - r["level_origin_i"],
            "Touches":     r["touch_count"],
            "ML Score":    f"{probs[idx]:.1%}",
            "Signal":      "✅ BUY ZONE" if probs[idx] >= SCORE_THRESH else "⚠️  WATCH",
        })

    results_df = pd.DataFrame(results).sort_values("ML Score", ascending=False)

    # Apply min_score filter and top-N limit
    if min_score > 0:
        results_df = results_df[results_df["ML Score"].str.rstrip("%").astype(float) >= min_score * 100]
    results_df = results_df.head(top)

    approaching = results_df[results_df["Distance"].str.split().str[0].astype(float) <= 3.0]
    print(f"Showing top {len(results_df)} levels | "
          f"{len(approaching)} within 3 ATR of current price\n")

    print(f"{'Level':<12} {'Distance':<14} {'Age':>8} {'Touches':>8} "
          f"{'ML Score':>10} {'Signal':<14}")
    print("-" * 70)
    for _, row in results_df.iterrows():
        dist_val = float(row['Distance'].split()[0])
        marker = " ◀ CLOSE" if dist_val <= 2.0 else ""
        print(f"{row['Level']:<12} {row['Distance']:<14} {row['Age (bars)']:>8} "
              f"{row['Touches']:>8} {row['ML Score']:>10} {row['Signal']:<14}{marker}")

    print(f"\n[SCAN] Current price: {current_price:.2f} | "
          f"ATR: {current_atr:.2f} | "
          f"Threshold: {SCORE_THRESH:.0%}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. Entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 11. Study — SL/TP optimisation via walk-forward OOS bar simulation
# ─────────────────────────────────────────────────────────────────────────────

def study(symbol="@MNQ", tf_label="H1", dataset_path=DATASET_PATH,
          min_score=0.65, n_bars=20000, max_fwd_bars=48):
    """
    Proper SL/TP study using actual bar data from MT5.

    Steps:
      1. Walk-forward score each retest OOS (no data leakage)
      2. For each OOS signal above min_score, simulate every SL x TP combo
         bar-by-bar on actual OHLC data from MT5
      3. Report real win rates, profit factors, MFE/MAE distributions
    """
    from xgboost import XGBClassifier

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run --mode train first.")

    # ── Load dataset ──────────────────────────────────────────────────────────
    df_feat = pd.read_csv(dataset_path, parse_dates=["timestamp"])
    df_feat.dropna(subset=["outcome"], inplace=True)
    df_feat["outcome"] = df_feat["outcome"].astype(int)
    df_feat.sort_values("timestamp", inplace=True)
    df_feat.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df_feat.columns if c not in ("timestamp", "outcome")]

    # ── Walk-forward OOS scoring ──────────────────────────────────────────────
    n_splits  = 5
    n         = len(df_feat)
    fold_size = n // n_splits
    oos_probs = np.full(n, np.nan)

    print(f"[STUDY] Walk-forward scoring {n} retests ...")
    for fold in range(1, n_splits):
        train_end  = fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n)
        if test_end <= test_start:
            break
        X_tr = df_feat[feature_cols].values[:train_end]
        y_tr = df_feat["outcome"].values[:train_end]
        X_te = df_feat[feature_cols].values[test_start:test_end]
        if len(np.unique(y_tr)) < 2:
            continue
        m = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
            scale_pos_weight=(y_tr == 0).sum() / max(y_tr.sum(), 1),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        oos_probs[test_start:test_end] = m.predict_proba(X_te)[:, 1]

    df_feat["oos_score"] = oos_probs
    signals = df_feat[~df_feat["oos_score"].isna() & (df_feat["oos_score"] >= min_score)].copy()
    print(f"[STUDY] {len(signals)} OOS signals at score >= {min_score:.0%}")

    if len(signals) < 15:
        print("[STUDY] Too few OOS signals. Lower --threshold or collect more data.")
        return

    # ── Pull raw bars from MT5 ────────────────────────────────────────────────
    tf = TF_LABEL_MAP[tf_label]
    connect()
    print(f"[STUDY] Pulling {n_bars} bars from MT5 ...")
    bars = get_bars(symbol, tf, n=n_bars)
    mt5.shutdown()

    atr_series = calc_atr(bars, ATR_PERIOD)

    # ── Parameter grid ────────────────────────────────────────────────────────
    sl_atrs = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50]
    tp_rs   = [0.5,  1.0,  1.5,  2.0,  2.5,  3.0]

    combo_results = {(sl, tp): [] for sl in sl_atrs for tp in tp_rs}
    mfe_ref_list  = []
    mae_ref_list  = []
    resolved = 0
    skipped  = 0

    # ── Bar-by-bar simulation ─────────────────────────────────────────────────
    for _, sig in signals.iterrows():
        ts      = sig["timestamp"]
        atr_val = sig.get("atr_at_retest", 0)

        # Locate bar in raw data
        pos = bars.index.searchsorted(ts)
        if pos >= len(bars) - max_fwd_bars - 1:
            skipped += 1
            continue

        # Verify timestamp match (within 1 bar)
        if abs((bars.index[pos] - ts).total_seconds()) > 7200:
            skipped += 1
            continue

        if atr_val <= 0:
            atr_val = atr_series.iloc[pos]
        if atr_val <= 0:
            skipped += 1
            continue

        # Entry = open of next bar after signal
        entry_bar_i = pos + 1
        if entry_bar_i >= len(bars):
            skipped += 1
            continue
        entry_price = bars["open"].iloc[entry_bar_i]
        fwd         = bars.iloc[entry_bar_i: entry_bar_i + max_fwd_bars]

        if len(fwd) < 3:
            skipped += 1
            continue

        # Simulate each combo
        for sl_atr in sl_atrs:
            sl_dist = sl_atr * atr_val
            for tp_r in tp_rs:
                tp_dist   = tp_r * sl_dist
                stop_px   = entry_price - sl_dist
                target_px = entry_price + tp_dist

                outcome_r = None
                trade_mfe = 0.0
                trade_mae = 0.0
                bars_held = 0

                for _, fb in fwd.iterrows():
                    bars_held += 1
                    up   = fb["high"] - entry_price
                    down = entry_price - fb["low"]
                    if up   > trade_mfe: trade_mfe = up
                    if down > trade_mae: trade_mae = down

                    if fb["low"] <= stop_px:
                        outcome_r = -1.0
                        break
                    if fb["high"] >= target_px:
                        outcome_r = float(tp_r)
                        break

                if outcome_r is None:
                    last_close = fwd["close"].iloc[-1]
                    outcome_r  = (last_close - entry_price) / sl_dist

                combo_results[(sl_atr, tp_r)].append(outcome_r)

                if sl_atr == 0.5 and tp_r == 1.5:
                    mfe_ref_list.append(trade_mfe / atr_val)
                    mae_ref_list.append(trade_mae / atr_val)

        resolved += 1

    print(f"[STUDY] Resolved {resolved} signals | Skipped {skipped}\n")

    if resolved == 0:
        print("[STUDY] No signals resolved. Try --bars 50000.")
        return

    # ── Build results table ───────────────────────────────────────────────────
    rows = []
    for sl_atr in sl_atrs:
        for tp_r in tp_rs:
            outcomes = np.array(combo_results[(sl_atr, tp_r)])
            if len(outcomes) == 0:
                continue
            wins   = int((outcomes > 0).sum())
            losses = int((outcomes <= 0).sum())
            total  = len(outcomes)
            wr     = wins / total
            gp     = float(outcomes[outcomes > 0].sum())
            gl     = float(abs(outcomes[outcomes <= 0].sum()))
            pf     = gp / gl if gl > 0 else float("inf")
            net_r  = gp - gl
            exp_r  = net_r / total
            rows.append({
                "sl_atr": sl_atr, "tp_r": tp_r,
                "trades": total, "wins": wins, "losses": losses,
                "win_rate": wr, "profit_factor": pf,
                "net_r": net_r, "expectancy_r": exp_r,
            })

    res = pd.DataFrame(rows)

    SEP = "=" * 72

    def heatmap(metric, title, fmt_fn):
        print(f"\n{SEP}\n{title}\n{SEP}")
        hdr = f"{'SL/TP':>8}" + "".join(f"{r:>9.1f}R" for r in tp_rs)
        print(hdr)
        print("-" * len(hdr))
        for sl in sl_atrs:
            vals = []
            for tp in tp_rs:
                m = res[(res.sl_atr == sl) & (res.tp_r == tp)]
                vals.append(m[metric].values[0] if len(m) > 0 else 0.0)
            best = max(vals)
            row  = f"{sl:>6.2f}x"
            for j, v in enumerate(vals):
                star = "*" if v == best else " "
                row += f"{star}{fmt_fn(v):>8}"
            print(row)

    heatmap("profit_factor", "PROFIT FACTOR  (rows=SL x ATR, cols=TP x R)",
            lambda v: "   inf" if v == float("inf") else f"{v:.2f}")
    heatmap("expectancy_r",  "EXPECTANCY  (R per trade)",
            lambda v: f"{v:+.3f}")
    heatmap("win_rate",      "WIN RATE",
            lambda v: f"{v:.1%}")

    # ── MFE / MAE distribution ────────────────────────────────────────────────
    if mfe_ref_list:
        mfe_arr = np.array(mfe_ref_list)
        mae_arr = np.array(mae_ref_list)
        print(f"\n{SEP}")
        print("MFE / MAE DISTRIBUTION  (SL=0.5 ATR, TP=1.5R as reference)")
        print(SEP)
        for pct in [25, 50, 75, 90, 95]:
            print(f"  p{pct:<3}  MFE: {np.percentile(mfe_arr, pct):.2f} ATR  |  "
                  f"MAE: {np.percentile(mae_arr, pct):.2f} ATR")
        print()
        mfe50 = np.percentile(mfe_arr, 50)
        mae75 = np.percentile(mae_arr, 75)
        if mae75 < 0.40:
            print("  Insight: MAE p75 < 0.4 ATR — tight stop (0.25-0.5 ATR) is survivable")
        elif mae75 < 0.75:
            print("  Insight: MAE p75 suggests 0.5-0.75 ATR stop is appropriate")
        else:
            print("  Insight: MAE p75 > 0.75 ATR — wider stop needed to avoid noise")
        if mfe50 > 2.0:
            print("  Insight: MFE p50 > 2 ATR — price often runs past 1.5R; "
                  "consider 2.0-2.5R target")
        elif mfe50 < 1.0:
            print("  Insight: MFE p50 < 1 ATR — short targets (0.5-1.0R) may outperform")

    # ── Top 10 ────────────────────────────────────────────────────────────────
    top10 = res.nlargest(10, "expectancy_r")
    print(f"\n{SEP}")
    print("TOP 10 COMBINATIONS BY EXPECTANCY  (OOS bar-by-bar simulation)")
    print(SEP)
    print(f"  {'SL (ATR)':>10} {'TP (R)':>8} {'Trades':>8} {'WinRate':>9} "
          f"{'PF':>8} {'Expect':>10} {'Net R':>8}")
    print("  " + "-" * 67)
    for _, r in top10.iterrows():
        pf_str = "     inf" if r.profit_factor == float("inf") else f"{r.profit_factor:>8.3f}"
        print(f"  {r.sl_atr:>10.2f} {r.tp_r:>8.1f} {r.trades:>8} "
              f"{r.win_rate:>9.1%} {pf_str} "
              f"{r.expectancy_r:>+10.4f} {r.net_r:>+8.2f}R")

    best    = res.nlargest(1, "expectancy_r").iloc[0]
    cur_atr = atr_series.iloc[-1]
    sl_pts  = best.sl_atr * cur_atr
    tp_pts  = best.tp_r * sl_pts

    print(f"\n{SEP}")
    print("RECOMMENDED PARAMETERS  (based on OOS bar-by-bar simulation)")
    print(SEP)
    print(f"  Best SL  : {best.sl_atr:.2f} x ATR below level")
    print(f"  Best TP  : {best.tp_r:.1f}R  ({best.tp_r:.1f} x SL distance)")
    print(f"  Win Rate : {best.win_rate:.1%}")
    print(f"  PF       : {best.profit_factor:.3f}" if best.profit_factor != float("inf")
          else "  PF       : inf")
    print(f"  Expect   : {best.expectancy_r:+.4f}R per trade")
    print(f"\n  For 1 MNQ contract @ current ATR {cur_atr:.0f} pts:")
    print(f"  SL : ~{sl_pts:.0f} pts  (${sl_pts * 2:.0f} risk per trade)")
    print(f"  TP : ~{tp_pts:.0f} pts  (${tp_pts * 2:.0f} target per trade)")
    print(f"\n[STUDY] Based on {resolved} OOS signals on actual OHLC bar data.")

    res.to_csv("study_results.csv", index=False)
    print("[STUDY] Full results saved -> study_results.csv")





# ─────────────────────────────────────────────────────────────────────────────
# 12. Backtest — long + optional short with daily 100 EMA trend filter
# ─────────────────────────────────────────────────────────────────────────────

def find_swing_highs(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.DataFrame:
    """
    Mirror of find_swing_lows for resistance levels.
    A swing high at index i: df['high'][i] is the maximum of
    the window [i-lookback : i+lookback+1].
    """
    highs    = df["high"].values
    atr_vals = calc_atr(df, ATR_PERIOD).values
    swings   = []
    for i in range(lookback, len(df) - lookback):
        window = highs[i - lookback: i + lookback + 1]
        if highs[i] == window.max() and list(window).count(highs[i]) == 1:
            swings.append({
                "bar_index":        i,
                "timestamp":        df.index[i],
                "price":            highs[i],
                "atr_at_formation": atr_vals[i],
            })
    return pd.DataFrame(swings)


def backtest(symbol="@MNQ", tf_label="H1", n_bars=20000,
             min_score=0.65, max_dist_atr=1.5,
             sl_atr=1.0, tp_r=2.0, max_bars_in_trade=48,
             show_last_n=10, enable_short=False):
    """
    Bar-by-bar backtest — identical logic to live bot.

    Direction rules:
      LONG  : daily close > daily 100 EMA  →  buy support levels
      SHORT : daily close < daily 100 EMA  →  sell resistance levels
               (only when enable_short=True)

    Both sides use the same ML model and level-detection logic,
    just mirrored: highs become resistance, approach from below.

    Single-position rule: no new trade while any position is open.
    """
    from xgboost import XGBClassifier

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No model at {MODEL_PATH}. Run --mode train first.")

    tf   = TF_LABEL_MAP[tf_label]
    htf  = HTF_MAP.get(tf, TIMEFRAME_H4)
    D1   = TIMEFRAME_D1

    # ── Pull bars ─────────────────────────────────────────────────────────────
    connect()
    print(f"[BT] Pulling {n_bars} bars for {symbol} {tf_label} ...")
    bars      = get_bars(symbol, tf,  n=n_bars)
    htf_bars  = get_bars(symbol, htf, n=n_bars // 4)
    daily     = get_bars(symbol, D1,  n=1000)
    mt5.shutdown()

    # ── Daily 100 EMA — trend filter ─────────────────────────────────────────
    daily_ema100 = calc_ema(daily["close"], 100)

    def daily_trend(ts):
        """Return 1 (bullish) or -1 (bearish) based on daily 100 EMA at ts."""
        prior = daily[daily.index <= ts]
        if len(prior) < 101:
            return 0   # not enough daily bars yet
        d_close  = prior["close"].iloc[-1]
        d_ema100 = daily_ema100.reindex(prior.index).iloc[-1]
        if np.isnan(d_ema100):
            return 0
        return 1 if d_close > d_ema100 else -1

    # ── Precompute indicators on H1 bar set ───────────────────────────────────
    atr_s    = calc_atr(bars, ATR_PERIOD)
    rsi_s    = calc_rsi(bars["close"])
    ema20_s  = calc_ema(bars["close"], 20)
    ema50_s  = calc_ema(bars["close"], 50)
    ema200_s = calc_ema(bars["close"], 200)
    atr_pct  = calc_atr_percentile(atr_s)
    vol_ma   = bars["volume"].rolling(20).mean()
    htf_ema20   = calc_ema(htf_bars["close"], 20)
    htf_ema50   = calc_ema(htf_bars["close"], 50)
    htf_atr_s   = calc_atr(htf_bars, ATR_PERIOD)

    # ── Walk-forward model training ───────────────────────────────────────────
    RETRAIN_EVERY = 200
    print("[BT] Pre-training walk-forward models ...")
    saved     = joblib.load(MODEL_PATH)
    feat_cols = saved["features"]

    df_feat = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    df_feat.dropna(subset=["outcome"], inplace=True)
    df_feat["outcome"] = df_feat["outcome"].astype(int)
    df_feat.sort_values("timestamp", inplace=True)
    df_feat.reset_index(drop=True, inplace=True)

    checkpoints = list(range(RETRAIN_EVERY, len(df_feat), RETRAIN_EVERY))
    if not checkpoints or checkpoints[-1] < len(df_feat):
        checkpoints.append(len(df_feat))

    trained_models = []
    for cp in checkpoints:
        subset = df_feat.iloc[:cp]
        X_cp   = subset[feat_cols].values
        y_cp   = subset["outcome"].values
        if len(np.unique(y_cp)) < 2 or len(y_cp) < 20:
            trained_models.append(None)
            continue
        m = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
            scale_pos_weight=(y_cp == 0).sum() / max(y_cp.sum(), 1),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        m.fit(X_cp, y_cp)
        trained_models.append(m)

    def get_model(bar_ts):
        prior  = df_feat[df_feat["timestamp"] <= bar_ts]
        cp_idx = min(len(prior) // RETRAIN_EVERY, len(trained_models) - 1)
        for i in range(cp_idx, -1, -1):
            if trained_models[i] is not None:
                return trained_models[i]
        return None

    def build_feature_row(i, ts, level, sw, direction, atr_val, close):
        """Build ML feature dict for a candidate level at bar i."""
        body       = abs(bars["close"].iloc[i] - bars["open"].iloc[i])
        bar_low    = bars["low"].iloc[i]
        bar_high   = bars["high"].iloc[i]
        lower_wick = min(bars["open"].iloc[i], bars["close"].iloc[i]) - bar_low
        upper_wick = bar_high - max(bars["open"].iloc[i], bars["close"].iloc[i])
        c_range    = bar_high - bar_low
        origin_i   = int(sw["bar_index"])

        touch_count = 0
        in_z = False
        zone_w = LEVEL_ZONE_ATR * sw["atr_at_formation"]
        for k in range(origin_i + 1, i):
            ref_price = bars["low"].iloc[k] if direction == 1 else bars["high"].iloc[k]
            in_zone_k = abs(ref_price - level) <= zone_w * 2
            if in_zone_k and not in_z:
                touch_count += 1
            in_z = in_zone_k

        if direction == 1:   # long — departure is upward from swing low
            ref_prices = bars["high"].iloc[origin_i + 1: i]
            max_h = ref_prices.max() if len(ref_prices) > 0 else level
            dep_h = (max_h - level) / atr_val if atr_val > 0 else 0
        else:                # short — departure is downward from swing high
            ref_prices = bars["low"].iloc[origin_i + 1: i]
            min_l = ref_prices.min() if len(ref_prices) > 0 else level
            dep_h = (level - min_l) / atr_val if atr_val > 0 else 0

        consec_dir = 0
        for k in range(i - 1, max(0, i - 8), -1):
            if direction == 1:
                if bars["close"].iloc[k] < bars["open"].iloc[k]:
                    consec_dir += 1
                else:
                    break
            else:
                if bars["close"].iloc[k] > bars["open"].iloc[k]:
                    consec_dir += 1
                else:
                    break

        vol_ratio = (bars["volume"].iloc[i] / vol_ma.iloc[i]
                     if vol_ma.iloc[i] > 0 else 1.0)

        htf_prior = htf_bars[htf_bars.index <= ts]
        if len(htf_prior) < 5:
            return None
        htf_c     = htf_prior["close"].iloc[-1]
        htf_e20   = htf_ema20.reindex(htf_prior.index).iloc[-1]
        htf_e50   = htf_ema50.reindex(htf_prior.index).iloc[-1]
        htf_a     = htf_atr_s.reindex(htf_prior.index).iloc[-1]
        htf_trend = 1 if htf_c > htf_e20 else -1
        htf_pct   = (htf_c - htf_e20) / htf_e20 if htf_e20 > 0 else 0
        htf_e20d  = abs(level - htf_e20) / htf_a if htf_a > 0 else 99
        htf_e50d  = abs(level - htf_e50) / htf_a if htf_a > 0 else 99
        htf_conf  = int(min(htf_e20d, htf_e50d) < 0.5)

        round_100  = round(level / 100) * 100
        dist_round = abs(level - round_100) / atr_val
        entry_proxy = close
        risk_proxy  = abs(entry_proxy - (level - sl_atr * atr_val
                          if direction == 1 else level + sl_atr * atr_val))

        # For short side, flip wick interpretation
        if direction == -1:
            lower_wick, upper_wick = upper_wick, lower_wick

        hour    = ts.hour
        session = 0
        if 7  <= hour < 13: session = 1
        if 13 <= hour < 21: session = 2

        return {
            "touch_count":         touch_count,
            "level_age_bars":      min(i - origin_i, 500),
            "departure_height":    dep_h,
            "origin_departure":    dep_h,
            "approach_drop_atr":   abs(bars["close"].iloc[max(0,i-5)] - close) / atr_val,
            "approach_consec_red": consec_dir,
            "approach_vol_ratio":  vol_ratio,
            "close_above_level":   int(close > level) if direction == 1 else int(close < level),
            "wick_touched_level":  int(bar_low <= level + LEVEL_ZONE_ATR * atr_val)
                                   if direction == 1
                                   else int(bar_high >= level - LEVEL_ZONE_ATR * atr_val),
            "wick_body_ratio":     lower_wick / body if body > 0 else 0,
            "close_pos_range":     (close - bar_low) / c_range if c_range > 0 else 0,
            "precision":           abs((bar_low if direction == 1 else bar_high) - level) / atr_val,
            "body_atr":            body / atr_val if atr_val > 0 else 0,
            "rsi":                 rsi_s.iloc[i],
            "pct_from_ema20":      (close - ema20_s.iloc[i]) / ema20_s.iloc[i],
            "pct_from_ema50":      (close - ema50_s.iloc[i]) / ema50_s.iloc[i],
            "pct_from_ema200":     (close - ema200_s.iloc[i]) / ema200_s.iloc[i],
            "atr_percentile":      atr_pct.iloc[i] if not np.isnan(atr_pct.iloc[i]) else 0.5,
            "htf_trend":           htf_trend,
            "htf_pct_ema20":       htf_pct,
            "htf_confluence":      htf_conf,
            "session":             session,
            "hour":                hour,
            "day_of_week":         ts.dayofweek,
            "dist_round_number":   dist_round,
            "risk_atr":            risk_proxy / atr_val if atr_val > 0 else 0,
        }

    # ── Precompute swing lows AND highs ───────────────────────────────────────
    MIN_BAR    = max(250, SWING_LOOKBACK * 2)
    closes_arr = bars["close"].values
    highs_arr  = bars["high"].values

    print("[BT] Precomputing swing lows and highs ...")
    all_lows  = find_swing_lows(bars,  lookback=SWING_LOOKBACK)
    all_highs = find_swing_highs(bars, lookback=SWING_LOOKBACK) if enable_short else pd.DataFrame()
    print(f"[BT] Swing lows: {len(all_lows)}  |  Swing highs: {len(all_highs)}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    position     = None
    trades       = []
    bar_count    = 0
    signal_count = 0

    total_bars = len(bars) - 1 - MIN_BAR
    print(f"[BT] Processing {total_bars:,} bars ...")

    for i in range(MIN_BAR, len(bars) - 1):
        if (i - MIN_BAR) % 1000 == 0 and i > MIN_BAR:
            pct = (i - MIN_BAR) / total_bars * 100
            print(f"[BT] {pct:.0f}% | trades: {len(trades)}", end="\r")

        ts      = bars.index[i]
        close   = bars["close"].iloc[i]
        atr_val = atr_s.iloc[i]
        hour    = ts.hour
        session = 0
        if 7  <= hour < 13: session = 1
        if 13 <= hour < 21: session = 2

        # ── Manage open position ──────────────────────────────────────────────
        if position is not None:
            bar_count += 1
            hi  = bars["high"].iloc[i]
            lo  = bars["low"].iloc[i]
            cl  = bars["close"].iloc[i]
            dir = position["direction"]

            exit_reason = None
            exit_price  = None

            if dir == 1:   # long
                if lo <= position["stop"]:
                    exit_reason, exit_price = "stop",      position["stop"]
                elif hi >= position["target"]:
                    exit_reason, exit_price = "target",    position["target"]
            else:          # short
                if hi >= position["stop"]:
                    exit_reason, exit_price = "stop",      position["stop"]
                elif lo <= position["target"]:
                    exit_reason, exit_price = "target",    position["target"]

            if exit_reason is None and bar_count >= max_bars_in_trade:
                exit_reason, exit_price = "time_stop", cl

            if exit_reason:
                pnl_pts = (exit_price - position["entry"]) * dir
                pnl_r   = pnl_pts / position["risk"]
                pnl_usd = pnl_pts * 2 * dir

                trades.append({
                    "direction":    "LONG" if dir == 1 else "SHORT",
                    "entry_time":   position["entry_time"],
                    "exit_time":    ts,
                    "level":        position["level"],
                    "ml_score":     position["ml_score"],
                    "entry":        position["entry"],
                    "stop":         position["stop"],
                    "target":       position["target"],
                    "exit_price":   exit_price,
                    "exit_reason":  exit_reason,
                    "pnl_pts":      round(pnl_pts, 2),
                    "pnl_r":        round(pnl_r, 3),
                    "pnl_usd":      round(pnl_usd, 2),
                    "bars_held":    bar_count,
                })
                position  = None
                bar_count = 0
            continue

        if session == 0:
            continue

        # ── Daily trend filter ────────────────────────────────────────────────
        trend = daily_trend(ts)
        if trend == 0:
            continue

        # ── Determine which directions are allowed this bar ───────────────────
        allowed_directions = []
        if trend == 1:
            allowed_directions.append((1,  all_lows))    # bullish → longs only
        if trend == -1 and enable_short:
            allowed_directions.append((-1, all_highs))   # bearish → shorts only

        if not allowed_directions:
            continue

        model = get_model(ts)
        if model is None:
            continue

        best_score = 0.0
        best_level = None
        best_sw    = None
        best_dir   = None

        for direction, swing_df in allowed_directions:
            if swing_df.empty:
                continue

            prior_swings = swing_df[swing_df["bar_index"] <= i - SWING_LOOKBACK]

            for _, sw in prior_swings.iterrows():
                level    = sw["price"]
                origin_i = int(sw["bar_index"])
                zone_lo  = level - LEVEL_ZONE_ATR * sw["atr_at_formation"]
                zone_hi  = level + LEVEL_ZONE_ATR * sw["atr_at_formation"]

                if direction == 1:
                    # Long: level must be below price, not broken (no close below zone)
                    subsequent = closes_arr[origin_i + 1: i + 1]
                    if len(subsequent) > 0 and np.any(subsequent < zone_lo):
                        continue
                    dist_atr = (close - level) / atr_val
                else:
                    # Short: level must be above price, not broken (no close above zone)
                    subsequent = closes_arr[origin_i + 1: i + 1]
                    if len(subsequent) > 0 and np.any(subsequent > zone_hi):
                        continue
                    dist_atr = (level - close) / atr_val

                if not (0 <= dist_atr <= max_dist_atr):
                    continue

                feat = build_feature_row(i, ts, level, sw, direction, atr_val, close)
                if feat is None:
                    continue

                X_row = np.array([[feat.get(c, 0) for c in feat_cols]])
                score = model.predict_proba(X_row)[0, 1]

                if score > best_score:
                    best_score = score
                    best_level = level
                    best_sw    = sw
                    best_dir   = direction

        # ── Enter if best signal clears threshold ─────────────────────────────
        if best_score >= min_score and best_level is not None:
            signal_count += 1
            next_open     = bars["open"].iloc[i + 1]
            sl_price      = (best_level - sl_atr * atr_val if best_dir == 1
                             else best_level + sl_atr * atr_val)
            risk          = abs(next_open - sl_price)
            if risk <= 0:
                continue
            tp_price = (next_open + tp_r * risk if best_dir == 1
                        else next_open - tp_r * risk)

            position = {
                "direction":  best_dir,
                "entry_time": bars.index[i + 1],
                "entry":      next_open,
                "stop":       sl_price,
                "target":     tp_price,
                "risk":       risk,
                "level":      best_level,
                "ml_score":   round(best_score, 4),
            }
            bar_count = 0

    # ── Results ───────────────────────────────────────────────────────────────
    if not trades:
        print(f"\n[BT] No trades. Signals: {signal_count}")
        return

    df_t  = pd.DataFrame(trades)
    longs  = df_t[df_t["direction"] == "LONG"]
    shorts = df_t[df_t["direction"] == "SHORT"]

    def print_stats(subset, label):
        if len(subset) == 0:
            print(f"\n  {label}: no trades")
            return
        wins   = (subset["pnl_r"] > 0).sum()
        losses = (subset["pnl_r"] <= 0).sum()
        total  = len(subset)
        wr     = wins / total
        gp     = subset[subset["pnl_r"] > 0]["pnl_r"].sum()
        gl     = subset[subset["pnl_r"] <= 0]["pnl_r"].abs().sum()
        pf     = gp / gl if gl > 0 else float("inf")
        net_r  = gp - gl
        exp_r  = net_r / total
        net_u  = subset["pnl_usd"].sum()
        cum    = subset["pnl_usd"].cumsum()
        dd     = (cum - cum.cummax()).min()
        print(f"\n  ── {label} ──")
        print(f"     Trades       : {total}  ({wins}W / {losses}L)")
        print(f"     Win Rate     : {wr:.1%}")
        print(f"     Profit Factor: {pf:.3f}")
        print(f"     Net R        : {net_r:+.2f}R")
        print(f"     Expectancy   : {exp_r:+.4f}R/trade")
        print(f"     Net P&L      : ${net_u:+,.2f}")
        print(f"     Max Drawdown : ${dd:,.2f}")
        for reason in ["target", "stop", "time_stop"]:
            sub = subset[subset["exit_reason"] == reason]
            if len(sub):
                print(f"     {reason:<12}: {len(sub):>4} trades  avg {sub['pnl_r'].mean():+.3f}R")

    SEP = "=" * 72
    print(f"\n{SEP}")
    mode_str = "LONG + SHORT" if enable_short else "LONG ONLY"
    print(f"BACKTEST RESULTS — {symbol} {tf_label}  |  {mode_str}")
    print(f"{bars.index[MIN_BAR].date()} → {bars.index[-1].date()}")
    print(SEP)
    print(f"  ML threshold : {min_score:.0%}  |  SL: {sl_atr:.2f}×ATR  |  "
          f"TP: {tp_r:.1f}R  |  Daily 100 EMA filter: ON")
    if enable_short:
        print(f"  LONG  when daily close > 100 EMA  (bullish trend)")
        print(f"  SHORT when daily close < 100 EMA  (bearish trend)")
    else:
        print(f"  LONG only when daily close > 100 EMA")

    print_stats(df_t,    "COMBINED")
    if enable_short:
        print_stats(longs,  "LONGS  (support bounce, bullish trend)")
        print_stats(shorts, "SHORTS (resistance rejection, bearish trend)")

    # ── Last N trades ─────────────────────────────────────────────────────────
    n = min(show_last_n, len(df_t))
    print(f"\n{SEP}")
    print(f"LAST {n} TRADES")
    print(SEP)
    print(f"  {'Dir':<6} {'Entry Time':<22} {'Exit Time':<22} "
          f"{'Level':>10} {'Score':>7} {'Entry':>10} "
          f"{'SL':>10} {'TP':>10} {'Exit':>10} "
          f"{'R':>7} {'$':>8} {'Bars':>5} {'Result':<16}")
    print("  " + "-" * 155)

    for _, t in df_t.tail(n).iterrows():
        mark = "✅" if t["pnl_r"] > 0 else "❌"
        d    = "▲ L" if t["direction"] == "LONG" else "▼ S"
        print(f"  {d:<6} {str(t['entry_time']):<22} {str(t['exit_time']):<22} "
              f"{t['level']:>10.2f} {t['ml_score']:>7.1%} "
              f"{t['entry']:>10.2f} {t['stop']:>10.2f} {t['target']:>10.2f} "
              f"{t['exit_price']:>10.2f} "
              f"{t['pnl_r']:>+7.3f} {t['pnl_usd']:>+8.2f} "
              f"{t['bars_held']:>5}  {mark} {t['exit_reason']:<12}")

    df_t.to_csv("backtest_trades.csv", index=False)
    print(f"\n[BT] Saved → backtest_trades.csv")
    print(f"[BT] Single-position rule: {signal_count - len(df_t)} signals skipped while in trade.")




# ─────────────────────────────────────────────────────────────────────────────
# 13. Retrain — automated monthly retraining with AUC degradation check
# ─────────────────────────────────────────────────────────────────────────────

def retrain(symbol="@MNQ", tf_label="H1", n_bars=50000,
            min_auc=0.65, auc_drop_threshold=0.03):
    """
    Automated retraining pipeline. Run this every 4-6 weeks.

    Steps:
      1. Collect fresh data (n_bars of history)
      2. Train new model on full dataset
      3. Walk-forward OOS validation
      4. Compare new AUC vs saved baseline AUC
      5. If AUC >= min_auc AND hasn't dropped > auc_drop_threshold → save new model
      6. If degraded → warn and keep old model, save report

    Outputs:
      levels_xgb_model.joblib   updated if AUC check passes
      levels_dataset.csv        always updated with fresh data
      retrain_report.json       full metrics for audit trail
    """
    from xgboost import XGBClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    REPORT_PATH   = "retrain_report.json"
    BASELINE_PATH = "retrain_baseline.json"

    print("\n" + "=" * 65)
    print("RETRAIN PIPELINE")
    print("=" * 65)

    # ── Step 1: Collect fresh data ────────────────────────────────────────────
    print(f"\n[RETRAIN] Step 1: Collecting {n_bars} bars for {symbol} {tf_label} ...")
    collect(symbol, tf_label, n_bars=n_bars)

    # ── Step 2: Load dataset ──────────────────────────────────────────────────
    df = pd.read_csv(DATASET_PATH, parse_dates=["timestamp"])
    df.dropna(subset=["outcome"], inplace=True)
    df["outcome"] = df["outcome"].astype(int)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    feature_cols = [c for c in df.columns if c not in ("timestamp", "outcome")]
    X = df[feature_cols]
    y = df["outcome"]

    n_total = len(df)
    n_wins  = int(y.sum())
    n_loss  = int((y == 0).sum())
    base_wr = y.mean()

    print(f"\n[RETRAIN] Dataset: {n_total} retests | "
          f"{n_wins} wins | {n_loss} losses | base WR: {base_wr:.1%}")
    print(f"[RETRAIN] Date range: {df['timestamp'].iloc[0].date()} "
          f"→ {df['timestamp'].iloc[-1].date()}")

    if n_total < 100:
        print("[RETRAIN] ⚠️  Less than 100 labeled samples — aborting. "
              "Increase --bars or check data.")
        return

    # ── Step 3: Walk-forward OOS validation ───────────────────────────────────
    print("\n[RETRAIN] Step 2: Walk-forward OOS validation ...")

    n_splits  = 5
    fold_size = n_total // n_splits
    oos_probs = np.full(n_total, np.nan)
    fold_aucs = []

    for fold in range(1, n_splits):
        train_end  = fold * fold_size
        test_start = train_end
        test_end   = min(test_start + fold_size, n_total)
        if test_end <= test_start:
            break

        X_tr = X.values[:train_end]
        y_tr = y.values[:train_end]
        X_te = X.values[test_start:test_end]
        y_te = y.values[test_start:test_end]

        if len(np.unique(y_tr)) < 2:
            continue

        m = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
            scale_pos_weight=(y_tr == 0).sum() / max(y_tr.sum(), 1),
            eval_metric="logloss", random_state=42, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        probs = m.predict_proba(X_te)[:, 1]
        oos_probs[test_start:test_end] = probs

        # Fold AUC
        from sklearn.metrics import roc_auc_score
        try:
            fold_auc = roc_auc_score(y_te, probs)
            fold_aucs.append(fold_auc)
            print(f"  Fold {fold}: train={train_end} | "
                  f"test={test_end - test_start} | AUC={fold_auc:.4f}")
        except Exception:
            pass

    oos_mask = ~np.isnan(oos_probs)
    df_oos   = df[oos_mask].copy()
    df_oos["prob"] = oos_probs[oos_mask]

    new_auc  = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    auc_std  = float(np.std(fold_aucs))  if fold_aucs else 0.0

    # OOS metrics at 65% threshold
    filtered = df_oos[df_oos["prob"] >= 0.65]
    if len(filtered) > 0:
        oos_wr = float((filtered["outcome"] == 1).mean())
        gp     = float((filtered["outcome"] == 1).sum() * 1.5)
        gl     = float((filtered["outcome"] == 0).sum() * 1.0)
        oos_pf = gp / gl if gl > 0 else float("inf")
        oos_exp= (gp - gl) / len(filtered)
        oos_n  = len(filtered)
    else:
        oos_wr = oos_pf = oos_exp = 0.0
        oos_n  = 0

    print(f"\n[RETRAIN] OOS Results (threshold 65%):")
    print(f"  Mean AUC    : {new_auc:.4f} ± {auc_std:.4f}")
    print(f"  OOS trades  : {oos_n}")
    print(f"  Win rate    : {oos_wr:.1%}")
    print(f"  Profit fac  : {oos_pf:.3f}")
    print(f"  Expectancy  : {oos_exp:+.4f}R/trade")

    # ── Step 4: Compare vs baseline ───────────────────────────────────────────
    baseline_auc = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            baseline = json.load(f)
            baseline_auc = baseline.get("auc")

    print(f"\n[RETRAIN] Step 3: AUC check ...")

    auc_ok     = new_auc >= min_auc
    drop_ok    = True
    auc_change = None

    if baseline_auc is not None:
        auc_change = new_auc - baseline_auc
        drop_ok    = auc_change >= -auc_drop_threshold
        print(f"  Baseline AUC : {baseline_auc:.4f}")
        print(f"  New AUC      : {new_auc:.4f}")
        print(f"  Change       : {auc_change:+.4f}")
        if drop_ok:
            print(f"  ✅ AUC change within threshold (>{-auc_drop_threshold:.2f})")
        else:
            print(f"  ⚠️  AUC dropped by {abs(auc_change):.4f} "
                  f"(threshold: {auc_drop_threshold:.2f})")
    else:
        print(f"  No baseline found — this will become the new baseline")

    if not auc_ok:
        print(f"  ❌ AUC {new_auc:.4f} below minimum {min_auc:.4f}")

    # ── Step 5: Train final model on full dataset ─────────────────────────────
    print(f"\n[RETRAIN] Step 4: Training final model on {n_total} samples ...")

    final_model = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
        scale_pos_weight=(y == 0).sum() / max(y.sum(), 1),
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    final_model.fit(X, y)

    imp = pd.Series(final_model.feature_importances_,
                    index=feature_cols).sort_values(ascending=False)
    print("\n  Top 10 features:")
    for feat, val in imp.head(10).items():
        bar = "█" * int(val * 200)
        print(f"  {feat:<28} {val:.4f}  {bar}")

    # ── Step 6: Save decision ─────────────────────────────────────────────────
    print(f"\n[RETRAIN] Step 5: Save decision ...")

    report = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "symbol":          symbol,
        "timeframe":       tf_label,
        "n_bars_pulled":   n_bars,
        "n_samples":       n_total,
        "n_wins":          n_wins,
        "n_losses":        n_loss,
        "date_range_start":str(df["timestamp"].iloc[0].date()),
        "date_range_end":  str(df["timestamp"].iloc[-1].date()),
        "auc":             new_auc,
        "auc_std":         auc_std,
        "auc_change":      auc_change,
        "baseline_auc":    baseline_auc,
        "oos_trades":      oos_n,
        "oos_win_rate":    oos_wr,
        "oos_pf":          oos_pf,
        "oos_expectancy":  oos_exp,
        "model_saved":     False,
        "reason":          "",
    }

    if auc_ok and drop_ok:
        # Back up old model before overwriting
        if os.path.exists(MODEL_PATH):
            backup = MODEL_PATH.replace(".joblib", "_backup.joblib")
            import shutil
            shutil.copy(MODEL_PATH, backup)
            print(f"  Old model backed up → {backup}")

        joblib.dump({"model": final_model, "features": feature_cols}, MODEL_PATH)
        report["model_saved"] = True
        report["reason"]      = "AUC checks passed"

        # Update baseline
        with open(BASELINE_PATH, "w") as f:
            json.dump({"auc": new_auc, "timestamp": report["timestamp"]}, f)

        print(f"  ✅ NEW MODEL SAVED → {MODEL_PATH}")
        print(f"  ✅ Baseline updated → {new_auc:.4f}")

    else:
        reasons = []
        if not auc_ok:
            reasons.append(f"AUC {new_auc:.4f} < min {min_auc:.4f}")
        if not drop_ok:
            reasons.append(f"AUC dropped {abs(auc_change):.4f} > threshold {auc_drop_threshold:.2f}")
        report["reason"] = " | ".join(reasons)

        print(f"  ⚠️  MODEL NOT SAVED — {report['reason']}")
        print(f"  Old model preserved at {MODEL_PATH}")
        print(f"  Review retrain_report.json and decide manually.")

    # Save report
    import json as _json
    with open(REPORT_PATH, "w") as f:
        _json.dump(report, f, indent=2)
    print(f"\n[RETRAIN] Report saved → {REPORT_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("RETRAIN SUMMARY")
    print("=" * 65)
    print(f"  Samples    : {n_total} ({n_wins}W / {n_loss}L)")
    print(f"  OOS AUC    : {new_auc:.4f} ± {auc_std:.4f}")
    print(f"  OOS trades : {oos_n} at 65% threshold")
    print(f"  OOS WR     : {oos_wr:.1%}  PF: {oos_pf:.3f}  Exp: {oos_exp:+.4f}R")
    if report["model_saved"]:
        print(f"\n  ✅ Model updated — copy {MODEL_PATH} to VPS and restart bot")
    else:
        print(f"\n  ⚠️  Model NOT updated — {report['reason']}")
        print(f"     Review report and keep monitoring")
    print("=" * 65)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Key Levels ML Pipeline")
    parser.add_argument("--mode",      choices=["collect", "train", "validate", "scan", "study", "backtest", "retrain"],
                        required=True)
    parser.add_argument("--symbol",    default="@MNQ")
    parser.add_argument("--tf",        default="H1", choices=list(TF_LABEL_MAP.keys()))
    parser.add_argument("--bars",      type=int,   default=10000)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESH,
                        help="ML score threshold (default 0.60)")
    parser.add_argument("--top",       type=int,   default=20,
                        help="Show top N levels in scan (default 20)")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Hide levels below this score in scan (default 0.0 = show all)")
    parser.add_argument("--sl-atr",   type=float, default=1.0,
                        help="Stop loss in ATR multiples (default 1.0)")
    parser.add_argument("--tp-r",    type=float, default=2.0,
                        help="Take profit in R multiples (default 2.0)")
    parser.add_argument("--max-dist",type=float, default=1.5,
                        help="Max distance from level in ATR (default 1.5)")
    parser.add_argument("--last-n",  type=int,   default=10,
                        help="Show last N trades in backtest (default 10)")
    parser.add_argument("--short",    action="store_true", default=False,
                        help="Enable short side (resistance rejection, bearish trend)")
    args = parser.parse_args()

    if args.mode == "collect":
        collect(args.symbol, args.tf, args.bars)
    elif args.mode == "train":
        train()
    elif args.mode == "validate":
        validate(threshold=args.threshold)
    elif args.mode == "scan":
        scan(args.symbol, args.tf, top=args.top, min_score=args.min_score)
    elif args.mode == "study":
        study(symbol=args.symbol, tf_label=args.tf, min_score=args.threshold)
    elif args.mode == "retrain":
        retrain(symbol=args.symbol, tf_label=args.tf, n_bars=args.bars)
    elif args.mode == "backtest":
        backtest(
            symbol=args.symbol,
            tf_label=args.tf,
            n_bars=args.bars,
            min_score=args.threshold,
            sl_atr=args.sl_atr,
            tp_r=args.tp_r,
            max_dist_atr=args.max_dist,
            show_last_n=args.last_n,
            enable_short=args.short,
        )
    elif args.mode == "study":
        study(min_score=args.threshold)
