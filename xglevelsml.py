#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from mt5linux import MetaTrader5

MT5_HOST = "127.0.0.1"
MT5_PORT = 18812

MODEL_PATH = "levels_xgb_model.joblib"
ATR_PERIOD = 14
SWING_LOOKBACK = 5
LEVEL_ZONE_ATR = 0.30
MIN_DEPARTURE_BARS = 3
MIN_DEPARTURE_ATR = 1.0

TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16408

TF_LABEL_MAP = {
    "M5": TIMEFRAME_M5,
    "M15": TIMEFRAME_M15,
    "H1": TIMEFRAME_H1,
    "H4": TIMEFRAME_H4,
    "D1": TIMEFRAME_D1,
}

HTF_MAP = {
    TIMEFRAME_M5: TIMEFRAME_M15,
    TIMEFRAME_M15: TIMEFRAME_H1,
    TIMEFRAME_H1: TIMEFRAME_H4,
}

SESSION_WINDOWS_UTC = {
    "london": [(7, 13)],
    "ny": [(13, 21)],
    "both": [(7, 21)],
    "all": [(0, 24)],
}


@dataclass
class ScannerConfig:
    symbol: str
    tf_label: str
    bars: int
    display_threshold: float
    good_threshold: float
    action_threshold: float
    history_threshold: float
    max_dist_atr: float
    top: int
    sl_atr: float
    tp_r: float
    session: str
    loop_seconds: int
    cooldown_minutes: int
    once: bool
    jsonl_log: str | None
    sort_by: str
    print_last_actionable: int


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def in_session(session_name: str, now_utc: datetime) -> bool:
    hour = now_utc.hour
    for start_h, end_h in SESSION_WINDOWS_UTC[session_name]:
        if start_h <= hour < end_h:
            return True
    return False


def session_label(now_utc: datetime) -> str:
    h = now_utc.hour
    if 7 <= h < 13:
        return "LONDON"
    if 13 <= h < 21:
        return "NEWYORK"
    return "OFF"


def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
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


def connect_mt5() -> MetaTrader5:
    mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 connection failed: {mt5.last_error()}")
    return mt5


def get_bars(mt5: MetaTrader5, symbol: str, tf: int, n: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data for {symbol} tf={tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    if "tick_volume" in df.columns:
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["open", "high", "low", "close", "volume"]].copy()


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


def extract_features(df: pd.DataFrame, htf_df: pd.DataFrame, retests: list[dict]) -> pd.DataFrame:
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
        atr_val = atr_s.iloc[i]
        if pd.isna(atr_val) or atr_val <= 0:
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
        if len(approach) > 0:
            approach_drop = (approach["close"].iloc[0] - df["close"].iloc[i]) / atr_val
        else:
            approach_drop = 0.0

        approach_consec_red = 0
        for j in range(i - 1, max(0, i - 8), -1):
            if df["close"].iloc[j] < df["open"].iloc[j]:
                approach_consec_red += 1
            else:
                break

        approach_vol_ratio = (df["volume"].iloc[i] / vol_ma20.iloc[i]) if vol_ma20.iloc[i] and not pd.isna(vol_ma20.iloc[i]) else 1.0

        o, h, lw, c = r["open"], r["high"], r["low"], r["close"]
        body = abs(c - o)
        lower_wick = min(o, c) - lw
        candle_range = h - lw
        close_above_level = int(c > level)
        wick_touched_level = int(lw <= level + LEVEL_ZONE_ATR * atr_val)
        wick_body_ratio = lower_wick / body if body > 0 else 0.0
        close_pos_range = (c - lw) / candle_range if candle_range > 0 else 0.0
        precision = abs(lw - level) / atr_val

        ts = df.index[i]
        htf_prior = htf_df[htf_df.index <= ts]
        if len(htf_prior) < 5:
            continue

        htf_c = htf_prior["close"].iloc[-1]
        htf_ema20_v = htf_ema20.reindex(htf_prior.index).iloc[-1]
        htf_ema50_v = htf_ema50.reindex(htf_prior.index).iloc[-1]
        htf_atr_v = htf_atr.reindex(htf_prior.index).iloc[-1]
        htf_trend = 1 if htf_c > htf_ema20_v else -1
        htf_pct_ema20 = (htf_c - htf_ema20_v) / htf_ema20_v if htf_ema20_v > 0 else 0.0
        htf_ema20_dist = abs(level - htf_ema20_v) / htf_atr_v if htf_atr_v > 0 else 99.0
        htf_ema50_dist = abs(level - htf_ema50_v) / htf_atr_v if htf_atr_v > 0 else 99.0
        htf_confluence = int(min(htf_ema20_dist, htf_ema50_dist) < 0.5)

        hour = ts.hour
        session = 0
        if 7 <= hour < 13:
            session = 1
        elif 13 <= hour < 21:
            session = 2

        round_100 = round(level / 100) * 100
        dist_round = abs(level - round_100) / atr_val
        entry_price = level
        stop_price = level - atr_val
        risk = abs(entry_price - stop_price)

        records.append({
            "timestamp": ts,
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
            "rsi": rsi_s.iloc[i],
            "pct_from_ema20": (c - ema20_s.iloc[i]) / ema20_s.iloc[i],
            "pct_from_ema50": (c - ema50_s.iloc[i]) / ema50_s.iloc[i],
            "pct_from_ema200": (c - ema200_s.iloc[i]) / ema200_s.iloc[i],
            "atr_percentile": atr_pct_s.iloc[i] if not pd.isna(atr_pct_s.iloc[i]) else 0.5,
            "htf_trend": htf_trend,
            "htf_pct_ema20": htf_pct_ema20,
            "htf_confluence": htf_confluence,
            "session": session,
            "hour": hour,
            "day_of_week": ts.dayofweek,
            "dist_round_number": dist_round,
            "risk_atr": risk / atr_val if atr_val > 0 else 0.0,
        })

    return pd.DataFrame(records)


def classify_status(score: float, good_threshold: float, action_threshold: float) -> str:
    if score >= action_threshold:
        return "ACTIONABLE"
    if score >= good_threshold:
        return "GOOD"
    return "WATCH"


def build_candidates(df: pd.DataFrame, htf_df: pd.DataFrame, model_pack: dict, cfg: ScannerConfig):
    model = model_pack["model"]
    feat_cols = model_pack["features"]
    current_price = float(df["close"].iloc[-1])
    current_atr = float(calc_atr(df, ATR_PERIOD).iloc[-1])
    last_i = len(df) - 1

    swings = find_swing_lows(df)
    if swings.empty:
        return [], current_price, current_atr

    closes = df["close"].values
    active_levels = []
    for _, sw in swings.iterrows():
        level = float(sw["price"])
        if level > current_price + current_atr:
            continue
        if abs(current_price - level) > 20 * current_atr:
            continue
        origin_i = int(sw["bar_index"])
        zone_lo = level - LEVEL_ZONE_ATR * float(sw["atr_at_formation"])
        subsequent = closes[origin_i + 1:]
        if len(subsequent) > 0 and np.any(subsequent < zone_lo):
            continue
        active_levels.append(sw)

    retests_now = []
    for sw in active_levels:
        level = float(sw["price"])
        origin_i = int(sw["bar_index"])
        dist_atr = (current_price - level) / current_atr if current_atr > 0 else 999
        if dist_atr < 0 or dist_atr > cfg.max_dist_atr:
            continue

        zone_width = LEVEL_ZONE_ATR * float(sw["atr_at_formation"])
        touch_count = 0
        in_zone = False
        for i in range(origin_i + 1, last_i):
            in_z = abs(float(df["low"].iloc[i]) - level) <= zone_width * 2
            if in_z and not in_zone:
                touch_count += 1
            in_zone = in_z

        retests_now.append({
            "level_price": level,
            "level_origin_i": origin_i,
            "level_origin_ts": sw["timestamp"],
            "retest_bar_i": last_i,
            "retest_ts": df.index[last_i],
            "touch_count": touch_count,
            "bars_since_origin": last_i - origin_i,
            "bars_outside": max(MIN_DEPARTURE_BARS, 5),
            "max_high_since_dep": current_price + max(current_atr * MIN_DEPARTURE_ATR, current_atr),
            "atr_at_retest": current_atr,
            "atr_at_formation": float(sw["atr_at_formation"]),
            "open": float(df["open"].iloc[last_i]),
            "high": float(df["high"].iloc[last_i]),
            "low": float(df["low"].iloc[last_i]),
            "close": float(df["close"].iloc[last_i]),
            "volume": float(df["volume"].iloc[last_i]),
            "dist_atr": dist_atr,
        })

    if not retests_now:
        return [], current_price, current_atr

    features = extract_features(df, htf_df, retests_now)
    if features.empty:
        return [], current_price, current_atr

    for col in feat_cols:
        if col not in features.columns:
            features[col] = 0

    X = features[feat_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    candidates = []
    for idx, ret in enumerate(retests_now[: len(probs)]):
        score = float(probs[idx])
        if score < cfg.display_threshold:
            continue

        level = float(ret["level_price"])
        atr = current_atr
        entry = level
        stop = level - cfg.sl_atr * atr
        risk = max(entry - stop, 0.01)
        target = entry + cfg.tp_r * risk
        points_to_level = current_price - level

        candidates.append({
            "timestamp": str(df.index[last_i]),
            "symbol": cfg.symbol,
            "tf": cfg.tf_label,
            "session": session_label(utc_now()),
            "current_price": round(current_price, 2),
            "atr": round(current_atr, 2),
            "level": round(level, 2),
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "risk_points": round(risk, 2),
            "reward_points": round(target - entry, 2),
            "distance_points": round(points_to_level, 2),
            "distance_atr": round(float(ret["dist_atr"]), 2),
            "touches": int(ret["touch_count"]),
            "age_bars": int(ret["bars_since_origin"]),
            "score": round(score, 4),
            "status": classify_status(score, cfg.good_threshold, cfg.action_threshold),
            "signal": "MANUAL LIMIT CANDIDATE",
        })

    if cfg.sort_by == "distance":
        candidates.sort(key=lambda x: (x["distance_atr"], -x["score"]))
    else:
        candidates.sort(key=lambda x: (-x["score"], x["distance_atr"]))

    return candidates[: cfg.top], current_price, current_atr


def build_historical_retests(df: pd.DataFrame) -> list[dict]:
    swings = find_swing_lows(df)
    if swings.empty:
        return []

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
        level = float(swing["price"])
        atr_f = float(swing["atr_at_formation"])
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
                    if bars_outside >= MIN_DEPARTURE_BARS and max_high_since_departure >= level + MIN_DEPARTURE_ATR * atr_f:
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


def build_last_actionable_retests(df: pd.DataFrame, htf_df: pd.DataFrame, model_pack: dict, cfg: ScannerConfig) -> list[dict]:
    model = model_pack["model"]
    feat_cols = model_pack["features"]

    historical_retests = build_historical_retests(df)
    if not historical_retests:
        return []

    features = extract_features(df, htf_df, historical_retests)
    if features.empty:
        return []

    usable_retests = historical_retests[: len(features)]

    for col in feat_cols:
        if col not in features.columns:
            features[col] = 0

    X = features[feat_cols].fillna(0)
    probs = model.predict_proba(X)[:, 1]

    rows = []
    for idx, ret in enumerate(usable_retests[: len(probs)]):
        score = float(probs[idx])
        if score < cfg.history_threshold:
            continue

        atr = float(ret["atr_at_retest"]) if ret["atr_at_retest"] and ret["atr_at_retest"] > 0 else 0.0
        if atr <= 0:
            continue

        entry = float(ret["level_price"])
        stop = entry - cfg.sl_atr * atr
        risk = max(entry - stop, 0.01)
        target = entry + cfg.tp_r * risk

        rows.append({
            "timestamp": str(ret["retest_ts"]),
            "level": round(entry, 2),
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "score": round(score, 4),
            "touches": int(ret["touch_count"]),
            "age_bars": int(ret["bars_since_origin"]),
            "status": classify_status(score, cfg.good_threshold, cfg.action_threshold),
        })

    rows.sort(key=lambda x: x["timestamp"], reverse=True)
    return rows[: cfg.print_last_actionable]


def alert_key(candidate: dict) -> str:
    raw = f"{candidate['symbol']}|{candidate['tf']}|{candidate['level']}|{candidate['score']}|{candidate['stop']}|{candidate['target']}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def print_header(cfg: ScannerConfig):
    print("=" * 112)
    print(
        f"LEVELS MANUAL SCANNER V2 | {cfg.symbol} {cfg.tf_label} | "
        f"display={cfg.display_threshold:.0%} | good={cfg.good_threshold:.0%} | "
        f"action={cfg.action_threshold:.0%} | SL={cfg.sl_atr:.2f} ATR | TP={cfg.tp_r:.2f}R"
    )
    print(
        f"session={cfg.session} | max_dist={cfg.max_dist_atr:.2f} ATR | top={cfg.top} | "
        f"sort={cfg.sort_by} | mode={'ONCE' if cfg.once else 'WATCH'}"
    )
    print("READ-ONLY scanner: prints levels only, places NO orders")
    print("=" * 112)


def print_candidates(candidates: list[dict], current_price: float, atr: float):
    print(f"\n[{utc_now().strftime('%Y-%m-%d %H:%M:%S UTC')}] price={current_price:.2f} atr={atr:.2f} displayed={len(candidates)}")
    if not candidates:
        print("No displayed candidates right now.")
        return

    print(f"{'Level':>10} {'Entry':>10} {'Stop':>10} {'Target':>10} {'DistATR':>8} {'Score':>8} {'Status':>12} {'Touch':>6} {'Age':>6}")
    print("-" * 112)
    for c in candidates:
        print(
            f"{c['level']:>10.2f} {c['entry']:>10.2f} {c['stop']:>10.2f} {c['target']:>10.2f} "
            f"{c['distance_atr']:>8.2f} {c['score']:>8.1%} {c['status']:>12} {c['touches']:>6} {c['age_bars']:>6}"
        )


def print_last_actionable(rows: list[dict], threshold: float):
    print(f"\nLast {len(rows)} historical retests with score >= {threshold:.0%}")
    if not rows:
        print("None found in loaded history.")
        return

    print(f"{'Timestamp':<24} {'Level':>10} {'Entry':>10} {'Stop':>10} {'Target':>10} {'Score':>8} {'Status':>12} {'Touch':>6} {'Age':>6}")
    print("-" * 112)
    for r in rows:
        print(
            f"{r['timestamp']:<24} {r['level']:>10.2f} {r['entry']:>10.2f} {r['stop']:>10.2f} {r['target']:>10.2f} "
            f"{r['score']:>8.1%} {r['status']:>12} {r['touches']:>6} {r['age_bars']:>6}"
        )


def append_jsonl(path: str, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def run_once(cfg: ScannerConfig, alerted: dict):
    now = utc_now()
    if not in_session(cfg.session, now):
        print(f"[{now.strftime('%Y-%m-%d %H:%M:%S UTC')}] Outside selected session ({cfg.session}).")
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Use your existing trained model or retrain first.")

    model_pack = joblib.load(MODEL_PATH)
    mt5 = connect_mt5()
    try:
        tf = TF_LABEL_MAP[cfg.tf_label]
        htf = HTF_MAP.get(tf, TIMEFRAME_H4)
        df = get_bars(mt5, cfg.symbol, tf, cfg.bars)
        htf_df = get_bars(mt5, cfg.symbol, htf, max(cfg.bars // 4, 500))
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass

    candidates, current_price, atr = build_candidates(df, htf_df, model_pack, cfg)
    recent_rows = build_last_actionable_retests(df, htf_df, model_pack, cfg)

    fresh = []
    cooldown_seconds = cfg.cooldown_minutes * 60
    now_ts = time.time()
    for c in candidates:
        key = alert_key(c)
        last_ts = alerted.get(key, 0)
        if now_ts - last_ts >= cooldown_seconds:
            alerted[key] = now_ts
            c["alert_key"] = key
            fresh.append(c)
            if cfg.jsonl_log:
                append_jsonl(cfg.jsonl_log, c)

    print_candidates(fresh, current_price, atr)
    print_last_actionable(recent_rows, cfg.history_threshold)


def main():
    parser = argparse.ArgumentParser(description="Read-only manual key levels scanner for Linux + mt5linux")
    parser.add_argument("--symbol", default="@MNQ")
    parser.add_argument("--tf", default="H1", choices=list(TF_LABEL_MAP.keys()))
    parser.add_argument("--bars", type=int, default=3000)

    parser.add_argument("--display-threshold", type=float, default=0.20)
    parser.add_argument("--good-threshold", type=float, default=0.50)
    parser.add_argument("--action-threshold", type=float, default=0.70)
    parser.add_argument("--history-threshold", type=float, default=0.70)

    parser.add_argument("--max-dist-atr", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--sl-atr", type=float, default=1.0)
    parser.add_argument("--tp-r", type=float, default=2.0)

    parser.add_argument("--session", choices=["london", "ny", "both", "all"], default="both")
    parser.add_argument("--loop-seconds", type=int, default=60)
    parser.add_argument("--cooldown-minutes", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--jsonl-log", default=None)

    parser.add_argument("--sort-by", choices=["score", "distance"], default="score")
    parser.add_argument("--print-last-actionable", type=int, default=5)

    args = parser.parse_args()

    cfg = ScannerConfig(
        symbol=args.symbol,
        tf_label=args.tf,
        bars=args.bars,
        display_threshold=args.display_threshold,
        good_threshold=args.good_threshold,
        action_threshold=args.action_threshold,
        history_threshold=args.history_threshold,
        max_dist_atr=args.max_dist_atr,
        top=args.top,
        sl_atr=args.sl_atr,
        tp_r=args.tp_r,
        session=args.session,
        loop_seconds=args.loop_seconds,
        cooldown_minutes=args.cooldown_minutes,
        once=args.once,
        jsonl_log=args.jsonl_log,
        sort_by=args.sort_by,
        print_last_actionable=args.print_last_actionable,
    )

    print_header(cfg)
    alerted = {}

    if cfg.once:
        run_once(cfg, alerted)
        return

    while True:
        try:
            run_once(cfg, alerted)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
        time.sleep(cfg.loop_seconds)


if __name__ == "__main__":
    main()
