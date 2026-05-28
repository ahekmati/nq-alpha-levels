#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from mt5linux import MetaTrader5

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except Exception:
    HMM_AVAILABLE = False


OUTPUT_DIR = Path("mnq_atr_dip_meta_scan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    atr_period: int = 7
    atr_threshold: float = 120.0
    reversal_lookback: int = 3
    daily_fast_ma: int = 20
    daily_slow_ma: int = 50
    daily_slope_lookback: int = 3
    stop_points: int = 200
    take_profit_points: int = 400
    max_hold_bars: int = 24
    n_splits: int = 5
    embargo_bars: int = 24
    hmm_states: int = 3


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="@MNQ")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18812)
    p.add_argument("--atr-threshold", type=float, default=120.0)
    p.add_argument("--stop-points", type=int, default=200)
    p.add_argument("--take-profit-points", type=int, default=400)
    p.add_argument("--max-hold-bars", type=int, default=24)
    return p.parse_args()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def fetch_rates(mt5, symbol: str, timeframe: int, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def choose_symbol(mt5, preferred: str, start_dt: datetime, end_dt: datetime) -> str:
    candidates = [preferred, "@MNQ", "MNQ", "MNQM26", "MNQU26"]
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    for sym in candidates:
        try:
            info = mt5.symbol_info(sym)
            if info is None:
                continue
            mt5.symbol_select(sym, True)
            test = mt5.copy_rates_range(sym, mt5.TIMEFRAME_H1, start_dt, end_dt)
            if test is not None and len(test) > 100:
                return sym
        except Exception:
            pass
    raise RuntimeError("Could not resolve MNQ symbol")


def build_daily_context(d1: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = d1.copy()
    df["ret1"] = df["close"].pct_change(1)
    df["ret5"] = df["close"].pct_change(5)
    df["vol10"] = df["ret1"].rolling(10).std()
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["rsi10"] = rsi(df["close"], 10)

    df["ma_fast"] = df["close"].rolling(cfg.daily_fast_ma).mean()
    df["ma_slow"] = df["close"].rolling(cfg.daily_slow_ma).mean()
    df["ma_fast_slope"] = df["ma_fast"] - df["ma_fast"].shift(cfg.daily_slope_lookback)
    df["ma_slow_slope"] = df["ma_slow"] - df["ma_slow"].shift(cfg.daily_slope_lookback)

    df["bull_ma_structure"] = (
        (df["close"] > df["ma_fast"]) &
        (df["ma_fast"] > df["ma_slow"]) &
        (df["ma_fast_slope"] > 0) &
        (df["ma_slow_slope"] > 0)
    ).astype(int)

    if HMM_AVAILABLE:
        feat = df[["ret1", "vol10", "range_pct"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        hmm = GaussianHMM(n_components=cfg.hmm_states, covariance_type="full", n_iter=300, random_state=42)
        hmm.fit(feat.values)
        states = hmm.predict(feat.values)
        feat["state"] = states
        bull_state = feat.groupby("state")["ret1"].mean().idxmax()
        df["hmm_state"] = np.nan
        df.loc[feat.index, "hmm_state"] = feat["state"]
        df["hmm_bull"] = (df["hmm_state"] == bull_state).astype(int)
    else:
        df["hmm_state"] = np.nan
        df["hmm_bull"] = df["bull_ma_structure"]

    df["bull_structure_strict"] = ((df["bull_ma_structure"] == 1) & (df["hmm_bull"] == 1)).astype(int)

    keep = [
        "time", "bull_ma_structure", "hmm_bull", "bull_structure_strict",
        "ret1", "ret5", "vol10", "range_pct", "rsi10",
        "ma_fast", "ma_slow", "ma_fast_slope", "ma_slow_slope"
    ]
    out = df[keep].copy()
    out["date"] = out["time"].dt.floor("D")
    out = out.drop(columns=["time"])

    for c in out.columns:
        if c != "date":
            out[c] = out[c].shift(1)

    return out


def build_hourly_dataset(h1: pd.DataFrame, daily_ctx: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = h1.copy()
    df["date"] = df["time"].dt.floor("D")

    df["atr7"] = atr(df, cfg.atr_period)
    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret6"] = df["close"].pct_change(6)
    df["ret12"] = df["close"].pct_change(12)
    df["rsi7"] = rsi(df["close"], 7)
    df["rsi14"] = rsi(df["close"], 14)

    df["h_ma12"] = df["close"].rolling(12).mean()
    df["h_ma24"] = df["close"].rolling(24).mean()
    df["h_ma48"] = df["close"].rolling(48).mean()
    df["dist_ma12"] = df["close"] - df["h_ma12"]
    df["dist_ma24"] = df["close"] - df["h_ma24"]
    df["dist_ma48"] = df["close"] - df["h_ma48"]

    df["bar_range"] = df["high"] - df["low"]
    df["bar_body"] = df["close"] - df["open"]
    df["bear_bar"] = (df["close"] < df["open"]).astype(int)
    df["bull_bar"] = (df["close"] > df["open"]).astype(int)
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek

    df = df.merge(daily_ctx, on="date", how="left")
    return df


def build_base_events(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    atr_prev = df["atr7"].shift(1)
    first_cross = (atr_prev < cfg.atr_threshold) & (df["atr7"] >= cfg.atr_threshold) & (df["bear_bar"] == 1)

    crossed_recently = first_cross.rolling(cfg.reversal_lookback).max().fillna(0) > 0
    reversal = (df["bull_bar"] == 1) & (df["close"] > df["high"].shift(1))
    base_signal = crossed_recently & reversal & (df["bull_structure_strict"] == 1)

    events = df.loc[base_signal].copy()
    events["event_idx"] = events.index
    return events


def triple_barrier_label(df: pd.DataFrame, event_idx: int, cfg: Config):
    entry_idx = event_idx + 1
    if entry_idx >= len(df):
        return np.nan, np.nan, np.nan, np.nan

    entry_price = df.iloc[entry_idx]["open"]
    stop_price = entry_price - cfg.stop_points
    take_price = entry_price + cfg.take_profit_points

    end_idx = min(entry_idx + cfg.max_hold_bars, len(df) - 1)

    for j in range(entry_idx, end_idx + 1):
        bar = df.iloc[j]
        low_ = bar["low"]
        high_ = bar["high"]

        if low_ <= stop_price and high_ >= take_price:
            return 0, entry_price, j, "stop_same_bar"

        if low_ <= stop_price:
            return 0, entry_price, j, "stop"

        if high_ >= take_price:
            return 1, entry_price, j, "take"

    final_close = df.iloc[end_idx]["close"]
    return int(final_close > entry_price), entry_price, end_idx, "time"


def make_event_dataset(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    events = build_base_events(df, cfg).copy()

    labels = []
    for idx in events["event_idx"]:
        y, entry, exit_idx, reason = triple_barrier_label(df, int(idx), cfg)
        labels.append((y, entry, exit_idx, reason))

    events["y"] = [x[0] for x in labels]
    events["entry_price"] = [x[1] for x in labels]
    events["exit_idx"] = [x[2] for x in labels]
    events["label_reason"] = [x[3] for x in labels]

    events["atr_excess"] = events["atr7"] - cfg.atr_threshold
    events["range_to_atr"] = events["bar_range"] / events["atr7"]
    events["body_to_range"] = events["bar_body"] / events["bar_range"].replace(0, np.nan)
    events["dist_ma12_atr"] = events["dist_ma12"] / events["atr7"]
    events["dist_ma24_atr"] = events["dist_ma24"] / events["atr7"]
    events["dist_ma48_atr"] = events["dist_ma48"] / events["atr7"]

    return events.replace([np.inf, -np.inf], np.nan).dropna(subset=["y"])


def purged_splits(n: int, n_splits: int, embargo: int):
    indices = np.arange(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1

    current = 0
    for fold_size in fold_sizes:
        test_start = current
        test_end = current + fold_size
        test_idx = indices[test_start:test_end]

        left_train = indices[:max(0, test_start - embargo)]
        right_train = indices[min(n, test_end + embargo):]
        train_idx = np.concatenate([left_train, right_train])

        if len(test_idx) > 0 and len(train_idx) > 20:
            yield train_idx, test_idx

        current = test_end


def model_library():
    return {
        "logreg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "rf": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "hgb": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.03,
                max_iter=300,
                min_samples_leaf=10,
                random_state=42
            ))
        ]),
    }


def evaluate_events(events: pd.DataFrame, cfg: Config):
    feature_cols = [
        c for c in events.columns
        if c not in {
            "time", "date", "event_idx", "entry_price", "exit_idx", "label_reason", "y",
            "open", "high", "low", "close", "tick_volume", "spread", "real_volume"
        }
    ]

    X = events[feature_cols]
    y = events["y"].astype(int).values

    rows = []
    pred_rows = []

    for name, model in model_library().items():
        oof = np.full(len(events), np.nan)

        for fold, (tr, te) in enumerate(purged_splits(len(events), cfg.n_splits, cfg.embargo_bars), start=1):
            mdl = clone(model)
            mdl.fit(X.iloc[tr], y[tr])

            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(X.iloc[te])[:, 1]
            else:
                s = mdl.decision_function(X.iloc[te])
                p = 1.0 / (1.0 + np.exp(-s))

            oof[te] = p

            yte = y[te]
            auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
            ap = average_precision_score(yte, p) if yte.sum() > 0 else np.nan

            rows.append({
                "model": name,
                "fold": fold,
                "roc_auc": auc,
                "avg_precision": ap,
                "n_test": len(te),
                "positive_rate": float(np.mean(yte))
            })

        valid = ~np.isnan(oof)
        if valid.sum() and len(np.unique(y[valid])) > 1:
            auc_all = roc_auc_score(y[valid], oof[valid])
            ap_all = average_precision_score(y[valid], oof[valid])
        else:
            auc_all = np.nan
            ap_all = np.nan

        for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
            take = valid & (oof >= thr)
            if take.sum() == 0:
                precision = np.nan
                count = 0
            else:
                precision = float(np.mean(y[take]))
                count = int(take.sum())

            rows.append({
                "model": name,
                "fold": "all",
                "roc_auc": auc_all,
                "avg_precision": ap_all,
                "n_test": int(valid.sum()),
                "positive_rate": float(np.mean(y[valid])) if valid.sum() else np.nan,
                "threshold": thr,
                "signals": count,
                "precision_at_thr": precision
            })

        pred_tmp = events[["time", "event_idx", "y"]].copy()
        pred_tmp["model"] = name
        pred_tmp["pred_prob"] = oof
        pred_rows.append(pred_tmp)

    return pd.DataFrame(rows), pd.concat(pred_rows, ignore_index=True)


def main():
    args = parse_args()
    cfg = Config(
        atr_threshold=args.atr_threshold,
        stop_points=args.stop_points,
        take_profit_points=args.take_profit_points,
        max_hold_bars=args.max_hold_bars,
    )

    end_dt = datetime.now(timezone.utc) if args.end is None else pd.Timestamp(args.end, tz="UTC").to_pydatetime()
    start_dt = pd.Timestamp(args.start, tz="UTC").to_pydatetime()

    mt5 = MetaTrader5(host=args.host, port=args.port)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    symbol = choose_symbol(mt5, args.symbol, start_dt, end_dt)
    mt5.symbol_select(symbol, True)

    d1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_D1, start_dt, end_dt)
    h1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)
    mt5.shutdown()

    if d1.empty or h1.empty:
        raise RuntimeError("No D1/H1 data returned.")

    daily_ctx = build_daily_context(d1, cfg)
    df = build_hourly_dataset(h1, daily_ctx, cfg).replace([np.inf, -np.inf], np.nan)
    events = make_event_dataset(df, cfg)

    eval_df, pred_df = evaluate_events(events, cfg)

    meta = {
        "symbol_used": symbol,
        "start": str(start_dt),
        "end": str(end_dt),
        "n_d1_bars": int(len(d1)),
        "n_h1_bars": int(len(h1)),
        "n_events": int(len(events)),
        "event_positive_rate": float(events["y"].mean()) if len(events) else np.nan,
        "atr_threshold": cfg.atr_threshold,
        "hmm_available": HMM_AVAILABLE,
    }

    df.to_csv(OUTPUT_DIR / "full_dataset.csv", index=False)
    events.to_csv(OUTPUT_DIR / "atr_dip_events.csv", index=False)
    eval_df.to_csv(OUTPUT_DIR / "meta_model_eval.csv", index=False)
    pred_df.to_csv(OUTPUT_DIR / "meta_model_predictions.csv", index=False)

    with open(OUTPUT_DIR / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))

    print("\n" + "=" * 90)
    print("META-LABEL EVENT MODELS")
    print("=" * 90)
    out = eval_df[(eval_df["fold"] == "all") & (eval_df["threshold"] == 0.60)].copy()
    if not out.empty:
        print(out.sort_values(["precision_at_thr", "roc_auc"], ascending=False).to_string(index=False))
    else:
        print("No aggregate rows found.")


if __name__ == "__main__":
    main()
