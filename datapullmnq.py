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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("research_mnq_model_scan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class HorizonConfig:
    bars_ahead: int = 12
    long_threshold: float = 0.006
    short_threshold: float = 0.006


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="MNQ", help="Preferred research symbol, e.g. MNQ or @MNQ")
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=18812)
    p.add_argument("--bars-ahead", type=int, default=12, help="Prediction horizon in H1 bars")
    p.add_argument("--long-threshold", type=float, default=0.006, help="e.g. 0.006 = 0.6%")
    p.add_argument("--short-threshold", type=float, default=0.006, help="e.g. 0.006 = 0.6%")
    return p.parse_args()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    ret = np.log(close / close.shift(1))
    return ret.rolling(window).std() * np.sqrt(window)


def zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd.replace(0, np.nan)


def fetch_rates(mt5, symbol: str, timeframe: int, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df.sort_values("time").reset_index(drop=True)


def choose_symbol(mt5, preferred: str, start_dt: datetime, end_dt: datetime) -> str:
    candidates = [preferred, "@MNQ", "MNQ", "MNQM26", "MNQU26", "MNQM26.cash"]
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
    raise RuntimeError("Could not resolve a usable MNQ research symbol in MT5.")


def build_daily_features(d1: pd.DataFrame) -> pd.DataFrame:
    df = d1.copy()
    df["d_ret1"] = df["close"].pct_change(1)
    df["d_ret5"] = df["close"].pct_change(5)
    df["d_ret20"] = df["close"].pct_change(20)
    df["d_rsi10"] = rsi(df["close"], 10)
    df["d_rsi14"] = rsi(df["close"], 14)
    df["d_atr14"] = atr(df, 14)
    df["d_atr_pct"] = df["d_atr14"] / df["close"]
    df["d_ma10"] = df["close"].rolling(10).mean()
    df["d_ma20"] = df["close"].rolling(20).mean()
    df["d_ma50"] = df["close"].rolling(50).mean()
    df["d_ma100"] = df["close"].rolling(100).mean()
    df["d_ma_gap_10_20"] = (df["d_ma10"] - df["d_ma20"]) / df["close"]
    df["d_ma_gap_20_50"] = (df["d_ma20"] - df["d_ma50"]) / df["close"]
    df["d_price_vs_ma20"] = (df["close"] - df["d_ma20"]) / df["close"]
    df["d_price_vs_ma50"] = (df["close"] - df["d_ma50"]) / df["close"]
    df["d_price_vs_ma100"] = (df["close"] - df["d_ma100"]) / df["close"]
    df["d_vol10"] = realized_vol(df["close"], 10)
    df["d_vol20"] = realized_vol(df["close"], 20)
    df["d_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["d_close_loc"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["d_z20"] = zscore(df["close"], 20)
    df["d_z50"] = zscore(df["close"], 50)
    return df


def build_hourly_features(h1: pd.DataFrame) -> pd.DataFrame:
    df = h1.copy()
    df["h_ret1"] = df["close"].pct_change(1)
    df["h_ret3"] = df["close"].pct_change(3)
    df["h_ret6"] = df["close"].pct_change(6)
    df["h_ret12"] = df["close"].pct_change(12)
    df["h_rsi7"] = rsi(df["close"], 7)
    df["h_rsi14"] = rsi(df["close"], 14)
    df["h_atr14"] = atr(df, 14)
    df["h_atr_pct"] = df["h_atr14"] / df["close"]
    df["h_ma6"] = df["close"].rolling(6).mean()
    df["h_ma24"] = df["close"].rolling(24).mean()
    df["h_ma72"] = df["close"].rolling(72).mean()
    df["h_gap_6_24"] = (df["h_ma6"] - df["h_ma24"]) / df["close"]
    df["h_gap_24_72"] = (df["h_ma24"] - df["h_ma72"]) / df["close"]
    df["h_price_vs_ma24"] = (df["close"] - df["h_ma24"]) / df["close"]
    df["h_price_vs_ma72"] = (df["close"] - df["h_ma72"]) / df["close"]
    df["h_vol24"] = realized_vol(df["close"], 24)
    df["h_vol72"] = realized_vol(df["close"], 72)
    df["h_range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["h_close_loc"] = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    df["h_z24"] = zscore(df["close"], 24)
    df["h_z72"] = zscore(df["close"], 72)
    df["hour"] = df["time"].dt.hour
    df["dow"] = df["time"].dt.dayofweek
    return df


def merge_features(h1: pd.DataFrame, d1_feat: pd.DataFrame, cfg: HorizonConfig) -> pd.DataFrame:
    hf = build_hourly_features(h1)
    df = hf.copy()

    d = d1_feat.copy()
    d["date"] = d["time"].dt.floor("D")
    daily_cols = [c for c in d.columns if c not in ["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]
    d = d[daily_cols].rename(columns={"time": "d_time"})
    df["date"] = df["time"].dt.floor("D")
    df = df.merge(d, on="date", how="left")

    fwd_close = df["close"].shift(-cfg.bars_ahead)
    fwd_ret = (fwd_close / df["close"]) - 1.0
    adverse_ret = ((df["close"].rolling(cfg.bars_ahead).min().shift(-cfg.bars_ahead + 1)) / df["close"]) - 1.0
    favorable_ret = ((df["close"].rolling(cfg.bars_ahead).max().shift(-cfg.bars_ahead + 1)) / df["close"]) - 1.0

    df["fwd_ret"] = fwd_ret
    df["favorable_ret"] = favorable_ret
    df["adverse_ret"] = adverse_ret
    df["target_long"] = (df["fwd_ret"] >= cfg.long_threshold).astype(int)
    df["target_short"] = (df["fwd_ret"] <= -cfg.short_threshold).astype(int)

    drop_cols = ["date", "d_time"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume",
        "fwd_ret", "favorable_ret", "adverse_ret", "target_long", "target_short"
    }
    return [c for c in df.columns if c not in excluded]


def model_library():
    return {
        "logreg_l2": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "rf_500": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=8,
                min_samples_leaf=25,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "hgb": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.03,
                max_iter=300,
                min_samples_leaf=50,
                random_state=42
            ))
        ]),
        "linear_svc_cal": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", max_iter=5000),
                method="sigmoid",
                cv=3
            ))
        ]),
    }


def evaluate_models(df: pd.DataFrame, target_col: str, model_dict: dict[str, Pipeline]) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = get_feature_columns(df)
    work = df.dropna(subset=[target_col]).copy()
    X = work[feature_cols]
    y = work[target_col].astype(int).values
    t = work["time"].values

    splitter = TimeSeriesSplit(n_splits=5)
    rows = []
    pred_frames = []

    for name, model in model_dict.items():
        oof_pred = np.full(len(work), np.nan)

        for fold, (tr_idx, te_idx) in enumerate(splitter.split(X)):
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]

            mdl = clone(model)
            mdl.fit(Xtr, ytr)

            if hasattr(mdl, "predict_proba"):
                p = mdl.predict_proba(Xte)[:, 1]
            else:
                s = mdl.decision_function(Xte)
                p = 1.0 / (1.0 + np.exp(-s))

            oof_pred[te_idx] = p

            fold_auc = roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan
            fold_ap = average_precision_score(yte, p) if yte.sum() > 0 else np.nan
            rows.append({
                "model": name,
                "fold": fold + 1,
                "target": target_col,
                "n_train": len(tr_idx),
                "n_test": len(te_idx),
                "positive_rate_test": float(np.mean(yte)),
                "roc_auc": fold_auc,
                "avg_precision": fold_ap
            })

        valid = ~np.isnan(oof_pred)
        if valid.sum() > 0 and len(np.unique(y[valid])) > 1:
            overall_auc = roc_auc_score(y[valid], oof_pred[valid])
            overall_ap = average_precision_score(y[valid], oof_pred[valid])
        else:
            overall_auc = np.nan
            overall_ap = np.nan

        thresholds = np.arange(0.50, 0.91, 0.05)
        for thr in thresholds:
            signal = (oof_pred >= thr).astype(int)
            take = valid & (signal == 1)
            if take.sum() == 0:
                precision = np.nan
                avg_fwd = np.nan
                median_fwd = np.nan
                hitrate = np.nan
            else:
                precision = float(np.mean(y[take]))
                avg_fwd = float(work.loc[take, "fwd_ret"].mean())
                median_fwd = float(work.loc[take, "fwd_ret"].median())
                if target_col == "target_long":
                    hitrate = float(np.mean(work.loc[take, "fwd_ret"] > 0))
                else:
                    hitrate = float(np.mean(work.loc[take, "fwd_ret"] < 0))

            rows.append({
                "model": name,
                "fold": "all",
                "target": target_col,
                "n_train": np.nan,
                "n_test": int(valid.sum()),
                "positive_rate_test": float(np.mean(y[valid])) if valid.sum() else np.nan,
                "roc_auc": overall_auc,
                "avg_precision": overall_ap,
                "threshold": thr,
                "signals": int(take.sum()),
                "precision_at_thr": precision,
                "mean_fwd_ret_at_thr": avg_fwd,
                "median_fwd_ret_at_thr": median_fwd,
                "directional_hitrate_at_thr": hitrate
            })

        pred_frames.append(pd.DataFrame({
            "time": t,
            "model": name,
            "target": target_col,
            "y_true": y,
            "pred_prob": oof_pred
        }))

    return pd.DataFrame(rows), pd.concat(pred_frames, ignore_index=True)


def fit_feature_importance(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    feature_cols = get_feature_columns(df)
    work = df.dropna(subset=[target_col]).copy()
    X = work[feature_cols]
    y = work[target_col].astype(int).values

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipe.fit(X, y)
    imp = pipe.named_steps["clf"].feature_importances_
    out = pd.DataFrame({"feature": feature_cols, "importance": imp, "target": target_col})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def main():
    args = parse_args()
    end_dt = datetime.now(timezone.utc) if args.end is None else pd.Timestamp(args.end, tz="UTC").to_pydatetime()
    start_dt = pd.Timestamp(args.start, tz="UTC").to_pydatetime()
    cfg = HorizonConfig(
        bars_ahead=args.bars_ahead,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
    )

    mt5 = MetaTrader5(host=args.host, port=args.port)
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    symbol = choose_symbol(mt5, args.symbol, start_dt, end_dt)
    mt5.symbol_select(symbol, True)

    d1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_D1, start_dt, end_dt)
    h1 = fetch_rates(mt5, symbol, mt5.TIMEFRAME_H1, start_dt, end_dt)

    if d1.empty or h1.empty:
        raise RuntimeError("No D1/H1 data returned from MT5.")

    d1_feat = build_daily_features(d1)
    dataset = merge_features(h1, d1_feat, cfg)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).sort_values("time").reset_index(drop=True)

    models = model_library()

    long_eval, long_preds = evaluate_models(dataset, "target_long", models)
    short_eval, short_preds = evaluate_models(dataset, "target_short", models)

    long_imp = fit_feature_importance(dataset, "target_long")
    short_imp = fit_feature_importance(dataset, "target_short")

    summary = {
        "symbol_used": symbol,
        "start": str(start_dt),
        "end": str(end_dt),
        "bars_ahead": cfg.bars_ahead,
        "long_threshold": cfg.long_threshold,
        "short_threshold": cfg.short_threshold,
        "n_d1_bars": int(len(d1)),
        "n_h1_bars": int(len(h1)),
        "n_dataset_rows": int(len(dataset)),
        "long_positive_rate": float(dataset["target_long"].mean()),
        "short_positive_rate": float(dataset["target_short"].mean()),
    }

    dataset.to_csv(OUTPUT_DIR / "mnq_h1_d1_feature_dataset.csv", index=False)
    long_eval.to_csv(OUTPUT_DIR / "model_eval_long.csv", index=False)
    short_eval.to_csv(OUTPUT_DIR / "model_eval_short.csv", index=False)
    long_preds.to_csv(OUTPUT_DIR / "model_oof_predictions_long.csv", index=False)
    short_preds.to_csv(OUTPUT_DIR / "model_oof_predictions_short.csv", index=False)
    long_imp.to_csv(OUTPUT_DIR / "feature_importance_long.csv", index=False)
    short_imp.to_csv(OUTPUT_DIR / "feature_importance_short.csv", index=False)

    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    mt5.shutdown()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
