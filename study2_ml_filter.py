"""
study2_ml_filter.py
====================
Trains an XGBoost classifier on Study 2 trade records to predict
whether a rally→selloff→overnight dip trade will win or lose.

If ML adds value (OOS AUC > 0.58, filtered WR > raw WR by >=5pp),
the model is saved and the evaluator can use it as a gate.

Modes:
  train    — train + walk-forward validate, save model if passes
  evaluate — re-run full validation on existing or new data
  scan     — score tonight's setup against the saved model

Usage:
  python study2_ml_filter.py --mode train --data ./mnq_study/study2_all_trades.csv
  python study2_ml_filter.py --mode evaluate --data ./mnq_study/study2_all_trades.csv
  python study2_ml_filter.py --mode scan --setup ./tonight_setup.json

Requirements:
  pip install xgboost scikit-learn pandas numpy joblib
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ET            = ZoneInfo("America/New_York")
MODEL_PATH    = Path(__file__).parent / "study2_xgb.joblib"
BASELINE_PATH = Path(__file__).parent / "study2_baseline.json"

# ML gate threshold — only take trade if ML score >= this
ML_THRESHOLD  = 0.58

# Minimum OOS AUC to accept and save a trained model
MIN_AUC       = 0.58

# Minimum win rate lift over base rate to accept model
MIN_LIFT      = 0.05    # 5 percentage points

# Features used for prediction — ORDER MUST NOT CHANGE after training
FEATURE_COLS  = [
    "rally_mult",          # overnight rally size in ATR multiples
    "selloff_pct",         # RTH retracement from overnight high (%)
    "retrace_ratio",       # selloff_pct / (rally_mult * 100)
    "atr_val",             # absolute ATR at time of setup
    "rth_close_norm",      # RTH close level (z-score normalised)
    "day_of_week",         # 0=Mon … 4=Fri
    "month",               # 1–12 seasonality
    "session_asian",       # 1 if entry was in Asian session
    "session_european",    # 1 if entry was in European session
    "session_premarket",   # 1 if entry was in Pre-market session
]


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML feature matrix from study2_all_trades.csv.
    Only rows where triggered=True and result in [win, loss] are kept.
    Returns sorted DataFrame with FEATURE_COLS + target + date_ts.
    """
    # Keep only triggered, resolved trades
    df = df[
        (df["triggered"] == True) &
        (df["result"].isin(["win", "loss"]))
    ].copy()

    if len(df) == 0:
        raise ValueError(
            "No resolved triggered trades found. "
            "Check dip_atr_mult and rr_ratio filter values."
        )

    # ── Target ────────────────────────────────────────────────
    df["target"] = (df["result"] == "win").astype(int)

    # ── Date features ─────────────────────────────────────────
    df["date_ts"]     = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date_ts"].dt.dayofweek
    df["month"]       = df["date_ts"].dt.month

    # ── Session dummies ───────────────────────────────────────
    df["session_asian"]     = (df["sub_session"] == "Asian").astype(int)
    df["session_european"]  = (df["sub_session"] == "European").astype(int)
    df["session_premarket"] = (df["sub_session"] == "Pre-market").astype(int)

    # ── Retracement ratio ─────────────────────────────────────
    # How much of the overnight rally did RTH give back?
    # selloff_pct is stored as percentage (e.g. 0.7 means 0.7%)
    # rally_mult is in ATR multiples
    # We normalise both to the same scale: selloff_pct / (rally_mult * 100)
    safe_rally = df["rally_mult"].replace(0, np.nan)
    df["retrace_ratio"] = (df["selloff_pct"] / (safe_rally * 100)).fillna(0).clip(0, 5)

    # ── Normalise rth_close (z-score rolling window) ──────────
    # Sort chronologically first so rolling window is correct
    df = df.sort_values("date_ts").reset_index(drop=True)
    roll_mean = df["rth_close"].rolling(100, min_periods=10).mean()
    roll_std  = df["rth_close"].rolling(100, min_periods=10).std().replace(0, np.nan)
    df["rth_close_norm"] = ((df["rth_close"] - roll_mean) / roll_std).fillna(0)

    # ── Select and validate ───────────────────────────────────
    needed = FEATURE_COLS + ["target", "date_ts"]
    # Verify all needed columns exist
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")

    df = df[needed].dropna()

    if len(df) == 0:
        raise ValueError("All rows dropped after NaN removal — check input data.")

    return df


# ─────────────────────────────────────────────
# TRAINING + WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

def train(data_path: str):
    print("\n" + "=" * 60)
    print("STUDY 2 ML FILTER  |  Training")
    print("=" * 60)

    df_raw = pd.read_csv(data_path)
    print(f"Raw records loaded: {len(df_raw):,}")

    # Use best RR/dip combo from backtesting (2.0x ATR, RR=2.5)
    # Filter to that specific param combo for training
    df_sub = df_raw[
        (df_raw["dip_atr_mult"] == 2.0) &
        (df_raw["rr_ratio"] == 2.5)
    ].copy()
    print(f"Records at dip=2.0x ATR / RR=2.5: {len(df_sub):,}")

    df = build_features(df_sub)
    n_total = len(df)
    n_wins  = int(df["target"].sum())
    n_loss  = n_total - n_wins

    print(f"Training samples: {n_total}  ({n_wins}W / {n_loss}L)")
    print(f"Base win rate   : {n_wins/n_total:.1%}")

    if n_total < 50:
        print("⚠  Fewer than 50 samples — ML filter may not be reliable.")
        print("   Collect more data before deploying.")

    X = df[FEATURE_COLS].values
    y = df["target"].values

    # ── Walk-forward cross-validation ────────────────────────
    tscv     = TimeSeriesSplit(n_splits=5)
    fold_aucs = []
    oos_preds = np.full(len(y), np.nan)

    print("\nWalk-forward validation (5 folds):")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if len(np.unique(y_val)) < 2:
            print(f"  Fold {fold+1}: skipped (single class in val)")
            continue

        model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.75,
            min_child_weight=5,
            scale_pos_weight=n_loss / max(n_wins, 1),
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_val)[:, 1]
        oos_preds[val_idx] = probs
        auc = roc_auc_score(y_val, probs)
        fold_aucs.append(auc)
        print(f"  Fold {fold+1}: AUC={auc:.4f}  n={len(y_val)}")

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.0
    std_auc  = float(np.std(fold_aucs))  if fold_aucs else 0.0
    print(f"\nOOS AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    # ── OOS performance at threshold ─────────────────────────
    valid_mask    = ~np.isnan(oos_preds)
    oos_probs     = oos_preds[valid_mask]
    oos_y         = y[valid_mask]
    base_wr       = float(oos_y.mean()) if len(oos_y) > 0 else 0.0

    filtered_mask = oos_probs >= ML_THRESHOLD
    n_filtered    = int(filtered_mask.sum())

    # Initialise with safe defaults — overwritten if n_filtered > 0
    filt_wr  = 0.0
    lift     = 0.0
    filt_exp = 0.0
    base_exp = (base_wr * 2.5) - (1 - base_wr)

    if n_filtered > 0:
        filt_wr  = float(oos_y[filtered_mask].mean())
        lift     = filt_wr - base_wr
        rr       = 2.5
        filt_exp = (filt_wr * rr) - (1 - filt_wr)
        base_exp = (base_wr * rr) - (1 - base_wr)
        print(f"\nAt threshold {ML_THRESHOLD}:")
        print(f"  Filtered trades  : {n_filtered} / {len(oos_y)} "
              f"({n_filtered/len(oos_y):.1%} pass rate)")
        print(f"  Filtered WR      : {filt_wr:.1%}  "
              f"(base={base_wr:.1%}  lift={lift:+.1%})")
        print(f"  Filtered Exp     : {filt_exp:+.3f}R  (base={base_exp:+.3f}R)")
    else:
        print(f"\n⚠  No OOS trades passed threshold {ML_THRESHOLD}")
        print(f"  Consider lowering ML_THRESHOLD or collecting more data.")

    ml_adds_value = (
        mean_auc   >= MIN_AUC  and
        n_filtered >  0        and
        lift       >= MIN_LIFT
    )

    # ── Feature importance ────────────────────────────────────
    final_model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.75, min_child_weight=5,
        scale_pos_weight=n_loss / max(n_wins, 1),
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    final_model.fit(X, y)

    imp = pd.Series(final_model.feature_importances_,
                    index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importance:")
    for feat, val in imp.items():
        bar = "█" * int(val * 40)
        print(f"  {feat:<22} {val:.4f}  {bar}")

    # ── Save decision ─────────────────────────────────────────
    print(f"\nML adds value: {'✅ YES' if ml_adds_value else '⚠  NO'}")

    if ml_adds_value:
        joblib.dump({
            "model"    : final_model,
            "features" : FEATURE_COLS,
            "threshold": ML_THRESHOLD,
            "auc"      : mean_auc,
            "base_wr"  : base_wr,
            "filt_wr"  : filt_wr,
            "lift"     : lift,
        }, MODEL_PATH)
        with open(BASELINE_PATH, "w") as f:
            json.dump({
                "auc"     : mean_auc,
                "base_wr" : base_wr,
                "filt_wr" : filt_wr,
                "lift"    : lift,
            }, f, indent=2)
        print(f"\n✅ Model saved → {MODEL_PATH}")
        print(f"   AUC={mean_auc:.4f}  base_WR={base_wr:.1%}  "
              f"filtered_WR={filt_wr:.1%}  lift={lift:+.1%}")
        print(f"   Uncomment scan step in run_overnight.bat to enable gate.")
    else:
        reasons = []
        if mean_auc < MIN_AUC:
            reasons.append(f"AUC {mean_auc:.4f} < min {MIN_AUC}")
        if n_filtered == 0:
            reasons.append(f"no OOS trades passed threshold {ML_THRESHOLD}")
        elif lift < MIN_LIFT:
            reasons.append(f"WR lift {lift:+.1%} < {MIN_LIFT:.0%} minimum")
        print(f"\n⚠  Model NOT saved — {' | '.join(reasons)}")
        print("   Rule-based Study 2 edge is sufficient for now.")
        print("   Re-train after accumulating more live trade data.")


# ─────────────────────────────────────────────
# SCORE A SINGLE SETUP (for evaluator integration)
# ─────────────────────────────────────────────

def score_setup(setup_dict: dict) -> float:
    """
    Score a single Study 2 setup from tonight_setup.json.
    Builds a feature vector matching FEATURE_COLS and returns
    the model's probability of a winning trade (0.0–1.0).
    Returns -1.0 if model file not found or setup not armed.
    """
    if not MODEL_PATH.exists():
        return -1.0

    bundle = joblib.load(MODEL_PATH)
    model  = bundle["model"]

    # Verify the saved model uses the same feature set
    saved_features = bundle.get("features", [])
    if saved_features != FEATURE_COLS:
        print(f"⚠  Model feature list mismatch — retrain recommended.")
        # Proceed anyway with current FEATURE_COLS

    s2 = setup_dict.get("study2", {})
    if not s2.get("armed"):
        return -1.0

    rally_mult  = float(s2.get("rally_mult", 0))
    selloff_pct = float(s2.get("selloff_pct", 0))
    atr_val     = float(s2.get("atr_val", 0))

    # retrace_ratio: same formula as build_features
    retrace_ratio = (selloff_pct / (rally_mult * 100)
                     if rally_mult > 0 else 0.0)
    retrace_ratio = float(np.clip(retrace_ratio, 0, 5))

    # rth_close_norm: we don't have the rolling mean at scan time,
    # use 0.0 (mean of the distribution) as a neutral value
    rth_close_norm = 0.0

    # Get current ET time for temporal features
    now = datetime.now(ET)
    day_of_week = now.weekday()    # 0=Mon … 4=Fri
    month       = now.month

    # Determine current session for session dummies
    t = now.hour * 60 + now.minute
    if t >= 16 * 60 or t < 7 * 60:
        session_asian, session_european, session_premarket = 1, 0, 0
    elif t < 9 * 60 + 30:
        session_asian, session_european, session_premarket = 0, 0, 1
    else:
        # RTH — shouldn't be scanning during RTH but handle gracefully
        session_asian, session_european, session_premarket = 0, 0, 0

    # Build feature vector in exact FEATURE_COLS order
    features = np.array([[
        rally_mult,
        selloff_pct,
        retrace_ratio,
        atr_val,
        rth_close_norm,
        day_of_week,
        month,
        session_asian,
        session_european,
        session_premarket,
    ]], dtype=np.float32)

    prob = float(model.predict_proba(features)[0, 1])
    return prob


def scan(setup_path: str):
    print("\n" + "=" * 60)
    print("STUDY 2 ML FILTER  |  Scan")
    print("=" * 60)

    if not MODEL_PATH.exists():
        print("⚠  No trained model found. Run: python study2_ml_filter.py --mode train")
        return

    bundle = joblib.load(MODEL_PATH)
    threshold = bundle.get("threshold", ML_THRESHOLD)

    with open(setup_path) as f:
        setup = json.load(f)

    s2 = setup.get("study2", {})
    if not s2.get("armed"):
        print("Study 2 not armed tonight — nothing to score.")
        return

    prob = score_setup(setup)
    decision = "✅ TAKE TRADE" if prob >= threshold else "❌ SKIP TRADE"

    print(f"Setup date   : {setup.get('date')}")
    print(f"Rally mult   : {s2.get('rally_mult')}x ATR")
    print(f"Selloff      : {s2.get('selloff_pct')}%")
    print(f"ML score     : {prob:.3f}  (threshold={threshold})")
    print(f"Decision     : {decision}")
    print(f"Base WR      : {bundle.get('base_wr', 0):.1%}  "
          f"Filtered WR: {bundle.get('filt_wr', 0):.1%}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study 2 ML Filter")
    parser.add_argument("--mode",  choices=["train", "evaluate", "scan"],
                        required=True)
    parser.add_argument("--data",  default="./mnq_study/study2_all_trades.csv",
                        help="Path to study2_all_trades.csv")
    parser.add_argument("--setup", default="./tonight_setup.json",
                        help="Path to tonight_setup.json (scan mode)")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data)
    elif args.mode == "evaluate":
        train(args.data)   # re-runs full validation
    elif args.mode == "scan":
        scan(args.setup)
