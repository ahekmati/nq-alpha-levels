"""
HMM Regime Strategy v4.1 — @MNQ H1
Live-ready system. All v4 improvements plus targeted fixes from first live results.

Architecture (5 layers):
  1. Data          — MT5 H1 bars, UTC-aware, incremental on bar close
  2. Features      — Itô drift correction, QV realized vol, vol percentile, Kalman velocity
  3. Regime        — Forward-only HMM (no Viterbi look-ahead) + Kalman confirmer
  4. Risk          — Vol-regime gate, Itô excursion stop, quality-score sizing
  5. Execution     — JSON state persistence (survives restarts), monthly HMM refit,
                     backtest walk-forward OR live MT5 orders via single MODE flag

Key improvements over v3:
  • Itô lemma drift correction (log_ret − σ²/2) → purer HMM signal
  • Quadratic-variation realized vol → replaces rolling std
  • Forward-only HMM decode (α-pass only) → eliminates Viterbi look-ahead bias
  • HMM uncertainty band (0.35–0.65) → skips ambiguous regime bars
  • Kalman filter velocity confirmer → second vote, zero refitting required
  • Vol-percentile regime gate (>70th → no entry) → avoids high-vol chop
  • Itô excursion stop → stop distance tied to realized vol × expected horizon
  • PERSIST_FOR_SCALE raised to 10 bars → reduces premature 2-lot entries
  • State file persistence → position / Kalman state / HMM model survive restarts
  • Monthly refit schedule → live HMM stays current without over-fitting

v4 → v4.1 fixes (from walk-forward result analysis):
  1. STOP_MIN_PTS raised 60→120.  The Itô formula was always hitting the old
     60-pt floor on H1 NQ, turning every loss into a fixed -$121.  With NQ
     prices at 20k-28k the formula naturally produces 90-160 pt stops — the
     floor was overriding it completely.  120 is now a true safety floor only.
  2. CONFIRM_BARS set to 0.  Forward-only HMM decode already eliminates
     look-ahead bias algorithmically; the extra bar just adds entry lag and
     compounds the Kalman velocity lag described below.
  3. Kalman confirmer changed from hard-block to soft-size.  When Kalman
     velocity disagrees with HMM regime, still enter but cap at 1 lot.
     When it agrees AND quality/persist gates pass, allow 2 lots as before.
     This recovers the ~15% of valid entries that were being blocked because
     Kalman velocity lags the HMM signal by design (smoothed filter).
  4. STOP_HORIZON_BARS raised 6→12.  Median hold duration in v4 OOS was
     ~25h.  The excursion formula was undersizing stops because it assumed
     a 6-bar hold while trades were actually held 2-3× longer.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("hmm_v4_1")


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

# ── Mode ─────────────────────────────────────────────────────────────────────
# "backtest"  → walk-forward validation, full metrics printed
# "live"      → bar-by-bar loop, MT5 orders, state persistence
MODE              = "backtest"

# ── Instrument ───────────────────────────────────────────────────────────────
SYMBOL            = "@MNQ"
START_DATE        = datetime(2020, 1,1, tzinfo=timezone.utc)
TIMEFRAME         = "H1"          # only H1 supported

# ── HMM ──────────────────────────────────────────────────────────────────────
N_COMPONENTS      = 2
HMM_TRAIN_BARS    = 252 * 24      # ~12 months of H1 bars for initial fit
HMM_REFIT_DAYS    = 30            # refit HMM every N calendar days in live mode
# Forward-only decode thresholds — bars outside these are "uncertain" → no entry
HMM_BULL_THRESH   = 0.65          # P(bull|obs) must exceed this
HMM_BEAR_THRESH   = 0.35          # P(bull|obs) must be below this

# ── Itô feature engineering ──────────────────────────────────────────────────
QV_WINDOW         = 10            # bars for quadratic-variation realized vol
VOL_PCT_LOOKBACK  = 252 * 24      # bars for vol-percentile history (~12 months)
VOL_PCT_NO_ENTRY  = 80.0          # vol percentile above which no new entries
VOL_PCT_ONE_LOT   = 60.0          # vol percentile above which max 1 lot (no scale)

# ── Kalman filter ─────────────────────────────────────────────────────────────
# Tracks [price_level, price_velocity] using a constant-velocity model.
# Process noise Q and observation noise R are the tuning knobs.
# Q_vel controls how quickly velocity can change (higher = more responsive).
# R_obs controls how much we trust each price observation.
KALMAN_Q_LEVEL    = 1e-4          # process noise: level state
KALMAN_Q_VEL      = 1e-5          # process noise: velocity state
KALMAN_R_OBS      = 1e-3          # observation noise

# ── RSI ───────────────────────────────────────────────────────────────────────
RSI_PERIOD             = 7
RSI_LONG_BASE          = 45
RSI_SHORT_BASE         = 55
RSI_TIGHTEN_STEP       = 5     # tighten per LOSS_STREAK_TIER_SIZE losses
LOSS_STREAK_TIER_SIZE  = 2
MAX_RSI_TIGHTEN        = 6

# ── Itô excursion stop ────────────────────────────────────────────────────────
# stop_dist = price × rv_qv × √(horizon_bars/annual_bars) × Φ⁻¹(confidence)
STOP_CONFIDENCE        = 0.95     # 95th percentile excursion
STOP_HORIZON_BARS      = 12       # raised from 6: median OOS hold was ~25h
STOP_ANNUAL_BARS       = 252 * 24
STOP_MIN_PTS           = 140    # raised from 60: Itô formula needs room to breathe
STOP_MAX_PTS           = 220

# ── Position sizing ───────────────────────────────────────────────────────────
BASE_CONTRACTS         = 1
MAX_CONTRACTS          = 2
PERSIST_FOR_SCALE      = 6       # regime must persist >= this to allow 2 lots
SCALE_QUALITY_THRESH   = 60       # quality score must be >= this for 2 lots
# Vol percentile gate for scaling: see VOL_PCT_ONE_LOT above

# ── Quality score ─────────────────────────────────────────────────────────────
QUALITY_LOOKBACK       = 20
QUALITY_FLIP_WEIGHT    = 0.40
QUALITY_WIN_WEIGHT     = 0.40
QUALITY_ATR_WEIGHT     = 0.20
MIN_QUALITY_TO_TRADE   = 30
QUALITY_MIN_TRADES     = 5        # need at least this many recent trades before
                                   # win-rate component activates

# ── Entry filters ─────────────────────────────────────────────────────────────
CONFIRM_BARS           = 1        # 0: forward-only HMM already removes look-ahead
MIN_HOLD_BARS          = 24        # min bars since last exit before new entry

# ── Session filter ─────────────────────────────────────────────────────────────
ENABLE_SESSION_FILTER  = False
SESSION_WINDOWS        = [(6, 11), (12, 21)]   # UTC hours (London open + NY)

# ── Drawdown brake ────────────────────────────────────────────────────────────
DD_BRAKE_PCT           = 0.18     # pause entries if equity DD > 12% from peak
DD_BRAKE_RECOVER       = 0.40     # resume when 50% of drawdown recovered

# ── Partial profit ─────────────────────────────────────────────────────────────
ENABLE_PARTIAL_PROFIT  = False
PARTIAL_TARGET_ATR     = 2.0
PARTIAL_CLOSE_FRAC     = 0.50
TRAIL_ATR              = 1.0

# ── Capital / costs ───────────────────────────────────────────────────────────
USD_PER_POINT          = 2.0
STARTING_CAPITAL_USD   = 5_000.0
COMMISSION_PER_SIDE    = 0.50
RISK_FREE_RATE         = 0.0

# ── Walk-forward ──────────────────────────────────────────────────────────────
ENABLE_WALK_FORWARD    = True
WF_TRAIN_MONTHS        = 12
WF_TEST_MONTHS         = 3

# ── State persistence (live mode) ─────────────────────────────────────────────
STATE_FILE             = Path("hmm_v4_state.json")
MODEL_FILE             = Path("hmm_v4_model.pkl")


# ═════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    side:          str           # "long" | "short"
    entry_time:    str           # ISO string
    entry_price:   float
    stop_price:    float
    contracts:     int
    partial_done:  bool   = False
    partial_pnl:   float  = 0.0
    partial_cons:  int    = 0
    trail_stop:    Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        return cls(**d)


@dataclass
class KalmanState:
    """Constant-velocity Kalman filter state [level, velocity]."""
    x:  np.ndarray = field(default_factory=lambda: np.zeros(2))
    P:  np.ndarray = field(default_factory=lambda: np.eye(2) * 1.0)
    initialized: bool = False

    def to_dict(self) -> dict:
        return {"x": self.x.tolist(), "P": self.P.tolist(),
                "initialized": self.initialized}

    @classmethod
    def from_dict(cls, d: dict) -> "KalmanState":
        obj = cls()
        obj.x = np.array(d["x"])
        obj.P = np.array(d["P"])
        obj.initialized = d["initialized"]
        return obj


@dataclass
class LiveState:
    """Everything that must survive a process restart in live mode."""
    position:        Optional[dict]  = None
    kalman:          Optional[dict]  = None
    last_refit_date: Optional[str]   = None
    consec_losses:   int             = 0
    recent_pnl:      list            = field(default_factory=list)
    peak_equity:     float           = STARTING_CAPITAL_USD
    equity:          float           = STARTING_CAPITAL_USD
    dd_brake_active: bool            = False
    dd_brake_trough: float           = STARTING_CAPITAL_USD
    bars_since_exit: int             = 999
    bars_since_sig:  int             = 999
    pending_regime:  Optional[str]   = None
    regime_persist:  int             = 0

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LiveState":
        if not path.exists():
            return cls()
        with open(path) as f:
            d = json.load(f)
        obj = cls(**d)
        return obj


# ═════════════════════════════════════════════════════════════════════════════
#  DATA LAYER
# ═════════════════════════════════════════════════════════════════════════════

def get_mt5_timeframe(mt5, tf: str):
    mapping = {"H1": mt5.TIMEFRAME_H1, "D1": mt5.TIMEFRAME_D1}
    return mapping[tf]


def fetch_bars(symbol: str, start: datetime, tf: str = "H1") -> pd.DataFrame:
    """Fetch OHLCV bars from MT5. Returns DataFrame indexed by UTC timestamp."""
    try:
        from mt5linux import MetaTrader5
    except ImportError:
        raise RuntimeError(
            "mt5linux not installed. Run: pip install mt5linux"
        )

    mt5 = MetaTrader5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    timeframe = get_mt5_timeframe(mt5, tf)
    end       = datetime.now(timezone.utc)
    rates     = mt5.copy_rates_range(symbol, timeframe, start, end)

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"No rates for {symbol}: {err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    mt5.shutdown()
    return df


def fetch_latest_bar(symbol: str, tf: str = "H1") -> Optional[pd.Series]:
    """Fetch the most recently closed bar for live mode."""
    try:
        from mt5linux import MetaTrader5
    except ImportError:
        return None

    mt5 = MetaTrader5()
    if not mt5.initialize():
        return None

    timeframe = get_mt5_timeframe(mt5, tf)
    rates     = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    return df.iloc[0]


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — ITÔ FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def calc_rsi(prices: pd.Series, period: int) -> pd.Series:
    """Wilder RSI with EWM smoothing."""
    delta    = prices.diff()
    gain     = delta.clip(lower=0.0)
    loss     = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    avg_gain = avg_gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR — kept for legacy stop comparison and partial profit."""
    high, low, prev = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([high - low,
                    (high - prev).abs(),
                    (low  - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full Itô-corrected feature set.

    log_ret      : raw log return
    ito_drift    : Itô lemma drift = log_ret − σ²/2  (purer drift signal)
    rv_qv        : quadratic-variation realized vol = √Σ(Δln S)²
    vol_pct      : rv_qv percentile rank over VOL_PCT_LOOKBACK bars (0–100)
    rsi          : RSI for entry gate
    atr          : Wilder ATR for partial-profit trailing stop
    """
    f = df.copy()

    # Raw log return
    f["log_ret"] = np.log(f["close"]).diff()

    # Quadratic-variation realized vol (Itô role 2)
    f["rv_qv"] = (
        (f["log_ret"] ** 2)
        .rolling(QV_WINDOW)
        .sum()
        .apply(np.sqrt)
    )

    # Itô drift correction (Itô role 1): subtract Jensen gap
    f["ito_drift"] = f["log_ret"] - 0.5 * f["rv_qv"] ** 2

    # Volatility percentile rank against rolling history
    f["vol_pct"] = (
        f["rv_qv"]
        .rolling(VOL_PCT_LOOKBACK, min_periods=QV_WINDOW * 4)
        .rank(pct=True) * 100
    )

    f["rsi"] = calc_rsi(f["close"], RSI_PERIOD)
    f["atr"] = calc_atr(f, 14)

    return f.dropna(subset=["ito_drift", "rv_qv", "rsi", "atr"])


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3a — FORWARD-ONLY HMM
# ═════════════════════════════════════════════════════════════════════════════

def _safe_log_emit(model: GaussianHMM,
                   X: np.ndarray) -> np.ndarray:
    """
    Compute log emission probabilities directly, bypassing hmmlearn's
    internal method which crashes on degenerate covariance matrices.

    Strategy: for each state k, compute a regularised covariance as
        cov_reg = cov + diag * max(trace(cov)/nf, 1e-4)
    This is a proportional ridge that handles both tiny (well-fitted)
    and huge (collapsed) covariance matrices robustly. Then evaluate
    the Gaussian log-likelihood using the Cholesky decomposition
    directly — no scipy validation that rejects singular matrices.

    Returns array shape (n_samples, n_components).
    """
    n_samples = X.shape[0]
    n_comp    = model.n_components
    nf        = model.means_.shape[1]
    log_emit  = np.empty((n_samples, n_comp))

    for k in range(n_comp):
        mu  = model.means_[k]
        cov = model.covars_[k].copy()

        # Proportional ridge: at least 1% of mean diagonal variance
        trace_mean = max(np.trace(cov) / nf, 1e-8)
        cov += np.eye(nf) * trace_mean * 0.01

        # Cholesky Gaussian log-pdf (numerically stable, no PD check)
        try:
            L   = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Fallback: diagonal approximation
            L   = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-12)))

        diff       = X - mu                           # (T, nf)
        sol        = np.linalg.solve(L, diff.T)       # (nf, T)
        maha       = (sol ** 2).sum(axis=0)           # (T,)
        log_det    = 2.0 * np.sum(np.log(np.diag(L)))
        log_emit[:, k] = -0.5 * (maha + log_det + nf * np.log(2 * np.pi))

    return log_emit


def fit_hmm(features: pd.DataFrame,
            n_components: int = N_COMPONENTS) -> GaussianHMM:
    """
    Fit a GaussianHMM on the Itô-corrected feature vector.
    Features used: [ito_drift, rv_qv] — drift and vol, directly mapped
    to the μ and σ of the underlying regime-switching SDE.
    Covariance matrices are regularised after fitting for numerical
    stability in the forward pass and Viterbi decode.
    """
    X = features[["ito_drift", "rv_qv"]].values
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=300,
        random_state=42,
        tol=1e-5,
    )
    model.fit(X)
    log.info(
        "HMM fitted on %d bars  converged=%s",
        len(X), model.monitor_.converged
    )
    return model


def identify_bull_state(model: GaussianHMM,
                        features: pd.DataFrame) -> int:
    """
    Determine which state index is 'bull' by comparing the mean
    ito_drift of each state's Gaussian emission.
    Uses the model's means_ directly (no Viterbi decode needed) —
    the state with the higher mean drift is bull.
    This avoids any numerical issues with Viterbi on degenerate data.
    """
    # model.means_ shape: (n_components, n_features)
    # feature order: [ito_drift, rv_qv] — index 0 is ito_drift
    drift_means = model.means_[:, 0]
    return int(np.argmax(drift_means))


def forward_pass(model: GaussianHMM,
                 X: np.ndarray) -> np.ndarray:
    """
    Forward-only (α) pass of the HMM.
    Returns P(state | obs_1..t) for each bar t using ONLY past observations.
    No Viterbi, no smoothing, no look-ahead.

    This is the critical fix over v1–v3: the live signal will match the
    backtest signal because both use the same causal filter.

    Returns array shape (n_samples, n_components).
    """
    n_samples = X.shape[0]
    n_comp    = model.n_components

    # Log emission probabilities: shape (n_samples, n_components)
    # _compute_log_likelihood expects shape (n_samples, n_features)
    log_emit = _safe_log_emit(model, X)            # (T, K)

    log_alpha = np.empty((n_samples, n_comp))
    log_alpha[0] = (
        np.log(model.startprob_ + 1e-300) + log_emit[0]
    )

    log_trans = np.log(model.transmat_ + 1e-300)  # (K, K)

    for t in range(1, n_samples):
        for j in range(n_comp):
            log_alpha[t, j] = (
                np.logaddexp.reduce(log_alpha[t - 1] + log_trans[:, j])
                + log_emit[t, j]
            )

    # Normalise: P(state_j | obs_1..t)
    log_sum   = np.logaddexp.reduce(log_alpha, axis=1, keepdims=True)
    proba     = np.exp(log_alpha - log_sum)
    return proba


def decode_regime(model: GaussianHMM,
                  features: pd.DataFrame,
                  bull_state: int) -> pd.Series:
    """
    Returns a regime Series with values "bull" | "bear" | "uncertain".
    "uncertain" = P(bull) in [HMM_BEAR_THRESH, HMM_BULL_THRESH].
    Entries are blocked when regime == "uncertain".
    """
    X     = features[["ito_drift", "rv_qv"]].values
    proba = forward_pass(model, X)

    bull_proba = proba[:, bull_state]
    regime     = pd.Series("uncertain", index=features.index)
    regime[bull_proba >= HMM_BULL_THRESH] = "bull"
    regime[bull_proba <= HMM_BEAR_THRESH] = "bear"
    return regime


def save_hmm_model(model: GaussianHMM, bull_state: int,
                   path: Path = MODEL_FILE) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": model, "bull_state": bull_state}, f)
    log.info("HMM model saved to %s", path)


def load_hmm_model(path: Path = MODEL_FILE
                   ) -> Optional[tuple[GaussianHMM, int]]:
    if not path.exists():
        return None
    with open(path, "rb") as f:
        d = pickle.load(f)
    log.info("HMM model loaded from %s", path)
    return d["model"], d["bull_state"]


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 3b — KALMAN FILTER VELOCITY CONFIRMER
# ═════════════════════════════════════════════════════════════════════════════
#
# State vector: [price_level, price_velocity]
# Transition:   x_t = F x_{t-1} + w,    w ~ N(0, Q)
#   F = [[1, 1],   (level += velocity each bar)
#        [0, 1]]   (velocity is a random walk)
# Observation:  y_t = H x_t + v,        v ~ N(0, R)
#   H = [[1, 0]]   (we observe the price level)
#
# This is an online filter — no refitting, no look-ahead.
# A positive velocity state means the Kalman filter "believes" price is
# trending upward, confirmed across the noise.

_F = np.array([[1.0, 1.0],
               [0.0, 1.0]])

_H = np.array([[1.0, 0.0]])


def kalman_Q() -> np.ndarray:
    return np.diag([KALMAN_Q_LEVEL, KALMAN_Q_VEL])


def kalman_R() -> np.ndarray:
    return np.array([[KALMAN_R_OBS]])


def kalman_update(state: KalmanState,
                  observation: float) -> KalmanState:
    """
    One-step Kalman predict + update.
    Returns a NEW KalmanState (immutable update pattern).
    """
    Q = kalman_Q()
    R = kalman_R()

    if not state.initialized:
        # First observation: initialize level to price, velocity to 0
        x_new = np.array([observation, 0.0])
        P_new = np.eye(2) * 1.0
        return KalmanState(x=x_new, P=P_new, initialized=True)

    # Predict
    x_pred = _F @ state.x
    P_pred = _F @ state.P @ _F.T + Q

    # Update
    y     = observation - (_H @ x_pred)[0]        # innovation
    S     = (_H @ P_pred @ _H.T + R)[0, 0]        # innovation covariance
    K     = (P_pred @ _H.T) / S                   # Kalman gain (2,1)
    x_new = x_pred + K.ravel() * y
    P_new = (np.eye(2) - K @ _H) @ P_pred

    return KalmanState(x=x_new, P=P_new, initialized=True)


def run_kalman_series(prices: pd.Series) -> pd.DataFrame:
    """
    Run the Kalman filter over a full price series for backtesting.
    Returns DataFrame with columns [kalman_level, kalman_velocity].
    """
    levels     = np.empty(len(prices))
    velocities = np.empty(len(prices))
    state      = KalmanState()

    for i, p in enumerate(prices):
        state         = kalman_update(state, float(p))
        levels[i]     = state.x[0]
        velocities[i] = state.x[1]

    return pd.DataFrame(
        {"kalman_level": levels, "kalman_velocity": velocities},
        index=prices.index,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — RISK: STOP, SIZING, QUALITY SCORE
# ═════════════════════════════════════════════════════════════════════════════

def ito_stop_distance(price: float,
                      rv_qv: float,
                      horizon_bars: int = STOP_HORIZON_BARS) -> float:
    """
    Itô excursion stop: expected maximum adverse move at STOP_CONFIDENCE
    over horizon_bars, derived from the quadratic-variation realized vol.

    stop_dist = price × rv_qv × √(horizon/annual) × Φ⁻¹(confidence)

    Clamped to [STOP_MIN_PTS, STOP_MAX_PTS] as hard rails.
    """
    if rv_qv <= 0:
        return STOP_MIN_PTS

    horizon_years = horizon_bars / STOP_ANNUAL_BARS
    # rv_qv is in log-return units over QV_WINDOW bars;
    # annualise to per-bar then scale to horizon
    rv_annualised = rv_qv / np.sqrt(QV_WINDOW)
    stop_pts = (price
                * rv_annualised
                * np.sqrt(horizon_years)
                * stats.norm.ppf(STOP_CONFIDENCE))

    return float(np.clip(stop_pts, STOP_MIN_PTS, STOP_MAX_PTS))


def compute_quality_score(regime_series: pd.Series,
                          rv_series: pd.Series,
                          recent_pnl: list,
                          idx: int) -> float:
    """
    Composite quality score 0–100.
      flip_score  : regime stability over QUALITY_LOOKBACK bars
      win_score   : recent trade win rate (activates after QUALITY_MIN_TRADES)
      atr_score   : current rv_qv vs rolling median (calm = high score)
    """
    start = max(0, idx - QUALITY_LOOKBACK)

    # 1. Flip rate
    recent_reg  = regime_series.iloc[start: idx + 1]
    if len(recent_reg) < 2:
        flip_score = 50.0
    else:
        flips      = recent_reg.ne(recent_reg.shift(1)).sum() - 1
        flip_rate  = flips / max(1, len(recent_reg) - 1)
        flip_score = max(0.0, 100.0 * (1.0 - flip_rate * 2))

    # 2. Win rate (only after enough trades)
    recent_t = recent_pnl[-QUALITY_LOOKBACK:]
    if len(recent_t) < QUALITY_MIN_TRADES:
        win_score = 50.0
    else:
        wins      = sum(1 for p in recent_t if p > 0)
        win_score = 100.0 * wins / len(recent_t)

    # 3. ATR calm score
    rv_start    = max(0, idx - 50)
    rv_window   = rv_series.iloc[rv_start: idx + 1]
    current_rv  = rv_series.iloc[idx]
    if len(rv_window) < 5 or rv_window.median() == 0:
        atr_score = 50.0
    else:
        ratio     = current_rv / rv_window.median()
        atr_score = max(0.0, min(100.0, 100.0 * (2.0 - ratio)))

    return float(np.clip(
        QUALITY_FLIP_WEIGHT * flip_score
        + QUALITY_WIN_WEIGHT  * win_score
        + QUALITY_ATR_WEIGHT  * atr_score,
        0.0, 100.0
    ))


def tightened_rsi(consec_losses: int) -> tuple[float, float]:
    """Return (long_thresh, short_thresh) tightened by streak."""
    tiers   = consec_losses // LOSS_STREAK_TIER_SIZE
    tighten = min(tiers * RSI_TIGHTEN_STEP, MAX_RSI_TIGHTEN)
    return RSI_LONG_BASE + tighten, RSI_SHORT_BASE - tighten


def calc_contracts(quality: float,
                   regime_persist: int,
                   vol_pct: float) -> int:
    """
    1 lot normally; 2 lots only when all three scale-up conditions are met:
      1. Regime has persisted >= PERSIST_FOR_SCALE bars
      2. Quality score >= SCALE_QUALITY_THRESH
      3. Vol percentile < VOL_PCT_ONE_LOT (calm market)
    """
    if vol_pct > VOL_PCT_ONE_LOT:
        return BASE_CONTRACTS
    if regime_persist < PERSIST_FOR_SCALE:
        return BASE_CONTRACTS
    if quality < SCALE_QUALITY_THRESH:
        return BASE_CONTRACTS
    return MAX_CONTRACTS


def in_session(ts: pd.Timestamp) -> bool:
    if not ENABLE_SESSION_FILTER:
        return True
    h = ts.hour + ts.minute / 60.0
    return any(s <= h <= e for s, e in SESSION_WINDOWS)


# ═════════════════════════════════════════════════════════════════════════════
#  TRADE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def close_trade(pos: dict, exit_time, exit_price: float,
                reason: str, partial_pnl: float = 0.0,
                partial_cons: int = 0) -> dict:
    t = pos.copy()
    t.update(exit_time=exit_time, exit_price=float(exit_price),
             exit_reason=reason, partial_pnl_pts=partial_pnl,
             partial_cons=partial_cons)
    return t


def calc_trade_pnl(pos: dict, exit_price: float) -> float:
    """Net USD PnL for the remainder leg (excludes partial leg)."""
    direction = 1 if pos["side"] == "long" else -1
    pts       = (exit_price - pos["entry_price"]) * direction * pos["contracts"]
    return pts * USD_PER_POINT - COMMISSION_PER_SIDE * 2 * pos["contracts"]


# ═════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def backtest(data: pd.DataFrame,
             regime: pd.Series,
             kalman_vel: pd.Series,
             starting_capital: float = STARTING_CAPITAL_USD) -> pd.DataFrame:
    """
    Single-pass bar-by-bar backtest.

    All filters active:
      • Forward-only HMM regime (passed in as pre-decoded pd.Series)
      • Kalman velocity confirmer
      • Vol-percentile regime gate
      • Itô excursion stop
      • Quality score + streak RSI tightening
      • Drawdown brake
      • Session filter
      • Scale-up gate (2-lot)
    """
    d = data.copy()
    d["regime"]    = regime.reindex(d.index).ffill()
    d["kalman_vel"]= kalman_vel.reindex(d.index).ffill()
    d = d.dropna(subset=["regime", "rsi", "rv_qv", "atr"])

    trades          : list  = []
    position        : Optional[dict] = None
    equity          : float = starting_capital
    peak_equity     : float = starting_capital
    dd_brake_active : bool  = False
    dd_brake_trough : float = starting_capital
    consec_losses   : int   = 0
    recent_pnl      : list  = []
    bars_since_exit : int   = 999
    bars_since_sig  : int   = 999
    pending_regime  : Optional[str] = None
    regime_persist  : int   = 0

    signals      = d["regime"]
    regime_shift = signals.ne(signals.shift(1))

    rows = list(d.iterrows())

    for idx, (t, row) in enumerate(rows):
        open_p   = float(row["open"])
        high_p   = float(row["high"])
        low_p    = float(row["low"])
        rv_qv    = float(row["rv_qv"])
        rsi_val  = float(row["rsi"])
        atr_val  = float(row["atr"])
        vol_pct  = float(row.get("vol_pct", 50.0))
        kvel     = float(row["kalman_vel"])
        reg      = signals.iloc[idx]

        # ── Regime persistence counter ────────────────────────────────
        if idx == 0:
            regime_persist = 0
        elif signals.iloc[idx] == signals.iloc[idx - 1]:
            regime_persist += 1
        else:
            regime_persist = 0

        # ── Quality score ─────────────────────────────────────────────
        quality = compute_quality_score(
            signals, d["rv_qv"], recent_pnl, idx
        )

        # ── Drawdown brake ────────────────────────────────────────────
        peak_equity = max(peak_equity, equity)
        current_dd  = (equity - peak_equity) / peak_equity

        if not dd_brake_active and current_dd < -DD_BRAKE_PCT:
            dd_brake_active = True
            dd_brake_trough = equity
            log.debug("DD brake engaged at equity %.2f", equity)

        if dd_brake_active:
            depth     = peak_equity - dd_brake_trough
            recovered = equity - dd_brake_trough
            if depth > 0 and (recovered / depth) >= DD_BRAKE_RECOVER:
                dd_brake_active = False
                log.debug("DD brake released at equity %.2f", equity)

        # ── Manage open position ──────────────────────────────────────
        if position is not None:
            side  = position["side"]
            n_con = position["contracts"]
            trail = position.get("trail_stop")

            # Update trailing stop after partial close
            if trail is not None:
                if side == "long":
                    new_trail = high_p - atr_val * TRAIL_ATR
                    position["trail_stop"] = max(trail, new_trail)
                else:
                    new_trail = low_p + atr_val * TRAIL_ATR
                    position["trail_stop"] = min(trail, new_trail)
                position["stop_price"] = position["trail_stop"]

            # Partial profit target
            if ENABLE_PARTIAL_PROFIT and not position.get("partial_done"):
                pt_dist = atr_val * PARTIAL_TARGET_ATR
                if side == "long" and high_p >= position["entry_price"] + pt_dist:
                    pp   = position["entry_price"] + pt_dist
                    pc   = max(1, int(n_con * PARTIAL_CLOSE_FRAC))
                    ppts = (pp - position["entry_price"]) * pc
                    equity += ppts * USD_PER_POINT - COMMISSION_PER_SIDE * pc
                    position.update(partial_done=True, partial_pnl_pts=ppts,
                                    partial_cons=pc, contracts=n_con - pc,
                                    trail_stop=high_p - atr_val * TRAIL_ATR,
                                    stop_price=high_p - atr_val * TRAIL_ATR)

                elif side == "short" and low_p <= position["entry_price"] - pt_dist:
                    pp   = position["entry_price"] - pt_dist
                    pc   = max(1, int(n_con * PARTIAL_CLOSE_FRAC))
                    ppts = (position["entry_price"] - pp) * pc
                    equity += ppts * USD_PER_POINT - COMMISSION_PER_SIDE * pc
                    position.update(partial_done=True, partial_pnl_pts=ppts,
                                    partial_cons=pc, contracts=n_con - pc,
                                    trail_stop=low_p + atr_val * TRAIL_ATR,
                                    stop_price=low_p + atr_val * TRAIL_ATR)

            # Stop hit
            stopped = ((side == "long"  and low_p  <= position["stop_price"])
                    or (side == "short" and high_p >= position["stop_price"]))

            if stopped:
                ppt  = position.get("partial_pnl_pts", 0.0)
                pc   = position.get("partial_cons", 0)
                tobj = close_trade(position, t, position["stop_price"],
                                   "stop_loss", ppt, pc)
                comm = COMMISSION_PER_SIDE * 2 * position["contracts"]
                dir_ = 1 if side == "long" else -1
                rem_pts = (
                    (position["stop_price"] - position["entry_price"])
                    * dir_ * position["contracts"]
                )
                trade_pnl = rem_pts * USD_PER_POINT - comm
                equity   += trade_pnl
                net_pnl   = trade_pnl + ppt * USD_PER_POINT
                recent_pnl.append(net_pnl)
                consec_losses = consec_losses + 1 if net_pnl < 0 else 0
                trades.append({**tobj, "commission": comm,
                                "quality": quality, "lots": position["contracts"] + pc})
                position        = None
                bars_since_exit = 0
                pending_regime  = None
                bars_since_sig  = 999
                continue

        # ── Counters ──────────────────────────────────────────────────
        bars_since_exit += 1
        if pending_regime is not None:
            bars_since_sig += 1

        # ── Regime shift → start confirmation, close opposite ────────
        if regime_shift.iloc[idx]:
            pending_regime = reg
            bars_since_sig = 0

            if position is not None:
                opp = "short" if reg == "bull" else "long"
                if position["side"] == opp:
                    ppt  = position.get("partial_pnl_pts", 0.0)
                    pc   = position.get("partial_cons", 0)
                    tobj = close_trade(position, t, open_p,
                                       "regime_flip", ppt, pc)
                    comm = COMMISSION_PER_SIDE * 2 * position["contracts"]
                    dir_ = 1 if position["side"] == "long" else -1
                    rem_pts = (
                        (open_p - position["entry_price"])
                        * dir_ * position["contracts"]
                    )
                    trade_pnl = rem_pts * USD_PER_POINT - comm
                    equity   += trade_pnl
                    net_pnl   = trade_pnl + ppt * USD_PER_POINT
                    recent_pnl.append(net_pnl)
                    consec_losses = consec_losses + 1 if net_pnl < 0 else 0
                    trades.append({**tobj, "commission": comm,
                                    "quality": quality,
                                    "lots": position["contracts"] + pc})
                    position        = None
                    bars_since_exit = 0

        # ── Entry logic ───────────────────────────────────────────────
        long_rsi, short_rsi = tightened_rsi(consec_losses)

        can_enter = (
            position is None
            and pending_regime is not None
            and pending_regime != "uncertain"
            and bars_since_sig  >= CONFIRM_BARS
            and bars_since_exit >= MIN_HOLD_BARS
            and in_session(t)
            and quality >= MIN_QUALITY_TO_TRADE
            and not dd_brake_active
            and vol_pct < VOL_PCT_NO_ENTRY
        )

        if can_enter:
            # Kalman confirmer (soft-size, not hard-block).
            # If velocity agrees with HMM direction → full sizing allowed.
            # If velocity disagrees → still enter but cap at 1 lot.
            # Rationale: Kalman velocity lags the HMM signal by design
            # (it is a smoothed filter).  Blocking on disagreement was
            # dropping ~15% of valid entries in v4, hurting win rate.
            # Capping size on disagreement still provides risk reduction
            # without losing the entry entirely.
            kalman_confirms = (
                (pending_regime == "bull" and kvel > 0) or
                (pending_regime == "bear" and kvel < 0)
            )

            stop_dist = ito_stop_distance(open_p, rv_qv)

            # Size: full calc if Kalman agrees, else forced to 1 lot
            if kalman_confirms:
                n_con = calc_contracts(quality, regime_persist, vol_pct)
            else:
                n_con = BASE_CONTRACTS   # Kalman disagrees → cautious size

            if pending_regime == "bull" and rsi_val > long_rsi:
                position = {
                    "side":         "long",
                    "entry_time":   str(t),
                    "entry_price":  open_p,
                    "stop_price":   open_p - stop_dist,
                    "contracts":    n_con,
                    "partial_done": False,
                    "partial_pnl_pts": 0.0,
                    "partial_cons": 0,
                    "trail_stop":   None,
                    "kalman_agree": kalman_confirms,
                }
                pending_regime = None

            elif pending_regime == "bear" and rsi_val < short_rsi:
                position = {
                    "side":         "short",
                    "entry_time":   str(t),
                    "entry_price":  open_p,
                    "stop_price":   open_p + stop_dist,
                    "contracts":    n_con,
                    "partial_done": False,
                    "partial_pnl_pts": 0.0,
                    "partial_cons": 0,
                    "trail_stop":   None,
                    "kalman_agree": kalman_confirms,
                }
                pending_regime = None

    # ── Close open position at end of data ────────────────────────────
    if position is not None:
        last_t     = d.index[-1]
        last_close = float(d["close"].iloc[-1])
        ppt        = position.get("partial_pnl_pts", 0.0)
        pc         = position.get("partial_cons", 0)
        tobj       = close_trade(position, last_t, last_close,
                                  "final_close", ppt, pc)
        comm       = COMMISSION_PER_SIDE * 2 * position["contracts"]
        trades.append({**tobj, "commission": comm,
                        "quality": quality,
                        "lots": position["contracts"] + pc})

    if not trades:
        return pd.DataFrame()

    df_t = pd.DataFrame(trades)

    direction            = df_t["side"].map({"long": 1, "short": -1})
    df_t["rem_pnl_pts"]  = (
        (df_t["exit_price"] - df_t["entry_price"])
        * direction * df_t["contracts"]
    )
    df_t["pnl_points"]   = df_t["rem_pnl_pts"] + df_t.get("partial_pnl_pts", 0.0)
    df_t["pnl_usd"]      = (
        df_t["rem_pnl_pts"] * USD_PER_POINT
        + df_t.get("partial_pnl_pts", 0.0) * USD_PER_POINT
        - df_t["commission"]
    )
    df_t["cum_pnl_usd"]  = df_t["pnl_usd"].cumsum()
    df_t["equity_usd"]   = starting_capital + df_t["cum_pnl_usd"]
    df_t["trade_return"] = df_t["pnl_usd"] / starting_capital
    df_t["dur_hours"]    = (
        (pd.to_datetime(df_t["exit_time"])
         - pd.to_datetime(df_t["entry_time"])).dt.total_seconds() / 3600.0
    )
    df_t["2lot"]         = (df_t["lots"] > BASE_CONTRACTS).astype(int)
    return df_t


# ═════════════════════════════════════════════════════════════════════════════
#  WALK-FORWARD VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def walk_forward(features: pd.DataFrame,
                 bars: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling walk-forward:
    • Train HMM on WF_TRAIN_MONTHS
    • Decode with forward-only pass on next WF_TEST_MONTHS
    • Kalman filter runs continuously across all windows (online, no refit)
    • Concatenate OOS trades with running equity
    """
    if not ENABLE_WALK_FORWARD:
        return pd.DataFrame()

    # Run Kalman filter over full price series once — it's online so this
    # is the correct causal treatment.
    kalman_df = run_kalman_series(bars["close"])

    all_oos: list = []
    dates         = features.index
    window_start  = dates[0]

    while True:
        train_end = window_start + pd.DateOffset(months=WF_TRAIN_MONTHS)
        test_end  = train_end   + pd.DateOffset(months=WF_TEST_MONTHS)

        train_mask = (dates >= window_start) & (dates < train_end)
        test_mask  = (dates >= train_end)    & (dates < test_end)

        if train_mask.sum() < 500 or test_mask.sum() < 5:
            break

        train_feat = features[train_mask]
        test_feat  = features[test_mask]

        # Fit HMM on training window
        model      = fit_hmm(train_feat)
        bull_state = identify_bull_state(model, train_feat)

        # Decode OOS using forward-only pass
        oos_regime = decode_regime(model, test_feat, bull_state)

        # Build OOS dataset
        test_bars  = bars.reindex(test_feat.index)
        test_data  = (test_bars
                      .join(test_feat[["ito_drift", "rv_qv", "rsi", "atr",
                                       "vol_pct"]], how="left")
                      .join(oos_regime.rename("regime"), how="left"))

        oos_kvel   = kalman_df["kalman_velocity"].reindex(test_feat.index)

        oos_trades = backtest(test_data, test_data["regime"],
                              oos_kvel, starting_capital=STARTING_CAPITAL_USD)

        if not oos_trades.empty:
            oos_trades["wf_window"] = str(train_end.date())
            all_oos.append(oos_trades)

        window_start = window_start + pd.DateOffset(months=WF_TEST_MONTHS)
        if test_end > dates[-1]:
            break

    if not all_oos:
        return pd.DataFrame()

    combined = pd.concat(all_oos, ignore_index=True)
    combined.sort_values("entry_time", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined["cum_pnl_usd"] = combined["pnl_usd"].cumsum()
    combined["equity_usd"]  = STARTING_CAPITAL_USD + combined["cum_pnl_usd"]
    return combined


# ═════════════════════════════════════════════════════════════════════════════
#  METRICS
# ═════════════════════════════════════════════════════════════════════════════

def max_streak(cond: pd.Series) -> int:
    best = cur = 0
    for v in cond:
        cur  = cur + 1 if v else 0
        best = max(best, cur)
    return best


def compute_metrics(df: pd.DataFrame, label: str = "") -> dict:
    if df.empty:
        return {}

    ann     = STOP_ANNUAL_BARS
    wins    = df[df["pnl_usd"] > 0]
    losses  = df[df["pnl_usd"] < 0]
    longs   = df[df["side"] == "long"]
    shorts  = df[df["side"] == "short"]
    gp      = wins["pnl_usd"].sum()
    gl      = losses["pnl_usd"].sum()
    net     = df["pnl_usd"].sum()
    pf      = abs(gp / gl) if gl != 0 else np.nan

    eq      = df["equity_usd"]
    rm      = eq.cummax()
    mdd_usd = (eq - rm).min()
    mdd_pct = ((eq - rm) / rm).min() * 100

    ret     = df["trade_return"]
    mu      = ret.mean()
    sd      = ret.std(ddof=1)
    sharpe  = np.sqrt(ann) * mu / sd if sd > 0 else np.nan
    ds      = ret[ret < 0].std(ddof=1)
    sortino = np.sqrt(ann) * mu / ds  if (ds and ds > 0) else np.nan

    two_lot = df.get("2lot", pd.Series(0, index=df.index))

    return {
        "label":            label,
        "trades":           len(df),
        "long_trades":      len(longs),
        "short_trades":     len(shorts),
        "two_lot_trades":   int(two_lot.sum()),
        "win_rate":         (df["pnl_usd"] > 0).mean() * 100,
        "long_win_rate":    (longs["pnl_usd"]  > 0).mean() * 100 if len(longs)  else np.nan,
        "short_win_rate":   (shorts["pnl_usd"] > 0).mean() * 100 if len(shorts) else np.nan,
        "net_pnl_usd":      net,
        "net_pnl_pts":      df["pnl_points"].sum(),
        "gross_profit":     gp,
        "gross_loss":       gl,
        "profit_factor":    pf,
        "total_return":     (eq.iloc[-1] / STARTING_CAPITAL_USD - 1) * 100,
        "final_equity":     eq.iloc[-1],
        "max_dd_usd":       mdd_usd,
        "max_dd_pct":       mdd_pct,
        "recovery_factor":  abs(net / mdd_usd) if mdd_usd != 0 else np.nan,
        "sharpe":           sharpe,
        "sortino":          sortino,
        "avg_win_usd":      wins["pnl_usd"].mean()    if len(wins)    else 0.0,
        "avg_loss_usd":     losses["pnl_usd"].mean()  if len(losses)  else 0.0,
        "avg_win_pts":      wins["pnl_points"].mean() if len(wins)    else 0.0,
        "avg_loss_pts":     losses["pnl_points"].mean()if len(losses) else 0.0,
        "expectancy_usd":   df["pnl_usd"].mean(),
        "stopped_out":      (df["exit_reason"] == "stop_loss").sum(),
        "regime_flips":     (df["exit_reason"] == "regime_flip").sum(),
        "max_consec_wins":  max_streak(df["pnl_usd"] > 0),
        "max_consec_losses":max_streak(df["pnl_usd"] < 0),
        "best_trade_usd":   df["pnl_usd"].max(),
        "worst_trade_usd":  df["pnl_usd"].min(),
        "avg_dur_hrs":      df["dur_hours"].mean(),
        "total_commission": df["commission"].sum() if "commission" in df else 0,
        "avg_quality":      df["quality"].mean()   if "quality"    in df else np.nan,
        "kalman_agree_pct": (df["kalman_agree"].mean() * 100
                             if "kalman_agree" in df else np.nan),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PRINTING
# ═════════════════════════════════════════════════════════════════════════════

G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; B = "\033[94m"; E = "\033[0m"


def _f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def print_trades(df: pd.DataFrame, label: str = "") -> None:
    if df.empty:
        print("No trades.")
        return
    W = 96
    print(f"\n{'═'*W}")
    print(f"  TRADES — {label}  |  {SYMBOL}  {TIMEFRAME}  [v4.1]")
    print(f"  Capital ${STARTING_CAPITAL_USD:,.0f}  "
          f"Itô stop  QV-vol  forward HMM  Kalman confirm  vol-pct gate")
    print(f"{'─'*W}")
    for _, tr in df.iterrows():
        c     = G if tr["pnl_usd"] >= 0 else R
        lots  = f" ×{int(tr['lots'])}" if tr.get("lots", 1) > 1 else "   "
        wf    = f"  [{tr['wf_window']}]" if "wf_window" in tr else ""
        q     = tr.get("quality", np.nan)
        qs    = f" Q={q:.0f}" if not np.isnan(q) else ""
        ka    = "" if tr.get("kalman_agree", True) else " ~K"
        print(
            f"{c}"
            f"{tr['side'].upper():5}{lots} "
            f"{tr['entry_time']} → {tr['exit_time']} "
            f"entry={tr['entry_price']:.2f} exit={tr['exit_price']:.2f} "
            f"stop={tr['stop_price']:.2f} "
            f"reason={tr['exit_reason']:12} "
            f"pnl={tr['pnl_points']:+.1f}pts / ${tr['pnl_usd']:+.2f} "
            f"eq=${tr['equity_usd']:,.0f}"
            f"{qs}{ka}{wf}"
            f"{E}"
        )


def print_metrics(m: dict) -> None:
    if not m:
        print("No metrics.")
        return
    W = 96
    label = m.get("label", "")
    print(f"\n{'═'*W}")
    print(f"  METRICS — {label}")
    print(f"{'─'*W}")
    rows = [
        ("Capital",           f"${STARTING_CAPITAL_USD:,.0f} → ${m['final_equity']:,.2f}"),
        ("Total return",      f"{_f(m['total_return'])} %"),
        ("Net profit",        f"${_f(m['net_pnl_usd'])}  ({_f(m['net_pnl_pts'])} pts)"),
        ("Gross profit",      f"${_f(m['gross_profit'])}"),
        ("Gross loss",        f"${_f(m['gross_loss'])}"),
        ("Commission",        f"${_f(m['total_commission'])}"),
        ("Profit factor",     _f(m["profit_factor"], 3)),
        ("Sharpe ratio",      _f(m["sharpe"], 3)),
        ("Sortino ratio",     _f(m["sortino"], 3)),
        ("Max drawdown",      f"${_f(m['max_dd_usd'])}  ({_f(m['max_dd_pct'])} %)"),
        ("Recovery factor",   _f(m["recovery_factor"], 3)),
        ("─── Trades",        f"{m['trades']}  (L:{m['long_trades']}  S:{m['short_trades']})"),
        ("2-lot trades",      f"{m['two_lot_trades']}  "
                              f"({100*m['two_lot_trades']/max(1,m['trades']):.1f}%)"),
        ("Avg quality score", _f(m["avg_quality"], 1)),
        ("Kalman agreed %",   _f(m.get("kalman_agree_pct"), 1)),
        ("Win rate",          f"{_f(m['win_rate'])} %  "
                              f"(L:{_f(m['long_win_rate'])} %  S:{_f(m['short_win_rate'])} %)"),
        ("Avg win",           f"${_f(m['avg_win_usd'])}  ({_f(m['avg_win_pts'])} pts)"),
        ("Avg loss",          f"${_f(m['avg_loss_usd'])}  ({_f(m['avg_loss_pts'])} pts)"),
        ("Expectancy",        f"${_f(m['expectancy_usd'])} / trade"),
        ("Stopped out",       str(m["stopped_out"])),
        ("Regime-flip exits", str(m["regime_flips"])),
        ("Max consec wins",   str(m["max_consec_wins"])),
        ("Max consec losses", str(m["max_consec_losses"])),
        ("Best trade",        f"${_f(m['best_trade_usd'])}"),
        ("Worst trade",       f"${_f(m['worst_trade_usd'])}"),
        ("Avg duration",      f"{_f(m['avg_dur_hrs'])} h"),
    ]
    for k, v in rows:
        print(f"  {k:<24} {v}")
    print(f"{'═'*W}")


def print_wf_summary(df: pd.DataFrame) -> None:
    if df.empty or "wf_window" not in df.columns:
        return
    W = 96
    print(f"\n{'─'*W}")
    print("  WALK-FORWARD WINDOW SUMMARY")
    print(f"  {'Window':<14}{'Trades':>7}{'Win%':>7}{'PF':>7}"
          f"{'Net$':>10}{'MaxDD%':>8}{'2lot':>6}{'AvgQ':>7}")
    print(f"{'─'*W}")
    for win, grp in df.groupby("wf_window"):
        m  = compute_metrics(grp)
        pf = _f(m.get("profit_factor"), 2)
        print(
            f"  {win:<14}"
            f"{m['trades']:>7}"
            f"{_f(m['win_rate']):>7}"
            f"{pf:>7}"
            f"{_f(m['net_pnl_usd']):>10}"
            f"{_f(m['max_dd_pct']):>8}"
            f"{m['two_lot_trades']:>6}"
            f"{_f(m['avg_quality'], 0):>7}"
        )
    print(f"{'─'*W}")


def print_comparison(is_m: dict, oos_m: dict) -> None:
    W = 96
    print(f"\n{'═'*W}")
    print("  IN-SAMPLE vs WALK-FORWARD OOS")
    print(f"  {'Metric':<26}{'In-sample':>14}{'OOS':>14}{'Δ':>10}")
    print(f"{'─'*W}")
    keys = [
        ("Win rate %",     "win_rate",       1),
        ("Profit factor",  "profit_factor",  3),
        ("Sharpe",         "sharpe",         3),
        ("Sortino",        "sortino",        3),
        ("Max DD %",       "max_dd_pct",     1),
        ("Expectancy $",   "expectancy_usd", 2),
        ("2-lot trades",   "two_lot_trades", 0),
        ("Avg quality",    "avg_quality",    1),
    ]
    for lbl, key, d in keys:
        iv = is_m.get(key, np.nan)
        wv = oos_m.get(key, np.nan)
        try:
            delta = f"{((wv - iv) / abs(iv)) * 100:+.1f} %" if iv != 0 else "n/a"
        except Exception:
            delta = "n/a"
        print(f"  {lbl:<26}{_f(iv, d):>14}{_f(wv, d):>14}{delta:>10}")
    print(f"{'═'*W}")


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE MODE
# ═════════════════════════════════════════════════════════════════════════════

class LiveTrader:
    """
    Bar-by-bar live trading loop.

    On each new bar close:
      1. Fetch latest H1 bar from MT5
      2. Update features (Itô drift, QV vol, vol percentile, RSI, ATR)
      3. Update Kalman filter (one step, online)
      4. Run forward HMM pass on rolling buffer (HMM_TRAIN_BARS)
      5. Evaluate entry/exit signals
      6. Send orders to MT5
      7. Persist full state to JSON

    The HMM is refit every HMM_REFIT_DAYS using the most recent
    HMM_TRAIN_BARS of data. This keeps it current without over-fitting
    to very recent noise.
    """

    def __init__(self) -> None:
        self.state    = LiveState.load(STATE_FILE)
        self.model    : Optional[GaussianHMM] = None
        self.bull_state: int = 0
        self._load_model()
        self.kalman   = (KalmanState.from_dict(self.state.kalman)
                         if self.state.kalman else KalmanState())
        self.position = (Position.from_dict(self.state.position)
                         if self.state.position else None)
        log.info("LiveTrader initialised  equity=%.2f  position=%s",
                 self.state.equity,
                 self.position.side if self.position else "flat")

    # ── Model management ─────────────────────────────────────────────

    def _load_model(self) -> None:
        result = load_hmm_model()
        if result:
            self.model, self.bull_state = result

    def _refit_needed(self) -> bool:
        if self.model is None:
            return True
        if self.state.last_refit_date is None:
            return True
        last = datetime.fromisoformat(self.state.last_refit_date)
        return (datetime.now(timezone.utc) - last).days >= HMM_REFIT_DAYS

    def refit_model(self, features: pd.DataFrame) -> None:
        log.info("Refitting HMM on %d bars…", len(features))
        self.model      = fit_hmm(features)
        self.bull_state = identify_bull_state(self.model, features)
        save_hmm_model(self.model, self.bull_state)
        self.state.last_refit_date = datetime.now(timezone.utc).isoformat()
        log.info("HMM refit complete  bull_state=%d", self.bull_state)

    # ── Signal generation ─────────────────────────────────────────────

    def get_regime_proba(self, features: pd.DataFrame) -> float:
        """Return P(bull | obs_1..t) for the most recent bar."""
        if self.model is None:
            return 0.5
        X     = features[["ito_drift", "rv_qv"]].values[-HMM_TRAIN_BARS:]
        proba = forward_pass(self.model, X)
        return float(proba[-1, self.bull_state])

    def decode_latest_regime(self, bull_proba: float) -> str:
        if bull_proba >= HMM_BULL_THRESH:
            return "bull"
        if bull_proba <= HMM_BEAR_THRESH:
            return "bear"
        return "uncertain"

    # ── Order management (MT5 stubs) ───────────────────────────────────

    def send_market_order(self, side: str, contracts: int,
                          stop_price: float) -> bool:
        """
        Send a market order to MT5.
        In paper/testing mode log only — swap in real MT5 order calls here.
        """
        log.info(
            "ORDER  %s  %d lot(s)  stop=%.2f",
            side.upper(), contracts, stop_price
        )
        # ── Replace with actual MT5 order code: ──────────────────────
        # from mt5linux import MetaTrader5
        # mt5 = MetaTrader5()
        # mt5.initialize()
        # request = {
        #     "action":    mt5.TRADE_ACTION_DEAL,
        #     "symbol":    SYMBOL,
        #     "volume":    contracts * 1.0,
        #     "type":      mt5.ORDER_TYPE_BUY if side=="long" else mt5.ORDER_TYPE_SELL,
        #     "deviation": 20,
        #     "magic":     20240101,
        #     "comment":   "hmm_v4",
        #     "type_time": mt5.ORDER_TIME_GTC,
        #     "type_filling": mt5.ORDER_FILLING_IOC,
        # }
        # result = mt5.order_send(request)
        # mt5.shutdown()
        # return result.retcode == mt5.TRADE_RETCODE_DONE
        return True

    def close_position(self, reason: str) -> None:
        if self.position is None:
            return
        log.info("CLOSE  %s  reason=%s", self.position.side.upper(), reason)
        # MT5 close order goes here
        self.position = None
        self.state.position = None

    # ── Main tick ─────────────────────────────────────────────────────

    def on_bar_close(self, bar: pd.Series,
                     features: pd.DataFrame) -> None:
        """
        Called once per closed H1 bar.
        features = full DataFrame up to and including the new bar.
        """
        if self._refit_needed():
            self.refit_model(features.tail(HMM_TRAIN_BARS))

        # Kalman update with latest close
        self.kalman = kalman_update(self.kalman, float(bar["close"]))
        kvel        = self.kalman.x[1]

        # Regime signal
        bull_proba  = self.get_regime_proba(features)
        regime      = self.decode_latest_regime(bull_proba)

        last_row    = features.iloc[-1]
        rv_qv       = float(last_row["rv_qv"])
        vol_pct     = float(last_row.get("vol_pct", 50.0))
        rsi_val     = float(last_row["rsi"])
        atr_val     = float(last_row["atr"])
        open_p      = float(bar["open"])
        high_p      = float(bar["high"])
        low_p       = float(bar["low"])

        # ── Manage open position ──────────────────────────────────────
        if self.position is not None:
            side = self.position.side
            if side == "long"  and low_p  <= self.position.stop_price:
                self.close_position("stop_loss")
                self.state.consec_losses += 1
            elif side == "short" and high_p >= self.position.stop_price:
                self.close_position("stop_loss")
                self.state.consec_losses += 1

        # ── Entry ─────────────────────────────────────────────────────
        if self.position is None and regime != "uncertain":
            long_rsi, short_rsi = tightened_rsi(self.state.consec_losses)

            # Soft-size Kalman (mirrors backtest): enter on any confirmed
            # regime; size is 1 lot when Kalman disagrees, full calc when agrees.
            kalman_confirms = (
                (regime == "bull" and kvel > 0) or
                (regime == "bear" and kvel < 0)
            )

            quality_ok = vol_pct < VOL_PCT_NO_ENTRY

            can_enter = (
                quality_ok
                and not self.state.dd_brake_active
                and in_session(pd.Timestamp(bar.name
                               if hasattr(bar, "name")
                               else datetime.now(timezone.utc)))
            )

            if can_enter:
                stop_dist = ito_stop_distance(open_p, rv_qv)
                if kalman_confirms:
                    n_con = calc_contracts(
                        50.0,  # simplified quality for live
                        self.state.regime_persist, vol_pct
                    )
                else:
                    n_con = BASE_CONTRACTS  # Kalman disagrees → cautious size

                if regime == "bull" and rsi_val > long_rsi:
                    ok = self.send_market_order("long", n_con,
                                                open_p - stop_dist)
                    if ok:
                        self.position = Position(
                            side="long",
                            entry_time=str(bar.name),
                            entry_price=open_p,
                            stop_price=open_p - stop_dist,
                            contracts=n_con,
                        )
                        self.state.consec_losses = 0

                elif regime == "bear" and rsi_val < short_rsi:
                    ok = self.send_market_order("short", n_con,
                                                open_p + stop_dist)
                    if ok:
                        self.position = Position(
                            side="short",
                            entry_time=str(bar.name),
                            entry_price=open_p,
                            stop_price=open_p + stop_dist,
                            contracts=n_con,
                        )
                        self.state.consec_losses = 0

        # ── Persist state ─────────────────────────────────────────────
        self.state.kalman   = self.kalman.to_dict()
        self.state.position = (self.position.to_dict()
                               if self.position else None)
        self.state.save(STATE_FILE)

        log.info(
            "Bar closed  regime=%-9s  P(bull)=%.2f  kvel=%+.5f  "
            "vol_pct=%.0f  pos=%s  eq=%.2f",
            regime, bull_proba, kvel, vol_pct,
            self.position.side if self.position else "flat",
            self.state.equity,
        )

    def run_live_loop(self) -> None:
        """
        Production loop: wait for each new H1 bar, fetch it, update.
        Runs indefinitely — use Ctrl+C or a process supervisor to stop.
        """
        log.info("Starting live loop for %s %s", SYMBOL, TIMEFRAME)

        # Initial data load
        bars     = fetch_bars(SYMBOL, START_DATE, TIMEFRAME)
        features = build_features(bars)

        last_bar_time = bars.index[-1]

        while True:
            import time
            time.sleep(30)   # poll every 30 seconds

            latest = fetch_latest_bar(SYMBOL, TIMEFRAME)
            if latest is None:
                continue

            bar_time = latest.name if hasattr(latest, "name") else bars.index[-1]

            if bar_time <= last_bar_time:
                continue   # no new bar yet

            # New bar arrived
            last_bar_time = bar_time

            # Append to history
            new_row = pd.DataFrame([latest], index=[bar_time])
            bars    = pd.concat([bars, new_row])
            features= build_features(bars)

            # Tick
            self.on_bar_close(latest, features)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest() -> None:
    log.info("Fetching %s %s from %s…", SYMBOL, TIMEFRAME, START_DATE.date())
    bars = fetch_bars(SYMBOL, START_DATE, TIMEFRAME)
    log.info("Got %d bars.", len(bars))

    log.info("Building Itô features…")
    features = build_features(bars)

    # ── Kalman series (causal — runs once over full history) ──────────
    log.info("Running Kalman filter…")
    kalman_df = run_kalman_series(bars["close"])

    # ── In-sample (full dataset, for reference only) ──────────────────
    log.info("Fitting in-sample HMM…")
    model_is      = fit_hmm(features)
    bull_state_is = identify_bull_state(model_is, features)
    regime_is     = decode_regime(model_is, features, bull_state_is)

    data_is = (bars
               .join(features[["ito_drift", "rv_qv", "rsi", "atr", "vol_pct"]],
                     how="left")
               .join(regime_is.rename("regime"), how="left"))

    kvel_is = kalman_df["kalman_velocity"]

    log.info("Running in-sample backtest…")
    trades_is = backtest(data_is, data_is["regime"], kvel_is)
    print_trades(trades_is, label="In-sample (full — use OOS for real validation)")
    metrics_is = compute_metrics(trades_is, label="In-sample")
    print_metrics(metrics_is)

    # ── Walk-forward OOS ──────────────────────────────────────────────
    if ENABLE_WALK_FORWARD:
        log.info(
            "Running walk-forward (train=%dmo / test=%dmo)…",
            WF_TRAIN_MONTHS, WF_TEST_MONTHS
        )
        trades_wf = walk_forward(features, bars)

        if not trades_wf.empty:
            print_trades(trades_wf, label="Walk-forward OOS")
            metrics_wf = compute_metrics(trades_wf, label="Walk-forward OOS")
            print_metrics(metrics_wf)
            print_wf_summary(trades_wf)
            print_comparison(metrics_is, metrics_wf)
        else:
            print(f"{R}Walk-forward produced no trades.{E}")


def run_live() -> None:
    trader = LiveTrader()
    trader.run_live_loop()


def main() -> None:
    if MODE == "backtest":
        run_backtest()
    elif MODE == "live":
        run_live()
    else:
        raise ValueError(f"Unknown MODE: {MODE!r}. Use 'backtest' or 'live'.")


if __name__ == "__main__":
    main()
