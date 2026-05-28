"""
HMM Regime Strategy v3 — @MNQ H1
Changes over v2:
  1. Regime Quality Score  — rolling metric (0–100) combining HMM flip rate,
     recent win rate, and ATR percentile. Replaces blunt loss counters.
  2. Adaptive RSI gate     — tightens after N consecutive losses, resets on
     first win. Keeps you in play without killing profitable regimes.
  3. 2-lot scale-up gate   — doubles position when regime has persisted long
     enough, ATR is calm (below rolling median), and quality score is high.
     Never scales into chop or high-volatility stops.
  4. Drawdown brake        — hard stop on new entries if equity drops more
     than DD_BRAKE_PCT from its recent peak. Resets when equity recovers
     DD_BRAKE_RECOVER fraction of the drawdown.
  5. Kelly scaffold kept   — off by default; quality score gates sizing first.
"""

from mt5linux import MetaTrader5
from hmmlearn.hmm import GaussianHMM
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════ CONFIG ═══════════════════════ #

SYMBOL            = "@MNQ"
START_DATE        = datetime(2019, 5, 3, tzinfo=timezone.utc)
MODE              = "H1"

# ── HMM ──────────────────────────────────────────────
N_COMPONENTS      = 2
ROLL_VOL_PERIOD   = 10

# ── RSI ──────────────────────────────────────────────
RSI_PERIOD             = 7
RSI_LONG_THRESHOLD     = 45       # base threshold for longs
RSI_SHORT_THRESHOLD    = 55       # base threshold for shorts
RSI_TIGHTEN_STEP       = 3        # tighten by this many points per streak tier
# After LOSS_STREAK_TIER_SIZE consecutive losses the RSI gate tightens by one
# RSI_TIGHTEN_STEP.  Resets to base on first winning trade.
# Example: base long = 45, after 4 losses → 48, after 8 → 51
LOSS_STREAK_TIER_SIZE  = 2        # losses per tighten tier
MAX_RSI_TIGHTEN        = 12       # never tighten more than this total (pts)

# ── Stop loss ─────────────────────────────────────────
ATR_PERIOD        = 14
ATR_MULTIPLIER    = 1.5
ATR_STOP_MIN_PTS  = 60
ATR_STOP_MAX_PTS  = 180

# ── Regime Quality Score ──────────────────────────────
# Score = 0 (choppy/losing) to 100 (clean trend/winning).
# Used both as a trade filter and as a gating condition for 2-lot scaling.
QUALITY_LOOKBACK       = 20      # bars to measure flip rate and win rate
QUALITY_WIN_WEIGHT     = 0.4     # weight of recent win rate component
QUALITY_FLIP_WEIGHT    = 0.4     # weight of regime stability component
QUALITY_ATR_WEIGHT     = 0.2     # weight of ATR calm component
MIN_QUALITY_TO_TRADE   = 30      # below this: skip new entries entirely
                                  # (still manages open positions)

# ── 2-lot scale-up gate ───────────────────────────────
# All three conditions must be true to trade 2 lots:
ENABLE_SCALE_UP        = True
PERSIST_FOR_SCALE      = 6       # regime must have persisted >= this many bars
ATR_MEDIAN_LOOKBACK    = 50      # rolling window for ATR median calculation
SCALE_QUALITY_THRESHOLD= 65      # quality score must be >= this
MAX_CONTRACTS          = 2       # hard ceiling on contracts

# ── Drawdown brake ────────────────────────────────────
DD_BRAKE_PCT           = 0.12    # pause new entries if DD from peak > 12 %
DD_BRAKE_RECOVER       = 0.50    # resume when 50 % of the drawdown recovered

# ── Partial profit (off by default) ──────────────────
ENABLE_PARTIAL_PROFIT  = True
PARTIAL_TARGET_ATR     = 2.0
PARTIAL_CLOSE_FRAC     = 0.50
TRAIL_ATR              = 1.0

# ── Session filter ────────────────────────────────────
ENABLE_SESSION_FILTER  = True
SESSION_WINDOWS = [
    (7,  10),   # London open
    (13, 20),   # New York session
]

# ── Hold / confirmation ──────────────────────────────
MIN_HOLD_BARS     = 3
CONFIRM_BARS      = 1

# ── Capital / costs ──────────────────────────────────
BASE_CONTRACTS         = 1
USD_PER_POINT          = 2.0
STARTING_CAPITAL_USD   = 5000
COMMISSION_PER_SIDE    = 0.50
RISK_FREE_RATE         = 0.0

# ── Walk-forward ─────────────────────────────────────
ENABLE_WALK_FORWARD    = True
WF_TRAIN_MONTHS        = 12
WF_TEST_MONTHS         = 3

# ══════════════════════════════════════════════════════ #


# ─────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────

def get_timeframe(mt5, mode: str):
    if mode == "D1":
        return mt5.TIMEFRAME_D1
    return mt5.TIMEFRAME_H1


def get_annualization_factor(mode: str) -> int:
    return 252 * 24 if mode == "H1" else 252


def fetch_bars(symbol: str, start: datetime, mode: str) -> pd.DataFrame:
    mt5 = MetaTrader5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    tf  = get_timeframe(mt5, mode)
    end = datetime.now(timezone.utc)
    rates = mt5.copy_rates_range(symbol, tf, start, end)
    if rates is None or len(rates) == 0:
        mt5.shutdown()
        raise RuntimeError(f"No rates for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)
    mt5.shutdown()
    return df


# ─────────────────────────────────────────────────────
#  FEATURES
# ─────────────────────────────────────────────────────

def calc_rsi(prices: pd.Series, period: int) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.clip(lower=0.0)
    loss     = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_gain = avg_gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = avg_loss.ewm(alpha=1 / period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["log_ret"] = np.log(f["close"]).diff()
    f["rv_10"]   = f["log_ret"].rolling(ROLL_VOL_PERIOD).std()
    f["rsi"]     = calc_rsi(f["close"], RSI_PERIOD)
    f["atr"]     = calc_atr(f, ATR_PERIOD)
    return f.dropna(subset=["log_ret", "rv_10", "rsi", "atr"])


# ─────────────────────────────────────────────────────
#  HMM
# ─────────────────────────────────────────────────────

def fit_hmm(features: pd.DataFrame) -> pd.Series:
    X = features[["log_ret", "rv_10"]].values
    model = GaussianHMM(
        n_components=N_COMPONENTS,
        covariance_type="full",
        n_iter=300,
        random_state=42,
    )
    model.fit(X)
    states = model.predict(X)
    feat = features.copy()
    feat["state"] = states
    bull_state = feat.groupby("state")["log_ret"].mean().idxmax()
    feat["regime"] = np.where(feat["state"] == bull_state, "bull", "bear")
    return feat["regime"]


# ─────────────────────────────────────────────────────
#  REGIME QUALITY SCORE
# ─────────────────────────────────────────────────────

def compute_quality_score(
    regime_series: pd.Series,
    atr_series: pd.Series,
    trade_results: list,   # list of recent pnl_usd values (most recent last)
    idx: int,
    lookback: int = QUALITY_LOOKBACK,
    atr_median_window: int = ATR_MEDIAN_LOOKBACK,
) -> float:
    """
    Returns a score 0–100.
    Three components weighted by config:
      flip_score : how stable the regime has been (fewer flips = higher score)
      win_score  : recent trade win rate
      atr_score  : how calm the ATR is vs its rolling median (calmer = higher)
    """
    # 1. Regime flip rate over lookback bars
    start_i = max(0, idx - lookback)
    recent_regimes = regime_series.iloc[start_i: idx + 1]
    if len(recent_regimes) < 2:
        flip_score = 50.0
    else:
        flips = recent_regimes.ne(recent_regimes.shift(1)).sum() - 1
        flip_rate = flips / max(1, len(recent_regimes) - 1)
        flip_score = max(0.0, 100.0 * (1.0 - flip_rate * 2))

    # 2. Recent trade win rate (last lookback trades)
    recent_trades = trade_results[-lookback:] if trade_results else []
    if len(recent_trades) < 3:
        win_score = 50.0
    else:
        wins = sum(1 for p in recent_trades if p > 0)
        win_score = 100.0 * wins / len(recent_trades)

    # 3. ATR calm score: current ATR vs rolling median
    start_atr = max(0, idx - atr_median_window)
    atr_window = atr_series.iloc[start_atr: idx + 1]
    if len(atr_window) < 5:
        atr_score = 50.0
    else:
        current_atr = atr_series.iloc[idx]
        median_atr  = atr_window.median()
        if median_atr == 0:
            atr_score = 50.0
        else:
            ratio = current_atr / median_atr
            # ratio < 1 means calm (good), > 1 means volatile (bad)
            atr_score = max(0.0, min(100.0, 100.0 * (2.0 - ratio)))

    score = (
        QUALITY_FLIP_WEIGHT * flip_score +
        QUALITY_WIN_WEIGHT  * win_score  +
        QUALITY_ATR_WEIGHT  * atr_score
    )
    return float(np.clip(score, 0.0, 100.0))


# ─────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────

def in_session(ts: pd.Timestamp) -> bool:
    if not ENABLE_SESSION_FILTER:
        return True
    h = ts.hour + ts.minute / 60.0
    for start_h, end_h in SESSION_WINDOWS:
        if start_h <= h <= end_h:
            return True
    return False


def tightened_thresholds(consec_losses: int):
    """
    Returns (long_threshold, short_threshold) after applying streak tightening.
    Tightens by RSI_TIGHTEN_STEP per LOSS_STREAK_TIER_SIZE losses, capped.
    Long threshold goes UP (harder to enter).
    Short threshold goes DOWN (harder to enter).
    """
    tiers   = consec_losses // LOSS_STREAK_TIER_SIZE
    tighten = min(tiers * RSI_TIGHTEN_STEP, MAX_RSI_TIGHTEN)
    return (RSI_LONG_THRESHOLD + tighten, RSI_SHORT_THRESHOLD - tighten)


def calc_contracts(
    equity: float,
    stop_pts: float,
    quality_score: float,
    regime_persist_bars: int,
    atr_series_window: pd.Series,
) -> int:
    """
    Returns 1 or 2 contracts.
    2 contracts only when all scale-up conditions are met.
    """
    if not ENABLE_SCALE_UP:
        return BASE_CONTRACTS

    # Condition 1: regime has been stable long enough
    if regime_persist_bars < PERSIST_FOR_SCALE:
        return BASE_CONTRACTS

    # Condition 2: quality score is high
    if quality_score < SCALE_QUALITY_THRESHOLD:
        return BASE_CONTRACTS

    # Condition 3: current ATR is below rolling median (calm stops)
    if len(atr_series_window) < 5:
        return BASE_CONTRACTS
    current_atr = atr_series_window.iloc[-1]
    median_atr  = atr_series_window.median()
    if current_atr > median_atr:
        return BASE_CONTRACTS

    return min(MAX_CONTRACTS, BASE_CONTRACTS + 1)


def close_trade(pos: dict, exit_time, exit_price: float, reason: str,
                partial_pnl_pts: float = 0.0, partial_contracts: int = 0) -> dict:
    pos = pos.copy()
    pos["exit_time"]         = exit_time
    pos["exit_price"]        = float(exit_price)
    pos["exit_reason"]       = reason
    pos["partial_pnl_pts"]   = partial_pnl_pts
    pos["partial_contracts"] = partial_contracts
    return pos


# ─────────────────────────────────────────────────────
#  BACKTEST CORE
# ─────────────────────────────────────────────────────

def backtest(data: pd.DataFrame, regime: pd.Series,
             starting_capital: float = STARTING_CAPITAL_USD) -> pd.DataFrame:
    d = data.copy()
    d["regime"] = regime.reindex(d.index).ffill()
    d = d.dropna(subset=["regime", "rsi", "atr"])

    signals      = d["regime"]
    atr_series   = d["atr"]
    regime_shift = signals.ne(signals.shift(1))

    trades               = []
    position             = None
    bars_since_exit      = 999
    bars_since_sig       = 999
    pending_regime       = None
    equity               = starting_capital
    peak_equity          = starting_capital
    dd_brake_active      = False
    dd_brake_trough      = starting_capital

    # Streak tracking
    consec_losses        = 0
    recent_pnl_list      = []    # running list for quality score

    # Regime persistence counter (bars since last flip)
    regime_persist_bars  = 0

    rows     = list(d.iterrows())
    n_rows   = len(rows)

    for idx, (t, row) in enumerate(rows):
        open_p  = float(row["open"])
        high_p  = float(row["high"])
        low_p   = float(row["low"])
        atr_val = float(row["atr"])
        rsi_val = float(row["rsi"])
        reg     = signals.loc[t]

        # ── Update regime persistence ──────────────────────────
        if idx == 0:
            regime_persist_bars = 0
        elif signals.iloc[idx] == signals.iloc[idx - 1]:
            regime_persist_bars += 1
        else:
            regime_persist_bars = 0

        # ── Quality score ───────────────────────────────────────
        atr_window_start = max(0, idx - ATR_MEDIAN_LOOKBACK)
        atr_window       = atr_series.iloc[atr_window_start: idx + 1]
        quality          = compute_quality_score(
            signals, atr_series, recent_pnl_list, idx
        )

        # ── Drawdown brake ──────────────────────────────────────
        peak_equity = max(peak_equity, equity)
        current_dd  = (equity - peak_equity) / peak_equity  # negative

        if not dd_brake_active and current_dd < -DD_BRAKE_PCT:
            dd_brake_active = True
            dd_brake_trough = equity

        if dd_brake_active:
            # Recover when we claw back DD_BRAKE_RECOVER of the drawdown
            dd_depth = peak_equity - dd_brake_trough
            recovered = equity - dd_brake_trough
            if dd_depth > 0 and (recovered / dd_depth) >= DD_BRAKE_RECOVER:
                dd_brake_active = False

        # ── Manage existing position ────────────────────────────
        if position is not None:
            side       = position["side"]
            stop_price = position["stop_price"]
            n_con      = position["contracts"]
            trail_stop = position.get("trail_stop")

            # Trailing stop update after partial close
            if trail_stop is not None:
                if side == "long":
                    new_trail = high_p - atr_val * TRAIL_ATR
                    position["trail_stop"] = max(trail_stop, new_trail)
                    stop_price = position["trail_stop"]
                elif side == "short":
                    new_trail = low_p + atr_val * TRAIL_ATR
                    position["trail_stop"] = min(trail_stop, new_trail)
                    stop_price = position["trail_stop"]
                position["stop_price"] = stop_price

            # Partial profit
            if ENABLE_PARTIAL_PROFIT and not position.get("partial_done", False):
                pt_dist = atr_val * PARTIAL_TARGET_ATR
                if side == "long" and high_p >= position["entry_price"] + pt_dist:
                    partial_px  = position["entry_price"] + pt_dist
                    partial_con = max(1, int(n_con * PARTIAL_CLOSE_FRAC))
                    partial_pts = (partial_px - position["entry_price"]) * partial_con
                    equity     += partial_pts * USD_PER_POINT - COMMISSION_PER_SIDE * partial_con
                    position.update({
                        "partial_done": True,
                        "partial_pnl_pts": partial_pts,
                        "partial_contracts": partial_con,
                        "contracts": n_con - partial_con,
                        "trail_stop": high_p - atr_val * TRAIL_ATR,
                        "stop_price": high_p - atr_val * TRAIL_ATR,
                    })
                elif side == "short" and low_p <= position["entry_price"] - pt_dist:
                    partial_px  = position["entry_price"] - pt_dist
                    partial_con = max(1, int(n_con * PARTIAL_CLOSE_FRAC))
                    partial_pts = (position["entry_price"] - partial_px) * partial_con
                    equity     += partial_pts * USD_PER_POINT - COMMISSION_PER_SIDE * partial_con
                    position.update({
                        "partial_done": True,
                        "partial_pnl_pts": partial_pts,
                        "partial_contracts": partial_con,
                        "contracts": n_con - partial_con,
                        "trail_stop": low_p + atr_val * TRAIL_ATR,
                        "stop_price": low_p + atr_val * TRAIL_ATR,
                    })

            # Stop loss check
            stopped = False
            if side == "long"  and low_p  <= position["stop_price"]:
                stopped = True
            elif side == "short" and high_p >= position["stop_price"]:
                stopped = True

            if stopped:
                partial_pts = position.get("partial_pnl_pts", 0.0)
                partial_con = position.get("partial_contracts", 0)
                t_obj       = close_trade(position, t, position["stop_price"],
                                          "stop_loss", partial_pts, partial_con)
                commission  = COMMISSION_PER_SIDE * 2 * position["contracts"]
                direction   = 1 if side == "long" else -1
                remainder_pts = (position["stop_price"] - position["entry_price"]) * direction
                trade_pnl_usd = remainder_pts * USD_PER_POINT * position["contracts"] - commission
                equity       += trade_pnl_usd
                recent_pnl_list.append(trade_pnl_usd + partial_pts * USD_PER_POINT)

                # streak tracking
                if trade_pnl_usd + partial_pts * USD_PER_POINT < 0:
                    consec_losses += 1
                else:
                    consec_losses = 0

                trades.append({**t_obj, "commission": commission,
                                "equity_before": equity - trade_pnl_usd,
                                "quality_score": quality,
                                "contracts_used": position["contracts"] + partial_con})
                position        = None
                bars_since_exit = 0
                pending_regime  = None
                bars_since_sig  = 999
                continue

        # ── Increment counters ──────────────────────────────────
        bars_since_exit += 1
        if pending_regime is not None:
            bars_since_sig += 1

        # ── Regime shift: close opposite, start confirmation ────
        if regime_shift.loc[t]:
            pending_regime = reg
            bars_since_sig = 0

            if position is not None:
                opp = ("short" if reg == "bull" else "long")
                if position["side"] == opp:
                    partial_pts = position.get("partial_pnl_pts", 0.0)
                    partial_con = position.get("partial_contracts", 0)
                    t_obj       = close_trade(position, t, open_p, "regime_flip",
                                              partial_pts, partial_con)
                    commission  = COMMISSION_PER_SIDE * 2 * position["contracts"]
                    direction   = 1 if position["side"] == "long" else -1
                    remainder_pts = (open_p - position["entry_price"]) * direction
                    trade_pnl_usd = remainder_pts * USD_PER_POINT * position["contracts"] - commission
                    equity       += trade_pnl_usd
                    recent_pnl_list.append(trade_pnl_usd + partial_pts * USD_PER_POINT)

                    if trade_pnl_usd + partial_pts * USD_PER_POINT < 0:
                        consec_losses += 1
                    else:
                        consec_losses = 0

                    trades.append({**t_obj, "commission": commission,
                                    "equity_before": equity - trade_pnl_usd,
                                    "quality_score": quality,
                                    "contracts_used": position["contracts"] + partial_con})
                    position        = None
                    bars_since_exit = 0

        # ── Entry logic ─────────────────────────────────────────
        long_rsi_thresh, short_rsi_thresh = tightened_thresholds(consec_losses)

        can_enter = (
            position is None
            and pending_regime is not None
            and bars_since_sig  >= CONFIRM_BARS
            and bars_since_exit >= MIN_HOLD_BARS
            and in_session(t)
            and quality >= MIN_QUALITY_TO_TRADE
            and not dd_brake_active
        )

        if can_enter:
            stop_dist = float(np.clip(
                atr_val * ATR_MULTIPLIER,
                ATR_STOP_MIN_PTS,
                ATR_STOP_MAX_PTS,
            ))
            n_con = calc_contracts(
                equity, stop_dist, quality,
                regime_persist_bars, atr_window
            )

            if pending_regime == "bull" and rsi_val > long_rsi_thresh:
                position = {
                    "side":          "long",
                    "entry_time":    t,
                    "entry_price":   open_p,
                    "stop_price":    open_p - stop_dist,
                    "contracts":     n_con,
                    "partial_done":  False,
                    "trail_stop":    None,
                }
                pending_regime = None

            elif pending_regime == "bear" and rsi_val < short_rsi_thresh:
                position = {
                    "side":          "short",
                    "entry_time":    t,
                    "entry_price":   open_p,
                    "stop_price":    open_p + stop_dist,
                    "contracts":     n_con,
                    "partial_done":  False,
                    "trail_stop":    None,
                }
                pending_regime = None

    # ── Close any open trade at end ─────────────────────────────
    if position is not None:
        last_t     = d.index[-1]
        last_close = float(d["close"].iloc[-1])
        partial_pts = position.get("partial_pnl_pts", 0.0)
        partial_con = position.get("partial_contracts", 0)
        t_obj = close_trade(position, last_t, last_close, "final_close",
                            partial_pts, partial_con)
        commission = COMMISSION_PER_SIDE * 2 * position["contracts"]
        trades.append({**t_obj, "commission": commission,
                        "equity_before": equity,
                        "quality_score": quality,
                        "contracts_used": position["contracts"] + partial_con})

    if not trades:
        return pd.DataFrame()

    df_t = pd.DataFrame(trades)

    direction = df_t["side"].map({"long": 1, "short": -1})
    df_t["remainder_pnl_pts"] = (
        (df_t["exit_price"] - df_t["entry_price"]) * direction * df_t["contracts"]
    )
    df_t["partial_pnl_pts"] = df_t.get("partial_pnl_pts", 0.0)
    df_t["pnl_points"] = df_t["remainder_pnl_pts"] + df_t["partial_pnl_pts"]
    df_t["pnl_usd"] = (
        df_t["remainder_pnl_pts"] * USD_PER_POINT
        + df_t.get("partial_pnl_pts", 0.0) * USD_PER_POINT
        - df_t["commission"]
    )
    df_t["cum_pnl_usd"]    = df_t["pnl_usd"].cumsum()
    df_t["equity_usd"]     = starting_capital + df_t["cum_pnl_usd"]
    df_t["trade_return"]   = df_t["pnl_usd"] / starting_capital
    df_t["duration_hours"] = (
        (df_t["exit_time"] - df_t["entry_time"]).dt.total_seconds() / 3600.0
    )
    df_t["2lot_trades"]    = (df_t.get("contracts_used", BASE_CONTRACTS) > BASE_CONTRACTS).astype(int)
    return df_t


# ─────────────────────────────────────────────────────
#  WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────────────

def walk_forward(features: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_WALK_FORWARD:
        return pd.DataFrame()

    all_oos_trades = []
    dates          = features.index
    window_start   = dates[0]

    while True:
        train_end = window_start + pd.DateOffset(months=WF_TRAIN_MONTHS)
        test_end  = train_end   + pd.DateOffset(months=WF_TEST_MONTHS)

        train_mask = (dates >= window_start) & (dates < train_end)
        test_mask  = (dates >= train_end)    & (dates < test_end)

        if train_mask.sum() < 200 or test_mask.sum() < 5:
            break

        train_feat = features[train_mask]
        test_feat  = features[test_mask]

        X_train = train_feat[["log_ret", "rv_10"]].values
        model = GaussianHMM(
            n_components=N_COMPONENTS,
            covariance_type="full",
            n_iter=300,
            random_state=42,
        )
        model.fit(X_train)

        X_test = test_feat[["log_ret", "rv_10"]].values
        states = model.predict(X_test)

        train_feat2 = train_feat.copy()
        train_feat2["state"] = model.predict(X_train)
        bull_state = train_feat2.groupby("state")["log_ret"].mean().idxmax()
        regime_oos = pd.Series(
            np.where(states == bull_state, "bull", "bear"),
            index=test_feat.index,
        )

        test_bars = bars.reindex(test_feat.index)
        test_data = test_bars.join(
            test_feat[["log_ret", "rv_10", "rsi", "atr"]], how="left"
        ).join(regime_oos.rename("regime"), how="left")

        oos_trades = backtest(test_data, test_data["regime"],
                              starting_capital=STARTING_CAPITAL_USD)
        if not oos_trades.empty:
            oos_trades["wf_window"] = str(train_end.date())
            all_oos_trades.append(oos_trades)

        window_start = window_start + pd.DateOffset(months=WF_TEST_MONTHS)
        if test_end > dates[-1]:
            break

    if not all_oos_trades:
        return pd.DataFrame()

    combined = pd.concat(all_oos_trades, ignore_index=True)
    combined.sort_values("entry_time", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined["cum_pnl_usd"] = combined["pnl_usd"].cumsum()
    combined["equity_usd"]  = STARTING_CAPITAL_USD + combined["cum_pnl_usd"]
    return combined


# ─────────────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────────────

def max_consecutive(cond: pd.Series) -> int:
    best = cur = 0
    for v in cond:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def compute_metrics(trades_df: pd.DataFrame, label: str = "") -> dict:
    if trades_df.empty:
        return {}

    ann    = get_annualization_factor(MODE)
    wins   = trades_df[trades_df["pnl_usd"] > 0]
    losses = trades_df[trades_df["pnl_usd"] < 0]
    longs  = trades_df[trades_df["side"] == "long"]
    shorts = trades_df[trades_df["side"] == "short"]
    two_lot = trades_df.get("2lot_trades", pd.Series(0, index=trades_df.index))

    gross_profit = wins["pnl_usd"].sum()
    gross_loss   = losses["pnl_usd"].sum()
    net_profit   = trades_df["pnl_usd"].sum()
    pf = abs(gross_profit / gross_loss) if gross_loss != 0 else np.nan

    equity      = trades_df["equity_usd"]
    running_max = equity.cummax()
    dd_usd      = equity - running_max
    dd_pct      = dd_usd / running_max
    mdd_usd     = dd_usd.min()
    mdd_pct     = dd_pct.min() * 100

    returns   = trades_df["trade_return"]
    ret_mean  = returns.mean()
    ret_std   = returns.std(ddof=1)
    sharpe    = np.sqrt(ann) * ret_mean / ret_std if ret_std > 0 else np.nan
    downside  = returns[returns < 0].std(ddof=1)
    sortino   = np.sqrt(ann) * ret_mean / downside if (downside and downside > 0) else np.nan
    vol_pct   = ret_std * np.sqrt(ann) * 100 if ret_std > 0 else np.nan

    avg_quality = trades_df["quality_score"].mean() if "quality_score" in trades_df else np.nan
    n_2lot      = two_lot.sum() if not two_lot.empty else 0

    return {
        "label":               label,
        "trades":              len(trades_df),
        "long_trades":         len(longs),
        "short_trades":        len(shorts),
        "two_lot_trades":      int(n_2lot),
        "win_rate_pct":        (trades_df["pnl_usd"] > 0).mean() * 100,
        "long_win_rate_pct":   (longs["pnl_usd"] > 0).mean() * 100  if len(longs)  > 0 else np.nan,
        "short_win_rate_pct":  (shorts["pnl_usd"] > 0).mean() * 100 if len(shorts) > 0 else np.nan,
        "net_profit_usd":      net_profit,
        "net_profit_pts":      trades_df["pnl_points"].sum(),
        "gross_profit_usd":    gross_profit,
        "gross_loss_usd":      gross_loss,
        "profit_factor":       pf,
        "total_return_pct":    (equity.iloc[-1] / STARTING_CAPITAL_USD - 1) * 100,
        "final_equity_usd":    equity.iloc[-1],
        "max_drawdown_usd":    mdd_usd,
        "max_drawdown_pct":    mdd_pct,
        "recovery_factor":     abs(net_profit / mdd_usd) if mdd_usd != 0 else np.nan,
        "sharpe_ratio":        sharpe,
        "sortino_ratio":       sortino,
        "volatility_pct":      vol_pct,
        "avg_win_usd":         wins["pnl_usd"].mean()    if len(wins)   > 0 else 0.0,
        "avg_loss_usd":        losses["pnl_usd"].mean()  if len(losses) > 0 else 0.0,
        "avg_win_pts":         wins["pnl_points"].mean() if len(wins)   > 0 else 0.0,
        "avg_loss_pts":        losses["pnl_points"].mean() if len(losses) > 0 else 0.0,
        "expectancy_usd":      trades_df["pnl_usd"].mean(),
        "stopped_out":         (trades_df["exit_reason"] == "stop_loss").sum(),
        "regime_flip_exits":   (trades_df["exit_reason"] == "regime_flip").sum(),
        "max_consec_wins":     max_consecutive(trades_df["pnl_usd"] > 0),
        "max_consec_losses":   max_consecutive(trades_df["pnl_usd"] < 0),
        "best_trade_usd":      trades_df["pnl_usd"].max(),
        "worst_trade_usd":     trades_df["pnl_usd"].min(),
        "avg_duration_hrs":    trades_df["duration_hours"].mean(),
        "total_commission":    trades_df["commission"].sum() if "commission" in trades_df else 0,
        "avg_quality_score":   avg_quality,
    }


# ─────────────────────────────────────────────────────
#  PRINTING
# ─────────────────────────────────────────────────────

G = "\033[92m"
R = "\033[91m"
Y = "\033[93m"
B = "\033[94m"
E = "\033[0m"
DIM = "\033[2m"


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{d}f}"


def color_val(v: float) -> str:
    return G if v >= 0 else R


def print_trades(trades_df: pd.DataFrame, label: str = "In-sample"):
    if trades_df.empty:
        print("No trades.")
        return
    print(f"\n{'═'*90}")
    print(f"  TRADES — {label}  |  {SYMBOL}  |  {MODE}")
    print(f"  Capital: ${STARTING_CAPITAL_USD:,.0f}   "
          f"Stop: ATR×{ATR_MULTIPLIER} ({ATR_STOP_MIN_PTS}–{ATR_STOP_MAX_PTS} pts)   "
          f"Scale-up: {'ON' if ENABLE_SCALE_UP else 'OFF'}   "
          f"DD brake: {DD_BRAKE_PCT*100:.0f}%")
    print(f"{'─'*90}")

    for _, tr in trades_df.iterrows():
        c        = color_val(tr["pnl_usd"])
        wf       = f"  [{tr['wf_window']}]" if "wf_window" in tr else ""
        n_con    = int(tr.get("contracts_used", BASE_CONTRACTS))
        qs       = tr.get("quality_score", np.nan)
        lots_tag = f" ×{n_con}" if n_con > 1 else "   "
        qs_tag   = f" Q={qs:.0f}" if not np.isnan(qs) else ""
        print(
            f"{c}"
            f"{tr['side'].upper():5}"
            f"{lots_tag} "
            f"{tr['entry_time']} → {tr['exit_time']} "
            f"entry={tr['entry_price']:.2f} "
            f"exit={tr['exit_price']:.2f} "
            f"reason={tr['exit_reason']:12} "
            f"pnl={tr['pnl_points']:+.1f}pts / ${tr['pnl_usd']:+.2f} "
            f"eq=${tr['equity_usd']:,.0f}"
            f"{qs_tag}{wf}"
            f"{E}"
        )


def print_metrics(m: dict):
    if not m:
        print("No metrics.")
        return

    label = m.get("label", "")
    hdr   = f"  METRICS — {label}" if label else "  METRICS"

    print(f"\n{'═'*90}")
    print(hdr)
    print(f"{'─'*90}")

    rows = [
        ("Capital",             f"${STARTING_CAPITAL_USD:,.0f} → ${m['final_equity_usd']:,.2f}"),
        ("Total return",        f"{fmt(m['total_return_pct'])} %"),
        ("Net profit",          f"${fmt(m['net_profit_usd'])}  ({fmt(m['net_profit_pts'])} pts)"),
        ("Gross profit",        f"${fmt(m['gross_profit_usd'])}"),
        ("Gross loss",          f"${fmt(m['gross_loss_usd'])}"),
        ("Total commission",    f"${fmt(m['total_commission'])}"),
        ("Profit factor",       fmt(m["profit_factor"], 3)),
        ("Sharpe ratio",        fmt(m["sharpe_ratio"], 3)),
        ("Sortino ratio",       fmt(m["sortino_ratio"], 3)),
        ("Ann. volatility",     f"{fmt(m['volatility_pct'])} %"),
        ("Max drawdown",        f"${fmt(m['max_drawdown_usd'])}  ({fmt(m['max_drawdown_pct'])} %)"),
        ("Recovery factor",     fmt(m["recovery_factor"], 3)),
        ("─── Trades",          f"{m['trades']}  (L:{m['long_trades']}  S:{m['short_trades']})"),
        ("2-lot trades",        f"{m['two_lot_trades']}  ({100*m['two_lot_trades']/max(1,m['trades']):.1f}% of trades)"),
        ("Avg quality score",   fmt(m["avg_quality_score"], 1)),
        ("Win rate",            f"{fmt(m['win_rate_pct'])} %  (L:{fmt(m['long_win_rate_pct'])} %  S:{fmt(m['short_win_rate_pct'])} %)"),
        ("Avg win",             f"${fmt(m['avg_win_usd'])}  ({fmt(m['avg_win_pts'])} pts)"),
        ("Avg loss",            f"${fmt(m['avg_loss_usd'])}  ({fmt(m['avg_loss_pts'])} pts)"),
        ("Expectancy",          f"${fmt(m['expectancy_usd'])} / trade"),
        ("Stopped out",         f"{m['stopped_out']}"),
        ("Regime-flip exits",   f"{m['regime_flip_exits']}"),
        ("Max consec wins",     f"{m['max_consec_wins']}"),
        ("Max consec losses",   f"{m['max_consec_losses']}"),
        ("Best trade",          f"${fmt(m['best_trade_usd'])}"),
        ("Worst trade",         f"${fmt(m['worst_trade_usd'])}"),
        ("Avg duration",        f"{fmt(m['avg_duration_hrs'])} h"),
    ]

    for k, v in rows:
        print(f"  {k:<24} {v}")
    print(f"{'═'*90}")


def print_wf_summary(oos_df: pd.DataFrame):
    if oos_df.empty or "wf_window" not in oos_df.columns:
        return
    print(f"\n{'─'*90}")
    print("  WALK-FORWARD WINDOW SUMMARY")
    print(f"  {'Window':<14} {'Trades':>7} {'Win%':>7} {'PF':>7} {'Net$':>10} {'MaxDD%':>8} {'2lot':>6} {'AvgQ':>6}")
    print(f"{'─'*90}")
    for win, grp in oos_df.groupby("wf_window"):
        w_m    = compute_metrics(grp)
        pf     = fmt(w_m.get("profit_factor", np.nan), 2)
        n2     = w_m.get("two_lot_trades", 0)
        avgq   = fmt(w_m.get("avg_quality_score", np.nan), 0)
        print(
            f"  {win:<14} "
            f"{w_m['trades']:>7} "
            f"{fmt(w_m['win_rate_pct']):>7} "
            f"{pf:>7} "
            f"{fmt(w_m['net_profit_usd']):>10} "
            f"{fmt(w_m['max_drawdown_pct']):>8} "
            f"{n2:>6} "
            f"{avgq:>6}"
        )
    print(f"{'─'*90}")


# ─────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────

def main():
    print(f"\n{B}Fetching {MODE} bars for {SYMBOL} from {START_DATE.date()} …{E}")
    bars = fetch_bars(SYMBOL, START_DATE, MODE)
    print(f"  Got {len(bars):,} bars.")

    print(f"{B}Building features …{E}")
    features = build_features(bars)

    print(f"{B}Fitting HMM (in-sample) …{E}")
    regime_is = fit_hmm(features)

    data_is = bars.join(features[["log_ret", "rv_10", "rsi", "atr"]], how="left")
    data_is = data_is.join(regime_is.rename("regime"), how="left")

    print(f"{B}Running in-sample backtest …{E}")
    trades_is = backtest(data_is, data_is["regime"])
    print_trades(trades_is, label="In-sample (full dataset)")
    metrics_is = compute_metrics(trades_is, label="In-sample")
    print_metrics(metrics_is)

    if ENABLE_WALK_FORWARD:
        print(f"\n{Y}Running walk-forward validation "
              f"(train={WF_TRAIN_MONTHS}mo / test={WF_TEST_MONTHS}mo) …{E}")
        trades_wf = walk_forward(features, bars)

        if not trades_wf.empty:
            print_trades(trades_wf, label="Walk-forward OOS")
            metrics_wf = compute_metrics(trades_wf, label="Walk-forward OOS")
            print_metrics(metrics_wf)
            print_wf_summary(trades_wf)

            print(f"\n{'═'*90}")
            print("  IN-SAMPLE vs WALK-FORWARD COMPARISON")
            print(f"  {'Metric':<24} {'In-sample':>14} {'OOS':>14} {'Decay':>10}")
            print(f"{'─'*90}")
            compare_keys = [
                ("Win rate %",     "win_rate_pct",     1),
                ("Profit factor",  "profit_factor",    3),
                ("Sharpe ratio",   "sharpe_ratio",     3),
                ("Max DD %",       "max_drawdown_pct", 1),
                ("Expectancy $",   "expectancy_usd",   2),
                ("2-lot trades",   "two_lot_trades",   0),
                ("Avg quality",    "avg_quality_score",1),
            ]
            for lbl, key, d in compare_keys:
                is_v = metrics_is.get(key, np.nan)
                wf_v = metrics_wf.get(key, np.nan)
                try:
                    if not (np.isnan(float(is_v)) or np.isnan(float(wf_v))) and is_v != 0:
                        decay = f"{((wf_v - is_v) / abs(is_v)) * 100:+.1f} %"
                    else:
                        decay = "n/a"
                except Exception:
                    decay = "n/a"
                print(f"  {lbl:<24} {fmt(is_v, d):>14} {fmt(wf_v, d):>14} {decay:>10}")
            print(f"{'═'*90}")
        else:
            print(f"{R}Walk-forward produced no trades.{E}")


if __name__ == "__main__":
    main()
