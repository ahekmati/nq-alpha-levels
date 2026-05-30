"""
MNQ Overnight Behavior Study  v3
==================================
Study 1 — THREE VARIANTS run in parallel:
  Qualifying "strong up day" filters (all must pass):
    1. Daily RSI(10) >= 60
    2. H1 close > H1 EMA(100)
    3. H1 EMA(20) > H1 EMA(100)
    4. RTH gain >= threshold  [0.8%, 1.0%, 1.2% tested side-by-side]

  Sub-session breakdown now shown for EVERY dip level (0.75x and 1.0x)
  not just the top combo, to confirm Asian session dominance.

Study 2 — rally_mult >= 2.5 filter applied:
  Only setups where overnight rally was >= 2.5x ATR are traded.
  Reversal% and expectancy re-measured on filtered universe.
  Sub-session breakdown and rally segmentation retained.

Symbol  : @MNQ (H1 bars via mt5linux)
Period  : ~7 years of available H1 data
Sessions: Asian      = 16:00-00:00 ET
          European   = 00:00-07:00 ET
          Pre-market = 07:00-09:30 ET
          RTH        = 09:30-16:00 ET

Run     : python mnq_overnight_study.py
Output  : CSV files in ./mnq_study/ + console summary
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

warnings.filterwarnings("ignore")

# ── mt5linux ──────────────────────────────────────────────────────────────────
from mt5linux import MetaTrader5
MT5_HOST     = "127.0.0.1"
MT5_PORT     = 18812
mt5          = MetaTrader5(host=MT5_HOST, port=MT5_PORT)
TIMEFRAME_H1 = 16385
TIMEFRAME_D1 = 16408

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SYMBOL     = "@MNQ"
N_BARS_H1  = 50000        # H1 bars  (~7 years with buffer)
N_BARS_D1  = 2000         # D1 bars  (~8 years, enough for RSI warmup)
OUTPUT_DIR = "./mnq_study"
ET         = pytz.timezone("America/New_York")

# ── Study 1 filters ───────────────────────────────────────────
DAILY_RSI_PERIOD    = 10
DAILY_RSI_MIN       = 60.0    # daily RSI must be >= this
H1_EMA_FAST         = 20      # H1 EMA fast
H1_EMA_SLOW         = 100     # H1 EMA slow (price must be above this)
RTH_GAIN_MIN_PCT    = 0.008   # RTH session must close up >= 0.8% (base)
RTH_GAIN_VARIANTS   = [0.008, 0.010, 0.012]  # 0.8%, 1.0%, 1.2% tested side-by-side

# ── Shared indicator ──────────────────────────────────────────
ATR_PERIOD = 14

# ── Parameter grids ───────────────────────────────────────────
DIP_ATR_MULTS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
RR_RATIOS     = [1.0, 1.5, 2.0, 2.5, 3.0]

# ── Study 2 thresholds ────────────────────────────────────────
RALLY_ATR_MULT      = 1.5     # minimum overnight range to qualify as a rally
RALLY_ATR_FILTER    = 2.5     # REFINED: only trade dips when rally was >= 2.5x ATR
SELLOFF_PCT         = 0.005


# ─────────────────────────────────────────────
# MT5 CONNECTION + DATA FETCH
# ─────────────────────────────────────────────

def connect():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 connection failed: {mt5.last_error()}")
    print("[MT5] Connected via mt5linux")


def get_bars(symbol: str, tf: int, n: int) -> pd.DataFrame:
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
    if rates is None or len(rates) == 0:
        raise ValueError(f"No data for {symbol} tf={tf}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "tick_volume": "Volume"
    }, inplace=True)
    return df[["Open", "High", "Low", "Close", "Volume"]].copy()


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────

def calc_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def calc_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def add_h1_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR, EMAs, session tag to H1 dataframe."""
    df["atr"]      = calc_atr(df)
    df["ema20"]    = calc_ema(df["Close"], H1_EMA_FAST)
    df["ema100"]   = calc_ema(df["Close"], H1_EMA_SLOW)
    df["green"]    = (df["Close"] > df["Open"]).astype(int)
    df["bar_size"] = df["Close"] - df["Open"]
    df["time_et"]  = df.index.tz_convert(ET)
    return df


def build_daily_rsi(df_d1: pd.DataFrame) -> pd.Series:
    """
    Compute daily RSI(10) from D1 bars.
    Returns a Series indexed by date (ET date of the D1 bar).
    """
    rsi = calc_rsi(df_d1["Close"], DAILY_RSI_PERIOD)
    rsi.index = pd.to_datetime(rsi.index).tz_convert(ET).normalize()
    return rsi


# ─────────────────────────────────────────────
# SESSION SLICERS
# ─────────────────────────────────────────────

def rth_bars(df: pd.DataFrame, date) -> pd.DataFrame:
    start = ET.localize(datetime(date.year, date.month, date.day, 9, 30))
    end   = ET.localize(datetime(date.year, date.month, date.day, 16, 0))
    return df[(df["time_et"] >= start) & (df["time_et"] < end)].copy()


def overnight_bars(df: pd.DataFrame, date) -> pd.DataFrame:
    """16:00 ET on date → 09:30 ET next calendar day, tagged by sub-session."""
    start    = ET.localize(datetime(date.year, date.month, date.day, 16, 0))
    next_day = date + timedelta(days=1)
    end      = ET.localize(datetime(next_day.year, next_day.month, next_day.day, 9, 30))
    sub      = df[(df["time_et"] >= start) & (df["time_et"] < end)].copy()

    def tag(ts):
        h = ts.hour + ts.minute / 60.0
        if h >= 16:   return "Asian"        # 16:00–00:00
        elif h < 7:   return "European"     # 00:00–07:00
        else:         return "Pre-market"   # 07:00–09:30

    if len(sub) > 0:
        sub["sub_session"] = sub["time_et"].apply(tag)
    return sub


# ─────────────────────────────────────────────
# STUDY 1 QUALIFYING DAY DETECTION  (v2)
# ─────────────────────────────────────────────

def is_strong_up_day(rth: pd.DataFrame,
                     daily_rsi_series: pd.Series,
                     date,
                     gain_override: float = None) -> tuple:
    """
    A qualifying strong up day requires ALL of:
      1. daily RSI(10) on this date >= DAILY_RSI_MIN (60)
      2. H1 close at end of RTH > H1 EMA(100)
      3. H1 EMA(20) > H1 EMA(100) at end of RTH
      4. RTH session gain >= RTH_GAIN_MIN_PCT (1.0%)

    Returns (bool, meta_dict)
    """
    if len(rth) < 2:
        return False, {}

    last_bar  = rth.iloc[-1]
    first_bar = rth.iloc[0]

    rth_open  = float(first_bar["Open"])
    rth_close = float(last_bar["Close"])
    atr_val   = float(last_bar["atr"])

    if rth_open == 0 or pd.isna(atr_val) or atr_val == 0:
        return False, {}

    # 1. Daily RSI check
    # Match date to daily RSI index (normalize to midnight ET)
    d_key = pd.Timestamp(date).tz_localize(ET).normalize()
    # Try same day and previous day (D1 bar may be stamped at open)
    d_rsi = None
    for offset in [0, -1, 1]:
        candidate = d_key + pd.Timedelta(days=offset)
        if candidate in daily_rsi_series.index:
            d_rsi = daily_rsi_series[candidate]
            break
    if d_rsi is None or pd.isna(d_rsi):
        return False, {}

    rsi_ok = float(d_rsi) >= DAILY_RSI_MIN

    # 2. Price above H1 EMA(100)
    ema100_val = float(last_bar["ema100"])
    price_above_ema100 = rth_close > ema100_val if not pd.isna(ema100_val) else False

    # 3. H1 EMA(20) > H1 EMA(100) — trend aligned
    ema20_val = float(last_bar["ema20"])
    ema_aligned = (ema20_val > ema100_val) if (
        not pd.isna(ema20_val) and not pd.isna(ema100_val)) else False

    # 4. RTH gain >= threshold
    gain_threshold = gain_override if gain_override is not None else RTH_GAIN_MIN_PCT
    rth_gain_pct = (rth_close - rth_open) / rth_open
    gain_ok = rth_gain_pct >= gain_threshold

    qualified = rsi_ok and price_above_ema100 and ema_aligned and gain_ok

    meta = {
        "daily_rsi"        : round(float(d_rsi), 2),
        "ema20"            : round(ema20_val, 2),
        "ema100"           : round(ema100_val, 2),
        "rth_gain_pct"     : round(rth_gain_pct * 100, 3),
        "rth_open"         : rth_open,
        "rth_close"        : rth_close,
        "day_high"         : float(rth["High"].max()),
        "day_low"          : float(rth["Low"].min()),
        "atr_val"          : atr_val,
        "filter_rsi"       : rsi_ok,
        "filter_above_ema" : price_above_ema100,
        "filter_ema_align" : ema_aligned,
        "filter_gain"      : gain_ok,
    }
    return qualified, meta


# ─────────────────────────────────────────────
# DIP DETECTION  (unchanged)
# ─────────────────────────────────────────────

def find_overnight_dip(overnight: pd.DataFrame,
                       reference_price: float,
                       atr_val: float,
                       dip_mult: float):
    threshold = reference_price - dip_mult * atr_val
    for i, (idx, row) in enumerate(overnight.iterrows()):
        if row["Low"] <= threshold:
            return i, idx, float(row["Low"]), row.get("sub_session", "unknown")
    return None, None, None, None


# ─────────────────────────────────────────────
# TRADE EVALUATION  (unchanged)
# ─────────────────────────────────────────────

def evaluate_trade(df: pd.DataFrame,
                   entry_idx,
                   entry_price: float,
                   atr_val: float,
                   rr_ratio: float,
                   stop_atr_mult: float = 1.0) -> tuple:
    stop   = entry_price - stop_atr_mult * atr_val
    target = entry_price + rr_ratio * stop_atr_mult * atr_val
    risk   = entry_price - stop

    future = df[df.index > entry_idx].head(24)
    mfe = mae = 0.0
    result = "open"

    for _, bar in future.iterrows():
        mfe = max(mfe, bar["High"] - entry_price)
        mae = max(mae, entry_price - bar["Low"])
        if bar["Low"] <= stop:
            result = "loss"
            break
        if bar["High"] >= target:
            result = "win"
            break

    return result, float(risk), float(mfe), float(mae), float(stop), float(target)


# ─────────────────────────────────────────────
# STUDY 1  —  Strong up day → overnight dip
# ─────────────────────────────────────────────

def _session_breakdown(triggered_df: pd.DataFrame,
                        dip_mult: float, rr_ratio: float) -> pd.DataFrame:
    """Helper: sub-session win rate for a specific dip/RR combo."""
    sub = triggered_df[
        (triggered_df["dip_atr_mult"] == dip_mult) &
        (triggered_df["rr_ratio"]     == rr_ratio)
    ]
    if len(sub) == 0:
        return pd.DataFrame()
    return (
        sub.groupby("sub_session")
        .apply(lambda x: pd.Series({
            "count"   : len(x),
            "wins"    : int((x["result"] == "win").sum()),
            "losses"  : int((x["result"] == "loss").sum()),
            "win_rate": round(
                (x["result"] == "win").sum() /
                max((x["result"].isin(["win","loss"])).sum(), 1), 3),
        }))
        .reset_index()
    )


def _run_study1_for_threshold(df: pd.DataFrame,
                               daily_rsi: pd.Series,
                               gain_thresh: float) -> tuple:
    """
    Run Study 1 for a single RTH gain threshold.
    Returns (trades_df, summary_df, n_qual, n_dates, filter_counts).
    """
    dates     = sorted(set(df["time_et"].dt.date))
    qual_days = []
    records   = []
    f_rsi = f_ema = f_align = f_gain = f_all = 0

    for date in dates:
        rth = rth_bars(df, date)
        if len(rth) < 2:
            continue

        # Temporarily override threshold
        qualified, meta = is_strong_up_day(rth, daily_rsi, date,
                                            gain_override=gain_thresh)
        if meta:
            if meta.get("filter_rsi"):       f_rsi   += 1
            if meta.get("filter_above_ema"): f_ema   += 1
            if meta.get("filter_ema_align"): f_align += 1
            if meta.get("filter_gain"):      f_gain  += 1
            if qualified:                    f_all   += 1

        if not qualified:
            continue

        qual_days.append(date)
        overnight = overnight_bars(df, date)
        if len(overnight) == 0:
            continue

        atr_val   = meta["atr_val"]
        rth_close = meta["rth_close"]

        for dip_mult in DIP_ATR_MULTS:
            _, entry_idx, entry_price, sub_sess = find_overnight_dip(
                overnight, rth_close, atr_val, dip_mult)

            if entry_idx is None:
                records.append({
                    "date"        : str(date),
                    "dip_atr_mult": dip_mult,
                    "rr_ratio"    : None,
                    "triggered"   : False,
                    "result"      : "no_trigger",
                    "sub_session" : None,
                    "gain_thresh" : gain_thresh,
                    **meta,
                })
                continue

            for rr in RR_RATIOS:
                result, risk, mfe, mae, stop, target = evaluate_trade(
                    df, entry_idx, entry_price, atr_val, rr)
                records.append({
                    "date"        : str(date),
                    "dip_atr_mult": dip_mult,
                    "rr_ratio"    : rr,
                    "triggered"   : True,
                    "result"      : result,
                    "sub_session" : sub_sess,
                    "entry_price" : entry_price,
                    "risk_pts"    : risk,
                    "mfe_pts"     : mfe,
                    "mae_pts"     : mae,
                    "gain_thresh" : gain_thresh,
                    **meta,
                })

    trades_df = pd.DataFrame(records)
    n_qual    = len(qual_days)
    n_dates   = len(dates)

    if n_qual == 0:
        return trades_df, pd.DataFrame(), n_qual, n_dates, \
               (f_rsi, f_ema, f_align, f_gain, f_all)

    triggered_df = trades_df[trades_df["triggered"] == True]
    rows = []
    for dip_mult in DIP_ATR_MULTS:
        for rr in RR_RATIOS:
            sub = triggered_df[
                (triggered_df["dip_atr_mult"] == dip_mult) &
                (triggered_df["rr_ratio"] == rr)
            ]
            if len(sub) == 0:
                continue
            wins   = int((sub["result"] == "win").sum())
            losses = int((sub["result"] == "loss").sum())
            opens  = int((sub["result"] == "open").sum())
            total  = wins + losses
            wr     = wins / total if total > 0 else 0
            exp    = (wr * rr) - (1 - wr)
            trate  = sub["date"].nunique() / n_qual
            rows.append({
                "gain_thresh"  : gain_thresh,
                "dip_atr_mult" : dip_mult,
                "rr_ratio"     : rr,
                "n_trades"     : total,
                "wins"         : wins,
                "losses"       : losses,
                "open"         : opens,
                "win_rate"     : round(wr, 3),
                "expectancy_R" : round(exp, 3),
                "trigger_rate" : round(trate, 3),
            })

    summary_df = pd.DataFrame(rows).sort_values("expectancy_R", ascending=False)
    return trades_df, summary_df, n_qual, n_dates, \
           (f_rsi, f_ema, f_align, f_gain, f_all)


def run_study1(df: pd.DataFrame, daily_rsi: pd.Series):
    print("\n" + "=" * 65)
    print("STUDY 1  |  Buy Overnight Dip After Strong Up Day (v3)")
    print("=" * 65)
    print(f"  Fixed filters: Daily RSI({DAILY_RSI_PERIOD}) >= {DAILY_RSI_MIN}  |  "
          f"Price > EMA({H1_EMA_SLOW})  |  EMA({H1_EMA_FAST}) > EMA({H1_EMA_SLOW})")
    print(f"  RTH gain threshold tested: {[f'{g*100:.1f}%' for g in RTH_GAIN_VARIANTS]}")

    all_trades   = []
    all_summaries = []

    for gain_thresh in RTH_GAIN_VARIANTS:
        trades_df, summary_df, n_qual, n_dates, fcounts = \
            _run_study1_for_threshold(df, daily_rsi, gain_thresh)

        f_rsi, f_ema, f_align, f_gain, f_all = fcounts

        print(f"\n{'─'*65}")
        print(f"  RTH gain >= {gain_thresh*100:.1f}%  →  {n_qual} qualifying days "
              f"({n_qual/n_dates*100:.1f}% of {n_dates} trading days)")
        print(f"  Filter pass rates (individual):")
        print(f"    Daily RSI >= {DAILY_RSI_MIN}      : {f_rsi:4d}  ({f_rsi/n_dates*100:.1f}%)")
        print(f"    Price > EMA({H1_EMA_SLOW})         : {f_ema:4d}  ({f_ema/n_dates*100:.1f}%)")
        print(f"    EMA({H1_EMA_FAST}) > EMA({H1_EMA_SLOW})        : {f_align:4d}  ({f_align/n_dates*100:.1f}%)")
        print(f"    RTH gain >= {gain_thresh*100:.1f}%          : {f_gain:4d}  ({f_gain/n_dates*100:.1f}%)")

        if n_qual == 0:
            print("  ⚠  No qualifying days.")
            continue

        triggered_df = trades_df[trades_df["triggered"] == True]

        # ── Trigger rates ──────────────────────────────────────
        print(f"\n  Dip trigger rates:")
        for dip_mult in DIP_ATR_MULTS:
            days_t = trades_df[
                (trades_df["dip_atr_mult"] == dip_mult) &
                (trades_df["triggered"] == True)
            ]["date"].nunique()
            print(f"    Dip >= {dip_mult:.2f}x ATR : {days_t}/{n_qual} days "
                  f"({days_t/n_qual*100:.1f}%)")

        # ── Top combos ────────────────────────────────────────
        if len(summary_df) > 0:
            print(f"\n  Top 10 combinations (gain={gain_thresh*100:.1f}%):")
            print(summary_df.head(10).to_string(index=False))

        # ── Sub-session breakdown for 0.75x, 1.0x, and best combo ─
        print(f"\n  Sub-session breakdown by dip level:")
        for focus_dip in [0.75, 1.0]:
            # Use best RR for this dip level
            best_rr_row = summary_df[summary_df["dip_atr_mult"] == focus_dip]
            if len(best_rr_row) == 0:
                continue
            best_rr = best_rr_row.iloc[0]["rr_ratio"]
            sess = _session_breakdown(triggered_df, focus_dip, best_rr)
            if len(sess) > 0:
                print(f"\n    dip={focus_dip}x ATR, best RR={best_rr}:")
                print(sess.to_string(index=False))

        # Also show best overall combo session breakdown
        if len(summary_df) > 0:
            best = summary_df.iloc[0]
            if best["dip_atr_mult"] not in [0.75, 1.0]:
                sess = _session_breakdown(triggered_df,
                                          best["dip_atr_mult"], best["rr_ratio"])
                if len(sess) > 0:
                    print(f"\n    Best overall (dip={best['dip_atr_mult']}x ATR, "
                          f"RR={best['rr_ratio']}):")
                    print(sess.to_string(index=False))

        all_trades.append(trades_df)
        all_summaries.append(summary_df)

    # ── Cross-threshold comparison table ──────────────────────
    print(f"\n{'='*65}")
    print("  CROSS-THRESHOLD COMPARISON  (best combo per gain threshold)")
    print(f"{'='*65}")
    print(f"  {'Threshold':>10}  {'QualDays':>9}  {'DipATR':>7}  "
          f"{'RR':>5}  {'N':>5}  {'WR':>7}  {'Exp_R':>7}  {'TrigRate':>9}")
    dates_total = len(sorted(set(df["time_et"].dt.date)))
    for gain_thresh, summary_df in zip(RTH_GAIN_VARIANTS, all_summaries):
        if len(summary_df) == 0:
            continue
        _, _, n_qual, _, _ = _run_study1_for_threshold.__wrapped__ \
            if hasattr(_run_study1_for_threshold, '__wrapped__') \
            else (None, None, None, None, None)
        # Re-derive n_qual from summary trigger_rate and n_trades
        best = summary_df.iloc[0]
        # n_qual = n_trades / trigger_rate (approx, integer)
        n_qual_est = int(round(best["n_trades"] / best["trigger_rate"])) \
            if best["trigger_rate"] > 0 else "?"
        print(f"  {gain_thresh*100:>9.1f}%  {str(n_qual_est):>9}  "
              f"{best['dip_atr_mult']:>7.2f}  {best['rr_ratio']:>5.1f}  "
              f"{best['n_trades']:>5}  {best['win_rate']:>7.1%}  "
              f"{best['expectancy_R']:>7.3f}  {best['trigger_rate']:>9.3f}")

    # Return base threshold results for equity curve + CSV
    if all_trades:
        base_idx = RTH_GAIN_VARIANTS.index(RTH_GAIN_MIN_PCT) \
            if RTH_GAIN_MIN_PCT in RTH_GAIN_VARIANTS else 0
        return all_trades[base_idx], \
               all_summaries[base_idx] if all_summaries else pd.DataFrame()
    return pd.DataFrame(), pd.DataFrame()


# ─────────────────────────────────────────────
# STUDY 2  —  Overnight rally → RTH selloff → overnight dip
# ─────────────────────────────────────────────

def run_study2(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("STUDY 2  |  Overnight Rally → RTH Selloff → Next Overnight Dip (v3)")
    print("=" * 65)
    print(f"  Overnight rally filter : >= {RALLY_ATR_MULT}x ATR  (setup qualifier)")
    print(f"  Trade entry filter     : rally_mult >= {RALLY_ATR_FILTER}x ATR  (refined)")
    print(f"  RTH selloff threshold  : >= {SELLOFF_PCT*100:.1f}%")

    dates      = sorted(set(df["time_et"].dt.date))
    setup_recs = []
    trade_recs = []

    for date in dates:
        prev_date = date - timedelta(days=1)
        on        = overnight_bars(df, prev_date)
        if len(on) < 3:
            continue

        on_low   = on["Low"].min()
        on_high  = on["High"].max()
        on_range = on_high - on_low
        avg_atr  = on["atr"].mean()

        if pd.isna(avg_atr) or avg_atr == 0:
            continue
        if on_range < RALLY_ATR_MULT * avg_atr:
            continue

        rth = rth_bars(df, date)
        if len(rth) < 3:
            continue

        rth_close   = float(rth.iloc[-1]["Close"])
        selloff_pts = on_high - rth_close
        selloff_pct = selloff_pts / on_high if on_high > 0 else 0
        selloff_ok  = selloff_pct >= SELLOFF_PCT
        rally_mult  = on_range / avg_atr

        setup_recs.append({
            "date"       : str(date),
            "on_low"     : round(on_low, 2),
            "on_high"    : round(on_high, 2),
            "on_range"   : round(on_range, 2),
            "avg_atr"    : round(avg_atr, 2),
            "rally_mult" : round(rally_mult, 2),
            "rth_close"  : round(rth_close, 2),
            "selloff_pct": round(selloff_pct, 4),
            "selloff_ok" : selloff_ok,
        })

        if not selloff_ok:
            continue

        # ── REFINED: only trade when rally was large enough ───
        if rally_mult < RALLY_ATR_FILTER:
            continue

        next_on = overnight_bars(df, date)
        if len(next_on) == 0:
            continue

        rth_atr = float(rth.iloc[-1]["atr"])
        if pd.isna(rth_atr) or rth_atr == 0:
            rth_atr = avg_atr

        for dip_mult in DIP_ATR_MULTS:
            _, entry_idx, entry_price, sub_sess = find_overnight_dip(
                next_on, rth_close, rth_atr, dip_mult)

            if entry_idx is None:
                trade_recs.append({
                    "date"        : str(date),
                    "dip_atr_mult": dip_mult,
                    "rr_ratio"    : None,
                    "triggered"   : False,
                    "result"      : "no_trigger",
                    "sub_session" : None,
                    "selloff_pct" : round(selloff_pct, 4),
                    "rally_mult"  : round(rally_mult, 2),
                })
                continue

            for rr in RR_RATIOS:
                result, risk, mfe, mae, stop, target = evaluate_trade(
                    df, entry_idx, entry_price, rth_atr, rr)
                trade_recs.append({
                    "date"        : str(date),
                    "dip_atr_mult": dip_mult,
                    "rr_ratio"    : rr,
                    "triggered"   : True,
                    "result"      : result,
                    "sub_session" : sub_sess,
                    "entry_price" : entry_price,
                    "rth_close"   : rth_close,
                    "atr_val"     : rth_atr,
                    "risk_pts"    : risk,
                    "mfe_pts"     : mfe,
                    "mae_pts"     : mae,
                    "selloff_pct" : round(selloff_pct, 4),
                    "rally_mult"  : round(rally_mult, 2),
                })

    setups_df  = pd.DataFrame(setup_recs)
    trades_df  = pd.DataFrame(trade_recs)

    n_rallies       = len(setups_df)
    n_selloffs_all  = int(setups_df["selloff_ok"].sum()) if len(setups_df) > 0 else 0
    n_selloffs      = int(
        setups_df[(setups_df["selloff_ok"] == True) &
                  (setups_df["rally_mult"] >= RALLY_ATR_FILTER)]["selloff_ok"].sum()
    ) if len(setups_df) > 0 else 0

    pct_all = n_selloffs_all / n_rallies * 100 if n_rallies > 0 else 0
    pct_flt = n_selloffs / n_rallies * 100 if n_rallies > 0 else 0
    print(f"\n  Overnight rally setups (>={RALLY_ATR_MULT}x ATR)  : {n_rallies}")
    print(f"  With RTH selloff (all rallies)               : {n_selloffs_all} ({pct_all:.1f}%)")
    print(f"  With RTH selloff AND rally>={RALLY_ATR_FILTER}x ATR    : {n_selloffs} ({pct_flt:.1f}%)")

    if n_selloffs == 0:
        print("  ⚠  No qualifying setups.")
        return trades_df, pd.DataFrame()

    # ── Reversal vs Continuation ──────────────────────────────
    # Report on BOTH unfiltered and filtered (>= RALLY_ATR_FILTER) setups
    selloff_df = setups_df[setups_df["selloff_ok"] == True].copy()
    selloff_filtered = selloff_df[selloff_df["rally_mult"] >= RALLY_ATR_FILTER].copy()

    for label, sdf in [("All rallies (unfiltered)", selloff_df),
                        (f"Rally >= {RALLY_ATR_FILTER}x ATR (filtered)", selloff_filtered)]:
        rev = cont = 0
        for _, row in sdf.iterrows():
            try:
                d   = datetime.strptime(row["date"], "%Y-%m-%d").date()
                non = overnight_bars(df, d)
                if len(non) == 0:
                    continue
                ref = row["rth_close"]
                a   = row["avg_atr"]
                if non["High"].max() >= ref + 0.5 * a:
                    rev  += 1
                elif non["Low"].min() < ref - a:
                    cont += 1
            except Exception:
                continue
        tc = rev + cont
        n  = len(sdf)
        print(f"\n  [{label}]  n={n} setups")
        if tc > 0:
            print(f"    Reversal     ≥0.5 ATR recovery : {rev:4d}  ({rev/tc*100:.1f}%)")
            print(f"    Continuation ≥1.0 ATR drop     : {cont:4d}  ({cont/tc*100:.1f}%)")
            print(f"    Unclassified                   : {n - tc}")

    # ── Summary grid ──────────────────────────────────────────
    triggered_df = trades_df[trades_df["triggered"] == True]
    if len(triggered_df) == 0:
        print("  ⚠  No dip trades triggered.")
        return trades_df, pd.DataFrame()

    rows = []
    for dip_mult in DIP_ATR_MULTS:
        for rr in RR_RATIOS:
            sub = triggered_df[
                (triggered_df["dip_atr_mult"] == dip_mult) &
                (triggered_df["rr_ratio"] == rr)
            ]
            if len(sub) == 0:
                continue
            wins   = int((sub["result"] == "win").sum())
            losses = int((sub["result"] == "loss").sum())
            total  = wins + losses
            wr     = wins / total if total > 0 else 0
            exp    = (wr * rr) - (1 - wr)
            trate  = sub["date"].nunique() / n_selloffs

            rows.append({
                "dip_atr_mult" : dip_mult,
                "rr_ratio"     : rr,
                "n_trades"     : total,
                "wins"         : wins,
                "losses"       : losses,
                "win_rate"     : round(wr, 3),
                "expectancy_R" : round(exp, 3),
                "trigger_rate" : round(trate, 3),
            })

    summary_df = pd.DataFrame(rows).sort_values("expectancy_R", ascending=False)
    print("\n  Top 10 Param Combinations (by Expectancy):")
    print(summary_df.head(10).to_string(index=False))

    # ── Sub-session breakdown ─────────────────────────────────
    if len(summary_df) > 0:
        best     = summary_df.iloc[0]
        best_sub = triggered_df[
            (triggered_df["dip_atr_mult"] == best["dip_atr_mult"]) &
            (triggered_df["rr_ratio"]     == best["rr_ratio"])
        ]
        sess = (
            best_sub.groupby("sub_session")
            .apply(lambda x: pd.Series({
                "count"   : len(x),
                "wins"    : int((x["result"] == "win").sum()),
                "losses"  : int((x["result"] == "loss").sum()),
                "win_rate": round(
                    (x["result"] == "win").sum() /
                    max((x["result"].isin(["win","loss"])).sum(), 1), 3),
            }))
            .reset_index()
        )
        print(f"\n  Sub-session breakdown  "
              f"(dip={best['dip_atr_mult']}x ATR, RR={best['rr_ratio']}):")
        print(sess.to_string(index=False))

        # ── Rally size segmentation ───────────────────────────
        print("\n  Edge by overnight rally magnitude:")
        for lb, ub, label in [(1.5, 2.5, "1.5–2.5x ATR"),
                               (2.5, 4.0, "2.5–4.0x ATR"),
                               (4.0, 99,  ">4.0x ATR")]:
            seg = triggered_df[
                (triggered_df["dip_atr_mult"] == best["dip_atr_mult"]) &
                (triggered_df["rr_ratio"]     == best["rr_ratio"]) &
                (triggered_df["rally_mult"]   >= lb) &
                (triggered_df["rally_mult"]   <  ub)
            ]
            if len(seg) == 0:
                continue
            w = int((seg["result"] == "win").sum())
            l = int((seg["result"] == "loss").sum())
            t = w + l
            wr = w / t if t > 0 else 0
            ex = (wr * best["rr_ratio"]) - (1 - wr)
            print(f"    {label:15s}  n={t:3d}  WR={wr:.1%}  Exp={ex:+.3f}R")

    return trades_df, summary_df


# ─────────────────────────────────────────────
# EQUITY CURVE BUILDER
# ─────────────────────────────────────────────

def build_equity_curve(trades_df, dip_mult, rr_ratio, label):
    sub = trades_df[
        (trades_df["triggered"] == True) &
        (trades_df["dip_atr_mult"] == dip_mult) &
        (trades_df["rr_ratio"] == rr_ratio)
    ].copy().sort_values("date")
    if len(sub) == 0:
        return pd.DataFrame()
    sub["pnl_R"] = sub["result"].map({"win": rr_ratio, "loss": -1.0, "open": 0.0})
    sub["cum_R"] = sub["pnl_R"].cumsum()
    sub["study"] = label
    return sub[["date","dip_atr_mult","rr_ratio","result",
                "pnl_R","cum_R","sub_session","study"]]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    connect()

    # ── Fetch data ────────────────────────────────────────────
    print(f"\nFetching H1 data for {SYMBOL} ({N_BARS_H1:,} bars)...")
    df_h1 = get_bars(SYMBOL, TIMEFRAME_H1, N_BARS_H1)
    df_h1 = add_h1_indicators(df_h1)

    print(f"Fetching D1 data for {SYMBOL} ({N_BARS_D1:,} bars)...")
    df_d1 = get_bars(SYMBOL, TIMEFRAME_D1, N_BARS_D1)

    daily_rsi = build_daily_rsi(df_d1)

    h1_start = df_h1["time_et"].min().strftime("%Y-%m-%d")
    h1_end   = df_h1["time_et"].max().strftime("%Y-%m-%d")
    print(f"H1  : {len(df_h1):,} bars  |  {h1_start} → {h1_end}")
    print(f"D1  : {len(df_d1):,} bars  |  daily RSI computed")

    # ── Studies ───────────────────────────────────────────────
    s1_trades, s1_summary = run_study1(df_h1, daily_rsi)
    s2_trades, s2_summary = run_study2(df_h1)

    # ── Equity curves ─────────────────────────────────────────
    ec_parts = []
    if len(s1_summary) > 0:
        b = s1_summary.iloc[0]
        ec_parts.append(build_equity_curve(
            s1_trades, b["dip_atr_mult"], b["rr_ratio"],
            "Study1_StrongUpDay_OvernightDip"))
    if len(s2_summary) > 0:
        b = s2_summary.iloc[0]
        ec_parts.append(build_equity_curve(
            s2_trades, b["dip_atr_mult"], b["rr_ratio"],
            "Study2_Rally_Selloff_Dip"))

    # ── Save CSVs ─────────────────────────────────────────────
    saved = {}
    def save(df_, name):
        if df_ is not None and len(df_) > 0:
            p = os.path.join(OUTPUT_DIR, name)
            df_.to_csv(p, index=False)
            saved[name] = p

    save(s1_trades,  "study1_all_trades.csv")
    save(s1_summary, "study1_summary.csv")
    save(s2_trades,  "study2_all_trades.csv")
    save(s2_summary, "study2_summary.csv")
    if ec_parts:
        save(pd.concat(ec_parts, ignore_index=True), "equity_curves.csv")

    print("\n" + "=" * 65)
    print("FILES SAVED")
    print("=" * 65)
    for name, path in saved.items():
        print(f"  {name:<40} → {path}")

    print("""
INTERPRETATION GUIDE
─────────────────────────────────────────────────────────────
Study 1 filter diagnostics:
  If qualifying days < 100, consider loosening ONE filter at
  a time. The "filter pass rates (individual)" lines show
  which filter is the tightest bottleneck.

Study 1 trade edge:
  expectancy_R > 0.20 = edge worth building on
  expectancy_R > 0.40 = strong edge
  trigger_rate > 0.30 = setup fires on 30%+ of qualifying days

Study 2 rally size segmentation:
  Larger overnight rallies (>2.5x ATR) should show higher
  reversal win rates — confirms institutional accumulation
  thesis behind the setup.
─────────────────────────────────────────────────────────────
""")

    mt5.shutdown()
    print("[MT5] Disconnected. Study complete.")


if __name__ == "__main__":
    main()
