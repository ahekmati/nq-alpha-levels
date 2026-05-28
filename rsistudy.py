from mt5linux import MetaTrader5
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
import json


# ---------------- CONFIG ---------------- #
SYMBOL = "MNQM26"  # adjust if needed
START_DATE = datetime(2021, 1, 1, tzinfo=timezone.utc)

MODE = "D1"  # "D1" for daily, "H1" for hourly
RSI_PERIOD = 10

OUTDIR = Path("output/mnq_rsi_analysis")
# ---------------------------------------- #


def get_timeframe(mt5, mode: str):
    if mode == "D1":
        return mt5.TIMEFRAME_D1
    elif mode == "H1":
        return mt5.TIMEFRAME_H1
    else:
        raise ValueError("MODE must be 'D1' or 'H1'")


def fetch_bars(symbol: str, start: datetime, mode: str) -> pd.DataFrame:
    mt5 = MetaTrader5()
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    timeframe = get_timeframe(mt5, mode)
    end = datetime.now(timezone.utc)

    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        mt5.shutdown()
        raise RuntimeError(f"No rates returned for {symbol}, error={err}")

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    mt5.shutdown()
    return df


def calc_rsi(prices: pd.Series, period: int = 10) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50)

    return rsi


def crossing_count(values: pd.Series, level: float = 50.0) -> int:
    above = values > level
    cross = (above != above.shift(1)) & above.notna() & above.shift(1).notna()
    return int(cross.sum())


def analyze_rsi(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d["rsi"] = calc_rsi(d["close"], RSI_PERIOD)
    d["rsi_chg_1"] = d["rsi"].diff()
    d["rsi_slope_3"] = d["rsi"] - d["rsi"].shift(3)
    d["rsi_slope_5"] = d["rsi"] - d["rsi"].shift(5)

    d["zone"] = pd.cut(
        d["rsi"],
        bins=[-np.inf, 30, 40, 50, 60, 70, np.inf],
        labels=["<30", "30-40", "40-50", "50-60", "60-70", ">70"],
    )

    d["cross_up_30"] = (d["rsi"].shift(1) < 30) & (d["rsi"] >= 30)
    d["cross_down_60"] = (d["rsi"].shift(1) > 60) & (d["rsi"] <= 60)
    d["cross_up_50"] = (d["rsi"].shift(1) <= 50) & (d["rsi"] > 50)
    d["cross_down_50"] = (d["rsi"].shift(1) >= 50) & (d["rsi"] < 50)
    d["neutral_40_60"] = d["rsi"].between(40, 60, inclusive="both")

    d["fwd_ret_5"] = d["close"].shift(-5) / d["close"] - 1
    d["fwd_ret_10"] = d["close"].shift(-10) / d["close"] - 1
    d["fwd_ret_20"] = d["close"].shift(-20) / d["close"] - 1

    return d


def build_zone_stats(d: pd.DataFrame) -> pd.DataFrame:
    return (
        d.dropna(subset=["zone"])
        .groupby("zone", observed=True)
        .agg(
            bars=("close", "count"),
            avg_5d=("fwd_ret_5", "mean"),
            med_5d=("fwd_ret_5", "median"),
            avg_10d=("fwd_ret_10", "mean"),
            med_10d=("fwd_ret_10", "median"),
            avg_20d=("fwd_ret_20", "mean"),
            med_20d=("fwd_ret_20", "median"),
        )
        .reset_index()
    )


def build_neutral_runs(d: pd.DataFrame) -> pd.DataFrame:
    neutral = d["neutral_40_60"].fillna(False)
    run_id = (neutral != neutral.shift(1)).cumsum()

    neutral_df = d.loc[neutral, ["rsi"]].copy()
    neutral_df["run_id"] = run_id.loc[neutral]
    neutral_df["time"] = neutral_df.index

    neutral_runs = (
        neutral_df.groupby("run_id")
        .agg(
            start=("time", "min"),
            end=("time", "max"),
            bars=("time", "count"),
            rsi_min=("rsi", "min"),
            rsi_max=("rsi", "max"),
        )
        .reset_index(drop=True)
        .sort_values("bars", ascending=False)
    )

    return neutral_runs


def build_arm_stats(d: pd.DataFrame) -> pd.DataFrame:
    buy_arms = d.index[d["cross_up_30"].fillna(False)].tolist()
    sell_arms = d.index[d["cross_down_60"].fillna(False)].tolist()

    def arm_stats(indices, side):
        rows = []
        for ts in indices:
            loc = d.index.get_loc(ts)
            win = d.iloc[loc:min(loc + 11, len(d))].copy()
            if win.empty:
                continue

            rsi_vals = win["rsi"].dropna()
            if rsi_vals.empty:
                continue

            rows.append(
                {
                    "time": ts,
                    "side": side,
                    "rsi_at_arm": float(d.at[ts, "rsi"]),
                    "max_rsi_next_10": float(rsi_vals.max()),
                    "min_rsi_next_10": float(rsi_vals.min()),
                    "crosses_50_next_10": crossing_count(rsi_vals, 50.0),
                    "stayed_neutral_next_5": bool(
                        d["neutral_40_60"].iloc[loc:min(loc + 6, len(d))].fillna(False).all()
                    ),
                    "ret_5d": d.at[ts, "fwd_ret_5"],
                    "ret_10d": d.at[ts, "fwd_ret_10"],
                    "ret_20d": d.at[ts, "fwd_ret_20"],
                }
            )
        return pd.DataFrame(rows)

    arm_df = pd.concat(
        [
            arm_stats(buy_arms, "buy_arm"),
            arm_stats(sell_arms, "sell_arm"),
        ],
        ignore_index=True,
    )

    return arm_df


def build_summary(d: pd.DataFrame, neutral_runs: pd.DataFrame) -> dict:
    return {
        "symbol": SYMBOL,
        "mode": MODE,
        "bars": int(len(d)),
        "start": d.index.min().isoformat() if len(d) else None,
        "end": d.index.max().isoformat() if len(d) else None,
        "rsi_period": RSI_PERIOD,
        "rsi_mean": float(d["rsi"].mean(skipna=True)),
        "rsi_median": float(d["rsi"].median(skipna=True)),
        "pct_neutral_40_60": float(d["neutral_40_60"].mean(skipna=True)),
        "buy_arm_count": int(d["cross_up_30"].sum(skipna=True)),
        "sell_arm_count": int(d["cross_down_60"].sum(skipna=True)),
        "cross_up_50_count": int(d["cross_up_50"].sum(skipna=True)),
        "cross_down_50_count": int(d["cross_down_50"].sum(skipna=True)),
        "neutral_run_count": int(len(neutral_runs)),
        "neutral_run_avg_bars": float(neutral_runs["bars"].mean()) if len(neutral_runs) else 0.0,
        "neutral_run_max_bars": int(neutral_runs["bars"].max()) if len(neutral_runs) else 0,
    }


def main():
    print(f"Fetching {MODE} bars for {SYMBOL} from {START_DATE.date()} ...")
    bars = fetch_bars(SYMBOL, START_DATE, MODE)
    print(f"Got {len(bars)} bars.")

    data = analyze_rsi(bars)
    zone_stats = build_zone_stats(data)
    neutral_runs = build_neutral_runs(data)
    arm_stats = build_arm_stats(data)
    summary = build_summary(data, neutral_runs)

    OUTDIR.mkdir(parents=True, exist_ok=True)

    data.to_csv(OUTDIR / "bars_with_rsi.csv")
    zone_stats.to_csv(OUTDIR / "rsi_zone_stats.csv", index=False)
    neutral_runs.to_csv(OUTDIR / "neutral_40_60_runs.csv", index=False)
    arm_stats.to_csv(OUTDIR / "rsi_arm_events.csv", index=False)

    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote analysis files to: {OUTDIR}")


if __name__ == "__main__":
    main()
