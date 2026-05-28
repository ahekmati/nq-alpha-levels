import time
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pandas as pd
from mt5linux import MetaTrader5


# =========================
# CONFIG
# =========================
BASE_DIR = Path("research_mnq_bear_model") / "promotion_early_short_proto_v1"
RULES_FILE = BASE_DIR / "live_rules_manifest.csv"
DATA_FILE = BASE_DIR / "h1_execution_dataset_with_proto_v1.csv"
STATE_FILE = BASE_DIR / "executor_state.csv"
LOG_FILE = BASE_DIR / "executor_log.csv"

RULE_NAME = "early_short_proto_v1"

MT5_HOST = "localhost"
MT5_PORT = 18812

SYMBOL = "MNQ"
LOT_SIZE = 1.0
MAGIC = 9001001
DEVIATION = 30
POLL_SECONDS = 20

DRY_RUN = True

USE_RAW_INDEX_POINTS = True
DEFAULT_POINT_SIZE = 1.0


mt5 = MetaTrader5(host=MT5_HOST, port=MT5_PORT)


def utc_now():
    return datetime.now(timezone.utc)


def log_event(event_type, message, extra=None):
    row = {
        "ts_utc": utc_now().isoformat(),
        "event_type": event_type,
        "message": message,
    }
    if extra:
        row.update(extra)

    df_new = pd.DataFrame([row])
    if LOG_FILE.exists():
        df_old = pd.read_csv(LOG_FILE)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(LOG_FILE, index=False)

    print(f"[{row['ts_utc']}] {event_type}: {message}")


def load_rules():
    rules = pd.read_csv(RULES_FILE)
    rules = rules[(rules["enabled"] == 1) & (rules["rule_name"] == RULE_NAME)].copy()
    if rules.empty:
        raise ValueError(f"No enabled rule found for {RULE_NAME}")
    return rules.iloc[0].to_dict()


def load_data():
    df = pd.read_csv(DATA_FILE)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def load_state():
    if not STATE_FILE.exists():
        return pd.DataFrame(columns=[
            "rule_name", "symbol", "signal_time", "position_ticket",
            "opened_at_utc", "expire_at_utc", "status", "mt5_comment"
        ])
    df = pd.read_csv(STATE_FILE)
    for c in ["signal_time", "opened_at_utc", "expire_at_utc"]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def save_state(df):
    df.to_csv(STATE_FILE, index=False)


def has_processed_bar(state, signal_time):
    if state.empty:
        return False
    m = (state["rule_name"] == RULE_NAME) & (state["signal_time"] == signal_time)
    return bool(m.any())


def get_symbol_info(symbol):
    info = mt5.symbol_info(symbol)
    return info


def point_distance_to_price(symbol, stop_points):
    if USE_RAW_INDEX_POINTS:
        return float(stop_points)

    info = get_symbol_info(symbol)
    if info is None:
        return float(stop_points) * DEFAULT_POINT_SIZE

    point = getattr(info, "point", None)
    if point is None:
        return float(stop_points) * DEFAULT_POINT_SIZE

    return float(stop_points) * float(point)


def get_open_proto_positions(symbol):
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return []

    out = []
    for p in positions:
        p_magic = getattr(p, "magic", None)
        p_comment = str(getattr(p, "comment", ""))
        if p_magic == MAGIC or RULE_NAME in p_comment:
            out.append(p)
    return out


def latest_closed_bar(df):
    now = utc_now()
    closed = df[df["time"] <= now - timedelta(hours=1)].copy()
    if closed.empty:
        return None
    return closed.iloc[-1]


def build_mt5_comment(signal_time):
    return f"{RULE_NAME}|{signal_time.isoformat()}"


def send_short(symbol, volume, stop_points, signal_time):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"No tick available for {symbol}")

    entry_price = float(tick.bid)
    stop_distance_price = point_distance_to_price(symbol, stop_points)
    sl = entry_price + stop_distance_price

    comment = build_mt5_comment(signal_time)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(volume),
        "type": mt5.ORDER_TYPE_SELL,
        "price": entry_price,
        "sl": sl,
        "deviation": int(DEVIATION),
        "magic": int(MAGIC),
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    if DRY_RUN:
        log_event("DRY_OPEN", "Would send short order", {
            "symbol": symbol,
            "volume": volume,
            "entry_price": entry_price,
            "sl": sl,
            "signal_time": signal_time.isoformat(),
            "request": str(request),
        })
        return {"retcode": "DRY_RUN", "order": None, "comment": comment}

    result = mt5.order_send(request)
    return {
        "retcode": getattr(result, "retcode", None),
        "order": getattr(result, "order", None),
        "comment": comment,
        "raw_result": result,
    }


def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    if tick is None:
        raise RuntimeError(f"No tick available for {position.symbol}")

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": float(position.volume),
        "type": mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": float(tick.ask),
        "deviation": int(DEVIATION),
        "magic": int(MAGIC),
        "comment": f"{RULE_NAME}|timed_exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    if DRY_RUN:
        log_event("DRY_CLOSE", "Would close position on timed exit", {
            "symbol": position.symbol,
            "position_ticket": getattr(position, "ticket", None),
            "request": str(request),
        })
        return {"retcode": "DRY_RUN", "raw_result": None}

    result = mt5.order_send(request)
    return {
        "retcode": getattr(result, "retcode", None),
        "raw_result": result,
    }


def reconcile_time_exits(state):
    open_positions = get_open_proto_positions(SYMBOL)
    if not open_positions:
        return state

    for pos in open_positions:
        row = state[
            (state["status"] == "open") &
            (
                (state["position_ticket"].astype(str) == str(getattr(pos, "ticket", None))) |
                (state["mt5_comment"].astype(str) == str(getattr(pos, "comment", "")))
            )
        ]

        if row.empty:
            continue

        expire_at = row.iloc[0]["expire_at_utc"]
        if pd.isna(expire_at):
            continue

        if utc_now() >= expire_at.to_pydatetime():
            result = close_position(pos)
            log_event("TIME_EXIT_CHECK", "Timed exit triggered", {
                "position_ticket": getattr(pos, "ticket", None),
                "retcode": result.get("retcode"),
            })

            if DRY_RUN or result.get("retcode") == getattr(mt5, "TRADE_RETCODE_DONE", None):
                idx = row.index[0]
                state.loc[idx, "status"] = "closed"

    return state


def maybe_open_new_trade(state, rule, df):
    bar = latest_closed_bar(df)
    if bar is None:
        log_event("NO_BAR", "No closed bar available yet")
        return state

    signal_time = pd.Timestamp(bar["time"])

    if has_processed_bar(state, signal_time):
        return state

    signal_value = int(bar.get(RULE_NAME, 0))
    if signal_value != 1:
        new_row = pd.DataFrame([{
            "rule_name": RULE_NAME,
            "symbol": SYMBOL,
            "signal_time": signal_time,
            "position_ticket": None,
            "opened_at_utc": utc_now(),
            "expire_at_utc": None,
            "status": "no_signal",
            "mt5_comment": None,
        }])
        log_event("NO_SIGNAL", "Latest closed bar has no proto signal", {
            "signal_time": signal_time.isoformat(),
        })
        return pd.concat([state, new_row], ignore_index=True)

    open_positions = get_open_proto_positions(SYMBOL)
    if open_positions:
        new_row = pd.DataFrame([{
            "rule_name": RULE_NAME,
            "symbol": SYMBOL,
            "signal_time": signal_time,
            "position_ticket": None,
            "opened_at_utc": utc_now(),
            "expire_at_utc": None,
            "status": "skipped_existing_position",
            "mt5_comment": None,
        }])
        log_event("SKIP_OPEN", "Signal present but existing proto position already open", {
            "signal_time": signal_time.isoformat(),
            "open_count": len(open_positions),
        })
        return pd.concat([state, new_row], ignore_index=True)

    try:
        result = send_short(
            symbol=SYMBOL,
            volume=LOT_SIZE,
            stop_points=float(rule["stop_points"]),
            signal_time=signal_time.to_pydatetime()
        )
    except Exception as e:
        new_row = pd.DataFrame([{
            "rule_name": RULE_NAME,
            "symbol": SYMBOL,
            "signal_time": signal_time,
            "position_ticket": None,
            "opened_at_utc": utc_now(),
            "expire_at_utc": None,
            "status": f"open_exception_{type(e).__name__}",
            "mt5_comment": None,
        }])
        log_event("OPEN_EXCEPTION", str(e), {
            "signal_time": signal_time.isoformat(),
        })
        return pd.concat([state, new_row], ignore_index=True)

    retcode = result.get("retcode")
    comment = result.get("comment")
    log_event("OPEN_RESULT", "Processed open request", {
        "signal_time": signal_time.isoformat(),
        "retcode": retcode,
        "position_ticket": result.get("order"),
    })

    if DRY_RUN or retcode == getattr(mt5, "TRADE_RETCODE_DONE", None):
        expire_at = utc_now() + timedelta(hours=float(rule["time_exit_h"]))
        status = "open"
        position_ticket = result.get("order")
    else:
        expire_at = None
        status = f"open_failed_{retcode}"
        position_ticket = None

    new_row = pd.DataFrame([{
        "rule_name": RULE_NAME,
        "symbol": SYMBOL,
        "signal_time": signal_time,
        "position_ticket": position_ticket,
        "opened_at_utc": utc_now(),
        "expire_at_utc": expire_at,
        "status": status,
        "mt5_comment": comment,
    }])

    return pd.concat([state, new_row], ignore_index=True)


def ensure_connection():
    ok = mt5.initialize()
    if not ok:
        last_error = None
        try:
            last_error = mt5.last_error()
        except Exception:
            last_error = "unknown"
        raise RuntimeError(f"mt5.initialize() failed: {last_error}")

    info = mt5.terminal_info()
    log_event("INIT", "MT5 connection initialized", {
        "terminal_info": str(info),
        "host": MT5_HOST,
        "port": MT5_PORT,
        "dry_run": DRY_RUN,
        "symbol": SYMBOL,
    })


def main():
    ensure_connection()

    try:
        while True:
            try:
                rule = load_rules()
                df = load_data()
                state = load_state()

                state = reconcile_time_exits(state)
                state = maybe_open_new_trade(state, rule, df)

                save_state(state)

            except Exception as loop_error:
                log_event("LOOP_ERROR", str(loop_error), {})
            time.sleep(POLL_SECONDS)
    finally:
        try:
            mt5.shutdown()
            log_event("SHUTDOWN", "MT5 connection shut down")
        except Exception as e:
            log_event("SHUTDOWN_ERROR", str(e))


if __name__ == "__main__":
    main()
