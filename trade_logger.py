"""
trade_logger.py — Real-time trade logging to CSV.
Every trade open and close is appended instantly so no data is lost
even if the agent is stopped mid-session.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from logger import get_logger
from paths import TRADES_CSV

log = get_logger("trade_logger")

LOG_FILE = TRADES_CSV

HEADERS = [
    "trade_id", "coin", "direction",
    "opened_at", "closed_at", "duration_mins",
    "entry_price", "exit_price",
    "size_usd", "leverage",
    "pnl_usd", "pnl_pct",
    "stop_loss", "take_profit",
    "exit_reason", "signal_score",
    "result",
]


def _file_has_headers() -> bool:
    if not LOG_FILE.exists():
        return False
    try:
        with LOG_FILE.open(newline="") as f:
            first = next(csv.reader(f), [])
    except Exception:
        return False
    return list(first) == HEADERS


def _normalize_legacy_log() -> None:
    if not LOG_FILE.exists() or _file_has_headers():
        return

    rows = []
    try:
        with LOG_FILE.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == len(HEADERS):
                    rows.append(row)
    except Exception as exc:
        log.warning(f"Could not normalize legacy trade log: {exc}")
        return

    tmp_path = Path(f"{LOG_FILE}.tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        writer.writerows(rows)
    tmp_path.replace(LOG_FILE)
    log.warning(f"Normalized legacy trade log without headers: {LOG_FILE}")


def _ensure_headers() -> None:
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()
        log.info(f"Created trade log: {LOG_FILE}")
        return
    _normalize_legacy_log()
    if LOG_FILE.exists() and LOG_FILE.stat().st_size == 0:
        with LOG_FILE.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()


def _read_rows() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    _ensure_headers()
    try:
        with LOG_FILE.open(newline="") as f:
            return list(csv.DictReader(f))
    except Exception as exc:
        log.warning(f"Trade log read failed: {exc}")
        return []


def read_closed_trades(limit: Optional[int] = None) -> list[dict]:
    rows = _read_rows()
    if limit is None:
        return rows
    return rows[-limit:]


def _load_last_id() -> int:
    rows = _read_rows()
    ids = []
    for row in rows:
        try:
            ids.append(int(row.get("trade_id", 0) or 0))
        except Exception:
            continue
    return max(ids or [0])


# In-memory open trades waiting to be closed
_open_trades: dict = {}
_trade_counter: int = _load_last_id()


def log_open(
    coin: str,
    direction: str,
    entry_price: float,
    size_usd: float,
    stop_loss: float,
    take_profit: float,
    signal_score: float,
    leverage: int = 3,
) -> int:
    """Call this when a trade is opened."""
    global _trade_counter
    _ensure_headers()
    _trade_counter += 1
    _open_trades[coin] = {
        "trade_id": _trade_counter,
        "coin": coin,
        "direction": direction,
        "opened_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "entry_price": entry_price,
        "size_usd": size_usd,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "signal_score": round(signal_score, 1),
    }
    log.debug(f"[{coin}] Trade #{_trade_counter} logged as open")
    return _trade_counter


def restore_open(
    coin: str,
    direction: str,
    entry_price: float,
    size_usd: float,
    stop_loss: float,
    take_profit: float,
    leverage: int = 3,
    signal_score: float = 0.0,
    opened_at: Optional[str] = None,
) -> None:
    """
    Restore an open trade so later closes still hit the CSV log.

    Closed rows are immutable. A recovered open position therefore receives the
    next globally unique ID instead of reusing the last closed ID for that coin.
    """
    _ensure_headers()
    if coin in _open_trades:
        return

    global _trade_counter
    _trade_counter = max(_trade_counter, _load_last_id()) + 1
    restored_id = _trade_counter

    _open_trades[coin] = {
        "trade_id": restored_id,
        "coin": coin,
        "direction": direction,
        "opened_at": opened_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "entry_price": entry_price,
        "size_usd": size_usd,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "signal_score": round(signal_score, 1),
    }
    log.info(f"[{coin}] Restored open trade #{restored_id} for logging continuity")


def update_open(coin: str, entry_price: float, size_usd: float, stop_loss: float, take_profit: float) -> None:
    """Adjust the currently open trade after a confirmed scale-in."""
    trade = _open_trades.get(coin)
    if not trade:
        log.warning(f"[{coin}] update_open called but no open trade found")
        return
    trade["entry_price"] = entry_price
    trade["size_usd"] = size_usd
    trade["stop_loss"] = stop_loss
    trade["take_profit"] = take_profit
    log.info(
        f"[{coin}] Updated open trade #{trade['trade_id']} after scale-in: "
        f"entry=${entry_price:.2f} size=${size_usd:.2f}"
    )


def get_open_trade(coin: str) -> Optional[dict]:
    trade = _open_trades.get(coin)
    return dict(trade) if trade else None


def log_close(coin: str, exit_price: float, exit_reason: str) -> Optional[dict]:
    """Call this when a trade is closed. Writes a full row to the CSV."""
    trade = _open_trades.pop(coin, None)
    if not trade:
        log.warning(f"[{coin}] log_close called but no open trade found")
        return None

    closed_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    opened_ts = datetime.strptime(trade["opened_at"], "%Y-%m-%d %H:%M")
    closed_ts = datetime.strptime(closed_at, "%Y-%m-%d %H:%M")
    duration = round((closed_ts - opened_ts).total_seconds() / 60, 1)

    entry = trade["entry_price"]
    if trade["direction"] == "LONG":
        pnl_pct = (exit_price - entry) / entry
    else:
        pnl_pct = (entry - exit_price) / entry

    pnl_usd = pnl_pct * trade["size_usd"]
    result = "WIN" if pnl_usd >= 0 else "LOSS"

    row = {
        "trade_id": trade["trade_id"],
        "coin": coin,
        "direction": trade["direction"],
        "opened_at": trade["opened_at"],
        "closed_at": closed_at,
        "duration_mins": duration,
        "entry_price": round(entry, 4),
        "exit_price": round(exit_price, 4),
        "size_usd": round(trade["size_usd"], 2),
        "leverage": trade["leverage"],
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct * 100, 2),
        "stop_loss": round(trade["stop_loss"], 4),
        "take_profit": round(trade["take_profit"], 4),
        "exit_reason": exit_reason,
        "signal_score": trade["signal_score"],
        "result": result,
    }

    _ensure_headers()
    with LOG_FILE.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADERS).writerow(row)

    log.info(
        f"[{coin}] Trade #{trade['trade_id']} logged: {result} "
        f"{pnl_pct*100:+.2f}% (${pnl_usd:+.2f}) | {exit_reason}"
    )
    return row
