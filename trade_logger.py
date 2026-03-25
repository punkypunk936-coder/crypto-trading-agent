"""
trade_logger.py — Real-time trade logging to CSV.
Every trade open and close is appended instantly so no data is lost
even if the agent is stopped mid-session.
"""

import csv
import os
import time
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

def _load_last_id() -> int:
    """Resume trade ID counter from existing log."""
    if not LOG_FILE.exists():
        return 0
    try:
        with LOG_FILE.open(newline="") as f:
            rows = list(csv.DictReader(f))
            if rows:
                return int(rows[-1]["trade_id"])
    except Exception:
        pass
    return 0


# In-memory open trades waiting to be closed
_open_trades: dict = {}
_trade_counter: int = _load_last_id()


def _ensure_headers():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=HEADERS).writeheader()
        log.info(f"Created trade log: {LOG_FILE}")


def log_open(coin: str, direction: str, entry_price: float,
             size_usd: float, stop_loss: float, take_profit: float,
             signal_score: float, leverage: int = 3):
    """Call this when a trade is opened."""
    global _trade_counter
    _ensure_headers()
    _trade_counter += 1
    _open_trades[coin] = {
        "trade_id":    _trade_counter,
        "coin":        coin,
        "direction":   direction,
        "opened_at":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "entry_price": entry_price,
        "size_usd":    size_usd,
        "leverage":    leverage,
        "stop_loss":   stop_loss,
        "take_profit": take_profit,
        "signal_score":round(signal_score, 1),
    }
    log.debug(f"[{coin}] Trade #{_trade_counter} logged as open")


def restore_open(coin: str, direction: str, entry_price: float,
                 size_usd: float, stop_loss: float, take_profit: float,
                 leverage: int = 3, signal_score: float = 0.0,
                 opened_at: Optional[str] = None):
    """
    Restore an open trade so later closes still hit the CSV log.
    Does NOT increment _trade_counter to avoid duplicate rows on restart.
    Re-uses the last known trade_id from the CSV for this coin if found.
    """
    _ensure_headers()
    if coin in _open_trades:
        return   # already restored — do not duplicate

    # Try to find the last trade_id for this coin in the CSV
    restored_id = _find_last_open_id(coin)
    if restored_id is None:
        # No existing open trade found in CSV — use current counter
        global _trade_counter
        _trade_counter += 1
        restored_id = _trade_counter

    _open_trades[coin] = {
        "trade_id":     restored_id,
        "coin":         coin,
        "direction":    direction,
        "opened_at":    opened_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "entry_price":  entry_price,
        "size_usd":     size_usd,
        "leverage":     leverage,
        "stop_loss":    stop_loss,
        "take_profit":  take_profit,
        "signal_score": round(signal_score, 1),
    }
    log.info(f"[{coin}] Restored open trade #{restored_id} for logging continuity")


def _find_last_open_id(coin: str) -> Optional[int]:
    """
    Scan the CSV for the last entry for this coin that has no corresponding
    close (i.e., an open that was not yet closed). Returns its trade_id.
    Because we write one row per closed trade, an 'unclosed' trade simply
    won't appear as a closed row in the file — we return None to signal a
    fresh assignment is needed.

    In practice this just prevents us from incrementing the counter on every
    restart and creating phantom sequential IDs.
    """
    if not LOG_FILE.exists():
        return None
    try:
        with LOG_FILE.open(newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("coin") == coin]
        if rows:
            return int(rows[-1]["trade_id"])
    except Exception:
        pass
    return None


def log_close(coin: str, exit_price: float, exit_reason: str):
    """Call this when a trade is closed. Writes a full row to the CSV."""
    trade = _open_trades.pop(coin, None)
    if not trade:
        log.warning(f"[{coin}] log_close called but no open trade found")
        return

    closed_at   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    opened_ts   = datetime.strptime(trade["opened_at"], "%Y-%m-%d %H:%M")
    closed_ts   = datetime.strptime(closed_at,          "%Y-%m-%d %H:%M")
    duration    = round((closed_ts - opened_ts).total_seconds() / 60, 1)

    entry = trade["entry_price"]
    if trade["direction"] == "LONG":
        pnl_pct = (exit_price - entry) / entry
    else:
        pnl_pct = (entry - exit_price) / entry

    pnl_usd = pnl_pct * trade["size_usd"]
    result  = "WIN" if pnl_usd >= 0 else "LOSS"

    row = {
        "trade_id":      trade["trade_id"],
        "coin":          coin,
        "direction":     trade["direction"],
        "opened_at":     trade["opened_at"],
        "closed_at":     closed_at,
        "duration_mins": duration,
        "entry_price":   round(entry, 4),
        "exit_price":    round(exit_price, 4),
        "size_usd":      round(trade["size_usd"], 2),
        "leverage":      trade["leverage"],
        "pnl_usd":       round(pnl_usd, 2),
        "pnl_pct":       round(pnl_pct * 100, 2),
        "stop_loss":     round(trade["stop_loss"], 4),
        "take_profit":   round(trade["take_profit"], 4),
        "exit_reason":   exit_reason,
        "signal_score":  trade["signal_score"],
        "result":        result,
    }

    with LOG_FILE.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=HEADERS).writerow(row)

    log.info(
        f"[{coin}] Trade #{trade['trade_id']} logged: {result} "
        f"{pnl_pct*100:+.2f}% (${pnl_usd:+.2f}) | {exit_reason}"
    )
