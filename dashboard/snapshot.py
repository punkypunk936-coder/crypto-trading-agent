"""
dashboard/snapshot.py
Shared dashboard payload builder used by the local Flask UI and remote sync.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable


def default_state() -> dict:
    return {
        "status": "offline",
        "last_cycle": None,
        "cycle_number": 0,
        "portfolio_usd": 0,
        "available_usd": 0,
        "positions": [],
        "signals": {},
        "pending_orders": [],
        "sentiment": {},
        "mode": "unknown",
    }


def default_control() -> dict:
    return {
        "kill": {
            "active": False,
            "reason": "",
            "requested_at": None,
            "acknowledged_at": None,
        }
    }


def normalize_control(control: Any) -> dict:
    base = default_control()
    if not isinstance(control, dict):
        return base
    kill = control.get("kill")
    if isinstance(kill, dict):
        base["kill"].update({
            "active": bool(kill.get("active", False)),
            "reason": str(kill.get("reason", "") or ""),
            "requested_at": kill.get("requested_at"),
            "acknowledged_at": kill.get("acknowledged_at"),
        })
    return base


def calc_stats(trades: Iterable[dict] | None) -> dict:
    safe_trades = list(trades or [])
    if not safe_trades:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "best": 0,
            "worst": 0,
        }

    closed = []
    for trade in safe_trades:
        try:
            if trade.get("exit_price") and float(trade.get("exit_price", 0)) > 0:
                closed.append(trade)
        except Exception:
            continue

    if not closed:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "best": 0,
            "worst": 0,
        }

    pnls = []
    for trade in closed:
        try:
            pnls.append(float(trade.get("pnl_usd", 0)))
        except Exception:
            pnls.append(0.0)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "total": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "total_pnl": round(sum(pnls), 2),
        "avg_win": round(sum(wins) / len(wins) if wins else 0, 2),
        "avg_loss": round(sum(losses) / len(losses) if losses else 0, 2),
        "best": round(max(pnls), 2),
        "worst": round(min(pnls), 2),
    }


def runtime_status(state: dict) -> dict:
    last_cycle = state.get("last_cycle")
    stale = False
    age_seconds = None
    interval = int(((state.get("config") or {}).get("check_interval_seconds")) or 120)
    if isinstance(last_cycle, str):
        try:
            age_seconds = int((datetime.now() - datetime.strptime(last_cycle, "%Y-%m-%d %H:%M:%S")).total_seconds())
            stale = age_seconds > max(interval * 2, 240)
        except Exception:
            age_seconds = None
    return {
        "stale": stale,
        "state_age_seconds": age_seconds,
    }


def decision_summary(state: dict) -> dict:
    signals = (state or {}).get("signals") or {}
    summary = {
        "long_count": 0,
        "short_count": 0,
        "flat_count": 0,
        "tradable_count": 0,
        "tradable_active_count": 0,
        "lead": None,
    }
    lead_rank = (-1, -1, -1.0)

    for coin, sig in signals.items():
        action = str(sig.get("action") or "FLAT").upper()
        if action not in {"LONG", "SHORT", "FLAT"}:
            action = "FLAT"
        summary[f"{action.lower()}_count"] += 1

        execution_mode = sig.get("execution_mode") or "observation_only"
        is_tradable = execution_mode == "tradable"
        if is_tradable:
            summary["tradable_count"] += 1
        if is_tradable and action != "FLAT":
            summary["tradable_active_count"] += 1

        try:
            strength = abs(float(sig.get("score", 50.0)) - 50.0)
        except Exception:
            strength = 0.0

        rank = (
            1 if action != "FLAT" else 0,
            1 if is_tradable else 0,
            strength,
        )
        if rank > lead_rank:
            lead_rank = rank
            summary["lead"] = {
                "coin": coin,
                "action": action,
                "score": sig.get("score", 50.0),
                "confidence": sig.get("confidence", "LOW"),
                "execution_mode": execution_mode,
                "reason": sig.get("decision_reason") or sig.get("reason") or sig.get("flat_reason") or "",
            }

    return summary


def augment_state(state: Any) -> dict:
    safe_state = dict(state or {})
    merged = default_state()
    merged.update(safe_state)
    merged["positions_count"] = len(merged.get("positions") or [])
    merged["decision_summary"] = decision_summary(merged)
    return merged


def server_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_dashboard_snapshot(
    state: Any,
    trades: Iterable[dict] | None,
    control: Any = None,
    *,
    server_timestamp: str | None = None,
) -> dict:
    safe_trades = list(trades or [])
    shaped_state = augment_state(state)
    return {
        "state": shaped_state,
        "trades": safe_trades[-50:][::-1],
        "stats": calc_stats(safe_trades),
        "control": normalize_control(control),
        "runtime": runtime_status(shaped_state),
        "server_time": server_timestamp or server_time(),
    }
