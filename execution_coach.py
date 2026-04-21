"""
execution_coach.py — separate execution specialist for live entries.

The strategy decides whether a trade idea is good.
The execution coach decides how it should be expressed:
  • PASSIVE    → rest a limit / maker order
  • AGGRESSIVE → sweep now because the window is clean
  • CHASE      → only allowed on urgent breakouts that are still tight
  • SKIP       → price stretched or execution quality is not good enough
"""

from __future__ import annotations

from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_bool(value: Any) -> bool:
    return bool(value)


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _normalize_mode(value: Any) -> str:
    mode = _safe_str(value, "market").lower()
    if mode not in {"market", "limit", "maker_limit"}:
        return "market"
    return mode


def _breakout_state(signal_snapshot: Mapping[str, Any], orderbook_signal: Any = None) -> str:
    if orderbook_signal is not None:
        breakout = _safe_str(getattr(orderbook_signal, "breakout_state", ""))
        if breakout:
            return breakout.upper()
    return _safe_str(
        (signal_snapshot or {}).get("orderbook_breakout_state")
        or (signal_snapshot or {}).get("orderbook_intracycle_breakout_state"),
        "NONE",
    ).upper()


def _tight_distance_bps(current_price: float, anchor_price: float) -> float:
    if current_price <= 0 or anchor_price <= 0:
        return 0.0
    return abs(current_price - anchor_price) / max(anchor_price, 1e-9) * 10_000.0


def _best_passive_price(
    *,
    direction: str,
    signal_snapshot: Mapping[str, Any],
    execution_plan: Mapping[str, Any],
    execution_quality: Mapping[str, Any],
    orderbook_signal: Any = None,
) -> float:
    quality_price = _safe_float(execution_quality.get("passive_limit_price"))
    if quality_price > 0:
        return quality_price

    plan_price = _safe_float(execution_plan.get("limit_price") or execution_plan.get("entry_price"))
    if plan_price > 0 and _normalize_mode(execution_plan.get("mode")) in {"limit", "maker_limit"}:
        return plan_price

    if orderbook_signal is not None:
        bid = _safe_float(getattr(orderbook_signal, "best_bid", 0.0))
        ask = _safe_float(getattr(orderbook_signal, "best_ask", 0.0))
        if direction == "LONG" and bid > 0:
            return bid
        if direction == "SHORT" and ask > 0:
            return ask

    best_bid = _safe_float((signal_snapshot or {}).get("orderbook_best_bid"))
    best_ask = _safe_float((signal_snapshot or {}).get("orderbook_best_ask"))
    if direction == "LONG" and best_bid > 0:
        return best_bid
    if direction == "SHORT" and best_ask > 0:
        return best_ask
    return 0.0


def _make_plan(mode: str, *, entry_price: float, limit_price: float = 0.0, max_wait_cycles: int = 6, reason: str = "") -> dict:
    plan = {
        "mode": mode,
        "entry_price": round(entry_price, 6) if entry_price > 0 else 0.0,
        "limit_price": round(limit_price, 6) if limit_price > 0 else 0.0,
        "max_wait_cycles": int(max_wait_cycles or 0),
        "reason": reason,
    }
    if mode == "market":
        plan["limit_price"] = 0.0
    return plan


def decide_execution(
    cfg,
    *,
    coin: str,
    signal_snapshot: Mapping[str, Any] | None,
    order,
    execution_quality: Mapping[str, Any] | None,
    orderbook_signal: Any = None,
) -> dict:
    snap = dict(signal_snapshot or {})
    quality = dict(execution_quality or {})
    direction = _safe_str(snap.get("action"), "").upper()
    base_plan = dict(snap.get("execution_plan") or {})
    expectancy = dict(snap.get("expectancy") or {})
    trade_plan = dict(snap.get("trade_plan") or {})
    thesis = dict(snap.get("thesis") or {})

    current_price = _safe_float(
        snap.get("live_price")
        or snap.get("price")
        or getattr(order, "price", 0.0)
        or base_plan.get("entry_price")
    )
    planned_entry = _safe_float(base_plan.get("entry_price") or getattr(order, "price", 0.0) or current_price)
    plan_mode = _normalize_mode(base_plan.get("mode"))
    max_wait_cycles = int(base_plan.get("max_wait_cycles") or getattr(cfg, "execution_limit_timeout_cycles", 6) or 6)
    breakout_state = _breakout_state(snap, orderbook_signal)
    confirmed_breakout = breakout_state in {
        "CONFIRMED_BULLISH_BREAKOUT",
        "CONFIRMED_BEARISH_BREAKDOWN",
        "PERSISTENT_BULLISH_BREAKOUT",
        "PERSISTENT_BEARISH_BREAKDOWN",
    }
    persistent_breakout = breakout_state in {
        "PERSISTENT_BULLISH_BREAKOUT",
        "PERSISTENT_BEARISH_BREAKDOWN",
    }
    probability = _safe_float(expectancy.get("probability"), 0.50)
    expectancy_score = _safe_float(expectancy.get("score"), _safe_float(snap.get("score"), 50.0))
    quality_score = _safe_float(quality.get("score"), 0.0)
    stretch_bps = _tight_distance_bps(current_price, planned_entry)
    risk_reward = _safe_float(trade_plan.get("risk_reward_ratio"))
    support_defense_long = _safe_bool(thesis.get("support_defense_long"))
    passive_price = _best_passive_price(
        direction=direction,
        signal_snapshot=snap,
        execution_plan=base_plan,
        execution_quality=quality,
        orderbook_signal=orderbook_signal,
    )

    chase_ready = (
        confirmed_breakout
        and quality_score >= _safe_float(getattr(cfg, "execution_coach_min_quality_score", 74.0), 74.0)
        and probability >= _safe_float(getattr(cfg, "execution_coach_min_breakout_probability", 0.66), 0.66)
        and expectancy_score >= _safe_float(getattr(cfg, "execution_coach_min_breakout_expectancy_score", 66.0), 66.0)
    )
    urgency_score = max(
        0.0,
        min(
            100.0,
            45.0
            + (probability - 0.50) * 80.0
            + (quality_score - 70.0) * 0.6
            + (8.0 if confirmed_breakout else 0.0)
            + (4.0 if persistent_breakout else 0.0)
            - stretch_bps * 0.45,
        ),
    )

    if direction not in {"LONG", "SHORT"}:
        return {
            "enabled": bool(getattr(cfg, "execution_coach_enabled", True)),
            "verdict": "SKIP",
            "style": "SKIP",
            "mode": "skip",
            "summary": "No directional thesis is live, so the execution coach stays idle.",
            "reason": "no directional thesis",
            "stretch_bps": round(stretch_bps, 3),
            "urgency_score": round(urgency_score, 2),
            "execution_plan": {},
            "blockers": ["no directional thesis"],
        }

    if not bool(getattr(cfg, "execution_coach_enabled", True)):
        baseline = "PASSIVE" if plan_mode in {"limit", "maker_limit"} else "AGGRESSIVE"
        return {
            "enabled": False,
            "verdict": baseline,
            "style": baseline,
            "mode": plan_mode,
            "summary": _safe_str(base_plan.get("reason"), "Execution coach disabled"),
            "reason": _safe_str(base_plan.get("reason"), "Execution coach disabled"),
            "stretch_bps": round(stretch_bps, 3),
            "urgency_score": round(urgency_score, 2),
            "execution_plan": dict(base_plan),
            "blockers": [],
        }

    passive_hold_bps = _safe_float(getattr(cfg, "execution_coach_passive_hold_distance_bps", 18.0), 18.0)
    aggressive_max_bps = _safe_float(getattr(cfg, "execution_coach_aggressive_max_stretch_bps", 10.0), 10.0)
    max_chase_bps = _safe_float(getattr(cfg, "execution_coach_max_chase_bps", 32.0), 32.0)
    skip_stretch_bps = _safe_float(getattr(cfg, "execution_coach_skip_stretch_bps", 48.0), 48.0)

    if not _safe_bool(quality.get("permitted", True)):
        if _safe_bool(quality.get("prefer_passive_entry")) and passive_price > 0:
            reason = _safe_str(
                quality.get("passive_summary") or quality.get("summary"),
                "Execution quality prefers a passive maker entry.",
            )
            plan = _make_plan(
                "maker_limit",
                entry_price=passive_price,
                limit_price=passive_price,
                max_wait_cycles=max_wait_cycles,
                reason=reason,
            )
            return {
                "enabled": True,
                "verdict": "PASSIVE",
                "style": "PASSIVE",
                "mode": "maker_limit",
                "summary": reason,
                "reason": reason,
                "stretch_bps": round(stretch_bps, 3),
                "urgency_score": round(urgency_score, 2),
                "execution_plan": plan,
                "blockers": [],
            }
        reason = _safe_str(quality.get("summary"), "Execution quality is not clean enough to trade.")
        return {
            "enabled": True,
            "verdict": "SKIP",
            "style": "SKIP",
            "mode": "skip",
            "summary": reason,
            "reason": reason,
            "stretch_bps": round(stretch_bps, 3),
            "urgency_score": round(urgency_score, 2),
            "execution_plan": {},
            "blockers": list(quality.get("blockers") or [reason])[:4],
        }

    if plan_mode in {"limit", "maker_limit"}:
        if passive_price > 0 and stretch_bps <= passive_hold_bps:
            reason = _safe_str(
                base_plan.get("reason"),
                "The setup is still close enough to lean passive and let price come in.",
            )
            passive_mode = plan_mode if plan_mode in {"limit", "maker_limit"} else "maker_limit"
            plan = _make_plan(
                passive_mode,
                entry_price=passive_price,
                limit_price=passive_price,
                max_wait_cycles=max_wait_cycles,
                reason=reason,
            )
            return {
                "enabled": True,
                "verdict": "PASSIVE",
                "style": "PASSIVE",
                "mode": passive_mode,
                "summary": reason,
                "reason": reason,
                "stretch_bps": round(stretch_bps, 3),
                "urgency_score": round(urgency_score, 2),
                "execution_plan": plan,
                "blockers": [],
            }
        if chase_ready and stretch_bps <= max_chase_bps:
            reason = (
                f"{coin} is running cleanly out of the passive zone, so the coach upgrades the order to a chase "
                "instead of letting a winner go without us."
            )
            plan = _make_plan(
                "market",
                entry_price=current_price or planned_entry,
                max_wait_cycles=0,
                reason=reason,
            )
            return {
                "enabled": True,
                "verdict": "CHASE",
                "style": "CHASE",
                "mode": "market",
                "summary": reason,
                "reason": reason,
                "stretch_bps": round(stretch_bps, 3),
                "urgency_score": round(urgency_score, 2),
                "execution_plan": plan,
                "blockers": [],
            }
        reason = (
            f"{coin} moved {stretch_bps:.1f}bps away from the planned passive entry without enough urgency to justify "
            "chasing it."
        )
        return {
            "enabled": True,
            "verdict": "SKIP",
            "style": "SKIP",
            "mode": "skip",
            "summary": reason,
            "reason": reason,
            "stretch_bps": round(stretch_bps, 3),
            "urgency_score": round(urgency_score, 2),
            "execution_plan": {},
            "blockers": [reason],
        }

    if stretch_bps <= aggressive_max_bps:
        reason = _safe_str(
            base_plan.get("reason"),
            "Execution is clean enough to hit now without overpaying for urgency.",
        )
        plan = _make_plan(
            "market",
            entry_price=current_price or planned_entry,
            max_wait_cycles=0,
            reason=reason,
        )
        return {
            "enabled": True,
            "verdict": "AGGRESSIVE",
            "style": "AGGRESSIVE",
            "mode": "market",
            "summary": reason,
            "reason": reason,
            "stretch_bps": round(stretch_bps, 3),
            "urgency_score": round(urgency_score, 2),
            "execution_plan": plan,
            "blockers": [],
        }

    if chase_ready and stretch_bps <= max_chase_bps:
        reason = (
            f"{coin} is breaking with enough quality and expectancy to justify a controlled chase "
            f"({stretch_bps:.1f}bps from the ideal entry)."
        )
        plan = _make_plan(
            "market",
            entry_price=current_price or planned_entry,
            max_wait_cycles=0,
            reason=reason,
        )
        return {
            "enabled": True,
            "verdict": "CHASE",
            "style": "CHASE",
            "mode": "market",
            "summary": reason,
            "reason": reason,
            "stretch_bps": round(stretch_bps, 3),
            "urgency_score": round(urgency_score, 2),
            "execution_plan": plan,
            "blockers": [],
        }

    if passive_price > 0 and support_defense_long and direction == "LONG" and stretch_bps <= skip_stretch_bps:
        reason = (
            "The reclaim is not urgent enough to chase. The coach keeps the bot calm and rests a passive order "
            "near defended support instead."
        )
        plan = _make_plan(
            "maker_limit",
            entry_price=passive_price,
            limit_price=passive_price,
            max_wait_cycles=max_wait_cycles,
            reason=reason,
        )
        return {
            "enabled": True,
            "verdict": "PASSIVE",
            "style": "PASSIVE",
            "mode": "maker_limit",
            "summary": reason,
            "reason": reason,
            "stretch_bps": round(stretch_bps, 3),
            "urgency_score": round(urgency_score, 2),
            "execution_plan": plan,
            "blockers": [],
        }

    reason = (
        f"{coin} is too stretched ({stretch_bps:.1f}bps) for a calm entry, and the breakout urgency is not strong enough "
        "to justify chasing it."
    )
    return {
        "enabled": True,
        "verdict": "SKIP",
        "style": "SKIP",
        "mode": "skip",
        "summary": reason,
        "reason": reason,
        "stretch_bps": round(stretch_bps, 3),
        "urgency_score": round(urgency_score, 2),
        "execution_plan": {},
        "blockers": [reason],
    }
