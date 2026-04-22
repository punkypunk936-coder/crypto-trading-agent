"""
asset_state_machine.py — derive a clean per-asset trading lifecycle state.

This gives the agent and dashboard a more honest operating model than
"LONG/SHORT/FLAT" alone. A setup can be armed, confirming, blocked on
execution, pending as a passive order, or already live.
"""

from __future__ import annotations

from typing import Any, Mapping


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def derive_major_catalyst_watch(signal_snapshot: Mapping[str, Any] | None) -> dict:
    snap = dict(signal_snapshot or {})
    if _safe_str(snap.get("instrument_type"), "crypto").lower() != "equity":
        return {"active": False}

    catalyst_score = _safe_float(snap.get("news_catalyst_score"))
    if catalyst_score < 3.0:
        return {"active": False}

    action = _safe_str(snap.get("action"), "FLAT").upper()
    candidate_action = _safe_str(snap.get("thesis_candidate_action"), action).upper()
    tradable = _safe_str(snap.get("execution_mode"), "observation_only") == "tradable"
    thesis_permitted = bool(snap.get("thesis_permitted", False))
    if tradable and action in {"LONG", "SHORT"} and thesis_permitted:
        return {"active": False}

    news_score = _safe_float(snap.get("news_score"), 50.0)
    market_bias = _safe_str(snap.get("market_map_bias"), "NEUTRAL").upper()
    narrative_bias = _safe_str(snap.get("narrative_headline_bias"), "NEUTRAL").upper()
    reclaim_confirmed = bool(snap.get("market_map_reclaim_confirmed"))
    live_reclaim = bool(snap.get("market_map_live_reclaim"))
    reclaim_lost = bool(snap.get("market_map_reclaim_lost"))
    breakdown_confirmed = bool(snap.get("market_map_breakdown_confirmed"))
    live_breakdown = bool(snap.get("market_map_live_breakdown"))
    breakout_states = {
        _safe_str(snap.get("orderbook_breakout_state"), "NONE").upper(),
        _safe_str(snap.get("orderbook_intracycle_breakout_state"), "NONE").upper(),
    }
    bullish_breakout_live = bool(
        breakout_states & {"PROBING_BULLISH_BREAKOUT", "CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"}
    )
    bearish_breakout_live = bool(
        breakout_states & {"PROBING_BEARISH_BREAKDOWN", "CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"}
    )

    direction = ""
    bullish_context = (
        candidate_action == "LONG"
        or market_bias == "BULLISH"
        or narrative_bias == "BULLISH"
        or news_score >= 55.0
        or reclaim_confirmed
        or bullish_breakout_live
    )
    bearish_context = (
        candidate_action == "SHORT"
        or market_bias == "BEARISH"
        or narrative_bias == "BEARISH"
        or news_score <= 45.0
        or breakdown_confirmed
        or bearish_breakout_live
    )
    if bullish_context and not bearish_context:
        direction = "LONG"
    elif bearish_context and not bullish_context:
        direction = "SHORT"
    elif candidate_action in {"LONG", "SHORT"}:
        direction = candidate_action
    else:
        return {"active": False}

    if direction == "LONG":
        trigger_price = _safe_float(
            snap.get("market_map_nearest_resistance")
            or snap.get("daily_breakout_level")
            or snap.get("price")
            or snap.get("live_price")
        )
        if reclaim_confirmed and not live_reclaim:
            next_unblock = (
                f"Major catalyst watch: hold back above {trigger_price:,.2f} and keep the reclaim to unlock the long."
                if trigger_price > 0
                else "Major catalyst watch: reclaim the trigger again and keep it defended before buying."
            )
            watch_type = "RECLAIM_RETAKE"
        elif bool(snap.get("market_map_block_longs")):
            next_unblock = (
                f"Major catalyst watch: reclaim above {trigger_price:,.2f} on a clean hold before buying."
                if trigger_price > 0
                else "Major catalyst watch: wait for a clean reclaim before buying."
            )
            watch_type = "WAIT_RECLAIM"
        elif bullish_breakout_live and not live_reclaim:
            next_unblock = (
                f"Major catalyst watch: breakout pressure is live, but it still needs a defended reclaim above {trigger_price:,.2f}."
                if trigger_price > 0
                else "Major catalyst watch: breakout pressure is live, but it still needs a defended reclaim."
            )
            watch_type = "BREAKOUT_PULLBACK"
        elif candidate_action == "LONG" and market_bias == "BULLISH":
            next_unblock = (
                f"Major catalyst watch: the catalyst is strong, but price still needs to accept above {trigger_price:,.2f}."
                if trigger_price > 0
                else "Major catalyst watch: the catalyst is strong, but price still needs a clean reclaim-retake."
            )
            watch_type = "THESIS_WAIT"
        else:
            return {"active": False}
    else:
        trigger_price = _safe_float(
            snap.get("market_map_nearest_support")
            or snap.get("daily_breakdown_level")
            or snap.get("price")
            or snap.get("live_price")
        )
        if breakdown_confirmed and not live_breakdown:
            next_unblock = (
                f"Major catalyst watch: lose {trigger_price:,.2f} again and keep the breakdown to unlock the short."
                if trigger_price > 0
                else "Major catalyst watch: re-break support and keep it lost before shorting."
            )
            watch_type = "BREAKDOWN_RETAKE"
        elif bool(snap.get("market_map_block_shorts")):
            next_unblock = (
                f"Major catalyst watch: break below {trigger_price:,.2f} on a clean hold before shorting."
                if trigger_price > 0
                else "Major catalyst watch: wait for a clean breakdown before shorting."
            )
            watch_type = "WAIT_BREAKDOWN"
        elif bearish_breakout_live and not live_breakdown:
            next_unblock = (
                f"Major catalyst watch: breakdown pressure is live, but it still needs acceptance below {trigger_price:,.2f}."
                if trigger_price > 0
                else "Major catalyst watch: breakdown pressure is live, but it still needs acceptance below support."
            )
            watch_type = "BREAKDOWN_PULLBACK"
        elif candidate_action == "SHORT" and market_bias == "BEARISH":
            next_unblock = (
                f"Major catalyst watch: the catalyst is strong, but price still needs to accept below {trigger_price:,.2f}."
                if trigger_price > 0
                else "Major catalyst watch: the catalyst is strong, but price still needs a clean breakdown."
            )
            watch_type = "THESIS_WAIT"
        else:
            return {"active": False}

    return {
        "active": True,
        "direction": direction,
        "label": "Major catalyst watch",
        "watch_type": watch_type,
        "trigger_price": round(trigger_price, 6) if trigger_price > 0 else 0.0,
        "next_unblock_reason": next_unblock,
        "summary": _safe_str(snap.get("news_catalyst_summary")),
    }


def build_asset_state(
    signal_snapshot: Mapping[str, Any] | None,
    *,
    stage: str = "",
    current_position: str = "",
    pending_limit: bool = False,
) -> dict:
    snap = dict(signal_snapshot or {})
    stage_key = _safe_str(stage or snap.get("decision_stage")).lower()
    current_position = _safe_str(current_position).upper()
    action = _safe_str(snap.get("action"), "FLAT").upper()
    candidate_action = _safe_str(snap.get("thesis_candidate_action"), action).upper()
    tradable = _safe_str(snap.get("execution_mode"), "observation_only") == "tradable"
    reliability = dict(snap.get("data_reliability") or {})
    execution_quality = dict(snap.get("execution_quality") or {})
    next_unblock = ""
    state = "OBSERVING"
    label = "Observing"
    catalyst_watch = derive_major_catalyst_watch(snap)

    if current_position:
        state = "LIVE_POSITION"
        label = "Live position"
        next_unblock = "No unblock needed. The bot is managing the open trade."
    elif pending_limit or stage_key in {"limit_entry_placed", "entry_limit_already_pending", "passive_rescue_limit_placed"}:
        state = "PENDING_ENTRY"
        label = "Pending entry"
        next_unblock = "Waiting for the resting limit order to fill, cancel, or expire."
    elif stage_key == "signal_streak_wait":
        remaining = max(0, _safe_int(snap.get("streak_confirmation_remaining"), 1))
        state = "WAITING_CONFIRMATION"
        label = "Waiting confirmation"
        next_unblock = (
            f"Need {remaining} more confirming cycle(s) before the entry is allowed."
            if remaining
            else "Waiting for one more confirming cycle before entry."
        )
    elif stage_key in {"precision_cadence_block", "reversal_cooldown_block"}:
        state = "COOLDOWN"
        label = "Cooling down"
        next_unblock = _safe_str(
            snap.get("decision_reason") or snap.get("flat_reason"),
            "The setup family is cooling down before it can be used again.",
        )
    elif stage_key == "portfolio_correlation_block":
        state = "PORTFOLIO_GUARD"
        label = "Portfolio guard"
        next_unblock = _safe_str(
            snap.get("portfolio_guard_summary"),
            "The book is already leaning too hard into the same correlated theme.",
        )
    elif stage_key == "risk_rejected":
        state = "RISK_BLOCKED"
        label = "Risk blocked"
        next_unblock = _safe_str(
            snap.get("decision_reason") or snap.get("flat_reason"),
            "Sizing or exposure rules rejected the order.",
        )
    elif stage_key == "data_reliability_block" or (reliability and not bool(reliability.get("permitted", True))):
        state = "DATA_QUALITY_HOLD"
        label = "Data quality hold"
        next_unblock = _safe_str(
            reliability.get("summary") or snap.get("decision_reason") or snap.get("flat_reason"),
            "The supporting data is not reliable enough to trade right now.",
        )
    elif stage_key in {"execution_quality_block", "execution_coach_skip"} or (action in {"LONG", "SHORT"} and execution_quality and not bool(execution_quality.get("permitted", True))):
        if bool(execution_quality.get("prefer_passive_entry", False)):
            state = "PASSIVE_ENTRY"
            label = "Passive entry"
            next_unblock = _safe_str(
                execution_quality.get("passive_summary") or execution_quality.get("summary"),
                "A passive limit order is safer than sweeping market liquidity here.",
            )
        else:
            state = "EXECUTION_BLOCKED"
            label = "Execution blocked"
            next_unblock = _safe_str(
                snap.get("execution_coach_summary") or execution_quality.get("summary"),
                "Execution quality still needs cleaner spread, depth, or slippage.",
            )
    elif catalyst_watch.get("active"):
        state = "MAJOR_CATALYST_WATCH"
        label = _safe_str(catalyst_watch.get("label"), "Major catalyst watch")
        next_unblock = _safe_str(
            catalyst_watch.get("next_unblock_reason"),
            "A major catalyst is active, but the retake confirmation still has to print.",
        )
    elif not tradable:
        state = "OBSERVATION_ONLY"
        label = "Observation only"
        next_unblock = _safe_str(
            snap.get("mode_detail"),
            "The venue is not allowing execution on this market yet.",
        )
    elif action in {"LONG", "SHORT"} and bool(snap.get("thesis_permitted", False)):
        state = "EXECUTABLE"
        label = "Executable"
        next_unblock = "The thesis is live. The next clean execution window can open the trade."
    elif candidate_action in {"LONG", "SHORT"}:
        state = "ARMED"
        label = "Armed"
        next_unblock = _safe_str(
            (snap.get("thesis_blockers") or [None])[0]
            or (snap.get("expectancy_blockers") or [None])[0]
            or snap.get("decision_reason")
            or snap.get("flat_reason"),
            "The context is forming, but one more condition still needs to align.",
        )

    return {
        "state": state,
        "label": label,
        "next_unblock_reason": next_unblock,
        "stage": stage_key or "analysis",
    }
