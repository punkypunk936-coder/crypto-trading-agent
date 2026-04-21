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

    if current_position:
        state = "LIVE_POSITION"
        label = "Live position"
        next_unblock = "No unblock needed. The bot is managing the open trade."
    elif pending_limit or stage_key in {"limit_entry_placed", "entry_limit_already_pending", "passive_rescue_limit_placed"}:
        state = "PENDING_ENTRY"
        label = "Pending entry"
        next_unblock = "Waiting for the resting limit order to fill, cancel, or expire."
    elif not tradable:
        state = "OBSERVATION_ONLY"
        label = "Observation only"
        next_unblock = _safe_str(
            snap.get("mode_detail"),
            "The venue is not allowing execution on this market yet.",
        )
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
