"""
feature_store.py — Model-ready feature rows for decisions and closed trades.

This module gives the agent a stable feature vocabulary that can be reused for:
  • historical analog retrieval
  • future ML training
  • audit / debugging of why the agent chose to trade or stay flat
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable, Mapping

from logger import get_logger
from paths import FEATURE_STORE_JSONL

log = get_logger("feature_store")

FEATURE_VERSION = 1

NUMERIC_FEATURE_KEYS = (
    "score",
    "analysis_price",
    "live_price",
    "price",
    "candle_score",
    "news_score",
    "news_articles",
    "news_catalyst_score",
    "narrative_score_adjustment",
    "narrative_uncertainty_delta",
    "memory_adj",
    "memory_cooldown",
    "rl_total_trades",
    "rl_win_rate",
    "rl_pattern_boost",
    "swing_high",
    "swing_low",
    "atr_pct",
    "funding_rate",
    "oi_change_pct",
    "foc_score",
    "orderbook_score",
    "orderbook_imbalance",
    "orderbook_imbalance_mean",
    "orderbook_imbalance_trend",
    "orderbook_imbalance_volatility",
    "orderbook_support_distance_pct",
    "orderbook_resistance_distance_pct",
    "orderbook_support_strength",
    "orderbook_resistance_strength",
    "orderbook_support_wall_persistence",
    "orderbook_resistance_wall_persistence",
    "orderbook_feed_age_seconds",
    "orderbook_feed_snapshot_count",
    "market_map_score_adjustment",
    "market_map_daily_close",
    "market_map_nearest_support",
    "market_map_nearest_resistance",
    "planned_risk_pct",
    "planned_reward_pct",
    "planned_risk_reward_ratio",
    "planned_stop_atr_multiple",
    "planned_target_atr_multiple",
    "planned_target_r_multiple",
    "thesis_alignment_points",
    "thesis_conflict_points",
    "thesis_conviction_score",
    "expectancy_probability",
    "expectancy_expected_r",
    "expectancy_uncertainty",
    "expectancy_score",
    "execution_quality_score",
    "execution_coach_urgency_score",
    "execution_coach_stretch_bps",
    "estimated_slippage_bps",
    "execution_persistence_cycles",
    "analog_sample_size",
    "analog_avg_similarity",
    "analog_reliability",
    "analog_win_rate",
    "analog_avg_pnl_pct",
    "analog_avg_captured_r",
    "analog_score_adjustment",
    "analog_probability_adjustment",
    "analog_expected_r_adjustment",
    "analog_uncertainty_adjustment",
    "data_reliability_score",
    "data_reliability_price_gap_pct",
    "portfolio_guard_size_multiplier",
    "same_direction_exposure_pct",
    "total_theme_exposure_pct",
    "llm_referee_confidence_score",
)

BOOL_FEATURE_KEYS = (
    "using_closed_candles",
    "narrative_event_risk_active",
    "orderbook_favor_longs",
    "orderbook_favor_shorts",
    "orderbook_block_longs",
    "orderbook_block_shorts",
    "orderbook_valid",
    "market_map_available",
    "market_map_favor_longs",
    "market_map_favor_shorts",
    "market_map_block_longs",
    "market_map_block_shorts",
    "thesis_permitted",
    "analog_supportive",
    "analog_adverse",
    "analog_hard_block",
    "data_reliability_permitted",
    "portfolio_guard_permitted",
    "llm_referee_used",
    "llm_referee_blocked",
    "execution_coach_used",
)

CATEGORICAL_FEATURE_KEYS = (
    "action",
    "decision",
    "confidence",
    "instrument_type",
    "mtf_bias",
    "candle_trend",
    "news_velocity",
    "narrative_event_importance",
    "narrative_headline_bias",
    "market_regime",
    "dominant_regime",
    "volatility_label",
    "msb_type",
    "structure_trend",
    "funding_label",
    "cvd_divergence",
    "orderbook_interaction",
    "orderbook_breakout_state",
    "orderbook_intracycle_breakout_state",
    "market_map_bias",
    "stop_basis",
    "target_basis",
    "thesis_candidate_action",
    "thesis_state",
    "thesis_quality",
    "execution_mode",
    "execution_coach_verdict",
    "analog_verdict",
    "asset_state",
    "decision_stage",
    "data_reliability_quality",
    "portfolio_theme",
    "llm_referee_verdict",
    "llm_referee_sentiment_bias",
)

TEXT_FEATURE_KEYS = (
    "decision_reason",
    "flat_reason",
    "price_action_summary",
    "thesis_summary",
    "expectancy_summary",
    "news_catalyst_summary",
    "execution_quality_summary",
    "execution_coach_summary",
    "market_map_summary",
    "market_map_notes",
    "narrative_summary",
    "analog_summary",
    "next_unblock_reason",
    "data_reliability_summary",
    "portfolio_guard_summary",
    "llm_referee_summary",
    "llm_referee_why_now",
    "llm_referee_principal_risk",
    "llm_referee_invalidation_focus",
    "llm_referee_next_unblock",
    "llm_referee_execution_style",
)


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _category(value: Any, default: str = "unknown") -> str:
    text = _safe_str(value, default).replace(" ", "_").lower()
    return text or default


def _compact_text(value: Any, limit: int = 220) -> str:
    text = _safe_str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def build_signal_feature_map(signal_snapshot: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict:
    snap = dict(signal_snapshot or {})
    ctx = dict(extra or {})
    features: dict[str, Any] = {}

    for key in NUMERIC_FEATURE_KEYS:
        features[key] = round(_safe_float(snap.get(key)), 6)

    for key in BOOL_FEATURE_KEYS:
        features[key] = _safe_bool(snap.get(key))

    for key in CATEGORICAL_FEATURE_KEYS:
        features[key] = _category(snap.get(key))

    for key in TEXT_FEATURE_KEYS:
        features[key] = _compact_text(snap.get(key))

    features["thesis_reason_count"] = float(len(list(snap.get("thesis_reasons") or [])))
    features["thesis_blocker_count"] = float(len(list(snap.get("thesis_blockers") or [])))
    features["expectancy_reason_count"] = float(len(list(snap.get("expectancy_reasons") or [])))
    features["expectancy_blocker_count"] = float(len(list(snap.get("expectancy_blockers") or [])))
    features["rl_guard_reason_count"] = float(len(list(snap.get("rl_guard_reasons") or [])))
    features["operator_review_reason_count"] = float(len(list(snap.get("operator_review_reasons") or [])))
    features["analog_match_count"] = float(len(list(snap.get("analog_top_matches") or [])))
    features["data_reliability_permitted"] = _safe_bool((snap.get("data_reliability") or {}).get("permitted", True))
    features["portfolio_guard_permitted"] = _safe_bool((snap.get("portfolio_guard") or {}).get("permitted", True))
    features["same_direction_exposure_pct"] = round(_safe_float((snap.get("portfolio_guard") or {}).get("same_direction_exposure_pct")), 6)
    features["total_theme_exposure_pct"] = round(_safe_float((snap.get("portfolio_guard") or {}).get("total_theme_exposure_pct")), 6)
    llm_referee = dict(snap.get("llm_referee") or {})
    confidence_map = {"LOW": 0.33, "MEDIUM": 0.66, "HIGH": 1.0}
    features["llm_referee_confidence_score"] = round(confidence_map.get(_safe_str(llm_referee.get("confidence")).upper(), 0.0), 6)
    features["llm_referee_used"] = _safe_bool(llm_referee.get("used", False))
    features["llm_referee_blocked"] = _safe_bool(_safe_str(llm_referee.get("verdict")).upper() == "BLOCK")
    features["llm_referee_verdict"] = _category(llm_referee.get("verdict"))
    features["llm_referee_sentiment_bias"] = _category(llm_referee.get("sentiment_bias"))
    features["llm_referee_summary"] = _compact_text(llm_referee.get("summary"))
    features["llm_referee_why_now"] = _compact_text(llm_referee.get("why_now"))
    features["llm_referee_principal_risk"] = _compact_text(llm_referee.get("principal_risk"))
    features["llm_referee_invalidation_focus"] = _compact_text(llm_referee.get("invalidation_focus"))
    features["llm_referee_next_unblock"] = _compact_text(llm_referee.get("next_unblock"))
    features["llm_referee_execution_style"] = _compact_text(llm_referee.get("execution_style"))

    for key, value in ctx.items():
        normalized = f"ctx_{_category(key)}"
        if isinstance(value, bool):
            features[normalized] = _safe_bool(value)
        elif isinstance(value, (int, float)):
            features[normalized] = round(float(value), 6)
        else:
            features[normalized] = _category(value)

    return features


def build_decision_feature_row(record: Mapping[str, Any]) -> dict:
    payload = dict(record or {})
    signal_snapshot = dict(payload.get("signal_snapshot") or {})
    metadata = {
        "stage": payload.get("stage", "decision"),
        "has_position": bool(payload.get("has_position", False)),
        "current_position": payload.get("current_position", ""),
        "tradable": bool(payload.get("tradable", False)),
        "executed": bool(payload.get("executed", False)),
        "blocked": bool(payload.get("blocked", False)),
        "pending_limit": bool(payload.get("pending_limit", False)),
    }
    feature_map = build_signal_feature_map(
        signal_snapshot,
        extra={
            "stage": metadata["stage"],
            "has_position": metadata["has_position"],
            "tradable": metadata["tradable"],
            "executed": metadata["executed"],
            "blocked": metadata["blocked"],
            "pending_limit": metadata["pending_limit"],
            "current_position": metadata["current_position"] or "none",
        },
    )
    return {
        "row_type": "decision",
        "feature_version": FEATURE_VERSION,
        "recorded_at_ts": payload.get("recorded_at_ts", time.time()),
        "decision_id": payload.get("decision_id"),
        "cycle_number": payload.get("cycle_number"),
        "coin": payload.get("coin"),
        "candidate_action": payload.get("candidate_action"),
        "final_action": payload.get("final_action"),
        "metadata": metadata,
        "features": feature_map,
        "labels": {
            "executed": bool(payload.get("executed", False)),
            "blocked": bool(payload.get("blocked", False)),
        },
    }


def build_closed_trade_feature_row(record: Mapping[str, Any]) -> dict:
    payload = dict(record or {})
    entry_context = dict(payload.get("entry_context") or {})
    thesis = dict(payload.get("thesis") or entry_context.get("thesis") or {})
    trade_plan = dict(payload.get("trade_plan") or entry_context.get("trade_plan") or {})
    execution_quality = dict(payload.get("execution_quality") or entry_context.get("execution_quality") or {})
    exit_context = dict(payload.get("exit_context") or {})

    signal_snapshot = {
        "action": payload.get("direction", "FLAT"),
        "decision": payload.get("direction", "FLAT"),
        "score": payload.get("signal_score", entry_context.get("score", 50.0)),
        "confidence": entry_context.get("confidence", "LOW"),
        "instrument_type": entry_context.get("instrument_type", "crypto"),
        "mtf_bias": entry_context.get("mtf_bias", exit_context.get("mtf_bias", "FLAT")),
        "candle_score": entry_context.get("candle_score", 50.0),
        "candle_trend": entry_context.get("candle_trend", "flat"),
        "news_score": entry_context.get("news_score", 50.0),
        "news_velocity": entry_context.get("news_velocity", "low"),
        "memory_adj": entry_context.get("memory_adj", 0.0),
        "rl_total_trades": entry_context.get("rl_total_trades", 0),
        "rl_win_rate": entry_context.get("rl_win_rate", 0.0),
        "rl_pattern_boost": entry_context.get("rl_pattern_boost", 0.0),
        "market_regime": entry_context.get("market_regime", exit_context.get("market_regime", "RANGING")),
        "dominant_regime": entry_context.get("dominant_regime", exit_context.get("dominant_regime", "MIXED")),
        "volatility_label": entry_context.get("volatility_label", "normal"),
        "foc_score": entry_context.get("foc_score", 50.0),
        "funding_label": entry_context.get("funding_label", "n/a"),
        "orderbook_score": entry_context.get("orderbook_score", 50.0),
        "market_map_bias": entry_context.get("market_map_bias", "neutral"),
        "market_map_summary": entry_context.get("market_map_summary", ""),
        "market_map_notes": entry_context.get("market_map_notes", ""),
        "planned_risk_pct": entry_context.get("planned_risk_pct", trade_plan.get("risk_pct", 0.0)),
        "planned_reward_pct": entry_context.get("planned_reward_pct", trade_plan.get("reward_pct", 0.0)),
        "planned_risk_reward_ratio": entry_context.get("planned_risk_reward_ratio", trade_plan.get("risk_reward_ratio", 0.0)),
        "planned_stop_atr_multiple": entry_context.get("planned_stop_atr_multiple", trade_plan.get("stop_atr_multiple", 0.0)),
        "planned_target_atr_multiple": entry_context.get("planned_target_atr_multiple", trade_plan.get("target_atr_multiple", 0.0)),
        "planned_target_r_multiple": entry_context.get("planned_target_r_multiple", trade_plan.get("target_r_multiple", 0.0)),
        "stop_basis": entry_context.get("stop_basis", trade_plan.get("stop_basis", "")),
        "target_basis": entry_context.get("target_basis", trade_plan.get("target_basis", "")),
        "price_action_summary": entry_context.get("price_action_summary", trade_plan.get("price_action_summary", "")),
        "thesis_candidate_action": thesis.get("candidate_action", payload.get("direction", "FLAT")),
        "thesis_state": thesis.get("state", ""),
        "thesis_permitted": thesis.get("permitted", False),
        "thesis_quality": thesis.get("quality", "LOW"),
        "thesis_alignment_points": thesis.get("alignment_points", 0.0),
        "thesis_conflict_points": thesis.get("conflict_points", 0.0),
        "thesis_conviction_score": thesis.get("conviction_score", payload.get("signal_score", 50.0)),
        "thesis_summary": thesis.get("summary", ""),
        "expectancy_probability": entry_context.get("expectancy_probability", exit_context.get("expectancy_probability", 0.50)),
        "expectancy_expected_r": entry_context.get("expectancy_expected_r", exit_context.get("expectancy_expected_r", 0.0)),
        "expectancy_uncertainty": entry_context.get("expectancy_uncertainty", exit_context.get("expectancy_uncertainty", 0.50)),
        "expectancy_score": entry_context.get("expectancy_score", exit_context.get("expectancy_score", payload.get("signal_score", 50.0))),
        "expectancy_summary": entry_context.get("expectancy_summary", ""),
        "execution_mode": entry_context.get("execution_mode", "tradable"),
        "execution_quality_score": execution_quality.get("score", 0.0),
        "execution_quality_summary": execution_quality.get("summary", ""),
        "estimated_slippage_bps": execution_quality.get("estimated_slippage_bps", 0.0),
        "execution_persistence_cycles": execution_quality.get("persistence_cycles", 0),
    }
    feature_map = build_signal_feature_map(
        signal_snapshot,
        extra={
            "trade_outcome": payload.get("outcome", "UNKNOWN"),
            "exit_reason": payload.get("exit_reason", ""),
        },
    )
    plan_outcome = dict(payload.get("plan_outcome") or {})
    return {
        "row_type": "closed_trade",
        "feature_version": FEATURE_VERSION,
        "recorded_at_ts": payload.get("recorded_at_ts", time.time()),
        "trade_id": payload.get("trade_id"),
        "coin": payload.get("coin"),
        "direction": payload.get("direction"),
        "features": feature_map,
        "labels": {
            "outcome": payload.get("outcome", "UNKNOWN"),
            "pnl_pct": round(_safe_float(payload.get("pnl_pct")), 6),
            "pnl_usd": round(_safe_float(payload.get("pnl_usd")), 6),
            "hold_minutes": round(_safe_float(payload.get("hold_minutes")), 6),
            "captured_r_multiple": round(_safe_float(plan_outcome.get("captured_r_multiple")), 6),
            "tp_progress_ratio": round(_safe_float(plan_outcome.get("tp_progress_ratio")), 6),
            "stop_pressure_ratio": round(_safe_float(plan_outcome.get("stop_pressure_ratio")), 6),
            "exit_reason": payload.get("exit_reason", ""),
        },
    }


def append_feature_row(row: Mapping[str, Any]) -> None:
    if not isinstance(row, Mapping) or not row:
        return
    FEATURE_STORE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with FEATURE_STORE_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(row), default=_json_default, sort_keys=True) + "\n")


def append_decision_feature_row(record: Mapping[str, Any]) -> None:
    append_feature_row(build_decision_feature_row(record))


def append_closed_trade_feature_row(record: Mapping[str, Any]) -> None:
    append_feature_row(build_closed_trade_feature_row(record))


def load_feature_rows(limit: int | None = None, row_type: str | None = None) -> list[dict]:
    if not FEATURE_STORE_JSONL.exists():
        return []

    rows: list[dict] = []
    with FEATURE_STORE_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:
                log.debug(f"Skipping malformed feature-store line: {exc}")
                continue
            if row_type and str(row.get("row_type", "")).lower() != row_type.lower():
                continue
            rows.append(row)
    if limit is None:
        return rows
    return rows[-limit:]


def iter_feature_rows(row_type: str | None = None) -> Iterable[dict]:
    for row in load_feature_rows(row_type=row_type):
        yield row
