"""
llm_referee.py — structured OpenAI referee for high-value or uncertain setups.

Design rules:
  • optional and fail-soft
  • structured JSON only
  • conservative: mostly blocks or annotates, never sends orders itself
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping

import requests

from logger import get_logger
from paths import LLM_REFEREE_REPORT_JSON

log = get_logger("llm_referee")

REFEREE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": ["SUPPORT", "WAIT", "BLOCK"]},
        "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
        "sentiment_bias": {"type": "string", "enum": ["BULLISH", "BEARISH", "NEUTRAL", "MIXED"]},
        "summary": {"type": "string"},
        "why_now": {"type": "string"},
        "principal_risk": {"type": "string"},
        "invalidation_focus": {"type": "string"},
        "next_unblock": {"type": "string"},
        "execution_style": {"type": "string"},
    },
    "required": [
        "verdict",
        "confidence",
        "sentiment_bias",
        "summary",
        "why_now",
        "principal_risk",
        "invalidation_focus",
        "next_unblock",
        "execution_style",
    ],
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _compact(value: Any, limit: int = 220) -> str:
    text = _safe_str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _load_report() -> dict:
    if not LLM_REFEREE_REPORT_JSON.exists():
        return {"enabled": False, "verdicts": {}}
    try:
        data = json.loads(LLM_REFEREE_REPORT_JSON.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data.setdefault("verdicts", {})
            return data
    except Exception:
        pass
    return {"enabled": False, "verdicts": {}}


def _save_report(report: dict) -> None:
    try:
        LLM_REFEREE_REPORT_JSON.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        log.debug("llm_referee_report.json write failed: %s", exc)


def _extract_output_text(payload: Mapping[str, Any]) -> str:
    output_text = _safe_str(payload.get("output_text"))
    if output_text:
        return output_text
    for item in list(payload.get("output") or []):
        for content in list((item or {}).get("content") or []):
            if str((content or {}).get("type") or "") == "output_text":
                return _safe_str((content or {}).get("text"))
    return ""


def _post_responses_request(*, api_key: str, base_url: str, payload: dict, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/responses"
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


class LLMReferee:
    def __init__(self, trading_cfg):
        self.cfg = trading_cfg
        self._cache: dict[str, dict] = {}

    def enabled(self) -> bool:
        return bool(
            getattr(self.cfg, "llm_referee_enabled", False)
            and _safe_str(getattr(self.cfg, "llm_referee_api_key", ""))
        )

    def default_report(self) -> dict:
        if self.enabled():
            return {
                "enabled": True,
                "model": _safe_str(getattr(self.cfg, "llm_referee_model", "gpt-5.4"), "gpt-5.4"),
                "updated_at": int(time.time()),
                "verdicts": dict((_load_report() or {}).get("verdicts") or {}),
            }
        return {
            "enabled": False,
            "reason": "OPENAI referee is disabled or OPENAI_API_KEY is missing",
            "updated_at": int(time.time()),
            "verdicts": dict((_load_report() or {}).get("verdicts") or {}),
        }

    def _fingerprint(self, coin: str, signal_snapshot: Mapping[str, Any], dossier: Mapping[str, Any] | None) -> str:
        material = {
            "coin": coin,
            "action": _safe_str(signal_snapshot.get("action")).upper(),
            "asset_state": _safe_str(signal_snapshot.get("asset_state")).upper(),
            "score": round(_safe_float(signal_snapshot.get("score")), 2),
            "probability": round(_safe_float(signal_snapshot.get("expectancy_probability"), 0.50), 4),
            "expected_r": round(_safe_float(signal_snapshot.get("expectancy_expected_r")), 4),
            "confidence": _safe_str(signal_snapshot.get("confidence")).upper(),
            "live_price": round(_safe_float(signal_snapshot.get("live_price") or signal_snapshot.get("price")), 6),
            "market_map_bias": _safe_str(signal_snapshot.get("market_map_bias")).upper(),
            "decision_reason": _compact(signal_snapshot.get("decision_reason")),
            "next_unblock_reason": _compact(signal_snapshot.get("next_unblock_reason")),
            "dossier_read": _compact(((dossier or {}).get("dossier") or {}).get("current_read")),
        }
        return hashlib.sha1(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()

    def should_review(self, coin: str, signal_snapshot: Mapping[str, Any], *, current_position: str = "") -> bool:
        if not self.enabled():
            return False
        if _safe_str(current_position):
            return False
        action = _safe_str(signal_snapshot.get("action")).upper()
        if action not in {"LONG", "SHORT"}:
            return False
        probability = _safe_float(signal_snapshot.get("expectancy_probability"), 0.50)
        score_distance = abs(_safe_float(signal_snapshot.get("score"), 50.0) - 50.0)
        asset_state = _safe_str(signal_snapshot.get("asset_state")).upper()
        allowed_states = {
            _safe_str(item).upper()
            for item in list(getattr(self.cfg, "llm_referee_review_on_asset_states", []) or [])
            if _safe_str(item)
        }
        return (
            probability >= float(getattr(self.cfg, "llm_referee_min_expectancy_probability", 0.56) or 0.56)
            or score_distance >= float(getattr(self.cfg, "llm_referee_min_score_distance", 10.0) or 10.0)
            or asset_state in allowed_states
        )

    def review_setup(
        self,
        coin: str,
        signal_snapshot: Mapping[str, Any],
        *,
        dossier: Mapping[str, Any] | None = None,
        missed_move_context: Mapping[str, Any] | None = None,
    ) -> dict:
        if not self.enabled():
            return {
                "enabled": False,
                "used": False,
                "verdict": "DISABLED",
                "summary": "OpenAI referee is disabled or OPENAI_API_KEY is missing",
            }

        fingerprint = self._fingerprint(coin, signal_snapshot, dossier)
        ttl_seconds = max(60, int(getattr(self.cfg, "llm_referee_cache_minutes", 45) or 45) * 60)
        cached = self._cache.get(fingerprint)
        now = time.time()
        if cached and (now - float(cached.get("reviewed_at_ts", 0.0) or 0.0)) < ttl_seconds:
            return dict(cached, cached=True)

        context = {
            "coin": coin,
            "signal": {
                "action": _safe_str(signal_snapshot.get("action")).upper(),
                "confidence": _safe_str(signal_snapshot.get("confidence")).upper(),
                "score": round(_safe_float(signal_snapshot.get("score"), 50.0), 2),
                "expectancy_probability": round(_safe_float(signal_snapshot.get("expectancy_probability"), 0.50), 4),
                "expectancy_expected_r": round(_safe_float(signal_snapshot.get("expectancy_expected_r"), 0.0), 4),
                "expectancy_uncertainty": round(_safe_float(signal_snapshot.get("expectancy_uncertainty"), 0.50), 4),
                "asset_state": _safe_str(signal_snapshot.get("asset_state")).upper(),
                "instrument_type": _safe_str(signal_snapshot.get("instrument_type"), "crypto"),
                "live_price": round(_safe_float(signal_snapshot.get("live_price") or signal_snapshot.get("price")), 6),
                "decision_reason": _compact(signal_snapshot.get("decision_reason")),
                "next_unblock_reason": _compact(signal_snapshot.get("next_unblock_reason")),
                "thesis_summary": _compact(signal_snapshot.get("thesis_summary")),
                "market_map_summary": _compact(signal_snapshot.get("market_map_summary")),
                "orderbook_summary": _compact(signal_snapshot.get("execution_quality_summary")),
                "narrative_summary": _compact(signal_snapshot.get("narrative_summary")),
                "analog_summary": _compact(signal_snapshot.get("analog_summary")),
                "orderbook_breakout_state": _safe_str(signal_snapshot.get("orderbook_breakout_state")),
                "orderbook_interaction": _safe_str(signal_snapshot.get("orderbook_interaction")),
            },
            "dossier": dict(dossier or {}),
            "missed_move_context": dict(missed_move_context or {}),
        }

        system_prompt = (
            "You are the referee for a deterministic trading agent. "
            "Be conservative. Prefer patience over activity. "
            "Only return JSON that matches the schema. "
            "Judge whether the directional setup should be SUPPORTED, put on WAIT, or BLOCKED. "
            "Use only the provided context. Do not invent headlines, prices, or catalysts."
        )
        user_prompt = (
            "Review this setup for a high-precision trading system. "
            "The goal is fewer trades, but better trades.\n\n"
            f"Context:\n{json.dumps(context, sort_keys=True)}"
        )

        payload = {
            "model": _safe_str(getattr(self.cfg, "llm_referee_model", "gpt-5.4"), "gpt-5.4"),
            "input": [
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "trade_referee",
                    "schema": REFEREE_SCHEMA,
                    "strict": True,
                }
            },
        }

        try:
            response = _post_responses_request(
                api_key=_safe_str(getattr(self.cfg, "llm_referee_api_key", "")),
                base_url=_safe_str(getattr(self.cfg, "llm_referee_base_url", "https://api.openai.com/v1"), "https://api.openai.com/v1"),
                payload=payload,
                timeout=float(getattr(self.cfg, "llm_referee_timeout_seconds", 18.0) or 18.0),
            )
            parsed = json.loads(_extract_output_text(response))
        except Exception as exc:
            log.debug("[%s] OpenAI referee skipped: %s", coin, exc)
            return {
                "enabled": True,
                "used": False,
                "verdict": "UNAVAILABLE",
                "summary": f"OpenAI referee unavailable: {exc}",
            }

        result = {
            "enabled": True,
            "used": True,
            "cached": False,
            "fingerprint": fingerprint,
            "coin": coin,
            "verdict": _safe_str(parsed.get("verdict"), "WAIT").upper(),
            "confidence": _safe_str(parsed.get("confidence"), "MEDIUM").upper(),
            "sentiment_bias": _safe_str(parsed.get("sentiment_bias"), "NEUTRAL").upper(),
            "summary": _compact(parsed.get("summary"), 260),
            "why_now": _compact(parsed.get("why_now"), 220),
            "principal_risk": _compact(parsed.get("principal_risk"), 220),
            "invalidation_focus": _compact(parsed.get("invalidation_focus"), 180),
            "next_unblock": _compact(parsed.get("next_unblock"), 180),
            "execution_style": _compact(parsed.get("execution_style"), 120),
            "reviewed_at_ts": now,
            "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
            "model": _safe_str(getattr(self.cfg, "llm_referee_model", "gpt-5.4"), "gpt-5.4"),
        }
        self._cache[fingerprint] = dict(result)

        report = _load_report()
        report.update({
            "enabled": True,
            "model": result["model"],
            "updated_at": int(now),
        })
        verdicts = dict(report.get("verdicts") or {})
        verdicts[coin] = dict(result)
        report["verdicts"] = verdicts
        _save_report(report)
        return result
