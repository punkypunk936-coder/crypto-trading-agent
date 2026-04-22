"""
narrative.py — narrative and macro-event gating for trade quality.

This layer is intentionally lightweight:
  - major headline flow comes from the existing news signal
  - optional macro events come from a local operator-owned JSON file

The goal is not to predict the news. It is to stop the agent from trading as if
narrative risk does not exist.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List

from logger import get_logger
from paths import MACRO_EVENTS_JSON

log = get_logger("narrative")


@dataclass
class NarrativeSignal:
    coin: str
    valid: bool = True
    summary: str = ""
    score_adjustment: float = 0.0
    uncertainty_delta: float = 0.0
    headline_bias: str = "NEUTRAL"
    headline_score: float = 50.0
    headline_count: int = 0
    event_risk_active: bool = False
    event_name: str = ""
    event_importance: str = "NONE"
    minutes_to_event: float | None = None
    block_longs: bool = False
    block_shorts: bool = False
    reasons: List[str] | None = None


def _parse_iso8601(value: str | None) -> float | None:
    if not value:
        return None
    try:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _load_macro_events() -> list[dict]:
    if not MACRO_EVENTS_JSON.exists():
        return []
    try:
        payload = json.loads(MACRO_EVENTS_JSON.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to load macro events file: %s", exc)
        return []
    if isinstance(payload, dict):
        payload = payload.get("events", [])
    return [item for item in payload if isinstance(item, dict)]


def _coin_tags(coin: str) -> set[str]:
    coin = coin.upper()
    tags = {"ALL", coin}
    if coin in {"BTC", "ETH", "SOL", "HYPE", "TAO"}:
        tags.update({"CRYPTO", "RISK"})
    if coin in {"SP500"}:
        tags.update({"INDEX", "MACRO", "RISK"})
    if coin in {"XAU"}:
        tags.update({"METAL", "MACRO", "DEFENSIVE"})
    if coin in {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "TSLA"}:
        tags.update({"EQUITY", "MAG7", "RISK"})
    if coin in {"BRENT", "WTI", "CL"}:
        tags.update({"ENERGY", "MACRO"})
    return tags


def _event_matches(event: dict, coin: str) -> bool:
    tags = _coin_tags(coin)
    symbols = event.get("symbols") or event.get("coins") or event.get("tags") or []
    if isinstance(symbols, str):
        symbols = [symbols]
    for symbol in symbols:
        if str(symbol).upper() in tags:
            return True
    return False


def _event_window_match(
    event: dict,
    *,
    now_ts: float,
    risk_window_minutes: int,
    post_event_cooldown_minutes: int,
) -> tuple[bool, float | None]:
    start_ts = _parse_iso8601(event.get("starts_at") or event.get("start") or event.get("time"))
    if start_ts is None:
        return False, None
    window_before = max(0, int(risk_window_minutes or 0)) * 60
    window_after = max(0, int(post_event_cooldown_minutes or 0)) * 60
    if (start_ts - window_before) <= now_ts <= (start_ts + window_after):
        minutes_to_event = (start_ts - now_ts) / 60.0
        return True, minutes_to_event
    return False, (start_ts - now_ts) / 60.0


def _impact_rank(value: str) -> int:
    mapping = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
    return mapping.get(str(value or "LOW").upper(), 1)


def _headline_bias(news_signal) -> tuple[str, float, int, bool, bool, list[str], float]:
    if not news_signal or not getattr(news_signal, "valid", False):
        return "NEUTRAL", 50.0, 0, False, False, [], 0.0

    score = float(getattr(news_signal, "score", 50.0) or 50.0)
    count = int(getattr(news_signal, "article_count", 0) or 0)
    catalyst_score = float(getattr(news_signal, "catalyst_score", 0.0) or 0.0)
    catalyst_summary = str(getattr(news_signal, "catalyst_summary", "") or "")
    reasons: list[str] = []
    block_longs = False
    block_shorts = False
    score_adjustment = 0.0

    if score >= 58.0:
        bias = "BULLISH"
        score_adjustment += min(4.0, (score - 50.0) / 4.0)
        reasons.append("headline flow is leaning bullish")
    elif score <= 42.0:
        bias = "BEARISH"
        score_adjustment -= min(4.0, (50.0 - score) / 4.0)
        reasons.append("headline flow is leaning bearish")
    else:
        bias = "NEUTRAL"

    if getattr(news_signal, "is_extreme", False):
        if score <= 25.0:
            block_longs = True
            reasons.append("extreme bearish headlines block longs")
        elif score >= 75.0:
            block_shorts = True
            reasons.append("extreme bullish headlines block shorts")

    if catalyst_score >= 3.0 and score >= 55.0:
        catalyst_adjustment = min(6.0, catalyst_score * 1.2)
        score_adjustment += catalyst_adjustment
        if catalyst_summary:
            reasons.append(f"catalyst checklist aligned: {catalyst_summary}")
        else:
            reasons.append("major catalyst checklist aligned with the move")
        if catalyst_score >= 4.0 and score >= 68.0:
            block_shorts = True
            reasons.append("major bullish catalyst means fading it needs exceptional evidence")

    return bias, score, count, block_longs, block_shorts, reasons, score_adjustment


def get_narrative_signal(
    coin: str,
    *,
    news_signal=None,
    risk_window_minutes: int = 90,
    post_event_cooldown_minutes: int = 45,
    now_ts: float | None = None,
) -> NarrativeSignal:
    coin = coin.upper()
    now_ts = float(now_ts or time.time())
    reasons: list[str] = []
    signal = NarrativeSignal(coin=coin, reasons=reasons)

    (
        signal.headline_bias,
        signal.headline_score,
        signal.headline_count,
        signal.block_longs,
        signal.block_shorts,
        headline_reasons,
        signal.score_adjustment,
    ) = _headline_bias(news_signal)
    reasons.extend(headline_reasons)

    applicable_event = None
    applicable_minutes = None
    for event in sorted(_load_macro_events(), key=lambda item: _impact_rank(item.get("impact")) * -1):
        if not _event_matches(event, coin):
            continue
        active, minutes_to_event = _event_window_match(
            event,
            now_ts=now_ts,
            risk_window_minutes=risk_window_minutes,
            post_event_cooldown_minutes=post_event_cooldown_minutes,
        )
        if active:
            applicable_event = event
            applicable_minutes = minutes_to_event
            break

    if applicable_event:
        signal.event_risk_active = True
        signal.event_name = str(applicable_event.get("name", "macro event") or "macro event")
        signal.event_importance = str(applicable_event.get("impact", "HIGH") or "HIGH").upper()
        signal.minutes_to_event = round(float(applicable_minutes or 0.0), 1)
        signal.uncertainty_delta += 0.14 if signal.event_importance == "HIGH" else 0.08
        if signal.minutes_to_event is not None and signal.minutes_to_event >= 0:
            reasons.append(
                f"{signal.event_name} is within the narrative risk window ({signal.minutes_to_event:.0f}m)"
            )
        else:
            reasons.append(f"{signal.event_name} cooldown is still active")

        event_bias = str(applicable_event.get("bias", "NEUTRAL") or "NEUTRAL").upper()
        if event_bias == "BULLISH":
            signal.score_adjustment += 2.0
        elif event_bias == "BEARISH":
            signal.score_adjustment -= 2.0

    if reasons:
        signal.summary = "; ".join(reasons[:3])
    else:
        signal.summary = "Narrative flow is neutral"
    return signal
