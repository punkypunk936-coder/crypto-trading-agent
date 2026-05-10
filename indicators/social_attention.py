"""
Optional trader/social attention feed.

The module intentionally avoids logged-in scraping. Configure public RSS/text
sources via social_attention_sources and it will scan for ticker/cashtag flow.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class SocialAttentionSignal:
    valid: bool
    enabled: bool
    score: float = 50.0
    sentiment: str = "NEUTRAL"
    attention_level: str = "LOW"
    mentions: int = 0
    bullish_terms: int = 0
    bearish_terms: int = 0
    summary: str = ""
    sources_checked: int = 0
    source_hits: list[str] = field(default_factory=list)


_CACHE: dict[str, tuple[float, SocialAttentionSignal]] = {}

BULLISH_WORDS = {
    "accumulation",
    "ai",
    "backlog",
    "beat",
    "breakout",
    "bullish",
    "buy",
    "call buying",
    "capex",
    "demand",
    "long",
    "momentum",
    "reclaim",
    "runner",
    "shortage",
    "squeeze",
    "strong",
    "upgrade",
}

BEARISH_WORDS = {
    "bearish",
    "crowded",
    "cut",
    "downgrade",
    "fade",
    "miss",
    "overbought",
    "reject",
    "sell",
    "short",
    "weak",
}


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _sources(config: Any) -> list[str]:
    raw = _cfg_value(config, "social_attention_sources", []) or []
    if isinstance(raw, str):
        items = raw.replace("\n", ",").split(",")
    else:
        items = list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        url = str(item or "").strip()
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    max_sources = int(_cfg_value(config, "social_attention_max_sources", 6) or 6)
    return out[: max(1, max_sources)]


def _terms_for_coin(coin: str) -> list[str]:
    symbol = str(coin or "").upper().strip()
    terms = [f"${symbol}"]
    if len(symbol) >= 4:
        terms.append(symbol)
    else:
        terms.append(f"{symbol} token")
    return terms


def _fetch_source_text(url: str, timeout: float) -> str:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "trading-agent-social-attention/1.0"},
    )
    response.raise_for_status()
    return response.text[:250_000]


def _count_terms(window: str, words: set[str]) -> int:
    lower = window.lower()
    return sum(1 for word in words if word in lower)


def _scan_text(coin: str, text: str) -> tuple[int, int, int]:
    mentions = 0
    bullish = 0
    bearish = 0
    for term in _terms_for_coin(coin):
        pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", re.IGNORECASE)
        for match in pattern.finditer(text or ""):
            mentions += 1
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 180)
            window = text[start:end]
            bullish += _count_terms(window, BULLISH_WORDS)
            bearish += _count_terms(window, BEARISH_WORDS)
    return mentions, bullish, bearish


def get_social_attention_signal(coin: str, config: Any = None) -> SocialAttentionSignal:
    enabled = bool(_cfg_value(config, "use_social_attention", True))
    if not enabled:
        return SocialAttentionSignal(valid=False, enabled=False, summary="social attention disabled")

    sources = _sources(config)
    if not sources:
        return SocialAttentionSignal(
            valid=False,
            enabled=True,
            summary="No public trader/social feeds configured yet.",
        )

    cache_seconds = float(_cfg_value(config, "social_attention_cache_seconds", 300) or 300)
    cache_key = f"{str(coin or '').upper()}:{','.join(sources)}"
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] <= cache_seconds:
        return cached[1]

    timeout = float(_cfg_value(config, "social_attention_timeout_seconds", 4.0) or 4.0)
    mentions = bullish = bearish = 0
    hits: list[str] = []
    checked = 0
    for source in sources:
        try:
            text = _fetch_source_text(source, timeout)
        except Exception:
            continue
        checked += 1
        source_mentions, source_bullish, source_bearish = _scan_text(coin, text)
        if source_mentions:
            hits.append(source)
            mentions += source_mentions
            bullish += source_bullish
            bearish += source_bearish

    score = max(0.0, min(100.0, 50.0 + min(25.0, mentions * 4.0) + bullish * 1.6 - bearish * 1.8))
    if score >= 62.0:
        sentiment = "BULLISH"
    elif score <= 42.0:
        sentiment = "BEARISH"
    else:
        sentiment = "NEUTRAL"
    if mentions >= 8:
        level = "HIGH"
    elif mentions >= 3:
        level = "MEDIUM"
    else:
        level = "LOW"
    if mentions:
        summary = f"{mentions} trader-feed mention(s), {sentiment.lower()} tilt."
    elif checked:
        summary = "Trader feeds checked; no meaningful ticker attention."
    else:
        summary = "Trader feeds were configured but unavailable."
    signal = SocialAttentionSignal(
        valid=checked > 0,
        enabled=True,
        score=round(score, 2),
        sentiment=sentiment,
        attention_level=level,
        mentions=mentions,
        bullish_terms=bullish,
        bearish_terms=bearish,
        summary=summary,
        sources_checked=checked,
        source_hits=hits[:5],
    )
    _CACHE[cache_key] = (now, signal)
    return signal
