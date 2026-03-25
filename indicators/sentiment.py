"""
indicators/sentiment.py
Fetches the Crypto Fear & Greed Index from Alternative.me (free, no key needed).

Score:  0  = Extreme Fear  → strong buy signal
Score: 100 = Extreme Greed → strong sell signal
"""

import time
import requests
from logger import get_logger

log = get_logger("sentiment")

FNG_URL = "https://api.alternative.me/fng/?limit=1"
CACHE_TTL = 3600   # Re-fetch at most once per hour (index updates daily)

_cache = {"ts": 0.0, "score": 50, "label": "Neutral"}


def get_fear_greed_score() -> dict:
    """
    Returns a dict with:
        score (int 0-100): raw F&G index value
        label (str)      : 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
        signal_score (float 0-100): converted to our scoring system
              low F&G (fear) → high signal_score (bullish bias)
              high F&G (greed) → low signal_score (bearish bias)
    """
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL:
        log.debug(f"Sentiment cache hit: {_cache['label']} ({_cache['score']})")
        return _build_result(_cache["score"], _cache["label"])

    try:
        resp = requests.get(FNG_URL, timeout=8)
        resp.raise_for_status()
        data  = resp.json()["data"][0]
        score = int(data["value"])
        label = data["value_classification"]
        _cache.update({"ts": now, "score": score, "label": label})
        log.info(f"Fear & Greed Index: {label} ({score}/100)")
        return _build_result(score, label)
    except Exception as e:
        log.warning(f"Could not fetch F&G index ({e}). Using last known value.")
        return _build_result(_cache["score"], _cache["label"])


def _build_result(fng_score: int, label: str) -> dict:
    """
    Convert raw Fear & Greed score (0=fear, 100=greed)
    into our signal score (0=bearish, 100=bullish).
    Invert the scale so extreme fear → bullish signal.
    """
    # Invert: signal_score = 100 - fng_score, with slight extremes boost
    signal_score = 100.0 - fng_score

    # Boost extremes: extreme fear / greed are stronger signals
    if fng_score <= 20:          # Extreme Fear → very bullish
        signal_score = min(90, signal_score + 10)
    elif fng_score >= 80:        # Extreme Greed → very bearish
        signal_score = max(10, signal_score - 10)

    return {
        "raw_score":     fng_score,
        "label":         label,
        "signal_score":  signal_score,
        "is_extreme":    fng_score <= 20 or fng_score >= 80,
    }


def sentiment_summary(result: dict) -> str:
    s = result["signal_score"]
    if s >= 70:
        bias = "BULLISH"
    elif s <= 30:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    return (f"Sentiment: {result['label']} (F&G={result['raw_score']}) → "
            f"{bias} signal ({s:.0f}/100)")
