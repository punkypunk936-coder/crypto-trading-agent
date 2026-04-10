"""
indicators/news.py — News sentiment for crypto trading.

Sources
───────
  1. CryptoPanic API (free, no key needed for public feed)
     https://cryptopanic.com/api/v1/posts/?auth_token=&currencies=BTC,ETH
  2. Hyperliquid-specific feed for HYPE token
     (GitHub releases + protocol announcements via RSS)

HYPE-specific logic
───────────────────
HYPE (Hyperliquid's token) moves on:
  • Protocol upgrades / new features
  • New market listings on Hyperliquid
  • Volume milestones / TVL records
  • Competitor news (disadvantage)
  • Airdrop / tokenomics updates

Scoring
───────
  Each headline is scored -100 (max bearish) → +100 (max bullish)
  by keyword matching. Score is averaged across recent articles.
  Converted to 0–100 scale (50 = neutral) for the indicator system.

  Very high velocity (many articles in 1h) adds a magnitude boost
  regardless of direction.
"""

from __future__ import annotations

import time
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import copy
import xml.etree.ElementTree as ET
import requests

from logger import get_logger

log = get_logger("news")

# Cache news for 10 minutes — no need to hit API every 2 min cycle
CACHE_TTL = 600
STALE_CACHE_TTL = 3600
SOURCE_BACKOFF_TTL = 900

# CryptoPanic free API (no token = public posts, 50/hr rate limit)
CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

# Yahoo Finance RSS — free, no API key — used for equity indexes
# ^GSPC = S&P 500 | ^IXIC = NASDAQ | ^DJI = Dow Jones
YAHOO_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Instruments routed to macro / commodity news instead of CryptoPanic.
INDEX_INSTRUMENTS = {"SP500", "XAU", "BRENT", "WTI", "CL", "NDX", "DJI", "VIX"}

# ── Macro / equity keyword weights ────────────────────────────────────────────
# Used for SP500 and other index instruments
MACRO_BULLISH_KEYWORDS: Dict[str, float] = {
    # Fed / monetary policy
    "rate cut": 30, "pivot": 25, "dovish": 22, "easy money": 20,
    "quantitative easing": 18, "qe": 18, "stimulus": 18, "liquidity": 12,
    # Economic strength
    "jobs beat": 25, "nonfarm payrolls beat": 25, "gdp beat": 22,
    "soft landing": 20, "strong earnings": 20, "earnings beat": 18,
    "record high": 18, "all-time high": 20, "bull market": 18,
    "rally": 15, "breakout": 15, "risk-on": 15, "buy the dip": 12,
    # Inflation (falling = good for equities)
    "inflation falls": 22, "cpi lower": 22, "disinflation": 18,
    "deflation": 15,
    # Geopolitical
    "trade deal": 15, "ceasefire": 12, "peace": 10,
}

MACRO_BEARISH_KEYWORDS: Dict[str, float] = {
    # Fed / monetary policy
    "rate hike": 30, "hawkish": 25, "tightening": 22, "quantitative tightening": 18,
    "qt": 15, "interest rate rise": 25,
    # Economic weakness
    "recession": 30, "stagflation": 28, "gdp miss": 22, "jobs miss": 22,
    "unemployment rise": 20, "earnings miss": 18, "profit warning": 18,
    "layoffs": 15, "default": 25, "debt ceiling": 20,
    # Inflation (rising = bad for equities)
    "inflation surge": 25, "cpi higher": 22, "hot inflation": 22,
    "hyperinflation": 28,
    # Market structure
    "bear market": 25, "crash": 30, "circuit breaker": 25,
    "sell-off": 20, "correction": 15, "yield curve inversion": 22,
    "bank run": 28, "banking crisis": 28,
    # Geopolitical
    "war escalation": 20, "sanctions": 15, "trade war": 18,
    "tariff": 12,
}

OIL_BULLISH_KEYWORDS: Dict[str, float] = {
    "opec cut": 28, "opec+ cut": 30, "production cut": 22, "supply cut": 22,
    "inventory draw": 18, "crude draw": 18, "drawdown": 14, "supply disruption": 22,
    "pipeline outage": 18, "middle east tension": 16, "sanctions": 14,
    "demand rebound": 16, "refinery outage": 14,
}

OIL_BEARISH_KEYWORDS: Dict[str, float] = {
    "output increase": 24, "production increase": 24, "inventory build": 18,
    "crude build": 18, "oversupply": 22, "demand slowdown": 18,
    "recession fears": 16, "opec output hike": 28, "price cap": 14,
    "strategic reserve release": 16, "export surge": 14,
}

GOLD_BULLISH_KEYWORDS: Dict[str, float] = {
    "safe haven": 24, "central bank buying": 24, "gold demand": 18,
    "weaker dollar": 18, "dollar weakens": 18, "yield falls": 18,
    "real yields fall": 22, "rate cut": 16, "geopolitical tension": 18,
    "inflation hedge": 18, "recession fears": 12,
}

GOLD_BEARISH_KEYWORDS: Dict[str, float] = {
    "strong dollar": 20, "dollar strengthens": 20, "yield rises": 18,
    "real yields rise": 22, "rate hike": 18, "hawkish fed": 18,
    "risk-on": 12, "profit taking": 10, "gold sell-off": 22,
}

# Bullish/bearish keyword weights
BULLISH_KEYWORDS: Dict[str, float] = {
    # Strong
    "all-time high": 30, "ath": 25, "record": 20, "surge": 20,
    "breakout": 20, "bullish": 20, "rally": 18, "mooning": 18,
    "listing": 15, "partnership": 15, "integration": 12,
    "upgrade": 12, "launch": 12, "milestone": 12,
    "adoption": 15, "institutional": 15, "etf": 20,
    "buy": 8, "long": 8, "accumulate": 10,
    # HYPE-specific
    "hyperliquid": 5, "perp": 5, "perpetual": 5,
    "new market": 15, "new listing": 15, "tvl record": 20,
    "volume record": 20, "airdrop": 10, "token": 5,
}

BEARISH_KEYWORDS: Dict[str, float] = {
    # Strong
    "crash": 30, "collapse": 30, "hack": 35, "exploit": 35,
    "rug": 35, "scam": 30, "fraud": 30, "ban": 25,
    "regulation": 15, "sec": 15, "lawsuit": 20, "fine": 15,
    "bearish": 20, "dump": 20, "sell-off": 20, "selloff": 20,
    "liquidation": 15, "delisting": 20, "shutdown": 25,
    "concern": 8, "risk": 8, "warning": 10, "fear": 12,
    "plunge": 20, "slump": 15, "drop": 10, "fall": 8,
    # HYPE-specific bearish
    "outage": 20, "downtime": 15, "vulnerability": 25,
    "competitor": 10, "dydx": 10, "hyperliquid hack": 50,
}


@dataclass
class NewsItem:
    title: str
    published_at: str
    url: str
    source: str
    sentiment_score: float   # -100 to +100
    is_hype_relevant: bool


@dataclass
class NewsSignal:
    coin: str
    score: float              # 0–100 (50 = neutral)
    raw_sentiment: float      # average headline score (-100 to +100)
    article_count: int
    velocity: str             # "LOW" | "NORMAL" | "HIGH" | "EXTREME"
    top_headlines: List[str]
    is_extreme: bool          # big news → magnitude boost
    valid: bool = True
    error: str  = ""


# ── Module-level cache ────────────────────────────────────────────────────

_cache: Dict[str, dict] = {}   # coin → {ts, signal}
_source_backoff: Dict[str, float] = {}

CRYPTO_NEWS_QUERIES: Dict[str, str] = {
    "BTC": "Bitcoin OR BTC crypto",
    "ETH": "Ethereum OR ETH crypto",
    "SOL": "Solana OR SOL crypto",
    "HYPE": "Hyperliquid OR HYPE token OR Hyperliquid protocol",
    "TAO": "Bittensor OR TAO token",
}

MACRO_NEWS_QUERIES: Dict[str, str] = {
    "SP500": "S&P 500 OR SPX OR US stocks OR Wall Street",
    "XAU": "gold OR XAU OR bullion OR treasury yields",
    "BRENT": "Brent crude OR oil OR OPEC",
    "WTI": "WTI crude OR oil OR OPEC",
    "CL": "WTI crude OR oil OR OPEC",
    "NDX": "Nasdaq OR NDX OR US tech stocks",
    "DJI": "Dow Jones OR DJI OR US industrials",
    "VIX": "VIX OR volatility index OR stock market volatility",
}


def _backoff_active(key: str) -> bool:
    return time.time() < float(_source_backoff.get(key, 0.0) or 0.0)


def _set_backoff(key: str, ttl_seconds: int = SOURCE_BACKOFF_TTL) -> None:
    _source_backoff[key] = time.time() + max(60, int(ttl_seconds or SOURCE_BACKOFF_TTL))


def _clear_backoff(key: str) -> None:
    _source_backoff.pop(key, None)


def _clone_signal(signal: NewsSignal, **updates) -> NewsSignal:
    payload = copy.deepcopy(signal.__dict__)
    payload.update(updates)
    return NewsSignal(**payload)


def _stale_cached_signal(coin: str) -> Optional[NewsSignal]:
    cached = _cache.get(coin)
    if not cached:
        return None
    age = time.time() - float(cached.get("ts", 0.0) or 0.0)
    if age > STALE_CACHE_TTL:
        return None
    signal = cached.get("signal")
    if not isinstance(signal, NewsSignal):
        return None
    return _clone_signal(
        signal,
        valid=True,
        error=(f"stale cache reused after live feed issue: {signal.error}" if signal.error else "stale cache reused"),
    )


def _parse_rss_titles(content: bytes, limit: int = 20) -> List[str]:
    try:
        root = ET.fromstring(content)
    except Exception:
        return []
    titles: List[str] = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        title = re.sub(r"\s*-\s*[^-]{1,60}$", "", title).strip()
        titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def _fetch_google_news_headlines(query: str, *, limit: int = 20) -> List[str]:
    params = {"q": f"{query} when:1d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
    resp = requests.get(GOOGLE_NEWS_RSS_URL, params=params, timeout=8, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    return _parse_rss_titles(resp.content, limit=limit)


def get_news_signal(coin: str, auth_token: str = "") -> NewsSignal:
    """
    Main entry point. Routes index instruments to macro news,
    crypto instruments to CryptoPanic. Returns cached result if fresh.
    """
    cached = _cache.get(coin)
    if cached and time.time() - cached["ts"] < CACHE_TTL:
        return cached["signal"]

    if coin.upper() in INDEX_INSTRUMENTS:
        signal = _fetch_macro_news(coin)
    else:
        signal = _fetch_and_score(coin, auth_token=auth_token)

    if not signal.valid:
        stale = _stale_cached_signal(coin)
        if stale:
            _cache[coin] = {"ts": time.time(), "signal": stale}
            log.info(f"[{coin}] Reusing stale cached news signal while live feed recovers")
            return stale

    _cache[coin] = {"ts": time.time(), "signal": signal}
    return signal


def _fetch_macro_news(coin: str) -> NewsSignal:
    """
    Fetch macro / commodity news for non-crypto instruments (SP500, BRENT, WTI etc.) via
    Yahoo Finance RSS. No API key required.
    """
    TICKER_MAP = {
        "SP500": "^GSPC",   # Hyperliquid S&P 500 perpetual → Yahoo Finance ^GSPC RSS
        "XAU":   "GC=F",
        "BRENT": "BZ=F",
        "WTI":   "CL=F",
        "CL":    "CL=F",
        "NDX":   "^IXIC",
        "DJI":   "^DJI",
        "VIX":   "^VIX",
    }
    ticker = TICKER_MAP.get(coin.upper(), "^GSPC")

    headlines: List[str] = []
    errors: List[str] = []
    yahoo_key = f"macro:yahoo:{coin.upper()}"
    google_key = f"macro:google:{coin.upper()}"

    if not _backoff_active(yahoo_key):
        try:
            params = {"s": ticker, "region": "US", "lang": "en-US"}
            resp = requests.get(YAHOO_RSS_URL, params=params, timeout=8, headers=REQUEST_HEADERS)
            resp.raise_for_status()
            headlines = _parse_rss_titles(resp.content, limit=20)
            if headlines:
                _clear_backoff(yahoo_key)
        except Exception as e:
            errors.append(f"Yahoo RSS: {e}")
            _set_backoff(yahoo_key)

    if not headlines and not _backoff_active(google_key):
        query = MACRO_NEWS_QUERIES.get(coin.upper(), f"{coin} markets")
        try:
            headlines = _fetch_google_news_headlines(query, limit=20)
            if headlines:
                _clear_backoff(google_key)
        except Exception as e:
            errors.append(f"Google News: {e}")
            _set_backoff(google_key)

    if not headlines:
        error = " | ".join(errors) if errors else "no macro headlines returned"
        log.warning(f"[{coin}] Macro news fetch failed: {error} — using neutral")
        return NewsSignal(coin=coin, score=50.0, raw_sentiment=0.0,
                          article_count=0, velocity="LOW",
                          top_headlines=[], is_extreme=False,
                          valid=False, error=error)

    if not headlines:
        return NewsSignal(coin=coin, score=50.0, raw_sentiment=0.0,
                          article_count=0, velocity="LOW",
                          top_headlines=[], is_extreme=False, valid=True)

    scores = []
    for title in headlines:
        score = _score_macro_headline(title, coin=coin)
        scores.append(score)

    raw = sum(scores) / len(scores) if scores else 0.0
    indicator_score = (raw + 100) / 2   # map -100…+100 → 0…100

    count = len(headlines)
    velocity = "EXTREME" if count >= 15 else "HIGH" if count >= 10 else "NORMAL" if count >= 5 else "LOW"
    is_extreme = abs(raw) >= 40 or velocity == "EXTREME"

    top = sorted(zip(headlines, scores), key=lambda x: abs(x[1]), reverse=True)
    top_headlines = [h for h, _ in top[:3]]

    log.info(
        f"[{coin}] Macro news: {count} headlines | raw={raw:+.1f} | "
        f"score={indicator_score:.1f}/100 | velocity={velocity}"
    )
    if top_headlines:
        log.info(f"[{coin}] Top macro headline: {top_headlines[0][:80]}")

    return NewsSignal(
        coin          = coin,
        score         = round(min(100, max(0, indicator_score)), 2),
        raw_sentiment = round(raw, 2),
        article_count = count,
        velocity      = velocity,
        top_headlines = top_headlines,
        is_extreme    = is_extreme,
        valid         = True,
    )


def _score_macro_headline(title: str, coin: str = "") -> float:
    """Score a macro / commodity headline from -100 to +100."""
    lower = title.lower()
    score = 0.0
    for kw, weight in MACRO_BULLISH_KEYWORDS.items():
        if kw in lower:
            score += weight
    for kw, weight in MACRO_BEARISH_KEYWORDS.items():
        if kw in lower:
            score -= weight
    if coin.upper() in {"BRENT", "WTI", "CL"}:
        for kw, weight in OIL_BULLISH_KEYWORDS.items():
            if kw in lower:
                score += weight
        for kw, weight in OIL_BEARISH_KEYWORDS.items():
            if kw in lower:
                score -= weight
    if coin.upper() == "XAU":
        for kw, weight in GOLD_BULLISH_KEYWORDS.items():
            if kw in lower:
                score += weight
        for kw, weight in GOLD_BEARISH_KEYWORDS.items():
            if kw in lower:
                score -= weight
    # S&P specific boost
    if coin.upper() == "SP500" and any(t in lower for t in ["s&p", "s&p 500", "spx", "sp500", "dow", "nasdaq"]):
        score *= 1.2
    if coin.upper() in {"BRENT", "WTI", "CL"} and any(t in lower for t in ["brent", "wti", "crude", "opec", "oil"]):
        score *= 1.15
    if coin.upper() == "XAU" and any(t in lower for t in ["gold", "xau", "bullion", "comex", "treasury", "dollar"]):
        score *= 1.15
    return max(-100, min(100, score))


def _fetch_and_score(coin: str, auth_token: str = "") -> NewsSignal:
    """Fetch news from CryptoPanic and score it."""
    items: List[NewsItem] = []
    errors: List[str] = []
    cryptopanic_key = f"crypto:cryptopanic:{coin.upper()}"
    google_key = f"crypto:google:{coin.upper()}"

    if not _backoff_active(cryptopanic_key):
        try:
            params = {
                "currencies": coin,
                "kind":       "news",
                "filter":     "important",
                "public":     "true",
            }
            if auth_token:
                params["auth_token"] = auth_token
            resp = requests.get(CRYPTOPANIC_URL, params=params, timeout=8, headers=REQUEST_HEADERS)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            for r in results[:20]:
                title = (r.get("title") or "").strip()
                if not title:
                    continue
                score = _score_headline(title, coin)
                items.append(NewsItem(
                    title            = title,
                    published_at     = r.get("published_at", ""),
                    url              = r.get("url", ""),
                    source           = r.get("source", {}).get("title", ""),
                    sentiment_score  = score,
                    is_hype_relevant = coin == "HYPE" and _is_hype_specific(title),
                ))
            if items:
                _clear_backoff(cryptopanic_key)
        except Exception as e:
            errors.append(f"CryptoPanic: {e}")
            _set_backoff(cryptopanic_key)

    if not items and not _backoff_active(google_key):
        query = CRYPTO_NEWS_QUERIES.get(coin.upper(), f"{coin} crypto")
        try:
            for title in _fetch_google_news_headlines(query, limit=20):
                score = _score_headline(title, coin)
                items.append(NewsItem(
                    title=title,
                    published_at="",
                    url="",
                    source="Google News",
                    sentiment_score=score,
                    is_hype_relevant=coin == "HYPE" and _is_hype_specific(title),
                ))
            if items:
                _clear_backoff(google_key)
        except Exception as e:
            errors.append(f"Google News: {e}")
            _set_backoff(google_key)

    if not items:
        error = " | ".join(errors) if errors else "no crypto headlines returned"
        log.warning(f"[{coin}] News fetch failed: {error} — using neutral")
        return NewsSignal(coin=coin, score=50.0, raw_sentiment=0.0,
                          article_count=0, velocity="LOW",
                          top_headlines=[], is_extreme=False,
                          valid=False, error=error)

    # Average headline sentiment
    scores = [it.sentiment_score for it in items]
    raw    = sum(scores) / len(scores) if scores else 0.0

    # HYPE: boost if Hyperliquid-specific positive/negative news found
    if coin == "HYPE":
        hype_items = [it for it in items if it.is_hype_relevant]
        if hype_items:
            hype_raw = sum(it.sentiment_score for it in hype_items) / len(hype_items)
            raw = raw * 0.4 + hype_raw * 0.6   # weight toward HYPE-specific news
            log.info(f"[HYPE] {len(hype_items)} Hyperliquid-specific articles "
                     f"found, raw sentiment: {hype_raw:+.1f}")

    # Convert -100/+100 → 0/100 scale
    indicator_score = (raw + 100) / 2   # -100→0, 0→50, +100→100

    # Velocity (articles per unit time)
    count = len(items)
    if count >= 15:
        velocity = "EXTREME"
    elif count >= 10:
        velocity = "HIGH"
    elif count >= 5:
        velocity = "NORMAL"
    else:
        velocity = "LOW"

    is_extreme = abs(raw) >= 50 or velocity == "EXTREME"

    top = [it.title for it in sorted(items, key=lambda x: abs(x.sentiment_score),
                                      reverse=True)[:3]]

    log.info(
        f"[{coin}] News: {count} articles | raw={raw:+.1f} | "
        f"score={indicator_score:.1f}/100 | velocity={velocity}"
        f"{' | public feed' if not auth_token else ''}"
    )
    if top:
        log.info(f"[{coin}] Top headline: {top[0][:80]}")

    return NewsSignal(
        coin          = coin,
        score         = round(min(100, max(0, indicator_score)), 2),
        raw_sentiment = round(raw, 2),
        article_count = count,
        velocity      = velocity,
        top_headlines = top,
        is_extreme    = is_extreme,
        valid         = True,
    )


def _score_headline(title: str, coin: str) -> float:
    """Score a headline from -100 to +100."""
    lower = title.lower()
    score = 0.0

    for kw, weight in BULLISH_KEYWORDS.items():
        if kw in lower:
            score += weight

    for kw, weight in BEARISH_KEYWORDS.items():
        if kw in lower:
            score -= weight

    # Coin-specific mentions boost the signal
    if coin.lower() in lower:
        score *= 1.3

    return max(-100, min(100, score))


def _is_hype_specific(title: str) -> bool:
    """True if the headline is specifically about Hyperliquid protocol."""
    lower = title.lower()
    return any(kw in lower for kw in [
        "hyperliquid", "hype token", "hl protocol",
        "hl dex", "hyperliquid dex", "hyperliquid perp",
    ])
