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

# Instruments routed to macro / equity news instead of CryptoPanic.
INDEX_INSTRUMENTS = {
    "SP500", "XAU", "BRENT", "WTI", "CL", "NDX", "DJI", "VIX",
    "AAPL", "AMZN", "GOOGL", "META", "MSFT", "TSLA",
    "NVDA", "INTC", "MU", "SNDK", "SKHX", "CRWV", "EWY", "HIMS",
}

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

EQUITY_BULLISH_KEYWORDS: Dict[str, float] = {
    "beats": 18, "beat": 14, "tops estimates": 18, "raises guidance": 22,
    "guidance raised": 22, "upgraded": 14, "upgrade": 12,
    "growth accelerates": 16, "strong demand": 14, "shares rise": 12,
    "stock rises": 12, "shares gain": 12, "stock gains": 12,
    "soars": 16, "jumps": 14, "surges": 14,
}

EQUITY_BEARISH_KEYWORDS: Dict[str, float] = {
    "misses": 18, "miss": 14, "cuts guidance": 22, "guidance cut": 22,
    "downgraded": 14, "downgrade": 12, "growth slows": 16,
    "weak demand": 14, "shares fall": 12, "stock falls": 12,
    "shares drop": 12, "stock drops": 12, "slides": 14, "slumps": 16,
}

EQUITY_STRATEGIC_DEAL_KEYWORDS = (
    "strategic collaboration",
    "expand collaboration",
    "expanded collaboration",
    "strategic partnership",
    "deepen ties",
    "deepens ties",
    "agreement",
    "deal",
    "partnership",
    "collaboration",
)

EQUITY_CAPITAL_COMMITMENT_KEYWORDS = (
    "invest",
    "investment",
    "stake",
    "funding",
    "financing",
    "inject",
)

EQUITY_DEMAND_COMMITMENT_KEYWORDS = (
    "commit",
    "commitment",
    "spend",
    "spending",
    "contract",
    "order",
    "backlog",
    "bookings",
    "commercial milestones",
    "revenue commitment",
)

EQUITY_CAPACITY_LOCKIN_KEYWORDS = (
    "capacity",
    "compute",
    "inference",
    "gigawatt",
    "gigawatts",
    "chip",
    "chips",
    "trainium",
    "graviton",
    "gpu",
)

EQUITY_DISTRIBUTION_EXPANSION_KEYWORDS = (
    "available within",
    "available on",
    "native console",
    "platform on aws",
    "through aws",
    "within aws",
    "bedrock",
    "marketplace",
    "distribution",
)

CATALYST_TAG_LABELS: Dict[str, str] = {
    "platform_anchor": "platform anchor",
    "partner_attached": "partner attached",
    "strategic_deal": "strategic deal",
    "capital_commitment": "capital commitment",
    "demand_commitment": "demand commitment",
    "capacity_lock_in": "capacity lock-in",
    "distribution_expand": "distribution expansion",
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
    catalyst_score: float = 0.0
    catalyst_summary: str = ""
    catalyst_tags: List[str] = field(default_factory=list)


@dataclass
class CatalystChecklist:
    score: float = 0.0
    summary: str = ""
    tags: List[str] = field(default_factory=list)


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
    "AAPL": "Apple stock OR AAPL earnings OR iPhone demand",
    "AMZN": "Amazon OR AMZN OR AWS OR Anthropic OR Trainium OR Bedrock",
    "GOOGL": "Alphabet OR GOOGL OR Google Cloud OR Gemini OR Waymo OR Vertex AI",
    "META": "Meta stock OR META earnings OR ad revenue OR AI spend",
    "MSFT": "Microsoft OR MSFT OR Azure OR OpenAI OR Copilot OR Foundry",
    "TSLA": "Tesla stock OR TSLA deliveries OR EV demand",
    "NVDA": "NVIDIA OR NVDA OR Blackwell OR GPU demand OR data center spend",
    "INTC": "Intel OR INTC OR foundry OR AI PC OR Gaudi",
    "MU": "Micron OR MU OR memory pricing OR HBM demand OR DRAM",
    "SNDK": "SanDisk OR SNDK OR NAND pricing OR flash memory",
    "SKHX": "SK Hynix OR SKHX OR HBM memory OR South Korea semiconductors",
    "CRWV": "CoreWeave OR CRWV OR neocloud OR GPU cloud OR AI infrastructure",
    "EWY": "EWY OR South Korea ETF OR Samsung OR SK Hynix OR Korea equities",
    "BRENT": "Brent crude OR oil OR OPEC",
    "WTI": "WTI crude OR oil OR OPEC",
    "CL": "WTI crude OR oil OR OPEC",
    "NDX": "Nasdaq OR NDX OR US tech stocks",
    "DJI": "Dow Jones OR DJI OR US industrials",
    "VIX": "VIX OR volatility index OR stock market volatility",
}

ASSET_NEWS_PROFILES: Dict[str, Dict[str, List[str]]] = {
    "BTC": {"primary": ["bitcoin", "btc"], "context": ["spot etf", "miners", "on-chain"]},
    "ETH": {"primary": ["ethereum", "eth"], "context": ["ether", "staking", "layer 2"]},
    "SOL": {"primary": ["solana", "sol"], "context": ["validator", "solana ecosystem"]},
    "HYPE": {"primary": ["hyperliquid", "hype"], "context": ["perp dex", "hyperliquid protocol"]},
    "TAO": {"primary": ["bittensor", "tao"], "context": ["subnet", "ai network"]},
    "XRP": {"primary": ["xrp", "ripple"], "context": ["ripple labs"]},
    "BNB": {"primary": ["bnb", "binance coin"], "context": ["binance"]},
    "DOGE": {"primary": ["dogecoin", "doge"], "context": ["memecoin"]},
    "ADA": {"primary": ["cardano", "ada"], "context": ["hoskinson"]},
    "BCH": {"primary": ["bitcoin cash", "bch"], "context": []},
    "LINK": {"primary": ["chainlink"], "context": ["oracle network", "$link"]},
    "TRX": {"primary": ["tron", "trx"], "context": ["justin sun"]},
    "SP500": {"primary": ["s&p 500", "sp500", "spx"], "context": ["wall street", "u.s. stocks", "stock market", "equities"]},
    "XAU": {"primary": ["gold", "bullion", "xau", "gold price", "spot gold"], "context": ["treasury yields", "central bank", "dollar"]},
    "AAPL": {"primary": ["apple", "aapl"], "context": ["iphone", "ios", "mac", "tim cook", "services"]},
    "AMZN": {
        "primary": ["amazon", "amzn"],
        "context": ["aws", "prime", "kindle", "e-commerce", "anthropic", "claude", "bedrock", "trainium", "graviton"],
        "strong_context": ["aws", "amazon web services", "bedrock", "trainium", "graviton"],
        "partner_context": ["anthropic", "claude"],
    },
    "GOOGL": {
        "primary": ["alphabet", "google", "googl"],
        "context": ["youtube", "android", "gemini", "cloud", "google cloud", "vertex ai", "waymo"],
        "strong_context": ["google cloud", "gcp", "vertex ai", "gemini", "waymo", "tpu"],
        "partner_context": ["anthropic"],
    },
    "META": {"primary": ["meta"], "context": ["facebook", "instagram", "whatsapp", "reels"]},
    "MSFT": {
        "primary": ["microsoft", "msft"],
        "context": ["azure", "copilot", "windows", "office", "satya nadella", "openai", "chatgpt", "foundry"],
        "strong_context": ["azure", "azure ai", "azure openai", "foundry"],
        "partner_context": ["openai", "chatgpt"],
    },
    "TSLA": {"primary": ["tesla", "tsla"], "context": ["musk", "deliveries", "robotaxi", "ev"]},
    "NVDA": {"primary": ["nvidia", "nvda"], "context": ["blackwell", "gpu", "data center", "ai chip", "cuda"]},
    "INTC": {"primary": ["intel", "intc"], "context": ["foundry", "xeon", "gaudi", "chipmaker"]},
    "MU": {"primary": ["micron", "mu"], "context": ["dram", "nand", "hbm", "memory"]},
    "SNDK": {"primary": ["sandisk", "sndk"], "context": ["nand", "flash memory", "storage"]},
    "SKHX": {"primary": ["sk hynix", "skhx"], "context": ["hbm", "dram", "memory", "south korea chip"]},
    "CRWV": {"primary": ["coreweave", "crwv"], "context": ["gpu cloud", "neocloud", "ai infrastructure", "data center"]},
    "EWY": {"primary": ["ewy", "south korea etf"], "context": ["samsung", "sk hynix", "korea equities"]},
    "HIMS": {"primary": ["hims", "hims & hers", "hims and hers"], "context": ["telehealth", "glp-1", "weight loss", "subscription"]},
    "BRENT": {"primary": ["brent", "oil"], "context": ["opec", "crude"]},
    "WTI": {"primary": ["wti", "oil"], "context": ["opec", "crude"]},
    "CL": {"primary": ["wti", "oil"], "context": ["opec", "crude"]},
    "NDX": {"primary": ["nasdaq", "ndx"], "context": ["tech stocks"]},
    "DJI": {"primary": ["dow jones", "dji", "dow"], "context": ["industrials"]},
    "VIX": {"primary": ["vix", "volatility index"], "context": ["stock market volatility"]},
}

RELEVANCE_THRESHOLD = 0.35


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


def _token_present(text: str, token: str) -> bool:
    token = str(token or "").strip().lower()
    if not token:
        return False
    if re.search(r"[a-z0-9]", token) and " " not in token and all(ch.isalnum() for ch in token):
        return re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", text) is not None
    return token in text


def _headline_relevance(title: str, coin: str) -> float:
    lower = str(title or "").lower()
    profile = ASSET_NEWS_PROFILES.get(str(coin or "").upper(), {})
    primary_terms = list(profile.get("primary") or [])
    context_terms = list(profile.get("context") or [])
    strong_context_terms = list(profile.get("strong_context") or [])
    partner_terms = list(profile.get("partner_context") or [])

    if not primary_terms and not context_terms and not strong_context_terms and not partner_terms:
        # Keep the old behavior for unmapped assets instead of inventing false strictness.
        return 1.0

    primary_hits = sum(1 for term in primary_terms if _token_present(lower, term))
    context_hits = sum(1 for term in context_terms if _token_present(lower, term))
    strong_context_hits = sum(1 for term in strong_context_terms if _token_present(lower, term))
    partner_hits = sum(1 for term in partner_terms if _token_present(lower, term))
    if primary_hits <= 0 and context_hits <= 0 and strong_context_hits <= 0 and partner_hits <= 0:
        return 0.0

    relevance = 0.0
    if primary_hits > 0:
        relevance += 0.55 + min(0.25, 0.12 * max(0, primary_hits - 1))
    if context_hits > 0:
        relevance += 0.15 + min(0.15, 0.08 * max(0, context_hits - 1))
    if strong_context_hits > 0:
        relevance += 0.30 + min(0.20, 0.10 * max(0, strong_context_hits - 1))
    if partner_hits > 0 and (primary_hits > 0 or strong_context_hits > 0):
        relevance += 0.15 + min(0.10, 0.05 * max(0, partner_hits - 1))
    if primary_hits == 0 and strong_context_hits == 0 and context_hits > 0:
        relevance -= 0.15
    if primary_hits == 0 and strong_context_hits > 0 and partner_hits > 0:
        relevance = max(relevance, 0.62)
    return max(0.0, min(1.0, relevance))


def _equity_catalyst_checklist(title: str, coin: str) -> CatalystChecklist:
    coin = str(coin or "").upper()
    if coin not in {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "TSLA", "NVDA", "INTC", "MU", "SNDK", "SKHX", "CRWV", "EWY", "HIMS"}:
        return CatalystChecklist()

    lower = str(title or "").lower()
    profile = ASSET_NEWS_PROFILES.get(coin, {})
    primary_terms = list(profile.get("primary") or [])
    strong_context_terms = list(profile.get("strong_context") or [])
    partner_terms = list(profile.get("partner_context") or [])

    primary_hits = sum(1 for term in primary_terms if _token_present(lower, term))
    strong_context_hits = sum(1 for term in strong_context_terms if _token_present(lower, term))
    partner_hits = sum(1 for term in partner_terms if _token_present(lower, term))

    tags: List[str] = []
    score = 0.0

    if primary_hits > 0 or strong_context_hits > 0:
        tags.append("platform_anchor")
        score += 1.5 if strong_context_hits > 0 else 1.0
    if partner_hits > 0 and (primary_hits > 0 or strong_context_hits > 0):
        tags.append("partner_attached")
        score += 1.0
    if any(keyword in lower for keyword in EQUITY_STRATEGIC_DEAL_KEYWORDS):
        tags.append("strategic_deal")
        score += 1.0
    if any(keyword in lower for keyword in EQUITY_CAPITAL_COMMITMENT_KEYWORDS):
        tags.append("capital_commitment")
        score += 1.0
    if any(keyword in lower for keyword in EQUITY_DEMAND_COMMITMENT_KEYWORDS):
        tags.append("demand_commitment")
        score += 1.25
    if any(keyword in lower for keyword in EQUITY_CAPACITY_LOCKIN_KEYWORDS):
        tags.append("capacity_lock_in")
        score += 1.0
    if any(keyword in lower for keyword in EQUITY_DISTRIBUTION_EXPANSION_KEYWORDS):
        tags.append("distribution_expand")
        score += 0.75

    summary = ""
    if tags:
        summary = " + ".join(CATALYST_TAG_LABELS[tag] for tag in tags[:4])

    return CatalystChecklist(
        score=round(score, 2),
        summary=summary,
        tags=tags,
    )


def _filter_relevant_headlines(headlines: List[str], coin: str) -> List[tuple[str, float]]:
    relevant: List[tuple[str, float]] = []
    for title in headlines:
        relevance = _headline_relevance(title, coin)
        if relevance >= RELEVANCE_THRESHOLD:
            relevant.append((title, relevance))
    return relevant


def _neutral_asset_specific_signal(coin: str, *, context: str) -> NewsSignal:
    return NewsSignal(
        coin=coin,
        score=50.0,
        raw_sentiment=0.0,
        article_count=0,
        velocity="LOW",
        top_headlines=[],
        is_extreme=False,
        valid=True,
        error=f"no asset-specific headlines ({context})",
    )


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
        "AAPL":  "AAPL",
        "AMZN":  "AMZN",
        "GOOGL": "GOOGL",
        "META":  "META",
        "MSFT":  "MSFT",
        "TSLA":  "TSLA",
        "NVDA":  "NVDA",
        "INTC":  "INTC",
        "MU":    "MU",
        "SNDK":  "SNDK",
        "CRWV":  "CRWV",
        "EWY":   "EWY",
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

    relevant_headlines = _filter_relevant_headlines(headlines, coin)
    if not relevant_headlines:
        log.info(f"[{coin}] Macro news returned headlines, but none were asset-specific enough to trust")
        return _neutral_asset_specific_signal(coin, context="macro/equity flow")

    scores = []
    weighted: List[tuple[str, float, float]] = []
    best_catalyst = CatalystChecklist()
    for title, relevance in relevant_headlines:
        score = _score_macro_headline(title, coin=coin)
        checklist = _equity_catalyst_checklist(title, coin)
        if checklist.score > best_catalyst.score:
            best_catalyst = checklist
        weighted_score = score * (0.65 + (0.35 * relevance))
        scores.append(weighted_score)
        weighted.append((title, weighted_score, relevance))

    raw = sum(scores) / len(scores) if scores else 0.0
    indicator_score = (raw + 100) / 2   # map -100…+100 → 0…100

    count = len(relevant_headlines)
    velocity = "EXTREME" if count >= 15 else "HIGH" if count >= 10 else "NORMAL" if count >= 5 else "LOW"
    is_extreme = abs(raw) >= 40 or velocity == "EXTREME" or best_catalyst.score >= 4.0

    top = sorted(weighted, key=lambda x: abs(x[1]), reverse=True)
    top_headlines = [headline for headline, _, _ in top[:3]]

    log.info(
        f"[{coin}] Macro news: {count} headlines | raw={raw:+.1f} | "
        f"score={indicator_score:.1f}/100 | velocity={velocity}"
    )
    if top_headlines:
        log.info(f"[{coin}] Top macro headline: {top_headlines[0][:80]}")
    if best_catalyst.summary:
        log.info(f"[{coin}] Catalyst checklist {best_catalyst.score:.2f}: {best_catalyst.summary}")

    return NewsSignal(
        coin          = coin,
        score         = round(min(100, max(0, indicator_score)), 2),
        raw_sentiment = round(raw, 2),
        article_count = count,
        velocity      = velocity,
        top_headlines = top_headlines,
        is_extreme    = is_extreme,
        valid         = True,
        catalyst_score = best_catalyst.score,
        catalyst_summary = best_catalyst.summary,
        catalyst_tags = best_catalyst.tags,
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
    if coin.upper() in {"AAPL", "AMZN", "GOOGL", "META", "MSFT", "TSLA", "NVDA", "INTC", "MU", "SNDK", "SKHX", "CRWV", "EWY", "HIMS"}:
        for kw, weight in EQUITY_BULLISH_KEYWORDS.items():
            if kw in lower:
                score += weight
        for kw, weight in EQUITY_BEARISH_KEYWORDS.items():
            if kw in lower:
                score -= weight
    # S&P specific boost
    if coin.upper() == "SP500" and any(t in lower for t in ["s&p", "s&p 500", "spx", "sp500", "dow", "nasdaq"]):
        score *= 1.2
    if coin.upper() in {"BRENT", "WTI", "CL"} and any(t in lower for t in ["brent", "wti", "crude", "opec", "oil"]):
        score *= 1.15
    if coin.upper() == "XAU" and any(t in lower for t in ["gold", "xau", "bullion", "comex", "treasury", "dollar"]):
        score *= 1.15
    if coin.upper() == "AAPL" and any(t in lower for t in ["apple", "aapl", "iphone", "app store", "services revenue"]):
        score *= 1.15
    if coin.upper() == "AMZN" and any(t in lower for t in ["amazon", "amzn", "aws", "prime", "e-commerce"]):
        score *= 1.15
    if coin.upper() == "GOOGL" and any(t in lower for t in ["alphabet", "google", "googl", "search ads", "youtube", "cloud"]):
        score *= 1.15
    if coin.upper() == "META" and any(t in lower for t in ["meta", "facebook", "instagram", "whatsapp", "reels"]):
        score *= 1.15
    if coin.upper() == "MSFT" and any(t in lower for t in ["microsoft", "msft", "azure", "copilot", "office"]):
        score *= 1.15
    if coin.upper() == "TSLA" and any(t in lower for t in ["tesla", "tsla", "deliveries", "ev", "autonomous", "robotaxi"]):
        score *= 1.15
    if coin.upper() == "NVDA" and any(t in lower for t in ["nvidia", "nvda", "gpu", "blackwell", "cuda", "data center"]):
        score *= 1.15
    if coin.upper() == "INTC" and any(t in lower for t in ["intel", "intc", "foundry", "gaudi", "ai pc"]):
        score *= 1.15
    if coin.upper() == "MU" and any(t in lower for t in ["micron", "mu", "dram", "nand", "hbm", "memory"]):
        score *= 1.15
    if coin.upper() == "SNDK" and any(t in lower for t in ["sandisk", "sndk", "nand", "flash"]):
        score *= 1.15
    if coin.upper() == "SKHX" and any(t in lower for t in ["sk hynix", "skhx", "hbm", "dram", "memory"]):
        score *= 1.15
    if coin.upper() == "CRWV" and any(t in lower for t in ["coreweave", "crwv", "gpu cloud", "neocloud", "ai infrastructure"]):
        score *= 1.15
    if coin.upper() == "EWY" and any(t in lower for t in ["ewy", "south korea", "samsung", "sk hynix", "korea equities"]):
        score *= 1.15
    catalyst = _equity_catalyst_checklist(title, coin)
    if catalyst.score >= 3.0:
        score += min(32.0, 10.0 + catalyst.score * 5.0)
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

    filtered_items = [
        (item, _headline_relevance(item.title, coin))
        for item in items
    ]
    filtered_items = [
        (item, relevance) for item, relevance in filtered_items
        if relevance >= RELEVANCE_THRESHOLD
    ]
    if not filtered_items:
        log.info(f"[{coin}] News returned headlines, but none were asset-specific enough to trust")
        return _neutral_asset_specific_signal(coin, context="headline flow")

    # Average headline sentiment
    items = [item for item, _ in filtered_items]
    scores = [
        item.sentiment_score * (0.65 + (0.35 * relevance))
        for item, relevance in filtered_items
    ]
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
