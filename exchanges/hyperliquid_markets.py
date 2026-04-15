"""
exchanges/hyperliquid_markets.py
Shared Hyperliquid market-catalog helpers.

This module keeps one canonical view of:
  - our internal symbol aliases (SP500, XAU, AMZN, ...)
  - the live Hyperliquid venue symbols (SPX, PAXG, @280, ...)
  - execution semantics (perp vs spot, shortable vs long-only)

The goal is to let data feeds, orderbook logic, dry-run execution, and the
real exchange client all reference the same market truth.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from logger import get_logger

log = get_logger("hl_markets")

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
CATALOG_CACHE_TTL_SECONDS = 300.0

_CATALOG_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "catalog": {},
}

_PERP_MARKETS: Dict[str, Dict[str, Any]] = {
    "BTC": {
        "venue_symbol": "BTC",
        "market_type": "perp",
        "instrument_type": "crypto",
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Bitcoin",
    },
    "ETH": {
        "venue_symbol": "ETH",
        "market_type": "perp",
        "instrument_type": "crypto",
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Ethereum",
    },
    "SOL": {
        "venue_symbol": "SOL",
        "market_type": "perp",
        "instrument_type": "crypto",
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Solana",
    },
    "HYPE": {
        "venue_symbol": "HYPE",
        "market_type": "perp",
        "instrument_type": "crypto",
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "HYPE",
    },
    "TAO": {
        "venue_symbol": "TAO",
        "market_type": "perp",
        "instrument_type": "crypto",
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Bittensor",
    },
    # We keep the internal aliases stable for the rest of the codebase/UI.
    "SP500": {
        "venue_symbol": "SPX",
        "market_type": "perp",
        "instrument_type": "index",
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "S&P 500",
    },
    "XAU": {
        "venue_symbol": "PAXG",
        "market_type": "perp",
        "instrument_type": "index",
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Gold",
    },
}

_SPOT_MARKETS: Dict[str, Dict[str, Any]] = {
    "AAPL": {
        "pair_name": "AAPL/USDC",
        "fallback_symbol": "@268",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": False,
        "display_name": "Apple",
    },
    "AMZN": {
        "pair_name": "AMZN/USDC",
        "fallback_symbol": "@280",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": False,
        "display_name": "Amazon",
    },
    "GOOGL": {
        "pair_name": "GOOGL/USDC",
        "fallback_symbol": "@266",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": False,
        "display_name": "Alphabet",
    },
    "META": {
        "pair_name": "META/USDC",
        "fallback_symbol": "@287",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": False,
        "display_name": "Meta",
    },
    "MSFT": {
        "pair_name": "MSFT/USDC",
        "fallback_symbol": "@289",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": False,
        "display_name": "Microsoft",
    },
    "TSLA": {
        "pair_name": "TSLA/USDC",
        "fallback_symbol": "@264",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": False,
        "display_name": "Tesla",
    },
}

_PREFERRED_ORDER = [
    "BTC", "ETH", "SOL", "HYPE", "TAO", "SP500", "XAU",
    "AAPL", "AMZN", "GOOGL", "META", "MSFT", "TSLA",
]


def _fetch_perp_names() -> Optional[set[str]]:
    try:
        resp = requests.post(HL_INFO_URL, json={"type": "meta"}, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.warning("Failed to refresh Hyperliquid perp meta: %s", exc)
        return None

    universe = payload.get("universe", []) or []
    names = {str(item.get("name") or "").upper() for item in universe if item.get("name")}
    return names or None


def _fetch_spot_pairs() -> Optional[Dict[str, Dict[str, Any]]]:
    try:
        resp = requests.post(HL_INFO_URL, json={"type": "spotMetaAndAssetCtxs"}, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.warning("Failed to refresh Hyperliquid spot meta: %s", exc)
        return None

    if not isinstance(payload, list) or not payload:
        return None
    spot_meta = payload[0] if isinstance(payload[0], dict) else {}
    universe = list(spot_meta.get("universe", []) or [])
    tokens = list(spot_meta.get("tokens", []) or [])
    if not universe or not tokens:
        return None

    pairs: Dict[str, Dict[str, Any]] = {}
    for market in universe:
        try:
            base_idx, quote_idx = market["tokens"]
            base_info = tokens[base_idx]
            quote_info = tokens[quote_idx]
            pair_name = f"{base_info['name']}/{quote_info['name']}"
            pairs[pair_name.upper()] = {
                "venue_symbol": str(market.get("name") or "").upper(),
                "pair_name": pair_name.upper(),
                "index": int(market.get("index", 0) or 0),
            }
        except Exception:
            continue
    return pairs or None


def _catalog_from_fallback() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    for coin, spec in _PERP_MARKETS.items():
        catalog[coin] = {
            "coin": coin,
            **spec,
        }
    for coin, spec in _SPOT_MARKETS.items():
        catalog[coin] = {
            "coin": coin,
            "venue_symbol": spec["fallback_symbol"],
            "pair_name": spec["pair_name"].upper(),
            **{k: v for k, v in spec.items() if k not in {"fallback_symbol", "pair_name"}},
        }
    return catalog


def _ordered_coins(coins: List[str]) -> List[str]:
    order = {coin: idx for idx, coin in enumerate(_PREFERRED_ORDER)}
    return sorted(
        {str(coin).upper() for coin in coins if coin},
        key=lambda coin: (order.get(coin, 999), coin),
    )


def get_hyperliquid_market_catalog(*, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    now = time.time()
    cached_catalog = _CATALOG_CACHE.get("catalog", {})
    cached_ts = float(_CATALOG_CACHE.get("ts", 0.0) or 0.0)
    if not force_refresh and cached_catalog and (now - cached_ts) < CATALOG_CACHE_TTL_SECONDS:
        return dict(cached_catalog)

    perp_names = _fetch_perp_names()
    spot_pairs = _fetch_spot_pairs()

    # If live refresh fails, fall back to the last good catalog first.
    if (perp_names is None or spot_pairs is None) and cached_catalog:
        return dict(cached_catalog)

    catalog: Dict[str, Dict[str, Any]] = {}

    if perp_names is None or spot_pairs is None:
        catalog = _catalog_from_fallback()
    else:
        for coin, spec in _PERP_MARKETS.items():
            venue_symbol = str(spec["venue_symbol"]).upper()
            if venue_symbol not in perp_names:
                continue
            catalog[coin] = {
                "coin": coin,
                **spec,
            }

        for coin, spec in _SPOT_MARKETS.items():
            pair_name = str(spec["pair_name"]).upper()
            live_spec = spot_pairs.get(pair_name)
            venue_symbol = str((live_spec or {}).get("venue_symbol") or spec["fallback_symbol"]).upper()
            if not venue_symbol:
                continue
            catalog[coin] = {
                "coin": coin,
                "venue_symbol": venue_symbol,
                "pair_name": pair_name,
                **{k: v for k, v in spec.items() if k not in {"fallback_symbol", "pair_name"}},
            }

    _CATALOG_CACHE["ts"] = now
    _CATALOG_CACHE["catalog"] = dict(catalog)
    return dict(catalog)


def get_hyperliquid_market_spec(coin: str) -> Optional[Dict[str, Any]]:
    coin_upper = str(coin or "").upper().strip()
    if not coin_upper:
        return None
    return get_hyperliquid_market_catalog().get(coin_upper)


def resolve_hyperliquid_symbol(coin: str) -> str:
    spec = get_hyperliquid_market_spec(coin)
    if spec:
        return str(spec.get("venue_symbol") or str(coin or "").upper()).upper()
    return str(coin or "").upper()


def is_hyperliquid_supported(coin: str) -> bool:
    return get_hyperliquid_market_spec(coin) is not None


def hyperliquid_supports_shorts(coin: str) -> bool:
    spec = get_hyperliquid_market_spec(coin)
    return bool(spec and spec.get("shortable", False))


def hyperliquid_instrument_type(coin: str, default: str = "crypto") -> str:
    spec = get_hyperliquid_market_spec(coin)
    return str((spec or {}).get("instrument_type") or default)


def hyperliquid_market_type(coin: str, default: str = "perp") -> str:
    spec = get_hyperliquid_market_spec(coin)
    return str((spec or {}).get("market_type") or default)


def hyperliquid_supports_live_execution(coin: str) -> bool:
    spec = get_hyperliquid_market_spec(coin)
    return bool(spec and spec.get("live_tradeable", False))


def hyperliquid_supports_paper_execution(coin: str) -> bool:
    spec = get_hyperliquid_market_spec(coin)
    return bool(spec and spec.get("paper_tradeable", False))


def get_hyperliquid_supported_coins(
    *,
    include_spot: bool = True,
    live_tradeable_only: bool = False,
) -> List[str]:
    out: List[str] = []
    for coin, spec in get_hyperliquid_market_catalog().items():
        if spec.get("market_type") == "spot" and not include_spot:
            continue
        if live_tradeable_only and not spec.get("live_tradeable", False):
            continue
        out.append(coin)
    return _ordered_coins(out)
