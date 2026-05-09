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
ACTIVITY_CACHE_TTL_SECONDS = 180.0
TRADEXYZ_DEX = "xyz"

_CATALOG_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "catalog": {},
}

_ACTIVITY_CACHE: Dict[str, Dict[str, Any]] = {}

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
    "MON": {
        "venue_symbol": "MON",
        "market_type": "perp",
        "instrument_type": "crypto",
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "MON",
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
        "live_tradeable": True,
        "display_name": "Apple",
    },
    "AMZN": {
        "pair_name": "AMZN/USDC",
        "fallback_symbol": "@280",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Amazon",
    },
    "GOOGL": {
        "pair_name": "GOOGL/USDC",
        "fallback_symbol": "@266",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Alphabet",
    },
    "META": {
        "pair_name": "META/USDC",
        "fallback_symbol": "@287",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Meta",
    },
    "MSFT": {
        "pair_name": "MSFT/USDC",
        "fallback_symbol": "@289",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Microsoft",
    },
    "TSLA": {
        "pair_name": "TSLA/USDC",
        "fallback_symbol": "@264",
        "market_type": "spot",
        "instrument_type": "equity",
        "shortable": False,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": "Tesla",
    },
}

TRADEXYZ_ASSET_METADATA: Dict[str, Dict[str, Any]] = {
    "AAPL": {"display_name": "Apple", "instrument_type": "equity", "categories": ["mag7"]},
    "ALUMINIUM": {"display_name": "Aluminium", "instrument_type": "index", "categories": ["commodities_metals"]},
    "AMD": {"display_name": "AMD", "instrument_type": "equity", "categories": ["semis_memory"]},
    "AMZN": {"display_name": "Amazon", "instrument_type": "equity", "categories": ["mag7"]},
    "BABA": {"display_name": "Alibaba", "instrument_type": "equity", "categories": ["asia_macro"]},
    "BIRD": {"display_name": "Allbirds", "instrument_type": "equity", "categories": ["growth"]},
    "BRENTOIL": {"display_name": "Brent Oil", "instrument_type": "index", "categories": ["energy"]},
    "BX": {"display_name": "Blackstone", "instrument_type": "equity", "categories": ["financials"]},
    "CBRS": {
        "display_name": "Cerebras Systems",
        "instrument_type": "equity",
        "categories": ["pre_ipo", "semis_memory", "ai_infra"],
        "pre_ipo": True,
    },
    "CL": {"display_name": "Crude Oil", "instrument_type": "index", "categories": ["energy"]},
    "COIN": {"display_name": "Coinbase", "instrument_type": "equity", "categories": ["crypto_equities"]},
    "COPPER": {"display_name": "Copper", "instrument_type": "index", "categories": ["commodities_metals"]},
    "CORN": {"display_name": "Corn", "instrument_type": "index", "categories": ["agriculture"]},
    "COST": {"display_name": "Costco", "instrument_type": "equity", "categories": ["consumer"]},
    "CRCL": {"display_name": "Circle", "instrument_type": "equity", "categories": ["crypto_equities"]},
    "CRWV": {"display_name": "CoreWeave", "instrument_type": "equity", "categories": ["neoclouds"]},
    "DKNG": {"display_name": "DraftKings", "instrument_type": "equity", "categories": ["consumer", "growth"]},
    "DRAM": {"display_name": "DRAM", "instrument_type": "index", "categories": ["semis_memory"]},
    "DXY": {"display_name": "US Dollar Index", "instrument_type": "index", "categories": ["fx_rates", "indices_macro"]},
    "EBAY": {"display_name": "eBay", "instrument_type": "equity", "categories": ["consumer"]},
    "EUR": {"display_name": "Euro", "instrument_type": "index", "categories": ["fx_rates"]},
    "EWJ": {"display_name": "Japan ETF", "instrument_type": "index", "categories": ["asia_macro", "indices_macro"]},
    "EWZ": {"display_name": "Brazil ETF", "instrument_type": "index", "categories": ["latam_macro", "indices_macro"]},
    "EWY": {"display_name": "South Korea ETF", "instrument_type": "index", "categories": ["asia_macro", "indices_macro"]},
    "GME": {"display_name": "GameStop", "instrument_type": "equity", "categories": ["meme_momentum"]},
    "GOLD": {"display_name": "Gold", "instrument_type": "index", "categories": ["commodities_metals"]},
    "GOOGL": {"display_name": "Alphabet", "instrument_type": "equity", "categories": ["mag7"]},
    "H100": {"display_name": "H100 Group", "instrument_type": "equity", "categories": ["crypto_equities", "microcap"]},
    "HIMS": {"display_name": "Hims & Hers", "instrument_type": "equity", "categories": ["biotech_glp1", "growth"]},
    "HOOD": {"display_name": "Robinhood", "instrument_type": "equity", "categories": ["crypto_equities"]},
    "HYUNDAI": {"display_name": "Hyundai", "instrument_type": "equity", "categories": ["asia_macro"]},
    "INTC": {"display_name": "Intel", "instrument_type": "equity", "categories": ["semis_memory"]},
    "JP225": {"display_name": "Nikkei 225", "instrument_type": "index", "categories": ["asia_macro", "indices_macro"]},
    "JPY": {"display_name": "Japanese Yen", "instrument_type": "index", "categories": ["fx_rates", "asia_macro"]},
    "KIOXIA": {"display_name": "Kioxia", "instrument_type": "equity", "categories": ["semis_memory", "asia_macro"]},
    "KRW": {"display_name": "Korean Won", "instrument_type": "index", "categories": ["fx_rates", "asia_macro"]},
    "KR200": {"display_name": "KOSPI 200", "instrument_type": "index", "categories": ["asia_macro", "indices_macro"]},
    "LITE": {"display_name": "LITE", "instrument_type": "equity", "categories": ["growth"]},
    "LLY": {"display_name": "Eli Lilly", "instrument_type": "equity", "categories": ["biotech_glp1"]},
    "META": {"display_name": "Meta", "instrument_type": "equity", "categories": ["mag7"]},
    "MRVL": {"display_name": "Marvell", "instrument_type": "equity", "categories": ["semis_memory"]},
    "MSFT": {"display_name": "Microsoft", "instrument_type": "equity", "categories": ["mag7"]},
    "MSTR": {"display_name": "MicroStrategy", "instrument_type": "equity", "categories": ["crypto_equities"]},
    "MU": {"display_name": "Micron", "instrument_type": "equity", "categories": ["semis_memory"]},
    "NATGAS": {"display_name": "Natural Gas", "instrument_type": "index", "categories": ["energy"]},
    "NFLX": {"display_name": "Netflix", "instrument_type": "equity", "categories": ["consumer", "growth"]},
    "NIFTY": {"display_name": "Nifty 50", "instrument_type": "index", "categories": ["asia_macro", "indices_macro"]},
    "NVDA": {"display_name": "NVIDIA", "instrument_type": "equity", "categories": ["mag7", "semis_memory"]},
    "ORCL": {"display_name": "Oracle", "instrument_type": "equity", "categories": ["ai_infra"]},
    "PALLADIUM": {"display_name": "Palladium", "instrument_type": "index", "categories": ["commodities_metals"]},
    "PLATINUM": {"display_name": "Platinum", "instrument_type": "index", "categories": ["commodities_metals"]},
    "PLTR": {"display_name": "Palantir", "instrument_type": "equity", "categories": ["ai_infra", "growth"]},
    "PURRDAT": {"display_name": "PurrDat", "instrument_type": "index", "categories": ["indices_macro"]},
    "RIVN": {"display_name": "Rivian", "instrument_type": "equity", "categories": ["growth"]},
    "RKLB": {"display_name": "Rocket Lab", "instrument_type": "equity", "categories": ["growth"]},
    "SILVER": {"display_name": "Silver", "instrument_type": "index", "categories": ["commodities_metals"]},
    "SKHX": {"display_name": "SK Hynix", "instrument_type": "equity", "categories": ["semis_memory", "asia_macro"]},
    "SMSN": {"display_name": "Samsung", "instrument_type": "equity", "categories": ["semis_memory", "asia_macro"]},
    "SNDK": {"display_name": "SanDisk", "instrument_type": "equity", "categories": ["semis_memory"]},
    "SOFTBANK": {"display_name": "SoftBank", "instrument_type": "equity", "categories": ["asia_macro", "ai_infra"]},
    "SP500": {"display_name": "S&P 500", "instrument_type": "index", "categories": ["indices_macro"]},
    "TSLA": {"display_name": "Tesla", "instrument_type": "equity", "categories": ["mag7"]},
    "TSM": {"display_name": "TSMC", "instrument_type": "equity", "categories": ["semis_memory", "asia_macro"]},
    "TTF": {"display_name": "Dutch TTF Gas", "instrument_type": "index", "categories": ["energy"]},
    "URANIUM": {"display_name": "Uranium", "instrument_type": "index", "categories": ["uranium"]},
    "URNM": {"display_name": "Uranium Miners ETF", "instrument_type": "index", "categories": ["uranium"]},
    "USAR": {"display_name": "USA Rare Earth", "instrument_type": "equity", "categories": ["uranium", "commodities_metals"]},
    "VIX": {"display_name": "VIX", "instrument_type": "index", "categories": ["volatility", "indices_macro"]},
    "VOL": {"display_name": "Volatility", "instrument_type": "index", "categories": ["volatility"]},
    "WHEAT": {"display_name": "Wheat", "instrument_type": "index", "categories": ["agriculture"]},
    "XLE": {"display_name": "Energy Select Sector SPDR", "instrument_type": "index", "categories": ["energy", "indices_macro"]},
    "XYZ100": {"display_name": "XYZ 100", "instrument_type": "index", "categories": ["indices_macro"]},
    "ZM": {"display_name": "Zoom Communications", "instrument_type": "equity", "categories": ["software", "growth"]},
}

_TRADEXYZ_PERP_MARKETS: Dict[str, Dict[str, Any]] = {
    coin: {
        "venue_symbol": f"xyz:{coin}",
        "market_type": "perp",
        "instrument_type": str(meta.get("instrument_type") or "equity"),
        "shortable": True,
        "paper_tradeable": True,
        "live_tradeable": True,
        "display_name": str(meta.get("display_name") or coin),
        "categories": list(meta.get("categories") or ["other_stocks"]),
        "pre_ipo": bool(meta.get("pre_ipo", False)),
        "dex": TRADEXYZ_DEX,
    }
    for coin, meta in TRADEXYZ_ASSET_METADATA.items()
}

_PREFERRED_ORDER = [
    "BTC", "ETH", "SOL", "HYPE", "MON", "TAO", "SP500", "XAU",
    "AAPL", "AMZN", "GOOGL", "META", "MSFT", "TSLA",
    "NVDA", "INTC", "MU", "SNDK", "SKHX", "CRWV", "CBRS", "EWY", "HIMS",
    *[coin for coin in TRADEXYZ_ASSET_METADATA if coin not in {
        "BTC", "ETH", "SOL", "HYPE", "MON", "TAO", "SP500", "XAU",
        "AAPL", "AMZN", "GOOGL", "META", "MSFT", "TSLA",
        "NVDA", "INTC", "MU", "SNDK", "SKHX", "CRWV", "CBRS", "EWY", "HIMS",
    }],
]

_TRADEXYZ_DYNAMIC_OVERRIDES: Dict[str, Dict[str, Any]] = {
    coin: {
        "instrument_type": str(meta.get("instrument_type") or "equity"),
        "display_name": str(meta.get("display_name") or coin),
        "categories": list(meta.get("categories") or ["other_stocks"]),
        "pre_ipo": bool(meta.get("pre_ipo", False)),
    }
    for coin, meta in TRADEXYZ_ASSET_METADATA.items()
}


def _fetch_perp_names(*, dex: str = "") -> Optional[set[str]]:
    try:
        payload = {"type": "meta"}
        if dex:
            payload["dex"] = dex
        resp = requests.post(HL_INFO_URL, json=payload, timeout=8)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        dex_label = dex or "native"
        log.warning("Failed to refresh Hyperliquid perp meta for %s: %s", dex_label, exc)
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
                "venue_symbol": str(market.get("name") or "").strip(),
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
    for coin, spec in _TRADEXYZ_PERP_MARKETS.items():
        catalog[coin] = {
            "coin": coin,
            **spec,
        }
    for coin, spec in _SPOT_MARKETS.items():
        existing = dict(catalog.get(coin) or {})
        if str(existing.get("dex") or "").strip() == TRADEXYZ_DEX:
            continue
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
    tradexyz_perp_names = _fetch_perp_names(dex=TRADEXYZ_DEX)
    spot_pairs = _fetch_spot_pairs()

    # If live refresh fails, fall back to the last good catalog first.
    if (perp_names is None or tradexyz_perp_names is None or spot_pairs is None) and cached_catalog:
        return dict(cached_catalog)

    catalog: Dict[str, Dict[str, Any]] = {}

    if perp_names is None or tradexyz_perp_names is None or spot_pairs is None:
        catalog = _catalog_from_fallback()
    else:
        manual_perp_by_venue = {
            str(spec.get("venue_symbol") or coin).upper(): (coin, spec)
            for coin, spec in _PERP_MARKETS.items()
        }
        for venue_symbol in sorted(perp_names):
            venue_key = str(venue_symbol or "").upper()
            manual_coin, manual = manual_perp_by_venue.get(venue_key, (None, None))
            if manual_coin and manual:
                catalog[manual_coin] = {
                    "coin": manual_coin,
                    **manual,
                }
                continue
            catalog[venue_key] = {
                "coin": venue_key,
                "venue_symbol": str(venue_symbol or "").strip() or venue_key,
                "market_type": "perp",
                "instrument_type": "crypto",
                "shortable": True,
                "paper_tradeable": True,
                "live_tradeable": True,
                "display_name": venue_key,
            }

        manual_tradexyz_by_venue = {
            str(spec.get("venue_symbol") or coin).upper(): (coin, spec)
            for coin, spec in _TRADEXYZ_PERP_MARKETS.items()
        }
        for venue_symbol in sorted(tradexyz_perp_names):
            venue_raw = str(venue_symbol or "").strip()
            venue_key = venue_raw.upper()
            manual_coin, manual = manual_tradexyz_by_venue.get(venue_key, (None, None))
            if not manual_coin or not manual:
                internal_coin = venue_key.split(":", 1)[-1].strip().upper()
                if not internal_coin:
                    continue
                override = dict(_TRADEXYZ_DYNAMIC_OVERRIDES.get(internal_coin) or {})
                catalog[internal_coin] = {
                    "coin": internal_coin,
                    "venue_symbol": venue_raw or venue_key,
                    "market_type": "perp",
                    "instrument_type": str(override.get("instrument_type") or "equity"),
                    "shortable": bool(override.get("shortable", True)),
                    "paper_tradeable": True,
                    "live_tradeable": True,
                    "display_name": str(override.get("display_name") or internal_coin),
                    "categories": list(override.get("categories") or ["other_stocks"]),
                    "pre_ipo": bool(override.get("pre_ipo", False)),
                    "dex": TRADEXYZ_DEX,
                }
                continue
            catalog[manual_coin] = {
                "coin": manual_coin,
                **manual,
            }

        for coin, spec in _SPOT_MARKETS.items():
            existing = dict(catalog.get(coin) or {})
            if str(existing.get("dex") or "").strip() == TRADEXYZ_DEX:
                continue
            pair_name = str(spec["pair_name"]).upper()
            live_spec = spot_pairs.get(pair_name)
            venue_symbol = str((live_spec or {}).get("venue_symbol") or spec["fallback_symbol"]).strip()
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


def resolve_hyperliquid_internal_coin(symbol: str) -> str:
    symbol_upper = str(symbol or "").upper().strip()
    if not symbol_upper:
        return ""
    catalog = get_hyperliquid_market_catalog()
    for coin, spec in catalog.items():
        if str(spec.get("venue_symbol") or coin).upper() == symbol_upper:
            return coin
    return symbol_upper


def get_hyperliquid_market_dex(coin: str) -> str:
    spec = get_hyperliquid_market_spec(coin) or {}
    return str(spec.get("dex") or "").strip()


def get_hyperliquid_supported_dexes() -> List[str]:
    dexs = {
        str(spec.get("dex") or "").strip()
        for spec in get_hyperliquid_market_catalog().values()
        if str(spec.get("market_type") or "").lower() == "perp" and str(spec.get("dex") or "").strip()
    }
    return sorted(dexs)


def _activity_max_age_seconds(spec: Optional[Dict[str, Any]]) -> int:
    market_type = str((spec or {}).get("market_type") or "").lower()
    instrument_type = str((spec or {}).get("instrument_type") or "").lower()
    if market_type == "spot" or instrument_type == "equity":
        return 48 * 3600
    if instrument_type == "index":
        return 12 * 3600
    return 6 * 3600


def get_hyperliquid_market_activity(coin: str, *, force_refresh: bool = False) -> Dict[str, Any]:
    coin_upper = str(coin or "").upper().strip()
    spec = get_hyperliquid_market_spec(coin_upper)
    if not spec:
        return {
            "coin": coin_upper,
            "active": False,
            "reason": "unsupported",
            "last_candle_ts": 0,
            "age_seconds": None,
        }

    cached = _ACTIVITY_CACHE.get(coin_upper)
    now = time.time()
    if (
        not force_refresh
        and cached
        and (now - float(cached.get("checked_at", 0.0) or 0.0)) < ACTIVITY_CACHE_TTL_SECONDS
    ):
        return dict(cached)

    venue_symbol = str(spec.get("venue_symbol") or coin_upper).strip()
    dex = str(spec.get("dex") or "").strip()
    start_ms = int((now - (120 * 3600)) * 1000)
    end_ms = int(now * 1000)
    activity = {
        "coin": coin_upper,
        "venue_symbol": venue_symbol,
        "active": False,
        "reason": "no_recent_candles",
        "last_candle_ts": 0,
        "age_seconds": None,
        "checked_at": now,
    }
    try:
        resp = requests.post(
            HL_INFO_URL,
            json={
                "type": "candleSnapshot",
                "req": {
                    "coin": venue_symbol,
                    "interval": "1h",
                    "startTime": start_ms,
                    "endTime": end_ms,
                },
                **({"dex": dex} if dex else {}),
            },
            timeout=8,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        activity["reason"] = f"activity_probe_failed:{type(exc).__name__}"
        if cached:
            return dict(cached)
        _ACTIVITY_CACHE[coin_upper] = dict(activity)
        return dict(activity)

    candles = payload if isinstance(payload, list) else []
    if candles:
        try:
            last_candle_ts = int(candles[-1].get("t", 0) or 0)
        except Exception:
            last_candle_ts = 0
        if last_candle_ts > 0:
            age_seconds = max(0.0, now - (last_candle_ts / 1000.0))
            max_age_seconds = _activity_max_age_seconds(spec)
            activity["last_candle_ts"] = last_candle_ts
            activity["age_seconds"] = age_seconds
            activity["active"] = age_seconds <= max_age_seconds
            activity["reason"] = "fresh_candles" if activity["active"] else "stale_candles"

    _ACTIVITY_CACHE[coin_upper] = dict(activity)
    return dict(activity)


def resolve_hyperliquid_symbol(coin: str) -> str:
    spec = get_hyperliquid_market_spec(coin)
    if spec:
        return str(spec.get("venue_symbol") or str(coin or "").upper()).strip()
    return str(coin or "").upper()


def is_hyperliquid_supported(coin: str) -> bool:
    return get_hyperliquid_market_spec(coin) is not None


def hyperliquid_market_is_active(coin: str, *, force_refresh: bool = False) -> bool:
    return bool(get_hyperliquid_market_activity(coin, force_refresh=force_refresh).get("active", False))


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
    active_only: bool = False,
) -> List[str]:
    out: List[str] = []
    for coin, spec in get_hyperliquid_market_catalog().items():
        if spec.get("market_type") == "spot" and not include_spot:
            continue
        if live_tradeable_only and not spec.get("live_tradeable", False):
            continue
        if active_only and not hyperliquid_market_is_active(coin):
            continue
        out.append(coin)
    return _ordered_coins(out)
