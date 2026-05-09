"""
market_universe.py — build a broader Hyperliquid scout universe from market-cap filters.

The goal is to widen observation without automatically widening execution.
We intersect Hyperliquid's live universe with respectable large-cap assets:
  - crypto perps via CoinGecko market caps
  - supported equities via a lightweight market-cap supplement
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

from exchanges.hyperliquid_markets import (
    get_hyperliquid_market_catalog,
    hyperliquid_market_is_active,
)
from logger import get_logger
from paths import MARKET_CAP_UNIVERSE_JSON

log = get_logger("market_universe")

COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"
NASDAQ_SUMMARY_URL = "https://api.nasdaq.com/api/quote/{symbol}/summary"
EQUITY_MARKET_CAP_OVERRIDES_USD = {
    # Fallbacks for Trade.xyz names that do not have a direct U.S. quote
    # endpoint, or where the live quote provider can temporarily block.
    "CBRS": 23_000_000_000.0,
    "H100": 54_000_000.0,
    "HYUNDAI": 45_000_000_000.0,
    "KIOXIA": 6_000_000_000.0,
    "SKHX": 120_000_000_000.0,
    "SMSN": 350_000_000_000.0,
    "SOFTBANK": 130_000_000_000.0,
    "INTC": 90_000_000_000.0,
    "HIMS": 14_000_000_000.0,
}
NASDAQ_MARKET_CAP_SKIP_SYMBOLS = {
    "CBRS",
    "H100",
    "HYUNDAI",
    "KIOXIA",
    "SKHX",
    "SMSN",
    "SOFTBANK",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _rank_or_default(value: Any, default: int = 999999) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except Exception:
        return default


def _parse_market_cap_value(value: Any) -> float:
    raw = str(value or "").strip()
    if not raw or raw.upper() == "N/A":
        return 0.0
    cleaned = raw.replace("$", "").replace(",", "").strip()
    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([TtBbMmKk])?$", cleaned)
    if not match:
        return _safe_float(cleaned)
    multiplier = {
        "T": 1_000_000_000_000.0,
        "B": 1_000_000_000.0,
        "M": 1_000_000.0,
        "K": 1_000.0,
    }.get(str(match.group(2) or "").upper(), 1.0)
    return _safe_float(match.group(1)) * multiplier


def _load_cache(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fetch_coingecko_market_caps(*, pages: int, per_page: int = 250) -> list[dict]:
    rows: list[dict] = []
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    for page in range(1, max(1, int(pages or 1)) + 1):
        try:
            resp = session.get(
                COINGECKO_MARKETS_URL,
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": min(250, max(1, int(per_page))),
                    "page": page,
                    "sparkline": "false",
                    "price_change_percentage": "24h",
                },
                timeout=20,
            )
            resp.raise_for_status()
            batch = resp.json()
        except Exception:
            if rows:
                log.info("CoinGecko market-cap refresh stopped early on page %s; keeping the rows already fetched", page)
                break
            raise
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
    return rows


def _fetch_equity_market_caps(symbols: list[str]) -> dict[str, dict]:
    tickers = [str(symbol or "").upper().strip() for symbol in symbols if str(symbol or "").strip()]
    if not tickers:
        return {}

    rows: dict[str, dict] = {}
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0",
    })
    try:
        for offset in range(0, len(tickers), 25):
            batch = tickers[offset: offset + 25]
            resp = session.get(
                YAHOO_QUOTE_URL,
                params={"symbols": ",".join(batch)},
                timeout=20,
            )
            resp.raise_for_status()
            payload = resp.json()
            for item in list((payload.get("quoteResponse") or {}).get("result") or []):
                symbol = str(item.get("symbol") or "").upper().strip()
                market_cap = _safe_float(item.get("marketCap"))
                if not symbol or market_cap <= 0:
                    continue
                rows[symbol] = {
                    "name": str(item.get("longName") or item.get("shortName") or symbol),
                    "market_cap": market_cap,
                    "market_cap_rank": None,
                    "price_change_percentage_24h": _safe_float(item.get("regularMarketChangePercent")),
                    "source": "yahoo_quote",
                }
    except Exception as exc:
        log.info("Yahoo equity market-cap refresh fell back to Nasdaq/overrides: %s", exc)

    def _fetch_nasdaq_market_cap(symbol: str) -> tuple[str, dict | None]:
        try:
            resp = requests.get(
                NASDAQ_SUMMARY_URL.format(symbol=symbol),
                params={"assetclass": "stocks"},
                headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
                timeout=8,
            )
            resp.raise_for_status()
            payload = resp.json()
            summary = dict(((payload.get("data") or {}).get("summaryData") or {}))
            market_cap = _parse_market_cap_value((summary.get("MarketCap") or {}).get("value"))
            if market_cap <= 0:
                return symbol, None
            return symbol, {
                "name": symbol,
                "market_cap": market_cap,
                "market_cap_rank": None,
                "price_change_percentage_24h": 0.0,
                "source": "nasdaq_summary",
            }
        except Exception as exc:
            log.debug("[%s] Nasdaq market-cap lookup skipped: %s", symbol, exc)
            return symbol, None

    missing = [
        symbol for symbol in tickers
        if symbol not in rows and symbol not in NASDAQ_MARKET_CAP_SKIP_SYMBOLS
    ]
    if missing:
        with ThreadPoolExecutor(max_workers=min(8, len(missing))) as pool:
            futures = [pool.submit(_fetch_nasdaq_market_cap, symbol) for symbol in missing]
            for future in as_completed(futures):
                symbol, row = future.result()
                if row:
                    rows[symbol] = row

    for symbol in tickers:
        if symbol in rows:
            continue
        override = _safe_float(EQUITY_MARKET_CAP_OVERRIDES_USD.get(symbol))
        if override <= 0:
            continue
        rows[symbol] = {
            "name": symbol,
            "market_cap": override,
            "market_cap_rank": None,
            "price_change_percentage_24h": 0.0,
            "source": "equity_override",
        }

    return rows


def _build_equity_candidates(
    *,
    catalog: dict[str, dict],
    min_market_cap_usd: float,
    active_only: bool,
) -> list[dict]:
    equity_symbols = [
        str(coin).upper()
        for coin, spec in catalog.items()
        if str(spec.get("instrument_type") or "").lower() == "equity"
    ]
    equity_market_caps = _fetch_equity_market_caps(equity_symbols)
    candidates: list[dict] = []
    for coin in equity_symbols:
        spec = dict(catalog.get(coin) or {})
        market_row = equity_market_caps.get(coin)
        if not market_row:
            continue
        market_cap = _safe_float(market_row.get("market_cap"))
        if market_cap < float(min_market_cap_usd or 0.0):
            continue
        active = hyperliquid_market_is_active(coin) if active_only else True
        if active_only and not active:
            continue
        candidates.append({
            "coin": coin,
            "name": str(market_row.get("name") or spec.get("display_name") or coin),
            "symbol": coin,
            "market_cap_usd": round(market_cap, 2),
            "market_cap_rank": _rank_or_default(market_row.get("market_cap_rank")),
            "price_change_pct_24h": _safe_float(market_row.get("price_change_percentage_24h")),
            "venue_symbol": str(spec.get("venue_symbol") or coin).strip(),
            "active": bool(active),
        })
    return candidates


def _normalize_records(
    records: list[dict] | None,
    *,
    catalog: dict[str, dict],
    min_market_cap_usd: float,
    active_only: bool,
) -> list[dict]:
    normalized: list[dict] = []
    seen: set[str] = set()
    for record in records or []:
        coin = str(record.get("coin") or record.get("symbol") or "").upper().strip()
        if not coin or coin in seen:
            continue
        market_cap = _safe_float(record.get("market_cap_usd") or record.get("market_cap"))
        if market_cap < float(min_market_cap_usd or 0.0):
            continue
        spec = dict(catalog.get(coin) or {})
        active = hyperliquid_market_is_active(coin) if active_only else bool(record.get("active", True))
        if active_only and not active:
            continue
        normalized.append({
            "coin": coin,
            "name": str(record.get("name") or spec.get("display_name") or coin),
            "symbol": coin,
            "market_cap_usd": round(market_cap, 2),
            "market_cap_rank": _rank_or_default(record.get("market_cap_rank")),
            "price_change_pct_24h": _safe_float(
                record.get("price_change_pct_24h")
                if record.get("price_change_pct_24h") is not None
                else record.get("price_change_percentage_24h")
            ),
            "venue_symbol": str(spec.get("venue_symbol") or record.get("venue_symbol") or coin).strip(),
            "active": bool(active),
        })
        seen.add(coin)
    return normalized


def _finalize_watchlist_payload(
    *,
    candidates: list[dict],
    min_market_cap_usd: float,
    active_only: bool,
    pages: int,
    max_coins: int,
    source: str,
) -> dict:
    candidates.sort(
        key=lambda item: (
            _rank_or_default(item.get("market_cap_rank")),
            -_safe_float(item.get("market_cap_usd")),
            str(item.get("coin")),
        )
    )
    limited = candidates[: max(1, int(max_coins or 60))]
    return {
        "generated_at_ts": time.time(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "min_market_cap_usd": float(min_market_cap_usd),
        "active_only": bool(active_only),
        "pages": int(pages or 1),
        "source": source,
        "coins": [row["coin"] for row in limited],
        "records": limited,
    }


def build_hyperliquid_market_cap_watchlist(
    *,
    min_market_cap_usd: float = 1_000_000_000.0,
    pages: int = 3,
    cache_hours: float = 6.0,
    active_only: bool = True,
    max_coins: int = 60,
    cache_path: Path | None = None,
    force_refresh: bool = False,
) -> dict:
    cache_target = Path(cache_path or MARKET_CAP_UNIVERSE_JSON).expanduser()
    cached = _load_cache(cache_target)
    max_age_seconds = max(900.0, float(cache_hours or 6.0) * 3600.0)
    catalog = get_hyperliquid_market_catalog(force_refresh=True)

    def _cached_payload_with_live_equities(source: str) -> dict | None:
        base_records = _normalize_records(
            list((cached or {}).get("records") or []),
            catalog=catalog,
            min_market_cap_usd=min_market_cap_usd,
            active_only=active_only,
        )
        merged: dict[str, dict] = {row["coin"]: row for row in base_records}
        if not merged:
            return None
        return _finalize_watchlist_payload(
            candidates=list(merged.values()),
            min_market_cap_usd=min_market_cap_usd,
            active_only=active_only,
            pages=pages,
            max_coins=max_coins,
            source=source,
        )

    if (
        not force_refresh
        and cached
        and (time.time() - _safe_float(cached.get("generated_at_ts"))) <= max_age_seconds
    ):
        return _cached_payload_with_live_equities(str(cached.get("source") or "cached")) or cached

    try:
        market_rows = _fetch_coingecko_market_caps(pages=pages)
    except Exception as exc:
        log.warning("Market-cap universe refresh failed: %s", exc)
        cached_payload = _cached_payload_with_live_equities("cached_plus_live_equities")
        if cached_payload:
            return cached_payload
        return {
            "generated_at_ts": time.time(),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "min_market_cap_usd": float(min_market_cap_usd),
            "coins": [],
            "records": [],
            "source": "empty_fallback",
        }

    strongest_by_symbol: dict[str, dict] = {}
    for row in market_rows:
        symbol = str(row.get("symbol") or "").upper().strip()
        market_cap = _safe_float(row.get("market_cap"))
        if not symbol or market_cap <= 0:
            continue
        current = strongest_by_symbol.get(symbol)
        if current is None or market_cap > _safe_float(current.get("market_cap")):
            strongest_by_symbol[symbol] = row

    candidates: list[dict] = []
    for coin, spec in catalog.items():
        if str(spec.get("market_type") or "").lower() != "perp":
            continue
        if str(spec.get("instrument_type") or "").lower() != "crypto":
            continue
        market_row = strongest_by_symbol.get(str(coin).upper())
        if not market_row:
            continue
        market_cap = _safe_float(market_row.get("market_cap"))
        if market_cap < float(min_market_cap_usd or 0.0):
            continue
        active = hyperliquid_market_is_active(coin) if active_only else True
        if active_only and not active:
            continue
        candidates.append({
            "coin": str(coin).upper(),
            "name": str(market_row.get("name") or coin),
            "symbol": str(market_row.get("symbol") or coin).upper(),
            "market_cap_usd": round(market_cap, 2),
            "market_cap_rank": _rank_or_default(market_row.get("market_cap_rank")),
            "price_change_pct_24h": _safe_float(market_row.get("price_change_percentage_24h")),
            "venue_symbol": str(spec.get("venue_symbol") or coin).strip(),
            "active": bool(active),
        })

    candidates.extend(
        _build_equity_candidates(
            catalog=catalog,
            min_market_cap_usd=min_market_cap_usd,
            active_only=active_only,
        )
    )

    payload = _finalize_watchlist_payload(
        candidates=candidates,
        min_market_cap_usd=min_market_cap_usd,
        active_only=active_only,
        pages=pages,
        max_coins=max_coins,
        source="coingecko_yahoo_x_hyperliquid",
    )
    try:
        _save_cache(cache_target, payload)
    except Exception as exc:
        log.debug("Could not write market-cap universe cache: %s", exc)
    return payload
