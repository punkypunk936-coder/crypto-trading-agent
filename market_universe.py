"""
market_universe.py — build a broader Hyperliquid scout universe from market-cap filters.

The goal is to widen observation without automatically widening execution.
We intersect Hyperliquid's live perp universe with large-cap crypto assets so
the agent can scout respectable names first and only promote them later if the
operator wants that.
"""

from __future__ import annotations

import json
import time
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
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
    return rows


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
    if (
        not force_refresh
        and cached
        and (time.time() - _safe_float(cached.get("generated_at_ts"))) <= max_age_seconds
    ):
        return cached

    try:
        market_rows = _fetch_coingecko_market_caps(pages=pages)
    except Exception as exc:
        log.warning("Market-cap universe refresh failed: %s", exc)
        if cached:
            return cached
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

    catalog = get_hyperliquid_market_catalog(force_refresh=True)
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
            "market_cap_rank": _safe_int(market_row.get("market_cap_rank")),
            "price_change_pct_24h": _safe_float(market_row.get("price_change_percentage_24h")),
            "venue_symbol": str(spec.get("venue_symbol") or coin).upper(),
            "active": bool(active),
        })

    candidates.sort(
        key=lambda item: (
            _safe_int(item.get("market_cap_rank"), 999999),
            -_safe_float(item.get("market_cap_usd")),
            str(item.get("coin")),
        )
    )
    limited = candidates[: max(1, int(max_coins or 60))]

    payload = {
        "generated_at_ts": time.time(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "min_market_cap_usd": float(min_market_cap_usd),
        "active_only": bool(active_only),
        "pages": int(pages or 1),
        "source": "coingecko_x_hyperliquid",
        "coins": [row["coin"] for row in limited],
        "records": limited,
    }
    try:
        _save_cache(cache_target, payload)
    except Exception as exc:
        log.debug("Could not write market-cap universe cache: %s", exc)
    return payload
