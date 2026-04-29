"""
data/market_data.py
Hyperliquid-first market-data layer.

Active runtime behavior:
  - supported venue symbols use Hyperliquid only
  - unsupported macro instruments can still fall back to Yahoo Finance

Legacy Lighter helpers remain below for historical tooling/tests, but the live
agent path is now Hyperliquid-first and does not quietly fall back away from
the venue for supported markets.
"""

import asyncio
import json
import time
import requests
import pandas as pd
from typing import Optional
from logger import get_logger
from exchanges.hyperliquid_markets import (
    get_hyperliquid_market_dex,
    get_hyperliquid_market_spec,
    is_hyperliquid_supported,
    resolve_hyperliquid_symbol,
)
from exchanges.lighter_client import COIN_TO_MARKET_ID, get_lighter_read_auth_headers

log = get_logger("market_data")

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
LIGHTER_API_BASE_URL = "https://mainnet.zklighter.elliot.ai"

# Simple in-memory cache: coin → (timestamp_fetched, DataFrame)
_cache: dict = {}
_price_cache: dict = {}
_reference_price_cache: dict = {}
CACHE_TTL_SECONDS = 60   # re-fetch after 60 s
STALE_CACHE_TTL_SECONDS = 1800
STALE_PRICE_TTL_SECONDS = 900
REFERENCE_PRICE_TTL_SECONDS = 60
DEFAULT_REFERENCE_MAX_DEVIATION_PCT = 2.0


def _hyperliquid_max_candle_staleness_seconds(coin: str, interval: str) -> int:
    spec = get_hyperliquid_market_spec(coin)
    market_type = str((spec or {}).get("market_type") or "").lower()
    instrument_type = str((spec or {}).get("instrument_type") or "").lower()

    if market_type == "spot" or instrument_type == "equity":
        # Equity spot listings can gap between sessions, but if the venue has
        # not printed a fresh hourly candle in ~2 days we should treat that
        # market as inactive instead of backfilling from elsewhere.
        base = 48 * 3600
    elif instrument_type == "index":
        base = 12 * 3600
    else:
        base = 6 * 3600

    interval_seconds = max(3600, _interval_to_seconds(interval))
    return max(base, int(interval_seconds * 2.0) + 1800)


def _hyperliquid_frame_is_recent(coin: str, interval: str, df: Optional[pd.DataFrame]) -> bool:
    if df is None or df.empty or "timestamp" not in df.columns:
        return False
    try:
        last_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    except Exception:
        return False
    age_seconds = max(0.0, time.time() - (last_ts.value / 1_000_000_000))
    return age_seconds <= _hyperliquid_max_candle_staleness_seconds(coin, interval)


def _cache_price(key: str, price: float) -> None:
    if price and price > 0:
        _price_cache[key] = (time.time(), float(price))


def _get_cached_price(key: str, *, max_age_seconds: int = STALE_PRICE_TTL_SECONDS) -> Optional[float]:
    cached = _price_cache.get(key)
    if not cached:
        return None
    ts, price = cached
    if time.time() - float(ts or 0.0) > max_age_seconds:
        return None
    try:
        value = float(price or 0.0)
    except Exception:
        return None
    return value if value > 0 else None


def _cache_reference_price(key: str, price: float) -> None:
    if price and price > 0:
        _reference_price_cache[key] = (time.time(), float(price))


def _get_cached_reference_price(
    key: str,
    *,
    max_age_seconds: int = REFERENCE_PRICE_TTL_SECONDS,
) -> Optional[float]:
    cached = _reference_price_cache.get(key)
    if not cached:
        return None
    ts, price = cached
    if time.time() - float(ts or 0.0) > max_age_seconds:
        return None
    try:
        value = float(price or 0.0)
    except Exception:
        return None
    return value if value > 0 else None


def _cache_frame(cache_key: str, df: Optional[pd.DataFrame], *, coin: str = "") -> None:
    if df is not None and not df.empty:
        _cache[cache_key] = (time.time(), df)
        try:
            last_close = float(df["close"].iloc[-1])
        except Exception:
            last_close = 0.0
        cache_coin = str(coin or "").upper().strip()
        if last_close > 0 and cache_coin:
            _cache_price(cache_coin, last_close)


def _get_stale_frame(cache_key: str, *, max_age_seconds: int = STALE_CACHE_TTL_SECONDS) -> Optional[pd.DataFrame]:
    cached = _cache.get(cache_key)
    if not cached:
        return None
    ts, df = cached
    if time.time() - float(ts or 0.0) > max_age_seconds:
        return None
    if df is None or df.empty:
        return None
    return df


def completed_candle_frame(df: Optional[pd.DataFrame], *, min_rows: int = 2) -> Optional[pd.DataFrame]:
    """
    Return a dataframe that excludes the still-forming latest candle.

    The live agent loops every 2 minutes while most conviction logic runs on 1H+
    candles. Using the in-progress bar for signal generation creates fake
    breakouts and brittle conviction, so we trim the newest row whenever we have
    enough history to do so safely.
    """
    if df is None:
        return None

    completed = df.copy()
    if len(completed) >= max(2, min_rows):
        completed = completed.iloc[:-1].copy()
    return completed.reset_index(drop=True)


LIGHTER_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "12h", "1d"}

# Yahoo fallbacks for instruments that are not routed to Hyperliquid directly.
INDEX_YAHOO_MAP = {
    "BRENT": "BZ=F",    # Brent crude futures
    "WTI":   "CL=F",    # WTI crude futures
    "CL":    "CL=F",    # Alias for WTI
    "EWY":   "EWY",
    "NDX":   "^IXIC",
    "DJI":   "^DJI",
    "VIX":   "^VIX",
}

EQUITY_YAHOO_MAP = {
    "AAPL": "AAPL",
    "AMZN": "AMZN",
    "GOOGL": "GOOGL",
    "META": "META",
    "MSFT": "MSFT",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "INTC": "INTC",
    "MU": "MU",
    "SNDK": "SNDK",
    "CRWV": "CRWV",
    "HIMS": "HIMS",
}

# Yahoo Finance interval map: our interval → yf period/interval params
_YF_INTERVAL_MAP = {
    "1m":  ("1d",  "1m"),
    "5m":  ("5d",  "5m"),
    "15m": ("5d",  "15m"),
    "30m": ("5d",  "30m"),
    "1h":  ("30d", "1h"),
    "4h":  ("60d", "1h"),   # yf has no 4h; use 1h and resample
    "12h": ("60d", "1h"),   # same — resample
    "1d":  ("1y",  "1d"),
}


def _run_async(coro):
    return asyncio.run(coro)


async def _lighter_api_get(method_name: str, **kwargs):
    import certifi
    import lighter

    api_client = lighter.ApiClient(
        lighter.Configuration(host=LIGHTER_API_BASE_URL, ssl_ca_cert=certifi.where())
    )
    try:
        api_name = "CandlestickApi" if method_name == "candles_without_preload_content" else "OrderApi"
        api = getattr(lighter, api_name)(api_client)
        request_kwargs = dict(kwargs)
        auth_headers = await get_lighter_read_auth_headers(api_base_url=LIGHTER_API_BASE_URL)
        if auth_headers:
            merged_headers = dict(request_kwargs.get("_headers") or {})
            merged_headers.update(auth_headers)
            request_kwargs["_headers"] = merged_headers
        response = await getattr(api, method_name)(**request_kwargs)
        return json.loads(await response.text())
    finally:
        await api_client.close()


def _fetch_candles_lighter(coin: str, interval: str, lookback: int) -> Optional[pd.DataFrame]:
    market_id = COIN_TO_MARKET_ID.get(coin.upper())
    if market_id is None:
        return None

    if interval not in LIGHTER_INTERVALS:
        log.warning(f"[{coin}] Lighter does not support interval {interval} — using Hyperliquid legacy feed")
        return None

    cache_key = f"{coin.upper()}_{interval}_lighter"
    now = time.time()
    if cache_key in _cache:
        ts, df = _cache[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            log.debug(f"[{coin}] Lighter candle cache hit ({interval})")
            return df

    try:
        payload = _run_async(
            _lighter_api_get(
                "candles_without_preload_content",
                market_id=market_id,
                resolution=interval,
                start_timestamp=int(now - _interval_to_seconds(interval) * max(lookback, 1)),
                end_timestamp=int(now),
                count_back=max(lookback, 1),
                set_timestamp_to_end=True,
            )
        )
    except Exception as e:
        log.error(f"[{coin}] Lighter candle fetch failed: {e}")
        stale = _get_stale_frame(cache_key)
        if stale is not None:
            log.info(f"[{coin}] Reusing stale Lighter candles while the live feed recovers")
            return stale
        return None

    candles = payload.get("c", []) or []
    rows = []
    for candle in candles:
        try:
            rows.append({
                "timestamp": pd.to_datetime(int(candle["t"]), unit="ms", utc=True),
                "open": float(candle["o"]),
                "high": float(candle["h"]),
                "low": float(candle["l"]),
                "close": float(candle["c"]),
                "volume": float(candle.get("v", 0) or 0),
                "trades": int(candle.get("i", 0) or 0),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not rows:
        log.warning(f"[{coin}] Lighter returned no valid candles")
        stale = _get_stale_frame(cache_key)
        if stale is not None:
            log.info(f"[{coin}] Reusing stale Lighter candles after empty response")
            return stale
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").tail(lookback).reset_index(drop=True)
    _cache_frame(cache_key, df, coin=coin)
    log.debug(f"[{coin}] Lighter candles: {len(df)} rows ({interval})")
    return df


def _get_current_price_lighter(coin: str) -> Optional[float]:
    market_id = COIN_TO_MARKET_ID.get(coin.upper())
    if market_id is None:
        return None
    try:
        payload = _run_async(
            _lighter_api_get(
                "recent_trades_without_preload_content",
                market_id=market_id,
                limit=1,
            )
        )
        trades = payload.get("trades", []) or []
        if not trades:
            log.warning(f"[{coin}] Lighter returned no recent trades")
            cached = _get_cached_price(coin.upper())
            if cached is not None:
                log.info(f"[{coin}] Reusing cached venue price after empty trades feed")
                return cached
            return None
        price = float(trades[0]["price"])
        _cache_price(coin.upper(), price)
        return price
    except Exception as e:
        log.error(f"[{coin}] Lighter price fetch failed: {e}")
        cached = _get_cached_price(coin.upper())
        if cached is not None:
            log.info(f"[{coin}] Reusing cached venue price while Lighter price feed recovers")
            return cached
        return None


def fetch_candles(
    coin: str,
    interval: str = "1h",
    lookback: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Fetch the last `lookback` OHLCV candles for `coin`.

    Supported venue markets use Hyperliquid candleSnapshot only.
    Unsupported macro instruments still fall back to Yahoo Finance.

    Returns a DataFrame with columns:
        timestamp, open, high, low, close, volume
    Or None if the request fails.
    """
    coin_upper = coin.upper()

    # ── Hyperliquid-first venue data ───────────────────────────────────────
    if is_hyperliquid_supported(coin_upper):
        hl_coin = resolve_hyperliquid_symbol(coin_upper)
        dex = get_hyperliquid_market_dex(coin_upper)
        cache_key = f"{hl_coin}_{interval}"
        now = time.time()

        if cache_key in _cache:
            ts, df = _cache[cache_key]
            if now - ts < CACHE_TTL_SECONDS:
                log.debug(f"Cache hit for {hl_coin} {interval}")
                return df

        end_ms = int(now * 1000)
        interval_ms = _interval_to_ms(interval)
        start_ms = end_ms - lookback * interval_ms

        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": hl_coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": end_ms,
            },
        }
        if dex:
            payload["dex"] = dex

        try:
            resp = requests.post(HL_INFO_URL, json=payload, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            log.error(f"Failed to fetch Hyperliquid candles for {coin_upper}/{hl_coin}: {e}")
            stale = _get_stale_frame(cache_key)
            if stale is not None:
                log.info(f"[{coin_upper}] Reusing stale Hyperliquid candles while the live feed recovers")
                return stale
            raw = []

        if raw:
            rows = []
            for c in raw:
                rows.append({
                    "timestamp": pd.to_datetime(int(c["t"]), unit="ms", utc=True),
                    "open": float(c["o"]),
                    "high": float(c["h"]),
                    "low": float(c["l"]),
                    "close": float(c["c"]),
                    "volume": float(c["v"]),
                    "trades": int(c.get("n", 0)),
                })

            df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
            if not _hyperliquid_frame_is_recent(coin_upper, interval, df):
                log.warning(
                    f"[{coin_upper}] Hyperliquid candle history for {hl_coin} is stale — "
                    "skipping until the venue prints fresh candles"
                )
                stale = _get_stale_frame(cache_key)
                if stale is not None and _hyperliquid_frame_is_recent(coin_upper, interval, stale):
                    log.info(f"[{coin_upper}] Reusing recent cached Hyperliquid candles while the venue recovers")
                    return stale
                return None
            _cache_frame(cache_key, df, coin=coin_upper)
            log.debug(f"Fetched {len(df)} Hyperliquid candles for {coin_upper}/{hl_coin} ({interval})")
            return df

        log.warning(f"[{coin_upper}] Hyperliquid returned no candle data for {hl_coin}")
        stale = _get_stale_frame(cache_key)
        if stale is not None and _hyperliquid_frame_is_recent(coin_upper, interval, stale):
            log.info(f"[{coin_upper}] Reusing recent cached Hyperliquid candles after empty venue response")
            return stale
        return None

    # ── Unsupported macro instruments: route to Yahoo Finance ──────────────
    if coin_upper in INDEX_YAHOO_MAP:
        return _fetch_candles_yahoo(coin_upper, interval, lookback)
    return None


def _fetch_candles_yahoo(
    coin: str,
    interval: str,
    lookback: int,
    *,
    override_ticker: str | None = None,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Yahoo Finance for index instruments (SP500 etc.).
    Uses the public download endpoint — no API key required.
    """
    yf_ticker = override_ticker or INDEX_YAHOO_MAP.get(coin) or EQUITY_YAHOO_MAP.get(coin) or "^GSPC"
    cache_key = f"{coin}_{interval}_yf"
    now = time.time()

    if cache_key in _cache:
        ts, df = _cache[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            log.debug(f"[{coin}] Yahoo Finance cache hit ({interval})")
            return df

    period, yf_interval = _YF_INTERVAL_MAP.get(interval, ("60d", "1h"))
    resample_to = None
    if interval in ("4h", "12h"):
        resample_to = interval   # we fetched 1h; resample after

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
        f"?interval={yf_interval}&range={period}&includePrePost=false"
    )
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        ts_list  = result["timestamp"]
        q        = result["indicators"]["quote"][0]
        opens    = q["open"]
        highs    = q["high"]
        lows     = q["low"]
        closes   = q["close"]
        volumes  = q["volume"]
    except Exception as e:
        log.error(f"[{coin}] Yahoo Finance candle fetch failed: {e}")
        stale = _get_stale_frame(cache_key)
        if stale is not None:
            log.info(f"[{coin}] Reusing stale Yahoo candles while Yahoo recovers")
            return stale
        return None

    rows = []
    for i, t in enumerate(ts_list):
        try:
            rows.append({
                "timestamp": pd.to_datetime(t, unit="s", utc=True),
                "open":      float(opens[i]   or 0),
                "high":      float(highs[i]   or 0),
                "low":       float(lows[i]    or 0),
                "close":     float(closes[i]  or 0),
                "volume":    float(volumes[i] or 0),
                "trades":    0,
            })
        except (TypeError, ValueError):
            continue

    if not rows:
        log.warning(f"[{coin}] Yahoo Finance returned no rows")
        stale = _get_stale_frame(cache_key)
        if stale is not None:
            log.info(f"[{coin}] Reusing stale Yahoo candles after empty response")
            return stale
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df = df[df["close"] > 0]   # drop bad rows

    # Resample 1h → 4h / 12h if needed
    if resample_to:
        rule = "4h" if resample_to == "4h" else "12h"
        df = df.set_index("timestamp").resample(rule).agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
            "trades": "sum",
        }).dropna().reset_index()

    # Trim to requested lookback
    df = df.tail(lookback).reset_index(drop=True)

    _cache_frame(cache_key, df, coin=coin)
    log.info(f"[{coin}] Yahoo Finance: {len(df)} candles ({interval}) | "
             f"last close={df['close'].iloc[-1]:.2f}")
    return df


def get_current_price(coin: str) -> Optional[float]:
    """
    Get the latest price for a coin.
    Supported venue markets use Hyperliquid allMids.
    Unsupported macro instruments still use Yahoo Finance.
    """
    coin_upper = coin.upper()

    if is_hyperliquid_supported(coin_upper):
        hl_coin = resolve_hyperliquid_symbol(coin_upper)
        dex = get_hyperliquid_market_dex(coin_upper)
        try:
            payload = {"type": "allMids"}
            if dex:
                payload["dex"] = dex
            resp = requests.post(HL_INFO_URL, json=payload, timeout=5)
            resp.raise_for_status()
            mids = resp.json()
            price = float(mids.get(hl_coin, 0) or 0.0)
            if price > 0:
                _cache_price(coin_upper, price)
                return price
            log.warning(f"Price for {coin_upper}/{hl_coin} came back 0 — venue returned no mid")
        except Exception as e:
            log.error(f"Failed to get Hyperliquid price for {coin_upper}/{hl_coin}: {e}")

        cached = _get_cached_price(coin_upper)
        if cached is not None:
            log.info(f"[{coin_upper}] Reusing cached Hyperliquid price while primary feed recovers")
            return cached
        return None

    # ── Macro fallback: Yahoo Finance latest close ─────────────────────────
    if coin_upper in INDEX_YAHOO_MAP:
        return _get_index_price_yahoo(coin_upper)
    return None


def get_reference_price_yahoo(coin: str) -> Optional[float]:
    """
    Fetch a reference quote without touching the executable venue price cache.

    Equity markets can execute on Hyperliquid/Trade.xyz while the operator
    naturally compares them against a public equity quote. Keep that comparison
    in a separate cache so a Yahoo reference never silently replaces the venue
    price used for execution.
    """
    coin_upper = str(coin or "").upper().strip()
    yf_ticker = INDEX_YAHOO_MAP.get(coin_upper) or EQUITY_YAHOO_MAP.get(coin_upper)
    if not yf_ticker:
        return None

    cache_key = f"{coin_upper}:yahoo_reference"
    cached = _get_cached_reference_price(cache_key)
    if cached is not None:
        return cached

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
        f"?interval=1m&range=1d&includePrePost=true"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        closes = result["indicators"]["quote"][0]["close"]
        price = next((float(c) for c in reversed(closes) if c), None)
        if price and price > 0:
            _cache_reference_price(cache_key, price)
            return price
        log.warning(f"[{coin_upper}] Yahoo reference quote returned no valid price")
        return None
    except Exception as e:
        log.debug(f"[{coin_upper}] Yahoo reference quote fetch failed: {e}")
        return _get_cached_reference_price(cache_key, max_age_seconds=STALE_PRICE_TTL_SECONDS)


def get_price_diagnostics(
    coin: str,
    *,
    venue_price: Optional[float] = None,
    max_deviation_pct: float = DEFAULT_REFERENCE_MAX_DEVIATION_PCT,
) -> dict:
    """Return source, venue symbol, and reference-price health for UI/state."""
    coin_upper = str(coin or "").upper().strip()
    if not coin_upper:
        return {
            "price_status": "UNKNOWN",
            "price_source": "",
            "price_source_label": "Unknown source",
            "price_warning": "Missing symbol for price diagnostics.",
        }

    supported = is_hyperliquid_supported(coin_upper)
    hl_coin = resolve_hyperliquid_symbol(coin_upper) if supported else ""
    dex = get_hyperliquid_market_dex(coin_upper) if supported else ""
    venue_name = "Trade.xyz" if str(dex or "").lower() == "xyz" else "Hyperliquid"
    source = f"{venue_name} allMids" if supported else "Yahoo Finance"
    source_label = f"{venue_name} {hl_coin}".strip() if supported else "Yahoo Finance"

    price = None
    try:
        price = float(venue_price or 0.0)
    except Exception:
        price = 0.0
    if price <= 0:
        price = get_current_price(coin_upper) if supported or coin_upper in INDEX_YAHOO_MAP else None

    reference_price = get_reference_price_yahoo(coin_upper)
    deviation_pct = None
    status = "OK"
    warning = ""
    if price and reference_price:
        deviation_pct = (float(price) - float(reference_price)) / float(reference_price) * 100.0
        if abs(deviation_pct) > float(max_deviation_pct or DEFAULT_REFERENCE_MAX_DEVIATION_PCT):
            status = "CHECK"
            warning = (
                f"{coin_upper} venue price is {deviation_pct:+.2f}% away from "
                f"Yahoo reference."
            )
    elif supported and coin_upper in EQUITY_YAHOO_MAP:
        status = "REFERENCE_MISSING"
        warning = "Yahoo reference quote unavailable; showing executable venue price only."

    return {
        "price_status": status,
        "price_source": source,
        "price_source_label": source_label,
        "venue": venue_name if supported else "Yahoo Finance",
        "venue_symbol": hl_coin,
        "venue_price": round(float(price), 6) if price else 0.0,
        "reference_price": round(float(reference_price), 6) if reference_price else 0.0,
        "reference_source": "Yahoo Finance" if reference_price else "",
        "price_deviation_pct": round(float(deviation_pct), 4) if deviation_pct is not None else None,
        "price_warning": warning,
    }


def _get_index_price_yahoo(coin: str, *, override_ticker: str | None = None) -> Optional[float]:
    """Fetch the latest price for an index instrument from Yahoo Finance."""
    yf_ticker = override_ticker or INDEX_YAHOO_MAP.get(coin) or EQUITY_YAHOO_MAP.get(coin) or "^GSPC"
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{yf_ticker}"
        f"?interval=1m&range=1d&includePrePost=false"
    )
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        data   = resp.json()
        result = data["chart"]["result"][0]
        closes = result["indicators"]["quote"][0]["close"]
        # Last non-null close
        price = next((float(c) for c in reversed(closes) if c), None)
        if price and price > 0:
            log.debug(f"[{coin}] Yahoo Finance price: {price:.2f}")
            _cache_price(coin.upper(), price)
            return price
        log.warning(f"[{coin}] Yahoo Finance returned no valid price")
        cached = _get_cached_price(coin.upper())
        if cached is not None:
            log.info(f"[{coin}] Reusing cached Yahoo price after empty response")
            return cached
        return None
    except Exception as e:
        log.error(f"[{coin}] Yahoo Finance price fetch failed: {e}")
        cached = _get_cached_price(coin.upper())
        if cached is not None:
            log.info(f"[{coin}] Reusing cached Yahoo price while feed recovers")
            return cached
        return None


def _interval_to_ms(interval: str) -> int:
    """Convert interval string like '1h', '15m', '4h' to milliseconds."""
    unit_map = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    try:
        num  = int(interval[:-1])
        unit = interval[-1]
        return num * unit_map[unit] * 1000
    except Exception:
        return 3600 * 1000   # default to 1h


def _interval_to_seconds(interval: str) -> int:
    return max(1, _interval_to_ms(interval) // 1000)
