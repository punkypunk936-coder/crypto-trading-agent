"""
data/market_data.py — Fetch OHLCV candle data from Hyperliquid's public API.
No API key required for reading market data.

HYPE token note
───────────────
HYPE is Hyperliquid's native token and trades on Hyperliquid itself.
The API ticker is "HYPE" — same format as BTC/ETH/SOL.
Hyperliquid's candleSnapshot API supports it directly.

Index / macro instruments (SP500, BRENT, WTI etc.)
───────────────────────────────
These instruments are not part of the current Lighter execution venue. Their
price data comes from Yahoo Finance (free, no API key) so the agent can still
analyse macro / commodity context in the same loop.
"""

import asyncio
import json
import time
import requests
import pandas as pd
from typing import Optional
from logger import get_logger
from exchanges.lighter_client import COIN_TO_MARKET_ID

log = get_logger("market_data")

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
LIGHTER_API_BASE_URL = "https://mainnet.zklighter.elliot.ai"

# Simple in-memory cache: coin → (timestamp_fetched, DataFrame)
_cache: dict = {}
CACHE_TTL_SECONDS = 60   # re-fetch after 60 s

# Hyperliquid uses these exact tickers — map any aliases
TICKER_MAP = {
    "HYPE": "HYPE",   # Hyperliquid native token — supported directly
    "BTC":  "BTC",
    "ETH":  "ETH",
    "SOL":  "SOL",
}

LIGHTER_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "12h", "1d"}

# Index / macro instruments not served by Lighter.
# Map our internal coin name → Yahoo Finance ticker for fallback OHLCV.
INDEX_YAHOO_MAP = {
    "SP500": "^GSPC",   # S&P 500 — Trade[XYZ] on Hyperliquid
    "BRENT": "BZ=F",    # Brent crude futures
    "WTI":   "CL=F",    # WTI crude futures
    "CL":    "CL=F",    # Alias for WTI
    "NDX":   "^IXIC",
    "DJI":   "^DJI",
    "VIX":   "^VIX",
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
        response = await getattr(api, method_name)(**kwargs)
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
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").tail(lookback).reset_index(drop=True)
    _cache[cache_key] = (now, df)
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
            return None
        return float(trades[0]["price"])
    except Exception as e:
        log.error(f"[{coin}] Lighter price fetch failed: {e}")
        return None


def fetch_candles(
    coin: str,
    interval: str = "1h",
    lookback: int = 100,
) -> Optional[pd.DataFrame]:
    """
    Fetch the last `lookback` OHLCV candles for `coin`.

    For crypto coins: Hyperliquid candleSnapshot API.
    For index coins (SP500 etc.): Yahoo Finance fallback — free, no API key.

    Returns a DataFrame with columns:
        timestamp, open, high, low, close, volume
    Or None if the request fails.
    """
    coin_upper = coin.upper()

    # ── Lighter-supported crypto markets use Lighter public data ───────────
    if coin_upper in COIN_TO_MARKET_ID:
        lighter_df = _fetch_candles_lighter(coin_upper, interval, lookback)
        if lighter_df is not None:
            return lighter_df

    # ── Index instruments: route to Yahoo Finance ──────────────────────────
    if coin_upper in INDEX_YAHOO_MAP:
        return _fetch_candles_yahoo(coin_upper, interval, lookback)

    # ── Crypto: Hyperliquid candleSnapshot ─────────────────────────────────
    hl_coin   = TICKER_MAP.get(coin_upper, coin_upper)
    cache_key = f"{hl_coin}_{interval}"
    now       = time.time()

    if cache_key in _cache:
        ts, df = _cache[cache_key]
        if now - ts < CACHE_TTL_SECONDS:
            log.debug(f"Cache hit for {hl_coin} {interval}")
            return df

    end_ms      = int(now * 1000)
    interval_ms = _interval_to_ms(interval)
    start_ms    = end_ms - lookback * interval_ms

    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin":      hl_coin,
            "interval":  interval,
            "startTime": start_ms,
            "endTime":   end_ms,
        }
    }

    try:
        resp = requests.post(HL_INFO_URL, json=payload, timeout=10)
        resp.raise_for_status()
        raw  = resp.json()
    except Exception as e:
        log.error(f"Failed to fetch candles for {hl_coin}: {e}")
        return None

    if not raw:
        log.warning(f"No candle data returned for {hl_coin} — "
                    f"{'HYPE trades on Hyperliquid; confirm ticker is correct' if hl_coin == 'HYPE' else 'check ticker'}")
        return None

    rows = []
    for c in raw:
        rows.append({
            "timestamp": pd.to_datetime(int(c["t"]), unit="ms", utc=True),
            "open":      float(c["o"]),
            "high":      float(c["h"]),
            "low":       float(c["l"]),
            "close":     float(c["c"]),
            "volume":    float(c["v"]),
            "trades":    int(c.get("n", 0)),
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    _cache[cache_key] = (now, df)
    log.debug(f"Fetched {len(df)} candles for {hl_coin} ({interval})")
    return df


def _fetch_candles_yahoo(coin: str, interval: str, lookback: int) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from Yahoo Finance for index instruments (SP500 etc.).
    Uses the public download endpoint — no API key required.
    """
    yf_ticker = INDEX_YAHOO_MAP.get(coin, "^GSPC")
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

    _cache[cache_key] = (now, df)
    log.info(f"[{coin}] Yahoo Finance: {len(df)} candles ({interval}) | "
             f"last close={df['close'].iloc[-1]:.2f}")
    return df


def get_current_price(coin: str) -> Optional[float]:
    """
    Get the latest price for a coin.
    Index instruments (SP500 etc.) use Yahoo Finance; crypto uses Hyperliquid allMids.
    """
    coin_upper = coin.upper()

    # ── Lighter-supported crypto markets use Lighter public data ───────────
    if coin_upper in COIN_TO_MARKET_ID:
        lighter_price = _get_current_price_lighter(coin_upper)
        if lighter_price is not None:
            return lighter_price

    # ── Index: Yahoo Finance latest close ─────────────────────────────────
    if coin_upper in INDEX_YAHOO_MAP:
        return _get_index_price_yahoo(coin_upper)

    # ── Crypto: Hyperliquid allMids ────────────────────────────────────────
    hl_coin = TICKER_MAP.get(coin_upper, coin_upper)
    try:
        resp  = requests.post(HL_INFO_URL, json={"type": "allMids"}, timeout=5)
        resp.raise_for_status()
        mids  = resp.json()
        price = float(mids.get(hl_coin, 0))
        if price <= 0:
            log.warning(f"Price for {hl_coin} came back 0 — not listed or ticker wrong")
            return None
        return price
    except Exception as e:
        log.error(f"Failed to get price for {hl_coin}: {e}")
        return None


def _get_index_price_yahoo(coin: str) -> Optional[float]:
    """Fetch the latest price for an index instrument from Yahoo Finance."""
    yf_ticker = INDEX_YAHOO_MAP.get(coin, "^GSPC")
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
            return price
        log.warning(f"[{coin}] Yahoo Finance returned no valid price")
        return None
    except Exception as e:
        log.error(f"[{coin}] Yahoo Finance price fetch failed: {e}")
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
