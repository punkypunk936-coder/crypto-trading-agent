"""
indicators/orderbook_levels.py
Live orderbook + higher-timeframe key-level intelligence.

Purpose:
  - Identify strong bid/ask walls from the active Lighter orderbook
  - Derive higher-timeframe key levels from daily swing highs/lows
  - Classify whether price is testing support, pressing resistance,
    or breaking a major level with/without daily confirmation

This module intentionally does not produce entries by itself. It gives the
strategy a "where are we relative to important liquidity + structural levels?"
answer so the agent avoids weak counter-level trades.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data.market_data import fetch_candles
from exchanges.lighter_client import (
    COIN_TO_MARKET_ID,
    DEFAULT_LIGHTER_API_BASE_URL,
    get_lighter_read_auth_headers,
)
from logger import get_logger

log = get_logger("orderbook_levels")

_CACHE: Dict[Tuple[str, int, int], Tuple[float, "OrderBookLevelSignal"]] = {}


@dataclass
class OrderBookLevel:
    price: float
    strength: float
    source: str
    label: str
    distance_pct: float = 0.0


@dataclass
class OrderBookLevelSignal:
    coin: str
    valid: bool = False
    score: float = 50.0
    description: str = ""
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    spread_bps: float = 0.0
    bid_notional: float = 0.0
    ask_notional: float = 0.0
    imbalance_ratio: float = 0.0
    nearest_support: float = 0.0
    nearest_support_strength: float = 0.0
    nearest_support_distance_pct: float = 0.0
    nearest_resistance: float = 0.0
    nearest_resistance_strength: float = 0.0
    nearest_resistance_distance_pct: float = 0.0
    daily_breakout_level: float = 0.0
    daily_breakdown_level: float = 0.0
    last_daily_close: float = 0.0
    breakout_state: str = "NONE"
    level_interaction: str = "BETWEEN_LEVELS"
    favor_longs: bool = False
    favor_shorts: bool = False
    block_longs: bool = False
    block_shorts: bool = False
    feed_age_seconds: float = 0.0
    feed_snapshot_count: int = 0
    imbalance_mean: float = 0.0
    imbalance_trend: float = 0.0
    imbalance_volatility: float = 0.0
    support_wall_persistence: int = 0
    resistance_wall_persistence: int = 0
    intracycle_breakout_state: str = "NONE"
    support_levels: List[Dict[str, float | str]] = field(default_factory=list)
    resistance_levels: List[Dict[str, float | str]] = field(default_factory=list)


@dataclass
class OrderBookFeedSnapshot:
    coin: str
    ts: float
    valid: bool = False
    ref_price: float = 0.0
    last_trade_price: float = 0.0
    price_decimals: int = 1
    book: Dict[str, Any] = field(default_factory=dict)
    bid_notional: float = 0.0
    ask_notional: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread_bps: float = 0.0
    imbalance_ratio: float = 0.0
    strongest_bid_wall: float = 0.0
    strongest_bid_wall_notional: float = 0.0
    strongest_ask_wall: float = 0.0
    strongest_ask_wall_notional: float = 0.0
    error: str = ""


class BackgroundOrderBookFeed:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._symbols: List[str] = []
        self._depth_limit: int = 120
        self._poll_interval_seconds: float = 3.0
        self._history_size: int = 120
        self._histories: Dict[str, deque[OrderBookFeedSnapshot]] = {}

    def configure(
        self,
        symbols: List[str],
        *,
        depth_limit: int = 120,
        poll_interval_seconds: float = 3.0,
        history_size: int = 120,
    ) -> None:
        filtered = sorted({str(symbol).upper() for symbol in symbols if str(symbol).upper() in COIN_TO_MARKET_ID})
        with self._lock:
            self._symbols = filtered
            self._depth_limit = max(20, min(int(depth_limit or 120), 250))
            self._poll_interval_seconds = max(0.5, float(poll_interval_seconds or 3.0))
            self._history_size = max(10, int(history_size or 120))
            for coin in filtered:
                history = self._histories.get(coin)
                if history is None or history.maxlen != self._history_size:
                    self._histories[coin] = deque(history or [], maxlen=self._history_size)

    def prime(self) -> None:
        symbols, depth_limit = self._snapshot_config()
        for coin in symbols:
            snapshot = _fetch_live_snapshot(coin, depth_limit=depth_limit)
            if snapshot.valid:
                self._store_snapshot(snapshot)

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if not self._symbols:
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="orderbook-feed",
                daemon=True,
            )
            self._thread.start()
        log.info("Orderbook feed started for %s", ", ".join(self._symbols))

    def stop(self, timeout: float = 5.0) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        if not thread:
            return
        self._stop_event.set()
        thread.join(timeout=timeout)
        self._stop_event.clear()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def get_recent_snapshots(
        self,
        coin: str,
        *,
        max_age_seconds: float = 45.0,
        limit: Optional[int] = None,
    ) -> List[OrderBookFeedSnapshot]:
        coin = coin.upper()
        now = time.time()
        with self._lock:
            history = list(self._histories.get(coin, []))
        if max_age_seconds > 0:
            history = [item for item in history if (now - item.ts) <= max_age_seconds]
        if limit:
            history = history[-int(limit):]
        return history

    def clear(self) -> None:
        self.stop()
        with self._lock:
            self._histories.clear()
            self._symbols = []

    def _snapshot_config(self) -> Tuple[List[str], int]:
        with self._lock:
            return list(self._symbols), int(self._depth_limit)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            symbols, depth_limit = self._snapshot_config()
            for coin in symbols:
                if self._stop_event.is_set():
                    break
                snapshot = _fetch_live_snapshot(coin, depth_limit=depth_limit)
                if snapshot.valid:
                    self._store_snapshot(snapshot)
            elapsed = time.time() - started
            with self._lock:
                sleep_for = max(0.5, self._poll_interval_seconds - elapsed)
            self._stop_event.wait(sleep_for)

    def _store_snapshot(self, snapshot: OrderBookFeedSnapshot) -> None:
        with self._lock:
            history = self._histories.get(snapshot.coin)
            if history is None or history.maxlen != self._history_size:
                history = deque(history or [], maxlen=self._history_size)
                self._histories[snapshot.coin] = history
            history.append(snapshot)


_BACKGROUND_ORDERBOOK_FEED = BackgroundOrderBookFeed()


def configure_background_orderbook_feed(
    symbols: List[str],
    *,
    depth_limit: int = 120,
    poll_interval_seconds: float = 3.0,
    history_size: int = 120,
) -> None:
    _BACKGROUND_ORDERBOOK_FEED.configure(
        symbols,
        depth_limit=depth_limit,
        poll_interval_seconds=poll_interval_seconds,
        history_size=history_size,
    )


def prime_background_orderbook_feed() -> None:
    _BACKGROUND_ORDERBOOK_FEED.prime()


def start_background_orderbook_feed() -> None:
    _BACKGROUND_ORDERBOOK_FEED.start()


def stop_background_orderbook_feed() -> None:
    _BACKGROUND_ORDERBOOK_FEED.stop()


def _run_async(coro):
    return asyncio.run(coro)


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    try:
        return json.loads(str(obj))
    except Exception:
        return {}


async def _lighter_orderbook_orders(market_id: int, limit: int) -> Dict[str, Any]:
    import certifi
    import lighter

    api_client = lighter.ApiClient(
        lighter.Configuration(host=DEFAULT_LIGHTER_API_BASE_URL, ssl_ca_cert=certifi.where())
    )
    try:
        api = lighter.OrderApi(api_client)
        auth_headers = await get_lighter_read_auth_headers(api_base_url=DEFAULT_LIGHTER_API_BASE_URL)
        request_kwargs: Dict[str, Any] = {"market_id": market_id, "limit": limit}
        if auth_headers:
            request_kwargs["_headers"] = auth_headers
        payload = await api.order_book_orders(**request_kwargs)
        return _safe_to_dict(payload)
    finally:
        await api_client.close()


async def _lighter_orderbook_details(market_id: int) -> Dict[str, Any]:
    import certifi
    import lighter

    api_client = lighter.ApiClient(
        lighter.Configuration(host=DEFAULT_LIGHTER_API_BASE_URL, ssl_ca_cert=certifi.where())
    )
    try:
        api = lighter.OrderApi(api_client)
        auth_headers = await get_lighter_read_auth_headers(api_base_url=DEFAULT_LIGHTER_API_BASE_URL)
        request_kwargs: Dict[str, Any] = {"market_id": market_id}
        if auth_headers:
            request_kwargs["_headers"] = auth_headers
        payload = await api.order_book_details(**request_kwargs)
        return _safe_to_dict(payload)
    finally:
        await api_client.close()


def _round_step(price: float) -> float:
    if price >= 50_000:
        return 500.0
    if price >= 5_000:
        return 50.0
    if price >= 1_000:
        return 25.0
    if price >= 100:
        return 5.0
    if price >= 20:
        return 1.0
    if price >= 5:
        return 0.25
    return 0.05


def _cluster_points(
    points: Iterable[Tuple[float, float, str, str]],
    tolerance_pct: float,
) -> List[Dict[str, Any]]:
    clusters: List[Dict[str, Any]] = []
    for price, strength, source, label in sorted(points, key=lambda item: item[0]):
        if price <= 0 or strength <= 0:
            continue
        if clusters:
            last = clusters[-1]
            tolerance = max(price, last["price"]) * tolerance_pct / 100.0
            if abs(price - last["price"]) <= tolerance:
                total_strength = last["strength"] + strength
                last["price"] = (
                    (last["price"] * last["strength"]) + (price * strength)
                ) / max(total_strength, 1e-9)
                last["strength"] = total_strength
                last["touches"] += 1
                last["sources"].add(source)
                last["labels"].add(label)
                continue
        clusters.append({
            "price": price,
            "strength": strength,
            "touches": 1,
            "sources": {source},
            "labels": {label},
        })
    return clusters


def _choose_level(
    current_price: float,
    levels: List[Dict[str, Any]],
    side: str,
) -> Tuple[float, float, float]:
    candidates = []
    for level in levels:
        price = _float(level.get("price"))
        strength = _float(level.get("strength"))
        if price <= 0 or strength <= 0:
            continue
        if side == "support" and price >= current_price:
            continue
        if side == "resistance" and price <= current_price:
            continue
        distance_pct = abs(current_price - price) / max(current_price, 1e-9) * 100
        score = strength / max(distance_pct + 0.12, 0.12)
        candidates.append((score, price, strength, distance_pct))

    if not candidates:
        return 0.0, 0.0, 0.0

    _, price, strength, distance_pct = max(candidates, key=lambda item: item[0])
    return price, min(1.5, strength), distance_pct


def _find_daily_pivots(df, strength: int = 2) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    highs = list(df["high"])
    lows = list(df["low"])
    n = len(df)
    swing_highs: List[Tuple[float, float]] = []
    swing_lows: List[Tuple[float, float]] = []

    for i in range(strength, n - strength):
        high = highs[i]
        low = lows[i]
        if all(high >= highs[i - j] and high >= highs[i + j] for j in range(1, strength + 1)):
            recency = 1.0 - ((n - 1 - i) / max(n - 1, 1))
            swing_highs.append((high, 0.55 + recency * 0.30))
        if all(low <= lows[i - j] and low <= lows[i + j] for j in range(1, strength + 1)):
            recency = 1.0 - ((n - 1 - i) / max(n - 1, 1))
            swing_lows.append((low, 0.55 + recency * 0.30))
    return swing_highs, swing_lows


def _daily_levels(coin: str, current_price: float, daily_lookback: int) -> Tuple[List[Tuple[float, float, str, str]], List[Tuple[float, float, str, str]], float]:
    df = fetch_candles(coin, interval="1d", lookback=max(daily_lookback, 40))
    if df is None or df.empty or len(df) < 8:
        return [], [], 0.0

    completed = df.iloc[:-1].copy() if len(df) >= 2 else df.copy()
    if completed.empty:
        return [], [], 0.0

    supports: List[Tuple[float, float, str, str]] = []
    resistances: List[Tuple[float, float, str, str]] = []
    last_close = float(completed["close"].iloc[-1])
    prev_high = float(completed["high"].iloc[-1])
    prev_low = float(completed["low"].iloc[-1])
    window20 = completed.tail(min(20, len(completed)))
    swing_highs, swing_lows = _find_daily_pivots(completed.tail(min(90, len(completed))))
    tolerance_pct = 0.45 if current_price >= 1_000 else 0.70

    for price, strength in swing_lows:
        supports.append((price, strength, "daily", "daily_swing_low"))
    for price, strength in swing_highs:
        resistances.append((price, strength, "daily", "daily_swing_high"))

    supports.extend([
        (prev_low, 0.70, "daily", "prev_day_low"),
        (float(window20["low"].min()), 0.80, "daily", "20d_low"),
    ])
    resistances.extend([
        (prev_high, 0.70, "daily", "prev_day_high"),
        (float(window20["high"].max()), 0.80, "daily", "20d_high"),
    ])

    step = _round_step(current_price)
    rounded_anchor = math.floor(current_price / step) * step
    for offset in range(-20, 21):
        level = rounded_anchor + offset * step
        if level <= 0:
            continue
        strength = 0.30 if abs(offset) > 6 else 0.42
        if level < current_price:
            supports.append((level, strength, "round", "round_level"))
        elif level > current_price:
            resistances.append((level, strength, "round", "round_level"))

    support_clusters = _cluster_points(supports, tolerance_pct)
    resistance_clusters = _cluster_points(resistances, tolerance_pct)

    return (
        [
            (cluster["price"], min(1.35, 0.35 + 0.18 * cluster["touches"] + cluster["strength"] * 0.35),
             ",".join(sorted(cluster["sources"])), ",".join(sorted(cluster["labels"])))
            for cluster in support_clusters
        ],
        [
            (cluster["price"], min(1.35, 0.35 + 0.18 * cluster["touches"] + cluster["strength"] * 0.35),
             ",".join(sorted(cluster["sources"])), ",".join(sorted(cluster["labels"])))
            for cluster in resistance_clusters
        ],
        last_close,
    )


def _bucket_orderbook(
    book: Dict[str, Any],
    current_price: float,
    price_decimals: int,
) -> Tuple[
    Dict[float, float],
    Dict[float, float],
    float,
    float,
    float,
    float,
    float,
]:
    bids = list(book.get("bids", []) or [])
    asks = list(book.get("asks", []) or [])
    best_bid = max((_float(item.get("price")) for item in bids), default=0.0)
    best_ask = min((_float(item.get("price")) for item in asks if _float(item.get("price")) > 0), default=0.0)

    anchor_price = current_price
    if anchor_price <= 0 and best_bid > 0 and best_ask > 0:
        anchor_price = (best_bid + best_ask) / 2.0
    if anchor_price <= 0:
        anchor_price = best_bid or best_ask or 1.0

    step = _round_step(anchor_price)
    bucket_size = max(step / 5.0, 10 ** (-price_decimals) * 10)

    bid_buckets: Dict[float, float] = {}
    ask_buckets: Dict[float, float] = {}
    bid_total = 0.0
    ask_total = 0.0

    for item in bids:
        price = _float(item.get("price"))
        amount = _float(item.get("remaining_base_amount") or item.get("initial_base_amount"))
        if price <= 0 or amount <= 0 or price >= anchor_price:
            continue
        notional = price * amount
        bid_total += notional
        bucket = round(price / bucket_size) * bucket_size
        bid_buckets[bucket] = bid_buckets.get(bucket, 0.0) + notional

    for item in asks:
        price = _float(item.get("price"))
        amount = _float(item.get("remaining_base_amount") or item.get("initial_base_amount"))
        if price <= 0 or amount <= 0 or price <= anchor_price:
            continue
        notional = price * amount
        ask_total += notional
        bucket = round(price / bucket_size) * bucket_size
        ask_buckets[bucket] = ask_buckets.get(bucket, 0.0) + notional

    spread_bps = 0.0
    if best_bid > 0 and best_ask > 0:
        mid = (best_bid + best_ask) / 2.0
        spread_bps = abs(best_ask - best_bid) / max(mid, 1e-9) * 10_000

    return bid_buckets, ask_buckets, bid_total, ask_total, best_bid, best_ask if best_ask > 0 else 0.0, spread_bps


def _orderbook_levels(
    book: Dict[str, Any],
    current_price: float,
    price_decimals: int,
) -> Tuple[
    List[Tuple[float, float, str, str]],
    List[Tuple[float, float, str, str]],
    float,
    float,
    float,
    float,
    float,
]:
    bid_buckets, ask_buckets, bid_total, ask_total, best_bid, best_ask, spread_bps = _bucket_orderbook(
        book,
        current_price,
        price_decimals,
    )

    max_bid = max(bid_buckets.values(), default=0.0)
    max_ask = max(ask_buckets.values(), default=0.0)

    supports = [
        (price, min(1.25, (notional / max(max_bid, 1e-9)) * 1.10), "orderbook", "bid_wall")
        for price, notional in bid_buckets.items() if notional > 0
    ]
    resistances = [
        (price, min(1.25, (notional / max(max_ask, 1e-9)) * 1.10), "orderbook", "ask_wall")
        for price, notional in ask_buckets.items() if notional > 0
    ]
    return supports, resistances, bid_total, ask_total, best_bid, best_ask, spread_bps


def _infer_ref_price(last_trade_price: float, best_bid: float, best_ask: float) -> float:
    if last_trade_price > 0:
        return last_trade_price
    if best_bid > 0 and best_ask > 0:
        return (best_bid + best_ask) / 2.0
    return best_bid or best_ask or 0.0


def _fetch_live_snapshot(coin: str, depth_limit: int) -> OrderBookFeedSnapshot:
    coin = coin.upper()
    snapshot = OrderBookFeedSnapshot(coin=coin, ts=time.time(), valid=False)
    market_id = COIN_TO_MARKET_ID.get(coin)
    if market_id is None:
        snapshot.error = "No public orderbook venue configured for this symbol"
        return snapshot

    try:
        details = _run_async(_lighter_orderbook_details(market_id))
        book = _run_async(_lighter_orderbook_orders(market_id, max(20, min(depth_limit, 250))))
    except Exception as exc:
        snapshot.error = str(exc)
        log.debug("[%s] Background orderbook fetch failed: %s", coin, exc)
        return snapshot

    detail_rows = list(details.get("order_book_details", []) or [])
    detail = detail_rows[0] if detail_rows else {}
    last_trade_price = _float(detail.get("last_trade_price"))
    price_decimals = int(detail.get("price_decimals", 1) or 1)
    bid_buckets, ask_buckets, bid_notional, ask_notional, best_bid, best_ask, spread_bps = _bucket_orderbook(
        book,
        last_trade_price,
        price_decimals,
    )
    ref_price = _infer_ref_price(last_trade_price, best_bid, best_ask)
    total_flow = bid_notional + ask_notional
    imbalance_ratio = (bid_notional - ask_notional) / max(total_flow, 1e-9) if total_flow > 0 else 0.0
    strongest_bid_wall, strongest_bid_notional = max(bid_buckets.items(), key=lambda item: item[1], default=(0.0, 0.0))
    strongest_ask_wall, strongest_ask_notional = max(ask_buckets.items(), key=lambda item: item[1], default=(0.0, 0.0))

    snapshot.valid = ref_price > 0
    snapshot.ref_price = round(ref_price, 6) if ref_price > 0 else 0.0
    snapshot.last_trade_price = round(last_trade_price, 6) if last_trade_price > 0 else 0.0
    snapshot.price_decimals = price_decimals
    snapshot.book = book
    snapshot.bid_notional = round(bid_notional, 2)
    snapshot.ask_notional = round(ask_notional, 2)
    snapshot.best_bid = round(best_bid, 6) if best_bid > 0 else 0.0
    snapshot.best_ask = round(best_ask, 6) if best_ask > 0 else 0.0
    snapshot.spread_bps = round(spread_bps, 3)
    snapshot.imbalance_ratio = round(imbalance_ratio, 4)
    snapshot.strongest_bid_wall = round(strongest_bid_wall, 6) if strongest_bid_wall > 0 else 0.0
    snapshot.strongest_bid_wall_notional = round(strongest_bid_notional, 2)
    snapshot.strongest_ask_wall = round(strongest_ask_wall, 6) if strongest_ask_wall > 0 else 0.0
    snapshot.strongest_ask_wall_notional = round(strongest_ask_notional, 2)
    return snapshot


def _serialize_levels(
    levels: List[Tuple[float, float, str, str]],
    current_price: float,
    side: str,
    limit: int = 4,
) -> List[Dict[str, float | str]]:
    if side == "support":
        ordered = sorted((lvl for lvl in levels if lvl[0] < current_price), key=lambda item: item[0], reverse=True)
    else:
        ordered = sorted((lvl for lvl in levels if lvl[0] > current_price), key=lambda item: item[0])

    out: List[Dict[str, float | str]] = []
    for price, strength, source, label in ordered[:limit]:
        distance_pct = abs(current_price - price) / max(current_price, 1e-9) * 100
        out.append({
            "price": round(price, 6),
            "strength": round(min(1.5, strength), 3),
            "source": source,
            "label": label,
            "distance_pct": round(distance_pct, 3),
        })
    return out


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    variance = sum((value - mu) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _wall_persistence(
    history: List[OrderBookFeedSnapshot],
    *,
    side: str,
) -> int:
    if not history:
        return 0
    latest = history[-1]
    if side == "bid":
        latest_price = float(latest.strongest_bid_wall or 0.0)
        latest_notional = float(latest.strongest_bid_wall_notional or 0.0)
        price_attr = "strongest_bid_wall"
        notional_attr = "strongest_bid_wall_notional"
    else:
        latest_price = float(latest.strongest_ask_wall or 0.0)
        latest_notional = float(latest.strongest_ask_wall_notional or 0.0)
        price_attr = "strongest_ask_wall"
        notional_attr = "strongest_ask_wall_notional"

    if latest_price <= 0 or latest_notional <= 0:
        return 0

    tolerance = max(abs(latest_price) * 0.0015, _round_step(abs(latest_price)) / 3.0, 1e-6)
    count = 0
    for sample in reversed(history):
        price = float(getattr(sample, price_attr, 0.0) or 0.0)
        notional = float(getattr(sample, notional_attr, 0.0) or 0.0)
        if price <= 0 or notional <= 0:
            break
        if abs(price - latest_price) > tolerance:
            break
        if notional < latest_notional * 0.35:
            break
        count += 1
    return count


def _intracycle_breakout_state(
    history: List[OrderBookFeedSnapshot],
    *,
    bullish_level: float,
    bearish_level: float,
    close_buffer: float,
    min_samples: int,
) -> Tuple[str, int]:
    if not history or min_samples <= 0:
        return "NONE", 0

    bullish_count = 0
    bearish_count = 0

    if bullish_level > 0:
        for sample in reversed(history):
            if float(sample.ref_price or 0.0) > bullish_level * (1 + close_buffer):
                bullish_count += 1
            else:
                break

    if bearish_level > 0:
        for sample in reversed(history):
            if float(sample.ref_price or 0.0) < bearish_level * (1 - close_buffer):
                bearish_count += 1
            else:
                break

    if bullish_count >= min_samples and bullish_count >= bearish_count:
        return "PERSISTENT_BULLISH_BREAKOUT", bullish_count
    if bearish_count >= min_samples and bearish_count > bullish_count:
        return "PERSISTENT_BEARISH_BREAKDOWN", bearish_count
    return "NONE", max(bullish_count, bearish_count)


def _build_orderbook_signal(
    coin: str,
    *,
    ref_price: float,
    price_decimals: int,
    book: Dict[str, Any],
    daily_lookback: int,
    guard_distance_pct: float,
    reaction_distance_pct: float,
    feed_history: Optional[List[OrderBookFeedSnapshot]] = None,
    feed_breakout_samples: int = 2,
) -> OrderBookLevelSignal:
    signal = OrderBookLevelSignal(coin=coin, valid=False)
    if ref_price <= 0:
        signal.description = "Orderbook returned no usable reference price"
        return signal

    daily_supports, daily_resistances, last_daily_close = _daily_levels(coin, ref_price, daily_lookback)
    (
        book_supports,
        book_resistances,
        bid_notional,
        ask_notional,
        best_bid,
        best_ask,
        spread_bps,
    ) = _orderbook_levels(book, ref_price, price_decimals)

    all_supports = daily_supports + book_supports
    all_resistances = daily_resistances + book_resistances

    support_price, support_strength, support_distance_pct = _choose_level(
        ref_price,
        [{"price": price, "strength": strength} for price, strength, _, _ in all_supports],
        "support",
    )
    resistance_price, resistance_strength, resistance_distance_pct = _choose_level(
        ref_price,
        [{"price": price, "strength": strength} for price, strength, _, _ in all_resistances],
        "resistance",
    )

    support_levels = _serialize_levels(all_supports, ref_price, "support")
    resistance_levels = _serialize_levels(all_resistances, ref_price, "resistance")

    daily_resistance_prices = [price for price, _, source, _ in daily_resistances if "daily" in source]
    daily_support_prices = [price for price, _, source, _ in daily_supports if "daily" in source]
    daily_resistance_above = [price for price in daily_resistance_prices if price > ref_price]
    daily_support_below = [price for price in daily_support_prices if price < ref_price]
    daily_broken_resistance = [price for price in daily_resistance_prices if price <= ref_price]
    daily_broken_support = [price for price in daily_support_prices if price >= ref_price]

    breakout_level = min(daily_resistance_above) if daily_resistance_above else 0.0
    breakdown_level = max(daily_support_below) if daily_support_below else 0.0
    broken_resistance = max(daily_broken_resistance) if daily_broken_resistance else 0.0
    broken_support = min(daily_broken_support) if daily_broken_support else 0.0

    close_buffer = 0.001
    persistent_breakout_state, persistent_breakout_samples = _intracycle_breakout_state(
        list(feed_history or []),
        bullish_level=broken_resistance,
        bearish_level=broken_support,
        close_buffer=close_buffer,
        min_samples=max(2, int(feed_breakout_samples or 2)),
    )

    breakout_state = "NONE"
    if broken_resistance > 0 and last_daily_close > broken_resistance * (1 + close_buffer):
        breakout_state = "CONFIRMED_BULLISH_BREAKOUT"
    elif broken_support > 0 and last_daily_close < broken_support * (1 - close_buffer):
        breakout_state = "CONFIRMED_BEARISH_BREAKDOWN"
    elif persistent_breakout_state != "NONE":
        breakout_state = persistent_breakout_state
    elif broken_resistance > 0 and ref_price > broken_resistance * (1 + close_buffer):
        breakout_state = "PROBING_BULLISH_BREAKOUT"
    elif broken_support > 0 and ref_price < broken_support * (1 - close_buffer):
        breakout_state = "PROBING_BEARISH_BREAKDOWN"

    total_flow = bid_notional + ask_notional
    imbalance_ratio = (bid_notional - ask_notional) / total_flow if total_flow > 0 else 0.0
    feed_history = list(feed_history or [])
    imbalance_values = [float(sample.imbalance_ratio or 0.0) for sample in feed_history]
    imbalance_mean = _mean(imbalance_values) if imbalance_values else imbalance_ratio
    imbalance_trend = (
        float(feed_history[-1].imbalance_ratio or 0.0) - float(feed_history[0].imbalance_ratio or 0.0)
        if len(feed_history) >= 2 else 0.0
    )
    imbalance_volatility = _stddev(imbalance_values)
    support_wall_persistence = _wall_persistence(feed_history, side="bid")
    resistance_wall_persistence = _wall_persistence(feed_history, side="ask")
    feed_age_seconds = max(0.0, time.time() - float(feed_history[-1].ts or 0.0)) if feed_history else 0.0

    support_reaction = (
        support_price > 0
        and support_distance_pct <= reaction_distance_pct
        and support_strength >= 0.45
    )
    resistance_reaction = (
        resistance_price > 0
        and resistance_distance_pct <= reaction_distance_pct
        and resistance_strength >= 0.45
    )
    range_compression = support_reaction and resistance_reaction

    score = 50.0 + imbalance_ratio * 14.0
    score += imbalance_mean * 8.0
    score += max(-4.5, min(4.5, imbalance_trend * 12.0))

    if support_price > 0 and support_distance_pct <= guard_distance_pct:
        support_effect = min(12.0, (1.0 - support_distance_pct / max(guard_distance_pct, 1e-9)) * 10.0 * support_strength)
        score += support_effect
    if resistance_price > 0 and resistance_distance_pct <= guard_distance_pct:
        resistance_effect = min(12.0, (1.0 - resistance_distance_pct / max(guard_distance_pct, 1e-9)) * 10.0 * resistance_strength)
        score -= resistance_effect

    if support_wall_persistence >= 2 and support_price > 0 and support_distance_pct <= guard_distance_pct:
        score += min(8.0, support_wall_persistence * 1.4)
    if resistance_wall_persistence >= 2 and resistance_price > 0 and resistance_distance_pct <= guard_distance_pct:
        score -= min(8.0, resistance_wall_persistence * 1.4)

    if breakout_state == "CONFIRMED_BULLISH_BREAKOUT":
        score += 12.0
    elif breakout_state == "PERSISTENT_BULLISH_BREAKOUT":
        score += 8.0
    elif breakout_state == "PROBING_BULLISH_BREAKOUT":
        score += 4.5
    elif breakout_state == "CONFIRMED_BEARISH_BREAKDOWN":
        score -= 12.0
    elif breakout_state == "PERSISTENT_BEARISH_BREAKDOWN":
        score -= 8.0
    elif breakout_state == "PROBING_BEARISH_BREAKDOWN":
        score -= 4.5

    if imbalance_volatility >= 0.18 and abs(imbalance_mean) <= 0.08:
        score = 50.0 + (score - 50.0) * 0.70

    level_interaction = "BETWEEN_LEVELS"
    if range_compression and breakout_state not in {
        "CONFIRMED_BULLISH_BREAKOUT",
        "CONFIRMED_BEARISH_BREAKDOWN",
        "PERSISTENT_BULLISH_BREAKOUT",
        "PERSISTENT_BEARISH_BREAKDOWN",
    }:
        level_interaction = "RANGE_COMPRESSION"
        score = 50.0 + (score - 50.0) * 0.35
    elif support_reaction:
        level_interaction = "AT_SUPPORT"
    elif resistance_reaction:
        level_interaction = "AT_RESISTANCE"
    elif support_price > 0 and support_distance_pct <= guard_distance_pct:
        level_interaction = "ABOVE_SUPPORT"
    elif resistance_price > 0 and resistance_distance_pct <= guard_distance_pct:
        level_interaction = "BELOW_RESISTANCE"
    elif breakout_state in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"}:
        level_interaction = "ABOVE_BREAKOUT"
    elif breakout_state in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"}:
        level_interaction = "BELOW_BREAKDOWN"

    bullish_breakouts = {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT", "PROBING_BULLISH_BREAKOUT"}
    bearish_breakdowns = {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN", "PROBING_BEARISH_BREAKDOWN"}

    block_longs = (
        resistance_price > 0
        and resistance_distance_pct <= guard_distance_pct
        and resistance_strength >= 0.55
        and breakout_state not in bullish_breakouts
    )
    block_shorts = (
        support_price > 0
        and support_distance_pct <= guard_distance_pct
        and support_strength >= 0.55
        and breakout_state not in bearish_breakdowns
    )

    if range_compression and breakout_state not in {
        "CONFIRMED_BULLISH_BREAKOUT",
        "CONFIRMED_BEARISH_BREAKDOWN",
        "PERSISTENT_BULLISH_BREAKOUT",
        "PERSISTENT_BEARISH_BREAKDOWN",
    }:
        block_longs = True
        block_shorts = True

    favor_longs = (
        breakout_state in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"}
        or (level_interaction in {"AT_SUPPORT", "ABOVE_SUPPORT"} and imbalance_mean >= -0.05)
        or (support_wall_persistence >= 2 and imbalance_mean >= 0.02)
    )
    favor_shorts = (
        breakout_state in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"}
        or (level_interaction in {"AT_RESISTANCE", "BELOW_RESISTANCE"} and imbalance_mean <= 0.05)
        or (resistance_wall_persistence >= 2 and imbalance_mean <= -0.02)
    )

    score = max(0.0, min(100.0, score))
    parts = []
    if support_price > 0:
        parts.append(f"support {support_price:,.2f} ({support_distance_pct:.2f}% away)")
    if resistance_price > 0:
        parts.append(f"resistance {resistance_price:,.2f} ({resistance_distance_pct:.2f}% away)")
    if breakout_state != "NONE":
        parts.append(breakout_state.replace("_", " ").title())
    parts.append(f"book imbalance {imbalance_ratio:+.2f}")
    if feed_history:
        parts.append(f"feed {len(feed_history)} snaps")
        parts.append(f"mean {imbalance_mean:+.2f}")
        if support_wall_persistence >= 2:
            parts.append(f"bid wall held {support_wall_persistence}x")
        if resistance_wall_persistence >= 2:
            parts.append(f"ask wall held {resistance_wall_persistence}x")
        if persistent_breakout_state != "NONE":
            parts.append(f"{persistent_breakout_state.replace('_', ' ').title()} ({persistent_breakout_samples}x)")

    return OrderBookLevelSignal(
        coin=coin,
        valid=True,
        score=round(score, 2),
        description=" | ".join(parts),
        best_bid=round(best_bid, 6),
        best_ask=round(best_ask, 6),
        mid_price=round(ref_price, 6),
        spread_bps=round(spread_bps, 3),
        bid_notional=round(bid_notional, 2),
        ask_notional=round(ask_notional, 2),
        imbalance_ratio=round(imbalance_ratio, 4),
        nearest_support=round(support_price, 6) if support_price else 0.0,
        nearest_support_strength=round(support_strength, 3),
        nearest_support_distance_pct=round(support_distance_pct, 3),
        nearest_resistance=round(resistance_price, 6) if resistance_price else 0.0,
        nearest_resistance_strength=round(resistance_strength, 3),
        nearest_resistance_distance_pct=round(resistance_distance_pct, 3),
        daily_breakout_level=round(breakout_level or broken_resistance, 6) if (breakout_level or broken_resistance) else 0.0,
        daily_breakdown_level=round(breakdown_level or broken_support, 6) if (breakdown_level or broken_support) else 0.0,
        last_daily_close=round(last_daily_close, 6) if last_daily_close else 0.0,
        breakout_state=breakout_state,
        level_interaction=level_interaction,
        favor_longs=favor_longs,
        favor_shorts=favor_shorts,
        block_longs=block_longs,
        block_shorts=block_shorts,
        feed_age_seconds=round(feed_age_seconds, 3),
        feed_snapshot_count=len(feed_history),
        imbalance_mean=round(imbalance_mean, 4),
        imbalance_trend=round(imbalance_trend, 4),
        imbalance_volatility=round(imbalance_volatility, 4),
        support_wall_persistence=int(support_wall_persistence),
        resistance_wall_persistence=int(resistance_wall_persistence),
        intracycle_breakout_state=persistent_breakout_state,
        support_levels=support_levels,
        resistance_levels=resistance_levels,
    )


def get_orderbook_levels(
    coin: str,
    *,
    current_price: float = 0.0,
    depth_limit: int = 120,
    daily_lookback: int = 120,
    cache_ttl_seconds: int = 25,
    guard_distance_pct: float = 1.25,
    reaction_distance_pct: float = 0.45,
    feed_max_age_seconds: float = 45.0,
    feed_breakout_samples: int = 2,
) -> OrderBookLevelSignal:
    coin = coin.upper()
    feed_history = _BACKGROUND_ORDERBOOK_FEED.get_recent_snapshots(
        coin,
        max_age_seconds=max(0.0, float(feed_max_age_seconds or 0.0)),
    )
    if feed_history:
        latest = feed_history[-1]
        ref_price = float(current_price or latest.ref_price or latest.last_trade_price or 0.0)
        signal = _build_orderbook_signal(
            coin,
            ref_price=ref_price,
            price_decimals=int(latest.price_decimals or 1),
            book=latest.book,
            daily_lookback=daily_lookback,
            guard_distance_pct=guard_distance_pct,
            reaction_distance_pct=reaction_distance_pct,
            feed_history=feed_history,
            feed_breakout_samples=feed_breakout_samples,
        )
        return signal

    cache_key = (coin, depth_limit, daily_lookback)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < cache_ttl_seconds:
        return cached[1]

    snapshot = _fetch_live_snapshot(coin, depth_limit=depth_limit)
    if not snapshot.valid:
        signal = OrderBookLevelSignal(coin=coin, valid=False)
        signal.description = snapshot.error or "Orderbook unavailable"
        return signal

    ref_price = float(current_price or snapshot.ref_price or snapshot.last_trade_price or 0.0)
    signal = _build_orderbook_signal(
        coin,
        ref_price=ref_price,
        price_decimals=int(snapshot.price_decimals or 1),
        book=snapshot.book,
        daily_lookback=daily_lookback,
        guard_distance_pct=guard_distance_pct,
        reaction_distance_pct=reaction_distance_pct,
        feed_history=[snapshot],
        feed_breakout_samples=feed_breakout_samples,
    )
    _CACHE[cache_key] = (now, signal)
    return signal
