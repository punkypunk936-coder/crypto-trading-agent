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
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from data.market_data import fetch_candles
from exchanges.lighter_client import COIN_TO_MARKET_ID, DEFAULT_LIGHTER_API_BASE_URL
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
    support_levels: List[Dict[str, float | str]] = field(default_factory=list)
    resistance_levels: List[Dict[str, float | str]] = field(default_factory=list)


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
        payload = await api.order_book_orders(market_id=market_id, limit=limit)
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
        payload = await api.order_book_details(market_id=market_id)
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
    bids = list(book.get("bids", []) or [])
    asks = list(book.get("asks", []) or [])
    best_bid = max((_float(item.get("price")) for item in bids), default=0.0)
    best_ask = min((_float(item.get("price")) for item in asks if _float(item.get("price")) > 0), default=0.0)

    step = _round_step(current_price or best_bid or best_ask or 1.0)
    bucket_size = max(step / 5.0, 10 ** (-price_decimals) * 10)

    bid_buckets: Dict[float, float] = {}
    ask_buckets: Dict[float, float] = {}
    bid_total = 0.0
    ask_total = 0.0

    for item in bids:
        price = _float(item.get("price"))
        amount = _float(item.get("remaining_base_amount") or item.get("initial_base_amount"))
        if price <= 0 or amount <= 0 or price >= current_price:
            continue
        notional = price * amount
        bid_total += notional
        bucket = round(price / bucket_size) * bucket_size
        bid_buckets[bucket] = bid_buckets.get(bucket, 0.0) + notional

    for item in asks:
        price = _float(item.get("price"))
        amount = _float(item.get("remaining_base_amount") or item.get("initial_base_amount"))
        if price <= 0 or amount <= 0 or price <= current_price:
            continue
        notional = price * amount
        ask_total += notional
        bucket = round(price / bucket_size) * bucket_size
        ask_buckets[bucket] = ask_buckets.get(bucket, 0.0) + notional

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

    spread_bps = 0.0
    if best_bid > 0 and best_ask > 0:
        mid = (best_bid + best_ask) / 2.0
        spread_bps = abs(best_ask - best_bid) / max(mid, 1e-9) * 10_000

    return supports, resistances, bid_total, ask_total, best_bid, best_ask if best_ask > 0 else 0.0, spread_bps


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


def get_orderbook_levels(
    coin: str,
    *,
    current_price: float = 0.0,
    depth_limit: int = 120,
    daily_lookback: int = 120,
    cache_ttl_seconds: int = 25,
    guard_distance_pct: float = 1.25,
    reaction_distance_pct: float = 0.45,
) -> OrderBookLevelSignal:
    coin = coin.upper()
    cache_key = (coin, depth_limit, daily_lookback)
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and (now - cached[0]) < cache_ttl_seconds:
        return cached[1]

    signal = OrderBookLevelSignal(coin=coin, valid=False)
    market_id = COIN_TO_MARKET_ID.get(coin)
    if market_id is None:
        signal.description = "No public orderbook venue configured for this symbol"
        return signal

    try:
        details = _run_async(_lighter_orderbook_details(market_id))
        book = _run_async(_lighter_orderbook_orders(market_id, max(20, min(depth_limit, 250))))
    except Exception as exc:
        signal.description = f"Orderbook unavailable: {exc}"
        log.debug("[%s] Orderbook fetch failed: %s", coin, exc)
        return signal

    detail_rows = list(details.get("order_book_details", []) or [])
    detail = detail_rows[0] if detail_rows else {}
    last_trade_price = _float(detail.get("last_trade_price"))
    price_decimals = int(detail.get("price_decimals", 1) or 1)

    ref_price = current_price or last_trade_price
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
        [
            {
                "price": price,
                "strength": strength,
            }
            for price, strength, _, _ in all_supports
        ],
        "support",
    )
    resistance_price, resistance_strength, resistance_distance_pct = _choose_level(
        ref_price,
        [
            {
                "price": price,
                "strength": strength,
            }
            for price, strength, _, _ in all_resistances
        ],
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

    breakout_state = "NONE"
    close_buffer = 0.001  # 0.10%
    if broken_resistance > 0 and last_daily_close > broken_resistance * (1 + close_buffer):
        breakout_state = "CONFIRMED_BULLISH_BREAKOUT"
    elif broken_resistance > 0 and ref_price > broken_resistance * (1 + close_buffer):
        breakout_state = "PROBING_BULLISH_BREAKOUT"
    elif broken_support > 0 and last_daily_close < broken_support * (1 - close_buffer):
        breakout_state = "CONFIRMED_BEARISH_BREAKDOWN"
    elif broken_support > 0 and ref_price < broken_support * (1 - close_buffer):
        breakout_state = "PROBING_BEARISH_BREAKDOWN"

    total_flow = bid_notional + ask_notional
    imbalance_ratio = 0.0
    if total_flow > 0:
        imbalance_ratio = (bid_notional - ask_notional) / total_flow

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

    if support_price > 0 and support_distance_pct <= guard_distance_pct:
        support_effect = min(12.0, (1.0 - support_distance_pct / max(guard_distance_pct, 1e-9)) * 10.0 * support_strength)
        score += support_effect
    if resistance_price > 0 and resistance_distance_pct <= guard_distance_pct:
        resistance_effect = min(12.0, (1.0 - resistance_distance_pct / max(guard_distance_pct, 1e-9)) * 10.0 * resistance_strength)
        score -= resistance_effect

    if breakout_state == "CONFIRMED_BULLISH_BREAKOUT":
        score += 12.0
    elif breakout_state == "PROBING_BULLISH_BREAKOUT":
        score += 4.5
    elif breakout_state == "CONFIRMED_BEARISH_BREAKDOWN":
        score -= 12.0
    elif breakout_state == "PROBING_BEARISH_BREAKDOWN":
        score -= 4.5

    level_interaction = "BETWEEN_LEVELS"
    if range_compression and breakout_state not in {
        "CONFIRMED_BULLISH_BREAKOUT",
        "CONFIRMED_BEARISH_BREAKDOWN",
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
    elif breakout_state == "CONFIRMED_BULLISH_BREAKOUT":
        level_interaction = "ABOVE_BREAKOUT"
    elif breakout_state == "CONFIRMED_BEARISH_BREAKDOWN":
        level_interaction = "BELOW_BREAKDOWN"

    block_longs = (
        resistance_price > 0
        and resistance_distance_pct <= guard_distance_pct
        and resistance_strength >= 0.55
        and breakout_state not in {"CONFIRMED_BULLISH_BREAKOUT", "PROBING_BULLISH_BREAKOUT"}
    )
    block_shorts = (
        support_price > 0
        and support_distance_pct <= guard_distance_pct
        and support_strength >= 0.55
        and breakout_state not in {"CONFIRMED_BEARISH_BREAKDOWN", "PROBING_BEARISH_BREAKDOWN"}
    )

    if range_compression and breakout_state not in {
        "CONFIRMED_BULLISH_BREAKOUT",
        "CONFIRMED_BEARISH_BREAKDOWN",
    }:
        block_longs = True
        block_shorts = True

    favor_longs = (
        breakout_state == "CONFIRMED_BULLISH_BREAKOUT"
        or (level_interaction in {"AT_SUPPORT", "ABOVE_SUPPORT"} and imbalance_ratio >= -0.08)
    )
    favor_shorts = (
        breakout_state == "CONFIRMED_BEARISH_BREAKDOWN"
        or (level_interaction in {"AT_RESISTANCE", "BELOW_RESISTANCE"} and imbalance_ratio <= 0.08)
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

    signal = OrderBookLevelSignal(
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
        support_levels=support_levels,
        resistance_levels=resistance_levels,
    )

    _CACHE[cache_key] = (now, signal)
    return signal
