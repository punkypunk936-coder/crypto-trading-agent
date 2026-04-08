"""
market_map.py — operator-owned daily key-level / market map workflow.

The goal is to let the human market operator define the higher-timeframe map
once per day, then let the agent respect that map throughout the session:
  - directional bias
  - supports / resistances
  - demand / supply zones
  - reclaim / breakdown levels that matter on a daily close
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from data.market_data import completed_candle_frame, fetch_candles
from logger import get_logger
from paths import DAILY_MARKET_MAP_JSON

log = get_logger("market_map")

_DAILY_CLOSE_CACHE: Dict[str, tuple[float, float]] = {}


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def default_market_map() -> dict:
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "updated_at": _now_str(),
        "global_notes": "",
        "coins": {},
    }


def _as_float_list(values: Any) -> List[float]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = [part.strip() for part in values.split(",")]
        values = [part for part in parts if part]
    out: List[float] = []
    for value in values or []:
        try:
            number = float(value)
        except Exception:
            continue
        if number > 0:
            out.append(round(number, 6))
    return sorted(dict.fromkeys(out))


def _normalize_zone(value: Any) -> dict:
    if not isinstance(value, dict):
        return {"low": 0.0, "high": 0.0}
    try:
        low = float(value.get("low") or 0.0)
    except Exception:
        low = 0.0
    try:
        high = float(value.get("high") or 0.0)
    except Exception:
        high = 0.0
    if low > high > 0:
        low, high = high, low
    return {"low": round(low, 6), "high": round(high, 6)}


def normalize_market_map(payload: Any) -> dict:
    base = default_market_map()
    if not isinstance(payload, dict):
        return base
    base["date"] = str(payload.get("date") or base["date"])
    base["updated_at"] = str(payload.get("updated_at") or base["updated_at"])
    base["global_notes"] = str(payload.get("global_notes") or "")

    coins = {}
    raw_coins = payload.get("coins") if isinstance(payload.get("coins"), dict) else {}
    for coin, raw_entry in raw_coins.items():
        if not isinstance(raw_entry, dict):
            continue
        coin_key = str(coin).upper().strip()
        if not coin_key:
            continue
        bias = str(raw_entry.get("bias") or "NEUTRAL").upper()
        if bias not in {"BULLISH", "BEARISH", "NEUTRAL"}:
            bias = "NEUTRAL"
        confidence = str(raw_entry.get("confidence") or "MEDIUM").upper()
        if confidence not in {"LOW", "MEDIUM", "HIGH"}:
            confidence = "MEDIUM"
        coins[coin_key] = {
            "bias": bias,
            "confidence": confidence,
            "supports": _as_float_list(raw_entry.get("supports")),
            "resistances": _as_float_list(raw_entry.get("resistances")),
            "daily_close_long_above": _as_float_list(raw_entry.get("daily_close_long_above")),
            "daily_close_short_below": _as_float_list(raw_entry.get("daily_close_short_below")),
            "demand_zone": _normalize_zone(raw_entry.get("demand_zone")),
            "supply_zone": _normalize_zone(raw_entry.get("supply_zone")),
            "notes": str(raw_entry.get("notes") or ""),
            "trade_mode": str(raw_entry.get("trade_mode") or ""),
            "updated_at": str(raw_entry.get("updated_at") or base["updated_at"]),
        }
    base["coins"] = coins
    return base


def load_market_map() -> dict:
    if not DAILY_MARKET_MAP_JSON.exists():
        return default_market_map()
    try:
        return normalize_market_map(json.loads(DAILY_MARKET_MAP_JSON.read_text()))
    except Exception as exc:
        log.warning(f"Failed to load daily market map: {exc}")
        return default_market_map()


def save_market_map(payload: dict) -> dict:
    normalized = normalize_market_map(payload)
    normalized["updated_at"] = _now_str()
    DAILY_MARKET_MAP_JSON.write_text(json.dumps(normalized, indent=2))
    return normalized


def upsert_market_map_entry(coin: str, update: dict) -> dict:
    market_map = load_market_map()
    coin_key = str(coin or "").upper().strip()
    if not coin_key:
        return market_map
    coins = dict(market_map.get("coins") or {})
    current = dict(coins.get(coin_key) or {})
    current.update({
        "bias": str(update.get("bias", current.get("bias", "NEUTRAL")) or "NEUTRAL").upper(),
        "confidence": str(update.get("confidence", current.get("confidence", "MEDIUM")) or "MEDIUM").upper(),
        "supports": _as_float_list(update.get("supports", current.get("supports", []))),
        "resistances": _as_float_list(update.get("resistances", current.get("resistances", []))),
        "daily_close_long_above": _as_float_list(update.get("daily_close_long_above", current.get("daily_close_long_above", []))),
        "daily_close_short_below": _as_float_list(update.get("daily_close_short_below", current.get("daily_close_short_below", []))),
        "demand_zone": _normalize_zone(update.get("demand_zone", current.get("demand_zone", {}))),
        "supply_zone": _normalize_zone(update.get("supply_zone", current.get("supply_zone", {}))),
        "notes": str(update.get("notes", current.get("notes", "")) or ""),
        "trade_mode": str(update.get("trade_mode", current.get("trade_mode", "")) or ""),
        "updated_at": _now_str(),
    })
    coins[coin_key] = current
    market_map["coins"] = coins
    if "global_notes" in update:
        market_map["global_notes"] = str(update.get("global_notes") or "")
    return save_market_map(market_map)


def delete_market_map_entry(coin: str) -> dict:
    market_map = load_market_map()
    coin_key = str(coin or "").upper().strip()
    if coin_key and coin_key in (market_map.get("coins") or {}):
        del market_map["coins"][coin_key]
    return save_market_map(market_map)


def _get_daily_close(coin: str, ttl_seconds: int = 300) -> float:
    coin = coin.upper()
    cached = _DAILY_CLOSE_CACHE.get(coin)
    now = time.time()
    if cached and (now - cached[0]) < ttl_seconds:
        return cached[1]
    df = fetch_candles(coin, interval="1d", lookback=5)
    completed = completed_candle_frame(df)
    daily_close = 0.0
    if completed is not None and not completed.empty:
        try:
            daily_close = float(completed["close"].iloc[-1])
        except Exception:
            daily_close = 0.0
    _DAILY_CLOSE_CACHE[coin] = (now, daily_close)
    return daily_close


def review_summary(market_map: dict) -> dict:
    coins = dict((market_map or {}).get("coins") or {})
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    for entry in coins.values():
        bias = str((entry or {}).get("bias") or "NEUTRAL").lower()
        if bias not in counts:
            bias = "neutral"
        counts[bias] += 1
    return {
        "count": len(coins),
        "bullish": counts["bullish"],
        "bearish": counts["bearish"],
        "neutral": counts["neutral"],
        "updated_at": (market_map or {}).get("updated_at"),
    }


@dataclass
class MarketMapSignal:
    coin: str
    available: bool = False
    bias: str = "NEUTRAL"
    confidence: str = "MEDIUM"
    valid: bool = False
    current_price: float = 0.0
    closed_price: float = 0.0
    daily_close: float = 0.0
    score_adjustment: float = 0.0
    favor_longs: bool = False
    favor_shorts: bool = False
    block_longs: bool = False
    block_shorts: bool = False
    nearest_support: float = 0.0
    nearest_support_distance_pct: float = 0.0
    nearest_resistance: float = 0.0
    nearest_resistance_distance_pct: float = 0.0
    in_demand_zone: bool = False
    in_supply_zone: bool = False
    above_reclaim_levels: List[float] = field(default_factory=list)
    below_breakdown_levels: List[float] = field(default_factory=list)
    summary: str = ""
    notes: str = ""


def _nearest_level(price: float, levels: List[float], *, side: str) -> tuple[float, float]:
    if price <= 0:
        return 0.0, 0.0
    candidates = []
    for level in levels:
        if side == "support" and level < price:
            candidates.append(level)
        elif side == "resistance" and level > price:
            candidates.append(level)
    if not candidates:
        return 0.0, 0.0
    level = max(candidates) if side == "support" else min(candidates)
    distance_pct = abs(price - level) / max(price, 1e-9) * 100.0
    return level, distance_pct


def get_market_map_signal(coin: str, current_price: float, closed_price: float = 0.0) -> MarketMapSignal:
    market_map = load_market_map()
    entry = dict((market_map.get("coins") or {}).get(coin.upper()) or {})
    signal = MarketMapSignal(
        coin=coin.upper(),
        available=bool(entry),
        valid=bool(entry),
        current_price=float(current_price or 0.0),
        closed_price=float(closed_price or current_price or 0.0),
    )
    if not entry:
        signal.summary = "No daily market map entry"
        return signal

    signal.bias = str(entry.get("bias") or "NEUTRAL").upper()
    signal.confidence = str(entry.get("confidence") or "MEDIUM").upper()
    signal.notes = str(entry.get("notes") or "")
    supports = _as_float_list(entry.get("supports"))
    resistances = _as_float_list(entry.get("resistances"))
    demand_zone = _normalize_zone(entry.get("demand_zone"))
    supply_zone = _normalize_zone(entry.get("supply_zone"))
    daily_reclaim = _as_float_list(entry.get("daily_close_long_above"))
    daily_breakdown = _as_float_list(entry.get("daily_close_short_below"))
    signal.daily_close = _get_daily_close(coin.upper())

    support, support_dist = _nearest_level(signal.current_price, supports, side="support")
    resistance, resistance_dist = _nearest_level(signal.current_price, resistances, side="resistance")
    signal.nearest_support = support
    signal.nearest_support_distance_pct = round(support_dist, 3)
    signal.nearest_resistance = resistance
    signal.nearest_resistance_distance_pct = round(resistance_dist, 3)
    signal.in_demand_zone = demand_zone["low"] > 0 and demand_zone["low"] <= signal.current_price <= max(demand_zone["high"], demand_zone["low"])
    signal.in_supply_zone = supply_zone["low"] > 0 and supply_zone["low"] <= signal.current_price <= max(supply_zone["high"], supply_zone["low"])
    signal.above_reclaim_levels = [level for level in daily_reclaim if signal.daily_close >= level]
    signal.below_breakdown_levels = [level for level in daily_breakdown if signal.daily_close <= level]

    bias_score = 0.0
    if signal.bias == "BULLISH":
        bias_score += 4.0
        signal.favor_longs = True
        signal.block_shorts = True
    elif signal.bias == "BEARISH":
        bias_score -= 4.0
        signal.favor_shorts = True
        signal.block_longs = True

    if signal.in_demand_zone:
        bias_score += 3.0
        signal.favor_longs = True
        signal.block_shorts = True
    if signal.in_supply_zone:
        bias_score -= 3.0
        signal.favor_shorts = True
        signal.block_longs = True

    guard_distance = 1.10
    if support > 0 and support_dist <= guard_distance:
        bias_score += 2.0
        signal.favor_longs = True
    if resistance > 0 and resistance_dist <= guard_distance:
        bias_score -= 2.0
        signal.favor_shorts = True

    if signal.above_reclaim_levels:
        bias_score += 3.0
        signal.favor_longs = True
        signal.block_shorts = True
    if signal.below_breakdown_levels:
        bias_score -= 3.0
        signal.favor_shorts = True
        signal.block_longs = True

    signal.score_adjustment = round(max(-12.0, min(12.0, bias_score)), 2)
    parts: List[str] = [f"{signal.bias.lower()} daily bias"]
    if signal.above_reclaim_levels:
        parts.append("daily reclaim confirmed")
    if signal.below_breakdown_levels:
        parts.append("daily breakdown confirmed")
    if signal.in_demand_zone:
        parts.append("price is sitting in mapped demand")
    if signal.in_supply_zone:
        parts.append("price is sitting in mapped supply")
    if resistance > 0 and resistance_dist <= guard_distance:
        parts.append("price is pressing mapped resistance")
    if support > 0 and support_dist <= guard_distance:
        parts.append("price is testing mapped support")
    signal.summary = "; ".join(parts[:4]) if parts else "Mapped levels loaded"
    return signal
