"""
market_map.py — higher-timeframe market map workflow.

Manual operator levels remain first-class, but the agent should never be
"unmapped". When no manual map exists, or when a manual map is partial, we
auto-synthesize a daily map from completed 1D structure so every tracked asset
has a thesis anchor.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from data.market_data import completed_candle_frame, fetch_candles
from logger import get_logger
from paths import DAILY_MARKET_MAP_JSON

log = get_logger("market_map")

_DAILY_CLOSE_CACHE: Dict[str, tuple[float, float]] = {}
_AUTO_ENTRY_CACHE: Dict[str, tuple[float, dict]] = {}


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


def _normalize_entry(raw_entry: dict, updated_at: str) -> dict:
    bias = str(raw_entry.get("bias") or "NEUTRAL").upper()
    if bias not in {"BULLISH", "BEARISH", "NEUTRAL"}:
        bias = "NEUTRAL"
    confidence = str(raw_entry.get("confidence") or "MEDIUM").upper()
    if confidence not in {"LOW", "MEDIUM", "HIGH"}:
        confidence = "MEDIUM"
    source = str(raw_entry.get("source") or "MANUAL").upper()
    if source not in {"MANUAL", "AUTO"}:
        source = "MANUAL"
    return {
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
        "summary": str(raw_entry.get("summary") or ""),
        "source": source,
        "auto_generated": bool(raw_entry.get("auto_generated", source == "AUTO")),
        "updated_at": str(raw_entry.get("updated_at") or updated_at),
    }


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
        coins[coin_key] = _normalize_entry(raw_entry, base["updated_at"])
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
        "summary": str(update.get("summary", current.get("summary", "")) or ""),
        "source": "MANUAL",
        "auto_generated": False,
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
    if price >= 1:
        return 0.10
    return 0.01


def _dedupe_levels(levels: List[float], tolerance_pct: float = 0.35) -> List[float]:
    out: List[float] = []
    for level in sorted(value for value in levels if value > 0):
        if not out:
            out.append(level)
            continue
        tolerance = max(level, out[-1]) * tolerance_pct / 100.0
        if abs(level - out[-1]) <= tolerance:
            out[-1] = round((out[-1] + level) / 2.0, 6)
        else:
            out.append(level)
    return [round(value, 6) for value in out]


def _atr_from_frame(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or df.empty:
        return 0.0
    try:
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        closes = df["close"].astype(float)
        prev_close = closes.shift(1).fillna(closes.iloc[0])
        tr = pd.concat(
            [
                highs - lows,
                (highs - prev_close).abs(),
                (lows - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = float(tr.rolling(period, min_periods=max(3, period // 2)).mean().iloc[-1])
        return max(atr, 0.0)
    except Exception:
        return 0.0


def _swing_levels(df: pd.DataFrame) -> tuple[List[float], List[float]]:
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    swing_highs: List[float] = []
    swing_lows: List[float] = []
    if len(df) < 5:
        return swing_lows, swing_highs
    for idx in range(2, len(df) - 2):
        window_highs = highs[idx - 2:idx + 3]
        window_lows = lows[idx - 2:idx + 3]
        if highs[idx] >= max(window_highs):
            swing_highs.append(highs[idx])
        if lows[idx] <= min(window_lows):
            swing_lows.append(lows[idx])
    return swing_lows[-8:], swing_highs[-8:]


def _auto_trade_mode(bias: str) -> str:
    if bias == "BULLISH":
        return "Buy pullbacks into support/demand; press harder after reclaim closes confirm."
    if bias == "BEARISH":
        return "Sell rallies into supply/resistance; press harder after breakdown closes confirm."
    return "Range the edges only; wait for reclaim/breakdown confirmation before pressing."


def _auto_summary(
    coin: str,
    bias: str,
    confidence: str,
    supports: List[float],
    resistances: List[float],
) -> str:
    parts = [f"auto {bias.lower()} map ({confidence.lower()} confidence)"]
    if supports:
        parts.append(f"support around {supports[-1]:,.2f}")
    if resistances:
        parts.append(f"resistance around {resistances[0]:,.2f}")
    return "; ".join(parts[:3])


def _build_auto_entry(coin: str, current_price: float = 0.0, closed_price: float = 0.0) -> dict:
    coin = coin.upper()
    now = time.time()
    cached = _AUTO_ENTRY_CACHE.get(coin)
    if cached and (now - cached[0]) < 300:
        entry = dict(cached[1])
        if current_price > 0:
            entry["updated_at"] = _now_str()
        return entry

    df = completed_candle_frame(fetch_candles(coin, interval="1d", lookback=140), min_rows=30)
    entry_price = float(current_price or closed_price or 0.0)
    if df is None or df.empty:
        step = _round_step(max(entry_price, 1.0))
        supports = _as_float_list([entry_price - step, entry_price - 2 * step]) if entry_price > 0 else []
        resistances = _as_float_list([entry_price + step, entry_price + 2 * step]) if entry_price > 0 else []
        entry = {
            "bias": "NEUTRAL",
            "confidence": "LOW",
            "supports": supports,
            "resistances": resistances,
            "daily_close_long_above": resistances[:2],
            "daily_close_short_below": supports[-2:],
            "demand_zone": {"low": supports[-1] if supports else 0.0, "high": supports[-1] if supports else 0.0},
            "supply_zone": {"low": resistances[0] if resistances else 0.0, "high": resistances[0] if resistances else 0.0},
            "notes": "Auto map fallback: limited history available, using neutral guardrails only.",
            "trade_mode": _auto_trade_mode("NEUTRAL"),
            "summary": "auto neutral fallback map",
            "source": "AUTO",
            "auto_generated": True,
            "updated_at": _now_str(),
        }
        _AUTO_ENTRY_CACHE[coin] = (now, dict(entry))
        return entry

    closes = df["close"].astype(float)
    highs = df["high"].astype(float)
    lows = df["low"].astype(float)
    last_close = float(closes.iloc[-1])
    ref_price = float(entry_price or last_close)
    ema20 = float(closes.ewm(span=min(20, len(closes)), adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=min(50, len(closes)), adjust=False).mean().iloc[-1])
    atr = _atr_from_frame(df)

    supports: List[float] = []
    resistances: List[float] = []
    swing_lows, swing_highs = _swing_levels(df)
    supports.extend(swing_lows)
    resistances.extend(swing_highs)

    for window in (5, 10, 20, 40):
        if len(df) > window:
            supports.append(float(lows.tail(window).min()))
            resistances.append(float(highs.tail(window).max()))

    supports.extend([ema20, ema50])
    resistances.extend([ema20, ema50])

    step = _round_step(ref_price)
    if ref_price > 0:
        anchor = round(ref_price / step) * step
        for mult in range(1, 4):
            supports.append(anchor - step * mult)
            resistances.append(anchor + step * mult)

    deduped_supports = [level for level in _dedupe_levels(supports) if 0 < level < ref_price]
    deduped_resistances = [level for level in _dedupe_levels(resistances) if level > ref_price]

    final_supports = deduped_supports[-3:]
    final_resistances = deduped_resistances[:3]

    if not final_supports and ref_price > 0:
        final_supports = _as_float_list([ref_price - step, ref_price - 2 * step])[-2:]
    if not final_resistances and ref_price > 0:
        final_resistances = _as_float_list([ref_price + step, ref_price + 2 * step])[:2]

    long_above = final_resistances[:2]
    short_below = final_supports[-2:]

    score = 0
    if ref_price >= ema20:
        score += 1
    else:
        score -= 1
    if ema20 >= ema50:
        score += 1
    else:
        score -= 1
    if len(closes) > 5:
        ref_compare = float(closes.iloc[-5])
        if ref_price > ref_compare:
            score += 1
        elif ref_price < ref_compare:
            score -= 1
    if len(df) > 20:
        prev_high = float(highs.iloc[-21:-1].max())
        prev_low = float(lows.iloc[-21:-1].min())
        if ref_price > prev_high:
            score += 2
        elif ref_price < prev_low:
            score -= 2

    if score >= 2:
        bias = "BULLISH"
    elif score <= -2:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    distance_to_trend = abs(ref_price - ema20) / max(atr, 1e-9) if atr > 0 else 0.0
    if abs(score) >= 4 or distance_to_trend >= 1.35:
        confidence = "HIGH"
    elif abs(score) >= 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    nearest_support = final_supports[-1] if final_supports else 0.0
    nearest_resistance = final_resistances[0] if final_resistances else 0.0
    zone_half_width = atr * 0.35 if atr > 0 else step * 0.75

    notes = (
        "Auto map from completed 1D trend, swing structure, round levels, and ATR zones. "
        "Manual overrides always take precedence."
    )
    summary = _auto_summary(coin, bias, confidence, final_supports, final_resistances)
    entry = {
        "bias": bias,
        "confidence": confidence,
        "supports": _as_float_list(final_supports),
        "resistances": _as_float_list(final_resistances),
        "daily_close_long_above": _as_float_list(long_above),
        "daily_close_short_below": _as_float_list(short_below),
        "demand_zone": {
            "low": round(max(0.0, nearest_support - zone_half_width), 6) if nearest_support else 0.0,
            "high": round(nearest_support + zone_half_width, 6) if nearest_support else 0.0,
        },
        "supply_zone": {
            "low": round(max(0.0, nearest_resistance - zone_half_width), 6) if nearest_resistance else 0.0,
            "high": round(nearest_resistance + zone_half_width, 6) if nearest_resistance else 0.0,
        },
        "notes": notes,
        "trade_mode": _auto_trade_mode(bias),
        "summary": summary,
        "source": "AUTO",
        "auto_generated": True,
        "updated_at": _now_str(),
    }
    _AUTO_ENTRY_CACHE[coin] = (now, dict(entry))
    return entry


def tracked_coins_from_state(state: Any) -> List[str]:
    coins: List[str] = []
    seen = set()
    safe_state = state if isinstance(state, dict) else {}
    config = safe_state.get("config") if isinstance(safe_state.get("config"), dict) else {}
    groups = [
        config.get("coins") or [],
        config.get("analysis_coins") or [],
        list((safe_state.get("signals") or {}).keys()),
        [pos.get("coin") for pos in (safe_state.get("positions") or []) if isinstance(pos, dict)],
    ]
    for group in groups:
        for value in group or []:
            coin = str(value or "").upper().strip()
            if coin and coin not in seen:
                seen.add(coin)
                coins.append(coin)
    return coins


def build_effective_market_map(
    tracked_coins: List[str] | None = None,
    *,
    current_prices: Dict[str, float] | None = None,
    closed_prices: Dict[str, float] | None = None,
    base_map: dict | None = None,
) -> dict:
    manual_map = normalize_market_map(base_map if isinstance(base_map, dict) else load_market_map())
    current_prices = {str(k).upper(): float(v or 0.0) for k, v in (current_prices or {}).items()}
    closed_prices = {str(k).upper(): float(v or 0.0) for k, v in (closed_prices or {}).items()}

    coins = list(dict.fromkeys([
        *(tracked_coins or []),
        *((manual_map.get("coins") or {}).keys()),
    ]))
    coins = [str(coin or "").upper().strip() for coin in coins if str(coin or "").strip()]

    merged = dict(manual_map)
    merged_coins: Dict[str, dict] = {}
    for coin in coins:
        manual_entry = dict((manual_map.get("coins") or {}).get(coin) or {})
        auto_entry = _build_auto_entry(
            coin,
            current_price=current_prices.get(coin, 0.0),
            closed_price=closed_prices.get(coin, 0.0),
        )
        if manual_entry:
            filled = dict(auto_entry)
            filled.update({
                key: value for key, value in manual_entry.items()
                if value not in (None, "", [], {}, 0.0)
            })
            filled["source"] = "MANUAL"
            filled["auto_generated"] = False
            filled["updated_at"] = manual_entry.get("updated_at", auto_entry.get("updated_at", _now_str()))
            merged_coins[coin] = _normalize_entry(filled, merged.get("updated_at") or _now_str())
        else:
            merged_coins[coin] = _normalize_entry(auto_entry, merged.get("updated_at") or _now_str())

    merged["coins"] = merged_coins
    merged["updated_at"] = _now_str()
    return merged


def review_summary(market_map: dict) -> dict:
    coins = dict((market_map or {}).get("coins") or {})
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    auto_count = 0
    manual_count = 0
    for entry in coins.values():
        bias = str((entry or {}).get("bias") or "NEUTRAL").lower()
        if bias not in counts:
            bias = "neutral"
        counts[bias] += 1
        if bool((entry or {}).get("auto_generated")) or str((entry or {}).get("source") or "").upper() == "AUTO":
            auto_count += 1
        else:
            manual_count += 1
    total = len(coins)
    return {
        "count": total,
        "bullish": counts["bullish"],
        "bearish": counts["bearish"],
        "neutral": counts["neutral"],
        "manual_count": manual_count,
        "auto_count": auto_count,
        "updated_at": (market_map or {}).get("updated_at"),
    }


@dataclass
class MarketMapSignal:
    coin: str
    available: bool = False
    bias: str = "NEUTRAL"
    confidence: str = "MEDIUM"
    valid: bool = False
    source: str = "AUTO"
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
    reclaim_levels: List[float] = field(default_factory=list)
    breakdown_levels: List[float] = field(default_factory=list)
    above_reclaim_levels: List[float] = field(default_factory=list)
    below_breakdown_levels: List[float] = field(default_factory=list)
    live_above_reclaim_levels: List[float] = field(default_factory=list)
    live_below_breakdown_levels: List[float] = field(default_factory=list)
    probing_above_reclaim_levels: List[float] = field(default_factory=list)
    probing_below_breakdown_levels: List[float] = field(default_factory=list)
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
    effective_map = build_effective_market_map(
        [coin.upper()],
        current_prices={coin.upper(): current_price},
        closed_prices={coin.upper(): closed_price},
    )
    entry = dict((effective_map.get("coins") or {}).get(coin.upper()) or {})
    signal = MarketMapSignal(
        coin=coin.upper(),
        available=bool(entry),
        valid=bool(entry),
        source=str(entry.get("source") or "AUTO").upper(),
        current_price=float(current_price or 0.0),
        closed_price=float(closed_price or current_price or 0.0),
    )
    if not entry:
        signal.summary = "No market map available"
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
    signal.reclaim_levels = daily_reclaim
    signal.breakdown_levels = daily_breakdown
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
    signal.live_above_reclaim_levels = [level for level in daily_reclaim if signal.current_price >= level]
    signal.live_below_breakdown_levels = [level for level in daily_breakdown if signal.current_price <= level]
    signal.probing_above_reclaim_levels = [level for level in daily_reclaim if signal.closed_price >= level]
    signal.probing_below_breakdown_levels = [level for level in daily_breakdown if signal.closed_price <= level]

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
    elif signal.probing_above_reclaim_levels:
        bias_score += 1.5
        signal.favor_longs = True

    if signal.below_breakdown_levels:
        bias_score -= 3.0
        signal.favor_shorts = True
        signal.block_longs = True
    elif signal.probing_below_breakdown_levels:
        bias_score -= 1.5
        signal.favor_shorts = True

    signal.score_adjustment = round(max(-12.0, min(12.0, bias_score)), 2)
    parts: List[str] = [f"{signal.source.lower()} {signal.bias.lower()} map"]
    if signal.above_reclaim_levels:
        if signal.live_above_reclaim_levels:
            parts.append("daily reclaim is holding")
        else:
            parts.append("daily reclaim was confirmed, but live price slipped back below reclaim")
    elif signal.probing_above_reclaim_levels:
        if signal.live_above_reclaim_levels:
            parts.append("intraday reclaim is live")
        else:
            parts.append("intraday reclaim in play")
    if signal.below_breakdown_levels:
        if signal.live_below_breakdown_levels:
            parts.append("daily breakdown is holding")
        else:
            parts.append("daily breakdown was confirmed, but live price bounced back above breakdown")
    elif signal.probing_below_breakdown_levels:
        if signal.live_below_breakdown_levels:
            parts.append("intraday breakdown is live")
        else:
            parts.append("intraday breakdown in play")
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
