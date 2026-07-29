"""US index regime, catalyst, and execution context for the trading agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


BENCHMARKS = (
    ("SP500", "S&P 500"),
    ("NDX", "Nasdaq"),
    ("VIXINDEX", "VIX"),
)
BENCHMARK_SYMBOLS = tuple(symbol for symbol, _label in BENCHMARKS)

_SUPPORTIVE_WORDS = {
    "beat", "cooling", "cut", "dovish", "growth", "reclaim", "relief",
    "strong demand", "upgrade", "upside",
}
_BEARISH_WORDS = {
    "decline", "downgrade", "higher oil", "hot inflation", "rate hike",
    "recession", "risk", "selloff", "tariff", "war", "weak",
}


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _short(value: Any, limit: int = 240) -> str:
    text = _text(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _fresh(signal: dict, *, now_ts: float, max_age_hours: float) -> bool:
    updated = _number(signal.get("analysis_updated_ts"))
    if updated <= 0:
        return False
    return 0 <= now_ts - updated <= max_age_hours * 3600.0


def _direction(signal: dict, *, symbol: str = "") -> str:
    action = _text(signal.get("action")).upper()
    if action in {"LONG", "SHORT"}:
        return action

    votes = 0
    market_bias = _text(signal.get("market_map_bias")).upper()
    if market_bias in {"BULLISH", "UPTREND", "RECLAIM"}:
        votes += 2
    elif market_bias in {"BEARISH", "DOWNTREND", "BREAKDOWN"}:
        votes -= 2

    structure = " ".join(
        _text(signal.get(key)).upper()
        for key in ("market_regime", "dominant_regime", "structure_trend", "mtf_bias")
    )
    if any(token in structure for token in ("UPTREND", "BULLISH", "TREND_UP")):
        votes += 1
    if any(token in structure for token in ("DOWNTREND", "BEARISH", "TREND_DOWN")):
        votes -= 1

    score = _number(signal.get("score"), 50.0)
    if score >= 57.0:
        votes += 1
    elif score <= 43.0:
        votes -= 1

    move = _number(signal.get("recent_move_pct") or signal.get("move_pct_24h"))
    threshold = 2.0 if symbol == "VIXINDEX" else 0.45
    if move >= threshold:
        votes += 1
    elif move <= -threshold:
        votes -= 1
    return "LONG" if votes > 0 else "SHORT" if votes < 0 else "FLAT"


def _first_level(signal: dict, keys: tuple[str, ...]) -> float:
    for key in keys:
        value = _number(signal.get(key))
        if value > 0:
            return value
    return 0.0


def _mapped_level(values: Any, *, price: float, support: bool) -> float:
    levels = sorted({_number(value) for value in (values or []) if _number(value) > 0})
    if not levels:
        return 0.0
    if support:
        below = [value for value in levels if value <= price]
        return max(below) if below else min(levels)
    above = [value for value in levels if value >= price]
    return min(above) if above else max(levels)


def _benchmark_row(
    symbol: str,
    label: str,
    signal: dict,
    mapped: dict,
    *,
    now_ts: float,
    max_age_hours: float,
) -> dict:
    price = _number(signal.get("live_price") or signal.get("price"))
    support = _first_level(signal, ("orderbook_support", "support", "daily_breakdown_level")) or _mapped_level(
        mapped.get("supports"),
        price=price,
        support=True,
    )
    resistance = _first_level(signal, ("orderbook_resistance", "resistance", "daily_breakout_level")) or _mapped_level(
        mapped.get("resistances"),
        price=price,
        support=False,
    )
    breakdown = _first_level(signal, ("daily_breakdown_level", "orderbook_support", "support")) or _mapped_level(
        mapped.get("daily_close_short_below") or mapped.get("supports"),
        price=price,
        support=True,
    )
    reclaim = _first_level(signal, ("daily_breakout_level", "orderbook_resistance", "resistance")) or _mapped_level(
        mapped.get("daily_close_long_above") or mapped.get("resistances"),
        price=price,
        support=False,
    )
    direction = _direction(signal, symbol=symbol)
    move = _number(signal.get("recent_move_pct") or signal.get("move_pct_24h"))
    return {
        "symbol": symbol,
        "label": label,
        "price": round(price, 4),
        "move_pct": round(move, 2),
        "direction": direction,
        "fresh": _fresh(signal, now_ts=now_ts, max_age_hours=max_age_hours),
        "support": round(support, 4),
        "resistance": round(resistance, 4),
        "breakdown": round(breakdown, 4),
        "reclaim": round(reclaim, 4),
        "structure": _text(signal.get("market_regime") or signal.get("structure_trend") or "UNKNOWN").upper(),
        "why": _short(
            signal.get("market_map_summary")
            or signal.get("price_action_summary")
            or signal.get("decision_reason")
            or signal.get("reason"),
            180,
        ),
        "updated_at": _text(signal.get("analysis_updated_at")),
    }


def _catalyst_tone(text: str) -> str:
    lowered = text.lower()
    supportive = sum(word in lowered for word in _SUPPORTIVE_WORDS)
    bearish = sum(word in lowered for word in _BEARISH_WORDS)
    return "BEARISH" if bearish > supportive else "SUPPORTIVE" if supportive > bearish else "MIXED"


def _looks_like_tag_soup(text: str) -> bool:
    words = text.split()
    return bool(
        len(words) < 4
        or (
            len(words) < 10
            and not any(char in text for char in ".,:;")
            and all(word.lower() == word for word in words)
        )
    )


def _collect_catalysts(signals: dict, benchmark_rows: list[dict]) -> list[dict]:
    priority = [row["symbol"] for row in benchmark_rows]
    equity_moves = sorted(
        (
            (symbol, abs(_number(signal.get("recent_move_pct") or signal.get("move_pct_24h"))))
            for symbol, signal in signals.items()
            if str(signal.get("instrument_type") or "").lower() == "equity"
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    priority.extend(symbol for symbol, _move in equity_moves[:8])

    seen: set[str] = set()
    catalysts: list[dict] = []
    for symbol in priority:
        signal = dict(signals.get(symbol) or {})
        candidates = (
            signal.get("news_headline"),
            signal.get("official_event_summary"),
            signal.get("news_event_summary"),
            signal.get("news_catalyst_summary"),
            signal.get("sec_event_summary"),
            signal.get("analyst_revision_summary"),
        )
        for candidate in candidates:
            text = _short(candidate, 220)
            normalized = text.lower()
            if not text or normalized in seen or _looks_like_tag_soup(text):
                continue
            seen.add(normalized)
            catalysts.append({
                "symbol": symbol,
                "tone": _catalyst_tone(text),
                "text": text,
            })
            break
        if len(catalysts) >= 5:
            break
    return catalysts


def _breadth(signals: dict, *, now_ts: float, max_age_hours: float) -> dict:
    rows = []
    for symbol, raw_signal in signals.items():
        signal = dict(raw_signal or {})
        if str(signal.get("instrument_type") or "").lower() != "equity":
            continue
        if not _fresh(signal, now_ts=now_ts, max_age_hours=max_age_hours):
            continue
        rows.append((symbol, _direction(signal, symbol=symbol)))
    bullish = sum(direction == "LONG" for _symbol, direction in rows)
    bearish = sum(direction == "SHORT" for _symbol, direction in rows)
    neutral = max(0, len(rows) - bullish - bearish)
    directional = bullish + bearish
    net = ((bullish - bearish) / directional) if directional else 0.0
    return {
        "tracked": len(rows),
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "net": round(net, 4),
        "summary": (
            f"{bearish}/{directional} directional stock reads are bearish"
            if bearish > bullish
            else f"{bullish}/{directional} directional stock reads are bullish"
            if bullish > bearish
            else f"{directional} directional stock reads are evenly split"
        ) if directional else "Stock breadth is waiting for fresh reads.",
    }


def _fmt_level(value: float) -> str:
    if value <= 0:
        return "mapped support"
    return f"{value:,.2f}"


def build_us_market_context(
    state: dict | None,
    *,
    market_map: dict | None = None,
    asia_context: dict | None = None,
    now: datetime | None = None,
    max_age_hours: float = 2.0,
) -> dict:
    safe_state = dict(state or {})
    signals = dict(safe_state.get("signals") or {})
    mapped_coins = dict((market_map or {}).get("coins") or {})
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_ts = now_dt.timestamp()

    rows = [
        _benchmark_row(
            symbol,
            label,
            dict(signals.get(symbol) or {}),
            dict(mapped_coins.get(symbol) or {}),
            now_ts=now_ts,
            max_age_hours=max_age_hours,
        )
        for symbol, label in BENCHMARKS
        if signals.get(symbol)
    ]
    fresh_rows = [row for row in rows if row["fresh"]]
    breadth = _breadth(signals, now_ts=now_ts, max_age_hours=max_age_hours)
    catalysts = _collect_catalysts(signals, fresh_rows)

    risk_score = 0.0
    directional_votes: list[int] = []
    for row in fresh_rows:
        direction = row["direction"]
        vote = 1 if direction == "LONG" else -1 if direction == "SHORT" else 0
        if row["symbol"] == "VIXINDEX":
            vote *= -1
            risk_score += vote * 0.75
        else:
            risk_score += vote
        if vote:
            directional_votes.append(vote)
    risk_score += breadth["net"] * 1.25
    asia_bias = _text((asia_context or {}).get("regional_bias")).upper()

    if risk_score <= -1.1:
        regime = "RISK_OFF"
    elif risk_score >= 1.1:
        regime = "RISK_ON"
    elif fresh_rows:
        regime = "RANGE"
    else:
        regime = "UNKNOWN"

    completeness = min(1.0, len(fresh_rows) / len(BENCHMARKS))
    agreement = abs(sum(directional_votes)) / len(directional_votes) if directional_votes else 0.0
    benchmark_net = (sum(directional_votes) / len(directional_votes)) if directional_votes else 0.0
    if not breadth["bullish"] and not breadth["bearish"]:
        breadth_confirmation = 0.20
    elif benchmark_net and benchmark_net * breadth["net"] > 0:
        breadth_confirmation = abs(breadth["net"])
    elif benchmark_net and benchmark_net * breadth["net"] < 0:
        breadth_confirmation = 0.0
    else:
        breadth_confirmation = 0.35
    confidence = 0.35 * completeness + 0.35 * agreement + 0.30 * breadth_confirmation
    confidence = round(min(0.95, max(0.0, confidence)), 4)

    row_map = {row["symbol"]: row for row in rows}
    spx = row_map.get("SP500", {})
    ndx = row_map.get("NDX", {})
    primary = spx or ndx
    support = _number(primary.get("support") or primary.get("breakdown"))
    resistance = _number(primary.get("resistance") or primary.get("reclaim"))

    if regime == "RISK_OFF":
        headline = "US risk is defensive: index structure and stock breadth favor protecting capital and selling failed rallies."
        commentary = (
            f"{breadth['summary']}. A short is valid only when the individual asset is also below structure; "
            "the market anchor is context, not permission to chase an extended red candle."
        )
        buy_plan = (
            f"Do not blind-buy the dip. Reconsider longs after the S&P reclaims {_fmt_level(resistance)} "
            "and Nasdaq plus breadth confirm the reclaim."
        )
        sell_plan = (
            f"Prefer a failed rally near {_fmt_level(resistance)} or a confirmed break below {_fmt_level(support)}; "
            "use 1x leverage and invalidate on a sustained reclaim."
        )
        invalidation = "Risk-off is invalidated by an S&P/Nasdaq reclaim, falling volatility, and breadth turning positive together."
        preferred_direction = "SHORT"
    elif regime == "RISK_ON":
        headline = "US risk is constructive: index structure and stock breadth favor buying supported pullbacks."
        commentary = (
            f"{breadth['summary']}. Favor leaders holding support; avoid shorting strength unless the individual thesis breaks first."
        )
        buy_plan = f"Look for support to hold near {_fmt_level(support)} or buy a confirmed breakout through {_fmt_level(resistance)}."
        sell_plan = "Shorts require an index failure plus asset-specific breakdown; a single weak candle is not enough."
        invalidation = "Risk-on is invalidated if S&P and Nasdaq lose mapped support while volatility and bearish breadth expand."
        preferred_direction = "LONG"
    elif regime == "RANGE":
        headline = "US risk is mixed: the indices do not offer a clean directional anchor yet."
        commentary = f"{breadth['summary']}. Trade the mapped edges selectively and keep size below normal."
        buy_plan = f"Buy only a defended support near {_fmt_level(support)} or a clean reclaim through {_fmt_level(resistance)}."
        sell_plan = f"Sell only a rejected resistance near {_fmt_level(resistance)} or a confirmed support loss."
        invalidation = "The range resolves when S&P, Nasdaq, VIX, and breadth agree in one direction."
        preferred_direction = "SELECTIVE"
    else:
        headline = "US market anchor is waiting for fresh S&P, Nasdaq, and volatility reads."
        commentary = "No market-wide directional permission is available until the benchmark feed refreshes."
        buy_plan = "Wait for fresh benchmark structure."
        sell_plan = "Wait for fresh benchmark structure."
        invalidation = "Not applicable until fresh data arrives."
        preferred_direction = "SELECTIVE"

    return {
        "enabled": True,
        "active": bool(fresh_rows),
        "updated_at": now_dt.isoformat(),
        "regime": regime,
        "confidence": confidence,
        "headline": headline,
        "commentary": commentary,
        "buy_plan": buy_plan,
        "sell_plan": sell_plan,
        "invalidation": invalidation,
        "benchmarks": rows,
        "breadth": breadth,
        "catalysts": catalysts,
        "asia_bias": asia_bias or "UNKNOWN",
        "execution_policy": {
            "preferred_direction": preferred_direction,
            "allow_tactical_short": regime == "RISK_OFF" and confidence >= 0.55,
            "max_aligned_short_leverage": 1,
            "aligned_short_size_multiplier": 0.50,
            "countertrend_size_multiplier": 0.35,
            "summary": (
                "Small 1x shorts are eligible when the stock's own bearish structure confirms the market regime."
                if regime == "RISK_OFF"
                else "Use the market regime as a filter; individual thesis and structure still control entry."
            ),
        },
        "fresh_count": len(fresh_rows),
        "risk_score": round(risk_score, 3),
    }


def assess_trade_alignment(
    context: dict | None,
    *,
    direction: str,
    instrument_type: str,
    signal: dict | None = None,
    min_confidence: float = 0.55,
    block_countertrend: bool = True,
) -> dict:
    safe = dict(context or {})
    signal = dict(signal or {})
    side = _text(direction).upper()
    instrument = _text(instrument_type).lower()
    regime = _text(safe.get("regime")).upper() or "UNKNOWN"
    confidence = _number(safe.get("confidence"))
    applies = instrument in {"equity", "index"} and side in {"LONG", "SHORT"}
    policy = dict(safe.get("execution_policy") or {})
    cross_market_state = _text(policy.get("cross_market_state")).upper() or "UNCONFIRMED"
    cross_market_multiplier = max(
        0.10,
        min(1.0, _number(policy.get("cross_market_size_multiplier"), 1.0)),
    )
    if not applies or not safe.get("active") or confidence < min_confidence:
        return {
            "active": False,
            "permitted": True,
            "aligned": False,
            "supporting_driver": False,
            "size_multiplier": 1.0,
            "cross_market_size_multiplier": cross_market_multiplier,
            "cross_market_state": cross_market_state,
            "leverage_cap": 0,
            "summary": "Global market anchor is not active for this entry.",
        }

    own_direction = _direction(signal)
    map_bias = _text(signal.get("market_map_bias")).upper()
    structure = " ".join(
        _text(signal.get(key)).upper()
        for key in ("market_regime", "structure_trend", "mtf_bias", "orderbook_breakout_state")
    )
    bearish_structure = own_direction == "SHORT" or map_bias == "BEARISH" or any(
        token in structure for token in ("DOWNTREND", "BEARISH", "BREAKDOWN")
    )
    bullish_structure = own_direction == "LONG" or map_bias == "BULLISH" or any(
        token in structure for token in ("UPTREND", "BULLISH", "RECLAIM")
    )
    aligned = (regime == "RISK_OFF" and side == "SHORT" and bearish_structure) or (
        regime == "RISK_ON" and side == "LONG" and bullish_structure
    )
    countertrend = (regime == "RISK_OFF" and side == "LONG" and not bullish_structure) or (
        regime == "RISK_ON" and side == "SHORT" and not bearish_structure
    )

    permitted = not (countertrend and block_countertrend and confidence >= 0.68)
    if regime == "RISK_OFF" and aligned:
        summary = "Risk-off alignment confirms a tactical short; keep it small, 1x, and invalidate on the index reclaim."
        size_multiplier = _number(policy.get("aligned_short_size_multiplier"), 0.50)
        leverage_cap = max(1, int(_number(policy.get("max_aligned_short_leverage"), 1)))
    elif aligned:
        summary = "Risk-on alignment supports the long, but the asset's own thesis still controls risk."
        size_multiplier = 0.80
        leverage_cap = 0
    elif countertrend:
        summary = (
            f"{regime.replace('_', ' ').title()} conflicts with this {side.lower()} and the asset has not confirmed a reversal."
        )
        size_multiplier = _number(policy.get("countertrend_size_multiplier"), 0.35)
        leverage_cap = 1
    else:
        summary = "Market regime is mixed for this entry; use selective size and asset-specific invalidation."
        size_multiplier = 0.65
        leverage_cap = 0

    size_multiplier *= cross_market_multiplier
    if cross_market_multiplier < 1.0:
        summary = (
            f"{summary} Cross-market state is {cross_market_state.lower()}, "
            f"so qualified size is reduced to {cross_market_multiplier:.0%}."
        )

    return {
        "active": True,
        "permitted": permitted,
        "aligned": aligned,
        "countertrend": countertrend,
        "supporting_driver": bool(aligned),
        "regime": regime,
        "confidence": round(confidence, 4),
        "size_multiplier": round(max(0.10, min(1.0, size_multiplier)), 4),
        "cross_market_size_multiplier": round(cross_market_multiplier, 4),
        "cross_market_state": cross_market_state,
        "leverage_cap": leverage_cap,
        "summary": summary,
        "invalidation": _text(safe.get("invalidation")),
    }
