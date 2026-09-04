"""Fresh, evidence-weighted weekly and monthly market outlooks.

This layer deliberately synthesizes existing agent evidence instead of making
another network request. It is a market prior, not an entry signal: individual
asset structure and risk checks still decide whether a trade is allowed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Iterable


MAJOR_CRYPTO = ("BTC", "ETH", "SOL", "HYPE")
US_BENCHMARKS = ("SP500", "NDX", "VIXINDEX")
MAX_SIGNAL_AGE_HOURS = 6.0


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value if value is not None else default)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _parse_datetime(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _signal_age_hours(signal: dict, *, now: datetime) -> float | None:
    timestamp = _number(signal.get("analysis_updated_ts"))
    if timestamp > 0:
        return max(0.0, (now.timestamp() - timestamp) / 3600.0)
    parsed = _parse_datetime(signal.get("analysis_updated_at"))
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds() / 3600.0)


def _fresh(signal: dict, *, now: datetime, max_age_hours: float = MAX_SIGNAL_AGE_HOURS) -> bool:
    age = _signal_age_hours(signal, now=now)
    return age is not None and age <= max_age_hours


def _contains_any(value: Any, positive: Iterable[str], negative: Iterable[str]) -> float:
    text = _text(value).upper()
    if any(token in text for token in positive):
        return 1.0
    if any(token in text for token in negative):
        return -1.0
    return 0.0


def _signal_bias(signal: dict) -> float:
    """Map independent directional fields to a bounded structural vote."""
    votes: list[tuple[float, float]] = []
    action = _text(signal.get("action") or signal.get("decision")).upper()
    if action in {"LONG", "SHORT"}:
        votes.append((1.0 if action == "LONG" else -1.0, 0.65))

    structure = _contains_any(
        " ".join(
            _text(signal.get(key))
            for key in ("market_regime", "structure_trend", "dominant_regime")
        ),
        ("UPTREND", "TREND_UP", "BULLISH", "BREAKOUT"),
        ("DOWNTREND", "TREND_DOWN", "BEARISH", "BREAKDOWN"),
    )
    if structure:
        votes.append((structure, 1.0))

    mtf = _contains_any(
        " ".join(_text(signal.get(key)) for key in ("mtf_bias", "mtf_reason")),
        ("BULLISH", "LONG"),
        ("BEARISH", "SHORT"),
    )
    if mtf:
        votes.append((mtf, 1.0))

    mapped = _contains_any(
        " ".join(
            _text(signal.get(key))
            for key in ("market_map_bias", "strategic_bias", "crypto_directional_bias")
        ),
        ("BULLISH", "LONG", "RISK_ON"),
        ("BEARISH", "SHORT", "RISK_OFF"),
    )
    if mapped:
        votes.append((mapped, 0.55))

    if not votes:
        return 0.0
    weight = sum(item[1] for item in votes)
    return round(_clamp(sum(value * item_weight for value, item_weight in votes) / weight), 4)


def _direction_counts(signals: list[dict]) -> dict:
    values = [_signal_bias(signal) for signal in signals]
    bullish = sum(value >= 0.25 for value in values)
    bearish = sum(value <= -0.25 for value in values)
    neutral = max(0, len(values) - bullish - bearish)
    directional = bullish + bearish
    net = ((bullish - bearish) / directional) if directional else 0.0
    return {
        "tracked": len(values),
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "directional": directional,
        "net": round(net, 4),
        "average_bias": round(sum(values) / len(values), 4) if values else 0.0,
    }


def _component(label: str, value: float | None, weight: float) -> dict:
    return {
        "label": label,
        "value": None if value is None else round(_clamp(value), 4),
        "weight": float(weight),
    }


def _forecast(
    components: list[dict],
    *,
    horizon: str,
    sample_ratio: float,
    confidence_cap: float = 0.78,
) -> dict:
    available = [row for row in components if row.get("value") is not None]
    total_weight = sum(_number(row.get("weight")) for row in components) or 1.0
    available_weight = sum(_number(row.get("weight")) for row in available)
    coverage = _clamp(available_weight / total_weight, 0.0, 1.0)
    if not available or coverage < 0.34:
        return {
            "horizon": horizon,
            "call": "NO FRESH CALL",
            "green_probability": 0.50,
            "red_probability": 0.50,
            "confidence": round(min(0.32, coverage), 4),
            "score": 0.0,
            "coverage": round(coverage, 4),
            "agreement": 0.0,
            "components": components,
        }

    score = sum(_number(row["value"]) * _number(row["weight"]) for row in available) / available_weight
    dispersion = sum(
        abs(_number(row["value"]) - score) * _number(row["weight"])
        for row in available
    ) / available_weight
    agreement = _clamp(1.0 - dispersion, 0.0, 1.0)
    confidence = min(
        confidence_cap,
        0.44 * coverage + 0.34 * agreement + 0.22 * _clamp(sample_ratio, 0.0, 1.0),
    )
    green_probability = _clamp(0.50 + score * 0.22 * confidence, 0.32, 0.68)
    call = (
        "GREEN LEAN"
        if green_probability >= 0.56
        else "RED LEAN"
        if green_probability <= 0.44
        else "MIXED"
    )
    return {
        "horizon": horizon,
        "call": call,
        "green_probability": round(green_probability, 4),
        "red_probability": round(1.0 - green_probability, 4),
        "confidence": round(confidence, 4),
        "score": round(score, 4),
        "coverage": round(coverage, 4),
        "agreement": round(agreement, 4),
        "components": components,
    }


def _call_phrase(forecast: dict, horizon: str) -> str:
    call = _text(forecast.get("call")).upper()
    green = round(_number(forecast.get("green_probability"), 0.50) * 100)
    if call == "GREEN LEAN":
        return f"{horizon} leans green ({green}% estimated chance)."
    if call == "RED LEAN":
        return f"{horizon} leans red ({100 - green}% estimated chance)."
    if call == "NO FRESH CALL":
        return f"{horizon} has no fresh evidence-backed call."
    return f"{horizon} is mixed ({green}% green / {100 - green}% red)."


def _fmt_level(value: Any) -> str:
    number = _number(value)
    if number <= 0:
        return "mapped structure"
    return f"{number:,.0f}" if number >= 1_000 else f"{number:,.2f}"


def _benchmark_rows(global_context: dict) -> list[dict]:
    return [
        dict(row or {})
        for row in list(global_context.get("benchmarks") or [])
        if row and row.get("symbol") in US_BENCHMARKS and row.get("fresh")
    ]


def _stock_outlook(
    signals: dict,
    global_context: dict,
    ai_context: dict,
    *,
    now: datetime,
) -> dict:
    equities_all = [
        dict(signal or {})
        for signal in signals.values()
        if _text(dict(signal or {}).get("instrument_type")).lower() == "equity"
    ]
    equities = [signal for signal in equities_all if _fresh(signal, now=now)]
    breadth = _direction_counts(equities)
    benchmarks = _benchmark_rows(global_context)

    benchmark_votes = []
    for row in benchmarks:
        direction = _text(row.get("direction")).upper()
        value = 1.0 if direction == "LONG" else -1.0 if direction == "SHORT" else 0.0
        if row.get("symbol") == "VIXINDEX":
            value *= -1.0
        benchmark_votes.append(value)
    benchmark_score = (
        sum(benchmark_votes) / len(benchmark_votes)
        if benchmark_votes
        else None
    )

    regime = _text(global_context.get("regime")).upper()
    global_score = 1.0 if regime == "RISK_ON" else -1.0 if regime == "RISK_OFF" else 0.0 if regime == "RANGE" else None
    tactical_bias = _text(dict(global_context.get("cross_market") or {}).get("tactical_bias")).upper()
    cross_score = 1.0 if tactical_bias == "LONG" else -1.0 if tactical_bias == "SHORT" else 0.0 if global_context.get("active") else None

    sectors = [dict(row or {}) for row in list(ai_context.get("sectors") or []) if row]
    ai_5d = None
    ai_20d = None
    ai_breadth = None
    if sectors:
        ai_5d = math.tanh(sum(_number(row.get("move_5d")) for row in sectors) / len(sectors) / 3.0)
        ai_20d = math.tanh(sum(_number(row.get("move_20d")) for row in sectors) / len(sectors) / 7.0)
        ai_breadth = (
            sum(_number(row.get("breadth_pct")) for row in sectors) / len(sectors) - 50.0
        ) / 50.0

    sample_ratio = min(1.0, len(equities) / max(12.0, float(len(equities_all) or 1)))
    next_week = _forecast(
        [
            _component("US index regime", global_score, 0.25),
            _component("US, Korea and Japan alignment", cross_score, 0.12),
            _component("S&P, Nasdaq and VIX", benchmark_score, 0.18),
            _component("Fresh stock breadth", breadth["net"] if equities else None, 0.25),
            _component("AI sector five-day trend", ai_5d, 0.12),
            _component("AI sector breadth", ai_breadth, 0.08),
        ],
        horizon="Next 7 days",
        sample_ratio=sample_ratio,
    )
    next_month = _forecast(
        [
            _component("Fresh stock structure", breadth["average_bias"] if equities else None, 0.32),
            _component("AI sector 20-day trend", ai_20d, 0.28),
            _component("S&P, Nasdaq and VIX", benchmark_score, 0.18),
            _component("US index regime", global_score, 0.12),
            _component("AI sector breadth", ai_breadth, 0.10),
        ],
        horizon="Next 30 days",
        sample_ratio=sample_ratio,
        confidence_cap=0.74,
    )

    directional = breadth["directional"]
    leading_sector = sectors[0] if sectors else {}
    lagging_sector = sectors[-1] if sectors else {}
    evidence = [
        {
            "label": "Index anchor",
            "value": regime.replace("_", " ") or "NO FRESH READ",
            "detail": (
                f"{len(benchmarks)}/3 S&P, Nasdaq and VIX inputs are fresh; "
                f"cross-market state is {_text(dict(global_context.get('cross_market') or {}).get('state')).replace('_', ' ').lower() or 'unconfirmed'}."
            ),
        },
        {
            "label": "Stock breadth",
            "value": f"{breadth['bullish']} bullish / {breadth['bearish']} bearish",
            "detail": f"{len(equities)} fresh stock reads; {directional} have a directional structure.",
        },
        {
            "label": "Sector proof",
            "value": (
                f"{_text(leading_sector.get('label'))} leads"
                if leading_sector
                else "Sector tape unavailable"
            ),
            "detail": (
                f"Leader {_number(leading_sector.get('move_5d')):+.1f}% over five days; "
                f"weakest {_text(lagging_sector.get('label')) or 'sector'} {_number(lagging_sector.get('move_5d')):+.1f}%."
                if leading_sector
                else "No sector claim is used until the reference tape refreshes."
            ),
        },
    ]

    primary = next_week
    spx = next((row for row in benchmarks if row.get("symbol") == "SP500"), {})
    support = spx.get("support") or spx.get("breakdown")
    resistance = spx.get("resistance") or spx.get("reclaim")
    if primary["call"] == "GREEN LEAN":
        invalidation = (
            f"The green lean fails if the S&P loses {_fmt_level(support)}, Nasdaq turns down, "
            "and bearish stock breadth expands together."
        )
    elif primary["call"] == "RED LEAN":
        invalidation = (
            f"The red lean fails if the S&P reclaims {_fmt_level(resistance)}, Nasdaq confirms, "
            "and stock breadth turns positive together."
        )
    else:
        invalidation = (
            f"A directional view starts only after the S&P clears {_fmt_level(resistance)} or loses "
            f"{_fmt_level(support)}, with Nasdaq, VIX, and breadth confirming."
        )

    active = len(equities) >= 3 and bool(len(benchmarks) >= 2 or sectors)
    if not active:
        next_week = _forecast([], horizon="Next 7 days", sample_ratio=0.0)
        next_month = _forecast([], horizon="Next 30 days", sample_ratio=0.0, confidence_cap=0.74)
    return {
        "key": "stocks",
        "label": "Stocks",
        "active": active,
        "next_week": next_week,
        "next_month": next_month,
        "thesis": f"{_call_phrase(next_week, 'The coming week')} {_call_phrase(next_month, 'The coming month')}",
        "evidence": evidence,
        "invalidation": invalidation,
        "data_quality": {
            "quality": "HIGH" if sample_ratio >= 0.75 and len(benchmarks) >= 2 else "MEDIUM" if active else "LOW",
            "fresh_inputs": len(equities) + len(benchmarks),
            "tracked_inputs": len(equities_all) + len(US_BENCHMARKS),
            "coverage": round(sample_ratio, 4),
        },
    }


def _crypto_outlook(signals: dict, *, now: datetime) -> dict:
    crypto_all = [
        dict(signal or {})
        for signal in signals.values()
        if _text(dict(signal or {}).get("instrument_type")).lower() == "crypto"
    ]
    crypto = [signal for signal in crypto_all if _fresh(signal, now=now)]
    fresh_by_symbol = {
        symbol: dict(signals.get(symbol) or {})
        for symbol in MAJOR_CRYPTO
        if signals.get(symbol) and _fresh(dict(signals.get(symbol) or {}), now=now)
    }
    majors = list(fresh_by_symbol.values())
    breadth = _direction_counts(crypto)
    major_breadth = _direction_counts(majors)

    mode_votes = []
    for signal in majors:
        mode = _text(signal.get("crypto_market_mode")).upper()
        if mode == "RISK_ON":
            mode_votes.append(1.0)
        elif mode in {"RISK_OFF", "DRAWDOWN"}:
            mode_votes.append(-1.0)
    mode_score = sum(mode_votes) / len(mode_votes) if mode_votes else None

    moves = [_number(signal.get("move_pct_24h") or signal.get("recent_move_pct")) for signal in majors]
    momentum_score = math.tanh((sum(moves) / len(moves)) / 2.5) if moves else None

    funding_values = [
        _number(signal.get("funding_rate"))
        for signal in crypto
        if signal.get("funding_rate") is not None
    ]
    average_funding = sum(funding_values) / len(funding_values) if funding_values else 0.0
    positioning_score = 0.0
    positioning_label = "BALANCED"
    if average_funding >= 0.0005:
        positioning_score = -0.35
        positioning_label = "LONGS CROWDED"
    elif average_funding <= -0.0005:
        positioning_score = 0.20
        positioning_label = "SHORTS CROWDED"

    sample_ratio = min(1.0, len(crypto) / max(10.0, float(len(crypto_all) or 1)))
    major_ratio = len(majors) / len(MAJOR_CRYPTO)
    next_week = _forecast(
        [
            _component("BTC, ETH, SOL and HYPE structure", major_breadth["average_bias"] if majors else None, 0.34),
            _component("Broad crypto structure", breadth["net"] if crypto else None, 0.24),
            _component("Crypto risk mode", mode_score, 0.18),
            _component("Major-coin 24-hour momentum", momentum_score, 0.14),
            _component("Perpetual positioning", positioning_score if funding_values else None, 0.10),
        ],
        horizon="Next 7 days",
        sample_ratio=min(sample_ratio, major_ratio),
    )
    next_month = _forecast(
        [
            _component("Major-coin multi-timeframe structure", major_breadth["average_bias"] if majors else None, 0.46),
            _component("Broad crypto structure", breadth["average_bias"] if crypto else None, 0.34),
            _component("Crypto risk mode", mode_score, 0.14),
            _component("Perpetual positioning", positioning_score if funding_values else None, 0.06),
        ],
        horizon="Next 30 days",
        sample_ratio=min(sample_ratio, major_ratio),
        confidence_cap=0.66,
    )

    major_names = list(fresh_by_symbol)
    btc = fresh_by_symbol.get("BTC", {})
    crypto_summary = _text(btc.get("crypto_structure_summary"))
    evidence = [
        {
            "label": "Major coins",
            "value": f"{major_breadth['bullish']} bullish / {major_breadth['bearish']} bearish",
            "detail": f"Fresh structure from {', '.join(major_names) or 'none of the four majors'}.",
        },
        {
            "label": "Crypto breadth",
            "value": f"{breadth['bullish']} bullish / {breadth['bearish']} bearish",
            "detail": f"{len(crypto)} fresh crypto reads. {crypto_summary or 'The broad structure model is still building.'}",
        },
        {
            "label": "Leverage check",
            "value": positioning_label,
            "detail": (
                f"Average sampled funding is {average_funding * 100:+.4f}% per interval across "
                f"{len(funding_values)} markets."
                if funding_values
                else "Funding evidence is unavailable and does not influence the call."
            ),
        },
    ]

    support = btc.get("market_map_nearest_support") or btc.get("daily_breakdown_level")
    resistance = btc.get("market_map_nearest_resistance") or btc.get("daily_breakout_level")
    if next_week["call"] == "GREEN LEAN":
        invalidation = (
            f"The green lean fails if BTC loses {_fmt_level(support)} and at least three of BTC, ETH, SOL, "
            "and HYPE turn structurally bearish."
        )
    elif next_week["call"] == "RED LEAN":
        invalidation = (
            f"The red lean fails if BTC reclaims {_fmt_level(resistance)} and at least three major coins "
            "turn structurally bullish."
        )
    else:
        invalidation = (
            f"A directional crypto call needs BTC above {_fmt_level(resistance)} or below {_fmt_level(support)}, "
            "confirmed by at least three major coins."
        )

    active = len(majors) >= 2 and bool(crypto)
    if not active:
        next_week = _forecast([], horizon="Next 7 days", sample_ratio=0.0)
        next_month = _forecast([], horizon="Next 30 days", sample_ratio=0.0, confidence_cap=0.66)
    return {
        "key": "crypto",
        "label": "Crypto",
        "active": active,
        "next_week": next_week,
        "next_month": next_month,
        "thesis": f"{_call_phrase(next_week, 'The coming week')} {_call_phrase(next_month, 'The coming month')}",
        "evidence": evidence,
        "invalidation": invalidation,
        "data_quality": {
            "quality": "HIGH" if major_ratio == 1.0 and sample_ratio >= 0.70 else "MEDIUM" if active else "LOW",
            "fresh_inputs": len(crypto),
            "tracked_inputs": len(crypto_all),
            "coverage": round(sample_ratio, 4),
            "fresh_majors": len(majors),
        },
    }


def build_weekly_outlook(
    state: dict | None,
    *,
    global_context: dict | None = None,
    ai_infrastructure: dict | None = None,
    now: datetime | None = None,
) -> dict:
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_dt = now_dt.astimezone(timezone.utc)
    safe_state = dict(state or {})
    signals = dict(safe_state.get("signals") or {})
    global_read = dict(
        global_context
        or safe_state.get("global_market_context")
        or safe_state.get("us_market_context")
        or {}
    )
    ai_read = dict(ai_infrastructure or safe_state.get("ai_infrastructure") or {})

    week_start = (now_dt - timedelta(days=now_dt.weekday())).date()
    valid_through = week_start + timedelta(days=7)
    stocks = _stock_outlook(signals, global_read, ai_read, now=now_dt)
    crypto = _crypto_outlook(signals, now=now_dt)
    active = bool(stocks.get("active") or crypto.get("active"))
    return {
        "enabled": True,
        "active": active,
        "week_id": f"{now_dt.isocalendar().year}-W{now_dt.isocalendar().week:02d}",
        "week_start": week_start.isoformat(),
        "valid_through": valid_through.isoformat(),
        "updated_at": now_dt.isoformat(),
        "title": "Punky's weekly note",
        "subtitle": "A probability-weighted view of the next week and month, built from the same evidence that governs trades.",
        "markets": {
            "stocks": stocks,
            "crypto": crypto,
        },
        "methodology": (
            "Calls combine fresh price structure, breadth, benchmark confirmation, cross-market context, "
            "sector trends, major-coin alignment, and positioning. Probabilities are bounded estimates, not promises; "
            "stale or thin evidence produces no call."
        ),
    }
