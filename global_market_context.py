"""Combined US, Korea, and Japan market context for analysis and execution."""

from __future__ import annotations

from datetime import datetime, timezone
from math import sqrt
from typing import Any

import asia_session
import us_market_context


REGION_REFERENCES = {
    "Korea": ("KR200", "EWY"),
    "Japan": ("JP225", "EWJ"),
}
US_REFERENCES = ("SP500", "NDX")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _bar_timestamp(bar: dict) -> int:
    raw = (
        bar.get("timestamp")
        or bar.get("time")
        or bar.get("open_time")
        or bar.get("t")
        or bar.get("date")
    )
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return int(parsed.timestamp())
        except ValueError:
            return 0
    value = _number(raw)
    if value > 10_000_000_000:
        value /= 1000.0
    return int(value)


def _bar_close(bar: dict) -> float:
    return _number(bar.get("close") or bar.get("c") or bar.get("price"))


def _return_series(signal: dict) -> dict[int, float]:
    raw_bars = list(dict(signal.get("price_action") or {}).get("bars") or [])
    points = sorted(
        (
            (_bar_timestamp(dict(bar or {})), _bar_close(dict(bar or {})))
            for bar in raw_bars
        ),
        key=lambda item: item[0],
    )
    points = [(timestamp, close) for timestamp, close in points if timestamp > 0 and close > 0]
    returns: dict[int, float] = {}
    previous = 0.0
    for timestamp, close in points:
        if previous > 0:
            returns[timestamp] = (close / previous) - 1.0
        previous = close
    return returns


def _pearson(left: dict[int, float], right: dict[int, float]) -> tuple[float | None, int]:
    timestamps = sorted(set(left).intersection(right))
    count = len(timestamps)
    if count < 12:
        return None, count
    left_values = [left[timestamp] for timestamp in timestamps]
    right_values = [right[timestamp] for timestamp in timestamps]
    left_mean = sum(left_values) / count
    right_mean = sum(right_values) / count
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_values, right_values)
    )
    left_variance = sum((value - left_mean) ** 2 for value in left_values)
    right_variance = sum((value - right_mean) ** 2 for value in right_values)
    denominator = sqrt(left_variance * right_variance)
    if denominator <= 0:
        return None, count
    return max(-1.0, min(1.0, numerator / denominator)), count


def _relationship(value: float | None) -> str:
    if value is None:
        return "INSUFFICIENT"
    if value >= 0.65:
        return "STRONG POSITIVE"
    if value >= 0.30:
        return "POSITIVE"
    if value <= -0.65:
        return "STRONG NEGATIVE"
    if value <= -0.30:
        return "NEGATIVE"
    return "WEAK"


def _sample_quality(count: int) -> str:
    if count >= 36:
        return "HIGH"
    if count >= 18:
        return "MEDIUM"
    if count >= 12:
        return "LOW"
    return "INSUFFICIENT"


def _current_us_direction(us_context: dict) -> str:
    regime = _text(us_context.get("regime")).upper()
    if regime == "RISK_ON":
        return "LONG"
    if regime == "RISK_OFF":
        return "SHORT"
    rows = [
        row for row in list(us_context.get("benchmarks") or [])
        if row.get("symbol") in US_REFERENCES and row.get("fresh")
    ]
    votes = [
        1 if row.get("direction") == "LONG" else -1
        for row in rows
        if row.get("direction") in {"LONG", "SHORT"}
    ]
    return "LONG" if sum(votes) > 0 else "SHORT" if sum(votes) < 0 else "FLAT"


def _region_direction(asia_context: dict, region: str) -> str:
    rows = [
        row for row in list(asia_context.get("benchmarks") or [])
        if row.get("region") == region and row.get("fresh")
    ]
    votes = [
        1 if row.get("direction") == "LONG" else -1
        for row in rows
        if row.get("direction") in {"LONG", "SHORT"}
    ]
    return "LONG" if sum(votes) > 0 else "SHORT" if sum(votes) < 0 else "FLAT"


def _best_correlation(signals: dict, region: str) -> dict:
    candidates = []
    for us_symbol in US_REFERENCES:
        us_returns = _return_series(dict(signals.get(us_symbol) or {}))
        if not us_returns:
            continue
        for regional_symbol in REGION_REFERENCES[region]:
            regional_returns = _return_series(dict(signals.get(regional_symbol) or {}))
            if not regional_returns:
                continue
            correlation, count = _pearson(us_returns, regional_returns)
            candidates.append({
                "us_symbol": us_symbol,
                "regional_symbol": regional_symbol,
                "value": correlation,
                "sample_size": count,
            })
    if not candidates:
        return {
            "us_symbol": "",
            "regional_symbol": "",
            "value": None,
            "sample_size": 0,
        }

    quality_rank = {
        "INSUFFICIENT": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
    }
    return max(
        candidates,
        key=lambda row: (
            quality_rank[_sample_quality(row["sample_size"])],
            abs(row["value"]) if row["value"] is not None else -1.0,
            row["sample_size"],
            row["us_symbol"] == "SP500",
            row["regional_symbol"] == REGION_REFERENCES[region][0],
        ),
    )


def _region_read(
    region: str,
    *,
    signals: dict,
    us_context: dict,
    asia_context: dict,
) -> dict:
    correlation = _best_correlation(signals, region)
    value = correlation["value"]
    relationship = _relationship(value)
    us_direction = _current_us_direction(us_context)
    regional_direction = _region_direction(asia_context, region)
    meaningful = value is not None and abs(value) >= 0.30
    implied_us_direction = regional_direction
    if meaningful and value < 0 and regional_direction in {"LONG", "SHORT"}:
        implied_us_direction = "SHORT" if regional_direction == "LONG" else "LONG"

    if (
        meaningful
        and us_direction in {"LONG", "SHORT"}
        and implied_us_direction in {"LONG", "SHORT"}
    ):
        status = "CONFIRMING" if implied_us_direction == us_direction else "DIVERGING"
    elif meaningful and implied_us_direction in {"LONG", "SHORT"}:
        status = "LEADING"
    elif value is not None and regional_direction in {"LONG", "SHORT"}:
        status = "WATCH"
    else:
        status = "NO EDGE"

    formatted = f"{value:+.2f}" if value is not None else "n/a"
    if status == "CONFIRMING":
        summary = (
            f"{region} confirms the US {us_direction.lower()} read. "
            f"{correlation['us_symbol']}/{correlation['regional_symbol']} hourly-return correlation is "
            f"{formatted} across {correlation['sample_size']} matched observations."
        )
    elif status == "DIVERGING":
        summary = (
            f"{region} contradicts the current US {us_direction.lower()} read after accounting for a "
            f"{relationship.lower()} relationship ({formatted}, {correlation['sample_size']} observations). "
            "Treat conviction as lower until the tapes reconnect."
        )
    elif status == "LEADING":
        summary = (
            f"{region} is an early {implied_us_direction.lower()} warning while the US tape is unresolved. "
            f"Its {relationship.lower()} US relationship is {formatted} across "
            f"{correlation['sample_size']} matched observations."
        )
    elif status == "WATCH":
        summary = (
            f"{region} is moving {regional_direction.lower()}, but its {relationship.lower()} US relationship "
            f"({formatted}, {correlation['sample_size']} observations) is not reliable enough to change the plan."
        )
    else:
        summary = f"{region} has no reliable matched-bar correlation edge yet."

    return {
        "region": region,
        "direction": regional_direction,
        "status": status,
        "correlation": round(value, 4) if value is not None else None,
        "relationship": relationship,
        "sample_size": correlation["sample_size"],
        "sample_quality": _sample_quality(correlation["sample_size"]),
        "us_symbol": correlation["us_symbol"],
        "regional_symbol": correlation["regional_symbol"],
        "implied_us_direction": implied_us_direction,
        "summary": summary,
    }


def build_global_market_context(
    state: dict | None,
    *,
    market_map: dict | None = None,
    asia_context: dict | None = None,
    us_context: dict | None = None,
    now: datetime | None = None,
    max_age_hours: float = 2.0,
) -> dict:
    safe_state = dict(state or {})
    signals = dict(safe_state.get("signals") or {})
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    asia = dict(asia_context or asia_session.build_asia_session(safe_state, now=now_dt))
    us = dict(
        us_context
        or us_market_context.build_us_market_context(
            safe_state,
            market_map=market_map,
            asia_context=asia,
            now=now_dt,
            max_age_hours=max_age_hours,
        )
    )
    correlations = [
        _region_read(
            region,
            signals=signals,
            us_context=us,
            asia_context=asia,
        )
        for region in REGION_REFERENCES
    ]
    confirmations = sum(row["status"] == "CONFIRMING" for row in correlations)
    divergences = sum(row["status"] == "DIVERGING" for row in correlations)
    leading_rows = [row for row in correlations if row["status"] == "LEADING"]
    leading_directions = {
        row["implied_us_direction"]
        for row in leading_rows
        if row["implied_us_direction"] in {"LONG", "SHORT"}
    }
    lead_direction = next(iter(leading_directions)) if len(leading_directions) == 1 else "FLAT"
    us_direction = _current_us_direction(us)
    if us_direction in {"LONG", "SHORT"}:
        if confirmations == len(correlations):
            cross_state = "CONFIRMED"
        elif confirmations and not divergences:
            cross_state = "SUPPORTED"
        elif confirmations and divergences:
            cross_state = "MIXED"
        elif divergences:
            cross_state = "DIVERGENT"
        else:
            cross_state = "UNCONFIRMED"
    elif len(leading_rows) == len(correlations) and lead_direction in {"LONG", "SHORT"}:
        cross_state = "ASIA_LEADS"
    elif leading_rows and lead_direction in {"LONG", "SHORT"}:
        cross_state = "EARLY_SIGNAL"
    elif len(leading_directions) > 1:
        cross_state = "MIXED"
    else:
        cross_state = "UNCONFIRMED"

    confidence_shift = {
        "CONFIRMED": 0.08,
        "SUPPORTED": 0.04,
        "ASIA_LEADS": 0.03,
        "EARLY_SIGNAL": 0.0,
        "MIXED": -0.08,
        "DIVERGENT": -0.15,
        "UNCONFIRMED": 0.0,
    }[cross_state]
    base_confidence = _number(us.get("confidence"))
    confidence = round(max(0.0, min(0.95, base_confidence + confidence_shift)), 4)
    size_multiplier = {
        "CONFIRMED": 1.0,
        "SUPPORTED": 1.0,
        "ASIA_LEADS": 0.75,
        "EARLY_SIGNAL": 0.85,
        "MIXED": 0.75,
        "DIVERGENT": 0.60,
        "UNCONFIRMED": 1.0,
    }[cross_state]
    tactical_bias = us_direction if us_direction in {"LONG", "SHORT"} else lead_direction
    if tactical_bias not in {"LONG", "SHORT"}:
        tactical_bias = "SELECTIVE"
    direction_text = tactical_bias.lower()
    breadth = dict(us.get("breadth") or {})
    breadth_net = _number(breadth.get("net"))
    breadth_conflicts = (
        tactical_bias == "SHORT" and breadth_net >= 0.25
    ) or (
        tactical_bias == "LONG" and breadth_net <= -0.25
    )
    if cross_state in {"ASIA_LEADS", "EARLY_SIGNAL"} and tactical_bias == "SHORT":
        display_bias = "DEFENSIVE"
    elif cross_state in {"ASIA_LEADS", "EARLY_SIGNAL"} and tactical_bias == "LONG":
        display_bias = "EARLY LONG"
    elif tactical_bias in {"LONG", "SHORT"}:
        display_bias = tactical_bias
    else:
        display_bias = "SELECTIVE"

    if cross_state == "CONFIRMED":
        cross_summary = (
            f"Korea and Japan both confirm the US {direction_text} read. "
            "Normal qualified size is available."
        )
    elif cross_state == "SUPPORTED":
        cross_summary = (
            f"One Asian market confirms the US {direction_text} read and none contradict it. "
            "Keep normal qualified size."
        )
    elif cross_state == "ASIA_LEADS":
        cross_summary = (
            f"Korea and Japan both point {direction_text} before the US tape confirms. "
            "Treat this as an early warning: require stock-level confirmation and cap qualified size at 75%."
        )
    elif cross_state == "EARLY_SIGNAL":
        cross_summary = (
            f"One Asian market points {direction_text} while the US tape is unresolved. "
            "Keep size at 85% and wait for either the second market or US price action to confirm."
        )
    elif cross_state == "MIXED":
        cross_summary = "Korea and Japan do not agree on the US handoff. Reduce qualified stock size to 75%."
    elif cross_state == "DIVERGENT":
        cross_summary = (
            f"Asian read-through contradicts the US {direction_text} read. "
            "Reduce qualified stock size to 60% until alignment returns."
        )
    else:
        cross_summary = "Cross-market history is not reliable enough to alter size. Use the US and stock-specific thesis."

    regime = _text(us.get("regime")).upper() or "UNKNOWN"
    region_clause = "; ".join(
        f"{row['region']} {row['status'].lower()} ({row['relationship'].lower()})"
        for row in correlations
    )
    if cross_state in {"ASIA_LEADS", "EARLY_SIGNAL"}:
        lead_label = "aligned Asia lead" if cross_state == "ASIA_LEADS" else "early Asia"
        headline = (
            f"World tape: {display_bias.lower()} bias. The US is unresolved; "
            f"Korea and Japan provide an {lead_label} signal."
        )
    elif regime == "RANGE" and us_direction in {"LONG", "SHORT"}:
        headline = (
            f"World tape: US range with a {direction_text} tilt, "
            f"{cross_state.lower()} by Korea and Japan."
        )
    else:
        headline = (
            f"World tape: {regime.replace('_', ' ').lower()} in the US, "
            f"{cross_state.lower()} by Korea and Japan."
        )
    conflict_clause = ""
    if breadth_conflicts:
        conflict_clause = (
            f" US breadth currently leans against the {direction_text} Asia warning, "
            "so this is a defensive tilt rather than an outright directional call."
        )
    commentary = (
        f"{_text(us.get('commentary'))}{conflict_clause} Cross-market check: {region_clause}. "
        "Correlation is a risk filter, not proof of causation or an entry signal."
    )
    policy = dict(us.get("execution_policy") or {})
    policy.update({
        "preferred_direction": tactical_bias,
        "allow_tactical_short": bool(
            tactical_bias == "SHORT"
            and cross_state in {"CONFIRMED", "SUPPORTED", "ASIA_LEADS"}
        ),
        "cross_market_state": cross_state,
        "cross_market_direction": tactical_bias,
        "cross_market_size_multiplier": size_multiplier,
        "cross_market_summary": cross_summary,
        "summary": f"{_text(policy.get('summary'))} {cross_summary}".strip(),
    })

    if tactical_bias == "SHORT" and cross_state in {"ASIA_LEADS", "EARLY_SIGNAL"}:
        execution = (
            f"Keep a defensive tilt and use fresh stock-level breakdown shorts only at "
            f"{size_multiplier:.0%} of normal qualified size. Use at most 1x leverage; "
            "do not fight a stock whose own structure remains bullish."
        )
    elif tactical_bias == "SHORT":
        execution = (
            f"Favor fresh stock-level breakdown shorts at {size_multiplier:.0%} of normal qualified size. "
            "Use at most 1x leverage and stay flat when the stock itself does not confirm."
        )
    elif tactical_bias == "LONG":
        execution = (
            f"Favor supported stock-level longs at {size_multiplier:.0%} of normal qualified size. "
            "Enter on a reclaim or held pullback, not on an extended candle."
        )
    else:
        execution = "No broad directional edge. Trade only the clearest asset-specific thesis and keep exposure selective."

    us_view = {
        "region": "US",
        "role": "PRIMARY ANCHOR",
        "direction": us_direction,
        "status": "ANCHOR",
        "correlation": None,
        "relationship": "PRIMARY",
        "sample_size": 0,
        "sample_quality": "PRIMARY",
        "summary": _text(us.get("headline")),
    }
    for row in correlations:
        row["role"] = "ASIA LEAD"
    result = dict(us)
    result.update({
        "updated_at": now_dt.isoformat(),
        "headline": headline,
        "commentary": commentary,
        "confidence": confidence,
        "base_us_confidence": round(base_confidence, 4),
        "market_views": [us_view, *correlations],
        "correlations": correlations,
        "cross_market": {
            "state": cross_state,
            "confirmations": confirmations,
            "divergences": divergences,
            "leads": len(leading_rows),
            "confidence_shift": confidence_shift,
            "size_multiplier": size_multiplier,
            "us_direction": us_direction,
            "tactical_bias": tactical_bias,
            "display_bias": display_bias,
            "breadth_conflicts": breadth_conflicts,
            "summary": cross_summary,
            "execution": execution,
        },
        "asia_session": asia,
        "semiconductor_readthrough": _text(asia.get("us_readthrough")),
        "execution_policy": policy,
    })
    return result
