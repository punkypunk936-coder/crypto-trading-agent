"""Asia-session context and US read-through for the operator dashboard."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any


BENCHMARKS = (
    ("KR200", "KOSPI 200", "Korea"),
    ("EWY", "South Korea ETF", "Korea"),
    ("JP225", "Nikkei 225", "Japan"),
    ("EWJ", "Japan ETF", "Japan"),
)
ASIA_SEMIS = ("SKHX", "SMSN", "KIOXIA", "TSM", "DRAM")
US_SEMI_READTHROUGH = ("MU", "SNDK", "AMD", "NVDA", "MRVL")


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _short(value: Any, limit: int = 180) -> str:
    text = _text(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _fresh(signal: dict, *, now_ts: float, max_age_hours: float = 18.0) -> bool:
    updated = _number(signal.get("analysis_updated_ts"))
    return bool(updated and now_ts - updated <= max_age_hours * 3600.0)


def _direction(signal: dict) -> str:
    action = _text(signal.get("action")).upper()
    if action in {"LONG", "SHORT"}:
        return action
    bias = _text(signal.get("market_map_bias")).upper()
    if bias == "BULLISH":
        return "LONG"
    if bias == "BEARISH":
        return "SHORT"
    score = _number(signal.get("score"), 50.0)
    return "LONG" if score >= 57.0 else "SHORT" if score <= 43.0 else "FLAT"


def _benchmark_row(symbol: str, label: str, region: str, signal: dict, *, now_ts: float) -> dict:
    direction = _direction(signal)
    move = _number(signal.get("recent_move_pct") or signal.get("move_pct_24h"))
    price = _number(signal.get("live_price") or signal.get("price"))
    map_summary = _text(signal.get("market_map_summary") or signal.get("price_action_summary"))
    event_summary = _text(
        signal.get("news_event_summary")
        or signal.get("news_catalyst_summary")
        or signal.get("news_headline")
    )
    bias = "bullish" if direction == "LONG" else "bearish" if direction == "SHORT" else "mixed"
    why_parts = []
    if move:
        why_parts.append(f"{move:+.1f}% over the latest 24h window")
    if map_summary:
        why_parts.append(map_summary)
    if event_summary:
        why_parts.append(event_summary)
    return {
        "symbol": symbol,
        "label": label,
        "region": region,
        "direction": direction,
        "bias": bias,
        "move_pct": round(move, 2),
        "price": round(price, 4),
        "fresh": _fresh(signal, now_ts=now_ts),
        "why": _short("; ".join(why_parts) or "Waiting for a fresh regional market read.", 220),
        "updated_at": _text(signal.get("analysis_updated_at")),
    }


def build_asia_session(state: dict | None, *, now: datetime | None = None) -> dict:
    safe_state = dict(state or {})
    signals = dict(safe_state.get("signals") or {})
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    now_ts = now_dt.timestamp()

    rows = [
        _benchmark_row(symbol, label, region, dict(signals.get(symbol) or {}), now_ts=now_ts)
        for symbol, label, region in BENCHMARKS
        if signals.get(symbol)
    ]
    fresh_rows = [row for row in rows if row.get("fresh")]
    directional = [row for row in fresh_rows if row.get("direction") in {"LONG", "SHORT"}]
    bullish_count = sum(row.get("direction") == "LONG" for row in directional)
    bearish_count = sum(row.get("direction") == "SHORT" for row in directional)

    semi_rows = []
    for symbol in ASIA_SEMIS:
        signal = dict(signals.get(symbol) or {})
        if not signal or not _fresh(signal, now_ts=now_ts):
            continue
        semi_rows.append({
            "symbol": symbol,
            "direction": _direction(signal),
            "move_pct": round(_number(signal.get("recent_move_pct") or signal.get("move_pct_24h")), 2),
            "why": _short(signal.get("news_catalyst_summary") or signal.get("market_map_summary"), 140),
        })
    semi_bulls = [row for row in semi_rows if row.get("direction") == "LONG"]
    semi_bears = [row for row in semi_rows if row.get("direction") == "SHORT"]

    if bullish_count > bearish_count:
        regional_bias = "RISK_ON"
        headline = f"Asia is constructive: {bullish_count}/{max(1, len(directional))} fresh benchmark reads are bullish."
    elif bearish_count > bullish_count:
        regional_bias = "RISK_OFF"
        headline = f"Asia is defensive: {bearish_count}/{max(1, len(directional))} fresh benchmark reads are bearish."
    else:
        regional_bias = "MIXED"
        headline = "Asia is mixed; Korea, Japan, and the regional semiconductor tape are not confirming one clean risk signal."

    if semi_bulls and bullish_count >= bearish_count:
        leaders = ", ".join(row["symbol"] for row in semi_bulls[:3])
        us_readthrough = (
            f"Regional semiconductor leadership is positive ({leaders}); that supports {', '.join(US_SEMI_READTHROUGH)} "
            "into the US session, provided the US names hold their own mapped support."
        )
        invalidation = "Read-through fails if Korea/Japan reverse lower and the Asian memory leaders lose support together."
    elif semi_bears and bearish_count >= bullish_count:
        leaders = ", ".join(row["symbol"] for row in semi_bears[:3])
        us_readthrough = (
            f"Regional semiconductor leadership is weak ({leaders}); treat strength in {', '.join(US_SEMI_READTHROUGH)} "
            "as less trustworthy until Asia stabilizes."
        )
        invalidation = "Bearish read-through fails if the regional benchmarks reclaim and Asian semiconductor breadth turns positive."
    else:
        us_readthrough = (
            "No clean semiconductor handoff yet. Use KOSPI/Nikkei direction as context, but require AMD, NVDA, MU, SNDK, and MRVL "
            "to confirm independently in the US session."
        )
        invalidation = "The mixed read resolves only when regional benchmarks and Asian semiconductor breadth align."

    return {
        "enabled": True,
        "active": bool(fresh_rows),
        "updated_at": now_dt.isoformat(),
        "regional_bias": regional_bias,
        "headline": headline,
        "us_readthrough": us_readthrough,
        "invalidation": invalidation,
        "benchmarks": rows,
        "semiconductor_leaders": semi_rows,
        "fresh_count": len(fresh_rows),
    }
