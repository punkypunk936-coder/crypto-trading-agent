"""
data_reliability.py — score whether the current market context is trustworthy.

The agent should trade only when its inputs are fresh enough and coherent
enough to deserve confidence. Reliability is treated separately from edge.
"""

from __future__ import annotations

from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def assess_reliability(trading_cfg, signal_snapshot: Mapping[str, Any] | None) -> dict:
    snap = dict(signal_snapshot or {})
    issues: list[str] = []
    blockers: list[str] = []
    score = 100.0

    tradable = _safe_str(snap.get("execution_mode"), "observation_only") == "tradable"
    instrument_type = _safe_str(snap.get("instrument_type"), "crypto").lower()
    action = _safe_str(snap.get("action"), "FLAT").upper()

    if not bool(snap.get("using_closed_candles", False)):
        score -= 8.0
        issues.append("conviction is leaning on a still-forming candle")

    analysis_price = _safe_float(snap.get("analysis_price"))
    live_price = _safe_float(snap.get("live_price") or snap.get("price"))
    price_gap_pct = 0.0
    if analysis_price > 0 and live_price > 0:
        price_gap_pct = abs(live_price - analysis_price) / analysis_price * 100.0
    max_gap_pct = float(getattr(trading_cfg, "data_reliability_max_live_analysis_gap_pct", 0.90) or 0.90)
    if price_gap_pct > max_gap_pct:
        score -= 18.0
        issues.append(
            f"live price drifted {price_gap_pct:.2f}% away from the analyzed close"
        )
        if tradable and action in {"LONG", "SHORT"}:
            blockers.append("price moved too far since the conviction snapshot")

    reference_deviation_pct = _safe_float(snap.get("price_deviation_pct"))
    max_reference_deviation_pct = float(
        getattr(trading_cfg, "data_reliability_max_reference_deviation_pct", 2.0) or 2.0
    )
    if abs(reference_deviation_pct) > max_reference_deviation_pct:
        score -= 14.0
        issues.append(
            f"venue price is {reference_deviation_pct:+.2f}% away from the reference quote"
        )

    if getattr(trading_cfg, "use_daily_market_map", True) and not bool(snap.get("market_map_available", False)):
        score -= 10.0
        issues.append("daily market map is missing")

    if getattr(trading_cfg, "use_news", False):
        articles = _safe_int(snap.get("news_articles"))
        min_articles = int(getattr(trading_cfg, "data_reliability_min_news_articles", 1) or 1)
        if articles < min_articles:
            score -= 12.0
            issues.append("news coverage is too thin to trust the narrative read")

    needs_orderbook = (
        tradable
        and getattr(trading_cfg, "use_orderbook_levels", True)
        and instrument_type in {"crypto", "index", "equity"}
    )
    if needs_orderbook:
        if not bool(snap.get("orderbook_valid", False)):
            score -= 35.0
            blockers.append("no valid orderbook snapshot is available")
        else:
            feed_age = _safe_float(snap.get("orderbook_feed_age_seconds"))
            max_age = float(
                getattr(
                    trading_cfg,
                    "orderbook_feed_max_snapshot_age_seconds",
                    45.0,
                ) or 45.0
            )
            if feed_age > max_age:
                score -= 20.0
                blockers.append(f"orderbook feed is stale ({feed_age:.0f}s old)")

            snapshots = _safe_int(snap.get("orderbook_feed_snapshot_count"))
            min_snapshots = int(getattr(trading_cfg, "data_reliability_min_orderbook_snapshots", 3) or 3)
            if snapshots < min_snapshots:
                score -= 10.0
                issues.append(
                    f"microstructure history is still thin ({snapshots}/{min_snapshots} snapshots)"
                )

    score = max(0.0, min(100.0, score))
    min_score = float(getattr(trading_cfg, "data_reliability_min_score", 58.0) or 58.0)
    permitted = not blockers and score >= min_score
    summary = (
        blockers[0]
        if blockers
        else (issues[0] if issues else "data quality is strong enough to trust the setup")
    )
    quality = "HIGH" if score >= 80 else "MEDIUM" if score >= min_score else "LOW"
    return {
        "permitted": permitted,
        "score": round(score, 2),
        "quality": quality,
        "summary": summary,
        "issues": issues[:4],
        "blockers": blockers[:4],
        "price_gap_pct": round(price_gap_pct, 4),
        "reference_deviation_pct": round(reference_deviation_pct, 4),
    }
