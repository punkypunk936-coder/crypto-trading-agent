"""
performance_intelligence.py - cumulative dry-run edge summaries.

This gives the agent and dashboard a compact answer to:
"What kinds of trades are making money, and what should we stop repeating?"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value not in (None, "") else default)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _direction(row: dict) -> str:
    return _safe_str(row.get("direction") or row.get("side"), "UNKNOWN").upper()


def _score(row: dict) -> float:
    return _safe_float(row.get("signal_score") or row.get("score") or row.get("entry_score"), 50.0)


def _entry_context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    context = row.get("entry_context", {}) if isinstance(row, Mapping) else {}
    return context if isinstance(context, Mapping) else {}


def _context_value(row: Mapping[str, Any], key: str, default: Any = "") -> Any:
    if not isinstance(row, Mapping):
        return default
    context = _entry_context(row)
    value = row.get(key)
    if value not in (None, ""):
        return value
    return context.get(key, default)


def _instrument_type(row: Mapping[str, Any]) -> str:
    return _safe_str(_context_value(row, "instrument_type", "crypto"), "crypto").lower()


def _asset_class(row: Mapping[str, Any]) -> str:
    coin = _safe_str(row.get("coin"), "UNKNOWN").upper()
    instrument = _instrument_type(row)
    if coin in {"XAU", "GOLD", "BRENT", "WTI", "CL"}:
        return "commodity"
    if instrument in {"crypto", "equity", "index", "commodity"}:
        return instrument
    return "other"


def _portfolio_theme(row: Mapping[str, Any]) -> str:
    return _safe_str(_context_value(row, "portfolio_theme", ""), "").upper()


def _market_regime(row: Mapping[str, Any]) -> str:
    return _safe_str(_context_value(row, "market_regime", row.get("market_regime", "RANGING")), "RANGING").upper()


def _dominant_regime(row: Mapping[str, Any]) -> str:
    return _safe_str(_context_value(row, "dominant_regime", row.get("dominant_regime", "MIXED")), "MIXED").upper()


def _structure_trend(row: Mapping[str, Any]) -> str:
    return _safe_str(_context_value(row, "structure_trend", "UNKNOWN"), "UNKNOWN").upper()


def _market_map_bias(row: Mapping[str, Any]) -> str:
    return _safe_str(_context_value(row, "market_map_bias", "NEUTRAL"), "NEUTRAL").upper()


def _crypto_market_mode(row: Mapping[str, Any]) -> str:
    return _safe_str(_context_value(row, "crypto_market_mode", _context_value(row, "market_mode", "UNKNOWN")), "UNKNOWN").upper()


def _crypto_directional_bias(row: Mapping[str, Any]) -> str:
    return _safe_str(_context_value(row, "crypto_directional_bias", _context_value(row, "directional_bias", "NEUTRAL")), "NEUTRAL").upper()


def score_bucket(direction: str, score: float) -> str:
    direction = str(direction or "").upper()
    score = _safe_float(score, 50.0)
    if direction == "LONG":
        if score >= 75.0:
            return "LONG_75_PLUS"
        if score >= 70.0:
            return "LONG_70_75"
        if score >= 65.0:
            return "LONG_65_70"
        return "LONG_UNDER_65"
    if direction == "SHORT":
        if score <= 25.0:
            return "SHORT_LE_25"
        if score <= 30.0:
            return "SHORT_25_30"
        if score <= 35.0:
            return "SHORT_30_35"
        return "SHORT_OVER_35"
    return "NO_DIRECTION"


def score_bucket_label(bucket: str) -> str:
    labels = {
        "LONG_75_PLUS": "LONG score 75+",
        "LONG_70_75": "LONG score 70-75",
        "LONG_65_70": "LONG score 65-70",
        "LONG_UNDER_65": "LONG score under 65",
        "SHORT_LE_25": "SHORT score <=25",
        "SHORT_25_30": "SHORT score 25-30",
        "SHORT_30_35": "SHORT score 30-35",
        "SHORT_OVER_35": "SHORT score over 35",
    }
    return labels.get(str(bucket or ""), str(bucket or "unknown").replace("_", " ").title())


def duration_bucket(minutes: float) -> str:
    minutes = _safe_float(minutes)
    if minutes < 60:
        return "UNDER_1H"
    if minutes < 360:
        return "1H_6H"
    if minutes < 1440:
        return "6H_1D"
    if minutes < 4320:
        return "1D_3D"
    return "3D_PLUS"


def duration_bucket_label(bucket: str) -> str:
    labels = {
        "UNDER_1H": "held under 1h",
        "1H_6H": "held 1-6h",
        "6H_1D": "held 6h-1d",
        "1D_3D": "held 1-3d",
        "3D_PLUS": "held 3d+",
    }
    return labels.get(str(bucket or ""), str(bucket or "unknown").replace("_", " ").lower())


def _duration(row: dict) -> float:
    return _safe_float(row.get("duration_mins") or row.get("hold_minutes") or row.get("duration_minutes"))


def _pnl(row: dict) -> float:
    return _safe_float(row.get("pnl_usd"))


def _pnl_pct(row: dict) -> float:
    return _safe_float(row.get("pnl_pct"))


def _result(row: dict) -> str:
    explicit = _safe_str(row.get("result") or row.get("outcome")).upper()
    if explicit in {"WIN", "LOSS", "FLAT"}:
        return explicit
    pnl = _pnl(row)
    if pnl > 0:
        return "WIN"
    if pnl < 0:
        return "LOSS"
    return "FLAT"


def _metric(label: str, rows: list[dict]) -> dict:
    total = len(rows)
    wins = sum(1 for row in rows if _result(row) == "WIN")
    losses = sum(1 for row in rows if _result(row) == "LOSS")
    pnl = sum(_pnl(row) for row in rows)
    durations = [_duration(row) for row in rows if _duration(row) > 0]
    win_rate = wins / total if total else 0.0
    return {
        "label": label,
        "samples": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "pnl_usd": round(pnl, 2),
        "avg_pnl_usd": round(pnl / total, 2) if total else 0.0,
        "avg_duration_minutes": round(sum(durations) / len(durations), 1) if durations else 0.0,
    }


def _lesson(edge: dict, positive: bool) -> str:
    label = _safe_str(edge.get("label"), "this bucket")
    wr = _safe_float(edge.get("win_rate")) * 100.0
    pnl = _safe_float(edge.get("pnl_usd"))
    verb = "Lean into" if positive else "Avoid"
    return f"{verb} {label}: {wr:.0f}% WR, ${pnl:+.2f}."


def _trade_title(row: Mapping[str, Any]) -> str:
    coin = _safe_str(row.get("coin"), "UNKNOWN").upper()
    direction = _direction(dict(row))
    pnl_pct = _pnl_pct(dict(row))
    pnl_usd = _pnl(dict(row))
    score = _score(dict(row))
    return f"{coin} {direction} score {score:.0f}: {pnl_pct:+.2f}% / ${pnl_usd:+.2f}"


def _trade_card(row: Mapping[str, Any] | None, *, similarity: float | None = None) -> dict:
    if not isinstance(row, Mapping) or not row:
        return {}
    context = _entry_context(row)
    thesis = context.get("thesis", {}) if isinstance(context.get("thesis", {}), Mapping) else {}
    card = {
        "trade_id": row.get("trade_id", ""),
        "coin": _safe_str(row.get("coin"), "UNKNOWN").upper(),
        "direction": _direction(dict(row)),
        "instrument_type": _instrument_type(row),
        "asset_class": _asset_class(row),
        "portfolio_theme": _portfolio_theme(row),
        "score": round(_score(dict(row)), 2),
        "score_bucket": score_bucket(_direction(dict(row)), _score(dict(row))),
        "pnl_usd": round(_pnl(dict(row)), 2),
        "pnl_pct": round(_pnl_pct(dict(row)), 4),
        "result": _result(dict(row)),
        "duration_minutes": round(_duration(dict(row)), 2),
        "duration_bucket": duration_bucket(_duration(dict(row))),
        "exit_reason": _safe_str(row.get("exit_reason"), "unknown"),
        "market_regime": _market_regime(row),
        "dominant_regime": _dominant_regime(row),
        "structure_trend": _structure_trend(row),
        "market_map_bias": _market_map_bias(row),
        "thesis_quality": _safe_str(thesis.get("quality") or context.get("thesis_quality"), "UNKNOWN"),
        "summary": _trade_title(row),
    }
    closed_at = row.get("closed_at") or row.get("exit_time") or row.get("closed_at_ts") or row.get("recorded_at_ts")
    if closed_at not in (None, ""):
        card["closed_at"] = closed_at
    if similarity is not None:
        card["similarity"] = round(max(0.0, min(1.0, similarity)), 4)
    return card


def _best_trade(rows: list[dict]) -> dict:
    if not rows:
        return {}
    row = max(rows, key=lambda item: (_pnl(item), _pnl_pct(item), _duration(item)))
    return _trade_card(row)


def _worst_trade(rows: list[dict]) -> dict:
    if not rows:
        return {}
    row = min(rows, key=lambda item: (_pnl(item), _pnl_pct(item), -_duration(item)))
    return _trade_card(row)


def _snapshot_target(
    *,
    coin: str,
    direction: str,
    score: float,
    signal_snapshot: Mapping[str, Any] | None = None,
) -> dict:
    snapshot = dict(signal_snapshot or {})
    target_row = {
        "coin": coin,
        "direction": direction,
        "signal_score": score,
        "entry_context": snapshot,
    }
    return {
        "coin": _safe_str(coin, "UNKNOWN").upper(),
        "direction": _safe_str(direction, "UNKNOWN").upper(),
        "instrument_type": _instrument_type(target_row),
        "asset_class": _asset_class(target_row),
        "portfolio_theme": _portfolio_theme(target_row),
        "score": _safe_float(score, 50.0),
        "score_bucket": score_bucket(direction, score),
        "market_regime": _market_regime(target_row),
        "dominant_regime": _dominant_regime(target_row),
        "structure_trend": _structure_trend(target_row),
        "market_map_bias": _market_map_bias(target_row),
        "crypto_market_mode": _crypto_market_mode(target_row),
        "crypto_directional_bias": _crypto_directional_bias(target_row),
    }


def _similarity(row: Mapping[str, Any], target: Mapping[str, Any]) -> float:
    if not isinstance(row, Mapping):
        return 0.0
    row_direction = _direction(dict(row))
    row_score = _score(dict(row))
    score_gap = abs(row_score - _safe_float(target.get("score"), 50.0))
    score_proximity = max(0.0, 1.0 - (score_gap / 35.0))
    checks = [
        (0.18, row_direction == target.get("direction")),
        (0.22, _safe_str(row.get("coin")).upper() == target.get("coin")),
        (0.18, _asset_class(row) == target.get("asset_class")),
        (0.10, bool(_portfolio_theme(row)) and _portfolio_theme(row) == target.get("portfolio_theme")),
        (0.14, score_bucket(row_direction, row_score) == target.get("score_bucket")),
        (0.08, _market_regime(row) == target.get("market_regime")),
        (0.04, _dominant_regime(row) == target.get("dominant_regime")),
        (0.03, _structure_trend(row) == target.get("structure_trend")),
        (0.02, _market_map_bias(row) == target.get("market_map_bias")),
        (0.03, _asset_class(row) != "crypto" or _crypto_market_mode(row) == target.get("crypto_market_mode")),
        (0.02, _asset_class(row) != "crypto" or _crypto_directional_bias(row) == target.get("crypto_directional_bias")),
        (0.03, score_proximity),
    ]
    total_weight = sum(weight for weight, _ in checks)
    score = 0.0
    for weight, matched in checks:
        score += weight * (float(matched) if not isinstance(matched, bool) else (1.0 if matched else 0.0))
    return score / max(total_weight, 1e-9)


def similar_trade_memory(
    trades: Iterable[dict] | None,
    *,
    coin: str,
    direction: str,
    score: float,
    signal_snapshot: Mapping[str, Any] | None = None,
    min_similarity: float = 0.58,
    limit: int = 60,
) -> dict:
    rows = [dict(row or {}) for row in list(trades or []) if isinstance(row, Mapping)]
    target = _snapshot_target(coin=coin, direction=direction, score=score, signal_snapshot=signal_snapshot)
    scored: list[tuple[float, dict]] = []
    for row in rows:
        sim = _similarity(row, target)
        if sim >= min_similarity:
            scored.append((sim, row))
    scored = sorted(
        scored,
        key=lambda item: (-item[0], -abs(_pnl_pct(item[1])), -abs(_pnl(item[1]))),
    )[:max(1, int(limit or 60))]
    matched = [row for _, row in scored]
    if not matched:
        return {
            "enabled": True,
            "active": False,
            "target": target,
            "min_similarity": round(min_similarity, 4),
            "matched_samples": 0,
            "summary": "No close historical trade analog yet.",
        }

    best_pair = max(scored, key=lambda item: (_pnl(item[1]), _pnl_pct(item[1]), item[0]))
    worst_pair = min(scored, key=lambda item: (_pnl(item[1]), _pnl_pct(item[1]), -item[0]))
    best = _trade_card(best_pair[1], similarity=best_pair[0])
    worst = _trade_card(worst_pair[1], similarity=worst_pair[0])
    return {
        "enabled": True,
        "active": True,
        "target": target,
        "min_similarity": round(min_similarity, 4),
        "matched_samples": len(matched),
        "best_match": best,
        "worst_match": worst,
        "top_matches": [_trade_card(row, similarity=sim) for sim, row in scored[:5]],
        "summary": f"Best comparable: {best.get('summary', 'n/a')}; worst comparable: {worst.get('summary', 'n/a')}.",
    }


def build_performance_edges(trades: Iterable[dict] | None, *, min_samples: int = 3) -> dict:
    rows = [
        dict(row or {})
        for row in list(trades or [])
        if _safe_str(
            (row or {}).get("closed_at")
            or (row or {}).get("exit_time")
            or (row or {}).get("closed_at_ts")
            or (row or {}).get("recorded_at_ts")
            or (row or {}).get("trade_id")
        )
    ]
    total = len(rows)
    overall = _metric("All closed trades", rows)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        direction = _direction(row)
        score_key = score_bucket(direction, _score(row))
        dur_key = duration_bucket(_duration(row))
        coin = _safe_str(row.get("coin"), "UNKNOWN").upper()
        exit_reason = _safe_str(row.get("exit_reason"), "unknown").lower()
        asset_class = _asset_class(row)
        instrument = _instrument_type(row)
        theme = _portfolio_theme(row)
        regime = _market_regime(row)
        dominant = _dominant_regime(row)
        structure = _structure_trend(row)
        crypto_mode = _crypto_market_mode(row)
        crypto_bias = _crypto_directional_bias(row)
        buckets[f"direction:{direction}"].append(row)
        buckets[f"score:{score_key}"].append(row)
        buckets[f"duration:{dur_key}"].append(row)
        buckets[f"exit:{exit_reason}"].append(row)
        buckets[f"asset_class_direction:{asset_class}:{direction}"].append(row)
        buckets[f"instrument_direction:{instrument}:{direction}"].append(row)
        buckets[f"regime_direction:{instrument}:{direction}:{regime}:{dominant}"].append(row)
        buckets[f"structure_direction:{instrument}:{direction}:{structure}"].append(row)
        if asset_class == "crypto":
            buckets[f"crypto_tape:{direction}:{crypto_mode}:{crypto_bias}"].append(row)
        if coin and direction:
            buckets[f"coin_direction:{coin}:{direction}"].append(row)
        if theme and direction:
            buckets[f"theme_direction:{theme}:{direction}"].append(row)

    metrics: list[dict] = []
    for key, grouped in buckets.items():
        kind, _, raw = key.partition(":")
        if kind == "score":
            label = score_bucket_label(raw)
        elif kind == "duration":
            label = duration_bucket_label(raw)
        elif kind == "coin_direction":
            label = raw.replace(":", " ")
        elif kind in {"asset_class_direction", "instrument_direction", "theme_direction", "regime_direction", "structure_direction", "crypto_tape"}:
            label = raw.replace(":", " ")
        elif kind == "exit":
            label = "exit " + raw.replace("_", " ")
        else:
            label = raw.replace("_", " ")
        metric = _metric(label, grouped)
        metric.update({"key": key, "kind": kind})
        metrics.append(metric)

    enough = [item for item in metrics if int(item.get("samples") or 0) >= min_samples]
    working = sorted(
        [item for item in enough if _safe_float(item.get("win_rate")) >= 0.58 and _safe_float(item.get("pnl_usd")) > 0],
        key=lambda item: (-_safe_float(item.get("win_rate")), -_safe_float(item.get("pnl_usd")), -int(item.get("samples") or 0)),
    )[:6]
    failing = sorted(
        [item for item in enough if _safe_float(item.get("win_rate")) <= 0.45 or _safe_float(item.get("pnl_usd")) < 0],
        key=lambda item: (_safe_float(item.get("win_rate")), _safe_float(item.get("pnl_usd")), -int(item.get("samples") or 0)),
    )[:6]

    asset_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        asset_groups[f"{_asset_class(row)}:{_direction(row)}"].append(row)
    asset_extremes = []
    for key, grouped in asset_groups.items():
        asset_extremes.append({
            "key": key,
            "best_trade": _best_trade(grouped),
            "worst_trade": _worst_trade(grouped),
            "samples": len(grouped),
        })

    best_trade = _best_trade(rows)
    worst_trade = _worst_trade(rows)
    return {
        "enabled": True,
        "summary": {
            **overall,
            "total_closed": total,
            "min_samples": min_samples,
            "working_edge_count": len(working),
            "failing_edge_count": len(failing),
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "top_lesson": _lesson(working[0], True) if working else (_lesson(failing[0], False) if failing else "Not enough closed trades to isolate an edge yet."),
        },
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "asset_class_extremes": sorted(asset_extremes, key=lambda item: str(item.get("key") or ""))[:20],
        "working_edges": working,
        "failing_edges": failing,
        "all_edges": sorted(metrics, key=lambda item: (-int(item.get("samples") or 0), str(item.get("key") or "")))[:40],
        "lessons": [_lesson(item, True) for item in working[:3]] + [_lesson(item, False) for item in failing[:3]],
    }


def setup_edge_verdict(
    trades: Iterable[dict] | None,
    *,
    coin: str,
    direction: str,
    score: float,
    signal_snapshot: Mapping[str, Any] | None = None,
    min_samples: int = 4,
    min_win_rate: float = 0.52,
) -> dict:
    direction = str(direction or "").upper()
    coin = str(coin or "").upper()
    score_key = score_bucket(direction, score)
    rows = [dict(row or {}) for row in list(trades or [])]
    memory = similar_trade_memory(
        rows,
        coin=coin,
        direction=direction,
        score=score,
        signal_snapshot=signal_snapshot,
    )
    target_asset = (memory.get("target") or {}).get("asset_class") or "unknown"
    target_regime = (memory.get("target") or {}).get("market_regime") or "RANGING"
    candidates = [
        ("coin_direction", f"{coin} {direction}", [
            row for row in rows
            if _safe_str(row.get("coin")).upper() == coin and _direction(row) == direction
        ]),
        ("asset_class_direction", f"{target_asset} {direction}", [
            row for row in rows
            if _asset_class(row) == target_asset and _direction(row) == direction
        ]),
        ("regime_direction", f"{target_asset} {direction} {target_regime}", [
            row for row in rows
            if _asset_class(row) == target_asset and _direction(row) == direction and _market_regime(row) == target_regime
        ]),
        ("score", score_bucket_label(score_key), [
            row for row in rows
            if _direction(row) == direction and score_bucket(direction, _score(row)) == score_key
        ]),
        ("direction", direction, [row for row in rows if _direction(row) == direction]),
    ]
    for kind, label, grouped in candidates:
        if len(grouped) < min_samples:
            continue
        metric = _metric(label, grouped)
        if _safe_float(metric.get("win_rate")) < min_win_rate or _safe_float(metric.get("pnl_usd")) < 0:
            return {
                "active": True,
                "permitted": False,
                "kind": kind,
                "label": label,
                "summary": _lesson(metric, False),
                "similar_trade_memory": memory,
                **metric,
            }
    return {
        "active": False,
        "permitted": True,
        "summary": "No toxic performance bucket found.",
        "similar_trade_memory": memory,
    }
