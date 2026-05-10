"""
performance_intelligence.py - cumulative dry-run edge summaries.

This gives the agent and dashboard a compact answer to:
"What kinds of trades are making money, and what should we stop repeating?"
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


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


def build_performance_edges(trades: Iterable[dict] | None, *, min_samples: int = 3) -> dict:
    rows = [
        dict(row or {})
        for row in list(trades or [])
        if _safe_str((row or {}).get("closed_at") or (row or {}).get("exit_time") or (row or {}).get("trade_id"))
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
        buckets[f"direction:{direction}"].append(row)
        buckets[f"score:{score_key}"].append(row)
        buckets[f"duration:{dur_key}"].append(row)
        buckets[f"exit:{exit_reason}"].append(row)
        if coin and direction:
            buckets[f"coin_direction:{coin}:{direction}"].append(row)

    metrics: list[dict] = []
    for key, grouped in buckets.items():
        kind, _, raw = key.partition(":")
        if kind == "score":
            label = score_bucket_label(raw)
        elif kind == "duration":
            label = duration_bucket_label(raw)
        elif kind == "coin_direction":
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

    return {
        "enabled": True,
        "summary": {
            **overall,
            "total_closed": total,
            "min_samples": min_samples,
            "working_edge_count": len(working),
            "failing_edge_count": len(failing),
            "top_lesson": _lesson(working[0], True) if working else (_lesson(failing[0], False) if failing else "Not enough closed trades to isolate an edge yet."),
        },
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
    min_samples: int = 4,
    min_win_rate: float = 0.52,
) -> dict:
    direction = str(direction or "").upper()
    coin = str(coin or "").upper()
    score_key = score_bucket(direction, score)
    rows = [dict(row or {}) for row in list(trades or [])]
    candidates = [
        ("coin_direction", f"{coin} {direction}", [
            row for row in rows
            if _safe_str(row.get("coin")).upper() == coin and _direction(row) == direction
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
                **metric,
            }
    return {"active": False, "permitted": True, "summary": "No toxic performance bucket found."}
