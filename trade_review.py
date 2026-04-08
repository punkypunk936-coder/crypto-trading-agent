"""
trade_review.py — operator review labels for closed trades.

These reviews are intentionally distinct from automated RL memory. They capture
human judgment about whether a trade had a good thesis, poor execution, or was
simply a bad trade idea. The agent can then use those labels as an additional
guardrail on future entries.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List

from logger import get_logger
from paths import TRADE_REVIEWS_JSON

log = get_logger("trade_review")

ALLOWED_VERDICTS = {
    "GOOD_TRADE",
    "BAD_THESIS",
    "BAD_EXECUTION",
    "BOTH_BAD",
    "MISS",
}
ALLOWED_QUALITY = {"STRONG", "MIXED", "WEAK", "GOOD", "OK", "POOR", ""}


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def default_reviews() -> dict:
    return {
        "updated_at": _now_str(),
        "reviews": {},
    }


def _normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",")]
    tags = []
    for item in value or []:
        text = str(item or "").strip()
        if text:
            tags.append(text)
    return list(dict.fromkeys(tags))


def _normalize_review(review: Any) -> dict | None:
    if not isinstance(review, dict):
        return None
    trade_id = str(review.get("trade_id") or "").strip()
    if not trade_id:
        return None
    verdict = str(review.get("verdict") or "GOOD_TRADE").upper()
    if verdict not in ALLOWED_VERDICTS:
        verdict = "GOOD_TRADE"
    thesis_quality = str(review.get("thesis_quality") or "").upper()
    execution_quality = str(review.get("execution_quality") or "").upper()
    if thesis_quality and thesis_quality not in ALLOWED_QUALITY:
        thesis_quality = ""
    if execution_quality and execution_quality not in ALLOWED_QUALITY:
        execution_quality = ""
    return {
        "trade_id": trade_id,
        "coin": str(review.get("coin") or "").upper(),
        "direction": str(review.get("direction") or "").upper(),
        "verdict": verdict,
        "thesis_quality": thesis_quality,
        "execution_quality": execution_quality,
        "tags": _normalize_tags(review.get("tags")),
        "notes": str(review.get("notes") or ""),
        "reviewed_at": str(review.get("reviewed_at") or _now_str()),
        "reviewer": str(review.get("reviewer") or "operator"),
    }


def normalize_reviews(payload: Any) -> dict:
    base = default_reviews()
    if not isinstance(payload, dict):
        return base
    raw_reviews = payload.get("reviews") if isinstance(payload.get("reviews"), dict) else {}
    normalized = {}
    for trade_id, review in raw_reviews.items():
        item = _normalize_review({"trade_id": trade_id, **(review if isinstance(review, dict) else {})})
        if item:
            normalized[item["trade_id"]] = item
    base["reviews"] = normalized
    base["updated_at"] = str(payload.get("updated_at") or base["updated_at"])
    return base


def load_reviews() -> dict:
    if not TRADE_REVIEWS_JSON.exists():
        return default_reviews()
    try:
        return normalize_reviews(json.loads(TRADE_REVIEWS_JSON.read_text()))
    except Exception as exc:
        log.warning(f"Failed to load trade reviews: {exc}")
        return default_reviews()


def save_reviews(payload: dict) -> dict:
    normalized = normalize_reviews(payload)
    normalized["updated_at"] = _now_str()
    TRADE_REVIEWS_JSON.write_text(json.dumps(normalized, indent=2))
    return normalized


def upsert_review(payload: dict) -> dict:
    reviews = load_reviews()
    item = _normalize_review(payload)
    if not item:
        return reviews
    current = dict((reviews.get("reviews") or {}).get(item["trade_id"], {}))
    current.update(item)
    reviews.setdefault("reviews", {})[item["trade_id"]] = current
    return save_reviews(reviews)


def get_review(trade_id: Any) -> dict | None:
    reviews = load_reviews().get("reviews") or {}
    return dict(reviews.get(str(trade_id), {}) or {}) or None


def merge_reviews_into_trades(trades: List[dict]) -> List[dict]:
    reviews = load_reviews().get("reviews") or {}
    merged = []
    for trade in trades or []:
        item = dict(trade or {})
        review = reviews.get(str(item.get("trade_id") or ""))
        if review:
            item["review"] = dict(review)
        merged.append(item)
    return merged


def review_summary(trades: List[dict] | None = None) -> dict:
    reviews = list((load_reviews().get("reviews") or {}).values())
    verdicts = Counter(str(review.get("verdict") or "") for review in reviews)
    thesis = Counter(str(review.get("thesis_quality") or "") for review in reviews if review.get("thesis_quality"))
    execution = Counter(str(review.get("execution_quality") or "") for review in reviews if review.get("execution_quality"))
    coverage = 0
    if trades:
        trade_ids = [str((trade or {}).get("trade_id") or "") for trade in trades if (trade or {}).get("trade_id") is not None]
        reviewed = sum(1 for trade_id in trade_ids if trade_id and str(trade_id) in (load_reviews().get("reviews") or {}))
        coverage = round(reviewed / len(trade_ids) * 100, 1) if trade_ids else 0
    return {
        "count": len(reviews),
        "coverage_pct": coverage,
        "verdicts": dict(verdicts),
        "thesis_quality": dict(thesis),
        "execution_quality": dict(execution),
        "updated_at": load_reviews().get("updated_at"),
    }


def get_directional_feedback(coin: str, direction: str) -> dict:
    reviews = list((load_reviews().get("reviews") or {}).values())
    relevant = [
        review for review in reviews
        if str(review.get("coin") or "").upper() == coin.upper()
        and str(review.get("direction") or "").upper() == direction.upper()
    ]
    relevant.sort(key=lambda review: str(review.get("reviewed_at") or ""), reverse=True)
    recent = relevant[:5]
    if not recent:
        return {"score_adjustment": 0.0, "hard_block": False, "reason": "", "reasons": []}

    bad_thesis = sum(1 for review in recent if review.get("verdict") in {"BAD_THESIS", "BOTH_BAD"})
    poor_execution = sum(1 for review in recent if review.get("verdict") in {"BAD_EXECUTION", "BOTH_BAD"})
    weak_thesis = sum(1 for review in recent if review.get("thesis_quality") == "WEAK")
    strong_thesis = sum(1 for review in recent if review.get("thesis_quality") == "STRONG" and review.get("verdict") == "GOOD_TRADE")
    reasons: List[str] = []
    score_adjustment = 0.0
    hard_block = False
    reason = ""

    if bad_thesis >= 3 or weak_thesis >= 3:
        hard_block = True
        reason = f"operator reviews keep flagging weak {coin.upper()} {direction.upper()} thesis"
        reasons.append(reason)
    else:
        if bad_thesis >= 2:
            score_adjustment -= 6.0
            reasons.append("recent operator reviews marked the thesis as bad")
        elif weak_thesis >= 2:
            score_adjustment -= 4.0
            reasons.append("operator reviews keep rating the thesis as weak")
        if poor_execution >= 2:
            score_adjustment -= 2.0
            reasons.append("execution quality has been poor recently")
        if strong_thesis >= 2 and bad_thesis == 0:
            score_adjustment += 2.0
            reasons.append("operator reviews confirm this setup family")
    return {
        "score_adjustment": round(score_adjustment, 2),
        "hard_block": hard_block,
        "reason": reason,
        "reasons": reasons[:3],
    }
