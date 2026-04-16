"""
decision_review_lab.py — label whether blocked/flat decisions were good passes.

This complements trade logs by reviewing the opportunities the agent skipped.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import precision_lab


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _load_review_candidates(data_dir: Path) -> list[dict]:
    dataset_path = data_dir / "decision_dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"decision dataset not found at {dataset_path}")

    rows: list[dict] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue

            snap = dict(record.get("signal_snapshot") or {})
            candidate = _safe_str(
                record.get("candidate_action")
                or snap.get("thesis_candidate_action")
                or snap.get("action")
                or "FLAT"
            ).upper()
            if candidate not in {"LONG", "SHORT"}:
                continue

            risk_pct = _safe_float(snap.get("planned_risk_pct")) / 100.0
            if risk_pct <= 0:
                continue

            stage = _safe_str(record.get("stage")).lower()
            executed = bool(record.get("executed", False)) or stage in {"market_entry_opened", "limit_entry_opened"}
            pending_only = bool(record.get("pending_limit", False)) and not executed
            if pending_only:
                continue

            rows.append({
                "decision_id": record.get("decision_id"),
                "coin": _safe_str(record.get("coin")).upper(),
                "action": candidate,
                "stage": stage,
                "blocked": bool(record.get("blocked", False)),
                "executed": executed,
                "ts": _safe_float(record.get("recorded_at_ts")),
                "risk_pct": risk_pct,
                "reward_pct": _safe_float(snap.get("planned_reward_pct")) / 100.0,
                "rr": _safe_float(snap.get("planned_risk_reward_ratio")),
                "prob": _safe_float(snap.get("expectancy_probability"), 0.50),
                "unc": _safe_float(snap.get("expectancy_uncertainty"), 0.50),
                "score": _safe_float(snap.get("expectancy_score"), 50.0),
                "confidence": _safe_str(snap.get("confidence"), "LOW").upper(),
                "thesis_quality": _safe_str(snap.get("thesis_quality"), "LOW").upper(),
                "breakout": _safe_str(snap.get("orderbook_breakout_state"), "NONE").lower(),
                "interaction": _safe_str(snap.get("orderbook_interaction"), "between_levels").lower(),
                "regime": _safe_str(snap.get("dominant_regime"), "mixed").lower(),
                "instrument_type": _safe_str(snap.get("instrument_type"), "crypto").lower(),
            })

    rows.sort(key=lambda item: item["ts"])
    return rows


def _collapse(rows: list[dict], *, dedupe_minutes: int) -> list[dict]:
    episodes: list[dict] = []
    max_gap = dedupe_minutes * 60
    for row in rows:
        if not episodes:
            episodes.append(dict(row))
            continue
        last = episodes[-1]
        same_family = (
            row["coin"] == last["coin"]
            and row["action"] == last["action"]
            and row["stage"] == last["stage"]
            and row["breakout"] == last["breakout"]
            and row["interaction"] == last["interaction"]
            and row["regime"] == last["regime"]
        )
        if same_family and (row["ts"] - last["ts"]) <= max_gap:
            continue
        episodes.append(dict(row))
    return episodes


def _classify(row: dict) -> str:
    if bool(row.get("executed", False)):
        return "GOOD_TRADE" if int(row.get("outcome", 0)) else "BAD_TRADE"
    return "MISSED_WIN" if int(row.get("outcome", 0)) else "CORRECT_PASS"


def build_report(
    *,
    data_dir: Path,
    target_r: float,
    horizon_minutes: int,
    interval: str,
    dedupe_minutes: int,
) -> dict:
    rows = _load_review_candidates(data_dir)
    episodes = _collapse(rows, dedupe_minutes=dedupe_minutes)
    labeled = [
        labeled_row
        for labeled_row in (
            precision_lab._label_episode(
                row,
                interval=interval,
                horizon_minutes=horizon_minutes,
                target_r=target_r,
            )
            for row in episodes
        )
        if labeled_row is not None
    ]
    for row in labeled:
        row["classification"] = _classify(row)

    counts = Counter(row["classification"] for row in labeled)
    missed_by_family: dict[str, int] = defaultdict(int)
    for row in labeled:
        if row["classification"] == "MISSED_WIN":
            missed_by_family[f"{row['coin']}:{row['action']}"] += 1

    report = {
        "data_dir": str(data_dir),
        "generated_at": int(time.time()),
        "target_r": target_r,
        "horizon_minutes": horizon_minutes,
        "interval": interval,
        "decision_rows": len(rows),
        "episodes": len(episodes),
        "labeled_episodes": len(labeled),
        "classifications": dict(counts),
        "missed_families": [
            {"family": family, "misses": misses}
            for family, misses in sorted(missed_by_family.items(), key=lambda item: (-item[1], item[0]))[:8]
        ],
    }
    return report


def build_and_save_report(
    *,
    data_dir: Path,
    target_r: float,
    horizon_minutes: int,
    interval: str,
    dedupe_minutes: int,
) -> dict:
    report = build_report(
        data_dir=data_dir,
        target_r=target_r,
        horizon_minutes=horizon_minutes,
        interval=interval,
        dedupe_minutes=dedupe_minutes,
    )
    (data_dir / "decision_review_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
