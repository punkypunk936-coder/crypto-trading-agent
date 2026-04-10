"""
decision_dataset.py — Structured per-cycle decision records for AI learning.

Unlike the closed-trade dataset, this captures every decision the agent makes:
trades it takes, trades it blocks, and moments where it correctly stays flat.
That gives later models the "what did we see?" context that pure trade logs miss.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable

from logger import get_logger
from paths import DECISION_DATASET_JSONL

log = get_logger("decision_dataset")


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def append_decision(record: dict) -> None:
    if not isinstance(record, dict) or not record:
        return

    payload = dict(record)
    payload.setdefault("recorded_at_ts", time.time())
    payload.setdefault(
        "decision_id",
        f"{payload.get('cycle_number', 0)}:{payload.get('coin', 'UNKNOWN')}:{payload.get('stage', 'decision')}:{int(payload['recorded_at_ts'] * 1000)}",
    )
    DECISION_DATASET_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_DATASET_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def load_decisions(limit: int | None = None) -> list[dict]:
    if not DECISION_DATASET_JSONL.exists():
        return []

    rows: list[dict] = []
    with DECISION_DATASET_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                log.debug(f"Skipping malformed decision dataset line: {exc}")
    if limit is None:
        return rows
    return rows[-limit:]


def iter_decisions() -> Iterable[dict]:
    for row in load_decisions():
        yield row
