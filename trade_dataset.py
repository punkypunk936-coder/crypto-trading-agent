"""
trade_dataset.py — Structured closed-trade dataset for learning and audit.

This complements the CSV log with a richer JSONL record per completed trade so
the agent can later train or analyze on the full trade thesis, execution
quality, and outcome geometry without scraping strings back out of the dashboard.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable

from logger import get_logger
from paths import TRADE_DATASET_JSONL

log = get_logger("trade_dataset")


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def append_closed_trade(record: dict) -> None:
    if not isinstance(record, dict) or not record:
        return

    payload = dict(record)
    payload.setdefault("recorded_at_ts", time.time())
    TRADE_DATASET_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with TRADE_DATASET_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def load_closed_trades(limit: int | None = None) -> list[dict]:
    if not TRADE_DATASET_JSONL.exists():
        return []

    rows: list[dict] = []
    with TRADE_DATASET_JSONL.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                log.debug(f"Skipping malformed trade dataset line: {exc}")
    if limit is None:
        return rows
    return rows[-limit:]


def iter_closed_trades() -> Iterable[dict]:
    for row in load_closed_trades():
        yield row
