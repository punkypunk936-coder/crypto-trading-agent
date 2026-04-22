"""
decision_dataset.py — Structured per-cycle decision records for AI learning.

Unlike the closed-trade dataset, this captures every decision the agent makes:
trades it takes, trades it blocks, and moments where it correctly stays flat.
That gives later models the "what did we see?" context that pure trade logs miss.
"""

from __future__ import annotations

from collections import deque
import json
import time
from pathlib import Path
from typing import Any, Iterable

from logger import get_logger
from paths import DATA_DIR, DECISION_DATASET_JSONL, FEATURE_STORE_JSONL

log = get_logger("decision_dataset")
RUNTIME_DATA_DIR = Path.home() / "Library" / "Application Support" / "crypto_trading_agent_runtime"


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _dataset_path(data_dir: Path | None = None) -> Path:
    return (Path(data_dir).expanduser() if data_dir else DECISION_DATASET_JSONL.parent) / DECISION_DATASET_JSONL.name


def _feature_store_path(data_dir: Path | None = None) -> Path:
    return (Path(data_dir).expanduser() if data_dir else FEATURE_STORE_JSONL.parent) / FEATURE_STORE_JSONL.name


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def resolve_richest_decision_data_dir(preferred: Path | None = None) -> Path:
    candidates: list[Path] = []
    for candidate in (preferred, DATA_DIR, RUNTIME_DATA_DIR):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path not in candidates:
            candidates.append(path)

    scored: list[tuple[tuple[int, int], Path]] = []
    for path in candidates:
        decision_rows = _count_jsonl_rows(_dataset_path(path))
        feature_rows = _count_jsonl_rows(_feature_store_path(path))
        scored.append(((decision_rows, feature_rows), path))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else Path(DATA_DIR).expanduser()


def append_decision(record: dict, *, data_dir: Path | None = None) -> None:
    if not isinstance(record, dict) or not record:
        return

    payload = dict(record)
    payload.setdefault("recorded_at_ts", time.time())
    payload.setdefault(
        "decision_id",
        f"{payload.get('cycle_number', 0)}:{payload.get('coin', 'UNKNOWN')}:{payload.get('stage', 'decision')}:{int(payload['recorded_at_ts'] * 1000)}",
    )
    dataset_path = _dataset_path(data_dir)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def load_decisions(limit: int | None = None, *, data_dir: Path | None = None) -> list[dict]:
    dataset_path = _dataset_path(data_dir)
    if not dataset_path.exists():
        return []

    rows: list[dict] | deque[dict]
    rows = deque(maxlen=limit) if limit is not None else []
    with dataset_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                log.debug(f"Skipping malformed decision dataset line: {exc}")
    if isinstance(rows, deque):
        return list(rows)
    return rows


def iter_decisions() -> Iterable[dict]:
    for row in load_decisions():
        yield row
