"""
decision_dataset.py — Structured per-cycle decision records for AI learning.

Unlike the closed-trade dataset, this captures every decision the agent makes:
trades it takes, trades it blocks, and moments where it correctly stays flat.
That gives later models the "what did we see?" context that pure trade logs miss.
"""

from __future__ import annotations

from collections import deque
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable

from logger import get_logger
from paths import DATA_DIR, DECISION_DATASET_JSONL, FEATURE_STORE_JSONL

log = get_logger("decision_dataset")
RUNTIME_DATA_DIR = Path.home() / "Library" / "Application Support" / "crypto_trading_agent_runtime"
MAX_ACTIVE_DATASET_BYTES = int(os.getenv("DECISION_DATASET_MAX_ACTIVE_MB", "512")) * 1024 * 1024
TAIL_READ_CHUNK_BYTES = 1024 * 1024
MAX_TAIL_READ_BYTES = int(os.getenv("DECISION_DATASET_MAX_TAIL_MB", "128")) * 1024 * 1024


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _dataset_path(data_dir: Path | None = None) -> Path:
    return (Path(data_dir).expanduser() if data_dir else DECISION_DATASET_JSONL.parent) / DECISION_DATASET_JSONL.name


def _feature_store_path(data_dir: Path | None = None) -> Path:
    return (Path(data_dir).expanduser() if data_dir else FEATURE_STORE_JSONL.parent) / FEATURE_STORE_JSONL.name


def _file_richness(path: Path) -> tuple[int, float]:
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime
    except Exception:
        return 0, 0.0


def _tail_lines(path: Path, limit: int) -> list[str]:
    if limit <= 0 or not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        chunks: list[bytes] = []
        newline_count = 0
        bytes_read = 0
        while position > 0 and newline_count <= limit and bytes_read < MAX_TAIL_READ_BYTES:
            read_size = min(TAIL_READ_CHUNK_BYTES, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
            bytes_read += read_size
    payload = b"".join(reversed(chunks))
    return payload.decode("utf-8", errors="replace").splitlines()[-limit:]


def _rotate_if_oversized(path: Path) -> None:
    try:
        if MAX_ACTIVE_DATASET_BYTES <= 0 or path.stat().st_size < MAX_ACTIVE_DATASET_BYTES:
            return
    except FileNotFoundError:
        return
    except OSError as exc:
        log.warning("Could not inspect decision dataset for rotation: %s", exc)
        return

    archive = path.with_name(f"{path.stem}.archive-{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    try:
        path.replace(archive)
        log.info("Rotated oversized decision dataset to %s", archive)
    except OSError as exc:
        log.warning("Could not rotate oversized decision dataset: %s", exc)


def resolve_richest_decision_data_dir(preferred: Path | None = None) -> Path:
    candidates: list[Path] = []
    for candidate in (preferred, DATA_DIR, RUNTIME_DATA_DIR):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path not in candidates:
            candidates.append(path)

    scored: list[tuple[tuple[int, float, int, float], Path]] = []
    for path in candidates:
        decision_size, decision_mtime = _file_richness(_dataset_path(path))
        feature_size, feature_mtime = _file_richness(_feature_store_path(path))
        scored.append(((decision_size, decision_mtime, feature_size, feature_mtime), path))

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
    _rotate_if_oversized(dataset_path)
    with dataset_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def load_decisions(limit: int | None = None, *, data_dir: Path | None = None) -> list[dict]:
    dataset_path = _dataset_path(data_dir)
    if not dataset_path.exists():
        return []

    source = _tail_lines(dataset_path, limit) if limit is not None else dataset_path.open(encoding="utf-8")
    rows: list[dict] | deque[dict] = deque(maxlen=limit) if limit is not None else []
    try:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                log.debug(f"Skipping malformed decision dataset line: {exc}")
    finally:
        if limit is None:
            source.close()
    return list(rows)


def iter_decisions() -> Iterable[dict]:
    for row in load_decisions():
        yield row
