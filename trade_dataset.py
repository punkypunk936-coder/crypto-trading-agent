"""
trade_dataset.py — Structured closed-trade dataset for learning and audit.

This complements the CSV log with a richer JSONL record per completed trade so
the agent can later train or analyze on the full trade thesis, execution
quality, and outcome geometry without scraping strings back out of the dashboard.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from logger import get_logger
from paths import DATA_DIR, TRADE_DATASET_JSONL, TRADES_CSV

log = get_logger("trade_dataset")
RUNTIME_DATA_DIR = Path.home() / "Library" / "Application Support" / "crypto_trading_agent_runtime"


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


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


def _dataset_path(data_dir: Path | None = None) -> Path:
    return (Path(data_dir).expanduser() if data_dir else TRADE_DATASET_JSONL.parent) / TRADE_DATASET_JSONL.name


def _csv_path(data_dir: Path | None = None) -> Path:
    return (Path(data_dir).expanduser() if data_dir else TRADES_CSV.parent) / TRADES_CSV.name


def _file_richness(path: Path) -> tuple[int, float]:
    try:
        stat = path.stat()
        return stat.st_size, stat.st_mtime
    except Exception:
        return 0, 0.0


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except Exception:
        return 0


def resolve_richest_history_data_dir(preferred: Path | None = None) -> Path:
    candidates: list[Path] = []
    for candidate in (preferred, DATA_DIR, RUNTIME_DATA_DIR):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path not in candidates:
            candidates.append(path)

    scored: list[tuple[tuple[int, float, int, float, int, int], Path]] = []
    for path in candidates:
        dataset_size, dataset_mtime = _file_richness(_dataset_path(path))
        csv_rows = _count_csv_rows(_csv_path(path))
        decision_size, _ = _file_richness(path / "decision_dataset.jsonl")
        feature_size, _ = _file_richness(path / "feature_store.jsonl")
        score = (
            max(dataset_size, csv_rows),  # best closed-trade history wins first
            dataset_mtime,
            dataset_size,
            float(csv_rows),
            decision_size,
            feature_size,
        )
        scored.append((score, path))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else Path(DATA_DIR).expanduser()


def resolve_history_data_dir(preferred: Path | None = None) -> Path:
    if preferred is not None:
        return Path(preferred).expanduser()
    return resolve_richest_history_data_dir()


def append_closed_trade(record: dict, *, data_dir: Path | None = None) -> None:
    if not isinstance(record, dict) or not record:
        return

    payload = dict(record)
    payload.setdefault("recorded_at_ts", time.time())
    dataset_path = _dataset_path(data_dir)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=_json_default, sort_keys=True) + "\n")


def _tail_lines(path: Path, limit: int, chunk_bytes: int = 1024 * 1024) -> list[str]:
    if limit <= 0 or not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        chunks: list[bytes] = []
        newline_count = 0
        while position > 0 and newline_count <= limit:
            read_size = min(chunk_bytes, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            chunks.append(chunk)
            newline_count += chunk.count(b"\n")
    payload = b"".join(reversed(chunks))
    return payload.decode("utf-8", errors="replace").splitlines()[-limit:]


def _load_jsonl_rows(path: Path, limit: int | None = None) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    source = _tail_lines(path, limit) if limit is not None else path.open(encoding="utf-8")
    try:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                log.debug(f"Skipping malformed trade dataset line: {exc}")
    finally:
        if limit is None:
            source.close()
    return rows


def _parse_timestamp(value: Any) -> float | None:
    text = _safe_str(value)
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            continue
    return None


def _normalize_csv_trade_row(row: dict) -> dict:
    outcome = _safe_str(row.get("result"), "UNKNOWN").upper()
    if outcome not in {"WIN", "LOSS", "BREAKEVEN"}:
        pnl_usd = _safe_float(row.get("pnl_usd"))
        outcome = "WIN" if pnl_usd > 0 else ("LOSS" if pnl_usd < 0 else "BREAKEVEN")

    opened_at_ts = _parse_timestamp(row.get("opened_at"))
    closed_at_ts = _parse_timestamp(row.get("closed_at"))
    hold_minutes = _safe_float(row.get("duration_mins"))
    if hold_minutes <= 0 and opened_at_ts and closed_at_ts and closed_at_ts >= opened_at_ts:
        hold_minutes = round((closed_at_ts - opened_at_ts) / 60.0, 4)

    return {
        "trade_id": _safe_int(row.get("trade_id"), 0),
        "coin": _safe_str(row.get("coin")).upper(),
        "direction": _safe_str(row.get("direction")).upper(),
        "entry_price": _safe_float(row.get("entry_price")),
        "exit_price": _safe_float(row.get("exit_price")),
        "size_usd": _safe_float(row.get("size_usd")),
        "signal_score": _safe_float(row.get("signal_score"), 50.0),
        "hold_minutes": hold_minutes,
        "pnl_usd": _safe_float(row.get("pnl_usd")),
        "pnl_pct": _safe_float(row.get("pnl_pct")),
        "outcome": outcome,
        "exit_reason": _safe_str(row.get("exit_reason")),
        "opened_at_ts": opened_at_ts,
        "closed_at_ts": closed_at_ts,
        "recorded_at_ts": closed_at_ts or time.time(),
        "exchange": "csv_backfill",
        "entry_context": {},
        "exit_context": {},
        "execution_quality": {},
        "trade_plan": {},
        "plan_outcome": {},
        "thesis": {
            "state": "UNKNOWN",
            "quality": "UNKNOWN",
            "candidate_action": _safe_str(row.get("direction")).upper(),
            "permitted": True,
            "summary": "Backfilled from CSV history",
        },
        "backfilled_from_csv": True,
    }


def load_csv_closed_trades(limit: int | None = None, *, data_dir: Path | None = None) -> list[dict]:
    csv_path = _csv_path(data_dir)
    if not csv_path.exists():
        return []

    rows: list[dict] = []
    try:
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if not row:
                    continue
                rows.append(_normalize_csv_trade_row(row))
    except Exception as exc:
        log.warning("Could not read CSV trade history %s: %s", csv_path, exc)
        return []

    if limit is None:
        return rows
    return rows[-limit:]


def ensure_backfilled_from_csv(*, data_dir: Path | None = None) -> dict:
    target_dir = resolve_history_data_dir(data_dir)
    dataset_path = _dataset_path(target_dir)
    structured_rows = _load_jsonl_rows(dataset_path)
    csv_rows = load_csv_closed_trades(data_dir=target_dir)

    existing_keys = {
        (
            _safe_int(row.get("trade_id"), 0),
            _safe_str(row.get("coin")).upper(),
            _safe_str(row.get("direction")).upper(),
            round(_safe_float(row.get("opened_at_ts")), 3),
            round(_safe_float(row.get("closed_at_ts")), 3),
        )
        for row in structured_rows
    }
    appended = 0
    for row in csv_rows:
        key = (
            _safe_int(row.get("trade_id"), 0),
            _safe_str(row.get("coin")).upper(),
            _safe_str(row.get("direction")).upper(),
            round(_safe_float(row.get("opened_at_ts")), 3),
            round(_safe_float(row.get("closed_at_ts")), 3),
        )
        if key in existing_keys:
            continue
        append_closed_trade(row, data_dir=target_dir)
        existing_keys.add(key)
        appended += 1

    return {
        "data_dir": str(target_dir),
        "csv_rows": len(csv_rows),
        "structured_rows_before": len(structured_rows),
        "appended_rows": appended,
    }


def load_closed_trades(
    limit: int | None = None,
    *,
    data_dir: Path | None = None,
    backfill_from_csv: bool = True,
) -> list[dict]:
    target_dir = resolve_history_data_dir(data_dir)
    dataset_path = _dataset_path(target_dir)
    rows = _load_jsonl_rows(dataset_path, limit=limit)

    csv_rows = load_csv_closed_trades(limit=limit, data_dir=target_dir) if backfill_from_csv else []
    if backfill_from_csv and limit is None and csv_rows and len(rows) < len(csv_rows):
        try:
            ensure_backfilled_from_csv(data_dir=target_dir)
            rows = _load_jsonl_rows(dataset_path, limit=limit)
        except Exception as exc:
            log.warning("CSV backfill into %s failed: %s", dataset_path, exc)

    if not rows and csv_rows:
        rows = list(csv_rows)

    if limit is None:
        return rows
    return rows[-limit:]


def iter_closed_trades() -> Iterable[dict]:
    for row in load_closed_trades():
        yield row
