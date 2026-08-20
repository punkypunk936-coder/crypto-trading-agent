"""Durable Ask Punky forecasts settled against the executable venue price path."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from exchanges.hyperliquid_markets import resolve_hyperliquid_symbol
from logger import get_logger
from paths import ASK_FORECAST_LEDGER_JSONL

log = get_logger("ask_learning")

HL_INFO_URL = "https://api.hyperliquid.xyz/info"
MAX_RECORDS = 2000
MAX_SETTLEMENTS_PER_PASS = 40
RESOLUTION_SOURCE = "venue_candle_path_v2"
_lock = threading.Lock()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if number == number else default
    except (TypeError, ValueError):
        return default


def _parse_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _ledger_path(data_dir: Path | None = None) -> Path:
    return (Path(data_dir).expanduser() if data_dir else ASK_FORECAST_LEDGER_JSONL.parent) / ASK_FORECAST_LEDGER_JSONL.name


def _normalize_record(source: dict) -> dict | None:
    ticker = str(source.get("ticker") or source.get("coin") or "").upper().strip()
    direction = str(source.get("direction") or "").upper().strip()
    analysed_at = str(source.get("analysedAt") or source.get("analysed_at") or "").strip()
    analysed_ts = _parse_ts(analysed_at)
    target = _safe_float(source.get("target"))
    invalidation = _safe_float(source.get("invalidation"))
    if not ticker or direction not in {"LONG", "SHORT"} or analysed_ts <= 0 or target <= 0 or invalidation <= 0:
        return None
    record_id = str(source.get("id") or f"{ticker}-{direction}-{int(analysed_ts * 1000)}")[:180]
    outcome = source.get("outcome")
    if outcome not in {0, 1} or str(source.get("resolutionSource") or "") != RESOLUTION_SOURCE:
        outcome = None
    return {
        "id": record_id,
        "ticker": ticker,
        "direction": direction,
        "decision": str(source.get("decision") or "")[:80],
        "query": str(source.get("query") or "")[:240],
        "analysedAt": _iso(analysed_ts),
        "horizonHours": max(1.0, min(168.0, _safe_float(source.get("horizonHours"), 24.0))),
        "venueSymbol": str(source.get("venueSymbol") or resolve_hyperliquid_symbol(ticker) or ticker).strip(),
        "current": _safe_float(source.get("current")),
        "target": target,
        "invalidation": invalidation,
        "rr": _safe_float(source.get("rr")),
        "rrBucket": str(source.get("rrBucket") or ""),
        "assetBucket": str(source.get("assetBucket") or ""),
        "probability": max(0.01, min(0.99, _safe_float(source.get("probability"), 0.5))),
        "evidenceQuality": max(0.0, min(100.0, _safe_float(source.get("evidenceQuality")))),
        "limitedHistory": bool(source.get("limitedHistory")),
        "outcome": outcome,
        "outcomeReason": str(source.get("outcomeReason") or "") if outcome is not None else "",
        "resolvedAt": source.get("resolvedAt") if outcome is not None else None,
        "resolutionSource": RESOLUTION_SOURCE if outcome is not None else "",
    }


def load_forecasts(*, data_dir: Path | None = None) -> list[dict]:
    path = _ledger_path(data_dir)
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    continue
                normalized = _normalize_record(row) if isinstance(row, dict) else None
                if normalized:
                    rows.append(normalized)
    except OSError as exc:
        log.debug("Ask forecast ledger read failed: %s", exc)
    return rows[-MAX_RECORDS:]


def _write_forecasts(rows: Iterable[dict], *, data_dir: Path | None = None) -> None:
    path = _ledger_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = [row for row in (_normalize_record(dict(item)) for item in rows) if row][-MAX_RECORDS:]
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            for row in normalized:
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temp_path = Path(handle.name)
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _fetch_candles(record: dict, now_ts: float) -> list[dict]:
    started_ts = _parse_ts(record.get("analysedAt"))
    expires_ts = started_ts + (_safe_float(record.get("horizonHours"), 24.0) * 3600.0)
    end_ts = min(now_ts, expires_ts)
    if started_ts <= 0 or end_ts <= started_ts:
        return []
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": str(record.get("venueSymbol") or record.get("ticker") or ""),
            "interval": "15m",
            "startTime": int((started_ts - 900.0) * 1000),
            "endTime": int((end_ts + 900.0) * 1000),
        },
    }
    response = requests.post(HL_INFO_URL, json=payload, timeout=8)
    response.raise_for_status()
    raw = response.json()
    return sorted(
        [item for item in raw if isinstance(item, dict) and _safe_float(item.get("t")) > 0],
        key=lambda item: _safe_float(item.get("t")),
    )


def _settle_record(record: dict, bars: list[dict], now_ts: float) -> dict:
    row = dict(record)
    if row.get("outcome") in {0, 1}:
        return row
    started_ts = _parse_ts(row.get("analysedAt"))
    expires_ts = started_ts + (_safe_float(row.get("horizonHours"), 24.0) * 3600.0)
    direction = str(row.get("direction") or "").upper()
    target = _safe_float(row.get("target"))
    invalidation = _safe_float(row.get("invalidation"))
    relevant = [
        bar for bar in bars
        if started_ts * 1000 <= _safe_float(bar.get("t")) <= expires_ts * 1000
    ]
    if not relevant:
        return row
    for bar in relevant:
        high = _safe_float(bar.get("h"))
        low = _safe_float(bar.get("l"))
        target_hit = high >= target if direction == "LONG" else low <= target
        invalid_hit = low <= invalidation if direction == "LONG" else high >= invalidation
        if not target_hit and not invalid_hit:
            continue
        ambiguous = target_hit and invalid_hit
        row["outcome"] = 0 if ambiguous or invalid_hit else 1
        row["outcomeReason"] = (
            "target_and_invalidation_same_candle" if ambiguous
            else "invalidation_first" if invalid_hit
            else "target_first"
        )
        close_ms = _safe_float(bar.get("T") or bar.get("t"))
        row["resolvedAt"] = _iso(close_ms / 1000.0)
        row["resolutionSource"] = RESOLUTION_SOURCE
        return row
    if now_ts >= expires_ts and len(relevant) >= 2:
        row["outcome"] = 0
        row["outcomeReason"] = "horizon_expired_before_target"
        row["resolvedAt"] = _iso(expires_ts)
        row["resolutionSource"] = RESOLUTION_SOURCE
    return row


def settle_forecasts(rows: Iterable[dict], *, now_ts: float | None = None) -> list[dict]:
    now_value = float(now_ts or time.time())
    output = [dict(row) for row in rows]
    pending = [index for index, row in enumerate(output) if row.get("outcome") not in {0, 1}]
    for index in pending[:MAX_SETTLEMENTS_PER_PASS]:
        try:
            output[index] = _settle_record(output[index], _fetch_candles(output[index], now_value), now_value)
        except Exception as exc:
            log.debug("Ask forecast settlement skipped for %s: %s", output[index].get("ticker"), exc)
    return output


def upsert_forecasts(incoming: Iterable[dict], *, data_dir: Path | None = None, settle: bool = True) -> list[dict]:
    with _lock:
        current = load_forecasts(data_dir=data_dir)
        by_id = {str(row.get("id")): dict(row) for row in current if row.get("id")}
        for raw in incoming:
            normalized = _normalize_record(dict(raw or {}))
            if not normalized:
                continue
            prior = by_id.get(normalized["id"], {})
            if prior.get("outcome") in {0, 1}:
                normalized.update({
                    "outcome": prior.get("outcome"),
                    "outcomeReason": prior.get("outcomeReason"),
                    "resolvedAt": prior.get("resolvedAt"),
                    "resolutionSource": prior.get("resolutionSource"),
                })
            by_id[normalized["id"]] = {**prior, **normalized}
        rows = sorted(by_id.values(), key=lambda row: _parse_ts(row.get("analysedAt")))[-MAX_RECORDS:]
        if settle:
            rows = settle_forecasts(rows)
        _write_forecasts(rows, data_dir=data_dir)
        return rows


def forecast_summary(rows: Iterable[dict]) -> dict:
    all_rows = list(rows)
    resolved = [row for row in all_rows if row.get("outcome") in {0, 1}]
    families: dict[str, list[dict]] = defaultdict(list)
    for row in resolved:
        families[f"{row.get('ticker')}:{row.get('direction')}"].append(row)
    family_stats = {}
    for key, items in families.items():
        wins = sum(int(item.get("outcome") or 0) for item in items)
        probability_error = sum(
            (_safe_float(item.get("probability"), 0.5) - int(item.get("outcome") or 0)) ** 2
            for item in items
        ) / max(1, len(items))
        family_stats[key] = {
            "samples": len(items),
            "wins": wins,
            "hit_rate": round(wins / max(1, len(items)), 4),
            "brier": round(probability_error, 4),
        }
    wins = sum(int(row.get("outcome") or 0) for row in resolved)
    return {
        "records": len(all_rows),
        "resolved": len(resolved),
        "wins": wins,
        "hit_rate": round(wins / max(1, len(resolved)), 4) if resolved else None,
        "families": family_stats,
        "resolution_source": RESOLUTION_SOURCE,
    }
