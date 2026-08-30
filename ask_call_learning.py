"""Durable Ask Punky forecasts settled against the executable venue price path."""

from __future__ import annotations

import json
import hashlib
import fcntl
import os
import tempfile
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
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


@contextmanager
def _ledger_write_guard(data_dir: Path | None = None):
    """Serialize read-modify-write cycles across the agent and dashboard processes."""
    path = _ledger_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _lock:
        with lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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
    current = _safe_float(source.get("current"))
    explicit_entry = _safe_float(source.get("entry"))
    entry = explicit_entry or current
    source_name = str(source.get("source") or "ask_punky")[:80]
    asset_bucket = str(source.get("assetBucket") or "").lower().strip()
    rr_value = _safe_float(source.get("rr"))
    rr_bucket = str(source.get("rrBucket") or _rr_bucket(rr_value))
    setup_type = str(source.get("setupType") or (
        "long_above_level" if direction == "LONG" else "short_below_level"
    ))[:80]
    activated_at = str(source.get("activatedAt") or "").strip() or None
    if not activated_at and entry > 0 and current > 0:
        entry_is_live = current >= entry if direction == "LONG" else current <= entry
        if entry_is_live or explicit_entry <= 0:
            activated_at = _iso(analysed_ts)
    outcome = source.get("outcome")
    if outcome not in {0, 1} or str(source.get("resolutionSource") or "") != RESOLUTION_SOURCE:
        outcome = None
    status = str(source.get("status") or "").strip().lower()
    if outcome is not None:
        status = "resolved"
    elif status not in {"pending_entry", "active", "superseded", "expired_untriggered"}:
        status = "active" if activated_at else "pending_entry"
    return {
        "id": record_id,
        "ticker": ticker,
        "direction": direction,
        "decision": str(source.get("decision") or "")[:80],
        "query": str(source.get("query") or "")[:240],
        "source": source_name,
        "analysedAt": _iso(analysed_ts),
        "horizonHours": max(1.0, min(168.0, _safe_float(source.get("horizonHours"), 24.0))),
        "venueSymbol": str(source.get("venueSymbol") or resolve_hyperliquid_symbol(ticker) or ticker).strip(),
        "current": current,
        "entry": entry,
        "entryMode": str(source.get("entryMode") or ("above" if direction == "LONG" else "below"))[:20],
        "target": target,
        "invalidation": invalidation,
        "rr": rr_value,
        "rrBucket": rr_bucket,
        "assetBucket": asset_bucket,
        "setupType": setup_type,
        "probability": max(0.01, min(0.99, _safe_float(source.get("probability"), 0.5))),
        "evidenceQuality": max(0.0, min(100.0, _safe_float(source.get("evidenceQuality")))),
        "limitedHistory": bool(source.get("limitedHistory")),
        "outcome": outcome,
        "outcomeReason": str(source.get("outcomeReason") or "") if outcome is not None or status in {"superseded", "expired_untriggered"} else "",
        "resolvedAt": source.get("resolvedAt") if outcome is not None or status in {"superseded", "expired_untriggered"} else None,
        "resolutionSource": RESOLUTION_SOURCE if outcome is not None else "",
        "activatedAt": activated_at,
        "status": status,
    }


def _rr_bucket(value: Any) -> str:
    rr = _safe_float(value)
    if rr < 1.5:
        return "under_1_5"
    if rr < 2.5:
        return "1_5_to_2_5"
    return "over_2_5"


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
    if row.get("outcome") in {0, 1} or row.get("status") in {"superseded", "expired_untriggered"}:
        return row
    started_ts = _parse_ts(row.get("analysedAt"))
    expires_ts = started_ts + (_safe_float(row.get("horizonHours"), 24.0) * 3600.0)
    direction = str(row.get("direction") or "").upper()
    entry = _safe_float(row.get("entry") or row.get("current"))
    target = _safe_float(row.get("target"))
    invalidation = _safe_float(row.get("invalidation"))
    activated_ts = _parse_ts(row.get("activatedAt"))
    relevant = [
        bar for bar in bars
        if started_ts * 1000 <= _safe_float(bar.get("t")) <= expires_ts * 1000
    ]
    if not relevant:
        return row
    for bar in relevant:
        high = _safe_float(bar.get("h"))
        low = _safe_float(bar.get("l"))
        if activated_ts <= 0:
            entry_hit = high >= entry if direction == "LONG" else low <= entry
            if not entry_hit:
                continue
            activated_ms = _safe_float(bar.get("T") or bar.get("t"))
            activated_ts = activated_ms / 1000.0
            row["activatedAt"] = _iso(activated_ts)
            row["status"] = "active"
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
        row["status"] = "resolved"
        return row
    if now_ts >= expires_ts and len(relevant) >= 2:
        if activated_ts <= 0:
            row["outcome"] = None
            row["outcomeReason"] = "entry_never_reached"
            row["status"] = "expired_untriggered"
        else:
            row["outcome"] = 0
            row["outcomeReason"] = "horizon_expired_before_target"
            row["status"] = "resolved"
        row["resolvedAt"] = _iso(expires_ts)
        row["resolutionSource"] = RESOLUTION_SOURCE if row.get("outcome") in {0, 1} else ""
    return row


def settle_forecasts(rows: Iterable[dict], *, now_ts: float | None = None) -> list[dict]:
    now_value = float(now_ts or time.time())
    output = [dict(row) for row in rows]
    pending = [
        index for index, row in enumerate(output)
        if row.get("outcome") not in {0, 1}
        and row.get("status") not in {"superseded", "expired_untriggered"}
    ]
    for index in pending[:MAX_SETTLEMENTS_PER_PASS]:
        try:
            output[index] = _settle_record(output[index], _fetch_candles(output[index], now_value), now_value)
        except Exception as exc:
            log.debug("Ask forecast settlement skipped for %s: %s", output[index].get("ticker"), exc)
    return output


def upsert_forecasts(incoming: Iterable[dict], *, data_dir: Path | None = None, settle: bool = True) -> list[dict]:
    with _ledger_write_guard(data_dir):
        current = load_forecasts(data_dir=data_dir)
        by_id = {str(row.get("id")): dict(row) for row in current if row.get("id")}
        for raw in incoming:
            normalized = _normalize_record(dict(raw or {}))
            if not normalized:
                continue
            if normalized.get("source") == "dashboard_plan":
                for prior_id, prior_row in list(by_id.items()):
                    if (
                        prior_id != normalized["id"]
                        and prior_row.get("source") == "dashboard_plan"
                        and prior_row.get("ticker") == normalized.get("ticker")
                        and prior_row.get("direction") == normalized.get("direction")
                        and prior_row.get("outcome") not in {0, 1}
                        and not prior_row.get("activatedAt")
                        and prior_row.get("status") not in {"superseded", "expired_untriggered"}
                    ):
                        by_id[prior_id] = {
                            **prior_row,
                            "status": "superseded",
                            "outcomeReason": "plan_revised_before_entry",
                            "resolvedAt": normalized.get("analysedAt"),
                        }
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
    setup_families: dict[str, list[dict]] = defaultdict(list)
    for row in resolved:
        families[f"{row.get('ticker')}:{row.get('direction')}"].append(row)
        setup_families[str(row.get("setupType") or "unknown")].append(row)
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
    setup_stats = {}
    for key, items in setup_families.items():
        wins = sum(int(item.get("outcome") or 0) for item in items)
        setup_stats[key] = {
            "samples": len(items),
            "wins": wins,
            "hit_rate": round(wins / max(1, len(items)), 4),
        }
    wins = sum(int(row.get("outcome") or 0) for row in resolved)
    recent = sorted(
        resolved,
        key=lambda row: _parse_ts(row.get("resolvedAt")),
        reverse=True,
    )[:8]
    return {
        "records": len(all_rows),
        "resolved": len(resolved),
        "wins": wins,
        "hit_rate": round(wins / max(1, len(resolved)), 4) if resolved else None,
        "families": family_stats,
        "setup_families": setup_stats,
        "pending_entry": sum(row.get("status") == "pending_entry" for row in all_rows),
        "active": sum(row.get("status") == "active" and row.get("outcome") not in {0, 1} for row in all_rows),
        "expired_untriggered": sum(row.get("status") == "expired_untriggered" for row in all_rows),
        "superseded": sum(row.get("status") == "superseded" for row in all_rows),
        "recent": [
            {
                "ticker": row.get("ticker"),
                "direction": row.get("direction"),
                "outcome": row.get("outcome"),
                "outcome_reason": row.get("outcomeReason"),
                "resolved_at": row.get("resolvedAt"),
            }
            for row in recent
        ],
        "resolution_source": RESOLUTION_SOURCE,
    }


def _plan_fingerprint(ticker: str, direction: str, entry: float, target: float, invalidation: float) -> str:
    geometry = ":".join(
        f"{value:.6g}"
        for value in (entry, target, invalidation)
    )
    return hashlib.sha1(f"{ticker}:{direction}:{geometry}".encode("utf-8")).hexdigest()[:16]


def displayed_plan_records(snapshot: dict) -> list[dict]:
    board = dict((snapshot or {}).get("action_board") or {})
    records: list[dict] = []
    for item in list(board.get("items") or []):
        plan = dict((item or {}).get("plain_plan") or {})
        direction = str(plan.get("direction") or item.get("thesis_direction") or "").upper()
        entry = _safe_float(plan.get("entry"))
        target = _safe_float(plan.get("target"))
        invalidation = _safe_float(plan.get("invalidation"))
        ticker = str(item.get("coin") or "").upper().strip()
        if (
            not ticker
            or direction not in {"LONG", "SHORT"}
            or not item.get("analysis_fresh")
            or min(entry, target, invalidation) <= 0
        ):
            continue
        geometry_valid = target > entry > invalidation if direction == "LONG" else target < entry < invalidation
        if not geometry_valid:
            continue
        analysed_ts = _safe_float(item.get("analysis_updated_ts")) or time.time()
        current = _safe_float(plan.get("current") or item.get("venue_price") or item.get("reference_price"))
        reward_risk = _safe_float(plan.get("reward_risk"))
        setup_type = "long_above_level" if direction == "LONG" else "short_below_level"
        record_id = f"displayed:{ticker}:{direction}:{_plan_fingerprint(ticker, direction, entry, target, invalidation)}"
        item["plan_record_id"] = record_id
        records.append({
            "id": record_id,
            "ticker": ticker,
            "direction": direction,
            "decision": str(plan.get("headline") or "")[:80],
            "query": "automatic displayed dashboard plan",
            "source": "dashboard_plan",
            "analysedAt": _iso(analysed_ts),
            "horizonHours": 24 if str(item.get("asset_bucket") or "") == "coin" else 72,
            "venueSymbol": str(item.get("venue_symbol") or resolve_hyperliquid_symbol(ticker) or ticker),
            "current": current,
            "entry": entry,
            "entryMode": "above" if direction == "LONG" else "below",
            "target": target,
            "invalidation": invalidation,
            "rr": reward_risk,
            "rrBucket": _rr_bucket(reward_risk),
            "assetBucket": str(item.get("asset_bucket") or ""),
            "setupType": setup_type,
            "probability": _safe_float(item.get("probability"), 0.5),
            "evidenceQuality": _safe_float(item.get("thesis_conviction_score") or item.get("score"), 50.0),
            "limitedHistory": bool(item.get("execution_mode") == "analysis_only_new_listing"),
        })
    return records


def attach_plan_accountability(snapshot: dict, rows: Iterable[dict] | None = None) -> dict:
    safe_snapshot = snapshot if isinstance(snapshot, dict) else {}
    all_rows = list(rows) if rows is not None else load_forecasts()
    summary = forecast_summary(all_rows)
    by_id = {str(row.get("id") or ""): row for row in all_rows if row.get("id")}
    for item in list((safe_snapshot.get("action_board") or {}).get("items") or []):
        direction = str(item.get("thesis_direction") or "").upper()
        ticker = str(item.get("coin") or "").upper()
        setup_type = "long_above_level" if direction == "LONG" else "short_below_level"
        ticker_stats = dict(summary.get("families", {}).get(f"{ticker}:{direction}") or {})
        setup_stats = dict(summary.get("setup_families", {}).get(setup_type) or {})
        current_record = by_id.get(str(item.get("plan_record_id") or ""), {})
        item["plan_accuracy"] = {
            "ticker": ticker_stats,
            "setup": setup_stats,
            "setup_type": setup_type,
            "current_status": str(current_record.get("status") or "untracked"),
            "current_outcome": current_record.get("outcome"),
            "current_outcome_reason": str(current_record.get("outcomeReason") or ""),
            "activated_at": current_record.get("activatedAt"),
            "resolved_at": current_record.get("resolvedAt"),
        }
    safe_snapshot["plan_accountability"] = summary
    return safe_snapshot


def refresh_snapshot_accountability(snapshot: dict, *, data_dir: Path | None = None, settle: bool = False) -> dict:
    incoming = displayed_plan_records(snapshot)
    rows = upsert_forecasts(incoming, data_dir=data_dir, settle=settle)
    return attach_plan_accountability(snapshot, rows)
