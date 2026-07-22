"""Season-gated earnings workflow for tracked equities."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


COMPANY_METRICS: dict[str, list[str]] = {
    "AAPL": ["iPhone revenue", "Services growth", "gross margin", "China demand", "guidance"],
    "AMD": ["Data Center revenue", "Instinct GPU revenue", "EPYC growth", "gross margin", "AI guidance"],
    "AMZN": ["AWS growth", "AWS operating margin", "retail margin", "capex", "guidance"],
    "GOOGL": ["Search revenue", "Cloud growth", "Cloud margin", "AI capex", "guidance"],
    "META": ["ad revenue", "ad impressions/pricing", "AI capex", "Reality Labs loss", "guidance"],
    "MSFT": ["Azure growth", "AI contribution", "Copilot adoption", "capex", "guidance"],
    "NVDA": ["Data Center revenue", "Blackwell/Rubin supply", "gross margin", "networking", "guidance"],
    "TSM": ["AI/HPC growth", "advanced-node mix", "CoWoS capacity", "gross margin", "capex"],
    "MU": ["DRAM/NAND pricing", "HBM revenue", "gross margin", "inventory", "guidance"],
    "SNDK": ["NAND pricing", "enterprise SSD demand", "bit shipments", "gross margin", "guidance"],
    "INTC": ["Data Center revenue", "Xeon share", "18A milestones", "foundry losses", "guidance"],
    "MRVL": ["Data Center growth", "custom silicon", "electro-optics", "gross margin", "guidance"],
    "NFLX": ["revenue", "operating margin", "engagement", "ad-tier growth", "guidance"],
    "TSLA": ["auto gross margin", "deliveries", "energy growth", "robotaxi/FSD", "capex"],
    "HIMS": ["subscribers", "GLP-1 revenue", "CAC", "retention", "guidance"],
    "RIVN": ["deliveries", "automotive gross margin", "cash burn", "R2 milestones", "guidance"],
}

SECTOR_METRICS: dict[str, list[str]] = {
    "semis_memory": ["revenue", "gross margin", "AI/data-centre demand", "inventory", "guidance"],
    "mag7": ["revenue growth", "operating margin", "AI monetization", "capex", "guidance"],
    "neoclouds": ["revenue", "backlog", "utilization", "financing/capex", "guidance"],
    "crypto_equities": ["revenue", "transaction volumes", "subscription revenue", "expenses", "guidance"],
    "consumer": ["revenue", "traffic/users", "gross margin", "operating margin", "guidance"],
}

UPCOMING_MARKERS = (
    "pre-event setup",
    "earnings calendar",
    "earnings today",
    "results today",
    "announce financial results",
    "conference call",
)
POST_MARKERS = (
    "reported results",
    "reports results",
    "earnings beat",
    "earnings miss",
    "beat estimates",
    "missed estimates",
    "raised guidance",
    "cut guidance",
)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _short(value: Any, limit: int = 220) -> str:
    text = _text(value)
    return text if len(text) <= limit else text[: max(0, limit - 3)].rstrip() + "..."


def _event_text(signal: dict) -> str:
    headlines = signal.get("news_top_headlines") or signal.get("top_headlines") or []
    if isinstance(headlines, str):
        headlines = [headlines]
    values = [
        signal.get("official_event_summary"),
        signal.get("news_event_summary"),
        signal.get("news_catalyst_summary"),
        signal.get("news_headline"),
        *list(headlines)[:4],
    ]
    return " | ".join(_text(value) for value in values if _text(value))


def _phase(event_text: str) -> str:
    lower = event_text.lower()
    if any(marker in lower for marker in POST_MARKERS):
        return "POST_REPORT"
    if "today" in lower or "just reported" in lower:
        return "LIVE"
    return "PRE_EVENT"


def _active_candidate(signal: dict, event_text: str) -> bool:
    lower = event_text.lower()
    event_score = max(_number(signal.get("news_event_score")), _number(signal.get("official_event_score")))
    has_window = any(marker in lower for marker in UPCOMING_MARKERS + POST_MARKERS)
    return bool(has_window and event_score >= 2.0)


def _metrics(coin: str, categories: list[str]) -> list[str]:
    if coin in COMPANY_METRICS:
        return list(COMPANY_METRICS[coin])
    for category in categories:
        if category in SECTOR_METRICS:
            return list(SECTOR_METRICS[category])
    return ["revenue", "earnings", "margin", "key operating KPI", "guidance"]


def _direction(signal: dict) -> str:
    action = _text(signal.get("action")).upper()
    if action in {"LONG", "SHORT"}:
        return action
    bias = _text(signal.get("market_map_bias")).upper()
    return "LONG" if bias == "BULLISH" else "SHORT" if bias == "BEARISH" else "FLAT"


def _result_excerpt(event_text: str) -> str:
    for part in event_text.split("|"):
        lower = part.lower()
        if any(marker in lower for marker in POST_MARKERS) or re.search(r"(?:revenue|eps|margin|guidance).*[$%0-9]", lower):
            return _short(part, 240)
    return ""


def _load_ledger(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {"events": {}}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {"events": {}}
    return payload if isinstance(payload, dict) else {"events": {}}


def _save_ledger(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        tmp_path.replace(path)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def build_earnings_session(
    state: dict | None,
    daily_radar: dict | None = None,
    *,
    ledger_path: Path | None = None,
    now: datetime | None = None,
) -> dict:
    safe_state = dict(state or {})
    signals = dict(safe_state.get("signals") or {})
    config = dict(safe_state.get("config") or {})
    category_map = dict(config.get("asset_categories") or {})
    radar_rows = {
        _text(row.get("coin")).upper(): dict(row or {})
        for row in (daily_radar or {}).get("top_assets") or []
        if _text((row or {}).get("coin"))
    }
    now_dt = now or datetime.now(timezone.utc)
    if now_dt.tzinfo is None:
        now_dt = now_dt.replace(tzinfo=timezone.utc)
    ledger = _load_ledger(ledger_path)
    events = dict(ledger.get("events") or {})

    current_rows: list[dict] = []
    for coin, signal_value in signals.items():
        signal = dict(signal_value or {})
        instrument = _text(signal.get("instrument_type") or (config.get("instrument_types") or {}).get(coin)).lower()
        if instrument != "equity":
            continue
        event_text = _event_text(signal)
        if not _active_candidate(signal, event_text):
            continue
        phase = _phase(event_text)
        categories = list(signal.get("asset_categories") or category_map.get(coin) or [])
        metrics = _metrics(coin, categories)
        radar = dict(radar_rows.get(coin) or {})
        implied_move = _number(signal.get("options_implied_move_pct"))
        direction = _direction(signal)
        result_excerpt = _result_excerpt(event_text)
        invalidation = _text(radar.get("invalidation") or signal.get("first_principles_wrong_if"))
        row = {
            "coin": coin,
            "phase": phase,
            "direction": direction,
            "event_score": round(max(_number(signal.get("news_event_score")), _number(signal.get("official_event_score"))), 2),
            "implied_move_pct": round(implied_move, 2),
            "metrics": metrics,
            "pre_thesis": _short(radar.get("thesis") or signal.get("first_principles_plain_thesis") or signal.get("decision_reason"), 240),
            "invalidation": invalidation,
            "event_context": _short(event_text, 260),
            "reported_figures": result_excerpt,
            "post_report_check": (
                f"Compare actual versus consensus for {', '.join(metrics[:4])}; then require guidance and price to agree before changing the strategic stance."
            ),
            "flip_condition": (
                f"Review or flip only if at least two core metrics miss, guidance weakens, and price confirms {invalidation.lower() or 'a structural break'}."
            ),
            "decision": "REVIEW" if phase == "POST_REPORT" else "WAIT FOR PRINT",
            "last_seen": now_dt.isoformat(),
        }
        revision = _number(signal.get("analyst_revision_score"))
        if phase == "POST_REPORT" and revision <= -0.75 and direction == "SHORT":
            row["decision"] = "FLIP RISK"
        elif phase == "POST_REPORT" and revision >= 0.75 and direction == "LONG":
            row["decision"] = "THESIS CONFIRMED"
        previous = dict(events.get(coin) or {})
        row["first_seen"] = previous.get("first_seen") or now_dt.isoformat()
        events[coin] = row
        current_rows.append(row)

    active = len(current_rows) >= 2 or any(row.get("phase") in {"LIVE", "POST_REPORT"} for row in current_rows)
    rows = sorted(
        current_rows,
        key=lambda row: (
            {"LIVE": 0, "POST_REPORT": 1, "PRE_EVENT": 2}.get(str(row.get("phase")), 3),
            -_number(row.get("event_score")),
            str(row.get("coin")),
        ),
    )[:8]
    ledger["events"] = events
    ledger["updated_at"] = now_dt.isoformat()
    _save_ledger(ledger_path, ledger)

    return {
        "enabled": True,
        "active": active,
        "updated_at": now_dt.isoformat(),
        "headline": f"Earnings season: {len(rows)} tracked company setups are active." if active else "Outside an active tracked earnings window.",
        "rows": rows if active else [],
        "pre_event_count": sum(row.get("phase") == "PRE_EVENT" for row in rows),
        "live_count": sum(row.get("phase") == "LIVE" for row in rows),
        "post_report_count": sum(row.get("phase") == "POST_REPORT" for row in rows),
    }
