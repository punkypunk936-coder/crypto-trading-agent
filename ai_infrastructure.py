"""Broad AI-infrastructure tape and sector-leadership scanner.

The trading loop should not need a full fundamental pass on every public stock
to notice that a vertical is moving. This module keeps that breadth scan cheap:
one batched Yahoo reference request covers the whole chain, while execution
remains restricted to the agent's supported Hyperliquid/Trade.xyz universe.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from datetime import datetime, timezone
from typing import Any

import requests


YAHOO_SPARK_URL = "https://query1.finance.yahoo.com/v7/finance/spark"
SCAN_CACHE_SECONDS = 300.0
REQUEST_CHUNK_SIZE = 20


SECTORS: dict[str, dict[str, Any]] = {
    "ai_chips": {
        "label": "AI Chips",
        "why": "Accelerators and silicon determine how much AI compute can be deployed.",
        "readthrough": "Strength should confirm in foundries, memory, networking, and server orders.",
        "members": {
            "NVDA": "NVDA", "AMD": "AMD", "AVGO": "AVGO", "ARM": "ARM",
            "QCOM": "QCOM", "INTC": "INTC", "CBRS": "CBRS",
        },
    },
    "foundries_fab": {
        "label": "Foundries & Fab",
        "why": "Foundry utilization and equipment demand confirm whether chip demand is becoming real capacity spend.",
        "readthrough": "Watch equipment orders, advanced packaging, and memory capex for confirmation.",
        "members": {
            "TSM": "TSM", "ASML": "ASML", "AMAT": "AMAT", "LRCX": "LRCX", "KLAC": "KLAC",
        },
    },
    "memory_storage": {
        "label": "Memory & Storage",
        "why": "HBM, DRAM, and storage content rise as AI servers become larger and denser.",
        "readthrough": "Sustained leadership implies the AI server bill of materials is broadening beyond GPUs.",
        "members": {
            "MU": "MU", "SNDK": "SNDK", "WDC": "WDC", "SMSN": "005930.KS",
            "SKHX": "000660.KS", "KIOXIA": "285A.T", "CXMT": "688825.SS",
        },
    },
    "neoclouds": {
        "label": "Neoclouds",
        "why": "Specialist GPU clouds are the high-beta expression of rented AI compute demand.",
        "readthrough": "Leadership needs backlog and utilization to outrun financing and supply concerns.",
        "members": {
            "CRWV": "CRWV", "NBIS": "NBIS", "IREN": "IREN", "APLD": "APLD",
            "WULF": "WULF", "CIFR": "CIFR",
        },
    },
    "hyperscalers": {
        "label": "Hyperscalers",
        "why": "The largest cloud platforms fund the capex cycle and turn AI demand into recurring revenue.",
        "readthrough": "Cloud growth, AI revenue, capex guidance, and free-cash-flow absorption decide durability.",
        "members": {
            "MSFT": "MSFT", "AMZN": "AMZN", "GOOGL": "GOOGL", "META": "META",
            "ORCL": "ORCL", "IBM": "IBM",
        },
    },
    "servers_racks": {
        "label": "Servers & Racks",
        "why": "Server and rack vendors show when chip demand is turning into deployable systems.",
        "readthrough": "A sustained move should pull networking, cooling, power, and component suppliers higher.",
        "members": {"DELL": "DELL", "HPE": "HPE", "SMCI": "SMCI"},
    },
    "networking_optics": {
        "label": "Networking & Optics",
        "why": "Large clusters need switches, interconnect, and optics to scale useful compute.",
        "readthrough": "Leadership confirms spend is shifting from isolated accelerators toward complete clusters.",
        "members": {
            "ANET": "ANET", "MRVL": "MRVL", "LITE": "LITE", "COHR": "COHR",
            "CRDO": "CRDO", "ALAB": "ALAB",
        },
    },
    "power_cooling": {
        "label": "Power & Cooling",
        "why": "Power delivery and thermal capacity are physical constraints on data-center growth.",
        "readthrough": "Persistent leadership signals that AI capex is reaching site construction and grid equipment.",
        "members": {
            "VRT": "VRT", "ETN": "ETN", "GEV": "GEV", "CEG": "CEG",
            "PWR": "PWR", "BE": "BE", "NRG": "NRG",
        },
    },
    "data_centers": {
        "label": "Data Centers",
        "why": "Data-center landlords and operators capture demand for powered, connected capacity.",
        "readthrough": "Watch leasing, occupancy, development yields, and power availability for confirmation.",
        "members": {"EQIX": "EQIX", "DLR": "DLR", "IRM": "IRM"},
    },
}


TRADEXYZ_AI_INFRA_SYMBOLS = {
    "AMD", "AMAT", "AMZN", "ARM", "ASML", "AVGO", "BE", "CBRS", "CRWV",
    "CXMT", "DELL", "GEV", "GOOGL", "IBM", "INTC", "KIOXIA", "LITE",
    "META", "MRVL", "MSFT", "MU", "NBIS", "NVDA", "ORCL", "QCOM", "SKHX",
    "SMSN", "SNDK", "TSM", "WDC",
}


_CACHE: dict[str, Any] = {"timestamp": 0.0, "report": None}
_CACHE_LOCK = threading.Lock()


def coverage_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for sector_key, sector in SECTORS.items():
        for symbol, yahoo_symbol in dict(sector.get("members") or {}).items():
            metadata[symbol] = {
                "display_name": symbol,
                "instrument_type": "equity",
                "categories": [sector_key, "ai_infra"],
                "sector": sector_key,
                "sector_label": str(sector.get("label") or sector_key),
                "yahoo_symbol": yahoo_symbol,
                "tradexyz": symbol in TRADEXYZ_AI_INFRA_SYMBOLS,
            }
    return metadata


AI_INFRA_COVERAGE_METADATA = coverage_metadata()
AI_INFRA_COVERAGE_COINS = list(AI_INFRA_COVERAGE_METADATA)
AI_INFRA_REFERENCE_ONLY_COINS = [
    symbol for symbol, meta in AI_INFRA_COVERAGE_METADATA.items() if not meta.get("tradexyz")
]


def _safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _move(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return (current / prior - 1.0) * 100.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _fetch_chunk(yahoo_symbols: list[str], *, session: Any = requests) -> dict[str, dict[str, Any]]:
    response = session.get(
        YAHOO_SPARK_URL,
        params={"symbols": ",".join(yahoo_symbols), "range": "1mo", "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    out: dict[str, dict[str, Any]] = {}
    for row in list(((payload.get("spark") or {}).get("result")) or []):
        yahoo_symbol = str((row or {}).get("symbol") or "").upper().strip()
        responses = list((row or {}).get("response") or [])
        if not yahoo_symbol or not responses:
            continue
        entry = dict(responses[0] or {})
        meta = dict(entry.get("meta") or {})
        quote_rows = list(((entry.get("indicators") or {}).get("quote")) or [])
        closes = [
            value for value in (_safe_float(raw) for raw in ((quote_rows[0] if quote_rows else {}).get("close") or []))
            if value is not None and value > 0
        ]
        timestamps = [int(value or 0) for value in list(entry.get("timestamp") or [])]
        current = _safe_float(meta.get("regularMarketPrice")) or (closes[-1] if closes else None)
        previous = _safe_float(meta.get("chartPreviousClose"))
        if previous is None and len(closes) >= 2:
            previous = closes[-2]
        out[yahoo_symbol] = {
            "current": current,
            "previous": previous,
            "closes": closes,
            "timestamps": timestamps[-len(closes):] if closes else [],
            "market_time": int(meta.get("regularMarketTime") or 0),
            "name": str(meta.get("shortName") or meta.get("longName") or yahoo_symbol),
        }
    return out


def fetch_reference_tape(*, session: Any = requests) -> tuple[dict[str, dict[str, Any]], list[str]]:
    metadata = AI_INFRA_COVERAGE_METADATA
    yahoo_symbols = list(dict.fromkeys(str(meta["yahoo_symbol"]).upper() for meta in metadata.values()))
    raw: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for start in range(0, len(yahoo_symbols), REQUEST_CHUNK_SIZE):
        chunk = yahoo_symbols[start:start + REQUEST_CHUNK_SIZE]
        try:
            raw.update(_fetch_chunk(chunk, session=session))
        except Exception as exc:
            errors.append(f"{chunk[0]}-{chunk[-1]}: {type(exc).__name__}")

    tape: dict[str, dict[str, Any]] = {}
    for symbol, meta in metadata.items():
        source = raw.get(str(meta["yahoo_symbol"]).upper())
        if not source or source.get("current") is None:
            continue
        closes = list(source.get("closes") or [])
        current = float(source["current"])
        move_1d = _move(current, _safe_float(source.get("previous")))
        move_5d = _move(current, closes[-6] if len(closes) >= 6 else (closes[0] if closes else None))
        move_20d = _move(current, closes[-21] if len(closes) >= 21 else (closes[0] if closes else None))
        tape[symbol] = {
            "coin": symbol,
            "name": source.get("name") or symbol,
            "sector": meta["sector"],
            "sector_label": meta["sector_label"],
            "price": round(current, 4),
            "move_1d": round(move_1d or 0.0, 2),
            "move_5d": round(move_5d or 0.0, 2),
            "move_20d": round(move_20d or 0.0, 2),
            "tradexyz": bool(meta.get("tradexyz")),
            "coverage_mode": "EXECUTABLE" if meta.get("tradexyz") else "REFERENCE_ONLY",
            "market_time": int(source.get("market_time") or 0),
            "history": [round(value, 4) for value in closes[-21:]],
        }
    return tape, errors


def _sector_setup(avg_1d: float, avg_5d: float, avg_20d: float, breadth: float) -> str:
    if avg_1d >= 1.0 and avg_5d <= 0.0 and breadth >= 60.0:
        return "EARLY ROTATION"
    if avg_1d > 0.0 and avg_5d > 0.0 and avg_20d > 0.0 and breadth >= 60.0:
        return "MOMENTUM INTACT"
    if avg_1d < 0.0 and avg_20d > 3.0:
        return "PULLBACK WATCH"
    if avg_1d < -0.5 and avg_5d < 0.0:
        return "FADING"
    return "MIXED"


def _build_sector_rows(tape: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sector_key, definition in SECTORS.items():
        members = [
            dict(tape[symbol]) for symbol in dict(definition.get("members") or {}) if symbol in tape
        ]
        if not members:
            continue
        moves_1d = [float(member["move_1d"]) for member in members]
        moves_5d = [float(member["move_5d"]) for member in members]
        moves_20d = [float(member["move_20d"]) for member in members]
        avg_1d = _mean(moves_1d)
        avg_5d = _mean(moves_5d)
        avg_20d = _mean(moves_20d)
        breadth = sum(1 for value in moves_1d if value > 0.0) / len(moves_1d) * 100.0
        traction_score = (avg_1d * 0.55) + ((avg_5d / 5.0) * 0.25) + ((avg_20d / 20.0) * 0.10) + ((breadth - 50.0) * 0.02)
        ranked = sorted(members, key=lambda item: float(item.get("move_1d") or 0.0), reverse=True)
        rows.append({
            "key": sector_key,
            "label": str(definition.get("label") or sector_key),
            "why": str(definition.get("why") or ""),
            "readthrough": str(definition.get("readthrough") or ""),
            "setup": _sector_setup(avg_1d, avg_5d, avg_20d, breadth),
            "traction_score": round(traction_score, 3),
            "move_1d": round(avg_1d, 2),
            "move_5d": round(avg_5d, 2),
            "move_20d": round(avg_20d, 2),
            "breadth_pct": round(breadth, 1),
            "covered_count": len(members),
            "universe_count": len(dict(definition.get("members") or {})),
            "leaders": ranked[:3],
            "laggards": list(reversed(ranked[-2:])),
            "members": ranked,
        })
    rows.sort(key=lambda row: float(row.get("traction_score") or 0.0), reverse=True)
    for index, row in enumerate(rows):
        score = float(row.get("traction_score") or 0.0)
        if index < 2 and score > 0.0:
            row["status"] = "LEADING"
        elif score >= 0.35:
            row["status"] = "IMPROVING"
        elif score <= -0.35:
            row["status"] = "LAGGING"
        else:
            row["status"] = "MIXED"
    return rows


def _build_report(tape: dict[str, dict[str, Any]], errors: list[str]) -> dict[str, Any]:
    sectors = _build_sector_rows(tape)
    all_members = sorted(tape.values(), key=lambda item: float(item.get("move_1d") or 0.0), reverse=True)
    unusual = sorted(
        [member for member in all_members if abs(float(member.get("move_1d") or 0.0)) >= 5.0],
        key=lambda member: abs(float(member.get("move_1d") or 0.0)),
        reverse=True,
    )
    positive_sectors = sum(1 for sector in sectors if float(sector.get("move_1d") or 0.0) > 0.0)
    positive_names = sum(1 for member in all_members if float(member.get("move_1d") or 0.0) > 0.0)
    breadth = positive_names / len(all_members) * 100.0 if all_members else 0.0
    lead = sectors[0] if sectors else {}
    lag = sectors[-1] if sectors else {}
    if positive_sectors >= max(6, math.ceil(len(sectors) * 0.67)) and breadth >= 60.0:
        regime = "BROAD AI RISK-ON"
    elif positive_sectors <= max(2, math.floor(len(sectors) * 0.33)) and breadth < 40.0:
        regime = "AI RISK-OFF"
    elif lead and float(lead.get("move_1d") or 0.0) >= 1.0:
        regime = "NARROW LEADERSHIP"
    else:
        regime = "MIXED ROTATION"
    lead_names = ", ".join(str(item.get("coin") or "") for item in list(lead.get("leaders") or [])[:2])
    headline = (
        f"{lead.get('label', 'AI infrastructure')} leads the AI stack"
        + (f", paced by {lead_names}." if lead_names else ".")
    )
    action = (
        f"Start with {lead.get('label', 'the leading vertical')}. Confirmation test: "
        f"{lead.get('readthrough', 'strength should spread into adjacent suppliers')}"
    )
    avoid = (
        f"Do not treat the whole AI basket as one trade: {lag.get('label', 'the weakest vertical')} "
        f"is the current weak link ({float(lag.get('move_1d') or 0.0):+.2f}% average)."
        if lag else "Do not force a basket-level conclusion without sector confirmation."
    )
    focus: list[str] = []
    focus_candidates = list(unusual[:6])
    for sector in sectors[:6]:
        leaders = list(sector.get("leaders") or [])
        if leaders:
            focus_candidates.append(leaders[0])
        executable_leader = next((member for member in leaders if member.get("tradexyz")), None)
        if executable_leader:
            focus_candidates.append(executable_leader)
    focus_candidates.extend(all_members)
    for member in focus_candidates:
        symbol = str(member.get("coin") or "")
        if symbol and symbol not in focus:
            focus.append(symbol)
        if len(focus) >= 12:
            break
    newest_market_time = max((int(member.get("market_time") or 0) for member in all_members), default=0)
    return {
        "enabled": True,
        "active": bool(sectors),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "market_time": datetime.fromtimestamp(newest_market_time, tz=timezone.utc).isoformat() if newest_market_time else "",
        "source": "Yahoo Finance reference tape",
        "regime": regime,
        "headline": headline,
        "action": action,
        "avoid": avoid,
        "focus_symbols": focus,
        "sectors": sectors,
        "leaders": all_members[:6],
        "laggards": list(reversed(all_members[-4:])),
        "unusual_movers": unusual[:12],
        "summary": {
            "universe_count": len(AI_INFRA_COVERAGE_METADATA),
            "covered_count": len(all_members),
            "sector_count": len(sectors),
            "positive_sector_count": positive_sectors,
            "breadth_pct": round(breadth, 1),
            "unusual_mover_count": len(unusual),
            "error_count": len(errors),
        },
        "errors": errors,
    }


def build_ai_infrastructure_context(*, force_refresh: bool = False, session: Any = requests) -> dict[str, Any]:
    now = time.time()
    with _CACHE_LOCK:
        cached = _CACHE.get("report")
        cached_at = float(_CACHE.get("timestamp") or 0.0)
        if not force_refresh and cached and now - cached_at < SCAN_CACHE_SECONDS:
            return dict(cached)

    tape, errors = fetch_reference_tape(session=session)
    report = _build_report(tape, errors)
    with _CACHE_LOCK:
        _CACHE["timestamp"] = now
        _CACHE["report"] = report
    return dict(report)
