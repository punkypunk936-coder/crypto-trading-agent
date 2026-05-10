"""
proactive_intelligence.py — proactive trading research layer.

This module turns the current signal surface into a research desk:
  - thesis ledger
  - morning scout book
  - read-through engine
  - starter basket optimizer
  - forecast calibration
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from logger import get_logger
from paths import DATA_DIR, FORECAST_LEDGER_JSONL, PROACTIVE_TRADER_REPORT_JSON, THESIS_LEDGER_JSONL

log = get_logger("proactive")

READ_THROUGH_LINKS: dict[str, list[str]] = {
    "crypto": ["crypto_equities", "meme_momentum", "growth"],
    "crypto_equities": ["crypto", "meme_momentum", "growth"],
    "mag7": ["semis_memory", "ai_infra", "neoclouds", "crypto_equities"],
    "semis_memory": ["mag7", "ai_infra", "neoclouds", "asia_macro"],
    "ai_infra": ["semis_memory", "neoclouds", "mag7"],
    "neoclouds": ["semis_memory", "ai_infra", "mag7"],
    "growth": ["mag7", "crypto", "meme_momentum"],
    "meme_momentum": ["crypto", "growth", "crypto_equities"],
    "indices_macro": ["mag7", "growth", "crypto"],
    "energy": ["indices_macro", "commodities_metals"],
    "commodities_metals": ["indices_macro", "energy", "asia_macro"],
    "asia_macro": ["semis_memory", "commodities_metals", "indices_macro"],
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now().isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _safe_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = value.replace("|", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = str(item or "").strip().lower()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _data_path(data_dir: Path | None, path: Path) -> Path:
    base = Path(data_dir).expanduser() if data_dir else Path(DATA_DIR).expanduser()
    return base / path.name


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
    except Exception as exc:
        log.debug("Could not read %s: %s", path, exc)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), default=_json_default, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True), encoding="utf-8")


def _parse_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = _safe_str(value)
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _config_maps(state: dict) -> tuple[dict, dict, dict]:
    config = dict((state or {}).get("config") or {})
    return (
        dict(config.get("instrument_types") or {}),
        dict(config.get("asset_categories") or {}),
        dict(config.get("portfolio_theme_map") or {}),
    )


def _instrument_type(coin: str, signal: dict, state: dict) -> str:
    types, _, _ = _config_maps(state)
    return _safe_str(signal.get("instrument_type") or types.get(coin), "crypto").lower()


def _categories(coin: str, signal: dict, state: dict) -> list[str]:
    _, category_map, _ = _config_maps(state)
    categories = _safe_list(signal.get("asset_categories") or signal.get("asset_category"))
    if categories:
        return categories
    categories = _safe_list(category_map.get(coin))
    if categories:
        return categories
    itype = _instrument_type(coin, signal, state)
    if itype == "index":
        return ["indices_macro"]
    if itype == "equity":
        return ["other_stocks"]
    return ["crypto"]


def _theme(coin: str, categories: list[str], state: dict) -> str:
    _, _, theme_map = _config_maps(state)
    explicit = _safe_str(theme_map.get(coin))
    if explicit:
        return explicit
    return _safe_str((categories or ["crypto"])[0], "crypto").upper()


def _asset_bucket(instrument_type: str) -> str:
    return "coin" if _safe_str(instrument_type, "crypto").lower() == "crypto" else "equity"


def _direction(coin: str, signal: dict, market_entry: dict | None = None) -> str:
    action = _safe_str(signal.get("action")).upper()
    if action in {"LONG", "SHORT"}:
        return action
    state = _safe_str(signal.get("asset_state")).upper()
    if "SHORT" in state or "BREAKDOWN" in state:
        return "SHORT"
    if "LONG" in state or "RECLAIM" in state:
        return "LONG"
    fp_direction = _safe_str(signal.get("first_principles_direction")).upper()
    if fp_direction in {"LONG", "SHORT"} and _safe_float(signal.get("first_principles_sequence_score")) >= 60.0:
        return fp_direction
    bias = _safe_str(signal.get("market_map_bias") or (market_entry or {}).get("bias")).upper()
    if bias == "BULLISH":
        return "LONG"
    if bias == "BEARISH":
        return "SHORT"
    score = _safe_float(signal.get("score"), 50.0)
    if score >= 56:
        return "LONG"
    if score <= 44:
        return "SHORT"
    return "FLAT"


def _side(direction: str) -> str:
    if direction == "LONG":
        return "bullish"
    if direction == "SHORT":
        return "bearish"
    return "neutral"


def _live_price(signal: dict) -> float:
    return _safe_float(signal.get("live_price") or signal.get("price") or signal.get("analysis_price"))


def _conviction_score(signal: dict, direction: str) -> float:
    score = _safe_float(signal.get("score"), 50.0)
    directional_strength = abs(score - 50.0) * 1.15
    if direction == "LONG" and score < 50:
        directional_strength *= 0.75
    if direction == "SHORT" and score > 50:
        directional_strength *= 0.75
    catalyst = _safe_float(signal.get("news_catalyst_score")) * 5.5
    event = _safe_float(signal.get("news_event_score")) * 6.0
    official = _safe_float(signal.get("official_event_score")) * 4.0
    revisions = max(0.0, _safe_float(signal.get("analyst_revision_score"))) * 3.0
    first_principles_score = max(0.0, _safe_float(signal.get("first_principles_sequence_score")) - 50.0) * 0.45
    fundamentals = max(0.0, _safe_float(signal.get("first_principles_fundamental_score")) - 50.0) * 0.35
    attention = max(0.0, _safe_float(signal.get("first_principles_attention_score") or signal.get("social_attention_score")) - 50.0) * 0.30
    thesis = max(0.0, _safe_float(signal.get("thesis_conviction_score"), score) - 50.0) * 0.55
    expectancy = max(0.0, _safe_float(signal.get("expectancy_probability")) - 0.50) * 100.0
    starter = 8.0 if bool(signal.get("conviction_entry_active")) else 0.0
    return round(max(0.0, min(100.0, 30.0 + directional_strength + catalyst + event + official + revisions + first_principles_score + fundamentals + attention + thesis + expectancy + starter)), 2)


def _thesis_summary(coin: str, direction: str, signal: dict, market_entry: dict | None) -> str:
    explicit = _safe_str(
        signal.get("first_principles_plain_thesis")
        or signal.get("thesis_summary")
        or signal.get("decision_reason")
        or signal.get("flat_reason")
    )
    if explicit:
        return explicit[:260]
    if _safe_float(signal.get("news_event_score")) >= 3:
        return f"{coin} has event/catalyst pressure that justifies a proactive {direction.lower()} thesis before full confirmation."
    bias = _safe_str(signal.get("market_map_bias") or (market_entry or {}).get("bias") or "neutral").lower()
    return f"{coin} has a {bias} map read and enough signal pressure to keep a proactive {direction.lower()} thesis alive."


def _invalidation(direction: str, signal: dict, market_entry: dict | None) -> str:
    if direction == "LONG":
        support = _safe_float(signal.get("market_map_nearest_support"))
        supports = (market_entry or {}).get("supports") or []
        if not support and supports:
            support = max(_safe_float(item) for item in supports)
        return f"Invalidate below {support:,.4g}" if support else "Invalidate if catalyst flow fades or reclaim fails."
    if direction == "SHORT":
        resistance = _safe_float(signal.get("market_map_nearest_resistance"))
        resistances = (market_entry or {}).get("resistances") or []
        if not resistance and resistances:
            positive = [_safe_float(item) for item in resistances if _safe_float(item) > 0]
            resistance = min(positive) if positive else 0.0
        return f"Invalidate above {resistance:,.4g}" if resistance else "Invalidate if price reclaims supply or bearish catalyst fades."
    return "No trade until direction becomes explicit."


def _candidate_rows(state: dict, market_map: dict, *, min_conviction: float = 48.0) -> list[dict]:
    signals = dict((state or {}).get("signals") or {})
    entries = dict((market_map or {}).get("coins") or {})
    rows: list[dict] = []
    for coin, raw_signal in signals.items():
        coin = _safe_str(coin).upper()
        if not coin:
            continue
        signal = dict(raw_signal or {})
        market_entry = dict(entries.get(coin) or {})
        direction = _direction(coin, signal, market_entry)
        if direction == "FLAT":
            continue
        conviction = _conviction_score(signal, direction)
        event_score = _safe_float(signal.get("news_event_score"))
        catalyst_score = _safe_float(signal.get("news_catalyst_score"))
        if conviction < min_conviction and event_score < 2.5 and catalyst_score < 2.5:
            continue
        instrument_type = _instrument_type(coin, signal, state)
        categories = _categories(coin, signal, state)
        rows.append({
            "coin": coin,
            "direction": direction,
            "side": _side(direction),
            "instrument_type": instrument_type,
            "asset_bucket": _asset_bucket(instrument_type),
            "categories": categories,
            "primary_category": categories[0] if categories else "crypto",
            "theme": _theme(coin, categories, state),
            "signal": signal,
            "market_entry": market_entry,
            "price": _live_price(signal),
            "score": _safe_float(signal.get("score"), 50.0),
            "conviction_score": conviction,
            "event_score": event_score,
            "catalyst_score": catalyst_score,
            "probability": _safe_float(signal.get("expectancy_probability")) or min(0.72, max(0.48, conviction / 100.0)),
        })
    rows.sort(key=lambda item: (-item["conviction_score"], item["coin"]))
    return rows


def build_thesis_ledger_report(
    state: dict,
    market_map: dict,
    *,
    config: Any = None,
    data_dir: Path | None = None,
) -> dict:
    max_theses = int(_cfg_value(config, "proactive_report_max_theses", 24) or 24)
    path = _data_path(data_dir, THESIS_LEDGER_JSONL)
    existing = _read_jsonl(path)
    by_id = {str(row.get("thesis_id") or ""): dict(row) for row in existing if row.get("thesis_id")}
    today = _now().date().isoformat()
    active: list[dict] = []
    for row in _candidate_rows(state, market_map, min_conviction=46.0)[:max_theses]:
        coin = row["coin"]
        direction = row["direction"]
        signal = row["signal"]
        market_entry = row["market_entry"]
        thesis_type = "event" if row["event_score"] >= 3.0 else ("catalyst" if row["catalyst_score"] >= 3.0 else "setup")
        thesis_id = f"{today}:{coin}:{direction}:{thesis_type}"
        previous = by_id.get(thesis_id, {})
        price = row["price"]
        probability = round(max(0.05, min(0.95, row["probability"])), 4)
        thesis = {
            "thesis_id": thesis_id,
            "created_at": previous.get("created_at") or _iso_now(),
            "updated_at": _iso_now(),
            "coin": coin,
            "direction": direction,
            "side": row["side"],
            "instrument_type": row["instrument_type"],
            "asset_bucket": row["asset_bucket"],
            "categories": row["categories"],
            "theme": row["theme"],
            "thesis_type": thesis_type,
            "entry_price_reference": round(price, 6) if price else 0.0,
            "conviction_score": row["conviction_score"],
            "probability": probability,
            "expected_surprise": _safe_str(signal.get("analyst_revision_summary") or signal.get("news_event_summary") or signal.get("news_catalyst_summary") or "Catalyst surprise not explicit yet."),
            "why_now": _thesis_summary(coin, direction, signal, market_entry),
            "first_principles": dict(signal.get("first_principles") or {}),
            "likely_path": _safe_str(signal.get("first_principles_likely_path") or "Likely path not explicit yet."),
            "attention_summary": _safe_str(signal.get("social_attention_summary") or "No configured trader-flow read yet."),
            "base_case": _safe_str(signal.get("thesis_summary") or signal.get("market_map_summary") or "Base case is continuation in the thesis direction if confirmation arrives."),
            "bull_case": f"{coin} expands in the thesis direction as catalyst flow, positioning, and structure align.",
            "bear_case": f"{coin} fails the setup if the market fades the catalyst or rejects the trigger zone.",
            "invalidation": _invalidation(direction, signal, market_entry),
            "event_tags": list(signal.get("news_event_tags") or []),
            "catalyst_summary": _safe_str(signal.get("news_catalyst_summary")),
            "event_summary": _safe_str(signal.get("news_event_summary")),
            "status": "active",
        }
        by_id[thesis_id] = thesis
        active.append(thesis)

    rows = sorted(by_id.values(), key=lambda item: _parse_ts(item.get("updated_at")), reverse=True)[:1000]
    try:
        _write_jsonl(path, rows)
    except Exception as exc:
        log.debug("thesis ledger write failed: %s", exc)
    return {
        "enabled": True,
        "path": str(path),
        "active_theses": active,
        "recent_theses": rows[:max_theses],
        "summary": {
            "active_count": len(active),
            "ledger_count": len(rows),
            "bullish_count": sum(1 for item in active if item.get("direction") == "LONG"),
            "bearish_count": sum(1 for item in active if item.get("direction") == "SHORT"),
        },
    }


def build_read_through_report(state: dict, market_map: dict, thesis_report: dict | None = None) -> dict:
    candidates = _candidate_rows(state, market_map, min_conviction=45.0)
    by_coin = {item["coin"]: item for item in candidates}
    impacts: list[dict] = []
    sources = [
        item for item in candidates
        if item["conviction_score"] >= 58.0 or item["event_score"] >= 3.0 or item["catalyst_score"] >= 3.0
    ][:12]
    for source in sources:
        source_cats = set(source["categories"])
        linked = set()
        for category in source_cats:
            linked.update(READ_THROUGH_LINKS.get(category, []))
        for target in candidates:
            if target["coin"] == source["coin"]:
                continue
            target_cats = set(target["categories"])
            overlap = bool(source_cats & target_cats)
            linked_hit = bool(target_cats & linked)
            if not overlap and not linked_hit:
                continue
            relation_weight = 1.0 if overlap else 0.68
            score = round((source["conviction_score"] * 0.55 + target["conviction_score"] * 0.35 + (target["event_score"] + target["catalyst_score"]) * 4.0) * relation_weight, 2)
            impacts.append({
                "source": source["coin"],
                "target": target["coin"],
                "direction": source["direction"],
                "side": source["side"],
                "source_theme": source["theme"],
                "target_theme": target["theme"],
                "relationship": "same category" if overlap else "linked theme",
                "score": score,
                "summary": f"{source['coin']} {source['side']} pressure can read through to {target['coin']} via {', '.join(sorted(source_cats & target_cats or target_cats & linked))}.",
            })
    impacts.sort(key=lambda item: (-_safe_float(item.get("score")), item.get("source", ""), item.get("target", "")))
    return {
        "enabled": True,
        "sources": sources,
        "top_impacts": impacts[:20],
        "summary": {
            "source_count": len(sources),
            "impact_count": len(impacts),
            "top_line": impacts[0]["summary"] if impacts else "No strong cross-asset read-through yet.",
        },
    }


def build_morning_scout_book(
    state: dict,
    market_map: dict,
    thesis_report: dict,
    read_through_report: dict,
    *,
    config: Any = None,
) -> dict:
    max_names = int(_cfg_value(config, "morning_scout_max_names", 14) or 14)
    read_bonus = defaultdict(float)
    read_notes = defaultdict(list)
    for impact in (read_through_report or {}).get("top_impacts", []) or []:
        target = _safe_str(impact.get("target")).upper()
        if not target:
            continue
        read_bonus[target] += min(16.0, _safe_float(impact.get("score")) / 8.0)
        if len(read_notes[target]) < 2:
            read_notes[target].append(_safe_str(impact.get("summary")))

    calls: list[dict] = []
    for thesis in (thesis_report or {}).get("active_theses", []) or []:
        coin = _safe_str(thesis.get("coin")).upper()
        direction = _safe_str(thesis.get("direction")).upper()
        if direction not in {"LONG", "SHORT"}:
            continue
        scout_score = round(min(100.0, _safe_float(thesis.get("conviction_score")) + read_bonus[coin]), 2)
        calls.append({
            "coin": coin,
            "direction": direction,
            "side": _side(direction),
            "asset_bucket": thesis.get("asset_bucket", "coin"),
            "categories": list(thesis.get("categories") or []),
            "theme": thesis.get("theme", ""),
            "thesis_id": thesis.get("thesis_id", ""),
            "scout_score": scout_score,
            "probability": _safe_float(thesis.get("probability")),
            "why": _safe_str(thesis.get("why_now")),
            "invalidation": _safe_str(thesis.get("invalidation")),
            "read_through": [note for note in read_notes[coin] if note],
            "event_setup": thesis.get("thesis_type") in {"event", "catalyst"},
        })

    calls.sort(key=lambda item: (-_safe_float(item.get("scout_score")), item.get("coin", "")))
    bullish = [item for item in calls if item.get("direction") == "LONG"][:max_names]
    bearish = [item for item in calls if item.get("direction") == "SHORT"][:max_names]
    starter_candidates = [
        item for item in calls
        if _safe_float(item.get("scout_score")) >= _safe_float(_cfg_value(config, "proactive_starter_min_conviction", 58.0), 58.0)
    ][:max_names]
    return {
        "enabled": True,
        "generated_at": _iso_now(),
        "session": "morning_scout_book",
        "bullish_calls": bullish,
        "bearish_calls": bearish,
        "starter_candidates": starter_candidates,
        "summary": {
            "call_count": len(calls),
            "bullish_count": len([item for item in calls if item.get("direction") == "LONG"]),
            "bearish_count": len([item for item in calls if item.get("direction") == "SHORT"]),
            "starter_count": len(starter_candidates),
            "top_call": calls[0] if calls else None,
        },
    }


def _existing_exposure(state: dict) -> tuple[float, dict[str, float], set[str]]:
    total = 0.0
    by_theme: dict[str, float] = defaultdict(float)
    occupied: set[str] = set()
    signals = dict((state or {}).get("signals") or {})
    for item in list((state or {}).get("positions") or []) + list((state or {}).get("pending_orders") or []):
        coin = _safe_str((item or {}).get("coin")).upper()
        if not coin:
            continue
        occupied.add(coin)
        signal = dict(signals.get(coin) or {})
        categories = _categories(coin, signal, state)
        theme = _theme(coin, categories, state)
        size = _safe_float((item or {}).get("size_usd"))
        total += size
        by_theme[theme] += size
    return total, by_theme, occupied


def build_starter_basket(
    state: dict,
    scout_book: dict,
    *,
    config: Any = None,
) -> dict:
    portfolio_usd = max(0.0, _safe_float((state or {}).get("portfolio_usd")))
    if portfolio_usd <= 0:
        return {"enabled": True, "allocations": [], "skipped": [], "summary": {"reason": "portfolio value unavailable"}}
    total_cap = portfolio_usd * _safe_float(_cfg_value(config, "event_risk_budget_max_portfolio_pct", 0.10), 0.10)
    single_cap = portfolio_usd * _safe_float(_cfg_value(config, "event_risk_budget_max_single_pct", 0.02), 0.02)
    theme_cap = portfolio_usd * _safe_float(_cfg_value(config, "event_risk_budget_max_theme_pct", 0.08), 0.08)
    min_trade = _safe_float(_cfg_value(config, "event_risk_budget_min_trade_usd", 100.0), 100.0)
    max_names = int(_cfg_value(config, "proactive_starter_basket_max_names", 6) or 6)
    strict_caps = bool(_cfg_value(config, "event_risk_budget_strict_caps", True))
    if not strict_caps:
        single_cap = max(single_cap, min_trade)
        theme_cap = max(theme_cap, min_trade)
        total_cap = max(total_cap, min_trade)
    total_used, theme_used, occupied = _existing_exposure(state)
    remaining = max(0.0, total_cap - total_used)
    allocations: list[dict] = []
    skipped: list[dict] = []
    for candidate in (scout_book or {}).get("starter_candidates", []) or []:
        if len(allocations) >= max_names:
            break
        coin = _safe_str(candidate.get("coin")).upper()
        if not coin:
            continue
        theme = _safe_str(candidate.get("theme"), "UNKNOWN")
        if coin in occupied:
            skipped.append({"coin": coin, "reason": "already has exposure"})
            continue
        theme_remaining = max(0.0, theme_cap - theme_used.get(theme, 0.0))
        if remaining < min_trade:
            skipped.append({"coin": coin, "reason": "event risk budget already used"})
            continue
        if theme_remaining < min_trade:
            skipped.append({"coin": coin, "reason": f"{theme} theme budget already used"})
            continue
        score = _safe_float(candidate.get("scout_score"))
        confidence_multiplier = max(0.35, min(1.0, score / 85.0))
        desired = min(single_cap * confidence_multiplier, remaining, theme_remaining)
        if desired < min_trade:
            skipped.append({"coin": coin, "reason": "starter size below minimum after budget caps"})
            continue
        size = round(desired, 2)
        allocation = {
            "coin": coin,
            "direction": candidate.get("direction"),
            "side": candidate.get("side"),
            "theme": theme,
            "size_usd": size,
            "portfolio_pct": round(size / portfolio_usd, 4),
            "scout_score": round(score, 2),
            "thesis_id": candidate.get("thesis_id", ""),
            "why": candidate.get("why", ""),
        }
        allocations.append(allocation)
        occupied.add(coin)
        remaining -= size
        theme_used[theme] += size
    return {
        "enabled": True,
        "allocations": allocations,
        "skipped": skipped[:12],
        "budget": {
            "portfolio_usd": round(portfolio_usd, 2),
            "total_cap_usd": round(total_cap, 2),
            "pre_existing_event_like_exposure_usd": round(total_used, 2),
            "remaining_after_plan_usd": round(max(0.0, remaining), 2),
            "single_cap_usd": round(single_cap, 2),
            "theme_cap_usd": round(theme_cap, 2),
        },
        "summary": {
            "allocation_count": len(allocations),
            "planned_usd": round(sum(_safe_float(item.get("size_usd")) for item in allocations), 2),
            "top_allocation": allocations[0] if allocations else None,
        },
    }


def build_pair_trade_book(
    state: dict,
    scout_book: dict,
    *,
    config: Any = None,
) -> dict:
    if not _cfg_value(config, "pair_trade_book_enabled", True):
        return {"enabled": False, "pairs": [], "hedge_allocations": [], "summary": {"reason": "pair trade book disabled"}}

    portfolio_usd = max(0.0, _safe_float((state or {}).get("portfolio_usd")))
    if portfolio_usd <= 0:
        return {"enabled": True, "pairs": [], "hedge_allocations": [], "summary": {"reason": "portfolio value unavailable"}}

    max_pairs = max(0, int(_cfg_value(config, "pair_trade_max_pairs", 2) or 2))
    if max_pairs <= 0:
        return {"enabled": True, "pairs": [], "hedge_allocations": [], "summary": {"reason": "pair limit is zero"}}

    min_equity_score = _safe_float(_cfg_value(config, "pair_trade_min_equity_score", 63.0), 63.0)
    min_crypto_score = _safe_float(_cfg_value(config, "pair_trade_min_crypto_score", 58.0), 58.0)
    pair_cap = portfolio_usd * _safe_float(_cfg_value(config, "pair_trade_max_notional_pct", 0.015), 0.015)
    hedge_ratio = max(0.10, min(1.0, _safe_float(_cfg_value(config, "pair_trade_hedge_ratio", 0.35), 0.35)))
    min_trade = max(
        _safe_float(_cfg_value(config, "min_trade_usd", 0.0), 0.0),
        _safe_float(_cfg_value(config, "event_risk_budget_min_trade_usd", 0.0), 0.0),
    )
    if pair_cap < min_trade:
        return {
            "enabled": True,
            "pairs": [],
            "hedge_allocations": [],
            "summary": {"reason": f"pair cap ${pair_cap:.0f} is below minimum ${min_trade:.0f}"},
        }

    allowed_crypto = {
        _safe_str(item).upper()
        for item in (_cfg_value(config, "pair_trade_crypto_hedge_candidates", []) or [])
        if _safe_str(item)
    }
    _, _, occupied = _existing_exposure(state)
    bullish_calls = list((scout_book or {}).get("bullish_calls") or [])
    bearish_calls = list((scout_book or {}).get("bearish_calls") or [])
    equity_longs = [
        item for item in bullish_calls
        if item.get("direction") == "LONG"
        and item.get("asset_bucket") == "equity"
        and _safe_float(item.get("scout_score")) >= min_equity_score
    ]
    crypto_shorts = [
        item for item in bearish_calls
        if item.get("direction") == "SHORT"
        and item.get("asset_bucket") == "coin"
        and _safe_float(item.get("scout_score")) >= min_crypto_score
        and (not allowed_crypto or _safe_str(item.get("coin")).upper() in allowed_crypto)
        and _safe_str(item.get("coin")).upper() not in occupied
    ]

    pairs: list[dict] = []
    hedge_allocations: list[dict] = []
    for equity in equity_longs:
        if len(pairs) >= max_pairs:
            break
        for crypto in crypto_shorts:
            if len(pairs) >= max_pairs:
                break
            short_coin = _safe_str(crypto.get("coin")).upper()
            if not short_coin or any(pair.get("short_coin") == short_coin for pair in pairs):
                continue
            long_coin = _safe_str(equity.get("coin")).upper()
            pair_score = round((_safe_float(equity.get("scout_score")) * 0.62 + _safe_float(crypto.get("scout_score")) * 0.38), 2)
            size = round(min(pair_cap, max(min_trade, pair_cap * hedge_ratio)), 2)
            summary = f"Long {long_coin} thesis paired with short {short_coin} while crypto tape is weaker."
            pair_id = f"{long_coin}_LONG__{short_coin}_SHORT"
            pair = {
                "pair_id": pair_id,
                "long_coin": long_coin,
                "long_direction": "LONG",
                "short_coin": short_coin,
                "short_direction": "SHORT",
                "pair_score": pair_score,
                "theme": equity.get("theme", ""),
                "hedge_theme": crypto.get("theme", "CRYPTO_BETA"),
                "hedge_size_usd": size,
                "hedge_portfolio_pct": round(size / portfolio_usd, 4),
                "hedge_ratio": round(hedge_ratio, 4),
                "why": summary,
                "long_invalidation": equity.get("invalidation", ""),
                "short_invalidation": crypto.get("invalidation", ""),
            }
            pairs.append(pair)
            hedge_allocations.append({
                "coin": short_coin,
                "direction": "SHORT",
                "side": "bearish",
                "theme": crypto.get("theme", "CRYPTO_BETA"),
                "size_usd": size,
                "portfolio_pct": round(size / portfolio_usd, 4),
                "scout_score": pair_score,
                "thesis_id": crypto.get("thesis_id", ""),
                "why": summary,
                "pair_trade": True,
                "pair_id": pair_id,
                "paired_long": long_coin,
                "paired_long_score": _safe_float(equity.get("scout_score")),
                "hedge_ratio": round(hedge_ratio, 4),
                "invalidation": crypto.get("invalidation", ""),
            })

    return {
        "enabled": True,
        "pairs": pairs,
        "hedge_allocations": hedge_allocations,
        "summary": {
            "pair_count": len(pairs),
            "hedge_allocation_count": len(hedge_allocations),
            "planned_hedge_usd": round(sum(_safe_float(item.get("size_usd")) for item in hedge_allocations), 2),
            "top_pair": pairs[0] if pairs else None,
            "reason": "no clean equity-long/crypto-short pair" if not pairs else "",
        },
    }


def _forecast_price_for_coin(coin: str, state: dict) -> float:
    signal = dict(((state or {}).get("signals") or {}).get(coin) or {})
    return _live_price(signal)


def build_forecast_calibration(
    state: dict,
    scout_book: dict,
    *,
    config: Any = None,
    data_dir: Path | None = None,
) -> dict:
    path = _data_path(data_dir, FORECAST_LEDGER_JSONL)
    rows = _read_jsonl(path)
    by_id = {str(row.get("forecast_id") or ""): dict(row) for row in rows if row.get("forecast_id")}
    now_ts = time.time()
    today = _now().date().isoformat()
    horizon_hours = _safe_float(_cfg_value(config, "proactive_forecast_horizon_hours", 24.0), 24.0)
    max_open = int(_cfg_value(config, "proactive_forecast_max_open", 60) or 60)
    for candidate in ((scout_book or {}).get("bullish_calls", []) + (scout_book or {}).get("bearish_calls", []))[:20]:
        coin = _safe_str(candidate.get("coin")).upper()
        direction = _safe_str(candidate.get("direction")).upper()
        if not coin or direction not in {"LONG", "SHORT"}:
            continue
        forecast_id = f"{today}:{coin}:{direction}:{int(horizon_hours)}h"
        if forecast_id in by_id:
            continue
        price = _forecast_price_for_coin(coin, state)
        if price <= 0:
            continue
        probability = max(0.05, min(0.95, _safe_float(candidate.get("probability")) or (_safe_float(candidate.get("scout_score")) / 100.0)))
        target_move = 0.035 if candidate.get("asset_bucket") == "coin" else (0.025 if candidate.get("event_setup") else 0.018)
        by_id[forecast_id] = {
            "forecast_id": forecast_id,
            "created_at": _iso_now(),
            "created_at_ts": now_ts,
            "coin": coin,
            "direction": direction,
            "asset_bucket": candidate.get("asset_bucket", ""),
            "theme": candidate.get("theme", ""),
            "horizon_hours": horizon_hours,
            "entry_price": price,
            "target_move_pct": target_move,
            "probability": round(probability, 4),
            "resolved": False,
            "source": "morning_scout_book",
            "thesis_id": candidate.get("thesis_id", ""),
        }

    for forecast in by_id.values():
        if forecast.get("resolved"):
            continue
        created_ts = _safe_float(forecast.get("created_at_ts")) or _parse_ts(forecast.get("created_at"))
        if not created_ts or (now_ts - created_ts) < (_safe_float(forecast.get("horizon_hours"), horizon_hours) * 3600.0):
            continue
        coin = _safe_str(forecast.get("coin")).upper()
        current = _forecast_price_for_coin(coin, state)
        entry = _safe_float(forecast.get("entry_price"))
        if current <= 0 or entry <= 0:
            continue
        raw_return = (current - entry) / entry
        directional_return = raw_return if forecast.get("direction") == "LONG" else -raw_return
        outcome = 1.0 if directional_return >= _safe_float(forecast.get("target_move_pct")) else 0.0
        probability = _safe_float(forecast.get("probability"), 0.5)
        forecast.update({
            "resolved": True,
            "resolved_at": _iso_now(),
            "resolved_price": current,
            "directional_return_pct": round(directional_return * 100.0, 4),
            "outcome": int(outcome),
            "brier": round((probability - outcome) ** 2, 6),
        })

    all_rows = sorted(by_id.values(), key=lambda item: _parse_ts(item.get("created_at")), reverse=True)[:1000]
    open_rows = [row for row in all_rows if not row.get("resolved")]
    if len(open_rows) > max_open:
        open_keep = {row.get("forecast_id") for row in open_rows[:max_open]}
        all_rows = [row for row in all_rows if row.get("resolved") or row.get("forecast_id") in open_keep]
    try:
        _write_jsonl(path, all_rows)
    except Exception as exc:
        log.debug("forecast ledger write failed: %s", exc)
    resolved = [row for row in all_rows if row.get("resolved")]
    brier = sum(_safe_float(row.get("brier")) for row in resolved) / len(resolved) if resolved else 0.0
    hit_rate = sum(1 for row in resolved if int(row.get("outcome") or 0) == 1) / len(resolved) if resolved else 0.0
    return {
        "enabled": True,
        "path": str(path),
        "open_forecasts": open_rows[:20],
        "recent_resolved": resolved[:20],
        "summary": {
            "open_count": len(open_rows),
            "resolved_count": len(resolved),
            "brier_score": round(brier, 4),
            "hit_rate": round(hit_rate, 4),
            "calibration_note": "Collecting forecasts." if not resolved else f"{hit_rate * 100:.0f}% hit rate, Brier {brier:.3f}",
        },
    }


def build_and_save_report(
    *,
    state: dict,
    market_map: dict,
    config: Any = None,
    data_dir: Path | None = None,
) -> dict:
    if not _cfg_value(config, "proactive_trader_enabled", True):
        return {"enabled": False, "summary": {"reason": "proactive trader disabled"}}
    thesis_report = (
        build_thesis_ledger_report(state, market_map, config=config, data_dir=data_dir)
        if _cfg_value(config, "thesis_ledger_enabled", True)
        else {"enabled": False, "active_theses": [], "summary": {}}
    )
    read_through = (
        build_read_through_report(state, market_map, thesis_report)
        if _cfg_value(config, "read_through_engine_enabled", True)
        else {"enabled": False, "top_impacts": [], "summary": {}}
    )
    scout_book = (
        build_morning_scout_book(state, market_map, thesis_report, read_through, config=config)
        if _cfg_value(config, "morning_scout_book_enabled", True)
        else {"enabled": False, "bullish_calls": [], "bearish_calls": [], "starter_candidates": [], "summary": {}}
    )
    starter_basket = (
        build_starter_basket(state, scout_book, config=config)
        if _cfg_value(config, "starter_basket_optimizer_enabled", True)
        else {"enabled": False, "allocations": [], "summary": {}}
    )
    pair_trade_book = (
        build_pair_trade_book(state, scout_book, config=config)
        if _cfg_value(config, "pair_trade_book_enabled", True)
        else {"enabled": False, "pairs": [], "hedge_allocations": [], "summary": {}}
    )
    forecast_calibration = (
        build_forecast_calibration(state, scout_book, config=config, data_dir=data_dir)
        if _cfg_value(config, "forecast_calibration_enabled", True)
        else {"enabled": False, "summary": {}}
    )
    top_call = (scout_book.get("summary") or {}).get("top_call") or {}
    report = {
        "enabled": True,
        "updated_at": _iso_now(),
        "thesis_ledger": thesis_report,
        "morning_scout_book": scout_book,
        "read_through_engine": read_through,
        "starter_basket_optimizer": starter_basket,
        "pair_trade_book": pair_trade_book,
        "forecast_calibration": forecast_calibration,
        "summary": {
            "top_call": top_call,
            "active_thesis_count": (thesis_report.get("summary") or {}).get("active_count", 0),
            "starter_plan_count": (starter_basket.get("summary") or {}).get("allocation_count", 0),
            "pair_trade_count": (pair_trade_book.get("summary") or {}).get("pair_count", 0),
            "read_through_count": (read_through.get("summary") or {}).get("impact_count", 0),
            "forecast_note": (forecast_calibration.get("summary") or {}).get("calibration_note", ""),
        },
    }
    try:
        _write_json(_data_path(data_dir, PROACTIVE_TRADER_REPORT_JSON), report)
    except Exception as exc:
        log.debug("proactive report write failed: %s", exc)
    return report
