"""
dashboard/snapshot.py
Shared dashboard payload builder used by the local Flask UI and remote sync.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable


def default_state() -> dict:
    return {
        "status": "offline",
        "last_cycle": None,
        "cycle_number": 0,
        "portfolio_usd": 0,
        "available_usd": 0,
        "positions": [],
        "signals": {},
        "pending_orders": [],
        "sentiment": {},
        "mode": "unknown",
    }


def default_control() -> dict:
    return {
        "kill": {
            "active": False,
            "reason": "",
            "requested_at": None,
            "acknowledged_at": None,
        }
    }


def normalize_control(control: Any) -> dict:
    base = default_control()
    if not isinstance(control, dict):
        return base
    kill = control.get("kill")
    if isinstance(kill, dict):
        base["kill"].update({
            "active": bool(kill.get("active", False)),
            "reason": str(kill.get("reason", "") or ""),
            "requested_at": kill.get("requested_at"),
            "acknowledged_at": kill.get("acknowledged_at"),
        })
    if not base["kill"]["active"]:
        base["kill"]["acknowledged_at"] = None
    return base


def default_market_map() -> dict:
    return {
        "date": None,
        "updated_at": None,
        "global_notes": "",
        "coins": {},
    }


def _normalize_coin_list(values: Any) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        coin = str(value or "").upper().strip()
        if coin and coin not in seen:
            seen.add(coin)
            items.append(coin)
    return items


def _normalize_asset_category_values(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_values = value.replace("|", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    categories: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        category = str(raw or "").strip().lower()
        if category and category not in seen:
            seen.add(category)
            categories.append(category)
    return categories


def _normalize_asset_category_map(raw: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, value in dict(raw or {}).items():
        coin = str(key or "").upper().strip()
        if not coin:
            continue
        categories = _normalize_asset_category_values(value)
        if categories:
            out[coin] = categories
    return out


def _merge_asset_category_maps(*maps: Any) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for raw in maps:
        normalized = _normalize_asset_category_map(raw)
        for coin, categories in normalized.items():
            current = list(merged.get(coin) or [])
            seen = set(current)
            for category in categories:
                if category in seen:
                    continue
                current.append(category)
                seen.add(category)
            if current:
                merged[coin] = current
    return merged


def _runtime_dashboard_config_defaults() -> dict:
    try:
        from config import config as runtime_config
    except Exception:
        return {}
    trading = getattr(runtime_config, "trading", None)
    if trading is None:
        return {}
    return {
        "coins": _normalize_coin_list(getattr(trading, "coins", []) or []),
        "analysis_coins": _normalize_coin_list(getattr(trading, "analysis_coins", []) or []),
        "dynamic_analysis_coins": _normalize_coin_list(getattr(trading, "dynamic_analysis_coins", []) or []),
        "instrument_types": {
            str(key or "").upper(): str(value or "")
            for key, value in dict(getattr(trading, "instrument_types", {}) or {}).items()
            if str(key or "").strip()
        },
        "asset_categories": _normalize_asset_category_map(getattr(trading, "asset_category_map", {}) or {}),
        "asset_category_labels": {
            str(key or "").strip().lower(): str(value or "")
            for key, value in dict(getattr(trading, "asset_category_labels", {}) or {}).items()
            if str(key or "").strip()
        },
        "check_interval_seconds": int(getattr(trading, "check_interval_seconds", 120) or 120),
        "dynamic_market_cap_min_usd": float(getattr(trading, "dynamic_market_cap_min_usd", 0.0) or 0.0),
        "use_daily_market_map": bool(getattr(trading, "use_daily_market_map", True)),
    }


def _merge_dashboard_config(config: Any) -> dict:
    current = dict(config or {}) if isinstance(config, dict) else {}
    defaults = _runtime_dashboard_config_defaults()
    merged = dict(defaults)
    merged.update({
        key: value
        for key, value in current.items()
        if key not in {
            "coins",
            "analysis_coins",
            "dynamic_analysis_coins",
            "instrument_types",
            "asset_categories",
            "asset_category_labels",
        }
    })
    merged["coins"] = _normalize_coin_list((defaults.get("coins") or []) + (current.get("coins") or []))
    merged["analysis_coins"] = _normalize_coin_list(
        (defaults.get("analysis_coins") or []) + (current.get("analysis_coins") or [])
    )
    merged["dynamic_analysis_coins"] = _normalize_coin_list(
        current.get("dynamic_analysis_coins") or defaults.get("dynamic_analysis_coins") or []
    )
    merged["instrument_types"] = dict(defaults.get("instrument_types") or {})
    merged["instrument_types"].update({
        str(key or "").upper(): str(value or "")
        for key, value in dict(current.get("instrument_types") or {}).items()
        if str(key or "").strip()
    })
    merged["asset_categories"] = _merge_asset_category_maps(
        defaults.get("asset_categories") or {},
        current.get("asset_categories") or {},
    )
    merged["asset_category_labels"] = dict(defaults.get("asset_category_labels") or {})
    merged["asset_category_labels"].update({
        str(key or "").strip().lower(): str(value or "")
        for key, value in dict(current.get("asset_category_labels") or {}).items()
        if str(key or "").strip()
    })
    return merged


def normalize_market_map(market_map: Any) -> dict:
    base = default_market_map()
    if not isinstance(market_map, dict):
        return base
    base["date"] = market_map.get("date")
    base["updated_at"] = market_map.get("updated_at")
    base["global_notes"] = str(market_map.get("global_notes") or "")
    base["coins"] = dict(market_map.get("coins") or {})
    return base


def default_trade_reviews() -> dict:
    return {
        "updated_at": None,
        "reviews": {},
    }


def normalize_trade_reviews(trade_reviews: Any) -> dict:
    base = default_trade_reviews()
    if not isinstance(trade_reviews, dict):
        return base
    base["updated_at"] = trade_reviews.get("updated_at")
    base["reviews"] = dict(trade_reviews.get("reviews") or {})
    return base


def _record_index(records: Iterable[dict] | None) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in list(records or []):
        if not isinstance(record, dict):
            continue
        trade_id = str(record.get("trade_id") or "").strip()
        if trade_id:
            indexed[trade_id] = dict(record)
    return indexed


def _humanize_key(value: Any) -> str:
    return str(value or "").replace("_", " ").strip().lower()


def _humanize_exit_reason(reason: Any) -> str:
    mapping = {
        "take_profit": "target was reached",
        "stop_loss": "hard invalidation was hit",
        "conviction_lost": "conviction faded after entry",
        "signal_reversal": "signal reversed against the trade",
        "micro_invalidation": "micro invalidation triggered early",
        "structure_invalidation": "structure invalidation triggered",
        "htf_invalidation": "higher-timeframe invalidation triggered",
        "time_stop": "time stop cut the trade",
    }
    key = str(reason or "").strip().lower()
    return mapping.get(key, _humanize_key(key) or "no close logic recorded")


def _compact_sentences(parts: Iterable[str], limit: int = 3) -> str:
    seen = set()
    out: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return " • ".join(out)


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _pick_level(values: Any, *, prefer: str = "min", fallback: Any = None) -> float | None:
    numbers: list[float] = []
    for value in list(values or []):
        number = _safe_float(value)
        if number > 0:
            numbers.append(number)
    if not numbers and fallback is not None:
        fallback_number = _safe_float(fallback)
        if fallback_number > 0:
            numbers.append(fallback_number)
    if not numbers:
        return None
    return min(numbers) if prefer == "min" else max(numbers)


def _instrument_type_for_coin(coin: str, signal: dict | None, config: dict | None) -> str:
    signal = dict(signal or {})
    config = dict(config or {})
    signal_type = str(signal.get("instrument_type") or "").strip().lower()
    if signal_type:
        return signal_type
    instrument_types = dict(config.get("instrument_types") or {})
    return str(instrument_types.get(str(coin or "").upper(), "crypto") or "crypto").strip().lower()


def _asset_bucket(instrument_type: str) -> str:
    normalized = str(instrument_type or "crypto").strip().lower()
    return "coin" if normalized == "crypto" else "equity"


def _asset_categories_for_coin(coin: str, instrument_type: str, config: dict | None) -> list[str]:
    config = dict(config or {})
    category_map = dict(config.get("asset_categories") or {})
    key = str(coin or "").upper()
    categories = _normalize_asset_category_values(category_map.get(key))
    if categories:
        return categories
    normalized_type = str(instrument_type or "crypto").strip().lower()
    if normalized_type == "index":
        return ["indices_macro"]
    if normalized_type == "equity":
        return ["other_stocks"]
    return ["crypto"]


def _asset_category_for_coin(coin: str, instrument_type: str, config: dict | None) -> str:
    categories = _asset_categories_for_coin(coin, instrument_type, config)
    return categories[0] if categories else "other_stocks"


def _asset_category_label(category: str, config: dict | None) -> str:
    config = dict(config or {})
    labels = {
        "indices_macro": "Indices & Macro",
        "mag7": "Mag7",
        "semis_memory": "Semis & Memory",
        "neoclouds": "Neoclouds",
        "ai_infra": "AI Infra",
        "crypto_equities": "Crypto Equities",
        "asia_macro": "Asia Macro",
        "commodities_metals": "Metals",
        "energy": "Energy",
        "agriculture": "Agriculture",
        "fx_rates": "FX & Rates",
        "uranium": "Uranium",
        "volatility": "Volatility",
        "consumer": "Consumer",
        "financials": "Financials",
        "biotech_glp1": "Biotech & GLP-1",
        "meme_momentum": "Meme Momentum",
        "growth": "Growth",
        "other_stocks": "Other Stocks",
        "crypto": "Coins",
    }
    labels.update({str(key or "").strip().lower(): str(value or "") for key, value in dict(config.get("asset_category_labels") or {}).items()})
    return labels.get(str(category or "").strip().lower(), _humanize_key(category) or "Other")


def _group_action_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    labels = {
        "coin": "Coins",
        "equity": "Stocks & Indices",
    }
    for bucket in ("coin", "equity"):
        bucket_items = [item for item in items if str(item.get("asset_bucket") or "") == bucket]
        tradable_items = [item for item in bucket_items if item.get("tradable")]
        observation_items = [item for item in bucket_items if not item.get("tradable")]
        groups[bucket] = {
            "key": bucket,
            "label": labels[bucket],
            "items": bucket_items,
            "tradable_items": tradable_items,
            "observation_items": observation_items,
            "count": len(bucket_items),
            "tradable_count": len(tradable_items),
            "observation_count": len(observation_items),
            "active_count": sum(
                1 for item in tradable_items
                if str(item.get("status") or "") in {"OPEN_LONG", "OPEN_SHORT", "READY_LONG", "READY_SHORT"}
            ),
            "pending_count": sum(
                1 for item in bucket_items
                if str(item.get("status") or "").upper() == "PENDING_ENTRY"
            ),
        }
    return groups


def _primary_reason(text: Any) -> str:
    normalized = str(text or "").replace("|", "·")
    parts = [str(part).strip() for part in normalized.split("·")]
    preferred = [
        part
        for part in parts
        if part
        and not part.lower().startswith("score ")
        and not part.lower().startswith("map:")
        and not part.lower().startswith("breakout state:")
        and not part.lower().startswith("key levels:")
    ]
    if preferred:
        return preferred[0]
    return parts[0] if parts else ""


def _setup_direction(status: str, action: str, bias: str) -> str:
    status_upper = str(status or "").upper()
    action_upper = str(action or "").upper()
    bias_upper = str(bias or "").upper()
    if "SHORT" in status_upper or "BREAKDOWN" in status_upper or action_upper == "SHORT" or bias_upper == "BEARISH":
        return "short"
    if "LONG" in status_upper or "RECLAIM" in status_upper or action_upper == "LONG" or bias_upper == "BULLISH":
        return "long"
    return "trade"


def _reason_clauses(text: Any) -> list[str]:
    normalized = str(text or "").replace("|", "·")
    return [str(part).strip() for part in normalized.split("·") if str(part).strip()]


def _directional_reason_score(clause: str, direction: str) -> int:
    text = str(clause or "").strip().lower()
    if not text:
        return -99

    score = 0
    if direction == "short":
        if "short blocked" in text:
            score += 14
        if "breakdown" in text or "break support" in text:
            score += 10
        if "bullish breakout" in text or "breaking above key resistance" in text:
            score += 9
        if "support" in text or "demand" in text:
            score += 7
        if "resistance" in text or "supply" in text:
            score += 5
        if "needs" in text and "for short" in text:
            score += 8
        if "needs" in text and "for long" in text:
            score -= 10
    elif direction == "long":
        if "long blocked" in text:
            score += 14
        if "reclaim" in text or "breakout" in text:
            score += 10
        if "bearish breakout" in text or "breaking below key support" in text:
            score += 9
        if "resistance" in text or "supply" in text:
            score += 7
        if "support" in text or "demand" in text:
            score += 5
        if "needs" in text and "for long" in text:
            score += 8
        if "needs" in text and "for short" in text:
            score -= 10

    generic_markers = (
        "absorption",
        "indecision",
        "trend flat",
        "ranging",
        "macro confirmation",
        "earnings confirmation",
        "doji",
        "inside bar",
        "memory:",
        "candles:",
        "regime:",
    )
    if any(marker in text for marker in generic_markers):
        score += 4

    return score


def _relevant_blocker_reason(text: Any, direction: str) -> str:
    clauses = _reason_clauses(text)
    if not clauses:
        return ""

    ranked = sorted(
        (
            (_directional_reason_score(clause, direction), -index, clause)
            for index, clause in enumerate(clauses)
        ),
        reverse=True,
    )
    if ranked and ranked[0][0] > 0:
        return ranked[0][2]

    for clause in clauses:
        lowered = clause.lower()
        if "needs" in lowered and (
            ("for long" in lowered and direction == "short")
            or ("for short" in lowered and direction == "long")
        ):
            continue
        return clause
    return ""


def _normalize_directional_blocker_reason(clause: str, direction: str) -> str:
    text = str(clause or "").strip()
    lowered = text.lower()
    if direction == "short" and lowered.startswith("long blocked") and (
        "breaking down" in lowered or "bearish breakout" in lowered
    ):
        remainder = text.split("—", 1)[-1].strip() if "—" in text else text
        return f"Breakdown already active — {remainder}"
    if direction == "long" and lowered.startswith("short blocked") and (
        "breaking above" in lowered or "bullish breakout" in lowered
    ):
        remainder = text.split("—", 1)[-1].strip() if "—" in text else text
        return f"Reclaim already active — {remainder}"
    return text


def _next_setup_reason(
    *,
    status: str,
    action: str,
    bias: str,
    entry_status: str,
    trigger: str,
    blocker: str,
    execution_note: str,
) -> str:
    status_upper = str(status or "").upper()
    direction = _setup_direction(status, action, bias)
    primary_blocker = _relevant_blocker_reason(blocker or execution_note, direction) or _primary_reason(blocker or execution_note)
    primary_blocker = _normalize_directional_blocker_reason(primary_blocker, direction)
    context = str(entry_status or trigger or "").strip()
    note = str(primary_blocker or execution_note or "").strip()

    if status_upper.startswith("OPEN_"):
        return _compact_sentences([
            f"Open {direction}: {context}" if context else f"Open {direction}.",
            execution_note,
        ], limit=2)
    if status_upper == "PENDING_ENTRY":
        return _compact_sentences([
            f"Order working: {context}" if context else "Order working.",
            note,
        ], limit=2)
    if status_upper in {"READY_LONG", "READY_SHORT"}:
        return _compact_sentences([
            f"Ready {direction}: {context}" if context else f"Ready {direction}.",
            note or "Final sizing and fill-quality checks still need to clear.",
        ], limit=2)
    if status_upper in {"WAIT_BREAKDOWN", "WAIT_RECLAIM", "WATCH_SHORT", "WATCH_LONG"}:
        return _compact_sentences([
            f"Not {direction} yet: {context}" if context else f"Not {direction} yet.",
            note or "The setup still needs confirmation before the bot can trade.",
        ], limit=2)
    if status_upper in {"EXECUTION_BLOCKED", "DATA_QUALITY_HOLD", "PORTFOLIO_GUARD", "RISK_BLOCKED"}:
        return _compact_sentences([
            f"Blocked: {context}" if context else "Blocked.",
            note,
        ], limit=2)
    if status_upper in {"ARMED", "WAITING_CONFIRMATION", "PASSIVE_ENTRY", "EXECUTABLE"}:
        return _compact_sentences([
            f"Waiting: {context}" if context else "Waiting for confirmation.",
            note,
        ], limit=2)
    return _compact_sentences([
        f"No trade: {note}" if note else "No trade: thesis is incomplete.",
        context,
    ], limit=2)


def _map_blurb(text: Any) -> str:
    summary = str(text or "").strip()
    if not summary:
        return ""
    replacements = {
        "auto bullish map": "bullish daily view",
        "auto bearish map": "bearish daily view",
        "auto neutral map": "neutral daily view",
        "daily reclaim confirmed": "reclaim is confirmed",
        "daily reclaim is holding": "reclaim is holding",
        "daily reclaim was confirmed, but live price slipped back below reclaim": "reclaim was confirmed, but price slipped back below the trigger",
        "daily breakdown confirmed": "breakdown is still active",
        "daily breakdown is holding": "breakdown is holding",
        "daily breakdown was confirmed, but live price bounced back above breakdown": "breakdown was confirmed, but price bounced back above the trigger",
        "price is sitting in mapped demand": "price is in demand",
        "price is sitting in mapped supply": "price is in supply",
        "price is pressing mapped resistance": "price is at resistance",
        "price is testing mapped support": "price is testing support",
    }
    cleaned = summary
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def _level_move(current_price: float, level: float) -> str:
    if current_price <= 0 or level <= 0:
        return ""
    delta = level - current_price
    pct = (delta / current_price) * 100.0 if current_price else 0.0
    return f"{delta:+,.2f} / {pct:+.2f}%"


def _level_gap(current_price: float, level: float) -> str:
    if current_price <= 0 or level <= 0:
        return ""
    delta = current_price - level
    pct = (delta / level) * 100.0 if level else 0.0
    return f"{delta:+,.2f} ({pct:+.2f}%)"


def _numeric_level_text(label: str, level: float | None, current_price: float = 0.0) -> str:
    safe_level = _safe_float(level)
    if safe_level <= 0:
        return ""
    move = _level_move(current_price, safe_level)
    return f"{label} {safe_level:,.2f}" + (f" ({move})" if move else "")


def _entry_status_text(current_price: float, level: float | None, label: str) -> str:
    safe_level = _safe_float(level)
    if current_price <= 0 or safe_level <= 0:
        return ""
    return f"Live {current_price:,.2f}; {label} {safe_level:,.2f}; gap {_level_gap(current_price, safe_level)}"


def _stop_target_text(current_price: float, stop_price: float | None, target_price: float | None) -> str:
    parts: list[str] = []
    stop_text = _numeric_level_text("SL", stop_price, current_price)
    target_text = _numeric_level_text("TP", target_price, current_price)
    if stop_text:
        parts.append(stop_text)
    if target_text:
        parts.append(target_text)
    return " • ".join(parts)


def _clip_text(text: Any, limit: int = 120) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[: max(limit - 1, 0)].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,.;:-") + "…"


def _contains_any(text: Any, keywords: Iterable[str]) -> bool:
    haystack = str(text or "").strip().lower()
    return any(str(keyword or "").strip().lower() in haystack for keyword in keywords if str(keyword or "").strip())


def _friction_item(key: str, label: str, status: str, summary: Any) -> dict[str, str]:
    normalized_status = str(status or "wait").strip().lower()
    if normalized_status not in {"clear", "wait", "block"}:
        normalized_status = "wait"
    return {
        "key": str(key or "").strip().lower(),
        "label": str(label or "").strip() or _humanize_key(key).title(),
        "status": normalized_status,
        "summary": _clip_text(summary, 120),
    }


def _flow_state_summary(sig: dict, direction: str) -> tuple[str, str]:
    orderbook_summary = str(
        sig.get("orderbook_summary")
        or sig.get("orderbook_context")
        or ""
    ).strip()
    interaction = str(sig.get("orderbook_interaction") or "").strip().upper()
    breakout_states = {
        str(sig.get("orderbook_breakout_state") or "").strip().upper(),
        str(sig.get("orderbook_intracycle_breakout_state") or "").strip().upper(),
    }
    bullish_states = {
        "PROBING_BULLISH_BREAKOUT",
        "CONFIRMED_BULLISH_BREAKOUT",
        "PERSISTENT_BULLISH_BREAKOUT",
    }
    bearish_states = {
        "PROBING_BEARISH_BREAKDOWN",
        "CONFIRMED_BEARISH_BREAKDOWN",
        "PERSISTENT_BEARISH_BREAKDOWN",
    }
    supportive = (
        direction == "long" and bool(breakout_states & bullish_states)
    ) or (
        direction == "short" and bool(breakout_states & bearish_states)
    )
    opposing = (
        direction == "long" and bool(breakout_states & bearish_states)
    ) or (
        direction == "short" and bool(breakout_states & bullish_states)
    )
    if orderbook_summary:
        summary = _primary_reason(orderbook_summary)
    elif supportive:
        summary = "directional breakout pressure is still live"
    elif opposing:
        summary = "opposite-side breakout pressure is getting in the way"
    elif interaction and interaction != "BETWEEN_LEVELS":
        summary = "order-flow: " + _humanize_key(interaction)
    else:
        summary = "order-flow still needs cleaner confirmation"

    if opposing:
        return "block", summary
    if supportive:
        return "clear", summary
    if _contains_any(summary, {"absorption", "between levels", "no clear direction", "mixed"}):
        return "wait", summary
    return ("wait" if orderbook_summary or interaction else "wait"), summary


def _reliability_state_summary(sig: dict, asset_state: str) -> tuple[str, str]:
    quality = str(sig.get("data_reliability_quality") or "").strip().upper()
    summary = str(sig.get("data_reliability_summary") or "").strip()
    if asset_state == "DATA_QUALITY_HOLD":
        return "block", _primary_reason(summary or "data quality is explicitly blocking this setup")
    if not summary and quality in {"", "UNKNOWN"}:
        return "clear", "no data-quality issue is currently flagged"
    text = summary or quality or "data quality is being monitored"
    if quality in {"HIGH", "STRONG"} or _contains_any(text, {"strong enough", "trust the setup", "reliable", "clean"}):
        return "clear", _primary_reason(text)
    if _contains_any(text, {"missing", "stale", "insufficient", "not ready", "not enough"}):
        return "block", _primary_reason(text)
    if quality in {"LOW", "WEAK"}:
        return "block", _primary_reason(text)
    if _contains_any(text, {"thin", "drifted", "partial", "lag", "waiting", "limited"}):
        return "wait", _primary_reason(text)
    return "wait", _primary_reason(text)


def _execution_state_summary(
    *,
    sig: dict,
    status: str,
    asset_state: str,
    tradable: bool,
    coach_verdict: str,
    execution_note: str,
    mode_detail: str,
) -> tuple[str, str]:
    status_upper = str(status or "").upper()
    coach_key = str(coach_verdict or "").strip().upper()
    quality_score = _safe_float(sig.get("execution_quality_score"))
    summary = str(
        sig.get("execution_coach_summary")
        or sig.get("execution_quality_summary")
        or execution_note
        or mode_detail
        or ""
    ).strip()
    if not tradable or asset_state == "EXECUTION_BLOCKED" or coach_key == "BLOCK":
        return "block", _primary_reason(summary or "execution lane is not clear enough yet")
    if status_upper.startswith("OPEN_") or status_upper == "PENDING_ENTRY" or coach_key == "GO":
        return "clear", _primary_reason(summary or "execution lane is already active")
    if quality_score >= 75:
        return "clear", _primary_reason(summary or "execution quality is clean")
    if quality_score and quality_score <= 55:
        return "wait", _primary_reason(summary or "execution quality is still messy")
    return "wait", _primary_reason(summary or "execution still needs cleaner tape")


def _structure_state_summary(
    *,
    sig: dict,
    status: str,
    bias: str,
    headline: str,
    map_summary: str,
) -> tuple[str, str]:
    status_upper = str(status or "").upper()
    thesis_quality = str(sig.get("thesis_quality") or "").strip().upper()
    thesis_summary = str(sig.get("thesis_summary") or "").strip()
    blockers = str(sig.get("thesis_blockers") or "").strip()
    text = thesis_summary or blockers or map_summary or headline or "structure is still forming"
    if _contains_any(text, {"invalid", "broken", "conflict", "misaligned", "not aligned"}):
        return "block", _primary_reason(text)
    if status_upper.startswith("OPEN_") or status_upper in {"READY_LONG", "READY_SHORT", "PENDING_ENTRY"}:
        return "clear", _primary_reason(text)
    if thesis_quality in {"HIGH", "STRONG"}:
        return "clear", _primary_reason(text)
    if status_upper in {"WAIT_RECLAIM", "WAIT_BREAKDOWN", "WATCH_LONG", "WATCH_SHORT", "WAITING_CONFIRMATION", "ARMED"}:
        return "wait", _primary_reason(text)
    if thesis_quality in {"LOW", "WEAK"} or status_upper in {"NO_SETUP", "RISK_BLOCKED"}:
        return "block", _primary_reason(text or f"{_humanize_key(bias)} structure is not ready")
    return "wait", _primary_reason(text)


def _build_friction_stack(
    *,
    sig: dict,
    status: str,
    action: str,
    bias: str,
    asset_state: str,
    tradable: bool,
    coach_verdict: str,
    headline: str,
    map_summary: str,
    execution_note: str,
    mode_detail: str,
) -> list[dict[str, str]]:
    direction = _setup_direction(status, action, bias)
    structure_status, structure_summary = _structure_state_summary(
        sig=sig,
        status=status,
        bias=bias,
        headline=headline,
        map_summary=map_summary,
    )
    flow_status, flow_summary = _flow_state_summary(sig, direction)
    reliability_status, reliability_summary = _reliability_state_summary(sig, asset_state)
    execution_status, execution_summary = _execution_state_summary(
        sig=sig,
        status=status,
        asset_state=asset_state,
        tradable=tradable,
        coach_verdict=coach_verdict,
        execution_note=execution_note,
        mode_detail=mode_detail,
    )
    return [
        _friction_item("structure", "Structure", structure_status, structure_summary),
        _friction_item("flow", "Flow", flow_status, flow_summary),
        _friction_item("data", "Data", reliability_status, reliability_summary),
        _friction_item("execution", "Execution", execution_status, execution_summary),
    ]


def _event_time_blurb(minutes_to_event: Any) -> str:
    minutes = _safe_int(minutes_to_event)
    if minutes <= 0:
        return "now"
    if minutes < 60:
        return f"in {minutes}m"
    hours, remainder = divmod(minutes, 60)
    if hours < 24:
        return f"in {hours}h" + (f" {remainder}m" if remainder else "")
    days, rem_hours = divmod(hours, 24)
    return f"in {days}d" + (f" {rem_hours}h" if rem_hours else "")


def _build_catalyst_rail(sig: dict) -> list[dict[str, str]]:
    rail: list[dict[str, str]] = []
    catalyst_score = _safe_float(sig.get("news_catalyst_score"))
    catalyst_summary = str(sig.get("news_catalyst_summary") or "").strip()
    news_headline = str(sig.get("news_headline") or "").strip()
    narrative_summary = str(sig.get("narrative_summary") or "").strip()
    event_name = str(sig.get("narrative_event_name") or "").strip()
    event_risk_active = bool(sig.get("narrative_event_risk_active"))
    if catalyst_score >= 3.0 or catalyst_summary:
        catalyst_text = catalyst_summary or news_headline or "fresh catalyst support is still present"
        label = "Catalyst"
        if catalyst_score >= 3.0:
            catalyst_text = "Major catalyst: " + catalyst_text
        elif catalyst_score >= 2.0:
            catalyst_text = "Fresh catalyst: " + catalyst_text
        rail.append({
            "label": label,
            "tone": "support" if catalyst_score >= 2.0 else "watch",
            "text": _clip_text(catalyst_text, 120),
        })
    if event_name or event_risk_active:
        event_parts = [event_name or "Event risk"]
        timing = _event_time_blurb(sig.get("narrative_minutes_to_event"))
        if timing:
            event_parts.append(timing)
        rail.append({
            "label": "Event",
            "tone": "risk" if event_risk_active else "watch",
            "text": _clip_text(" ".join(event_parts), 90),
        })
    narrative_text = narrative_summary or news_headline
    if narrative_text and not _contains_any(
        narrative_text,
        {"neutral", "no clear catalyst", "no strong narrative", "no catalyst", "flat narrative"},
    ):
        existing_texts = {str(item.get("text") or "").lower() for item in rail}
        clipped = _clip_text(_primary_reason(narrative_text), 110)
        if clipped and clipped.lower() not in existing_texts:
            rail.append({
                "label": "Narrative" if narrative_summary else "Headline",
                "tone": "watch",
                "text": clipped,
            })
    return rail[:3]


def _remaining_gate_blurb(status: str, sig: dict, direction: str) -> str:
    status_upper = str(status or "").upper()
    if status_upper == "WAIT_RECLAIM" or (
        status_upper == "WATCH_LONG" and bool(sig.get("market_map_reclaim_confirmed")) and not bool(sig.get("market_map_live_reclaim"))
    ):
        return "only the reclaim retake remains"
    if status_upper == "WAIT_BREAKDOWN" or (
        status_upper == "WATCH_SHORT" and bool(sig.get("market_map_breakdown_confirmed")) and not bool(sig.get("market_map_live_breakdown"))
    ):
        return "only the breakdown confirmation remains"
    if status_upper in {"WATCH_LONG", "WATCH_SHORT", "WAITING_CONFIRMATION", "ARMED", "PASSIVE_ENTRY", "EXECUTABLE"}:
        return "only the final confirmation remains"
    if status_upper in {"READY_LONG", "READY_SHORT"}:
        return "only sizing and fill checks remain"
    if status_upper == "PENDING_ENTRY":
        return "the order is already working"
    if status_upper.startswith("OPEN_"):
        return "the trade is already live"
    if direction == "long":
        return "the long still needs cleaner confirmation"
    if direction == "short":
        return "the short still needs cleaner confirmation"
    return "the setup still needs one final unlock"


def _build_why_this_lead(
    *,
    sig: dict,
    status: str,
    action: str,
    bias: str,
    probability: dict[str, Any],
    friction_stack: list[dict[str, str]],
    catalyst_rail: list[dict[str, str]],
) -> str:
    direction = _setup_direction(status, action, bias)
    reasons: list[str] = []
    probability_pct = _safe_int(probability.get("probability_pct"))
    probability_label = str(probability.get("probability_label") or "").strip().lower()
    if probability_pct > 0:
        lead_label = probability_label or "calibrated odds"
        reasons.append(f"{lead_label} {probability_pct}%")

    if any(str(item.get("tone") or "") == "support" for item in catalyst_rail):
        reasons.append("catalyst is still live")

    analog_count = _safe_int(sig.get("analog_count"))
    analog_win_rate = _safe_float(sig.get("analog_win_rate"))
    if analog_count >= 3 and analog_win_rate >= 0.55:
        reasons.append(f"{analog_count} analogs support it")

    clear_count = sum(1 for item in friction_stack if str(item.get("status") or "") == "clear")
    block_count = sum(1 for item in friction_stack if str(item.get("status") or "") == "block")
    if clear_count >= 3 and block_count == 0:
        reasons.append("friction stack is mostly clean")
    elif clear_count >= 2 and block_count <= 1:
        reasons.append("most checks are already aligned")

    reasons.append(_remaining_gate_blurb(status, sig, direction))

    fallback = (
        _primary_reason(sig.get("expectancy_summary"))
        or _primary_reason(sig.get("analog_summary"))
        or _primary_reason(sig.get("thesis_summary"))
        or _remaining_gate_blurb(status, sig, direction)
    )
    return _compact_sentences(reasons, limit=3) or fallback


def _fallback_probability(score: float, confidence: str, conviction_score: float = 0.0) -> float:
    anchor = conviction_score if conviction_score > 0 else score
    confidence_adjustments = {
        "LOW": -0.05,
        "MEDIUM": 0.0,
        "HIGH": 0.06,
    }
    probability = 0.50 + ((_clamp(anchor, 0.0, 100.0) - 50.0) / 50.0) * 0.18
    probability += confidence_adjustments.get(str(confidence or "MEDIUM").upper(), 0.0)
    return _clamp(probability, 0.18, 0.82)


def _watch_calibration_distance_band(current_price: float, level: float | None, *, direction: str) -> str:
    safe_level = _safe_float(level)
    if current_price <= 0 or safe_level <= 0:
        return "unknown"
    if direction == "below":
        distance_pct = max(0.0, (current_price - safe_level) / safe_level)
    else:
        distance_pct = max(0.0, (safe_level - current_price) / safe_level)
    if distance_pct <= 0.0025:
        return "tight"
    if distance_pct <= 0.0075:
        return "near"
    if distance_pct <= 0.015:
        return "stretch"
    return "far"


def _watch_calibration_candidate_state(asset_state: str, stage: str) -> bool:
    return asset_state in {
        "ARMED",
        "WAITING_CONFIRMATION",
        "EXECUTION_BLOCKED",
        "PENDING_ENTRY",
        "MAJOR_CATALYST_WATCH",
        "PASSIVE_ENTRY",
        "DATA_QUALITY_HOLD",
    } or stage in {
        "SIGNAL_STREAK_WAIT",
        "EXECUTION_QUALITY_BLOCK",
        "EXECUTION_COACH_SKIP",
        "PRECISION_CADENCE_BLOCK",
        "MAJOR_CATALYST_WATCH",
        "OBSERVATION_ONLY",
        "GUARDRAILS_FLAT",
        "LIMIT_ENTRY_PLACED",
    }


def _watch_calibration_text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def _watch_calibration_reclaim_success(row: dict) -> bool:
    snap = dict(row.get("signal_snapshot") or {})
    final_action = str(row.get("final_action") or snap.get("action") or "").upper()
    return bool(
        snap.get("market_map_live_reclaim")
        or (snap.get("market_map_reclaim_confirmed") and not snap.get("market_map_reclaim_lost"))
        or ((final_action == "LONG" or row.get("executed") or row.get("pending_limit")) and not row.get("blocked"))
    )


def _watch_calibration_breakdown_success(row: dict) -> bool:
    snap = dict(row.get("signal_snapshot") or {})
    final_action = str(row.get("final_action") or snap.get("action") or "").upper()
    return bool(
        snap.get("market_map_live_breakdown")
        or (snap.get("market_map_breakdown_confirmed") and not snap.get("market_map_breakdown_lost"))
        or ((final_action == "SHORT" or row.get("executed") or row.get("pending_limit")) and not row.get("blocked"))
    )


def _watch_calibration_event_from_record(record: dict) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return None

    snap = dict(record.get("signal_snapshot") or {})
    candidate_action = str(
        record.get("candidate_action")
        or snap.get("thesis_candidate_action")
        or snap.get("action")
        or ""
    ).upper()
    final_action = str(record.get("final_action") or snap.get("action") or "").upper()
    asset_state = str(record.get("asset_state") or snap.get("asset_state") or "").upper()
    stage = str(record.get("stage") or "").upper()
    bias = str(snap.get("market_map_bias") or "").upper()
    text = _watch_calibration_text(
        record.get("decision_reason"),
        record.get("stage"),
        record.get("asset_state"),
        snap.get("flat_reason"),
        snap.get("next_unblock_reason"),
    )
    if not _watch_calibration_candidate_state(asset_state, stage):
        return None

    if candidate_action == "LONG" and bias == "BULLISH":
        reclaim_confirmed = bool(snap.get("market_map_reclaim_confirmed"))
        live_reclaim = bool(snap.get("market_map_live_reclaim"))
        if live_reclaim or (final_action == "LONG" and not bool(record.get("blocked"))):
            return None
        if not (reclaim_confirmed or bool(snap.get("market_map_block_longs")) or "reclaim" in text):
            return None
        long_trigger = _pick_level(
            snap.get("daily_close_long_above"),
            prefer="min",
            fallback=snap.get("market_map_nearest_resistance") or snap.get("daily_breakout_level"),
        )
        live_anchor = _safe_float(snap.get("live_price") or snap.get("price") or snap.get("analysis_price"))
        instrument_type = str(snap.get("instrument_type") or "crypto").strip().lower() or "crypto"
        return {
            "scenario": "reclaim_retake" if reclaim_confirmed else "reclaim_initial",
            "asset_bucket": _asset_bucket(instrument_type),
            "major_catalyst": _safe_float(snap.get("news_catalyst_score")) >= 3.0,
            "distance_band": _watch_calibration_distance_band(live_anchor, long_trigger, direction="above"),
        }

    if candidate_action == "SHORT" and bias == "BEARISH":
        breakdown_confirmed = bool(snap.get("market_map_breakdown_confirmed"))
        live_breakdown = bool(snap.get("market_map_live_breakdown"))
        if live_breakdown or (final_action == "SHORT" and not bool(record.get("blocked"))):
            return None
        if not (breakdown_confirmed or bool(snap.get("market_map_block_shorts")) or "breakdown" in text or "break support" in text):
            return None
        short_trigger = _pick_level(
            snap.get("daily_close_short_below"),
            prefer="max",
            fallback=snap.get("market_map_nearest_support") or snap.get("daily_breakdown_level"),
        )
        live_anchor = _safe_float(snap.get("live_price") or snap.get("price") or snap.get("analysis_price"))
        instrument_type = str(snap.get("instrument_type") or "crypto").strip().lower() or "crypto"
        return {
            "scenario": "breakdown_retake" if breakdown_confirmed else "breakdown_initial",
            "asset_bucket": _asset_bucket(instrument_type),
            "major_catalyst": _safe_float(snap.get("news_catalyst_score")) >= 3.0,
            "distance_band": _watch_calibration_distance_band(live_anchor, short_trigger, direction="below"),
        }

    return None


def _watch_calibration_future_success(coin_rows: list[dict], start_index: int, scenario: str, *, horizon: int = 24) -> bool:
    origin_cycle = _safe_int(coin_rows[start_index].get("cycle_number"))
    seen = 0
    for row in coin_rows[start_index + 1:]:
        if _safe_int(row.get("cycle_number")) <= origin_cycle:
            continue
        seen += 1
        if scenario.startswith("reclaim"):
            if _watch_calibration_reclaim_success(row):
                return True
            if _watch_calibration_breakdown_success(row):
                return False
        else:
            if _watch_calibration_breakdown_success(row):
                return True
            if _watch_calibration_reclaim_success(row):
                return False
        if seen >= horizon:
            break
    return False


def _watch_calibration_bucket_keys(event: dict[str, Any]) -> list[tuple[Any, ...]]:
    catalyst_band = "major" if event.get("major_catalyst") else "normal"
    return [
        (
            "scenario_bucket_catalyst_distance",
            event.get("scenario"),
            event.get("asset_bucket"),
            catalyst_band,
            event.get("distance_band"),
        ),
        (
            "scenario_bucket_catalyst",
            event.get("scenario"),
            event.get("asset_bucket"),
            catalyst_band,
        ),
        ("scenario_bucket", event.get("scenario"), event.get("asset_bucket")),
        ("scenario", event.get("scenario")),
    ]


def _build_watch_probability_calibration(records: Iterable[dict] | None) -> dict[tuple[Any, ...], dict[str, Any]]:
    calibration: dict[tuple[Any, ...], dict[str, Any]] = {}
    by_coin: dict[str, list[dict]] = defaultdict(list)
    for record in list(records or []):
        if not isinstance(record, dict):
            continue
        coin = str(record.get("coin") or "").upper().strip()
        if not coin:
            continue
        by_coin[coin].append(dict(record))

    for coin_rows in by_coin.values():
        coin_rows.sort(
            key=lambda row: (
                _safe_int(row.get("cycle_number")),
                _safe_float(row.get("recorded_at_ts")),
            )
        )
        index = 0
        while index < len(coin_rows):
            row = coin_rows[index]
            event = _watch_calibration_event_from_record(row)
            if not event:
                index += 1
                continue
            success = _watch_calibration_future_success(coin_rows, index, str(event.get("scenario") or ""))
            for key in _watch_calibration_bucket_keys(event):
                bucket = calibration.setdefault(key, {"count": 0, "success": 0})
                bucket["count"] += 1
                bucket["success"] += 1 if success else 0
            index += 1
            while index < len(coin_rows):
                next_event = _watch_calibration_event_from_record(coin_rows[index])
                if not next_event:
                    break
                if (
                    str(next_event.get("scenario") or "") != str(event.get("scenario") or "")
                    or str(next_event.get("asset_bucket") or "") != str(event.get("asset_bucket") or "")
                ):
                    break
                index += 1

    for key, bucket in calibration.items():
        count = int(bucket.get("count") or 0)
        success = int(bucket.get("success") or 0)
        bucket["rate"] = (success / count) if count > 0 else 0.0
        bucket["key"] = key
    return calibration


def _pick_watch_probability_calibration(
    calibration: dict[tuple[Any, ...], dict[str, Any]] | None,
    *,
    scenario: str,
    asset_bucket: str,
    major_catalyst: bool,
    distance_band: str,
) -> dict[str, Any] | None:
    if not calibration:
        return None
    catalyst_band = "major" if major_catalyst else "normal"
    lookup = [
        (("scenario_bucket_catalyst_distance", scenario, asset_bucket, catalyst_band, distance_band), 10),
        (("scenario_bucket_catalyst", scenario, asset_bucket, catalyst_band), 12),
        (("scenario_bucket", scenario, asset_bucket), 18),
        (("scenario", scenario), 28),
    ]
    fallback: dict[str, Any] | None = None
    for key, min_count in lookup:
        bucket = calibration.get(key)
        if not bucket:
            continue
        if int(bucket.get("count") or 0) >= min_count:
            return bucket
        if fallback is None or int(bucket.get("count") or 0) > int(fallback.get("count") or 0):
            fallback = bucket
    if fallback and int(fallback.get("count") or 0) >= 5:
        return fallback
    return None


def _calibrate_watch_probability(
    *,
    calibration: dict[tuple[Any, ...], dict[str, Any]] | None,
    probability: float,
    scenario: str,
    asset_bucket: str,
    major_catalyst: bool,
    distance_band: str,
) -> dict[str, Any] | None:
    bucket = _pick_watch_probability_calibration(
        calibration,
        scenario=scenario,
        asset_bucket=asset_bucket,
        major_catalyst=major_catalyst,
        distance_band=distance_band,
    )
    if not bucket:
        return None

    samples = int(bucket.get("count") or 0)
    empirical_rate = _clamp(_safe_float(bucket.get("rate")), 0.05, 0.95)
    prior_weight = 14.0
    if samples < 10:
        prior_weight += 10.0
    elif samples < 20:
        prior_weight += 6.0
    elif samples < 35:
        prior_weight += 3.0
    if major_catalyst:
        prior_weight += 2.0
    calibrated_probability = ((empirical_rate * samples) + (probability * prior_weight)) / (samples + prior_weight)
    noun = "reclaim watches" if scenario.startswith("reclaim") else "breakdown watches"
    return {
        "probability": _clamp(calibrated_probability, 0.05, 0.95),
        "empirical_rate": empirical_rate,
        "samples": samples,
        "history_note": f"history {int(round(empirical_rate * 100.0))}% across {samples} similar {noun}",
    }


def _setup_probability(
    *,
    sig: dict[str, Any],
    status: str,
    action: str,
    confidence: str,
    score: float,
    conviction_score: float,
    live_anchor: float,
    long_trigger: float | None,
    short_trigger: float | None,
    reclaim_confirmed: bool,
    live_reclaim: bool,
    reclaim_lost: bool,
    bullish_breakout_live: bool,
    bias: str,
    tradable: bool,
    coach_verdict: str,
    asset_bucket: str,
    asset_state: str,
    calibration_model: dict[tuple[Any, ...], dict[str, Any]] | None,
) -> dict[str, Any]:
    expectancy_probability = _safe_float(sig.get("expectancy_probability"))
    uncertainty = _safe_float(sig.get("expectancy_uncertainty"))
    analog_adjustment = _clamp(_safe_float(sig.get("analog_probability_adjustment")), -0.08, 0.08)
    catalyst_score = _safe_float(sig.get("news_catalyst_score"))
    referee = dict(sig.get("llm_referee") or {})
    referee_verdict = str(referee.get("verdict") or sig.get("llm_referee_verdict") or "").upper()
    execution_quality_score = _safe_float(sig.get("execution_quality_score"))
    status_key = str(status or "").upper()
    action_key = str(action or "").upper()
    coach_key = str(coach_verdict or "").upper()
    base_probability = (
        expectancy_probability
        if 0.0 < expectancy_probability <= 1.0
        else _fallback_probability(score, confidence, conviction_score)
    )
    probability = base_probability
    probability_source = "heuristic"
    empirical_rate: float | None = None
    empirical_samples = 0
    label = "Setup odds"
    reasons: list[str] = []

    if uncertainty > 0:
        if uncertainty <= 0.22:
            probability += 0.02
            reasons.append("low model uncertainty")
        elif uncertainty >= 0.40:
            probability -= 0.04
            reasons.append("high model uncertainty")

    if analog_adjustment:
        probability += analog_adjustment
        reasons.append("historical analogs are supportive" if analog_adjustment > 0 else "historical analogs are cautious")

    if referee_verdict == "SUPPORT":
        probability += 0.03
        reasons.append("referee supports the setup")
    elif referee_verdict == "WAIT":
        probability -= 0.02
        reasons.append("referee still wants cleaner confirmation")
    elif referee_verdict == "BLOCK":
        probability -= 0.05
        reasons.append("referee is blocking the setup")

    if coach_key == "BLOCK":
        probability -= 0.06
        reasons.append("execution coach is blocking here")
    elif coach_key == "GO":
        probability += 0.03
        reasons.append("execution coach likes the tape")

    if execution_quality_score > 0:
        if execution_quality_score >= 75.0:
            probability += 0.03
            reasons.append("execution quality is clean")
        elif execution_quality_score <= 55.0:
            probability -= 0.03
            reasons.append("execution quality is messy")

    if catalyst_score >= 3.0:
        probability += 0.04
        reasons.append("major catalyst still supports the move")
    elif catalyst_score >= 2.0:
        probability += 0.02
        reasons.append("fresh catalyst support is still there")

    waiting_reclaim = status_key == "WAIT_RECLAIM" or (
        status_key == "WATCH_LONG" and long_trigger and not live_reclaim and (reclaim_confirmed or bias == "BULLISH")
    )
    waiting_break = status_key == "WAIT_BREAKDOWN" or (
        status_key == "WATCH_SHORT" and short_trigger and not bool(sig.get("market_map_live_breakdown")) and bias == "BEARISH"
    )
    calibration_context: dict[str, Any] | None = None

    if waiting_reclaim and long_trigger:
        label = "Reclaim odds"
        calibration_context = {
            "scenario": "reclaim_retake" if reclaim_confirmed else "reclaim_initial",
            "asset_bucket": asset_bucket,
            "major_catalyst": catalyst_score >= 3.0 or asset_state == "MAJOR_CATALYST_WATCH",
            "distance_band": _watch_calibration_distance_band(live_anchor, long_trigger, direction="above"),
        }
        if reclaim_confirmed:
            probability += 0.07
            reasons.append("prior reclaim already printed")
        if reclaim_lost and not live_reclaim:
            probability -= 0.03
            reasons.append("intraday hold was lost")
        if bullish_breakout_live:
            probability += 0.04
            reasons.append("breakout pressure is still live")
        if live_anchor > 0:
            distance_pct = max(0.0, (long_trigger - live_anchor) / long_trigger)
            if distance_pct <= 0.0025:
                probability += 0.06
                reasons.append(f"{distance_pct * 100:.2f}% below reclaim")
            elif distance_pct <= 0.0075:
                probability += 0.03
                reasons.append(f"{distance_pct * 100:.2f}% below reclaim")
            elif distance_pct >= 0.03:
                probability -= 0.06
                reasons.append(f"{distance_pct * 100:.2f}% below reclaim")
            elif distance_pct >= 0.015:
                probability -= 0.03
                reasons.append(f"{distance_pct * 100:.2f}% below reclaim")
    elif waiting_break and short_trigger:
        label = "Break odds"
        breakdown_confirmed = bool(sig.get("market_map_breakdown_confirmed"))
        live_breakdown = bool(sig.get("market_map_live_breakdown"))
        breakdown_lost = breakdown_confirmed and not live_breakdown
        calibration_context = {
            "scenario": "breakdown_retake" if breakdown_confirmed else "breakdown_initial",
            "asset_bucket": asset_bucket,
            "major_catalyst": catalyst_score >= 3.0 or asset_state == "MAJOR_CATALYST_WATCH",
            "distance_band": _watch_calibration_distance_band(live_anchor, short_trigger, direction="below"),
        }
        bearish_breakout_live = bool(
            {
                str(sig.get("orderbook_breakout_state") or "").upper(),
                str(sig.get("orderbook_intracycle_breakout_state") or "").upper(),
            } & {"CONFIRMED_BEARISH_BREAKOUT", "PERSISTENT_BEARISH_BREAKOUT"}
        )
        if breakdown_confirmed:
            probability += 0.07
            reasons.append("prior breakdown already printed")
        if breakdown_lost and not live_breakdown:
            probability -= 0.03
            reasons.append("intraday breakdown was lost")
        if bearish_breakout_live:
            probability += 0.04
            reasons.append("sell pressure is still live")
        if live_anchor > 0:
            distance_pct = max(0.0, (live_anchor - short_trigger) / short_trigger)
            if distance_pct <= 0.0025:
                probability += 0.06
                reasons.append(f"{distance_pct * 100:.2f}% above breakdown")
            elif distance_pct <= 0.0075:
                probability += 0.03
                reasons.append(f"{distance_pct * 100:.2f}% above breakdown")
            elif distance_pct >= 0.03:
                probability -= 0.06
                reasons.append(f"{distance_pct * 100:.2f}% above breakdown")
            elif distance_pct >= 0.015:
                probability -= 0.03
                reasons.append(f"{distance_pct * 100:.2f}% above breakdown")
    elif status_key in {"OPEN_LONG", "OPEN_SHORT"}:
        label = "Hold odds"
    elif status_key in {"READY_LONG", "READY_SHORT", "PENDING_ENTRY", "WAITING_CONFIRMATION", "EXECUTABLE", "PASSIVE_ENTRY"}:
        label = "Entry odds"
    elif action_key in {"LONG", "SHORT"}:
        label = "Entry odds"

    if tradable and label in {"Entry odds", "Setup odds"}:
        probability += 0.01

    calibration_result = None
    if calibration_context:
        calibration_result = _calibrate_watch_probability(
            calibration=calibration_model,
            probability=probability,
            scenario=str(calibration_context.get("scenario") or ""),
            asset_bucket=str(calibration_context.get("asset_bucket") or asset_bucket),
            major_catalyst=bool(calibration_context.get("major_catalyst")),
            distance_band=str(calibration_context.get("distance_band") or "unknown"),
        )
    if calibration_result:
        probability = _safe_float(calibration_result.get("probability"))
        empirical_rate = _safe_float(calibration_result.get("empirical_rate"))
        empirical_samples = int(calibration_result.get("samples") or 0)
        probability_source = "calibrated"
        reasons = [str(calibration_result.get("history_note") or "")] + reasons

    probability = _clamp(probability, 0.05, 0.95)
    probability_pct = int(round(probability * 100.0))
    if probability_pct >= 65:
        tier = "high"
    elif probability_pct >= 50:
        tier = "medium"
    else:
        tier = "low"
    note = _compact_sentences(reasons, limit=3)
    detail = f"{label} {probability_pct}%"
    if note:
        detail += f" • {note}"
    return {
        "probability": round(probability, 4),
        "probability_pct": probability_pct,
        "probability_label": label,
        "probability_text": f"{label} {probability_pct}%",
        "probability_detail": detail,
        "probability_tier": tier,
        "probability_source": probability_source,
        "probability_empirical": round(empirical_rate, 4) if empirical_rate is not None else None,
        "probability_empirical_pct": int(round(empirical_rate * 100.0)) if empirical_rate is not None else None,
        "probability_empirical_samples": empirical_samples,
    }


def action_board(
    state: dict,
    market_map: dict,
    decision_dataset_records: Iterable[dict] | None = None,
) -> dict:
    signals = dict((state or {}).get("signals") or {})
    positions = list((state or {}).get("positions") or [])
    positions_by_coin = {str(item.get("coin") or "").upper(): dict(item or {}) for item in positions}
    config = dict((state or {}).get("config") or {})
    dynamic_analysis = [
        str(coin or "").upper()
        for coin in (config.get("dynamic_analysis_coins") or [])
        if str(coin or "").strip()
    ]
    tracked = set()
    tracked.update(str(coin or "").upper() for coin in config.get("coins", []) or [])
    tracked.update(str(coin or "").upper() for coin in config.get("analysis_coins", []) or [])
    tracked.update(str(coin or "").upper() for coin in (market_map or {}).get("coins", {}).keys())
    tracked.update(str(coin or "").upper() for coin in signals.keys())
    tracked.update(str(coin or "").upper() for coin in positions_by_coin.keys())

    entries = dict((market_map or {}).get("coins") or {})
    probability_calibration = _build_watch_probability_calibration(decision_dataset_records)
    items: list[dict[str, Any]] = []
    order = {
        "OPEN_LONG": 0,
        "OPEN_SHORT": 0,
        "PENDING_ENTRY": 1,
        "EXECUTABLE": 1,
        "PASSIVE_ENTRY": 1,
        "READY_LONG": 1,
        "READY_SHORT": 1,
        "WAITING_CONFIRMATION": 2,
        "WAIT_RECLAIM": 2,
        "WATCH_LONG": 2,
        "WAIT_BREAKDOWN": 2,
        "WATCH_SHORT": 2,
        "PORTFOLIO_GUARD": 3,
        "DATA_QUALITY_HOLD": 3,
        "EXECUTION_BLOCKED": 3,
        "RISK_BLOCKED": 3,
        "COOLDOWN": 3,
        "ARMED": 3,
        "NO_SETUP": 3,
    }
    configured_tradeable = {
        str(coin or "").upper().strip()
        for coin in config.get("coins", []) or []
        if str(coin or "").strip()
    }

    for coin in sorted(item for item in tracked if item):
        sig = dict(signals.get(coin) or {})
        pos = positions_by_coin.get(coin)
        map_entry = dict(entries.get(coin) or {})
        tradable = (
            coin in configured_tradeable
            or (sig.get("execution_mode") or "observation_only") == "tradable"
            or pos is not None
        )
        execution_mode = "tradable" if tradable else str(sig.get("execution_mode") or "observation_only")
        bias = str(
            sig.get("market_map_bias")
            or map_entry.get("bias")
            or "NEUTRAL"
        ).upper()
        instrument_type = _instrument_type_for_coin(coin, sig, config)
        asset_bucket = _asset_bucket(instrument_type)
        asset_categories = _asset_categories_for_coin(coin, instrument_type, config)
        asset_category = asset_categories[0] if asset_categories else "other_stocks"
        asset_category_label = _asset_category_label(asset_category, config)
        asset_category_labels = [
            _asset_category_label(category, config)
            for category in asset_categories
        ]
        support = _pick_level(
            map_entry.get("supports"),
            prefer="max",
            fallback=sig.get("market_map_nearest_support"),
        )
        resistance = _pick_level(
            map_entry.get("resistances"),
            prefer="min",
            fallback=sig.get("market_map_nearest_resistance"),
        )
        long_trigger = _pick_level(
            map_entry.get("daily_close_long_above"),
            prefer="min",
            fallback=resistance,
        )
        short_trigger = _pick_level(
            map_entry.get("daily_close_short_below"),
            prefer="max",
            fallback=support,
        )
        current_logic = str(
            (pos or {}).get("current_logic")
            or (pos or {}).get("entry_logic")
            or sig.get("decision_reason")
            or sig.get("flat_reason")
            or ""
        ).strip()
        blocker = _primary_reason(sig.get("flat_reason") or sig.get("decision_reason") or "")
        map_summary = _map_blurb(sig.get("market_map_summary") or map_entry.get("summary") or map_entry.get("notes") or "")
        confidence = str(sig.get("confidence") or "LOW").upper()
        score = _safe_float(sig.get("score") or 50.0)
        conviction_score = _safe_float(sig.get("thesis_conviction_score") or score)
        action = str(sig.get("action") or "FLAT").upper()
        live_anchor = _safe_float(sig.get("live_price") or sig.get("price"))
        reclaim_confirmed = bool(sig.get("market_map_reclaim_confirmed"))
        live_reclaim = bool(sig.get("market_map_live_reclaim"))
        reclaim_lost = bool(sig.get("market_map_reclaim_lost"))
        if long_trigger and live_anchor > 0 and live_anchor >= long_trigger:
            live_reclaim = True
        breakout_states = {
            str(sig.get("orderbook_breakout_state") or "").upper(),
            str(sig.get("orderbook_intracycle_breakout_state") or "").upper(),
        }
        bullish_breakout_live = bool(
            {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"} & breakout_states
        )
        asset_state = str(sig.get("asset_state") or "").upper()
        asset_state_label = str(sig.get("asset_state_label") or "").strip()
        catalyst_watch_label = asset_state_label if asset_state == "MAJOR_CATALYST_WATCH" else ""
        next_unblock = str(sig.get("next_unblock_reason") or "").strip()
        state_override = asset_state in {
            "PENDING_ENTRY",
            "WAITING_CONFIRMATION",
            "PORTFOLIO_GUARD",
            "RISK_BLOCKED",
            "DATA_QUALITY_HOLD",
            "PASSIVE_ENTRY",
            "EXECUTION_BLOCKED",
            "COOLDOWN",
            "ARMED",
            "OBSERVATION_ONLY",
        }

        execution_note = ""
        entry_status = ""

        if pos:
            direction = str(pos.get("direction") or "").upper() or action
            status = f"OPEN_{direction or 'LONG'}"
            label = f"In {direction or 'LONG'}"
            headline = _primary_reason(current_logic) or "Trade is live and being managed."
            stop = _safe_float(pos.get("stop_loss"))
            target = _safe_float(pos.get("take_profit"))
            live_reference = live_anchor or _safe_float(pos.get("current_price"))
            entry_status = _entry_status_text(live_reference, _safe_float(pos.get("entry_price")), "entry")
            trigger = _stop_target_text(live_reference, stop, target) or "Trade is already open."
            execution_note = "Position is already open and under active management."
        elif asset_state == "PENDING_ENTRY":
            status = "PENDING_ENTRY"
            label = asset_state_label or "Pending entry"
            headline = next_unblock or "A resting limit order is already on the book."
            anchor_price = _safe_float(sig.get("price") or sig.get("live_price"))
            entry_status = _entry_status_text(live_anchor, anchor_price, "limit")
            trigger = _numeric_level_text("Limit", anchor_price, live_anchor) or "Waiting for the resting limit order to resolve."
            execution_note = next_unblock or "The order is already working. The next event is a fill, cancel, or expiry."
        elif state_override or (asset_state == "EXECUTABLE" and action == "FLAT"):
            status = asset_state or "ARMED"
            label = asset_state_label or "Setup pending"
            headline = _primary_reason(next_unblock or current_logic or blocker) or "The setup is still gated."
            if action == "LONG" and live_anchor > 0:
                entry_status = _entry_status_text(live_anchor, long_trigger, "trigger")
                trigger = _numeric_level_text("Trigger", long_trigger, live_anchor) or f"Live {live_anchor:,.2f}"
            elif action == "SHORT" and live_anchor > 0:
                entry_status = _entry_status_text(live_anchor, short_trigger, "trigger")
                trigger = _numeric_level_text("Trigger", short_trigger, live_anchor) or f"Live {live_anchor:,.2f}"
            elif long_trigger and bias == "BULLISH":
                entry_status = _entry_status_text(live_anchor, long_trigger, "trigger")
                trigger = _numeric_level_text("Trigger", long_trigger, live_anchor) or f"Trigger {long_trigger:,.2f}"
            elif short_trigger and bias == "BEARISH":
                entry_status = _entry_status_text(live_anchor, short_trigger, "trigger")
                trigger = _numeric_level_text("Trigger", short_trigger, live_anchor) or f"Trigger {short_trigger:,.2f}"
            else:
                trigger = "Wait for the next unblock."
            execution_note = (
                next_unblock
                or str(sig.get("portfolio_guard_summary") or "")
                or str(sig.get("data_reliability_summary") or "")
                or str(sig.get("execution_quality_summary") or "")
                or "The setup is still waiting on one final unlock."
            )
        elif action == "LONG":
            status = "READY_LONG"
            label = "Long thesis live"
            headline = _primary_reason(sig.get("decision_reason") or current_logic) or "Long thesis is live."
            entry_status = _entry_status_text(live_anchor, long_trigger, "trigger")
            trigger_level = long_trigger or _safe_float(sig.get("price")) or _safe_float(sig.get("live_price"))
            trigger = _numeric_level_text("Entry", trigger_level, live_anchor)
            if not trigger:
                trigger = "Watching for a clean long trigger."
            if tradable:
                execution_note = "The thesis qualifies, but the bot still waits for confirmation, sizing, and clean fills before sending the order."
            else:
                execution_note = "The thesis qualifies, but venue support or live market quality is not ready enough to execute yet."
        elif action == "SHORT":
            status = "READY_SHORT"
            label = "Short thesis live"
            headline = _primary_reason(sig.get("decision_reason") or current_logic) or "Short thesis is live."
            entry_status = _entry_status_text(live_anchor, short_trigger, "trigger")
            trigger_level = short_trigger or _safe_float(sig.get("price")) or _safe_float(sig.get("live_price"))
            trigger = _numeric_level_text("Entry", trigger_level, live_anchor)
            if not trigger:
                trigger = "Watching for a clean short trigger."
            if tradable:
                execution_note = "The thesis qualifies, but the bot still waits for confirmation, sizing, and clean fills before sending the order."
            else:
                execution_note = "The thesis qualifies, but venue support or live market quality is not ready enough to execute yet."
        elif bias == "BULLISH" and long_trigger and (reclaim_confirmed or bullish_breakout_live):
            status = "WATCH_LONG"
            label = catalyst_watch_label or "Bullish watch"
            headline = (
                "Daily bias is bullish, and the reclaim is on the board."
                if not map_summary
                else f"Daily bias is bullish, and {map_summary}."
            )
            entry_status = _entry_status_text(live_anchor, long_trigger, "reclaim")
            if reclaim_lost and not live_reclaim:
                trigger = _numeric_level_text("Reclaim", long_trigger, live_anchor)
                default_note = (
                    "The bot saw the reclaim, but live price slipped back below the trigger. "
                    "It wants that level held again before buying."
                )
                execution_note = next_unblock or default_note
            else:
                trigger = _numeric_level_text("Hold", long_trigger, live_anchor)
                default_note = (
                    "The bot sees the reclaim, but it still wants stronger continuation, "
                    "safer entry quality, and final sizing checks before buying."
                )
                execution_note = next_unblock or default_note
        elif bias == "BULLISH" and bool(sig.get("market_map_block_longs")) and long_trigger:
            status = "WAIT_RECLAIM"
            label = catalyst_watch_label or "Wait for reclaim"
            headline = (
                "Daily bias is bullish, but the long is still blocked until price reclaims resistance."
                if not map_summary
                else f"Daily bias is bullish, but {map_summary}."
            )
            entry_status = _entry_status_text(live_anchor, long_trigger, "reclaim")
            trigger = _numeric_level_text("Reclaim", long_trigger, live_anchor)
            execution_note = next_unblock or "The higher-timeframe view is constructive, but the reclaim still has to confirm before the bot can buy."
        elif bias == "BEARISH" and bool(sig.get("market_map_block_shorts")) and short_trigger:
            status = "WAIT_BREAKDOWN"
            label = catalyst_watch_label or "Wait for breakdown"
            headline = (
                "Daily bias is bearish, but the short is still blocked until price breaks support."
                if not map_summary
                else f"Daily bias is bearish, but {map_summary}."
            )
            entry_status = _entry_status_text(live_anchor, short_trigger, "breakdown")
            trigger = _numeric_level_text("Break", short_trigger, live_anchor)
            execution_note = next_unblock or "The higher-timeframe view is bearish, but the breakdown still has to confirm before the bot can short."
        elif bias == "BULLISH":
            status = "WATCH_LONG"
            label = catalyst_watch_label or "Bullish watch"
            headline = (
                "Higher-timeframe bias is bullish, but the entry is not ready."
                if not map_summary
                else f"Higher-timeframe bias is bullish, and {map_summary}."
            )
            entry_status = _entry_status_text(live_anchor, long_trigger, "trigger")
            trigger = _numeric_level_text("Trigger", long_trigger, live_anchor) or "Wait for cleaner long confirmation."
            execution_note = next_unblock or "The agent is reading this as bullish context only. It still needs a qualified live thesis before any order can go out."
        elif bias == "BEARISH":
            status = "WATCH_SHORT"
            label = catalyst_watch_label or "Bearish watch"
            headline = (
                "Higher-timeframe bias is bearish, but the entry is not ready."
                if not map_summary
                else f"Higher-timeframe bias is bearish, and {map_summary}."
            )
            entry_status = _entry_status_text(live_anchor, short_trigger, "trigger")
            trigger = _numeric_level_text("Trigger", short_trigger, live_anchor) or "Wait for cleaner short confirmation."
            execution_note = next_unblock or "The agent is reading this as bearish context only. It still needs a qualified live thesis before any order can go out."
        else:
            status = "NO_SETUP"
            label = "No setup"
            headline = blocker or "No clean edge right now."
            trigger = "Wait for structure and order-flow to agree."
            execution_note = "No trade is allowed right now because the thesis is still incomplete."

        coach_verdict = str(sig.get("execution_coach_verdict") or "").upper()
        coach_summary = str(sig.get("execution_coach_summary") or "").strip()
        if coach_summary and status in {"READY_LONG", "READY_SHORT", "PASSIVE_ENTRY", "EXECUTION_BLOCKED"}:
            execution_note = coach_summary

        probability = _setup_probability(
            sig=sig,
            status=status,
            action=action,
            confidence=confidence,
            score=score,
            conviction_score=conviction_score,
            live_anchor=live_anchor,
            long_trigger=long_trigger,
            short_trigger=short_trigger,
            reclaim_confirmed=reclaim_confirmed,
            live_reclaim=live_reclaim,
            reclaim_lost=reclaim_lost,
            bullish_breakout_live=bullish_breakout_live,
            bias=bias,
            tradable=tradable,
            coach_verdict=coach_verdict,
            asset_bucket=asset_bucket,
            asset_state=asset_state,
            calibration_model=probability_calibration,
        )

        if status in {"WATCH_LONG", "WAIT_RECLAIM", "READY_LONG", "OPEN_LONG"} and support:
            risk = _numeric_level_text("Lose", support, live_anchor) or f"Lose {support:,.2f}"
        elif status in {"WATCH_SHORT", "WAIT_BREAKDOWN", "READY_SHORT", "OPEN_SHORT"} and resistance:
            risk = _numeric_level_text("Reclaim", resistance, live_anchor) or f"Reclaim {resistance:,.2f}"
        else:
            risk = map_summary or ""

        next_setup_reason = _next_setup_reason(
            status=status,
            action=action,
            bias=bias,
            entry_status=entry_status,
            trigger=trigger,
            blocker=str(sig.get("flat_reason") or sig.get("decision_reason") or blocker or ""),
            execution_note=execution_note,
        )

        if tradable:
            mode_label = "EXECUTABLE"
            mode_meta = "Executable"
            mode_badge = "EXEC"
            mode_detail = "This market can execute on the active venue as soon as the thesis, sizing, and fill-quality checks align."
        else:
            mode_label = "SUPPORT PENDING"
            mode_meta = "Support pending"
            mode_badge = "PENDING"
            mode_detail = "The agent is tracking this market, but venue support or live data quality is not ready enough to execute it yet."

        friction_stack = _build_friction_stack(
            sig=sig,
            status=status,
            action=action,
            bias=bias,
            asset_state=asset_state,
            tradable=tradable,
            coach_verdict=coach_verdict,
            headline=headline,
            map_summary=map_summary,
            execution_note=execution_note,
            mode_detail=mode_detail,
        )
        catalyst_rail = _build_catalyst_rail(sig)
        why_this_lead = _build_why_this_lead(
            sig=sig,
            status=status,
            action=action,
            bias=bias,
            probability=probability,
            friction_stack=friction_stack,
            catalyst_rail=catalyst_rail,
        )

        items.append(
            {
                "coin": coin,
                "tradable": tradable,
                "execution_mode": execution_mode,
                "mode_label": mode_label,
                "mode_meta": mode_meta,
                "mode_badge": mode_badge,
                "mode_detail": mode_detail,
                "bias": bias,
                "instrument_type": instrument_type,
                "asset_bucket": asset_bucket,
                "asset_category": asset_category,
                "asset_categories": asset_categories,
                "asset_category_label": asset_category_label,
                "asset_category_labels": asset_category_labels,
                "asset_state": asset_state,
                "asset_state_label": asset_state_label,
                "next_unblock_reason": next_unblock,
                "status": status,
                "label": label,
                "headline": headline,
                "next_setup_reason": next_setup_reason,
                "trigger": trigger,
                "entry_status": entry_status or execution_note,
                "execution_note": execution_note,
                "risk": risk,
                "map_summary": map_summary,
                "friction_stack": friction_stack,
                "catalyst_rail": catalyst_rail,
                "why_this_lead": why_this_lead,
                "confidence": confidence,
                "score": round(score, 1),
                "thesis_conviction_score": round(conviction_score or score, 1),
                "pnl_usd": _safe_float(pos.get("unrealised_pnl")) if pos else 0.0,
                "llm_referee": dict(sig.get("llm_referee") or {}),
                "llm_referee_summary": str(sig.get("llm_referee_summary") or ""),
                "llm_referee_why_now": str(sig.get("llm_referee_why_now") or ""),
                "execution_coach_verdict": coach_verdict,
                "execution_coach_summary": coach_summary,
                **probability,
            }
        )

    items.sort(
        key=lambda item: (
            0 if item.get("tradable") else 1,
            order.get(str(item.get("status") or "NO_SETUP"), 9),
            -abs(_safe_float(item.get("score")) - 50.0),
            str(item.get("coin") or ""),
        )
    )
    tradable_items = [item for item in items if item.get("tradable")]
    observation_items = [item for item in items if not item.get("tradable")]
    pending_count = sum(1 for item in items if str(item.get("status") or "").upper() == "PENDING_ENTRY")
    groups = _group_action_items(items)
    return {
        "updated_at": state.get("last_cycle"),
        "lead": items[0] if items else None,
        "items": items,
        "tradable_items": tradable_items,
        "observation_items": observation_items,
        "groups": groups,
        "summary": {
            "tradable_count": len(tradable_items),
            "observation_count": len(observation_items),
            "active_tradable_count": sum(
                1 for item in tradable_items
                if str(item.get("status") or "") in {"OPEN_LONG", "OPEN_SHORT", "READY_LONG", "READY_SHORT"}
            ),
            "pending_count": pending_count,
            "scout_count": len(dynamic_analysis),
            "scout_preview": dynamic_analysis[:8],
            "scout_market_cap_min_usd": _safe_float(config.get("dynamic_market_cap_min_usd")),
            "bucket_counts": {
                bucket: {
                    "count": data.get("count", 0),
                    "tradable_count": data.get("tradable_count", 0),
                    "observation_count": data.get("observation_count", 0),
                    "active_count": data.get("active_count", 0),
                    "pending_count": data.get("pending_count", 0),
                }
                for bucket, data in groups.items()
            },
        },
    }


def _entry_logic_from_record(record: dict | None, trade: dict) -> str:
    record = dict(record or {})
    entry_ctx = dict(record.get("entry_context") or {})
    thesis = dict(record.get("thesis") or entry_ctx.get("thesis") or {})
    trade_plan = dict(record.get("trade_plan") or entry_ctx.get("trade_plan") or {})
    parts = [
        entry_ctx.get("reason"),
        thesis.get("summary"),
        entry_ctx.get("market_map_summary"),
    ]
    interaction = str(entry_ctx.get("orderbook_interaction") or "")
    breakout = str(entry_ctx.get("orderbook_breakout_state") or "")
    if interaction and interaction.upper() != "BETWEEN_LEVELS":
        parts.append("levels: " + _humanize_key(interaction))
    if breakout and breakout.upper() != "NONE":
        parts.append("breakout: " + _humanize_key(breakout))
    try:
        rr = float(trade_plan.get("risk_reward_ratio") or 0.0)
    except Exception:
        rr = 0.0
    if rr > 0:
        parts.append(f"planned R:R {rr:.2f}")
    fallback = trade.get("open_logic") or trade.get("reason") or "No opening logic recorded"
    return _compact_sentences(parts) or str(fallback)


def _close_logic_from_record(record: dict | None, trade: dict) -> str:
    record = dict(record or {})
    exit_ctx = dict(record.get("exit_context") or {})
    parts = [
        _humanize_exit_reason(trade.get("exit_reason") or record.get("exit_reason")),
        exit_ctx.get("thesis_summary"),
    ]
    interaction = str(exit_ctx.get("orderbook_interaction") or "")
    breakout = str(exit_ctx.get("orderbook_breakout_state") or "")
    if interaction and interaction.upper() != "BETWEEN_LEVELS":
        parts.append("exit near " + _humanize_key(interaction))
    if breakout and breakout.upper() != "NONE":
        parts.append("market was in " + _humanize_key(breakout))
    return _compact_sentences(parts) or str(trade.get("exit_reason") or "No closing logic recorded")


def _agent_lesson_from_record(record: dict | None, trade: dict) -> str:
    record = dict(record or {})
    entry_ctx = dict(record.get("entry_context") or {})
    direction = str(trade.get("direction") or record.get("direction") or "").upper()
    exit_reason = str(trade.get("exit_reason") or record.get("exit_reason") or "").lower()
    pnl = float(trade.get("pnl_usd") or record.get("pnl_usd") or 0.0)
    interaction = str(entry_ctx.get("orderbook_interaction") or "").upper()

    if pnl > 0:
        if exit_reason == "take_profit":
            return "The thesis followed through cleanly. Similar structure can stay tradeable when the same alignment shows up."
        if exit_reason in {"conviction_lost", "time_stop"}:
            return "The move worked, but momentum faded before the full target. Take cleaner partials when follow-through stalls."
        return "This setup paid. Keep favoring trades where structure, levels, and invalidation stay this coherent."

    if exit_reason == "stop_loss":
        if direction == "SHORT" and interaction in {"AT_SUPPORT", "ABOVE_SUPPORT"}:
            return "Avoid shorting straight into defended support and demand."
        if direction == "LONG" and interaction in {"AT_RESISTANCE", "BELOW_RESISTANCE"}:
            return "Avoid longing straight into heavy overhead resistance."
        return "The invalidation was hit quickly. Demand cleaner alignment before taking this setup again."
    if exit_reason in {"conviction_lost", "time_stop"}:
        return "The thesis never developed enough follow-through. Wait for stronger structure before committing capital."
    if exit_reason in {"signal_reversal", "structure_invalidation", "htf_invalidation", "micro_invalidation"}:
        return "Structure turned against the trade. Respect invalidation faster when the higher timeframe disagrees."
    return "Only re-take this pattern when the market map, structure, and order-flow line up more cleanly."


def merge_dataset_into_trades(trades: Iterable[dict] | None, dataset_records: Iterable[dict] | None) -> list[dict]:
    records = _record_index(dataset_records)
    out = []
    for trade in list(trades or []):
        item = dict(trade or {})
        record = records.get(str(item.get("trade_id") or ""))
        if record:
            item["dataset_record"] = record
        item["open_logic"] = _entry_logic_from_record(record, item)
        item["close_logic"] = _close_logic_from_record(record, item)
        item["agent_lesson"] = _agent_lesson_from_record(record, item)
        out.append(item)
    return out


def learning_summary(trades: Iterable[dict] | None) -> dict:
    safe_trades = [dict(trade or {}) for trade in list(trades or [])]
    recent = list(reversed(safe_trades[-8:]))
    lessons = [
        {
            "trade_id": trade.get("trade_id"),
            "coin": trade.get("coin"),
            "direction": trade.get("direction"),
            "pnl_usd": round(_safe_float(trade.get("pnl_usd")), 2),
            "result": "WIN" if float(trade.get("pnl_usd") or 0.0) > 0 else "LOSS" if float(trade.get("pnl_usd") or 0.0) < 0 else "FLAT",
            "open_logic": trade.get("open_logic", ""),
            "close_logic": trade.get("close_logic", ""),
            "lesson": trade.get("agent_lesson", ""),
        }
        for trade in recent
    ]
    latest = lessons[0] if lessons else None
    wins = sum(1 for lesson in lessons if lesson["result"] == "WIN")
    losses = sum(1 for lesson in lessons if lesson["result"] == "LOSS")
    return {
        "count": len(lessons),
        "wins": wins,
        "losses": losses,
        "latest": latest,
        "recent_lessons": lessons,
    }


def market_map_summary(market_map: dict) -> dict:
    coins = dict((market_map or {}).get("coins") or {})
    bullish = 0
    bearish = 0
    neutral = 0
    manual_count = 0
    auto_count = 0
    for entry in coins.values():
        bias = str((entry or {}).get("bias") or "NEUTRAL").upper()
        if bias == "BULLISH":
            bullish += 1
        elif bias == "BEARISH":
            bearish += 1
        else:
            neutral += 1
        if bool((entry or {}).get("auto_generated")) or str((entry or {}).get("source") or "").upper() == "AUTO":
            auto_count += 1
        else:
            manual_count += 1
    return {
        "count": len(coins),
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "manual_count": manual_count,
        "auto_count": auto_count,
        "updated_at": (market_map or {}).get("updated_at"),
    }


def merge_reviews_into_trades(trades: Iterable[dict] | None, trade_reviews: dict) -> list[dict]:
    reviews = dict((trade_reviews or {}).get("reviews") or {})
    out = []
    for trade in list(trades or []):
        item = dict(trade or {})
        review = reviews.get(str(item.get("trade_id") or ""))
        if review:
            item["review"] = dict(review)
        out.append(item)
    return out


def review_summary(trades: Iterable[dict] | None, trade_reviews: dict) -> dict:
    reviews = list(dict((trade_reviews or {}).get("reviews") or {}).values())
    verdicts: dict[str, int] = {}
    thesis_quality: dict[str, int] = {}
    execution_quality: dict[str, int] = {}
    for review in reviews:
        verdict = str(review.get("verdict") or "")
        if verdict:
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
        thesis = str(review.get("thesis_quality") or "")
        if thesis:
            thesis_quality[thesis] = thesis_quality.get(thesis, 0) + 1
        execution = str(review.get("execution_quality") or "")
        if execution:
            execution_quality[execution] = execution_quality.get(execution, 0) + 1
    safe_trades = list(trades or [])
    reviewed = sum(1 for trade in safe_trades if dict(trade or {}).get("review"))
    coverage = round(reviewed / len(safe_trades) * 100, 1) if safe_trades else 0.0
    return {
        "count": len(reviews),
        "coverage_pct": coverage,
        "verdicts": verdicts,
        "thesis_quality": thesis_quality,
        "execution_quality": execution_quality,
        "updated_at": (trade_reviews or {}).get("updated_at") if reviews else None,
    }


def calc_stats(trades: Iterable[dict] | None) -> dict:
    safe_trades = list(trades or [])
    if not safe_trades:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "best": 0,
            "worst": 0,
        }

    closed = []
    for trade in safe_trades:
        try:
            if trade.get("exit_price") and float(trade.get("exit_price", 0)) > 0:
                closed.append(trade)
        except Exception:
            continue

    if not closed:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "best": 0,
            "worst": 0,
        }

    pnls = []
    for trade in closed:
        try:
            pnls.append(float(trade.get("pnl_usd", 0)))
        except Exception:
            pnls.append(0.0)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "total": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1),
        "total_pnl": round(sum(pnls), 2),
        "avg_win": round(sum(wins) / len(wins) if wins else 0, 2),
        "avg_loss": round(sum(losses) / len(losses) if losses else 0, 2),
        "best": round(max(pnls), 2),
        "worst": round(min(pnls), 2),
    }


def runtime_status(state: dict) -> dict:
    last_cycle = state.get("last_cycle")
    stale = False
    age_seconds = None
    interval = int(((state.get("config") or {}).get("check_interval_seconds")) or 120)
    if isinstance(last_cycle, str):
        try:
            age_seconds = int((datetime.now() - datetime.strptime(last_cycle, "%Y-%m-%d %H:%M:%S")).total_seconds())
            stale = age_seconds > max(interval * 2, 240)
        except Exception:
            age_seconds = None
    return {
        "stale": stale,
        "state_age_seconds": age_seconds,
    }


def decision_summary(state: dict) -> dict:
    signals = (state or {}).get("signals") or {}
    summary = {
        "long_count": 0,
        "short_count": 0,
        "flat_count": 0,
        "tradable_count": 0,
        "tradable_active_count": 0,
        "lead": None,
    }
    lead_rank = (-1, -1, -1.0)

    for coin, sig in signals.items():
        action = str(sig.get("action") or "FLAT").upper()
        if action not in {"LONG", "SHORT", "FLAT"}:
            action = "FLAT"
        summary[f"{action.lower()}_count"] += 1

        execution_mode = sig.get("execution_mode") or "observation_only"
        is_tradable = execution_mode == "tradable"
        if is_tradable:
            summary["tradable_count"] += 1
        if is_tradable and action != "FLAT":
            summary["tradable_active_count"] += 1

        try:
            strength = abs(float(sig.get("score", 50.0)) - 50.0)
        except Exception:
            strength = 0.0

        rank = (
            1 if action != "FLAT" else 0,
            1 if is_tradable else 0,
            strength,
        )
        if rank > lead_rank:
            lead_rank = rank
            summary["lead"] = {
                "coin": coin,
                "action": action,
                "score": sig.get("score", 50.0),
                "confidence": sig.get("confidence", "LOW"),
                "execution_mode": execution_mode,
                "reason": sig.get("decision_reason") or sig.get("reason") or sig.get("flat_reason") or "",
            }

    return summary


def augment_state(state: Any) -> dict:
    safe_state = dict(state or {})
    merged = default_state()
    merged.update(safe_state)
    merged["config"] = _merge_dashboard_config(merged.get("config"))
    merged["positions_count"] = len(merged.get("positions") or [])
    merged["decision_summary"] = decision_summary(merged)
    return merged


def server_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_dashboard_snapshot(
    state: Any,
    trades: Iterable[dict] | None,
    control: Any = None,
    market_map: Any = None,
    trade_reviews: Any = None,
    trade_dataset_records: Iterable[dict] | None = None,
    decision_dataset_records: Iterable[dict] | None = None,
    decision_review_report: Any = None,
    challenger_report: Any = None,
    missed_move_report: Any = None,
    asset_dossiers: Any = None,
    llm_referee_report: Any = None,
    playbook_distiller_report: Any = None,
    *,
    server_timestamp: str | None = None,
) -> dict:
    normalized_market_map = normalize_market_map(market_map)
    normalized_trade_reviews = normalize_trade_reviews(trade_reviews)
    enriched_trades = merge_dataset_into_trades(trades or [], trade_dataset_records)
    safe_trades = merge_reviews_into_trades(enriched_trades, normalized_trade_reviews)
    shaped_state = augment_state(state)
    return {
        "state": shaped_state,
        "trades": safe_trades[-50:][::-1],
        "stats": calc_stats(safe_trades),
        "control": normalize_control(control),
        "action_board": action_board(shaped_state, normalized_market_map, decision_dataset_records),
        "market_map": normalized_market_map,
        "market_map_summary": market_map_summary(normalized_market_map),
        "trade_reviews": normalized_trade_reviews,
        "review_summary": review_summary(safe_trades, normalized_trade_reviews),
        "learning_summary": learning_summary(safe_trades),
        "decision_review_report": dict(decision_review_report or {}),
        "challenger_report": dict(challenger_report or {}),
        "missed_move_report": dict(missed_move_report or {}),
        "asset_dossiers": dict(asset_dossiers or {}),
        "llm_referee_report": dict(llm_referee_report or {}),
        "playbook_distiller_report": dict(playbook_distiller_report or {}),
        "runtime": runtime_status(shaped_state),
        "server_time": server_timestamp or server_time(),
    }
