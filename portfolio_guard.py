"""
portfolio_guard.py — keep the agent from stacking the same idea repeatedly.

The goal is not to eliminate conviction. It is to stop the book from becoming
an over-concentrated bet on one correlated theme.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _theme_for_coin(trading_cfg, coin: str, instrument_type: str) -> str:
    mapping = dict(getattr(trading_cfg, "portfolio_theme_map", {}) or {})
    coin_upper = _safe_str(coin).upper()
    if coin_upper in mapping:
        return _safe_str(mapping[coin_upper], "UNMAPPED")
    inst = _safe_str(instrument_type, "unknown").lower()
    if inst == "equity":
        return "MEGA_CAP_TECH"
    if inst == "index":
        return "MACRO_INDEX"
    return "CRYPTO_BETA"


def _metadata_for_item(item: Any) -> Mapping[str, Any]:
    if isinstance(item, Mapping):
        return item.get("metadata", {}) or {}
    return getattr(item, "metadata", {}) or {}


def _entry_context_for_item(item: Any) -> Mapping[str, Any]:
    metadata = _metadata_for_item(item)
    if isinstance(metadata, Mapping):
        context = metadata.get("entry_context", {}) or {}
        if isinstance(context, Mapping):
            return context
    if isinstance(item, Mapping):
        context = item.get("entry_context", {}) or item.get("signal_snapshot", {}) or {}
        if isinstance(context, Mapping):
            return context
    return {}


def _is_event_risk_item(item: Any) -> bool:
    context = _entry_context_for_item(item)
    if not isinstance(context, Mapping):
        return False
    if bool(context.get("event_risk_budget_active") or context.get("conviction_entry_event")):
        return True
    conviction_entry = context.get("conviction_entry")
    if isinstance(conviction_entry, Mapping) and bool(conviction_entry.get("event_conviction")):
        return True
    thesis = context.get("thesis")
    if isinstance(thesis, Mapping):
        thesis_entry = thesis.get("conviction_entry")
        if isinstance(thesis_entry, Mapping) and bool(thesis_entry.get("event_conviction")):
            return True
    event_tags = context.get("news_event_tags") or context.get("event_tags") or []
    if isinstance(event_tags, (list, tuple, set)) and any(str(tag).strip() for tag in event_tags):
        return bool(context.get("news_event_score", 0.0) or context.get("event_score", 0.0))
    return False


def _event_budget_entries(
    trading_cfg,
    *,
    instrument_type: str,
    open_positions: Iterable[Any] | None,
    pending_orders: Iterable[Any] | None,
) -> list[dict]:
    entries: list[dict] = []
    for item in list(open_positions or []):
        if not _is_event_risk_item(item):
            continue
        coin = _safe_str(item.get("coin") if isinstance(item, Mapping) else getattr(item, "coin", "")).upper()
        direction = _safe_str(item.get("direction") if isinstance(item, Mapping) else getattr(item, "direction", "")).upper()
        size_usd = _safe_float(item.get("size_usd") if isinstance(item, Mapping) else getattr(item, "size_usd", 0.0))
        context = _entry_context_for_item(item)
        item_instrument = _safe_str(context.get("instrument_type"), instrument_type)
        entries.append({
            "coin": coin,
            "direction": direction,
            "size_usd": size_usd,
            "theme": _theme_for_coin(trading_cfg, coin, item_instrument),
        })
    for pending in list(pending_orders or []):
        if not _is_event_risk_item(pending):
            continue
        coin = _safe_str(getattr(pending, "coin", "")).upper()
        direction = _safe_str(getattr(pending, "direction", "")).upper()
        size_usd = _safe_float(getattr(pending, "size_usd", 0.0))
        context = _entry_context_for_item(pending)
        item_instrument = _safe_str(context.get("instrument_type"), instrument_type)
        entries.append({
            "coin": coin,
            "direction": direction,
            "size_usd": size_usd,
            "theme": _theme_for_coin(trading_cfg, coin, item_instrument),
        })
    return entries


def assess_correlation(
    trading_cfg,
    *,
    coin: str,
    direction: str,
    instrument_type: str,
    portfolio_usd: float,
    proposed_size_usd: float,
    open_positions: Iterable[Any] | None = None,
    pending_orders: Iterable[Any] | None = None,
    event_starter: bool = False,
) -> dict:
    theme = _theme_for_coin(trading_cfg, coin, instrument_type)
    same_direction = _safe_str(direction).upper()
    same_theme_positions: list[dict] = []
    same_theme_same_direction: list[dict] = []

    for pos in list(open_positions or []):
        item = pos if isinstance(pos, Mapping) else {
            "coin": getattr(pos, "coin", ""),
            "direction": getattr(pos, "direction", ""),
            "size_usd": getattr(pos, "size_usd", 0.0),
            "instrument_type": getattr(pos, "metadata", {}).get("entry_context", {}).get("instrument_type", ""),
        }
        pos_coin = _safe_str(item.get("coin")).upper()
        pos_theme = _theme_for_coin(
            trading_cfg,
            pos_coin,
            _safe_str(item.get("instrument_type"), instrument_type),
        )
        if pos_coin == _safe_str(coin).upper():
            continue
        if pos_theme != theme:
            continue
        enriched = {
            "coin": pos_coin,
            "direction": _safe_str(item.get("direction")).upper(),
            "size_usd": _safe_float(item.get("size_usd")),
            "theme": pos_theme,
        }
        same_theme_positions.append(enriched)
        if enriched["direction"] == same_direction:
            same_theme_same_direction.append(enriched)

    for pending in list(pending_orders or []):
        pending_coin = _safe_str(getattr(pending, "coin", "")).upper()
        if not pending_coin or pending_coin == _safe_str(coin).upper():
            continue
        pending_theme = _theme_for_coin(trading_cfg, pending_coin, instrument_type)
        if pending_theme != theme:
            continue
        enriched = {
            "coin": pending_coin,
            "direction": _safe_str(getattr(pending, "direction", "")).upper(),
            "size_usd": _safe_float(getattr(pending, "size_usd", 0.0)),
            "theme": pending_theme,
        }
        same_theme_positions.append(enriched)
        if enriched["direction"] == same_direction:
            same_theme_same_direction.append(enriched)

    same_direction_exposure = sum(item["size_usd"] for item in same_theme_same_direction)
    total_theme_exposure = sum(item["size_usd"] for item in same_theme_positions)
    next_same_direction_exposure_pct = (
        (same_direction_exposure + max(0.0, proposed_size_usd)) / max(portfolio_usd, 1e-9)
        if portfolio_usd > 0 else 0.0
    )
    total_theme_exposure_pct = (
        total_theme_exposure / max(portfolio_usd, 1e-9)
        if portfolio_usd > 0 else 0.0
    )

    base_max_positions = int(getattr(trading_cfg, "portfolio_theme_max_positions", 2) or 2)
    event_extra_slots = int(getattr(trading_cfg, "portfolio_theme_event_starter_extra_slots", 0) or 0)
    max_positions = base_max_positions + (event_extra_slots if event_starter else 0)
    max_same_dir_exposure_pct = float(
        getattr(trading_cfg, "portfolio_theme_max_same_direction_exposure_pct", 0.18) or 0.18
    )
    warning_pct = float(getattr(trading_cfg, "portfolio_theme_warning_exposure_pct", 0.10) or 0.10)
    soft_penalty = float(getattr(trading_cfg, "portfolio_correlation_soft_penalty", 0.65) or 0.65)
    secondary_penalty = float(getattr(trading_cfg, "portfolio_correlation_secondary_penalty", 0.82) or 0.82)
    event_extra_penalty = float(getattr(trading_cfg, "portfolio_correlation_event_starter_extra_penalty", 0.50) or 0.50)
    event_budget_enabled = bool(getattr(trading_cfg, "event_risk_budget_enabled", True))

    blockers: list[str] = []
    warnings: list[str] = []
    size_multiplier = 1.0
    event_budget = {
        "enabled": bool(event_budget_enabled),
        "active": bool(event_starter and event_budget_enabled),
        "permitted": True,
        "summary": "",
        "size_multiplier": 1.0,
        "total_exposure_pct": 0.0,
        "theme_exposure_pct": 0.0,
        "projected_total_exposure_pct": 0.0,
        "projected_theme_exposure_pct": 0.0,
        "single_trade_cap_pct": 0.0,
        "related_coins": [],
    }

    if len(same_theme_same_direction) >= max_positions:
        blockers.append(
            f"{theme} already has {len(same_theme_same_direction)}/{max_positions} same-direction slots in use"
        )
    elif event_starter and len(same_theme_same_direction) >= base_max_positions:
        extra_slot_number = max(1, len(same_theme_same_direction) - base_max_positions + 1)
        size_multiplier *= event_extra_penalty ** extra_slot_number
        warnings.append(f"{theme} event starter is using the extra scout slot; trimming size")

    if next_same_direction_exposure_pct > max_same_dir_exposure_pct:
        blockers.append(
            f"{theme} same-direction exposure would rise to {next_same_direction_exposure_pct * 100:.1f}% "
            f"(limit {max_same_dir_exposure_pct * 100:.1f}%)"
        )

    if len(same_theme_same_direction) >= 1:
        size_multiplier *= soft_penalty
        warnings.append(f"{theme} already has a same-direction live idea; trimming size")
    elif total_theme_exposure_pct >= warning_pct:
        size_multiplier *= secondary_penalty
        warnings.append(f"{theme} theme exposure is already elevated; trimming size")

    if event_starter and event_budget_enabled:
        event_entries = _event_budget_entries(
            trading_cfg,
            instrument_type=instrument_type,
            open_positions=open_positions,
            pending_orders=pending_orders,
        )
        total_event_exposure = sum(item["size_usd"] for item in event_entries)
        theme_event_exposure = sum(item["size_usd"] for item in event_entries if item["theme"] == theme)
        proposed = max(0.0, _safe_float(proposed_size_usd))
        portfolio_base = max(_safe_float(portfolio_usd), 1e-9)
        max_total_pct = max(0.0, float(getattr(trading_cfg, "event_risk_budget_max_portfolio_pct", 0.10) or 0.10))
        max_theme_pct = max(0.0, float(getattr(trading_cfg, "event_risk_budget_max_theme_pct", 0.08) or 0.08))
        max_single_pct = max(0.0, float(getattr(trading_cfg, "event_risk_budget_max_single_pct", 0.02) or 0.02))
        soft_budget_pct = max(0.0, min(0.95, float(getattr(trading_cfg, "event_risk_budget_soft_penalty_pct", 0.65) or 0.65)))
        min_trade_usd = max(
            _safe_float(getattr(trading_cfg, "min_trade_usd", 0.0)),
            _safe_float(getattr(trading_cfg, "event_risk_budget_min_trade_usd", 100.0)),
        )
        strict_caps = bool(getattr(trading_cfg, "event_risk_budget_strict_caps", True))
        total_cap_usd = portfolio_base * max_total_pct
        theme_cap_usd = portfolio_base * max_theme_pct
        single_cap_usd = portfolio_base * max_single_pct
        if not strict_caps:
            total_cap_usd = max(total_cap_usd, min_trade_usd)
            theme_cap_usd = max(theme_cap_usd, min_trade_usd)
            single_cap_usd = max(single_cap_usd, min_trade_usd)
        event_multiplier = 1.0

        def apply_cap(label: str, cap_usd: float, used_usd: float) -> None:
            nonlocal event_multiplier
            if proposed <= 0 or cap_usd <= 0:
                blockers.append(f"event risk budget has no {label} capacity")
                return
            allowed_size = max(0.0, cap_usd - used_usd)
            if allowed_size <= 0:
                blockers.append(f"event risk budget is full at the {label} level")
                return
            if proposed > allowed_size:
                if allowed_size < min_trade_usd:
                    blockers.append(
                        f"event risk {label} budget has only ${allowed_size:.0f} left, below the ${min_trade_usd:.0f} starter minimum"
                    )
                    return
                event_multiplier = min(event_multiplier, allowed_size / max(proposed, 1e-9))
                warnings.append(f"event risk {label} budget trims starter to ${allowed_size:.0f}")

        apply_cap("single-name", single_cap_usd, 0.0)
        apply_cap("theme", theme_cap_usd, theme_event_exposure)
        apply_cap("portfolio", total_cap_usd, total_event_exposure)

        if total_cap_usd > 0 and total_event_exposure >= total_cap_usd * soft_budget_pct:
            event_multiplier *= 0.82
            warnings.append("event risk budget is already heavily used; trimming starter")

        size_multiplier *= event_multiplier
        event_budget.update({
            "permitted": not any("event risk" in blocker for blocker in blockers),
            "summary": (
                "event budget has room for a starter"
                if not any("event risk" in blocker for blocker in blockers)
                else next((blocker for blocker in blockers if "event risk" in blocker), "event risk budget blocks starter")
            ),
            "size_multiplier": round(max(0.0, min(1.0, event_multiplier)), 4),
            "total_exposure_pct": round(total_event_exposure / portfolio_base * 100.0, 3),
            "theme_exposure_pct": round(theme_event_exposure / portfolio_base * 100.0, 3),
            "projected_total_exposure_pct": round((total_event_exposure + proposed * event_multiplier) / portfolio_base * 100.0, 3),
            "projected_theme_exposure_pct": round((theme_event_exposure + proposed * event_multiplier) / portfolio_base * 100.0, 3),
            "single_trade_cap_pct": round(max_single_pct * 100.0, 3),
            "related_coins": [item["coin"] for item in event_entries[:6]],
        })

    permitted = not blockers
    summary = (
        blockers[0]
        if blockers
        else (warnings[0] if warnings else f"{theme} exposure is clean enough to add")
    )
    return {
        "permitted": permitted,
        "theme": theme,
        "summary": summary,
        "blockers": blockers[:4],
        "warnings": warnings[:4],
        "size_multiplier": round(max(0.1, min(1.0, size_multiplier)), 4),
        "same_theme_count": len(same_theme_positions),
        "same_theme_same_direction_count": len(same_theme_same_direction),
        "same_direction_exposure_pct": round(next_same_direction_exposure_pct * 100.0, 3),
        "total_theme_exposure_pct": round(total_theme_exposure_pct * 100.0, 3),
        "related_coins": [item["coin"] for item in same_theme_positions[:5]],
        "event_budget": event_budget,
        "event_budget_summary": event_budget.get("summary", ""),
        "event_budget_size_multiplier": event_budget.get("size_multiplier", 1.0),
        "event_budget_total_exposure_pct": event_budget.get("projected_total_exposure_pct", 0.0),
        "event_budget_theme_exposure_pct": event_budget.get("projected_theme_exposure_pct", 0.0),
    }
