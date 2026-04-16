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

    max_positions = int(getattr(trading_cfg, "portfolio_theme_max_positions", 2) or 2)
    max_same_dir_exposure_pct = float(
        getattr(trading_cfg, "portfolio_theme_max_same_direction_exposure_pct", 0.18) or 0.18
    )
    warning_pct = float(getattr(trading_cfg, "portfolio_theme_warning_exposure_pct", 0.10) or 0.10)
    soft_penalty = float(getattr(trading_cfg, "portfolio_correlation_soft_penalty", 0.65) or 0.65)
    secondary_penalty = float(getattr(trading_cfg, "portfolio_correlation_secondary_penalty", 0.82) or 0.82)

    blockers: list[str] = []
    warnings: list[str] = []
    size_multiplier = 1.0

    if len(same_theme_same_direction) >= max_positions:
        blockers.append(
            f"{theme} already has {len(same_theme_same_direction)}/{max_positions} same-direction slots in use"
        )

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
    }
