"""
asset_context.py - shared asset metadata helpers.

Keep the product surfaces lean by making one small module own instrument type,
category, bucket, and theme normalization.
"""

from __future__ import annotations

from typing import Any


def safe_text(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def normalize_symbol(value: Any) -> str:
    return safe_text(value).upper()


def normalize_asset_category_values(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_values = value.replace("|", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    categories: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        category = safe_text(raw).lower()
        if category and category not in seen:
            seen.add(category)
            categories.append(category)
    return categories


def config_from_state(state: dict | None) -> dict:
    return dict((state or {}).get("config") or {})


def instrument_type_for_coin(
    coin: str,
    *,
    signal: dict | None = None,
    item: dict | None = None,
    state: dict | None = None,
    config: dict | None = None,
    default: str = "crypto",
) -> str:
    cfg = dict(config or config_from_state(state))
    instrument_types = dict(cfg.get("instrument_types") or {})
    symbol = normalize_symbol(coin)
    return safe_text(
        (item or {}).get("instrument_type")
        or (signal or {}).get("instrument_type")
        or instrument_types.get(symbol),
        default,
    ).lower()


def asset_bucket(instrument_type: str) -> str:
    return "coin" if safe_text(instrument_type, "crypto").lower() == "crypto" else "equity"


def asset_categories_for_coin(
    coin: str,
    *,
    signal: dict | None = None,
    item: dict | None = None,
    state: dict | None = None,
    config: dict | None = None,
    instrument_type: str | None = None,
) -> list[str]:
    cfg = dict(config or config_from_state(state))
    symbol = normalize_symbol(coin)
    category_map = dict(cfg.get("asset_categories") or {})

    for source in (
        (item or {}).get("asset_categories") or (item or {}).get("asset_category"),
        (signal or {}).get("asset_categories") or (signal or {}).get("asset_category"),
        category_map.get(symbol),
    ):
        categories = normalize_asset_category_values(source)
        if categories:
            return categories

    normalized_type = safe_text(
        instrument_type
        or instrument_type_for_coin(symbol, signal=signal, item=item, state=state, config=cfg),
        "crypto",
    ).lower()
    if normalized_type == "index":
        return ["indices_macro"]
    if normalized_type == "equity":
        return ["other_stocks"]
    return ["crypto"]


def theme_for_coin(
    coin: str,
    categories: list[str] | None = None,
    *,
    state: dict | None = None,
    config: dict | None = None,
    default: str = "crypto",
) -> str:
    cfg = dict(config or config_from_state(state))
    theme_map = dict(cfg.get("portfolio_theme_map") or {})
    explicit = safe_text(theme_map.get(normalize_symbol(coin)))
    if explicit:
        return explicit
    return safe_text((categories or [default])[0], default).upper()
