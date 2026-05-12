"""
asset_context.py - shared asset metadata helpers.

Keep the product surfaces lean by making one small module own instrument type,
category, bucket, and theme normalization.
"""

from __future__ import annotations

from typing import Any

DEFAULT_ASSET_CATEGORY_LABELS = {
    "crypto": "Coins",
    "indices_macro": "Indices & Macro",
    "pre_ipo": "Pre-IPO",
    "mag7": "Mag7",
    "semis_memory": "Semis & Memory",
    "neoclouds": "Neoclouds",
    "ai_infra": "AI Infra",
    "crypto_equities": "Crypto Equities",
    "asia_macro": "Asia Macro",
    "latam_macro": "LatAm Macro",
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
    "software": "Software",
    "other_stocks": "Other Stocks",
}

DEFAULT_ASSET_CATEGORY_DESCRIPTIONS = {
    "indices_macro": "Macro-linked index and ETF proxies with slower, cleaner tape.",
    "mag7": "The big US platform leaders where catalyst reactions matter most.",
    "semis_memory": "Chip, storage, and memory names that often move together on AI demand.",
    "neoclouds": "GPU-cloud and AI infrastructure names tied to compute demand.",
    "ai_infra": "AI infrastructure names where cloud commitments and capex cycles can drive re-rates.",
    "crypto_equities": "Public market crypto beta, brokers, exchanges, and treasury-linked stocks.",
    "asia_macro": "Asia-linked equity, index, and currency proxies for regional risk appetite.",
    "latam_macro": "LatAm beta is mainly about local rates, commodities, FX, and global risk appetite.",
    "commodities_metals": "Metals and hard-asset proxies with macro and supply-chain sensitivity.",
    "energy": "Oil, gas, and energy equity proxies where inventory and geopolitical catalysts matter.",
    "agriculture": "Agricultural contracts with cleaner macro and weather-driven trend behavior.",
    "fx_rates": "Currency and dollar-index proxies that can anchor macro risk direction.",
    "uranium": "Uranium spot and miner exposure with policy, supply, and nuclear-demand catalysts.",
    "volatility": "Volatility instruments for stress, hedging, and regime-change reads.",
    "consumer": "Consumer and discretionary names where spending trends drive tape.",
    "financials": "Financials and private-market exposure with rate and liquidity sensitivity.",
    "biotech_glp1": "Healthcare and GLP-1-linked growth where product catalysts matter.",
    "meme_momentum": "High-reflexivity tickers where flow, attention, and positioning dominate.",
    "growth": "Single-name growth setups outside the big platform and chip baskets.",
    "software": "Software longs need durable growth, margin leverage, and AI monetization proof.",
    "other_stocks": "Everything executable that does not fit the main desks yet.",
}

THEME_BY_CATEGORY = {
    "crypto": "CRYPTO_BETA",
    "indices_macro": "US_MACRO_BETA",
    "pre_ipo": "PRE_IPO_EVENT",
    "mag7": "MEGA_CAP_TECH",
    "semis_memory": "SEMIS_MEMORY",
    "neoclouds": "NEOCLOUDS",
    "ai_infra": "AI_INFRA",
    "crypto_equities": "CRYPTO_EQUITIES",
    "asia_macro": "ASIA_MACRO",
    "latam_macro": "LATAM_MACRO",
    "commodities_metals": "COMMODITIES_METALS",
    "energy": "ENERGY_COMPLEX",
    "agriculture": "AGRICULTURE",
    "fx_rates": "FX_RATES",
    "uranium": "URANIUM",
    "volatility": "VOLATILITY",
    "consumer": "CONSUMER_GROWTH",
    "financials": "FINANCIALS",
    "biotech_glp1": "BIOTECH_GLP1",
    "meme_momentum": "MEME_MOMENTUM",
    "growth": "US_GROWTH",
    "software": "SOFTWARE_GROWTH",
    "other_stocks": "OTHER_STOCKS",
}


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


def asset_category_label(category: str, *, config: dict | None = None) -> str:
    key = safe_text(category).lower()
    labels = dict(DEFAULT_ASSET_CATEGORY_LABELS)
    labels.update({
        safe_text(raw_key).lower(): safe_text(raw_value)
        for raw_key, raw_value in dict((config or {}).get("asset_category_labels") or {}).items()
        if safe_text(raw_key) and safe_text(raw_value)
    })
    return labels.get(key, key.replace("_", " ").title() or "Other")


def asset_category_descriptions(*, config: dict | None = None) -> dict[str, str]:
    descriptions = dict(DEFAULT_ASSET_CATEGORY_DESCRIPTIONS)
    descriptions.update({
        safe_text(raw_key).lower(): safe_text(raw_value)
        for raw_key, raw_value in dict((config or {}).get("asset_category_descriptions") or {}).items()
        if safe_text(raw_key) and safe_text(raw_value)
    })
    return descriptions


def asset_category_description(category: str, *, config: dict | None = None) -> str:
    return asset_category_descriptions(config=config).get(safe_text(category).lower(), "")


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
    return theme_from_categories(categories or [default], instrument_type_for_coin(coin, state=state, config=cfg))


def theme_from_categories(categories: list[str] | None, instrument_type: str = "") -> str:
    primary = safe_text((categories or [""])[0]).lower()
    if primary in THEME_BY_CATEGORY:
        return THEME_BY_CATEGORY[primary]
    normalized_type = safe_text(instrument_type).lower()
    if normalized_type == "crypto":
        return "CRYPTO_BETA"
    if normalized_type == "index":
        return "US_MACRO_BETA"
    return (primary or "OTHER_STOCKS").upper()
