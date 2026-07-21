"""
tradexyz_profile.py - Trade.xyz dashboard intelligence.

This module owns the Trade.xyz-specific view model so dashboard snapshot code
does not reimplement segment, thesis, and high-timeframe hold logic.
"""

from __future__ import annotations

from typing import Any

from asset_context import (
    asset_categories_for_coin,
    asset_category_label,
    instrument_type_for_coin,
    normalize_asset_category_values,
    normalize_symbol,
)
from exchanges.hyperliquid_markets import TRADEXYZ_ASSET_METADATA


CATEGORY_SEGMENTS = {
    "pre_ipo": "Pre-IPO",
    "mag7": "Megacap AI",
    "semis_memory": "Semis & Memory",
    "neoclouds": "Neo Cloud",
    "ai_infra": "AI Infrastructure",
    "crypto_equities": "Crypto Equity Beta",
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
    "biotech_glp1": "Healthcare / GLP-1",
    "meme_momentum": "Meme Momentum",
    "growth": "High-Beta Growth",
    "software": "Software",
    "other_stocks": "Other Stocks",
    "indices_macro": "Macro Indices",
}

SEGMENT_OVERRIDES = {
    "AMD": "AI Compute",
    "CBRS": "AI Compute",
    "DRAM": "Memory",
    "INTC": "AI Compute",
    "KIOXIA": "Memory",
    "LITE": "Optics",
    "MRVL": "Interconnect",
    "MU": "Memory",
    "NVDA": "AI Compute",
    "SKHX": "Memory",
    "SMSN": "Memory",
    "SNDK": "Memory",
    "TSM": "AI Compute",
    "CRWV": "Neo Cloud",
    "GOOGL": "Megacap AI",
    "META": "Megacap AI",
    "AMZN": "Megacap AI",
    "MSFT": "Megacap AI",
    "ORCL": "AI Infrastructure",
    "PLTR": "AI Infrastructure",
    "SOFTBANK": "AI Infrastructure",
}

SEGMENT_THESES = {
    "AI Compute": "Compute demand is the AI bottleneck; supply, backlog, and capex surprises can keep repricing the group.",
    "Optics": "AI clusters need faster optical links; bookings and interconnect spend are the key tells.",
    "Memory": "AI servers are memory-heavy; HBM, DRAM, and NAND tightness can extend the cycle.",
    "Interconnect": "Custom silicon and high-speed connectivity benefit when cloud capex shifts from raw GPUs to full systems.",
    "Megacap AI": "Platform leaders convert AI demand into cloud revenue, ads, software leverage, and buyback capacity.",
    "Neo Cloud": "GPU-cloud names work when utilization, backlog, and financing stay ahead of supply concerns.",
    "AI Infrastructure": "Infra software, cloud plumbing, and strategic AI assets benefit from enterprise AI deployment.",
    "Pre-IPO": "Private-market AI listings can move on scarcity, valuation marks, and event-flow more than fundamentals.",
    "Crypto Equity Beta": "Equity wrappers for crypto volumes, treasury beta, and exchange activity amplify coin cycles.",
    "Asia Macro": "Asia-linked contracts read through to semis, FX, exporters, and regional risk appetite.",
    "LatAm Macro": "LatAm beta is mainly about local rates, commodities, FX, and global risk appetite.",
    "Metals": "Metals track industrial demand, supply friction, dollar liquidity, and hard-asset hedging.",
    "Energy": "Energy contracts key off inventory, geopolitics, weather, and global demand surprises.",
    "Agriculture": "Agricultural contracts move on weather, crop balance sheets, and supply shocks.",
    "FX & Rates": "Currencies and rates proxies anchor the macro regime for risk assets.",
    "Uranium": "Uranium exposure follows nuclear demand, supply discipline, and policy support.",
    "Volatility": "Volatility products are regime-change and stress hedges, not default long holds.",
    "Consumer": "Consumer names need clean demand, margins, and guidance to sustain a long hold.",
    "Financials": "Financials depend on liquidity, rates, credit quality, and deal activity.",
    "Healthcare / GLP-1": "Healthcare growth works when product demand and guidance revisions stay favorable.",
    "Meme Momentum": "Flow and attention dominate; holds require strict invalidation because reflexivity cuts both ways.",
    "High-Beta Growth": "Growth names need revenue acceleration plus risk-on tape to justify longer holds.",
    "Software": "Software longs need durable growth, margin leverage, and AI monetization proof.",
    "Macro Indices": "Macro index contracts express broad risk appetite, inflation, and liquidity regimes.",
    "Other Stocks": "Unclassified Trade.xyz names stay watch-only until the agent has a cleaner driver.",
}

NAME_THESES = {
    "CBRS": "Cerebras is a scarce AI-compute pure play; only scale when liquidity and event risk are clean.",
    "CRWV": "CoreWeave is tied to GPU-cloud demand; backlog, utilization, and financing are the core checks.",
    "GOOGL": "Google needs AI cloud, search monetization, and capex efficiency to keep the long thesis alive.",
    "META": "Meta works when AI ad tools, engagement, and margin discipline beat capex worry.",
    "AMZN": "Amazon needs AWS reacceleration and retail margin leverage to justify a longer hold.",
    "MSFT": "Microsoft is a cloud and AI software compounder; the hold depends on Azure and Copilot proof.",
    "INTC": "Intel is a turnaround and AI-supply optionality trade; hold only while evidence improves.",
    "LITE": "Lumentum is the optics read-through; bookings and datacenter demand need to confirm.",
    "MRVL": "Marvell is custom silicon plus interconnect exposure; AI design wins need to translate into revisions.",
    "MU": "Micron is levered to HBM and DRAM tightness; pricing and guide revisions decide the hold.",
    "SNDK": "SanDisk is storage-cycle exposure; NAND pricing must stay firm for a durable long.",
    "SKHX": "SK Hynix is HBM leadership exposure; memory pricing and AI server demand are the drivers.",
    "SMSN": "Samsung is a memory and Korea-tech read-through; HBM progress matters most.",
    "TSM": "TSMC is the AI foundry toll road; capex, utilization, and customer demand anchor the thesis.",
    "NVDA": "NVIDIA is the AI accelerator benchmark; supply, backlog, margins, and cloud capex drive the tape.",
    "AMD": "AMD is AI accelerator catch-up exposure; evidence of share gain matters more than sympathy moves.",
    "ORCL": "Oracle is AI cloud capacity exposure; bookings and remaining-performance obligations are key.",
    "PLTR": "Palantir needs enterprise AI demand to keep supporting premium multiples.",
    "HIMS": "Hims needs GLP-1 and core subscription strength to support a longer growth hold.",
}

STRUCTURAL_LONG_SEGMENTS = {
    "AI Compute",
    "Optics",
    "Memory",
    "Interconnect",
    "Megacap AI",
    "Neo Cloud",
    "AI Infrastructure",
    "Pre-IPO",
    "Crypto Equity Beta",
    "Healthcare / GLP-1",
}

SEGMENT_ORDER = [
    "AI Compute",
    "Optics",
    "Memory",
    "Interconnect",
    "Megacap AI",
    "Neo Cloud",
    "AI Infrastructure",
    "Pre-IPO",
    "Crypto Equity Beta",
    "Healthcare / GLP-1",
    "Software",
    "High-Beta Growth",
    "Consumer",
    "Financials",
    "Meme Momentum",
    "Asia Macro",
    "LatAm Macro",
    "Macro Indices",
    "Metals",
    "Energy",
    "Agriculture",
    "FX & Rates",
    "Uranium",
    "Volatility",
    "Other Stocks",
]


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _clip_text(text: Any, limit: int = 120) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    clipped = cleaned[: max(limit - 1, 0)].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(" ,.;:-") + "..."


def _first_nonempty_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _fundamental_driver_text(sig: dict, *, limit: int = 130) -> str:
    fp = dict(sig.get("first_principles") or {})
    driver = _first_nonempty_text(
        fp.get("why_now"),
        fp.get("fundamental_driver"),
        sig.get("first_principles_why_now"),
        sig.get("first_principles_fundamental_driver"),
        sig.get("official_event_summary"),
        sig.get("news_event_summary"),
        sig.get("analyst_revision_summary"),
        sig.get("news_catalyst_summary"),
        sig.get("options_summary"),
    )
    return _clip_text(driver, limit)


def default_tradexyz_assets_config() -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for coin, meta in dict(TRADEXYZ_ASSET_METADATA or {}).items():
        coin_upper = normalize_symbol(coin)
        if not coin_upper:
            continue
        assets[coin_upper] = {
            "display_name": str((meta or {}).get("display_name") or coin_upper),
            "instrument_type": str((meta or {}).get("instrument_type") or "equity").strip().lower(),
            "categories": normalize_asset_category_values((meta or {}).get("categories") or ["other_stocks"]),
            "pre_ipo": bool((meta or {}).get("pre_ipo", False)),
            "venue_symbol": f"xyz:{coin_upper}",
        }
    return assets


def normalize_tradexyz_assets(raw: Any) -> dict[str, dict[str, Any]]:
    assets: dict[str, dict[str, Any]] = {}
    for key, value in dict(raw or {}).items():
        coin = normalize_symbol(key)
        if not coin:
            continue
        meta = dict(value or {}) if isinstance(value, dict) else {}
        categories = normalize_asset_category_values(meta.get("categories") or meta.get("asset_categories") or [])
        assets[coin] = {
            "display_name": str(meta.get("display_name") or meta.get("name") or coin),
            "instrument_type": str(meta.get("instrument_type") or "equity").strip().lower(),
            "categories": categories or ["other_stocks"],
            "pre_ipo": bool(meta.get("pre_ipo", False)),
            "venue_symbol": str(meta.get("venue_symbol") or f"xyz:{coin}"),
        }
    return assets


def segment_for_coin(coin: str, categories: list[str], instrument_type: str) -> str:
    coin_upper = normalize_symbol(coin)
    if coin_upper in SEGMENT_OVERRIDES:
        return SEGMENT_OVERRIDES[coin_upper]
    for category in categories or []:
        segment = CATEGORY_SEGMENTS.get(str(category or "").strip().lower())
        if segment:
            return segment
    if str(instrument_type or "").strip().lower() == "index":
        return "Macro Indices"
    return "Other Stocks"


def segment_key(segment: str) -> str:
    return "_".join(str(segment or "other").strip().lower().replace("/", " ").split()) or "other"


def segment_rank(segment: str) -> int:
    try:
        return SEGMENT_ORDER.index(segment)
    except ValueError:
        return len(SEGMENT_ORDER) + 1


def _name_thesis(coin: str, display_name: str, segment: str, item: dict | None, sig: dict | None) -> str:
    item = dict(item or {})
    sig = dict(sig or {})
    driver = _first_nonempty_text(
        item.get("fundamental_driver"),
        item.get("first_principles_why_now"),
        _fundamental_driver_text(sig, limit=118),
        item.get("headline"),
    )
    if driver:
        return _clip_text(driver, 118)
    override = NAME_THESES.get(normalize_symbol(coin))
    if override:
        return override
    return _clip_text(
        f"{display_name or coin} gives the agent {str(segment or 'Trade.xyz').lower()} exposure; wait for fundamentals, flows, and price to align.",
        118,
    )


def _htf_view(
    coin: str,
    segment: str,
    item: dict | None,
    sig: dict | None,
    pos: dict | None,
    config: dict | None = None,
) -> tuple[str, str, bool, float]:
    item = dict(item or {})
    sig = dict(sig or {})
    pos = dict(pos or {})
    config = dict(config or {})
    status = str(item.get("status") or "").upper()
    action = str(sig.get("action") or item.get("action") or item.get("candidate_action") or "").upper()
    bias = str(item.get("bias") or sig.get("market_map_bias") or "NEUTRAL").upper()
    direction = str(pos.get("direction") or "").upper()
    conviction = max(
        _safe_float(item.get("thesis_conviction_score")),
        _safe_float(sig.get("thesis_conviction_score")),
        _safe_float(sig.get("score")),
        _safe_float(item.get("score")),
    )
    sequence = max(
        _safe_float(item.get("first_principles_sequence_score")),
        _safe_float(sig.get("first_principles_sequence_score")),
    )
    fundamental = max(
        _safe_float(item.get("first_principles_fundamental_score")),
        _safe_float(sig.get("first_principles_fundamental_score")),
    )
    confidence = str(sig.get("confidence") or item.get("confidence") or "").upper()
    strong_confidence = confidence in {"HIGH", "VERY_HIGH", "EXTREME"}
    structural = segment in STRUCTURAL_LONG_SEGMENTS
    bearish = (
        direction == "SHORT"
        or status in {"OPEN_SHORT", "READY_SHORT", "WATCH_SHORT", "WAIT_BREAKDOWN"}
        or action == "SHORT"
        or bias == "BEARISH"
    )
    bullish = (
        direction == "LONG"
        or status in {"OPEN_LONG", "READY_LONG", "WATCH_LONG", "WAIT_RECLAIM"}
        or action == "LONG"
        or bias == "BULLISH"
    )
    core_thesis = dict(sig.get("core_thesis") or pos.get("core_thesis") or {})
    core_names = {
        normalize_symbol(value)
        for value in (config.get("core_long_thesis_coins") or [])
        if normalize_symbol(value)
    }
    core_eligible = bool(core_thesis.get("eligible") or normalize_symbol(coin) in core_names)
    core_broken = bool(core_thesis.get("break_confirmed"))
    core_countertrend = bool(core_thesis.get("countertrend") or bearish)

    if direction == "LONG" or status == "OPEN_LONG":
        if core_eligible and not core_broken:
            label = "Core pullback" if core_countertrend else "Core long"
            reason = str(core_thesis.get("summary") or "Core long thesis remains active through short-term volatility.")
            return label, reason, True, max(conviction, sequence, fundamental)
        return "HTF hold", "Already long; hold while thesis and invalidation stay intact.", True, max(conviction, sequence)
    if core_eligible:
        if core_broken:
            return "Thesis broken", "Price damage and fundamental deterioration confirmed the strategic break.", False, max(conviction, sequence, fundamental)
        if core_countertrend:
            return (
                "Core pullback",
                str(core_thesis.get("summary") or "Tactical weakness only; keep the strategic long bias and do not flip short."),
                True,
                max(conviction, sequence, fundamental),
            )
        return (
            "Core long",
            str(core_thesis.get("summary") or "Curated long-term thesis; wait for a clean entry and stay with it through noise."),
            True,
            max(conviction, sequence, fundamental),
        )
    if bearish:
        return "No", "No long hold while the live read is bearish.", False, max(conviction, sequence)
    if bullish and (conviction >= 64.0 or sequence >= 65.0 or fundamental >= 70.0 or strong_confidence):
        return "HTF long", "Long bias qualifies; size only inside caps and invalidate fast.", True, max(conviction, sequence, fundamental)
    if structural:
        return "Starter only", "Structurally interesting, but not enough live proof for a full hold yet.", False, max(conviction, sequence, fundamental)
    return "No", "Track only until the driver becomes cleaner.", False, max(conviction, sequence, fundamental)


def _assets_from_state(state: dict) -> dict[str, dict[str, Any]]:
    config = dict((state or {}).get("config") or {})
    assets = normalize_tradexyz_assets(config.get("tradexyz_assets") or {})
    signals = dict((state or {}).get("signals") or {})
    configured = [
        normalize_symbol(coin)
        for coin in (config.get("coins") or [])
        + (config.get("analysis_coins") or [])
        + (config.get("dynamic_analysis_coins") or [])
        if normalize_symbol(coin)
    ]
    for coin in configured:
        sig = dict(signals.get(coin) or {})
        venue_symbol = str(sig.get("venue_symbol") or "").strip().lower()
        price_source = str(sig.get("price_source") or sig.get("price_source_label") or "").strip().lower()
        if coin not in assets and not (venue_symbol.startswith("xyz:") or "trade.xyz" in price_source):
            continue
        instrument_type = instrument_type_for_coin(coin, signal=sig, config=config)
        categories = asset_categories_for_coin(coin, signal=sig, config=config, instrument_type=instrument_type)
        existing = dict(assets.get(coin) or {})
        existing.update({
            "display_name": existing.get("display_name") or str(sig.get("display_name") or coin),
            "instrument_type": instrument_type,
            "categories": existing.get("categories") or categories or ["other_stocks"],
            "venue_symbol": existing.get("venue_symbol") or str(sig.get("venue_symbol") or f"xyz:{coin}"),
        })
        assets[coin] = existing
    return assets


def build_xyz_section(state: dict, board: dict | None = None) -> dict:
    safe_state = dict(state or {})
    config = dict(safe_state.get("config") or {})
    board_items = {
        normalize_symbol(item.get("coin")): dict(item or {})
        for item in list((board or {}).get("items") or [])
        if isinstance(item, dict) and normalize_symbol(item.get("coin"))
    }
    signals = {
        normalize_symbol(coin): dict(sig or {})
        for coin, sig in dict(safe_state.get("signals") or {}).items()
    }
    positions = {
        normalize_symbol(pos.get("coin")): dict(pos or {})
        for pos in list(safe_state.get("positions") or [])
        if isinstance(pos, dict) and normalize_symbol(pos.get("coin"))
    }
    assets = _assets_from_state(safe_state)
    items: list[dict[str, Any]] = []
    segment_counts: dict[str, dict[str, Any]] = {}
    configured_core_names = {
        normalize_symbol(value)
        for value in (config.get("core_long_thesis_coins") or [])
        if normalize_symbol(value)
    }

    for coin in sorted(
        assets.keys(),
        key=lambda value: (
            segment_rank(segment_for_coin(value, assets[value].get("categories") or [], assets[value].get("instrument_type") or "")),
            value,
        ),
    ):
        meta = dict(assets.get(coin) or {})
        sig = dict(signals.get(coin) or {})
        item = dict(board_items.get(coin) or {})
        pos = dict(positions.get(coin) or {})
        core_thesis = dict(sig.get("core_thesis") or pos.get("core_thesis") or {})
        instrument_type = str(meta.get("instrument_type") or instrument_type_for_coin(coin, signal=sig, config=config) or "equity").strip().lower()
        categories = normalize_asset_category_values(meta.get("categories") or item.get("asset_categories") or [])
        if not categories:
            categories = asset_categories_for_coin(coin, signal=sig, config=config, instrument_type=instrument_type)
        category_labels = [asset_category_label(category, config=config) for category in categories]
        segment = segment_for_coin(coin, categories, instrument_type)
        key = segment_key(segment)
        display_name = str(meta.get("display_name") or coin)
        htf_label, htf_reason, htf_hold, htf_score = _htf_view(coin, segment, item, sig, pos, config)
        name_thesis = _name_thesis(coin, display_name, segment, item, sig)
        configured_core = coin in configured_core_names
        row = {
            "coin": coin,
            "name": display_name,
            "venue_symbol": str(meta.get("venue_symbol") or item.get("venue_symbol") or sig.get("venue_symbol") or f"xyz:{coin}"),
            "instrument_type": instrument_type,
            "categories": categories,
            "category_labels": category_labels,
            "segment": segment,
            "segment_key": key,
            "segment_thesis": SEGMENT_THESES.get(segment) or SEGMENT_THESES["Other Stocks"],
            "name_thesis": name_thesis,
            "htf_label": htf_label,
            "htf_reason": htf_reason,
            "htf_long_hold": htf_hold,
            "htf_score": round(htf_score, 1),
            "strategic_bias": str(
                core_thesis.get("strategic_bias")
                or sig.get("strategic_bias")
                or ("LONG" if configured_core else "NEUTRAL")
            ).upper(),
            "tactical_state": str(
                core_thesis.get("tactical_state")
                or sig.get("tactical_state")
                or ("CORE_WATCH" if configured_core else "UNCLASSIFIED")
            ).upper(),
            "current_action": str(item.get("label") or item.get("status") or sig.get("action") or "Watch").strip(),
            "status": str(item.get("status") or "WATCH").upper(),
            "bias": str(item.get("bias") or sig.get("market_map_bias") or "NEUTRAL").upper(),
            "tradable": bool(item.get("tradable", coin in set(config.get("coins") or []))),
            "invalidation": str(item.get("invalidation_short") or item.get("invalidation") or ""),
            "next": str(item.get("trigger") or item.get("next_setup_reason") or ""),
        }
        items.append(row)
        bucket = segment_counts.setdefault(key, {
            "key": key,
            "label": segment,
            "count": 0,
            "htf_count": 0,
        })
        bucket["count"] += 1
        if htf_hold:
            bucket["htf_count"] += 1

    items.sort(
        key=lambda row: (
            segment_rank(str(row.get("segment") or "")),
            0 if row.get("htf_long_hold") else 1,
            -_safe_float(row.get("htf_score")),
            str(row.get("coin") or ""),
        )
    )
    segments = sorted(
        segment_counts.values(),
        key=lambda row: (segment_rank(str(row.get("label") or "")), str(row.get("label") or "")),
    )
    return {
        "title": "xyz",
        "updated_at": safe_state.get("last_cycle"),
        "items": items,
        "segments": segments,
        "summary": {
            "count": len(items),
            "segment_count": len(segments),
            "htf_long_count": sum(1 for item in items if item.get("htf_long_hold")),
            "starter_count": sum(1 for item in items if str(item.get("htf_label") or "") == "Starter only"),
        },
    }
