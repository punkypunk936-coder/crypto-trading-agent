"""
first_principles.py - compact fundamentals-first reasoning for the agent.

The intent is to force every setup through the same mental order:
fundamentals/catalysts, attention/flows, positioning, then price action.
"""

from __future__ import annotations

from typing import Any


ATTENTION_THEMES: dict[str, dict[str, Any]] = {
    "semis_memory": {
        "label": "Semis/AI compute",
        "what_matters": "AI compute demand, supply discipline, capex, revisions",
        "base_attention": 72.0,
    },
    "ai_infra": {
        "label": "AI infrastructure",
        "what_matters": "AI capex, capacity, cloud demand, backlog",
        "base_attention": 70.0,
    },
    "mag7": {
        "label": "Mag7",
        "what_matters": "earnings quality, AI monetization, cloud/ad demand, revisions",
        "base_attention": 64.0,
    },
    "neoclouds": {
        "label": "Neoclouds",
        "what_matters": "GPU access, customer demand, financing, utilization",
        "base_attention": 66.0,
    },
    "crypto_equities": {
        "label": "Crypto equities",
        "what_matters": "crypto beta, ETF/treasury flows, market liquidity",
        "base_attention": 61.0,
    },
    "crypto": {
        "label": "Crypto",
        "what_matters": "liquidity, leverage, funding, token catalyst, unlock risk",
        "base_attention": 56.0,
    },
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value not in (None, "") else default)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _clip(value: Any, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _clamp(value: float, lower: float = 0.0, upper: float = 100.0) -> float:
    return max(lower, min(upper, value))


def _as_categories(raw: Any) -> list[str]:
    if isinstance(raw, str):
        values = raw.replace("|", ",").split(",")
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip().lower()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _categories_for_signal(coin: str, signal: dict, config: Any = None) -> list[str]:
    categories = _as_categories(signal.get("asset_categories") or signal.get("asset_category"))
    if categories:
        return categories
    category_map = {}
    if isinstance(config, dict):
        category_map = dict(config.get("asset_category_map") or config.get("asset_categories") or {})
    elif config is not None:
        category_map = dict(getattr(config, "asset_category_map", {}) or {})
    categories = _as_categories(category_map.get(str(coin or "").upper()))
    if categories:
        return categories
    instrument_type = _safe_str(signal.get("instrument_type"), "crypto").lower()
    if instrument_type == "equity":
        return ["other_stocks"]
    if instrument_type == "index":
        return ["indices_macro"]
    return ["crypto"]


def _theme(categories: list[str]) -> dict[str, Any]:
    for category in categories:
        if category in ATTENTION_THEMES:
            return dict(ATTENTION_THEMES[category])
    return {
        "label": "General",
        "what_matters": "earnings, liquidity, flows, positioning, and clean risk levels",
        "base_attention": 50.0,
    }


def _direction(signal: dict) -> str:
    action = _safe_str(signal.get("action") or signal.get("decision")).upper()
    if action in {"LONG", "SHORT"}:
        return action
    candidate = _safe_str(signal.get("thesis_candidate_action")).upper()
    if candidate in {"LONG", "SHORT"}:
        return candidate
    state = _safe_str(signal.get("asset_state")).upper()
    if "SHORT" in state or "BREAKDOWN" in state:
        return "SHORT"
    if "LONG" in state or "RECLAIM" in state:
        return "LONG"
    bias = _safe_str(signal.get("market_map_bias")).upper()
    if bias == "BULLISH":
        return "LONG"
    if bias == "BEARISH":
        return "SHORT"
    score = _safe_float(signal.get("score"), 50.0)
    if score >= 56.0:
        return "LONG"
    if score <= 44.0:
        return "SHORT"
    return "FLAT"


def _level_text(direction: str, signal: dict) -> str:
    stop = _safe_float(signal.get("planned_stop_loss"))
    if stop > 0:
        return ("below " if direction == "LONG" else "above ") + f"{stop:,.4g}"
    if direction == "LONG":
        support = _safe_float(signal.get("market_map_nearest_support") or signal.get("orderbook_support"))
        return f"below {support:,.4g}" if support > 0 else "if catalyst/flow support fades"
    if direction == "SHORT":
        resistance = _safe_float(signal.get("market_map_nearest_resistance") or signal.get("orderbook_resistance"))
        return f"above {resistance:,.4g}" if resistance > 0 else "if price reclaims supply"
    return "until direction becomes explicit"


def build_first_principles_view(coin: str, signal: dict | None, config: Any = None) -> dict:
    signal = dict(signal or {})
    coin = _safe_str(coin).upper()
    categories = _categories_for_signal(coin, signal, config)
    theme = _theme(categories)
    direction = _direction(signal)
    side = "long" if direction == "LONG" else "short" if direction == "SHORT" else "flat"

    event_score = max(
        _safe_float(signal.get("news_event_score")),
        _safe_float(signal.get("official_event_score")),
        _safe_float(signal.get("sec_event_score")),
    )
    catalyst_score = max(
        _safe_float(signal.get("news_catalyst_score")),
        max(0.0, _safe_float(signal.get("analyst_revision_score"))),
    )
    social_score = _safe_float(signal.get("social_attention_score"), 50.0)
    social_mentions = int(_safe_float(signal.get("social_attention_mentions"), 0.0))
    implied_move = _safe_float(signal.get("options_implied_move_pct"))
    score = _safe_float(signal.get("score"), 50.0)
    expectancy_probability = _safe_float(signal.get("expectancy_probability"), 0.50)
    expectancy_r = _safe_float(signal.get("expectancy_expected_r"))

    theme_attention = _safe_float(theme.get("base_attention"), 50.0)
    fundamental_score = _clamp(
        42.0
        + event_score * 8.5
        + catalyst_score * 8.0
        + max(0.0, _safe_float(signal.get("analyst_revision_score"))) * 4.0
        + (7.0 if implied_move >= 4.0 else 0.0)
        + max(0.0, theme_attention - 50.0) * 0.32
    )
    attention_score = _clamp(
        theme_attention * 0.55
        + social_score * 0.35
        + min(14.0, social_mentions * 2.0)
        + event_score * 2.0
        + catalyst_score * 2.0
    )
    flow_score = _clamp(
        attention_score * 0.45
        + _safe_float(signal.get("foc_score"), 50.0) * 0.20
        + _safe_float(signal.get("orderbook_score"), 50.0) * 0.20
        + _safe_float(signal.get("market_map_score_adjustment")) * 1.4
        + 25.0
    )
    price_score = _clamp(50.0 + abs(score - 50.0) * 1.6 + _safe_float(signal.get("market_map_score_adjustment")) * 1.2)
    sequence_score = _clamp(fundamental_score * 0.42 + attention_score * 0.28 + flow_score * 0.18 + price_score * 0.12)

    event_summary = _safe_str(
        signal.get("official_event_summary")
        or signal.get("sec_event_summary")
        or signal.get("news_event_summary")
    )
    catalyst_summary = _safe_str(
        signal.get("analyst_revision_summary")
        or signal.get("news_catalyst_summary")
        or signal.get("news_headline")
    )
    social_summary = _safe_str(signal.get("social_attention_summary"))
    map_summary = _safe_str(signal.get("market_map_summary") or signal.get("price_action_summary"))

    fundamentals_text = event_summary or catalyst_summary or theme.get("what_matters", "")
    flow_text = social_summary or f"{theme.get('label', 'Theme')} attention is the main flow read"
    price_text = map_summary or _safe_str(signal.get("decision_reason") or signal.get("flat_reason"), "price action is a final confirmation layer")

    if direction == "FLAT":
        likely_path = "No prediction yet; wait for fundamentals/flows to point one way."
        plain_thesis = "No clean thesis yet."
    elif sequence_score >= 72.0 and fundamental_score >= 64.0:
        likely_path = f"{coin} can keep moving {side} if fundamentals and attention stay aligned."
        plain_thesis = f"{side.title()} thesis: fundamentals and flows line up before price confirmation."
    elif fundamental_score >= 62.0 or attention_score >= 66.0:
        likely_path = f"{coin} deserves starter-size {side} exposure only while the thesis stays intact."
        plain_thesis = f"{side.title()} watch: thesis is early, so keep risk small."
    else:
        likely_path = f"{coin} needs stronger fundamentals/flows before trusting the {side} setup."
        plain_thesis = f"{side.title()} idea is not proven yet."

    wrong_if = f"Wrong { _level_text(direction, signal) }"
    if direction == "FLAT":
        wrong_if = "Wrong if the setup stays catalyst-light and price stays noisy."

    decision = "wait"
    if direction in {"LONG", "SHORT"} and sequence_score >= 72.0 and expectancy_probability >= 0.56:
        decision = "press"
    elif direction in {"LONG", "SHORT"} and (fundamental_score >= 62.0 or attention_score >= 66.0):
        decision = "starter"
    elif direction in {"LONG", "SHORT"} and price_score > 70.0 and fundamental_score < 55.0:
        decision = "price-only wait"

    return {
        "enabled": True,
        "coin": coin,
        "direction": direction,
        "side": side,
        "categories": categories,
        "theme": theme.get("label", "General"),
        "what_matters": _clip(theme.get("what_matters", ""), 120),
        "fundamental_score": round(fundamental_score, 2),
        "attention_score": round(attention_score, 2),
        "flow_score": round(flow_score, 2),
        "price_score": round(price_score, 2),
        "sequence_score": round(sequence_score, 2),
        "event_score": round(event_score, 2),
        "catalyst_score": round(catalyst_score, 2),
        "social_attention_score": round(social_score, 2),
        "social_attention_mentions": social_mentions,
        "decision": decision,
        "plain_thesis": _clip(plain_thesis, 110),
        "likely_path": _clip(likely_path, 130),
        "wrong_if": _clip(wrong_if, 110),
        "summary": _clip(f"{plain_thesis} {likely_path} {wrong_if}", 220),
        "sequence": [
            {"step": "Fundamentals", "score": round(fundamental_score, 1), "takeaway": _clip(fundamentals_text, 105)},
            {"step": "Attention", "score": round(attention_score, 1), "takeaway": _clip(flow_text, 105)},
            {"step": "Flows", "score": round(flow_score, 1), "takeaway": _clip(_safe_str(signal.get("funding_label") or signal.get("orderbook_interaction") or "positioning read is neutral"), 105)},
            {"step": "Price", "score": round(price_score, 1), "takeaway": _clip(price_text, 105)},
        ],
        "expectancy_probability": round(expectancy_probability, 4),
        "expected_r": round(expectancy_r, 4),
    }
