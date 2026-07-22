"""
daily_radar.py - compact daily focus list for the trading dashboard.

The radar is deliberately plain: it turns the agent's heavy signal surface into
the few questions an operator actually needs answered today.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from asset_context import asset_bucket, asset_categories_for_coin, instrument_type_for_coin


COMPANY_DRIVERS: dict[str, str] = {
    "AAPL": "services growth, device demand, and the next AI-enabled upgrade cycle",
    "AMD": "AI accelerators and server CPUs expanding the data-centre revenue runway",
    "AMZN": "AWS growth, Trainium adoption, and improving retail operating leverage",
    "GOOGL": "Search cash flow funding Cloud, Gemini, TPU capacity, and Waymo optionality",
    "META": "ad monetization funding a durable AI infrastructure and engagement cycle",
    "MSFT": "Azure and Copilot converting AI demand into recurring enterprise revenue",
    "NVDA": "Blackwell and Rubin demand sustaining the accelerated-compute platform cycle",
    "TSM": "leading-edge capacity and advanced packaging capturing broad AI-chip demand",
    "MU": "HBM and server-memory demand tightening supply and lifting memory economics",
    "SNDK": "NAND discipline and enterprise SSD demand improving flash pricing power",
    "SKHX": "HBM leadership and AI-memory demand supporting a multi-year capacity cycle",
    "MRVL": "custom silicon and interconnect demand broadening the AI infrastructure buildout",
    "INTC": "Xeon demand and 18A foundry execution determining whether the turnaround compounds",
    "CRWV": "contracted GPU-cloud capacity converting AI demand into a visible revenue backlog",
    "CBRS": "wafer-scale deployments offering differentiated exposure to AI inference demand",
    "COIN": "crypto trading activity, stablecoin economics, and institutional adoption driving operating leverage",
    "HIMS": "subscriber growth and retention determining whether consumer-health distribution compounds",
    "COST": "membership renewal, traffic, and pricing power sustaining defensive earnings growth",
    "BABA": "cloud and AI monetization improving while China consumption expectations reset",
}

SECTOR_DRIVERS: dict[str, str] = {
    "semis_memory": "AI compute demand, supply discipline, and estimate revisions",
    "mag7": "earnings durability, AI monetization, and capital-spending returns",
    "neoclouds": "contracted capacity, financing discipline, and GPU-cloud utilization",
    "ai_infra": "AI infrastructure deployments, backlog conversion, and customer concentration",
    "crypto_equities": "crypto volumes, stablecoin economics, and risk appetite",
    "consumer": "traffic, pricing power, margins, and forward guidance",
    "biotech_glp1": "patient growth, retention, access, and margin durability",
    "financials": "asset growth, fee income, credit quality, and capital returns",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _short(value: Any, limit: int = 120) -> str:
    text = " ".join(_safe_str(value).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _coin_maps(state: dict, market_map: dict, action_board: dict) -> tuple[dict, dict, dict, dict]:
    signals = dict((state or {}).get("signals") or {})
    positions = {
        _safe_str(pos.get("coin")).upper(): dict(pos or {})
        for pos in (state or {}).get("positions") or []
        if _safe_str((pos or {}).get("coin"))
    }
    map_entries = dict((market_map or {}).get("coins") or {})
    board_items = {
        _safe_str(item.get("coin")).upper(): dict(item or {})
        for item in (action_board or {}).get("items") or []
        if _safe_str((item or {}).get("coin"))
    }
    return signals, positions, map_entries, board_items


def _direction(signal: dict, item: dict, position: dict, map_entry: dict) -> str:
    if position:
        direction = _safe_str(position.get("direction")).upper()
        if direction in {"LONG", "SHORT"}:
            return direction
    status = _safe_str(item.get("status") or signal.get("asset_state")).upper()
    if "SHORT" in status or "BREAKDOWN" in status:
        return "SHORT"
    if "LONG" in status or "RECLAIM" in status:
        return "LONG"
    action = _safe_str(item.get("action") or signal.get("action")).upper()
    if action in {"LONG", "SHORT"}:
        return action
    fp_direction = _safe_str(signal.get("first_principles_direction")).upper()
    if fp_direction in {"LONG", "SHORT"} and _safe_float(signal.get("first_principles_sequence_score")) >= 58.0:
        return fp_direction
    bias = _safe_str(item.get("bias") or signal.get("market_map_bias") or map_entry.get("bias")).upper()
    if bias == "BULLISH":
        return "LONG"
    if bias == "BEARISH":
        return "SHORT"
    score = _safe_float(signal.get("score") or item.get("score"), 50.0)
    if score >= 57.0:
        return "LONG"
    if score <= 43.0:
        return "SHORT"
    return "FLAT"


def _side(direction: str) -> str:
    if direction == "LONG":
        return "bullish"
    if direction == "SHORT":
        return "bearish"
    return "neutral"


def _status_label(item: dict, signal: dict, position: dict, direction: str) -> str:
    if position:
        return "OPEN"
    status = _safe_str(item.get("status") or signal.get("asset_state")).upper()
    if status == "PENDING_ENTRY":
        return "ORDER LIVE"
    if status in {"READY_LONG", "READY_SHORT", "EXECUTABLE", "PASSIVE_ENTRY"}:
        return "READY"
    if direction in {"LONG", "SHORT"}:
        return "WATCH"
    return "WAIT"


def _first_principles(signal: dict) -> dict:
    view = dict(signal.get("first_principles") or {})
    return {
        "why_now": _safe_str(view.get("why_now") or signal.get("first_principles_why_now")),
        "fundamental_driver": _safe_str(view.get("fundamental_driver") or signal.get("first_principles_fundamental_driver")),
        "attention_driver": _safe_str(view.get("attention_driver") or signal.get("first_principles_attention_driver")),
        "flow_driver": _safe_str(view.get("flow_driver") or signal.get("first_principles_flow_driver")),
        "price_confirmation": _safe_str(view.get("price_confirmation") or signal.get("first_principles_price_confirmation")),
        "plain_thesis": _safe_str(view.get("plain_thesis") or signal.get("first_principles_plain_thesis")),
        "likely_path": _safe_str(view.get("likely_path") or signal.get("first_principles_likely_path")),
        "wrong_if": _safe_str(view.get("wrong_if") or signal.get("first_principles_wrong_if")),
        "fundamental_score": _safe_float(view.get("fundamental_score") or signal.get("first_principles_fundamental_score")),
        "attention_score": _safe_float(view.get("attention_score") or signal.get("first_principles_attention_score")),
        "flow_score": _safe_float(view.get("flow_score") or signal.get("first_principles_flow_score")),
        "price_score": _safe_float(view.get("price_score") or signal.get("first_principles_price_score")),
        "sequence_score": _safe_float(view.get("sequence_score") or signal.get("first_principles_sequence_score")),
        "theme": _safe_str(view.get("theme") or signal.get("first_principles_theme")),
        "what_matters": _safe_str(view.get("what_matters") or signal.get("first_principles_what_matters")),
    }


def _sentence(value: Any) -> str:
    text = " ".join(_safe_str(value).split()).strip(" ,.;")
    return text[:1].upper() + text[1:] if text else ""


def _revision_evidence(signal: dict) -> tuple[str, str]:
    summary = _safe_str(signal.get("analyst_revision_summary"))
    score = _safe_float(signal.get("analyst_revision_score"))
    if not summary:
        return "", "neutral"
    eps = re.search(r"EPS revisions\s+(\d+)\s+up/(\d+)\s+down", summary, flags=re.IGNORECASE)
    street = re.search(
        r"street\s+(\d+)\s+buy(?:/(\d+)\s+hold)?/(\d+)\s+sell|street\s+(\d+)\s+buy\s+vs\s+(\d+)\s+sell",
        summary,
        flags=re.IGNORECASE,
    )
    parts: list[str] = []
    if eps:
        parts.append(f"EPS revisions are {eps.group(1)} up versus {eps.group(2)} down")
    if street:
        buys = street.group(1) or street.group(4)
        sells = street.group(3) or street.group(5)
        parts.append(f"the street is {buys} buy versus {sells} sell")
    if not parts:
        bias = "positive" if score > 0.75 else "negative" if score < -0.75 else "mixed"
        parts.append(f"the live analyst-revision read is {bias}")
    tone = "positive" if score > 0.75 else "negative" if score < -0.75 else "mixed"
    return "; ".join(parts), tone


def _price_evidence(signal: dict, map_entry: dict) -> str:
    summary = _safe_str(signal.get("market_map_summary") or map_entry.get("summary")).lower()
    if "reclaim was confirmed" in summary and "slipped back below" in summary:
        return "the larger reclaim is intact, but live price is retesting it from below"
    if "breakdown was confirmed" in summary and "bounced back above" in summary:
        return "buyers absorbed the daily breakdown and reclaimed the level"
    if "testing mapped support" in summary:
        return "price is testing mapped support rather than breaking structure"
    if "sitting in mapped demand" in summary and "pressing mapped resistance" in summary:
        return "price is holding demand but still has to clear mapped resistance"
    if "sitting in mapped demand" in summary:
        return "price is holding inside mapped demand"
    if "pressing mapped resistance" in summary:
        return "price is pressing mapped resistance"
    if "bearish" in summary or "breakdown" in summary:
        return "price structure remains damaged and needs a reclaim"
    if "bullish" in summary or "reclaim" in summary or "breakout" in summary:
        return "price structure is confirming the bullish case"
    return _safe_str(signal.get("first_principles_price_confirmation") or signal.get("price_action_summary"))


def _business_driver(coin: str, categories: list[str], fp: dict) -> str:
    if coin in COMPANY_DRIVERS:
        return COMPANY_DRIVERS[coin]
    for category in categories:
        if category in SECTOR_DRIVERS:
            return SECTOR_DRIVERS[category]
    return _safe_str(fp.get("what_matters") or fp.get("fundamental_driver"), "the earnings and demand setup")


def _merit_text(
    coin: str,
    direction: str,
    fp: dict,
    signal: dict,
    item: dict,
    map_entry: dict,
    categories: list[str],
) -> str:
    driver = _business_driver(coin, categories, fp)
    revision, revision_tone = _revision_evidence(signal)
    price = _price_evidence(signal, map_entry)
    event_text = _safe_str(
        signal.get("official_event_summary")
        or signal.get("news_event_summary")
        or fp.get("why_now")
    )
    event_is_specific = bool(event_text and not re.fullmatch(r"[a-z _+\-/]+", event_text.lower()))

    if direction == "LONG":
        if revision_tone == "negative":
            lead = f"Bullish tactically despite a fundamental warning: {revision}"
        else:
            lead = f"Bullish because {driver}"
    elif direction == "SHORT":
        if revision_tone == "positive":
            lead = f"Tactical bearish call, not a durable short: {revision}"
        else:
            lead = f"Bearish because {revision or driver + ' is weakening'}"
    else:
        lead = f"No directional trade yet: {driver} matters, but the evidence is not aligned"

    evidence: list[str] = []
    if revision and not (direction == "SHORT" and revision_tone == "positive") and revision.lower() not in lead.lower():
        evidence.append(revision)
    if event_is_specific:
        evidence.append(_sentence(event_text))
    if price:
        evidence.append(price)
    if not evidence:
        evidence.append(_safe_str(item.get("headline") or fp.get("plain_thesis") or signal.get("decision_reason")))
    joined = ". ".join(_sentence(value) for value in evidence if value)
    thesis = lead.rstrip(".") + (f". {joined}" if joined else "")
    return _short(thesis, 290)


def _next_text(direction: str, signal: dict, item: dict, map_entry: dict, position: dict) -> str:
    if position:
        logic = _safe_str(position.get("current_logic_short") or position.get("current_logic"))
        if logic:
            return _short(logic, 96)
        return "Hold while thesis and invalidation stay intact."
    direct = _safe_str(item.get("trigger") or item.get("next_setup_reason") or signal.get("next_unblock_reason"))
    if direct:
        return _short(direct, 96)
    if direction == "LONG":
        levels = map_entry.get("daily_close_long_above") or []
        positive = [_safe_float(x) for x in levels if _safe_float(x) > 0]
        if positive:
            return f"Buy/press only after {min(positive):,.4g} clears."
    if direction == "SHORT":
        levels = map_entry.get("daily_close_short_below") or []
        positive = [_safe_float(x) for x in levels if _safe_float(x) > 0]
        if positive:
            return f"Short/press only below {max(positive):,.4g}."
    return "Wait for the next clean trigger."


def _invalidation(direction: str, fp: dict, signal: dict, item: dict, map_entry: dict, position: dict) -> str:
    direct = _safe_str(
        item.get("invalidation_short")
        or item.get("invalidation")
        or fp.get("wrong_if")
        or signal.get("first_principles_wrong_if")
    )
    if direct:
        return _short(direct.replace("Wrong if", "Invalid if"), 96)
    stop = _safe_float(position.get("stop_loss") or signal.get("planned_stop_loss"))
    if stop:
        return ("Invalid above " if direction == "SHORT" else "Invalid below ") + f"{stop:,.4g}"
    if direction == "LONG":
        support = _safe_float(signal.get("market_map_nearest_support"))
        supports = [_safe_float(x) for x in map_entry.get("supports") or [] if _safe_float(x) > 0]
        if supports:
            support = max(support, max(supports))
        return f"Invalid below {support:,.4g}" if support else "Invalid if catalyst/flows fade."
    if direction == "SHORT":
        resistance = _safe_float(signal.get("market_map_nearest_resistance"))
        resistances = [_safe_float(x) for x in map_entry.get("resistances") or [] if _safe_float(x) > 0]
        if resistances:
            resistance = min([x for x in [resistance] + resistances if x > 0])
        return f"Invalid above {resistance:,.4g}" if resistance else "Invalid if supply is reclaimed."
    return "No trade until invalidation is explicit."


def _durable_profile(
    coin: str,
    *,
    categories: list[str],
    config: dict,
    fp: dict,
    signal: dict,
    tactical_direction: str,
    tactical_invalidation: str,
    thesis: str,
) -> dict:
    core_names = {str(value or "").upper() for value in config.get("core_long_thesis_coins") or []}
    semiconductor = "semis_memory" in categories
    eligible = coin in core_names or semiconductor
    fundamental = _safe_float(fp.get("fundamental_score"))
    sequence = _safe_float(fp.get("sequence_score"))
    revision = _safe_float(signal.get("analyst_revision_score"))
    if not eligible or fundamental < 62.0 or sequence < 66.0 or revision < -0.75:
        return {"active": False}

    strength = min(
        100.0,
        fundamental * 0.42
        + sequence * 0.38
        + max(0.0, revision) * 2.2
        + (8.0 if coin in core_names else 4.0),
    )
    if strength < 72.0:
        return {"active": False}
    maturity = "DURABLE" if strength >= 88.0 else "DEVELOPING"
    cycles = 4 if semiconductor else int(config.get("core_long_break_confirmation_cycles") or 3)
    invalidation = tactical_invalidation or "Invalid if price structure and revisions break together."
    level_clause = invalidation.replace("Wrong ", "").replace("Invalid ", "").strip().rstrip(".")
    driver = _business_driver(coin, categories, fp)
    if level_clause.lower().startswith(("above ", "below ")):
        price_condition = f"price remains {level_clause}"
    else:
        price_condition = f"the tactical invalidation persists ({invalidation.rstrip('.')})"
    strategic_invalidation = (
        f"Thesis breaks only if {driver} deteriorates, revisions turn negative, and {price_condition} "
        f"for {cycles} fresh checks."
    )
    tactical_state = "TACTICAL PULLBACK" if tactical_direction == "SHORT" else "TACTICALLY ALIGNED" if tactical_direction == "LONG" else "WAITING FOR ENTRY"
    return {
        "active": True,
        "coin": coin,
        "label": maturity,
        "strength": round(strength, 1),
        "strategic_direction": "LONG",
        "tactical_direction": tactical_direction,
        "tactical_state": tactical_state,
        "thesis": thesis,
        "invalidation": _short(strategic_invalidation, 240),
        "categories": categories,
    }


def _scope_text(direction: str, signal: dict, item: dict, position: dict, score: float) -> tuple[str, bool]:
    status = _status_label(item, signal, position, direction)
    starter = bool(signal.get("conviction_entry_active") or (signal.get("conviction_entry") or {}).get("active"))
    eventish = (
        _safe_float(signal.get("official_event_score"))
        + _safe_float(signal.get("news_event_score"))
        + _safe_float(signal.get("news_catalyst_score"))
    ) >= 3.0
    tradable = bool(item.get("tradable")) or _safe_str(signal.get("execution_mode")) == "tradable"
    if position and score >= 70.0:
        return "Can add small only if price confirms; keep original inval.", True
    if position:
        return "Hold, do not add until the next trigger improves.", False
    if starter:
        return "Small starter allowed inside event-risk caps.", True
    if tradable and direction in {"LONG", "SHORT"} and (score >= 68.0 or eventish):
        return "Starter/entry allowed only after the listed trigger.", True
    if direction in {"LONG", "SHORT"}:
        return "Track only until conviction improves.", False
    return "No position. Keep watching.", False


def _sequence_steps(fp: dict, signal: dict, map_entry: dict) -> list[dict]:
    fundamentals = _safe_str(
        fp.get("fundamental_driver")
        or fp.get("why_now")
        or signal.get("official_event_summary")
        or signal.get("news_event_summary")
        or signal.get("analyst_revision_summary")
        or signal.get("news_catalyst_summary")
        or fp.get("what_matters")
        or "Business/fundamental edge is still being measured."
    )
    attention = _safe_str(
        fp.get("attention_driver")
        or signal.get("social_attention_summary")
        or signal.get("news_catalyst_summary")
        or "Attention/flow read is not loud yet."
    )
    flows = _safe_str(
        fp.get("flow_driver")
        or signal.get("orderbook_summary")
        or signal.get("funding_oi_cvd_summary")
        or signal.get("execution_quality_summary")
        or "Flow confirmation is still forming."
    )
    price = _safe_str(
        fp.get("price_confirmation")
        or signal.get("market_map_summary")
        or map_entry.get("summary")
        or "Price map is waiting for the next trigger."
    )
    return [
        {"label": "Fundamentals", "score": round(_safe_float(fp.get("fundamental_score")), 1), "text": _short(fundamentals, 82)},
        {"label": "Attention", "score": round(_safe_float(fp.get("attention_score")), 1), "text": _short(attention, 82)},
        {"label": "Flows", "score": round(_safe_float(fp.get("flow_score")), 1), "text": _short(flows, 82)},
        {"label": "Price", "score": round(_safe_float(fp.get("price_score")), 1), "text": _short(price, 82)},
    ]


def _radar_score(signal: dict, item: dict, position: dict, direction: str, proactive_coins: set[str], coin: str) -> float:
    base_score = _safe_float(item.get("score") or signal.get("score"), 50.0)
    fp = _first_principles(signal)
    score = 28.0 + abs(base_score - 50.0) * 0.75
    score += max(0.0, _safe_float(fp.get("sequence_score")) - 50.0) * 0.38
    score += max(0.0, _safe_float(fp.get("fundamental_score")) - 50.0) * 0.24
    score += max(0.0, _safe_float(fp.get("attention_score")) - 50.0) * 0.20
    score += _safe_float(signal.get("official_event_score")) * 4.5
    score += _safe_float(signal.get("news_event_score")) * 4.0
    score += _safe_float(signal.get("news_catalyst_score")) * 3.6
    score += max(0.0, _safe_float(signal.get("analyst_revision_score"))) * 2.2
    score += max(0.0, _safe_float(item.get("probability_pct")) - 50.0) * 0.30
    if position:
        score += 14.0
    if bool(signal.get("conviction_entry_active") or (signal.get("conviction_entry") or {}).get("active")):
        score += 9.0
    if bool(item.get("tradable")) or _safe_str(signal.get("execution_mode")) == "tradable":
        score += 5.0
    if coin in proactive_coins:
        score += 7.0
    if direction == "FLAT":
        score *= 0.72
    return round(max(0.0, min(100.0, score)), 2)


def _proactive_focus_coins(proactive_report: dict) -> set[str]:
    coins: set[str] = set()
    scout = dict((proactive_report or {}).get("morning_scout_book") or {})
    basket = dict((proactive_report or {}).get("starter_basket_optimizer") or {})
    ledger = dict((proactive_report or {}).get("thesis_ledger") or {})
    for key in ("bullish_calls", "bearish_calls"):
        for row in scout.get(key) or []:
            coin = _safe_str((row or {}).get("coin")).upper()
            if coin:
                coins.add(coin)
    for row in basket.get("allocations") or []:
        coin = _safe_str((row or {}).get("coin")).upper()
        if coin:
            coins.add(coin)
    for row in ledger.get("active_theses") or []:
        coin = _safe_str((row or {}).get("coin")).upper()
        if coin:
            coins.add(coin)
    return coins


def build_daily_radar(
    state: dict,
    market_map: dict | None = None,
    action_board: dict | None = None,
    proactive_report: dict | None = None,
    *,
    limit: int = 12,
) -> dict:
    """Build a compact daily radar from the current state and research stack."""

    safe_state = dict(state or {})
    safe_market_map = dict(market_map or {})
    safe_action_board = dict(action_board or {})
    proactive_coins = _proactive_focus_coins(dict(proactive_report or {}))
    signals, positions, map_entries, board_items = _coin_maps(safe_state, safe_market_map, safe_action_board)
    config = dict(safe_state.get("config") or {})
    configured = set(config.get("coins") or []) | set(config.get("analysis_coins") or []) | set(config.get("dynamic_analysis_coins") or [])
    coins = {
        _safe_str(coin).upper()
        for coin in set(signals) | set(positions) | set(map_entries) | set(board_items) | proactive_coins | configured
        if _safe_str(coin)
    }

    rows: list[dict] = []
    for coin in sorted(coins):
        signal = dict(signals.get(coin) or {})
        item = dict(board_items.get(coin) or {})
        position = dict(positions.get(coin) or {})
        map_entry = dict(map_entries.get(coin) or {})
        fp = _first_principles(signal)
        direction = _direction(signal, item, position, map_entry)
        score = _radar_score(signal, item, position, direction, proactive_coins, coin)
        status = _status_label(item, signal, position, direction)
        if score < 36.0 and status not in {"OPEN", "ORDER LIVE", "READY"} and coin not in proactive_coins:
            continue
        instrument_type = instrument_type_for_coin(coin, signal=signal, item=item, state=safe_state)
        categories = asset_categories_for_coin(
            coin,
            signal=signal,
            item=item,
            state=safe_state,
            instrument_type=instrument_type,
        )
        scope, add_candidate = _scope_text(direction, signal, item, position, score)
        invalidation = _invalidation(direction, fp, signal, item, map_entry, position)
        thesis = _merit_text(coin, direction, fp, signal, item, map_entry, categories)
        durable = _durable_profile(
            coin,
            categories=categories,
            config=config,
            fp=fp,
            signal=signal,
            tactical_direction=direction,
            tactical_invalidation=invalidation,
            thesis=thesis,
        )
        rows.append({
            "coin": coin,
            "direction": direction,
            "side": _side(direction),
            "status": status,
            "radar_score": score,
            "instrument_type": instrument_type,
            "asset_bucket": asset_bucket(instrument_type),
            "categories": categories,
            "theme": _safe_str(fp.get("theme") or (categories[0] if categories else ""), "crypto"),
            "tradable": bool(item.get("tradable")) or _safe_str(signal.get("execution_mode")) == "tradable" or bool(position),
            "open_position": bool(position),
            "add_candidate": add_candidate,
            "why": thesis,
            "thesis": thesis,
            "next": _next_text(direction, signal, item, map_entry, position),
            "invalidation": invalidation,
            "scope": scope,
            "stick_with_it": "Stay with it until invalidation hits." if direction in {"LONG", "SHORT"} else "No thesis to hold yet.",
            "first_principles": _sequence_steps(fp, signal, map_entry),
            "sequence_score": round(_safe_float(fp.get("sequence_score")), 1),
            "fundamental_score": round(_safe_float(fp.get("fundamental_score")), 1),
            "attention_score": round(_safe_float(fp.get("attention_score")), 1),
            "price": _safe_float(signal.get("live_price") or signal.get("price") or position.get("current_price")),
            "durable_thesis": bool(durable.get("active")),
            "durable_profile": durable,
        })

    rows.sort(key=lambda row: (
        0 if row.get("open_position") else 1,
        0 if row.get("add_candidate") else 1,
        -_safe_float(row.get("radar_score")),
        str(row.get("coin") or ""),
    ))
    top_assets = rows[: max(1, int(limit or 12))]
    bullish = [row for row in top_assets if row.get("side") == "bullish"]
    bearish = [row for row in top_assets if row.get("side") == "bearish"]
    add_candidates = [row for row in top_assets if row.get("add_candidate")]
    open_assets = [row for row in top_assets if row.get("open_position")]
    durable_theses = sorted(
        [dict(row.get("durable_profile") or {}) for row in rows if row.get("durable_thesis")],
        key=lambda row: (-_safe_float(row.get("strength")), str(row.get("coin") or "")),
    )[:6]
    themes: dict[str, int] = {}
    for row in top_assets:
        theme = _safe_str(row.get("theme"), "other")
        themes[theme] = themes.get(theme, 0) + 1
    top_line = "No high-quality daily radar yet."
    if top_assets:
        lead = top_assets[0]
        top_line = f"{lead['coin']} leads radar: {lead['scope']} {lead['invalidation']}"
    return {
        "enabled": True,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "focus_count": len(top_assets),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "open_count": len(open_assets),
            "add_candidate_count": len(add_candidates),
            "top_line": top_line,
            "themes": themes,
        },
        "top_assets": top_assets,
        "bullish": bullish,
        "bearish": bearish,
        "add_candidates": add_candidates,
        "open_assets": open_assets,
        "durable_theses": durable_theses,
    }
