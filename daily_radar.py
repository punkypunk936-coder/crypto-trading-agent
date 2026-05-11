"""
daily_radar.py - compact daily focus list for the trading dashboard.

The radar is deliberately plain: it turns the agent's heavy signal surface into
the few questions an operator actually needs answered today.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


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


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw = value.replace("|", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
    else:
        raw = []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        text = _safe_str(item).lower()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


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


def _instrument_type(coin: str, signal: dict, item: dict, state: dict) -> str:
    config = dict((state or {}).get("config") or {})
    types = dict(config.get("instrument_types") or {})
    return _safe_str(
        item.get("instrument_type") or signal.get("instrument_type") or types.get(coin),
        "crypto",
    ).lower()


def _asset_bucket(instrument_type: str) -> str:
    return "coin" if instrument_type == "crypto" else "equity"


def _categories(coin: str, signal: dict, item: dict, state: dict) -> list[str]:
    config = dict((state or {}).get("config") or {})
    categories = _as_list(item.get("asset_categories") or item.get("asset_category"))
    if categories:
        return categories
    categories = _as_list(signal.get("asset_categories") or signal.get("asset_category"))
    if categories:
        return categories
    categories = _as_list((config.get("asset_categories") or {}).get(coin))
    if categories:
        return categories
    itype = _instrument_type(coin, signal, item, state)
    if itype == "equity":
        return ["other_stocks"]
    if itype == "index":
        return ["indices_macro"]
    return ["crypto"]


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


def _merit_text(coin: str, direction: str, fp: dict, signal: dict, item: dict, map_entry: dict) -> str:
    explicit = _safe_str(
        fp.get("plain_thesis")
        or item.get("headline")
        or item.get("why_this_lead")
        or signal.get("thesis_summary")
        or signal.get("decision_reason")
        or signal.get("news_catalyst_summary")
        or signal.get("news_event_summary")
        or map_entry.get("summary")
    )
    if explicit:
        return _short(explicit, 132)
    if direction == "LONG":
        return f"{coin} has enough fundamental, attention, and setup pressure to stay on the bullish radar."
    if direction == "SHORT":
        return f"{coin} has enough weakness or crowded-risk pressure to stay on the bearish radar."
    return f"{coin} is tracked, but merit is not strong enough for a directional call yet."


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
        fp.get("what_matters")
        or signal.get("official_event_summary")
        or signal.get("analyst_revision_summary")
        or "Business/fundamental edge is still being measured."
    )
    attention = _safe_str(
        signal.get("social_attention_summary")
        or signal.get("news_catalyst_summary")
        or "Attention/flow read is not loud yet."
    )
    flows = _safe_str(
        signal.get("orderbook_summary")
        or signal.get("funding_oi_cvd_summary")
        or signal.get("execution_quality_summary")
        or "Flow confirmation is still forming."
    )
    price = _safe_str(
        signal.get("market_map_summary")
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
        instrument_type = _instrument_type(coin, signal, item, safe_state)
        categories = _categories(coin, signal, item, safe_state)
        scope, add_candidate = _scope_text(direction, signal, item, position, score)
        invalidation = _invalidation(direction, fp, signal, item, map_entry, position)
        rows.append({
            "coin": coin,
            "direction": direction,
            "side": _side(direction),
            "status": status,
            "radar_score": score,
            "instrument_type": instrument_type,
            "asset_bucket": _asset_bucket(instrument_type),
            "categories": categories,
            "theme": _safe_str(fp.get("theme") or (categories[0] if categories else ""), "crypto"),
            "tradable": bool(item.get("tradable")) or _safe_str(signal.get("execution_mode")) == "tradable" or bool(position),
            "open_position": bool(position),
            "add_candidate": add_candidate,
            "why": _merit_text(coin, direction, fp, signal, item, map_entry),
            "next": _next_text(direction, signal, item, map_entry, position),
            "invalidation": invalidation,
            "scope": scope,
            "stick_with_it": "Stay with it until invalidation hits." if direction in {"LONG", "SHORT"} else "No thesis to hold yet.",
            "first_principles": _sequence_steps(fp, signal, map_entry),
            "sequence_score": round(_safe_float(fp.get("sequence_score")), 1),
            "fundamental_score": round(_safe_float(fp.get("fundamental_score")), 1),
            "attention_score": round(_safe_float(fp.get("attention_score")), 1),
            "price": _safe_float(signal.get("live_price") or signal.get("price") or position.get("current_price")),
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
    }
