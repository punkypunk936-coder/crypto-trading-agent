"""
dashboard/snapshot.py
Shared dashboard payload builder used by the local Flask UI and remote sync.
"""

from __future__ import annotations

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


def _group_action_items(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    labels = {
        "coin": "Coins",
        "equity": "Equity Perps",
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


def action_board(state: dict, market_map: dict) -> dict:
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
    tracked.update(str(coin or "").upper() for coin in signals.keys())
    tracked.update(str(coin or "").upper() for coin in positions_by_coin.keys())

    entries = dict((market_map or {}).get("coins") or {})
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

    for coin in sorted(item for item in tracked if item):
        sig = dict(signals.get(coin) or {})
        pos = positions_by_coin.get(coin)
        map_entry = dict(entries.get(coin) or {})
        tradable = (sig.get("execution_mode") or "observation_only") == "tradable" or pos is not None
        execution_mode = "tradable" if tradable else str(sig.get("execution_mode") or "observation_only")
        bias = str(
            sig.get("market_map_bias")
            or map_entry.get("bias")
            or "NEUTRAL"
        ).upper()
        instrument_type = _instrument_type_for_coin(coin, sig, config)
        asset_bucket = _asset_bucket(instrument_type)
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

        if pos:
            direction = str(pos.get("direction") or "").upper() or action
            status = f"OPEN_{direction or 'LONG'}"
            label = f"In {direction or 'LONG'}"
            headline = _primary_reason(current_logic) or "Trade is live and being managed."
            stop = _safe_float(pos.get("stop_loss"))
            target = _safe_float(pos.get("take_profit"))
            trigger = (
                f"Stop {stop:,.2f} • Target {target:,.2f}"
                if stop > 0 and target > 0
                else "Trade is already open."
            )
            execution_note = "Position is already open and under active management."
        elif asset_state == "PENDING_ENTRY":
            status = "PENDING_ENTRY"
            label = asset_state_label or "Pending entry"
            headline = next_unblock or "A resting limit order is already on the book."
            anchor_price = _safe_float(sig.get("price") or sig.get("live_price"))
            trigger = f"Working order around {anchor_price:,.2f}" if anchor_price > 0 else "Waiting for the resting limit order to resolve."
            execution_note = next_unblock or "The order is already working. The next event is a fill, cancel, or expiry."
        elif state_override or (asset_state == "EXECUTABLE" and action == "FLAT"):
            status = asset_state or "ARMED"
            label = asset_state_label or "Setup pending"
            headline = _primary_reason(next_unblock or current_logic or blocker) or "The setup is still gated."
            if action == "LONG" and live_anchor > 0:
                trigger = f"Long is tracking around {live_anchor:,.2f}"
            elif action == "SHORT" and live_anchor > 0:
                trigger = f"Short is tracking around {live_anchor:,.2f}"
            elif long_trigger and bias == "BULLISH":
                trigger = f"Best long trigger stays above {long_trigger:,.2f}"
            elif short_trigger and bias == "BEARISH":
                trigger = f"Best short trigger stays below {short_trigger:,.2f}"
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
            trigger = (
                f"Watching entry around {(_safe_float(sig.get('live_price')) or _safe_float(sig.get('price'))):,.2f} "
                "once final entry checks stay clean."
            )
            if tradable:
                execution_note = "The thesis qualifies, but the bot still waits for confirmation, sizing, and clean fills before sending the order."
            else:
                execution_note = "The thesis qualifies, but venue support or live market quality is not ready enough to execute yet."
        elif action == "SHORT":
            status = "READY_SHORT"
            label = "Short thesis live"
            headline = _primary_reason(sig.get("decision_reason") or current_logic) or "Short thesis is live."
            trigger = (
                f"Watching entry around {(_safe_float(sig.get('live_price')) or _safe_float(sig.get('price'))):,.2f} "
                "once final entry checks stay clean."
            )
            if tradable:
                execution_note = "The thesis qualifies, but the bot still waits for confirmation, sizing, and clean fills before sending the order."
            else:
                execution_note = "The thesis qualifies, but venue support or live market quality is not ready enough to execute yet."
        elif bias == "BULLISH" and long_trigger and (reclaim_confirmed or bullish_breakout_live):
            status = "WATCH_LONG"
            label = "Bullish watch"
            headline = (
                "Daily bias is bullish, and the reclaim is on the board."
                if not map_summary
                else f"Daily bias is bullish, and {map_summary}."
            )
            if reclaim_lost and not live_reclaim:
                trigger = f"Needs to hold back above {long_trigger:,.2f}"
                execution_note = (
                    "The bot saw the reclaim, but live price slipped back below the trigger. "
                    "It wants that level held again before buying."
                )
            else:
                trigger = f"Reclaim above {long_trigger:,.2f} is already on the board; waiting for cleaner continuation or pullback entry."
                execution_note = (
                    "The bot sees the reclaim, but it still wants stronger continuation, "
                    "safer entry quality, and final sizing checks before buying."
                )
        elif bias == "BULLISH" and bool(sig.get("market_map_block_longs")) and long_trigger:
            status = "WAIT_RECLAIM"
            label = "Wait for reclaim"
            headline = (
                "Daily bias is bullish, but the long is still blocked until price reclaims resistance."
                if not map_summary
                else f"Daily bias is bullish, but {map_summary}."
            )
            trigger = f"Long only after a reclaim above {long_trigger:,.2f}"
            execution_note = "The higher-timeframe view is constructive, but the reclaim still has to confirm before the bot can buy."
        elif bias == "BEARISH" and bool(sig.get("market_map_block_shorts")) and short_trigger:
            status = "WAIT_BREAKDOWN"
            label = "Wait for breakdown"
            headline = (
                "Daily bias is bearish, but the short is still blocked until price breaks support."
                if not map_summary
                else f"Daily bias is bearish, but {map_summary}."
            )
            trigger = f"Short only after a breakdown below {short_trigger:,.2f}"
            execution_note = "The higher-timeframe view is bearish, but the breakdown still has to confirm before the bot can short."
        elif bias == "BULLISH":
            status = "WATCH_LONG"
            label = "Bullish watch"
            headline = (
                "Higher-timeframe bias is bullish, but the entry is not ready."
                if not map_summary
                else f"Higher-timeframe bias is bullish, and {map_summary}."
            )
            trigger = (
                f"Best long trigger is above {long_trigger:,.2f}"
                if long_trigger
                else "Wait for cleaner long confirmation."
            )
            execution_note = "The agent is reading this as bullish context only. It still needs a qualified live thesis before any order can go out."
        elif bias == "BEARISH":
            status = "WATCH_SHORT"
            label = "Bearish watch"
            headline = (
                "Higher-timeframe bias is bearish, but the entry is not ready."
                if not map_summary
                else f"Higher-timeframe bias is bearish, and {map_summary}."
            )
            trigger = (
                f"Best short trigger is below {short_trigger:,.2f}"
                if short_trigger
                else "Wait for cleaner short confirmation."
            )
            execution_note = "The agent is reading this as bearish context only. It still needs a qualified live thesis before any order can go out."
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

        if status in {"WATCH_LONG", "WAIT_RECLAIM", "READY_LONG", "OPEN_LONG"} and support:
            risk = f"Risk if it loses {support:,.2f}"
        elif status in {"WATCH_SHORT", "WAIT_BREAKDOWN", "READY_SHORT", "OPEN_SHORT"} and resistance:
            risk = f"Risk if it reclaims {resistance:,.2f}"
        else:
            risk = map_summary or ""

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
                "asset_state": asset_state,
                "asset_state_label": asset_state_label,
                "next_unblock_reason": next_unblock,
                "status": status,
                "label": label,
                "headline": headline,
                "trigger": trigger,
                "execution_note": execution_note,
                "risk": risk,
                "map_summary": map_summary,
                "confidence": confidence,
                "score": round(score, 1),
                "pnl_usd": _safe_float(pos.get("unrealised_pnl")) if pos else 0.0,
                "llm_referee": dict(sig.get("llm_referee") or {}),
                "llm_referee_summary": str(sig.get("llm_referee_summary") or ""),
                "llm_referee_why_now": str(sig.get("llm_referee_why_now") or ""),
                "execution_coach_verdict": coach_verdict,
                "execution_coach_summary": coach_summary,
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
        "action_board": action_board(shaped_state, normalized_market_map),
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
