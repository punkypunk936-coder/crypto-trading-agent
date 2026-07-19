"""Deterministic PnL attribution for the agent, dashboard, and learning stores.

The functions in this module explain recorded facts. They deliberately avoid
inventing market causes when the source data is missing, and surface that
missing context as part of the explanation instead.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _humanize(value: Any) -> str:
    return _safe_str(value).replace("_", " ").strip().lower()


def _directional_price_move(direction: str, entry_price: float, current_price: float) -> tuple[float, float]:
    if entry_price <= 0 or current_price <= 0:
        return 0.0, 0.0
    raw_move_pct = (current_price - entry_price) / entry_price * 100.0
    pnl_move_pct = raw_move_pct if direction == "LONG" else -raw_move_pct
    return raw_move_pct, pnl_move_pct


def _level_conflict(direction: str, interaction: str) -> bool:
    interaction = interaction.upper()
    return bool(
        (direction == "LONG" and interaction in {"AT_RESISTANCE", "BELOW_RESISTANCE"})
        or (direction == "SHORT" and interaction in {"AT_SUPPORT", "ABOVE_SUPPORT"})
    )


def _market_alignment(direction: str, context: Mapping[str, Any]) -> dict:
    structure = _safe_str(context.get("structure_trend"), "UNKNOWN").upper()
    mtf_bias = _safe_str(context.get("mtf_bias"), "UNKNOWN").upper()
    breakout = _safe_str(context.get("orderbook_breakout_state"), "NONE").upper()
    interaction = _safe_str(context.get("orderbook_interaction"), "BETWEEN_LEVELS").upper()

    bullish_structure = structure in {"UP", "UPTREND", "BULLISH"}
    bearish_structure = structure in {"DOWN", "DOWNTREND", "BEARISH"}
    bullish_mtf = mtf_bias in {"UP", "UPTREND", "BULLISH", "LONG"}
    bearish_mtf = mtf_bias in {"DOWN", "DOWNTREND", "BEARISH", "SHORT"}
    bullish_breakout = breakout in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"}
    bearish_breakout = breakout in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"}

    aligned = bool(
        (direction == "LONG" and (bullish_structure or bullish_mtf or bullish_breakout))
        or (direction == "SHORT" and (bearish_structure or bearish_mtf or bearish_breakout))
    )
    against = bool(
        (direction == "LONG" and (bearish_structure or bearish_mtf or bearish_breakout))
        or (direction == "SHORT" and (bullish_structure or bullish_mtf or bullish_breakout))
    )
    return {
        "aligned": aligned,
        "against": against,
        "level_conflict": _level_conflict(direction, interaction),
        "structure_trend": structure,
        "mtf_bias": mtf_bias,
        "breakout_state": breakout,
        "level_interaction": interaction,
    }


def _thesis_view(position: Mapping[str, Any], signal: Mapping[str, Any], entry_context: Mapping[str, Any]) -> dict:
    live_thesis = dict(signal.get("thesis") or {})
    entry_thesis = dict(entry_context.get("thesis") or {})
    thesis = live_thesis or entry_thesis
    state = _safe_str(thesis.get("state") or signal.get("thesis_state"), "UNKNOWN").upper()
    quality = _safe_str(thesis.get("quality") or signal.get("thesis_quality"), "UNKNOWN").upper()
    conflicts = _safe_float(thesis.get("conflict_points", signal.get("thesis_conflict_points", 0.0)))
    guard_active = bool(position.get("loss_realization_guard_active"))
    opposite_signal = _safe_str(signal.get("action")).upper() in {"LONG", "SHORT"} and _safe_str(
        signal.get("action")
    ).upper() != _safe_str(position.get("direction")).upper()
    invalid = state in {"NO_TRADE", "INVALID", "INVALIDATED", "BROKEN"}
    intact = bool(guard_active or (not invalid and not opposite_signal and conflicts < 2.0 and thesis))
    summary = _safe_str(
        position.get("current_logic")
        or thesis.get("summary")
        or signal.get("thesis_summary")
        or position.get("entry_logic"),
        "No thesis narrative was recorded.",
    )
    return {
        "state": state,
        "quality": quality,
        "conflicts": round(conflicts, 2),
        "intact": intact,
        "guard_active": guard_active,
        "summary": summary,
    }


def _driver(code: str, label: str, detail: str, impact: str) -> dict:
    return {"code": code, "label": label, "detail": detail, "impact": impact}


def explain_open_position(position: Mapping[str, Any], signal: Mapping[str, Any] | None = None) -> dict:
    pos = dict(position or {})
    sig = dict(signal or {})
    entry_context = dict(pos.get("entry_context") or {})
    direction = _safe_str(pos.get("direction"), "UNKNOWN").upper()
    coin = _safe_str(pos.get("coin"), "UNKNOWN").upper()
    entry_price = _safe_float(pos.get("entry_price"))
    current_price = _safe_float(pos.get("current_price"))
    notional_usd = _safe_float(pos.get("size_usd"))
    leverage = max(1.0, _safe_float(pos.get("leverage"), 1.0))
    margin_usd = _safe_float(pos.get("margin_usd")) or (notional_usd / leverage if leverage else 0.0)
    reported_pnl = _safe_float(pos.get("unrealised_pnl", pos.get("unrealized_pnl_usd")))
    raw_move_pct, pnl_pct = _directional_price_move(direction, entry_price, current_price)
    calculated_pnl = notional_usd * pnl_pct / 100.0
    pnl_usd = reported_pnl if pos.get("unrealised_pnl") is not None else calculated_pnl
    stop_price = _safe_float(pos.get("loss_realization_hard_stop") or pos.get("stop_loss"))
    risk_pct = abs(entry_price - stop_price) / entry_price * 100.0 if entry_price > 0 and stop_price > 0 else 0.0
    current_r = pnl_pct / risk_pct if risk_pct > 0 else 0.0
    hold_minutes = _safe_float(pos.get("hold_minutes"))
    min_hold_minutes = _safe_float(pos.get("min_hold_minutes"))
    context = {**entry_context, **sig}
    alignment = _market_alignment(direction, context)
    thesis = _thesis_view(pos, sig, entry_context)

    state = "profit" if pnl_usd > 0 else "loss" if pnl_usd < 0 else "flat"
    move_word = "rose" if raw_move_pct > 0 else "fell" if raw_move_pct < 0 else "did not move"
    direction_effect = "helped" if pnl_pct > 0 else "hurt" if pnl_pct < 0 else "did not change"
    headline = (
        f"{coin} {direction} is {pnl_usd:+.2f} unrealized because price {move_word} "
        f"{abs(raw_move_pct):.2f}% from entry; that move {direction_effect} the position."
    )
    calculation = (
        f"{notional_usd:,.2f} notional x {pnl_pct:+.3f}% directional move = "
        f"{calculated_pnl:+.2f} calculated uPnL."
    )

    cause_codes: list[str] = []
    drivers = [
        _driver(
            "PRICE_PATH",
            "Price path",
            f"Entry {entry_price:,.6g} to current {current_price:,.6g}: raw move {raw_move_pct:+.3f}%, {direction} PnL move {pnl_pct:+.3f}%.",
            "positive" if pnl_pct > 0 else "negative" if pnl_pct < 0 else "neutral",
        ),
        _driver(
            "EXPOSURE",
            "Exposure",
            f"{notional_usd:,.2f} notional is backed by about {margin_usd:,.2f} margin at {leverage:.1f}x leverage.",
            "neutral",
        ),
    ]

    if alignment["against"]:
        cause_codes.append("MARKET_STRUCTURE_AGAINST_TRADE")
        drivers.append(_driver(
            "MARKET_STRUCTURE_AGAINST_TRADE",
            "Market structure",
            f"Structure {alignment['structure_trend']}, higher-timeframe bias {alignment['mtf_bias']}, breakout {alignment['breakout_state']} now lean against the {direction.lower()}.",
            "negative",
        ))
    elif alignment["aligned"]:
        cause_codes.append("MARKET_STRUCTURE_ALIGNED")
        drivers.append(_driver(
            "MARKET_STRUCTURE_ALIGNED",
            "Market structure",
            f"Structure {alignment['structure_trend']}, higher-timeframe bias {alignment['mtf_bias']}, breakout {alignment['breakout_state']} still support the {direction.lower()}.",
            "positive",
        ))
    else:
        cause_codes.append("MARKET_STRUCTURE_MIXED")
        drivers.append(_driver(
            "MARKET_STRUCTURE_MIXED",
            "Market structure",
            "Recorded structure is mixed or incomplete, so price path is the only confirmed PnL driver.",
            "neutral",
        ))

    if alignment["level_conflict"]:
        cause_codes.append("ENTRY_INTO_OPPOSING_LEVEL")
        drivers.append(_driver(
            "ENTRY_INTO_OPPOSING_LEVEL",
            "Key level",
            f"The {direction.lower()} is interacting with {alignment['level_interaction'].lower().replace('_', ' ')}, an opposing level for this direction.",
            "negative",
        ))

    if thesis["intact"]:
        drivers.append(_driver("THESIS_INTACT", "Thesis", thesis["summary"], "positive" if pnl_usd >= 0 else "neutral"))
    else:
        cause_codes.append("THESIS_WEAK_OR_INVALID")
        drivers.append(_driver("THESIS_WEAK_OR_INVALID", "Thesis", thesis["summary"], "negative"))

    if pnl_usd < 0 and hold_minutes and min_hold_minutes and hold_minutes < min_hold_minutes:
        cause_codes.append("EARLY_ADVERSE_VOLATILITY")
        drivers.append(_driver(
            "EARLY_ADVERSE_VOLATILITY",
            "Time in trade",
            f"Held {hold_minutes:.0f} of the planned {min_hold_minutes:.0f} minimum minutes; this loss is still inside the early thesis window.",
            "neutral",
        ))

    if state == "loss" and not any(code for code in cause_codes if code not in {"MARKET_STRUCTURE_MIXED"}):
        cause_codes.append("ADVERSE_PRICE_MOVE")
    elif state == "profit" and not alignment["aligned"]:
        cause_codes.append("FAVORABLE_PRICE_MOVE")

    if bool(pos.get("loss_realization_guard_active")) and pnl_usd < 0:
        decision = "HOLD_TO_HARD_INVALIDATION"
        decision_detail = _safe_str(
            pos.get("loss_realization_guard_reason"),
            "Thesis remains intact, so the loss stays unrealized until hard invalidation.",
        )
    elif bool(pos.get("runner_active")):
        decision = "RUNNER_HOLD"
        decision_detail = _safe_str(pos.get("runner_reason"), "The profitable runner remains active.")
    elif not thesis["intact"] and pnl_usd < 0:
        decision = "REVIEW_EXIT"
        decision_detail = "The latest recorded thesis is weak or invalid; exit logic should review the position."
    else:
        decision = "MONITOR"
        decision_detail = _safe_str(pos.get("current_logic"), "Monitor until the recorded invalidation or target is reached.")

    discrepancy = reported_pnl - calculated_pnl
    data_warnings: list[str] = []
    if entry_price <= 0 or current_price <= 0:
        data_warnings.append("Entry or current price is missing, so price attribution is incomplete.")
    if notional_usd <= 0:
        data_warnings.append("Position notional is missing, so dollar attribution is incomplete.")
    if abs(discrepancy) > max(0.05, abs(reported_pnl) * 0.05):
        data_warnings.append(
            f"Reported uPnL differs from price x notional math by {discrepancy:+.2f}; fees, fills, or stale marks may explain it."
        )
    if not sig:
        data_warnings.append("No live signal context was available for structure attribution.")

    if state == "profit":
        primary_cause = "MARKET_STRUCTURE_ALIGNED" if alignment["aligned"] else "FAVORABLE_PRICE_MOVE"
    else:
        primary_cause = next(
            (code for code in cause_codes if code not in {"MARKET_STRUCTURE_ALIGNED", "MARKET_STRUCTURE_MIXED"}),
            cause_codes[0] if cause_codes else "PRICE_PATH",
        )
    reward_r = current_r if risk_pct > 0 else pnl_pct / 1.0
    return {
        "version": 1,
        "scope": "open_position",
        "provisional": True,
        "coin": coin,
        "direction": direction,
        "state": state,
        "headline": headline,
        "summary": f"{headline} {decision_detail}",
        "calculation": calculation,
        "pnl_usd": round(pnl_usd, 4),
        "pnl_pct": round(pnl_pct, 6),
        "raw_price_move_pct": round(raw_move_pct, 6),
        "notional_usd": round(notional_usd, 4),
        "margin_usd": round(margin_usd, 4),
        "leverage": round(leverage, 4),
        "risk_distance_pct": round(risk_pct, 6),
        "current_r_multiple": round(current_r, 6),
        "primary_cause_code": primary_cause,
        "cause_codes": cause_codes,
        "drivers": drivers,
        "thesis": thesis,
        "market_alignment": alignment,
        "decision": decision,
        "decision_detail": decision_detail,
        "reinforcement": {
            "provisional": True,
            "reward_r": round(reward_r, 6),
            "reward_normalized": round(_clamp(reward_r / 2.0, -1.0, 1.0), 6),
            "primary_cause_code": primary_cause,
            "thesis_intact": thesis["intact"],
        },
        "data_quality": {
            "complete": not data_warnings,
            "warnings": data_warnings,
            "reported_minus_calculated_usd": round(discrepancy, 4),
        },
    }


def _closed_context(trade: Mapping[str, Any], record: Mapping[str, Any] | None) -> tuple[dict, dict, dict, dict]:
    source = dict(record or {})
    entry_context = dict(source.get("entry_context") or trade.get("entry_context") or {})
    exit_context = dict(source.get("exit_context") or trade.get("exit_context") or {})
    trade_plan = dict(source.get("trade_plan") or trade.get("trade_plan") or entry_context.get("trade_plan") or {})
    plan_outcome = dict(source.get("plan_outcome") or trade.get("plan_outcome") or {})
    return entry_context, exit_context, trade_plan, plan_outcome


def explain_closed_trade(trade: Mapping[str, Any], record: Mapping[str, Any] | None = None) -> dict:
    item = dict(trade or {})
    source = {**dict(record or {}), **item}
    entry_context, exit_context, trade_plan, plan_outcome = _closed_context(item, record)
    direction = _safe_str(source.get("direction"), "UNKNOWN").upper()
    coin = _safe_str(source.get("coin"), "UNKNOWN").upper()
    entry_price = _safe_float(source.get("entry_price"))
    exit_price = _safe_float(source.get("exit_price"))
    notional_usd = _safe_float(source.get("size_usd"))
    pnl_usd = _safe_float(source.get("pnl_usd"))
    raw_move_pct, calculated_pct = _directional_price_move(direction, entry_price, exit_price)
    pnl_pct = _safe_float(source.get("pnl_pct"), calculated_pct)
    calculated_pnl = notional_usd * calculated_pct / 100.0
    exit_reason = _safe_str(source.get("exit_reason"), "unknown").lower()
    hold_minutes = _safe_float(source.get("hold_minutes", source.get("duration_mins")))
    planned_stop = _safe_float(trade_plan.get("stop_loss") or entry_context.get("planned_stop_loss") or source.get("stop_loss"))
    risk_pct = abs(entry_price - planned_stop) / entry_price * 100.0 if entry_price > 0 and planned_stop > 0 else 0.0
    captured_r = _safe_float(plan_outcome.get("captured_r_multiple"))
    if not captured_r and risk_pct > 0:
        captured_r = pnl_pct / risk_pct
    context = {**entry_context, **exit_context}
    alignment = _market_alignment(direction, context)
    outcome = "WIN" if pnl_usd > 0 else "LOSS" if pnl_usd < 0 else "FLAT"

    cause_codes: list[str] = []
    drivers: list[dict] = [
        _driver(
            "PRICE_PATH",
            "Price path",
            f"Entry {entry_price:,.6g} to exit {exit_price:,.6g}: raw move {raw_move_pct:+.3f}%, {direction} return {pnl_pct:+.3f}%.",
            "positive" if pnl_usd > 0 else "negative" if pnl_usd < 0 else "neutral",
        ),
        _driver(
            "EXIT_TRIGGER",
            "Exit trigger",
            _humanize(exit_reason) or "No exit trigger was recorded.",
            "positive" if pnl_usd > 0 else "negative" if pnl_usd < 0 else "neutral",
        ),
    ]

    if pnl_usd < 0:
        if alignment["against"]:
            cause_codes.append("COUNTERTREND_OR_STRUCTURE_CONFLICT")
            drivers.append(_driver(
                "COUNTERTREND_OR_STRUCTURE_CONFLICT",
                "Structure conflict",
                f"Structure {alignment['structure_trend']}, higher-timeframe bias {alignment['mtf_bias']}, or breakout {alignment['breakout_state']} opposed the {direction.lower()}.",
                "negative",
            ))
        if alignment["level_conflict"]:
            cause_codes.append("ENTRY_INTO_OPPOSING_LEVEL")
            drivers.append(_driver(
                "ENTRY_INTO_OPPOSING_LEVEL",
                "Key-level conflict",
                f"The {direction.lower()} was taken around {alignment['level_interaction'].lower().replace('_', ' ')}, where opposing liquidity could absorb the move.",
                "negative",
            ))
        reason_codes = {
            "stop_loss": "HARD_STOP_HIT",
            "structure_invalidation": "STRUCTURE_INVALIDATION",
            "htf_invalidation": "HIGHER_TIMEFRAME_INVALIDATION",
            "micro_invalidation": "MICROSTRUCTURE_INVALIDATION",
            "signal_reversal": "SIGNAL_REVERSAL",
            "conviction_lost": "THESIS_DECAY",
            "scalp_failed_followthrough": "FAILED_FOLLOW_THROUGH",
            "scalp_time_stop": "NO_FOLLOW_THROUGH",
            "time_stop": "NO_FOLLOW_THROUGH",
            "stale_adverse": "STALE_ADVERSE_POSITION",
        }
        cause_codes.append(reason_codes.get(exit_reason, "ADVERSE_PRICE_MOVE"))
    elif pnl_usd > 0:
        cause_codes.append("TARGET_CAPTURED" if exit_reason == "take_profit" else "FAVORABLE_FOLLOW_THROUGH")
        if alignment["aligned"]:
            cause_codes.append("THESIS_AND_STRUCTURE_ALIGNED")
    else:
        cause_codes.append("FLAT_EXIT")

    execution_quality = dict(source.get("execution_quality") or entry_context.get("execution_quality") or {})
    execution_score = _safe_float(execution_quality.get("score"))
    if execution_score:
        drivers.append(_driver(
            "EXECUTION_QUALITY",
            "Execution",
            _safe_str(execution_quality.get("summary"), f"Recorded execution quality score {execution_score:.0f}."),
            "positive" if execution_score >= 70 else "negative" if execution_score < 45 else "neutral",
        ))

    primary_cause = cause_codes[0] if cause_codes else "PRICE_PATH"
    reward_r = captured_r if risk_pct > 0 or captured_r else pnl_pct / 1.0
    thesis_outcome = "worked" if pnl_usd > 0 else "failed" if any(
        code in cause_codes
        for code in {"COUNTERTREND_OR_STRUCTURE_CONFLICT", "STRUCTURE_INVALIDATION", "HIGHER_TIMEFRAME_INVALIDATION", "THESIS_DECAY"}
    ) else "unresolved"
    entry_credit = -1.0 if any(code in cause_codes for code in {"COUNTERTREND_OR_STRUCTURE_CONFLICT", "ENTRY_INTO_OPPOSING_LEVEL"}) else 0.75 if pnl_usd > 0 else -0.25
    exit_credit = 1.0 if exit_reason == "take_profit" else 0.5 if pnl_usd > 0 else -0.25 if exit_reason in {"time_stop", "scalp_time_stop", "scalp_failed_followthrough"} else 0.0
    headline = (
        f"{coin} {direction} realized {pnl_usd:+.2f}: price moved {raw_move_pct:+.2f}% from entry and the trade closed on "
        f"{_humanize(exit_reason) or 'an unrecorded trigger'}."
    )
    calculation = (
        f"{notional_usd:,.2f} notional x {calculated_pct:+.3f}% directional move = "
        f"{calculated_pnl:+.2f} before any recorded fees or fill differences."
    )
    data_warnings: list[str] = []
    if not entry_context:
        data_warnings.append("Entry context is unavailable; entry-cause attribution is limited.")
    if not exit_context:
        data_warnings.append("Exit market context is unavailable; only the recorded exit reason is confirmed.")
    if not planned_stop:
        data_warnings.append("Planned stop is unavailable, so realized R cannot be independently reconstructed.")
    discrepancy = pnl_usd - calculated_pnl
    if abs(discrepancy) > max(0.05, abs(pnl_usd) * 0.05):
        data_warnings.append(
            f"Reported PnL differs from price x notional math by {discrepancy:+.2f}; fees, fills, or rounded history may explain it."
        )

    return {
        "version": 1,
        "scope": "closed_trade",
        "provisional": False,
        "coin": coin,
        "direction": direction,
        "state": "profit" if pnl_usd > 0 else "loss" if pnl_usd < 0 else "flat",
        "headline": headline,
        "summary": f"{headline} Primary attribution: {_humanize(primary_cause)}.",
        "calculation": calculation,
        "pnl_usd": round(pnl_usd, 4),
        "pnl_pct": round(pnl_pct, 6),
        "raw_price_move_pct": round(raw_move_pct, 6),
        "notional_usd": round(notional_usd, 4),
        "hold_minutes": round(hold_minutes, 4),
        "exit_reason": exit_reason,
        "risk_distance_pct": round(risk_pct, 6),
        "captured_r_multiple": round(captured_r, 6),
        "primary_cause_code": primary_cause,
        "cause_codes": cause_codes,
        "drivers": drivers,
        "market_alignment": alignment,
        "reinforcement": {
            "provisional": False,
            "reward_r": round(reward_r, 6),
            "reward_normalized": round(_clamp(reward_r / 2.0, -1.0, 1.0), 6),
            "primary_cause_code": primary_cause,
            "thesis_outcome": thesis_outcome,
            "entry_credit": round(entry_credit, 4),
            "exit_credit": round(exit_credit, 4),
            "execution_credit": round(_clamp((execution_score - 50.0) / 50.0, -1.0, 1.0), 4) if execution_score else 0.0,
        },
        "data_quality": {
            "complete": not data_warnings,
            "warnings": data_warnings,
            "reported_minus_calculated_usd": round(discrepancy, 4),
        },
    }


def build_pnl_attribution_summary(
    trades: Iterable[Mapping[str, Any]] | None,
    positions: Iterable[Mapping[str, Any]] | None = None,
) -> dict:
    closed = [dict(item or {}) for item in list(trades or [])]
    open_positions = [dict(item or {}) for item in list(positions or [])]
    realized_pnl = sum(_safe_float(item.get("pnl_usd")) for item in closed)
    gross_profit = sum(max(0.0, _safe_float(item.get("pnl_usd"))) for item in closed)
    gross_loss = sum(abs(min(0.0, _safe_float(item.get("pnl_usd")))) for item in closed)
    open_pnl = sum(_safe_float(item.get("unrealised_pnl", item.get("unrealized_pnl_usd"))) for item in open_positions)
    wins = sum(1 for item in closed if _safe_float(item.get("pnl_usd")) > 0)
    losses = sum(1 for item in closed if _safe_float(item.get("pnl_usd")) < 0)
    flats = sum(1 for item in closed if _safe_float(item.get("pnl_usd")) == 0)
    cause_totals: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "loss_usd": 0.0, "coins": set()})
    for item in closed:
        pnl = _safe_float(item.get("pnl_usd"))
        if pnl >= 0:
            continue
        explanation = dict(item.get("pnl_explanation") or explain_closed_trade(item))
        code = _safe_str(explanation.get("primary_cause_code"), "UNATTRIBUTED_LOSS")
        cause_totals[code]["count"] += 1
        cause_totals[code]["loss_usd"] += abs(pnl)
        cause_totals[code]["coins"].add(_safe_str(item.get("coin"), "UNKNOWN").upper())

    top_loss_causes = sorted(
        (
            {
                "code": code,
                "label": _humanize(code),
                "count": data["count"],
                "loss_usd": round(data["loss_usd"], 2),
                "coins": sorted(data["coins"])[:8],
            }
            for code, data in cause_totals.items()
        ),
        key=lambda item: (-item["loss_usd"], -item["count"], item["code"]),
    )
    top = top_loss_causes[0] if top_loss_causes else None
    if realized_pnl < 0:
        headline = (
            f"Realized PnL is {realized_pnl:+.2f}: {losses} losses cost {gross_loss:.2f}, "
            f"more than {wins} wins earned {gross_profit:.2f}; {flats} trades closed flat."
        )
    elif realized_pnl > 0:
        headline = (
            f"Realized PnL is {realized_pnl:+.2f}: {wins} wins earned {gross_profit:.2f} "
            f"against {losses} losses costing {gross_loss:.2f}; {flats} trades closed flat."
        )
    else:
        headline = "Realized PnL is flat across the recorded closed trades."
    if top:
        headline += f" Largest attributed loss bucket: {top['label']} ({top['count']} trades, -{top['loss_usd']:.2f})."

    return {
        "version": 1,
        "headline": headline,
        "realized_pnl_usd": round(realized_pnl, 2),
        "open_unrealized_pnl_usd": round(open_pnl, 2),
        "tracked_pnl_usd": round(realized_pnl + open_pnl, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "closed_trades": len(closed),
        "open_positions": len(open_positions),
        "top_loss_causes": top_loss_causes[:8],
        "data_scope": "Realized PnL uses closed trades; open uPnL is shown separately and remains provisional.",
    }
