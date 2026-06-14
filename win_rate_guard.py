"""
win_rate_guard.py - entry governor focused on lifting quality win rate.

The existing strategy stack can produce good ideas, but a high win-rate system
needs a final veto layer that asks: "Has this kind of setup actually worked?"
This module converts closed-trade history and live setup quality into hard
blocks or size haircuts before exposure reaches the book.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping

import performance_intelligence
import trade_dataset


_HISTORY_CACHE: dict[str, Any] = {"ts": 0.0, "rows": []}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    return text if text else default


def _norm_token(value: Any, default: str = "UNKNOWN") -> str:
    text = _safe_str(value, default).upper()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or default


def _quality_rank(value: Any) -> int:
    return {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
        "EXTREME": 3,
    }.get(_norm_token(value, "LOW"), 0)


def _score_bucket(score: float, direction: str) -> str:
    conviction = abs(_safe_float(score, 50.0) - 50.0)
    if conviction >= 35:
        return "EXTREME"
    if conviction >= 25:
        return "HIGH"
    if conviction >= 15:
        return "MEDIUM"
    return "BASE"


def _prob_bucket(probability: float) -> str:
    p = _safe_float(probability, 0.50)
    if p >= 0.70:
        return "P70"
    if p >= 0.62:
        return "P62"
    if p >= 0.58:
        return "P58"
    if p >= 0.54:
        return "P54"
    return "PLOW"


def _unc_bucket(uncertainty: float) -> str:
    u = _safe_float(uncertainty, 0.50)
    if u <= 0.20:
        return "U20"
    if u <= 0.32:
        return "U32"
    if u <= 0.44:
        return "U44"
    return "UHIGH"


def _rr_bucket(risk_reward: float) -> str:
    rr = _safe_float(risk_reward, 0.0)
    if rr >= 3.0:
        return "RR3"
    if rr >= 2.0:
        return "RR2"
    if rr >= 1.5:
        return "RR15"
    return "RRLOW"


def _entry_context(row: Mapping[str, Any]) -> Mapping[str, Any]:
    context = row.get("entry_context", {}) if isinstance(row, Mapping) else {}
    return context if isinstance(context, Mapping) else {}


def _trade_plan_from(context: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    plan = context.get("trade_plan", {}) if isinstance(context, Mapping) else {}
    if isinstance(plan, Mapping):
        return plan
    return {}


def _thesis_quality_from(context: Mapping[str, Any], row: Mapping[str, Any] | None = None) -> str:
    thesis = context.get("thesis", {}) if isinstance(context, Mapping) else {}
    if isinstance(thesis, Mapping):
        quality = _safe_str(thesis.get("quality"))
        if quality:
            return quality
    return _safe_str(context.get("thesis_quality") if isinstance(context, Mapping) else "", "UNKNOWN")


def _event_like(context: Mapping[str, Any]) -> bool:
    if bool(context.get("event_risk_budget_active") or context.get("conviction_entry_event")):
        return True
    if _safe_float(context.get("news_event_score"), 0.0) > 0:
        return True
    tags = context.get("news_event_tags") or context.get("event_tags") or []
    return isinstance(tags, (list, tuple, set)) and any(_safe_str(tag) for tag in tags)


def _quality_win(row: Mapping[str, Any], trading_cfg) -> bool:
    pnl_pct = _safe_float(row.get("pnl_pct"), 0.0)
    pnl_usd = _safe_float(row.get("pnl_usd"), 0.0)
    min_pct = _safe_float(getattr(trading_cfg, "north_star_quality_win_min_pct", 0.15), 0.15)
    min_usd = _safe_float(getattr(trading_cfg, "north_star_quality_win_min_usd", 0.10), 0.10)
    return pnl_pct >= min_pct or pnl_usd >= min_usd


def _load_history(trading_cfg, *, force_refresh: bool = False) -> list[dict]:
    ttl = max(10.0, _safe_float(getattr(trading_cfg, "win_rate_guard_cache_seconds", 180.0), 180.0))
    now = time.time()
    if not force_refresh and _HISTORY_CACHE.get("rows") and now - _safe_float(_HISTORY_CACHE.get("ts")) < ttl:
        return list(_HISTORY_CACHE.get("rows") or [])
    limit = max(20, _safe_int(getattr(trading_cfg, "win_rate_guard_history_limit", 300), 300))
    try:
        rows = trade_dataset.load_closed_trades(limit=limit)
    except Exception:
        rows = []
    clean_rows = [row for row in rows if isinstance(row, dict)]
    _HISTORY_CACHE["ts"] = now
    _HISTORY_CACHE["rows"] = clean_rows
    return list(clean_rows)


def _snapshot_context(
    *,
    coin: str,
    direction: str,
    signal_snapshot: Mapping[str, Any],
    signal: Any = None,
) -> dict:
    thesis = dict(getattr(signal, "thesis", {}) or signal_snapshot.get("thesis", {}) or {})
    expectancy = dict(getattr(signal, "expectancy", {}) or signal_snapshot.get("expectancy", {}) or {})
    trade_plan = dict(getattr(signal, "trade_plan", {}) or signal_snapshot.get("trade_plan", {}) or {})
    score = _safe_float(getattr(signal, "score", signal_snapshot.get("score", 50.0)), 50.0)
    raw_quality = _safe_str(thesis.get("quality") or signal_snapshot.get("thesis_quality") or signal_snapshot.get("confidence"), "")
    conviction_entry = thesis.get("conviction_entry") if isinstance(thesis.get("conviction_entry"), Mapping) else {}
    eventish = bool(_event_like(signal_snapshot) or (isinstance(conviction_entry, Mapping) and conviction_entry.get("event_conviction")))
    quality = raw_quality or ("MEDIUM" if eventish and score >= 60.0 else "LOW")
    return {
        "coin": _safe_str(coin).upper(),
        "direction": _safe_str(direction).upper(),
        "instrument_type": _safe_str(signal_snapshot.get("instrument_type"), "crypto").lower(),
        "portfolio_theme": _safe_str(signal_snapshot.get("portfolio_theme"), ""),
        "score": score,
        "score_bucket": _score_bucket(score, direction),
        "confidence": _safe_str(getattr(signal, "confidence", signal_snapshot.get("confidence", "")), ""),
        "thesis_quality": quality,
        "quality_rank": _quality_rank(quality),
        "probability": _safe_float(expectancy.get("probability", signal_snapshot.get("expectancy_probability", 0.50)), 0.50),
        "expected_r": _safe_float(expectancy.get("expected_r", signal_snapshot.get("expectancy_expected_r", 0.0)), 0.0),
        "uncertainty": _safe_float(expectancy.get("uncertainty", signal_snapshot.get("expectancy_uncertainty", 0.50)), 0.50),
        "risk_reward": _safe_float(trade_plan.get("risk_reward_ratio", signal_snapshot.get("planned_risk_reward_ratio", 0.0)), 0.0),
        "market_regime": _norm_token(signal_snapshot.get("market_regime"), "RANGING"),
        "dominant_regime": _norm_token(signal_snapshot.get("dominant_regime"), "MIXED"),
        "structure_trend": _norm_token(signal_snapshot.get("structure_trend"), "UNKNOWN"),
        "breakout_state": _norm_token(signal_snapshot.get("orderbook_breakout_state"), "NONE"),
        "level_interaction": _norm_token(signal_snapshot.get("orderbook_interaction"), "BETWEEN_LEVELS"),
        "market_map_bias": _norm_token(signal_snapshot.get("market_map_bias"), "NEUTRAL"),
        "crypto_market_mode": _norm_token(signal_snapshot.get("crypto_market_mode") or signal_snapshot.get("market_mode"), "UNKNOWN"),
        "crypto_directional_bias": _norm_token(signal_snapshot.get("crypto_directional_bias") or signal_snapshot.get("directional_bias"), "NEUTRAL"),
        "crypto_risk_off": bool(signal_snapshot.get("crypto_risk_off") or signal_snapshot.get("risk_off")),
        "event_like": _event_like(signal_snapshot),
        "execution_quality_score": _safe_float(signal_snapshot.get("execution_quality_score"), 0.0),
        "data_reliability_score": _safe_float(signal_snapshot.get("data_reliability_score"), 0.0),
        "analog_sample_size": _safe_int(signal_snapshot.get("analog_sample_size"), 0),
        "analog_win_rate": _safe_float(signal_snapshot.get("analog_win_rate"), 0.0),
        "analog_adverse": bool(signal_snapshot.get("analog_adverse")),
        "analog_hard_block": bool(signal_snapshot.get("analog_hard_block")),
    }


def _row_context(row: Mapping[str, Any]) -> dict:
    context = _entry_context(row)
    plan = _trade_plan_from(context, row)
    expectancy = context.get("expectancy", {}) if isinstance(context.get("expectancy", {}), Mapping) else {}
    thesis = context.get("thesis", {}) if isinstance(context.get("thesis", {}), Mapping) else {}
    direction = _safe_str(row.get("direction") or context.get("action")).upper()
    score = _safe_float(context.get("score", row.get("signal_score", 50.0)), 50.0)
    quality = _safe_str(thesis.get("quality") or context.get("thesis_quality") or context.get("confidence"), "UNKNOWN")
    return {
        "coin": _safe_str(row.get("coin") or context.get("coin")).upper(),
        "direction": direction,
        "instrument_type": _safe_str(context.get("instrument_type"), "crypto").lower(),
        "portfolio_theme": _safe_str(context.get("portfolio_theme"), ""),
        "score": score,
        "score_bucket": _score_bucket(score, direction),
        "thesis_quality": quality,
        "probability": _safe_float(expectancy.get("probability", context.get("expectancy_probability", 0.50)), 0.50),
        "expected_r": _safe_float(expectancy.get("expected_r", context.get("expectancy_expected_r", 0.0)), 0.0),
        "uncertainty": _safe_float(expectancy.get("uncertainty", context.get("expectancy_uncertainty", 0.50)), 0.50),
        "risk_reward": _safe_float(plan.get("risk_reward_ratio", context.get("planned_risk_reward_ratio", 0.0)), 0.0),
        "market_regime": _norm_token(context.get("market_regime", row.get("market_regime")), "RANGING"),
        "dominant_regime": _norm_token(context.get("dominant_regime", row.get("dominant_regime")), "MIXED"),
        "structure_trend": _norm_token(context.get("structure_trend"), "UNKNOWN"),
        "breakout_state": _norm_token(context.get("orderbook_breakout_state"), "NONE"),
        "level_interaction": _norm_token(context.get("orderbook_interaction"), "BETWEEN_LEVELS"),
        "market_map_bias": _norm_token(context.get("market_map_bias"), "NEUTRAL"),
        "crypto_market_mode": _norm_token(context.get("crypto_market_mode") or context.get("market_mode"), "UNKNOWN"),
        "crypto_directional_bias": _norm_token(context.get("crypto_directional_bias") or context.get("directional_bias"), "NEUTRAL"),
        "crypto_risk_off": bool(context.get("crypto_risk_off") or context.get("risk_off")),
        "event_like": _event_like(context),
    }


def _family_keys(ctx: Mapping[str, Any]) -> list[tuple[str, str]]:
    coin = _safe_str(ctx.get("coin")).upper()
    direction = _safe_str(ctx.get("direction")).upper()
    instrument = _norm_token(ctx.get("instrument_type"), "UNKNOWN")
    theme = _norm_token(ctx.get("portfolio_theme"), "")
    score_bucket = _norm_token(ctx.get("score_bucket"), "BASE")
    quality = _norm_token(ctx.get("thesis_quality"), "UNKNOWN")
    prob = _prob_bucket(_safe_float(ctx.get("probability"), 0.50))
    unc = _unc_bucket(_safe_float(ctx.get("uncertainty"), 0.50))
    rr = _rr_bucket(_safe_float(ctx.get("risk_reward"), 0.0))
    regime = _norm_token(ctx.get("market_regime"), "RANGING")
    dominant = _norm_token(ctx.get("dominant_regime"), "MIXED")
    structure = _norm_token(ctx.get("structure_trend"), "UNKNOWN")
    breakout = _norm_token(ctx.get("breakout_state"), "NONE")
    interaction = _norm_token(ctx.get("level_interaction"), "BETWEEN_LEVELS")
    map_bias = _norm_token(ctx.get("market_map_bias"), "NEUTRAL")
    event = "EVENT" if bool(ctx.get("event_like")) else "NOEVENT"
    keys = [
        ("coin_direction", f"{coin}:{direction}"),
        ("instrument_regime", f"{instrument}:{direction}:{regime}:{dominant}"),
        ("structure_breakout", f"{instrument}:{direction}:{structure}:{breakout}:{interaction}"),
        ("quality_odds", f"{direction}:{quality}:{score_bucket}:{prob}:{unc}:{rr}"),
        ("map_regime", f"{instrument}:{direction}:{map_bias}:{regime}:{event}"),
    ]
    if theme:
        keys.append(("theme_direction", f"{theme}:{direction}:{event}"))
    return keys


def _stats_for_family(
    rows: Iterable[Mapping[str, Any]],
    trading_cfg,
    *,
    family_key: str,
    family_label: str,
) -> dict:
    matched: list[Mapping[str, Any]] = []
    for row in rows:
        ctx = _row_context(row)
        if (family_label, family_key) in _family_keys(ctx):
            matched.append(row)
    if not matched:
        return {"family": family_key, "label": family_label, "samples": 0}
    quality_wins = sum(1 for row in matched if _quality_win(row, trading_cfg))
    raw_wins = sum(1 for row in matched if _safe_float(row.get("pnl_usd"), 0.0) > 0 or _safe_float(row.get("pnl_pct"), 0.0) > 0)
    avg_pnl_pct = sum(_safe_float(row.get("pnl_pct"), 0.0) for row in matched) / max(len(matched), 1)
    avg_pnl_usd = sum(_safe_float(row.get("pnl_usd"), 0.0) for row in matched) / max(len(matched), 1)
    return {
        "family": family_key,
        "label": family_label,
        "samples": len(matched),
        "quality_win_rate": round(quality_wins / max(len(matched), 1), 4),
        "raw_win_rate": round(raw_wins / max(len(matched), 1), 4),
        "avg_pnl_pct": round(avg_pnl_pct, 4),
        "avg_pnl_usd": round(avg_pnl_usd, 4),
    }


def _overall_stats(rows: list[Mapping[str, Any]], trading_cfg) -> dict:
    if not rows:
        return {"samples": 0, "quality_win_rate": 0.0, "raw_win_rate": 0.0}
    quality_wins = sum(1 for row in rows if _quality_win(row, trading_cfg))
    raw_wins = sum(1 for row in rows if _safe_float(row.get("pnl_usd"), 0.0) > 0 or _safe_float(row.get("pnl_pct"), 0.0) > 0)
    pnl_usd = sum(_safe_float(row.get("pnl_usd"), 0.0) for row in rows)
    return {
        "samples": len(rows),
        "quality_win_rate": round(quality_wins / len(rows), 4),
        "raw_win_rate": round(raw_wins / len(rows), 4),
        "pnl_usd": round(pnl_usd, 2),
    }


def assess_entry(
    trading_cfg,
    *,
    coin: str,
    direction: str,
    signal_snapshot: Mapping[str, Any] | None = None,
    signal: Any = None,
    event_starter: bool = False,
    pair_trade: bool = False,
    history_rows: list[dict] | None = None,
    source: str = "entry",
) -> dict:
    if not bool(getattr(trading_cfg, "win_rate_guard_enabled", True)):
        return {"enabled": False, "permitted": True, "active": False, "summary": "win-rate guard disabled"}

    snapshot = dict(signal_snapshot or {})
    ctx = _snapshot_context(coin=coin, direction=direction, signal_snapshot=snapshot, signal=signal)
    if event_starter:
        ctx["event_like"] = True

    rows = list(history_rows) if history_rows is not None else _load_history(trading_cfg)
    history_limit = max(20, _safe_int(getattr(trading_cfg, "win_rate_guard_history_limit", 300), 300))
    rows = rows[-history_limit:]
    overall = _overall_stats(rows, trading_cfg)
    target_wr = _safe_float(getattr(trading_cfg, "win_rate_guard_target_quality_win_rate", 0.70), 0.70)
    min_rows = max(1, _safe_int(getattr(trading_cfg, "win_rate_guard_min_history_trades", 10), 10))
    strict_mode = overall["samples"] >= min_rows and overall["quality_win_rate"] < target_wr

    blockers: list[str] = []
    warnings: list[str] = []
    size_multiplier = 1.0

    family_min_samples = max(1, _safe_int(getattr(trading_cfg, "win_rate_guard_min_family_samples", 3), 3))
    toxic_wr = _safe_float(getattr(trading_cfg, "win_rate_guard_toxic_family_win_rate", 0.40), 0.40)
    trim_wr = _safe_float(getattr(trading_cfg, "win_rate_guard_trim_family_win_rate", 0.55), 0.55)
    families = [
        _stats_for_family(rows, trading_cfg, family_label=label, family_key=key)
        for label, key in _family_keys(ctx)
    ]
    active_families = [item for item in families if int(item.get("samples", 0) or 0) >= family_min_samples]
    toxic_families = [
        item for item in active_families
        if _safe_float(item.get("quality_win_rate"), 1.0) <= toxic_wr
        and (_safe_float(item.get("avg_pnl_pct"), 0.0) <= 0.0 or _safe_float(item.get("avg_pnl_usd"), 0.0) <= 0.0)
    ]
    if toxic_families:
        worst = sorted(toxic_families, key=lambda item: (item.get("quality_win_rate", 1.0), item.get("avg_pnl_pct", 0.0)))[0]
        blockers.append(
            f"historical setup family is toxic: {worst['label']} {worst['quality_win_rate'] * 100:.0f}% quality WR on {worst['samples']} trades"
        )
    else:
        weak_families = [
            item for item in active_families
            if _safe_float(item.get("quality_win_rate"), 1.0) < trim_wr
        ]
        if weak_families:
            weak = sorted(weak_families, key=lambda item: item.get("quality_win_rate", 1.0))[0]
            warnings.append(
                f"weak setup family: {weak['label']} {weak['quality_win_rate'] * 100:.0f}% quality WR; trimming"
            )
            size_multiplier = min(
                size_multiplier,
                _safe_float(getattr(trading_cfg, "win_rate_guard_size_multiplier", 0.45), 0.45),
            )

    same_direction_rows = [
        row for row in rows
        if _safe_str(row.get("coin")).upper() == ctx["coin"]
        and _safe_str(row.get("direction")).upper() == ctx["direction"]
    ]
    recent_window = max(1, _safe_int(getattr(trading_cfg, "win_rate_guard_recent_loss_window", 6), 6))
    recent_loss_limit = max(1, _safe_int(getattr(trading_cfg, "win_rate_guard_recent_loss_limit", 3), 3))
    recent = same_direction_rows[-recent_window:]
    recent_losses = sum(1 for row in recent if not _quality_win(row, trading_cfg))
    if len(recent) >= recent_loss_limit and recent_losses >= recent_loss_limit:
        blockers.append(f"{ctx['coin']} {ctx['direction']} has {recent_losses}/{len(recent)} recent non-quality outcomes")

    event_like = bool(event_starter or ctx.get("event_like"))
    min_prob = _safe_float(getattr(trading_cfg, "win_rate_guard_min_probability", 0.58), 0.58)
    min_expected_r = _safe_float(getattr(trading_cfg, "win_rate_guard_min_expected_r", 0.20), 0.20)
    max_uncertainty = _safe_float(getattr(trading_cfg, "win_rate_guard_max_uncertainty", 0.44), 0.44)
    min_exec_quality = _safe_float(getattr(trading_cfg, "win_rate_guard_min_execution_quality", 68.0), 68.0)
    min_quality_name = _safe_str(getattr(trading_cfg, "win_rate_guard_min_thesis_quality", "MEDIUM"), "MEDIUM")
    event_size_multiplier: float | None = None

    if strict_mode and not pair_trade:
        min_prob = max(min_prob, _safe_float(getattr(trading_cfg, "win_rate_guard_strict_min_probability", 0.62), 0.62))
        min_expected_r = max(min_expected_r, _safe_float(getattr(trading_cfg, "win_rate_guard_strict_min_expected_r", 0.26), 0.26))
        max_uncertainty = min(max_uncertainty, _safe_float(getattr(trading_cfg, "win_rate_guard_strict_max_uncertainty", 0.38), 0.38))
        min_exec_quality = max(min_exec_quality, _safe_float(getattr(trading_cfg, "win_rate_guard_strict_min_execution_quality", 74.0), 74.0))
        min_quality_name = _safe_str(getattr(trading_cfg, "win_rate_guard_strict_min_thesis_quality", "HIGH"), "HIGH")
        warnings.append(f"north-star quality WR is {overall['quality_win_rate'] * 100:.0f}% vs {target_wr * 100:.0f}%; strict selection active")

    if event_like:
        min_prob = min(min_prob, _safe_float(getattr(trading_cfg, "win_rate_guard_event_min_probability", 0.54), 0.54))
        min_expected_r = min(
            min_expected_r,
            _safe_float(getattr(trading_cfg, "win_rate_guard_event_min_expected_r", 0.12), 0.12),
        )
        max_uncertainty = max(
            max_uncertainty,
            _safe_float(getattr(trading_cfg, "win_rate_guard_event_max_uncertainty", 0.58), 0.58),
        )
        min_quality_name = _safe_str(
            getattr(trading_cfg, "win_rate_guard_event_min_thesis_quality", "MEDIUM"),
            "MEDIUM",
        )
        event_size_multiplier = _safe_float(getattr(trading_cfg, "win_rate_guard_event_size_multiplier", 0.60), 0.60)
    elif pair_trade:
        size_multiplier = min(size_multiplier, 0.65)

    if (
        bool(getattr(trading_cfg, "crypto_drawdown_entry_guard_enabled", True))
        and ctx["instrument_type"] == "crypto"
        and ctx["crypto_market_mode"] in {"DRAWDOWN", "RISK_OFF"}
        and ctx["crypto_directional_bias"] == "BEARISH"
    ):
        bullish_reclaim = bool(
            ctx["market_map_bias"] in {"BULLISH", "UPTREND", "RECLAIM"}
            or ctx["breakout_state"] in {
                "PROBING_BULLISH_BREAKOUT",
                "CONFIRMED_BULLISH_BREAKOUT",
                "PERSISTENT_BULLISH_BREAKOUT",
            }
            or ctx["structure_trend"] in {"BULLISH", "UPTREND"}
        )
        if ctx["direction"] == "LONG":
            min_prob = max(
                min_prob,
                _safe_float(getattr(trading_cfg, "crypto_drawdown_long_min_probability", 0.64), 0.64),
            )
            min_expected_r = max(
                min_expected_r,
                _safe_float(getattr(trading_cfg, "crypto_drawdown_long_min_expected_r", 0.30), 0.30),
            )
            size_multiplier = min(
                size_multiplier,
                _safe_float(getattr(trading_cfg, "crypto_risk_off_long_size_multiplier", 0.55), 0.55),
            )
            if (
                ctx["crypto_market_mode"] == "DRAWDOWN"
                and bool(getattr(trading_cfg, "crypto_drawdown_block_longs_without_reclaim", True))
                and not bullish_reclaim
                and not event_like
            ):
                blockers.append("crypto drawdown mode blocks fresh LONG until majors reclaim structure")
            else:
                warnings.append("crypto risk-off mode requires a confirmed reclaim for LONG exposure")
        elif ctx["direction"] == "SHORT":
            warnings.append("crypto majors are risk-off; SHORT thesis has structural tailwind, but odds still must clear")

    if ctx["probability"] < min_prob:
        blockers.append(f"hit-rate governor needs p>={min_prob * 100:.0f}% (has {ctx['probability'] * 100:.0f}%)")
    if ctx["expected_r"] < min_expected_r:
        blockers.append(f"hit-rate governor needs expected R>={min_expected_r:.2f} (has {ctx['expected_r']:.2f})")
    if ctx["uncertainty"] > max_uncertainty:
        blockers.append(f"hit-rate governor blocks uncertainty {ctx['uncertainty'] * 100:.0f}% > {max_uncertainty * 100:.0f}%")
    if _quality_rank(ctx["thesis_quality"]) < _quality_rank(min_quality_name):
        blockers.append(f"hit-rate governor needs thesis quality >= {min_quality_name}")
    if ctx["execution_quality_score"] > 0 and ctx["execution_quality_score"] < min_exec_quality:
        blockers.append(f"execution quality {ctx['execution_quality_score']:.0f} < win-rate floor {min_exec_quality:.0f}")
    if ctx["analog_hard_block"] and ctx["analog_adverse"]:
        blockers.append("historical analogs hard-block this setup")
    elif ctx["analog_sample_size"] >= _safe_int(getattr(trading_cfg, "win_rate_guard_min_analog_samples", 4), 4):
        min_analog_wr = _safe_float(getattr(trading_cfg, "win_rate_guard_min_analog_win_rate", 0.55), 0.55)
        if ctx["analog_win_rate"] > 0 and ctx["analog_win_rate"] < min_analog_wr:
            blockers.append(f"analog win rate {ctx['analog_win_rate'] * 100:.0f}% < {min_analog_wr * 100:.0f}%")

    similar_memory: dict[str, Any] = {
        "enabled": False,
        "active": False,
        "summary": "similar trade memory disabled",
    }
    if bool(getattr(trading_cfg, "similar_trade_memory_enabled", True)):
        min_similarity = _safe_float(getattr(trading_cfg, "similar_trade_min_similarity", 0.62), 0.62)
        similar_memory = performance_intelligence.similar_trade_memory(
            rows,
            coin=ctx["coin"],
            direction=ctx["direction"],
            score=ctx["score"],
            signal_snapshot=snapshot,
            min_similarity=min_similarity,
        )
        worst_match = dict(similar_memory.get("worst_match") or {})
        best_match = dict(similar_memory.get("best_match") or {})
        worst_loss_pct = abs(min(0.0, _safe_float(worst_match.get("pnl_pct"), 0.0)))
        worst_similarity = _safe_float(worst_match.get("similarity"), 0.0)
        hard_loss_pct = _safe_float(getattr(trading_cfg, "similar_trade_hard_block_loss_pct", 0.75), 0.75)
        haircut_loss_pct = _safe_float(getattr(trading_cfg, "similar_trade_haircut_loss_pct", 0.25), 0.25)
        if worst_match and worst_similarity >= min_similarity and worst_loss_pct >= hard_loss_pct and not event_like:
            blockers.append(
                f"closest prior similar trade was a major loser: {worst_match.get('summary', 'worst comparable lost')}"
            )
        elif worst_match and worst_similarity >= min_similarity and worst_loss_pct >= haircut_loss_pct:
            warnings.append(
                f"similar prior loser found: {worst_match.get('summary', 'worst comparable lost')}; trimming"
            )
            size_multiplier = min(
                size_multiplier,
                _safe_float(getattr(trading_cfg, "similar_trade_size_multiplier", 0.50), 0.50),
            )
        best_gain_pct = max(0.0, _safe_float(best_match.get("pnl_pct"), 0.0))
        if best_match and _safe_float(best_match.get("similarity"), 0.0) >= min_similarity and best_gain_pct > 0:
            warnings.append(f"best comparable supports thesis: {best_match.get('summary', 'similar winner')}")

    active = bool(strict_mode or blockers or warnings or active_families)
    if event_like and event_size_multiplier is not None and active:
        size_multiplier = min(size_multiplier, event_size_multiplier)
    permitted = not blockers
    if not permitted:
        size_multiplier = 0.0
    summary = blockers[0] if blockers else (
        warnings[0] if warnings else "setup clears win-rate governor"
    )
    return {
        "enabled": True,
        "active": active,
        "permitted": permitted,
        "strict_mode": bool(strict_mode),
        "summary": summary,
        "blockers": blockers[:5],
        "warnings": warnings[:5],
        "size_multiplier": round(max(0.0, min(1.0, size_multiplier)), 4),
        "overall": overall,
        "families": active_families[:6],
        "similar_trade_memory": similar_memory,
        "event_like": bool(event_like),
        "pair_trade": bool(pair_trade),
        "source": source,
        "requirements": {
            "min_probability": round(min_prob, 4),
            "min_expected_r": round(min_expected_r, 4),
            "max_uncertainty": round(max_uncertainty, 4),
            "min_execution_quality": round(min_exec_quality, 2),
            "min_thesis_quality": min_quality_name,
        },
    }
