"""
playbook_distiller.py — rolling rewrite of what is actually working.

This is the calm memory layer for the agent:
  • summarize which asset / direction / regime families are paying
  • flag which families are bleeding
  • distill them into plain-English playbooks that dossiers and the UI can reuse
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import trade_dataset
from logger import get_logger
from paths import PLAYBOOK_DISTILLER_REPORT_JSON

log = get_logger("playbook_distiller")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _trade_outcome(row: dict) -> str:
    outcome = _safe_str(row.get("outcome")).upper()
    if outcome in {"WIN", "LOSS", "BREAKEVEN"}:
        return outcome
    pnl_usd = _safe_float(row.get("pnl_usd"))
    if pnl_usd > 0:
        return "WIN"
    if pnl_usd < 0:
        return "LOSS"
    return "BREAKEVEN"


def _normalize_row(row: dict) -> dict | None:
    if not isinstance(row, dict):
        return None
    coin = _safe_str(row.get("coin")).upper()
    direction = _safe_str(row.get("direction")).upper()
    if not coin or direction not in {"LONG", "SHORT"}:
        return None

    entry_context = dict(row.get("entry_context") or {})
    exit_context = dict(row.get("exit_context") or {})
    trade_plan = dict(row.get("trade_plan") or {})
    plan_outcome = dict(row.get("plan_outcome") or {})

    closed_at_ts = _safe_float(row.get("closed_at_ts") or row.get("recorded_at_ts"))
    instrument_type = _safe_str(entry_context.get("instrument_type"), "crypto").lower()
    regime = _safe_str(
        entry_context.get("dominant_regime")
        or exit_context.get("dominant_regime")
        or entry_context.get("market_regime")
        or "UNKNOWN",
        "UNKNOWN",
    ).upper()
    execution_style = _safe_str(
        entry_context.get("execution_coach_verdict")
        or (entry_context.get("execution_plan") or {}).get("mode")
        or entry_context.get("entry_type")
        or "UNKNOWN",
        "UNKNOWN",
    ).upper()
    open_logic = _safe_str(
        row.get("open_logic")
        or entry_context.get("reason")
        or entry_context.get("price_action_summary")
        or trade_plan.get("price_action_summary"),
    )

    return {
        "coin": coin,
        "direction": direction,
        "instrument_type": instrument_type,
        "regime": regime,
        "execution_style": execution_style,
        "closed_at_ts": closed_at_ts,
        "hold_minutes": _safe_float(row.get("hold_minutes")),
        "pnl_usd": _safe_float(row.get("pnl_usd")),
        "pnl_pct": _safe_float(row.get("pnl_pct")),
        "captured_r_multiple": _safe_float(plan_outcome.get("captured_r_multiple")),
        "tp_progress_ratio": _safe_float(plan_outcome.get("tp_progress_ratio")),
        "risk_reward_ratio": _safe_float(
            trade_plan.get("risk_reward_ratio") or entry_context.get("planned_risk_reward_ratio")
        ),
        "outcome": _trade_outcome(row),
        "open_logic": open_logic,
    }


def _profit_factor(rows: list[dict]) -> float:
    gross_win = sum(max(0.0, _safe_float(row.get("pnl_usd"))) for row in rows)
    gross_loss = sum(abs(min(0.0, _safe_float(row.get("pnl_usd")))) for row in rows)
    if gross_loss <= 0:
        return gross_win if gross_win > 0 else 0.0
    return gross_win / gross_loss


def _family_stats(rows: list[dict]) -> dict:
    if not rows:
        return {
            "samples": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_pnl_pct": 0.0,
            "avg_pnl_usd": 0.0,
            "profit_factor": 0.0,
            "avg_hold_minutes": 0.0,
            "avg_captured_r": 0.0,
            "avg_tp_progress_ratio": 0.0,
            "execution_styles": {},
            "latest_logic": "",
        }

    wins = sum(1 for row in rows if _trade_outcome(row) == "WIN")
    losses = sum(1 for row in rows if _trade_outcome(row) == "LOSS")
    count = len(rows)
    execution_styles = Counter(_safe_str(row.get("execution_style"), "UNKNOWN").upper() for row in rows)
    latest_row = max(rows, key=lambda row: _safe_float(row.get("closed_at_ts")))
    return {
        "samples": count,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / count, 4) if count else 0.0,
        "avg_pnl_pct": round(sum(_safe_float(row.get("pnl_pct")) for row in rows) / count, 4),
        "avg_pnl_usd": round(sum(_safe_float(row.get("pnl_usd")) for row in rows) / count, 4),
        "profit_factor": round(_profit_factor(rows), 4),
        "avg_hold_minutes": round(sum(_safe_float(row.get("hold_minutes")) for row in rows) / count, 2),
        "avg_captured_r": round(sum(_safe_float(row.get("captured_r_multiple")) for row in rows) / count, 4),
        "avg_tp_progress_ratio": round(sum(_safe_float(row.get("tp_progress_ratio")) for row in rows) / count, 4),
        "execution_styles": dict(execution_styles.most_common(3)),
        "latest_logic": _safe_str(latest_row.get("open_logic")),
    }


def _family_payload(key: tuple[str, str], rows: list[dict]) -> dict:
    direction, regime = key
    stats = _family_stats(rows)
    lead_style = next(iter(stats["execution_styles"].keys()), "UNKNOWN")
    return {
        "direction": direction,
        "regime": regime,
        "execution_style": lead_style,
        **stats,
        "headline": (
            f"{direction} in {regime} is winning {stats['win_rate'] * 100:.0f}% "
            f"across {stats['samples']} trades"
        ),
    }


def _best_family(families: list[dict], *, min_samples: int, min_win_rate: float) -> dict | None:
    eligible = [
        family for family in families
        if int(family.get("samples", 0) or 0) >= min_samples
        and float(family.get("win_rate", 0.0) or 0.0) >= min_win_rate
        and float(family.get("avg_pnl_pct", 0.0) or 0.0) > 0
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda family: (
            float(family.get("win_rate", 0.0) or 0.0),
            float(family.get("avg_pnl_pct", 0.0) or 0.0),
            int(family.get("samples", 0) or 0),
        ),
    )


def _worst_family(families: list[dict], *, min_samples: int, max_losing_win_rate: float) -> dict | None:
    eligible = [
        family for family in families
        if int(family.get("samples", 0) or 0) >= min_samples
        and (
            float(family.get("win_rate", 0.0) or 0.0) <= max_losing_win_rate
            or float(family.get("avg_pnl_pct", 0.0) or 0.0) < 0
        )
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda family: (
            float(family.get("win_rate", 0.0) or 0.0),
            float(family.get("avg_pnl_pct", 0.0) or 0.0),
            -int(family.get("samples", 0) or 0),
        ),
    )


def _distilled_playbook(coin: str, best_family: dict | None, worst_family: dict | None) -> str:
    if best_family and worst_family:
        return (
            f"Lean into {coin} {best_family['direction']} in {best_family['regime']} "
            f"({best_family['win_rate'] * 100:.0f}% wins, {best_family['samples']} samples) "
            f"and stay away from {worst_family['direction']} in {worst_family['regime']} "
            f"({worst_family['win_rate'] * 100:.0f}% wins)."
        )
    if best_family:
        return (
            f"The cleanest recent edge is {coin} {best_family['direction']} in {best_family['regime']} "
            f"with {best_family['execution_style']} execution."
        )
    if worst_family:
        return (
            f"Recent history says avoid {coin} {worst_family['direction']} in {worst_family['regime']} "
            f"until that family earns trust back."
        )
    return f"{coin} does not have enough closed-trade history yet for a distilled playbook."


def build_report(cfg, *, data_dir: Path) -> dict:
    target_dir = trade_dataset.resolve_history_data_dir(data_dir)
    lookback_days = int(getattr(cfg.trading, "playbook_distiller_lookback_days", 28) or 28)
    min_samples = int(getattr(cfg.trading, "playbook_distiller_min_samples", 3) or 3)
    min_win_rate = float(getattr(cfg.trading, "playbook_distiller_min_win_rate", 0.55) or 0.55)
    max_losing_win_rate = float(getattr(cfg.trading, "playbook_distiller_max_losing_win_rate", 0.45) or 0.45)
    cutoff_ts = time.time() - max(1, lookback_days) * 86_400.0

    raw_rows = trade_dataset.load_closed_trades(data_dir=target_dir, backfill_from_csv=True)
    normalized_rows = []
    for raw in raw_rows:
        normalized = _normalize_row(raw)
        if not normalized:
            continue
        closed_at_ts = _safe_float(normalized.get("closed_at_ts"))
        if closed_at_ts > 0 and closed_at_ts < cutoff_ts:
            continue
        normalized_rows.append(normalized)

    by_coin: dict[str, list[dict]] = defaultdict(list)
    for row in normalized_rows:
        by_coin[str(row.get("coin"))].append(row)

    assets: dict[str, dict] = {}
    working_families: list[dict] = []
    failing_families: list[dict] = []

    for coin in sorted(by_coin):
        rows = by_coin[coin]
        by_family: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            family_key = (_safe_str(row.get("direction")).upper(), _safe_str(row.get("regime"), "UNKNOWN").upper())
            by_family[family_key].append(row)

        family_payloads = [
            _family_payload(key, family_rows)
            for key, family_rows in sorted(by_family.items(), key=lambda item: (item[0][0], item[0][1]))
        ]
        best_family = _best_family(family_payloads, min_samples=min_samples, min_win_rate=min_win_rate)
        worst_family = _worst_family(family_payloads, min_samples=min_samples, max_losing_win_rate=max_losing_win_rate)
        asset_stats = _family_stats(rows)
        lead_style = next(iter(asset_stats["execution_styles"].keys()), "UNKNOWN")

        assets[coin] = {
            "coin": coin,
            "instrument_type": _safe_str(rows[-1].get("instrument_type"), "crypto"),
            "samples": asset_stats["samples"],
            "win_rate": asset_stats["win_rate"],
            "avg_pnl_pct": asset_stats["avg_pnl_pct"],
            "avg_pnl_usd": asset_stats["avg_pnl_usd"],
            "profit_factor": asset_stats["profit_factor"],
            "lead_execution_style": lead_style,
            "playbook": _distilled_playbook(coin, best_family, worst_family),
            "best_family": best_family or {},
            "avoid_family": worst_family or {},
            "families": sorted(
                family_payloads,
                key=lambda family: (
                    float(family.get("win_rate", 0.0) or 0.0),
                    float(family.get("avg_pnl_pct", 0.0) or 0.0),
                    int(family.get("samples", 0) or 0),
                ),
                reverse=True,
            )[:6],
            "latest_logic": asset_stats["latest_logic"],
        }

        if best_family:
            working_families.append({"coin": coin, **best_family})
        if worst_family:
            failing_families.append({"coin": coin, **worst_family})

    working_families.sort(
        key=lambda family: (
            float(family.get("win_rate", 0.0) or 0.0),
            float(family.get("avg_pnl_pct", 0.0) or 0.0),
            int(family.get("samples", 0) or 0),
        ),
        reverse=True,
    )
    failing_families.sort(
        key=lambda family: (
            float(family.get("win_rate", 0.0) or 0.0),
            float(family.get("avg_pnl_pct", 0.0) or 0.0),
            -int(family.get("samples", 0) or 0),
        ),
    )

    summary = {
        "asset_count": len(assets),
        "closed_trade_count": len(normalized_rows),
        "working_family_count": len(working_families),
        "failing_family_count": len(failing_families),
        "top_working_families": working_families[:6],
        "top_failing_families": failing_families[:6],
        "weekly_rewrite": (
            "The agent should keep leaning into the strongest asset/regime families, "
            "stay patient on everything else, and stop recycling the families that are still losing."
        ),
    }

    return {
        "generated_at": int(time.time()),
        "data_dir": str(target_dir),
        "lookback_days": lookback_days,
        "min_samples": min_samples,
        "closed_trade_count": len(normalized_rows),
        "summary": summary,
        "assets": assets,
    }


def build_and_save_report(cfg, *, data_dir: Path) -> dict:
    report = build_report(cfg, data_dir=data_dir)
    report_path = Path(data_dir).expanduser() / PLAYBOOK_DISTILLER_REPORT_JSON.name
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
