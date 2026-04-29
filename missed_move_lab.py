"""
missed_move_lab.py — richer review of obvious winners the agent skipped.

This sits on top of the decision dataset and turns "missed winners" into:
  • recent examples the operator can inspect
  • top blocker families the bot should stop repeating
  • coin-level pressure that can feed asset dossiers and LLM review context
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import precision_lab
from logger import get_logger
from paths import DECISION_DATASET_JSONL, MISSED_MOVE_REPORT_JSON

log = get_logger("missed_move_lab")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _ts_iso(value: Any) -> str:
    ts = _safe_float(value)
    if ts <= 0:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _primary_reason(*values: Any) -> str:
    for raw in values:
        text = str(raw or "").replace("|", "·")
        for part in [piece.strip() for piece in text.split("·")]:
            lowered = part.lower()
            if not part:
                continue
            if lowered.startswith("score ") or lowered.startswith("map:") or lowered.startswith("breakout state:"):
                continue
            return part
    return ""


def _load_review_candidates(data_dir: Path) -> list[dict]:
    dataset_path = data_dir / DECISION_DATASET_JSONL.name
    if not dataset_path.exists():
        raise FileNotFoundError(f"decision dataset not found at {dataset_path}")

    rows: list[dict] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue

            snap = dict(record.get("signal_snapshot") or {})
            candidate = _safe_str(
                record.get("candidate_action")
                or snap.get("thesis_candidate_action")
                or snap.get("action")
                or "FLAT"
            ).upper()
            if candidate not in {"LONG", "SHORT"}:
                continue

            risk_pct = _safe_float(snap.get("planned_risk_pct")) / 100.0
            if risk_pct <= 0:
                continue

            stage = _safe_str(record.get("stage")).lower()
            executed = bool(record.get("executed", False)) or stage in {"market_entry_opened", "limit_entry_opened"}
            pending_only = bool(record.get("pending_limit", False)) and not executed
            if pending_only:
                continue

            decision_reason = _safe_str(record.get("decision_reason") or snap.get("decision_reason"))
            flat_reason = _safe_str(snap.get("flat_reason"))
            next_unblock = _safe_str(record.get("next_unblock_reason") or snap.get("next_unblock_reason"))
            rows.append({
                "decision_id": record.get("decision_id"),
                "coin": _safe_str(record.get("coin")).upper(),
                "action": candidate,
                "stage": stage,
                "blocked": bool(record.get("blocked", False)),
                "executed": executed,
                "ts": _safe_float(record.get("recorded_at_ts")),
                "risk_pct": risk_pct,
                "reward_pct": _safe_float(snap.get("planned_reward_pct")) / 100.0,
                "rr": _safe_float(snap.get("planned_risk_reward_ratio")),
                "prob": _safe_float(snap.get("expectancy_probability"), 0.50),
                "unc": _safe_float(snap.get("expectancy_uncertainty"), 0.50),
                "score": _safe_float(snap.get("expectancy_score"), 50.0),
                "price": _safe_float(snap.get("live_price") or snap.get("analysis_price") or snap.get("price")),
                "live_price": _safe_float(snap.get("live_price")),
                "analysis_price": _safe_float(snap.get("analysis_price")),
                "confidence": _safe_str(snap.get("confidence"), "LOW").upper(),
                "thesis_quality": _safe_str(snap.get("thesis_quality"), "LOW").upper(),
                "news_catalyst_score": _safe_float(snap.get("news_catalyst_score")),
                "news_event_score": _safe_float(snap.get("news_event_score")),
                "conviction_entry_event": bool(snap.get("conviction_entry_event")),
                "event_budget_summary": _safe_str(snap.get("event_budget_summary")),
                "portfolio_guard_summary": _safe_str(snap.get("portfolio_guard_summary")),
                "breakout": _safe_str(snap.get("orderbook_breakout_state"), "NONE").lower(),
                "interaction": _safe_str(snap.get("orderbook_interaction"), "between_levels").lower(),
                "regime": _safe_str(snap.get("dominant_regime"), "mixed").lower(),
                "instrument_type": _safe_str(snap.get("instrument_type"), "crypto").lower(),
                "decision_reason": decision_reason,
                "flat_reason": flat_reason,
                "next_unblock_reason": next_unblock,
                "thesis_summary": _safe_str(snap.get("thesis_summary")),
                "market_map_summary": _safe_str(snap.get("market_map_summary")),
                "analog_summary": _safe_str(snap.get("analog_summary")),
            })

    rows.sort(key=lambda item: item["ts"])
    return rows


def _collapse(rows: list[dict], *, dedupe_minutes: int) -> list[dict]:
    episodes: list[dict] = []
    max_gap = dedupe_minutes * 60
    for row in rows:
        if not episodes:
            episodes.append(dict(row))
            continue
        last = episodes[-1]
        same_family = (
            row["coin"] == last["coin"]
            and row["action"] == last["action"]
            and row["stage"] == last["stage"]
            and row["breakout"] == last["breakout"]
            and row["interaction"] == last["interaction"]
            and row["regime"] == last["regime"]
        )
        if same_family and (row["ts"] - last["ts"]) <= max_gap:
            continue
        episodes.append(dict(row))
    return episodes


def _classify(row: dict) -> str:
    if bool(row.get("executed", False)):
        return "GOOD_TRADE" if int(row.get("outcome", 0)) else "BAD_TRADE"
    return "MISSED_WIN" if int(row.get("outcome", 0)) else "CORRECT_PASS"


def _miss_summary(row: dict) -> str:
    blocker = _primary_reason(
        row.get("decision_reason"),
        row.get("flat_reason"),
        row.get("next_unblock_reason"),
        row.get("thesis_summary"),
        row.get("market_map_summary"),
        row.get("analog_summary"),
    )
    if blocker:
        return blocker
    return f"{row.get('coin', '')} {row.get('action', '')} was skipped even though it later reached the review target."


def _build_daily_top_mover_replay(
    rows: list[dict],
    *,
    lookback_hours: int = 24,
    top_n: int = 8,
    now_ts: float | None = None,
) -> dict:
    if not rows:
        return {
            "lookback_hours": lookback_hours,
            "top_movers": [],
            "missed_top_movers": [],
            "lessons": [],
        }
    now_value = _safe_float(now_ts) or max(_safe_float(row.get("ts")) for row in rows)
    cutoff = now_value - max(1, int(lookback_hours or 24)) * 3600
    recent = [row for row in rows if _safe_float(row.get("ts")) >= cutoff]
    by_coin: dict[str, list[dict]] = defaultdict(list)
    for row in recent:
        if _safe_float(row.get("price")) > 0:
            by_coin[str(row.get("coin") or "").upper()].append(row)

    movers: list[dict] = []
    for coin, coin_rows in by_coin.items():
        ordered = sorted(coin_rows, key=lambda item: _safe_float(item.get("ts")))
        if len(ordered) < 2:
            continue
        first = ordered[0]
        last = ordered[-1]
        start_price = _safe_float(first.get("price"))
        end_price = _safe_float(last.get("price"))
        if start_price <= 0 or end_price <= 0:
            continue
        move_pct = (end_price - start_price) / start_price * 100.0
        if abs(move_pct) < 0.05:
            continue
        direction = "LONG" if move_pct > 0 else "SHORT"
        movers.append({
            "coin": coin,
            "direction": direction,
            "move_pct": round(move_pct, 3),
            "start_price": round(start_price, 6),
            "end_price": round(end_price, 6),
            "start_ts": first.get("ts"),
            "end_ts": last.get("ts"),
            "start_at": _ts_iso(first.get("ts")),
            "end_at": _ts_iso(last.get("ts")),
        })

    movers.sort(key=lambda item: abs(_safe_float(item.get("move_pct"))), reverse=True)
    movers = movers[:top_n]
    missed: list[dict] = []
    for mover in movers:
        coin = mover["coin"]
        direction = mover["direction"]
        candidates = [
            row for row in recent
            if row.get("coin") == coin
            and row.get("action") == direction
            and not bool(row.get("executed", False))
            and (bool(row.get("blocked", False)) or str(row.get("stage") or "").endswith("_block"))
            and _safe_float(row.get("ts")) <= _safe_float(mover.get("end_ts"))
        ]
        if not candidates:
            continue
        candidate = sorted(candidates, key=lambda item: _safe_float(item.get("ts")), reverse=True)[0]
        missed.append({
            "coin": coin,
            "direction": direction,
            "move_pct": mover["move_pct"],
            "blocked_stage": candidate.get("stage"),
            "blocked_at": _ts_iso(candidate.get("ts")),
            "summary": _miss_summary(candidate),
            "score": round(_safe_float(candidate.get("score"), 50.0), 2),
            "probability": round(_safe_float(candidate.get("prob"), 0.50), 4),
            "news_catalyst_score": round(_safe_float(candidate.get("news_catalyst_score")), 2),
            "news_event_score": round(_safe_float(candidate.get("news_event_score")), 2),
            "conviction_entry_event": bool(candidate.get("conviction_entry_event")),
            "event_budget_summary": candidate.get("event_budget_summary", ""),
        })

    lessons = []
    if missed:
        top = missed[0]
        starter_note = (
            "event starter was active, so replay should inspect sizing/execution"
            if top.get("conviction_entry_event")
            else "consider whether an event starter should have been opened before confirmation"
        )
        lessons.append(
            f"{top['coin']} moved {top['move_pct']:+.2f}% after a blocked {top['direction']} candidate; {starter_note}."
        )
    return {
        "lookback_hours": lookback_hours,
        "top_movers": movers,
        "missed_top_movers": missed,
        "lessons": lessons,
    }


def build_report(
    *,
    data_dir: Path,
    target_r: float,
    horizon_minutes: int,
    interval: str,
    dedupe_minutes: int,
) -> dict:
    rows = _load_review_candidates(data_dir)
    episodes = _collapse(rows, dedupe_minutes=dedupe_minutes)
    labeled = [
        labeled_row
        for labeled_row in (
            precision_lab._label_episode(
                row,
                interval=interval,
                horizon_minutes=horizon_minutes,
                target_r=target_r,
            )
            for row in episodes
        )
        if labeled_row is not None
    ]
    for row in labeled:
        row["classification"] = _classify(row)

    counts = Counter(row["classification"] for row in labeled)
    missed = [row for row in labeled if row["classification"] == "MISSED_WIN"]
    missed_by_coin: dict[str, int] = Counter(row["coin"] for row in missed)
    missed_by_family: dict[str, int] = Counter(f"{row['coin']}:{row['action']}" for row in missed)
    missed_by_stage: dict[str, int] = Counter(str(row.get("stage") or "unknown") for row in missed)
    blocker_counts: dict[str, int] = Counter(_miss_summary(row) for row in missed if _miss_summary(row))
    daily_top_mover_replay = _build_daily_top_mover_replay(rows)

    recent_missed = []
    for row in sorted(missed, key=lambda item: float(item.get("ts", 0.0) or 0.0), reverse=True)[:10]:
        recent_missed.append({
            "coin": row.get("coin"),
            "action": row.get("action"),
            "stage": row.get("stage"),
            "ts": row.get("ts"),
            "recorded_at": _ts_iso(row.get("ts")),
            "score": round(_safe_float(row.get("score"), 50.0), 2),
            "probability": round(_safe_float(row.get("prob"), 0.50), 4),
            "risk_reward_ratio": round(_safe_float(row.get("rr"), 0.0), 2),
            "summary": _miss_summary(row),
            "interaction": row.get("interaction"),
            "breakout": row.get("breakout"),
            "regime": row.get("regime"),
        })

    report = {
        "generated_at": int(time.time()),
        "data_dir": str(data_dir),
        "target_r": target_r,
        "horizon_minutes": horizon_minutes,
        "interval": interval,
        "decision_rows": len(rows),
        "episodes": len(episodes),
        "labeled_episodes": len(labeled),
        "classifications": dict(counts),
        "summary": {
            "missed_win_count": int(counts.get("MISSED_WIN", 0)),
            "correct_pass_count": int(counts.get("CORRECT_PASS", 0)),
            "good_trade_count": int(counts.get("GOOD_TRADE", 0)),
            "bad_trade_count": int(counts.get("BAD_TRADE", 0)),
        },
        "top_missed_assets": [
            {"coin": coin, "misses": misses}
            for coin, misses in missed_by_coin.most_common(8)
        ],
        "top_missed_families": [
            {"family": family, "misses": misses}
            for family, misses in missed_by_family.most_common(8)
        ],
        "top_missed_stages": [
            {"stage": stage, "misses": misses}
            for stage, misses in missed_by_stage.most_common(8)
        ],
        "top_blockers": [
            {"blocker": blocker, "misses": misses}
            for blocker, misses in blocker_counts.most_common(8)
        ],
        "recent_missed_moves": recent_missed,
        "daily_top_mover_replay": daily_top_mover_replay,
    }
    return report


def build_and_save_report(
    *,
    data_dir: Path,
    target_r: float,
    horizon_minutes: int,
    interval: str,
    dedupe_minutes: int,
) -> dict:
    report = build_report(
        data_dir=data_dir,
        target_r=target_r,
        horizon_minutes=horizon_minutes,
        interval=interval,
        dedupe_minutes=dedupe_minutes,
    )
    try:
        MISSED_MOVE_REPORT_JSON.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        log.debug("missed_move_report.json write failed: %s", exc)
    return report
