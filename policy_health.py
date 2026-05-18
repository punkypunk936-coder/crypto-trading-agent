"""
policy_health.py - compact architecture and policy health report.

The agent already records a large amount of trade and decision data. This
module turns the highest-signal evidence into a small report that answers:
where is win rate leaking, and what should be fixed next?
"""

from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from paths import (
    DECISION_DATASET_JSONL,
    POLICY_HEALTH_REPORT_JSON,
    TRADES_CSV,
)


INVALIDATION_EXITS = {
    "conviction_lost",
    "htf_invalidation",
    "micro_invalidation",
    "stale_adverse",
    "stop_loss",
    "structure_invalidation",
    "time_stop",
}


def _cfg_value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value not in (None, "") else default)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value if value not in (None, "") else default))
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _data_path(data_dir: Path | None, path: Path) -> Path:
    if data_dir is None:
        return path
    return Path(data_dir) / path.name


def _tail_lines(path: Path, max_lines: int, *, block_size: int = 64 * 1024) -> list[bytes]:
    if max_lines <= 0 or not path.exists():
        return []
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            position = handle.tell()
            chunks: list[bytes] = []
            line_count = 0
            while position > 0 and line_count <= max_lines:
                read_size = min(block_size, position)
                position -= read_size
                handle.seek(position)
                chunk = handle.read(read_size)
                chunks.append(chunk)
                line_count += chunk.count(b"\n")
            payload = b"".join(reversed(chunks))
            return payload.splitlines()[-max_lines:]
    except Exception:
        return []


def _load_closed_trades(data_dir: Path | None = None) -> list[dict]:
    path = _data_path(data_dir, TRADES_CSV)
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if _safe_str(row.get("closed_at") or row.get("exit_time")):
                    rows.append(dict(row))
    except Exception:
        return []
    return rows


def _load_recent_decisions(data_dir: Path | None = None, *, sample_lines: int = 5000) -> list[dict]:
    path = _data_path(data_dir, DECISION_DATASET_JSONL)
    rows: list[dict] = []
    for raw in _tail_lines(path, sample_lines):
        try:
            rows.append(json.loads(raw.decode("utf-8", errors="ignore")))
        except Exception:
            continue
    return rows


def _pnl(row: dict) -> float:
    return _safe_float(row.get("pnl_usd"))


def _duration(row: dict) -> float:
    return _safe_float(row.get("duration_mins") or row.get("duration_minutes") or row.get("hold_minutes"))


def _family_metrics(rows: Iterable[dict], *, min_samples: int = 3, limit: int = 8) -> tuple[list[dict], list[dict]]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        coin = _safe_str(row.get("coin")).upper()
        direction = _safe_str(row.get("direction") or row.get("side")).upper()
        if coin and direction:
            buckets[(coin, direction)].append(row)

    metrics: list[dict] = []
    for (coin, direction), items in buckets.items():
        if len(items) < min_samples:
            continue
        wins = sum(1 for item in items if _pnl(item) > 0)
        pnl = sum(_pnl(item) for item in items)
        metrics.append({
            "family": f"{coin}:{direction}",
            "coin": coin,
            "direction": direction,
            "samples": len(items),
            "win_rate": _pct(wins, len(items)),
            "pnl_usd": round(pnl, 2),
            "avg_pnl_usd": round(pnl / len(items), 2) if items else 0.0,
        })

    worst = sorted(metrics, key=lambda item: (item["pnl_usd"], -item["samples"], item["family"]))[:limit]
    best = sorted(metrics, key=lambda item: (-item["pnl_usd"], -item["samples"], item["family"]))[:limit]
    return worst, best


def _trade_summary(rows: list[dict], *, short_hold_minutes: float) -> dict:
    total = len(rows)
    wins = sum(1 for row in rows if _pnl(row) > 0)
    losses = sum(1 for row in rows if _pnl(row) < 0)
    pnl_values = [_pnl(row) for row in rows]
    win_pnls = [value for value in pnl_values if value > 0]
    loss_pnls = [value for value in pnl_values if value < 0]
    durations = [_duration(row) for row in rows if _duration(row) >= 0]
    exit_counts = Counter(_safe_str(row.get("exit_reason"), "unknown") for row in rows)
    invalidation_count = sum(count for reason, count in exit_counts.items() if reason in INVALIDATION_EXITS)
    short_hold_count = sum(1 for value in durations if value < short_hold_minutes)
    very_short_count = sum(1 for value in durations if value < 15.0)
    worst, best = _family_metrics(rows)

    gross_win = sum(win_pnls)
    gross_loss = abs(sum(loss_pnls))
    return {
        "closed_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": _pct(wins, total),
        "pnl_usd": round(sum(pnl_values), 2),
        "avg_pnl_usd": round(sum(pnl_values) / total, 2) if total else 0.0,
        "avg_win_usd": round(sum(win_pnls) / len(win_pnls), 2) if win_pnls else 0.0,
        "avg_loss_usd": round(sum(loss_pnls) / len(loss_pnls), 2) if loss_pnls else 0.0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else 0.0,
        "median_hold_minutes": round(statistics.median(durations), 2) if durations else 0.0,
        "avg_hold_minutes": round(sum(durations) / len(durations), 2) if durations else 0.0,
        "short_hold_rate": _pct(short_hold_count, len(durations)),
        "very_short_hold_rate": _pct(very_short_count, len(durations)),
        "take_profit_rate": _pct(exit_counts.get("take_profit", 0), total),
        "invalidation_exit_rate": _pct(invalidation_count, total),
        "exit_reasons": [
            {"reason": reason, "count": count, "rate": _pct(count, total)}
            for reason, count in exit_counts.most_common(12)
        ],
        "worst_families": worst,
        "best_families": best,
    }


def _decision_direction(record: dict, snap: dict) -> str:
    action = _safe_str(record.get("candidate_action") or snap.get("thesis_candidate_action") or snap.get("action")).upper()
    return action if action in {"LONG", "SHORT"} else "FLAT"


def _first_principles_scores(snap: dict) -> dict:
    view = dict(snap.get("first_principles") or {})
    return {
        "fundamental": _safe_float(view.get("fundamental_score") or snap.get("first_principles_fundamental_score")),
        "attention": _safe_float(view.get("attention_score") or snap.get("first_principles_attention_score")),
        "flow": _safe_float(view.get("flow_score") or snap.get("first_principles_flow_score")),
        "price": _safe_float(view.get("price_score") or snap.get("first_principles_price_score")),
        "sequence": _safe_float(view.get("sequence_score") or snap.get("first_principles_sequence_score")),
    }


def _decision_summary(records: list[dict], *, decision_file_size_bytes: int = 0) -> dict:
    total = len(records)
    stage_counts = Counter(_safe_str(record.get("stage"), "unknown") for record in records)
    final_counts = Counter(_safe_str(record.get("final_action"), "FLAT").upper() for record in records)
    candidate_counts = Counter(_safe_str(record.get("candidate_action"), "FLAT").upper() for record in records)

    directional: list[dict] = []
    for record in records:
        snap = dict(record.get("signal_snapshot") or {})
        direction = _decision_direction(record, snap)
        if direction in {"LONG", "SHORT"}:
            directional.append(record)

    blocked = sum(1 for record in directional if bool(record.get("blocked", False)))
    executed = sum(1 for record in directional if bool(record.get("executed", False)))
    low_context = 0
    no_news_context = 0
    no_social_coverage = 0
    price_only = 0
    directional_stage_counts = Counter()
    for record in directional:
        snap = dict(record.get("signal_snapshot") or {})
        directional_stage_counts[_safe_str(record.get("stage"), "unknown")] += 1
        event_score = max(
            _safe_float(snap.get("news_event_score")),
            _safe_float(snap.get("official_event_score")),
            _safe_float(snap.get("sec_event_score")),
        )
        catalyst_score = max(
            _safe_float(snap.get("news_catalyst_score")),
            max(0.0, _safe_float(snap.get("analyst_revision_score"))),
        )
        social_score = _safe_float(snap.get("social_attention_score"), 50.0)
        social_mentions = _safe_int(snap.get("social_attention_mentions"))
        social_sources = _safe_int(snap.get("social_attention_sources_checked"))
        no_news = event_score <= 0.0 and catalyst_score <= 0.0
        no_social = social_sources <= 0 or (social_mentions <= 0 and 45.0 <= social_score <= 55.0)
        no_news_context += 1 if no_news else 0
        no_social_coverage += 1 if no_social else 0
        low_context += 1 if no_news and no_social else 0
        fp = _first_principles_scores(snap)
        if fp["price"] >= 70.0 and fp["fundamental"] < 62.0 and fp["attention"] < 62.0:
            price_only += 1

    directional_total = len(directional)
    return {
        "sampled_decisions": total,
        "decision_file_size_mb": round(decision_file_size_bytes / (1024 * 1024), 2) if decision_file_size_bytes else 0.0,
        "directional_decisions": directional_total,
        "blocked_directional": blocked,
        "executed_directional": executed,
        "blocked_directional_rate": _pct(blocked, directional_total),
        "executed_directional_rate": _pct(executed, directional_total),
        "low_context_directional_rate": _pct(low_context, directional_total),
        "missing_news_context_rate": _pct(no_news_context, directional_total),
        "missing_social_coverage_rate": _pct(no_social_coverage, directional_total),
        "price_only_directional_rate": _pct(price_only, directional_total),
        "stage_counts": [
            {"stage": stage, "count": count, "rate": _pct(count, total)}
            for stage, count in stage_counts.most_common(14)
        ],
        "directional_stage_counts": [
            {"stage": stage, "count": count, "rate": _pct(count, directional_total)}
            for stage, count in directional_stage_counts.most_common(14)
        ],
        "final_actions": dict(final_counts),
        "candidate_actions": dict(candidate_counts),
    }


def _finding(priority: str, area: str, title: str, evidence: str, fix: str) -> dict:
    return {
        "priority": priority,
        "area": area,
        "title": title,
        "evidence": evidence,
        "fix": fix,
    }


def _build_findings(trades: dict, decisions: dict, *, config: Any = None) -> list[dict]:
    findings: list[dict] = []
    target_wr = float(_cfg_value(config, "north_star_target_quality_win_rate", 0.70) or 0.70)
    win_rate = _safe_float(trades.get("win_rate"))
    closed = _safe_int(trades.get("closed_trades"))
    if closed >= 20 and win_rate < target_wr:
        findings.append(_finding(
            "P0",
            "Measurement",
            "Win-rate target is not being controlled by a promotion gate",
            f"{closed} closed trades show {win_rate * 100:.1f}% win rate vs {target_wr * 100:.0f}% target.",
            "Require every new entry/exit policy to show replay or live cohort improvement before increasing size.",
        ))

    short_hold_rate = _safe_float(trades.get("short_hold_rate"))
    median_hold = _safe_float(trades.get("median_hold_minutes"))
    if closed >= 20 and (short_hold_rate >= 0.40 or median_hold < 60.0):
        findings.append(_finding(
            "P0",
            "Exit Policy",
            "The system is churning positions before theses can work",
            f"Median hold is {median_hold:.0f}m and {short_hold_rate * 100:.0f}% of closed trades lasted under 60m.",
            "Build an exit-policy replay by thesis class; require hard invalidation for thesis runners and delay soft conviction exits.",
        ))

    invalidation_rate = _safe_float(trades.get("invalidation_exit_rate"))
    take_profit_rate = _safe_float(trades.get("take_profit_rate"))
    if closed >= 20 and invalidation_rate >= 0.60 and take_profit_rate <= 0.10:
        findings.append(_finding(
            "P1",
            "Trade Management",
            "Most exits are invalidation exits, not planned winners",
            f"Invalidation-style exits are {invalidation_rate * 100:.0f}% of closes; take-profit exits are {take_profit_rate * 100:.0f}%.",
            "Separate scalp exits from high-timeframe exits and learn per-family stop width, time stop, and runner rules.",
        ))

    directional = _safe_int(decisions.get("directional_decisions"))
    blocked_rate = _safe_float(decisions.get("blocked_directional_rate"))
    executed_rate = _safe_float(decisions.get("executed_directional_rate"))
    if directional >= 100 and blocked_rate >= 0.75:
        findings.append(_finding(
            "P1",
            "Guard Stack",
            "The guard stack blocks most directional ideas without ranking missed winners",
            f"Recent sample has {directional} directional ideas; {blocked_rate * 100:.0f}% blocked and {executed_rate * 100:.1f}% executed.",
            "Add per-guard counterfactual attribution: which blocks saved losses, which blocks missed winners, and which should be relaxed.",
        ))

    missing_social_rate = _safe_float(decisions.get("missing_social_coverage_rate"))
    low_context_rate = _safe_float(decisions.get("low_context_directional_rate"))
    if directional >= 50 and missing_social_rate >= 0.50:
        findings.append(_finding(
            "P1",
            "Attention Data",
            "The meme/attention layer is effectively missing on recent decisions",
            f"{missing_social_rate * 100:.0f}% of directional decisions had no usable social/trader-feed coverage.",
            "Wire reliable public trader feeds or paid social/attention feeds, then treat missing attention as unknown rather than neutral.",
        ))
    if directional >= 50 and low_context_rate >= 0.30:
        findings.append(_finding(
            "P1",
            "Fundamental Context",
            "Too many directional ideas have neither catalyst nor attention context",
            f"{low_context_rate * 100:.0f}% of recent directional ideas had no catalyst/news context and no social coverage.",
            "Make the morning scout book the source of truth for what deserves risk that day; demote out-of-radar names to observation.",
        ))

    price_only_rate = _safe_float(decisions.get("price_only_directional_rate"))
    if directional >= 50 and price_only_rate >= 0.20:
        findings.append(_finding(
            "P2",
            "Signal Quality",
            "Price-action-only ideas still appear in the directional stream",
            f"{price_only_rate * 100:.0f}% of sampled directional ideas were price-heavy without enough fundamental/attention score.",
            "Keep the first-principles guard on, but measure the blocked cohort so we know if it improves precision.",
        ))

    decision_file_mb = _safe_float(decisions.get("decision_file_size_mb"))
    large_mb = float(_cfg_value(config, "policy_health_large_decision_log_mb", 500.0) or 500.0)
    if decision_file_mb >= large_mb:
        findings.append(_finding(
            "P2",
            "Data Ops",
            "Decision history is too large for fast analysis",
            f"Decision log is {decision_file_mb:.0f}MB; full reads can stall the agent/dashboard.",
            "Add daily rollups and archive old JSONL shards; dashboards should read compact reports, not the raw firehose.",
        ))

    worst = list(trades.get("worst_families") or [])
    if worst:
        item = worst[0]
        findings.append(_finding(
            "P2",
            "Allocation",
            "Capital is not yet aggressively avoiding the worst families",
            f"Worst family: {item.get('family')} with {item.get('samples')} samples and ${item.get('pnl_usd'):+.2f} PnL.",
            "Tie family-level edge to max size, allowed direction, and cooldown before fresh exposure is admitted.",
        ))

    return findings


def _roadmap() -> list[dict]:
    return [
        {
            "name": "Exit policy replay",
            "why": "Win rate is being hurt by churn and invalidation exits.",
            "build": "Replay current exits against alternatives by thesis class: scalp, event starter, high-timeframe runner.",
        },
        {
            "name": "Guard counterfactual attribution",
            "why": "Blocked ideas need labels so the agent knows which guards help vs miss winners.",
            "build": "Track per-guard saved-loss and missed-win rates, then auto-relax or tighten thresholds by evidence.",
        },
        {
            "name": "Attention feed coverage",
            "why": "Fundamentals plus meme/attention flow is a core edge, but missing feeds currently look neutral.",
            "build": "Add approved trader/social/volume feeds and an attention-unknown state that reduces confidence.",
        },
        {
            "name": "Family allocator",
            "why": "Some ticker-direction families clearly pay while others bleed.",
            "build": "Set per-family max size, leverage, cooldown, and starter eligibility from realized edge.",
        },
        {
            "name": "Daily policy rollup",
            "why": "The raw decision log is too large to reason about directly.",
            "build": "Persist compact daily summaries and archive old JSONL shards.",
        },
    ]


def build_report(
    *,
    data_dir: Path | None = None,
    config: Any = None,
    decision_sample_lines: int | None = None,
) -> dict:
    data_dir = Path(data_dir) if data_dir is not None else None
    sample_lines = int(
        decision_sample_lines
        if decision_sample_lines is not None
        else _cfg_value(config, "policy_health_decision_sample_lines", 5000)
        or 5000
    )
    short_hold_minutes = float(_cfg_value(config, "policy_health_short_hold_minutes", 60.0) or 60.0)
    trade_rows = _load_closed_trades(data_dir)
    decision_path = _data_path(data_dir, DECISION_DATASET_JSONL)
    decision_rows = _load_recent_decisions(data_dir, sample_lines=sample_lines)
    trade_summary = _trade_summary(trade_rows, short_hold_minutes=short_hold_minutes)
    decision_summary = _decision_summary(
        decision_rows,
        decision_file_size_bytes=decision_path.stat().st_size if decision_path.exists() else 0,
    )
    findings = _build_findings(trade_summary, decision_summary, config=config)
    return {
        "generated_at": int(time.time()),
        "data_dir": str(data_dir or DECISION_DATASET_JSONL.parent),
        "status": "ATTENTION_REQUIRED" if findings else "OK",
        "summary": {
            "headline": findings[0]["title"] if findings else "No major policy-health issues detected.",
            "finding_count": len(findings),
            "closed_trades": trade_summary.get("closed_trades", 0),
            "win_rate": trade_summary.get("win_rate", 0.0),
            "pnl_usd": trade_summary.get("pnl_usd", 0.0),
            "median_hold_minutes": trade_summary.get("median_hold_minutes", 0.0),
            "directional_sample": decision_summary.get("directional_decisions", 0),
            "blocked_directional_rate": decision_summary.get("blocked_directional_rate", 0.0),
            "low_context_directional_rate": decision_summary.get("low_context_directional_rate", 0.0),
        },
        "trades": trade_summary,
        "recent_decisions": decision_summary,
        "findings": findings,
        "roadmap": _roadmap(),
    }


def build_and_save_report(
    *,
    data_dir: Path | None = None,
    config: Any = None,
    decision_sample_lines: int | None = None,
) -> dict:
    report = build_report(
        data_dir=data_dir,
        config=config,
        decision_sample_lines=decision_sample_lines,
    )
    path = _data_path(data_dir, POLICY_HEALTH_REPORT_JSON)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
