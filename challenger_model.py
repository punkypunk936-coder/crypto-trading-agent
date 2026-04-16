"""
challenger_model.py — champion/challenger report for live expectancy logic.

This is a conservative promotion scaffold, not a black-box trading model.
It keeps one "champion" live profile, evaluates the best replay-backed
"challenger" rule set in shadow, and recommends promotion only when the
challenger has a clear edge on enough labeled decisions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import decision_review_lab
import precision_lab
from logger import get_logger
from paths import DECISION_REVIEW_REPORT_JSON

log = get_logger("challenger_model")


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


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read %s: %s", path, exc)
        return None


def _ensure_precision_report(cfg, data_dir: Path) -> dict:
    target_r = float(getattr(cfg.trading, "decision_review_target_r", 0.25) or 0.25)
    horizon_minutes = int(getattr(cfg.trading, "decision_review_horizon_minutes", 720) or 720)
    interval = str(getattr(cfg.trading, "decision_review_interval", "5m") or "5m")
    dedupe_minutes = int(getattr(cfg.trading, "decision_review_dedupe_minutes", 30) or 30)
    report = precision_lab.build_report(
        data_dir=data_dir,
        target_r=target_r,
        horizon_minutes=horizon_minutes,
        interval=interval,
        dedupe_minutes=dedupe_minutes,
    )
    report_path = data_dir / "precision_lab_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _ensure_decision_review_report(cfg, data_dir: Path) -> dict:
    report_path = data_dir / "decision_review_report.json"
    report = _load_json(report_path) or _load_json(DECISION_REVIEW_REPORT_JSON)
    if report:
        return report
    report = decision_review_lab.build_report(
        data_dir=data_dir,
        target_r=float(getattr(cfg.trading, "decision_review_target_r", 0.25) or 0.25),
        horizon_minutes=int(getattr(cfg.trading, "decision_review_horizon_minutes", 720) or 720),
        interval=str(getattr(cfg.trading, "decision_review_interval", "5m") or "5m"),
        dedupe_minutes=int(getattr(cfg.trading, "decision_review_dedupe_minutes", 30) or 30),
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


def build_report(cfg, *, data_dir: Path) -> dict:
    precision_path = data_dir / "precision_lab_report.json"
    precision_report = _load_json(precision_path) or _ensure_precision_report(cfg, data_dir)
    decision_review_report = _ensure_decision_review_report(cfg, data_dir)

    champion_wr = _safe_float(precision_report.get("overall_win_rate"))
    champion_samples = _safe_int(precision_report.get("labeled_episodes"))
    challenger_rule = dict(((precision_report.get("best_rules") or [{}])[0]) or {})
    challenger_wr = _safe_float(challenger_rule.get("win_rate"))
    challenger_samples = _safe_int(challenger_rule.get("samples"))

    review_counts = dict(decision_review_report.get("classifications") or {})
    labeled_decisions = _safe_int(decision_review_report.get("labeled_episodes"))
    missed_wins = _safe_int(review_counts.get("MISSED_WIN"))
    correct_passes = _safe_int(review_counts.get("CORRECT_PASS"))
    executed_bad = _safe_int(review_counts.get("BAD_TRADE"))
    executed_good = _safe_int(review_counts.get("GOOD_TRADE"))

    min_labeled = int(getattr(cfg.trading, "challenger_min_labeled_decisions", 25) or 25)
    min_edge = float(getattr(cfg.trading, "challenger_min_win_rate_edge", 0.04) or 0.04)
    shadow_ready = labeled_decisions >= min_labeled and challenger_samples >= max(3, min_labeled // 4)
    outperforming = challenger_wr >= (champion_wr + min_edge)
    promote = shadow_ready and outperforming

    if not shadow_ready:
        status = "INSUFFICIENT_DATA"
        summary = (
            f"Need more labeled decisions before trusting a challenger "
            f"({labeled_decisions}/{min_labeled})."
        )
    elif promote:
        status = "CHALLENGER_READY"
        summary = (
            f"Best challenger replay is beating the champion by "
            f"{(challenger_wr - champion_wr) * 100:.1f} pts on {challenger_samples} samples."
        )
    else:
        status = "CHAMPION_HOLDS"
        summary = "Current live logic still looks stronger than the best challenger preset."

    return {
        "generated_at": int(time.time()),
        "data_dir": str(data_dir),
        "status": status,
        "shadow_ready": shadow_ready,
        "promote": promote,
        "summary": summary,
        "champion": {
            "overall_win_rate": round(champion_wr, 4),
            "labeled_episodes": champion_samples,
        },
        "challenger": {
            "rule": challenger_rule,
            "win_rate": round(challenger_wr, 4),
            "samples": challenger_samples,
            "edge_vs_champion": round(challenger_wr - champion_wr, 4),
        },
        "decision_review": {
            "labeled_episodes": labeled_decisions,
            "missed_wins": missed_wins,
            "correct_passes": correct_passes,
            "good_trades": executed_good,
            "bad_trades": executed_bad,
            "missed_families": list(decision_review_report.get("missed_families") or [])[:6],
        },
    }


def build_and_save_report(cfg, *, data_dir: Path) -> dict:
    report = build_report(cfg, data_dir=data_dir)
    (data_dir / "challenger_model_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report
