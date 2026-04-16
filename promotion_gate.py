"""
promotion_gate.py — live-capital promotion gate backed by real agent history.

The goal is not to promise perfection. It is to block live trading until the
agent has earned it through recent closed-trade performance plus replay-tested
decision quality from the precision lab.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import challenger_model
from logger import get_logger
from paths import DATA_DIR
import precision_lab
import trade_dataset

log = get_logger("promotion_gate")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _load_precision_report(report_path: Path) -> dict | None:
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read precision report %s: %s", report_path, exc)
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _count_decision_rows(data_dir: Path) -> int:
    dataset_path = data_dir / "decision_dataset.jsonl"
    if not dataset_path.exists():
        return 0
    try:
        with dataset_path.open(encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except Exception:
        return 0


def _resolve_precision_data_dir(preferred: Path | None = None) -> Path:
    candidates: list[Path] = []
    for candidate in (preferred, DATA_DIR, trade_dataset.RUNTIME_DATA_DIR):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path not in candidates:
            candidates.append(path)

    scored: list[tuple[tuple[int, int, int], Path]] = []
    for path in candidates:
        decision_rows = _count_decision_rows(path)
        feature_rows = 0
        feature_path = path / "feature_store.jsonl"
        if feature_path.exists():
            try:
                with feature_path.open(encoding="utf-8") as handle:
                    feature_rows = sum(1 for line in handle if line.strip())
            except Exception:
                feature_rows = 0
        trade_rows = len(trade_dataset.load_csv_closed_trades(data_dir=path))
        scored.append(((decision_rows, feature_rows, trade_rows), path))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1] if scored else Path(preferred or DATA_DIR).expanduser()


def _ensure_trade_history(cfg, data_dir: Path) -> tuple[list[dict], dict]:
    lookback = int(getattr(cfg.trading, "live_promotion_lookback_closed_trades", 40) or 40)
    try:
        backfill_info = trade_dataset.ensure_backfilled_from_csv(data_dir=data_dir)
    except Exception as exc:
        log.warning("Closed-trade backfill failed for %s: %s", data_dir, exc)
        backfill_info = {"data_dir": str(data_dir), "appended_rows": 0, "csv_rows": 0, "structured_rows_before": 0}

    try:
        rows = trade_dataset.load_closed_trades(limit=lookback, data_dir=data_dir, backfill_from_csv=True)
    except TypeError:
        # Test doubles may still expose the legacy signature.
        rows = trade_dataset.load_closed_trades(limit=lookback)

    normalized = [
        row for row in rows
        if str(row.get("outcome", "")).upper() in {"WIN", "LOSS", "BREAKEVEN"}
    ]
    return normalized, backfill_info


def _ensure_precision_report(cfg, data_dir: Path) -> tuple[dict | None, Path]:
    report_path = data_dir / "precision_lab_report.json"
    report = _load_precision_report(report_path)
    max_age_seconds = float(
        getattr(cfg.trading, "live_promotion_report_max_age_hours", 12.0) or 12.0
    ) * 3600.0
    report_age = time.time() - _safe_float((report or {}).get("generated_at"))
    decision_rows_on_disk = _count_decision_rows(data_dir)
    report_decision_rows = _safe_int((report or {}).get("decision_rows"))
    report_zeroed = bool(report and report_decision_rows <= 0 and decision_rows_on_disk > 0)
    is_fresh = bool(report and report_age <= max_age_seconds and not report_zeroed)
    if is_fresh:
        return report, report_path

    try:
        report = precision_lab.build_report(
            data_dir=data_dir,
            target_r=float(getattr(cfg.trading, "live_promotion_precision_target_r", 0.25) or 0.25),
            horizon_minutes=int(
                getattr(cfg.trading, "live_promotion_precision_horizon_minutes", 720) or 720
            ),
            interval=str(getattr(cfg.trading, "live_promotion_precision_interval", "5m") or "5m"),
            dedupe_minutes=int(
                getattr(cfg.trading, "live_promotion_precision_dedupe_minutes", 30) or 30
            ),
        )
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report, report_path
    except Exception as exc:
        log.warning("Precision report refresh failed: %s", exc)
        return report, report_path


def evaluate_live_promotion(cfg, data_dir: Path | None = None) -> dict:
    preferred_dir = Path(data_dir or DATA_DIR).expanduser()
    trade_history_dir = trade_dataset.resolve_richest_history_data_dir(preferred_dir)
    precision_dir = _resolve_precision_data_dir(preferred_dir)
    lookback = int(getattr(cfg.trading, "live_promotion_lookback_closed_trades", 40) or 40)
    min_closed = int(getattr(cfg.trading, "live_promotion_min_closed_trades", 20) or 20)
    min_wr = float(getattr(cfg.trading, "live_promotion_min_win_rate", 0.58) or 0.58)
    min_avg_pnl_pct = float(getattr(cfg.trading, "live_promotion_min_avg_pnl_pct", 0.10) or 0.10)
    min_profit_factor = float(getattr(cfg.trading, "live_promotion_min_profit_factor", 1.15) or 1.15)
    min_precision_samples = int(getattr(cfg.trading, "live_promotion_min_precision_samples", 6) or 6)
    min_precision_wr = float(getattr(cfg.trading, "live_promotion_min_precision_win_rate", 0.60) or 0.60)

    closed_rows, backfill_info = _ensure_trade_history(cfg, trade_history_dir)
    wins = sum(1 for row in closed_rows if str(row.get("outcome", "")).upper() == "WIN")
    losses = sum(1 for row in closed_rows if str(row.get("outcome", "")).upper() == "LOSS")
    breakeven = sum(1 for row in closed_rows if str(row.get("outcome", "")).upper() == "BREAKEVEN")
    total = len(closed_rows)
    avg_pnl_pct = (
        sum(_safe_float(row.get("pnl_pct")) for row in closed_rows) / total
        if total else 0.0
    )
    gross_profit = sum(max(0.0, _safe_float(row.get("pnl_usd"))) for row in closed_rows)
    gross_loss = sum(abs(min(0.0, _safe_float(row.get("pnl_usd")))) for row in closed_rows)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    win_rate = (wins / total) if total else 0.0

    precision_report, report_path = _ensure_precision_report(cfg, precision_dir)
    precision_samples = int((precision_report or {}).get("labeled_episodes", 0) or 0)
    precision_wr = _safe_float((precision_report or {}).get("overall_win_rate"))
    best_rule = dict(((precision_report or {}).get("best_rules") or [{}])[0] or {})
    challenger_report = {}
    if getattr(cfg.trading, "challenger_model_enabled", True):
        try:
            challenger_report = challenger_model.build_and_save_report(cfg, data_dir=precision_dir)
        except Exception as exc:
            log.warning("Challenger report refresh failed: %s", exc)
            challenger_report = {}

    blockers: list[str] = []
    if total < min_closed:
        blockers.append(f"only {total}/{min_closed} recent closed trades available")
    if total and win_rate < min_wr:
        blockers.append(f"recent win rate {win_rate * 100:.1f}% is below {min_wr * 100:.1f}%")
    if total and avg_pnl_pct < min_avg_pnl_pct:
        blockers.append(f"recent average pnl {avg_pnl_pct:.2f}% is below {min_avg_pnl_pct:.2f}%")
    if total and profit_factor < min_profit_factor:
        blockers.append(f"profit factor {profit_factor:.2f} is below {min_profit_factor:.2f}")
    if precision_samples < min_precision_samples:
        blockers.append(f"precision lab has only {precision_samples}/{min_precision_samples} labeled episodes")
    elif precision_wr < min_precision_wr:
        blockers.append(
            f"precision replay win rate {precision_wr * 100:.1f}% is below {min_precision_wr * 100:.1f}%"
        )

    return {
        "passed": not blockers,
        "data_dir": str(trade_history_dir),
        "data_dirs": {
            "trade_history": str(trade_history_dir),
            "precision_replay": str(precision_dir),
        },
        "history_source": {
            "structured_trade_rows": backfill_info.get("structured_rows_before", 0) + backfill_info.get("appended_rows", 0),
            "csv_trade_rows": backfill_info.get("csv_rows", 0),
            "backfilled_trade_rows": backfill_info.get("appended_rows", 0),
        },
        "trade_metrics": {
            "closed_trades": total,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "win_rate": round(win_rate, 4),
            "avg_pnl_pct": round(avg_pnl_pct, 4),
            "profit_factor": round(profit_factor, 4),
        },
        "precision_report_path": str(report_path),
        "precision_metrics": {
            "labeled_episodes": precision_samples,
            "overall_win_rate": round(precision_wr, 4),
            "best_rule": best_rule,
        },
        "challenger_metrics": {
            "status": challenger_report.get("status", ""),
            "shadow_ready": bool(challenger_report.get("shadow_ready", False)),
            "promote": bool(challenger_report.get("promote", False)),
            "summary": challenger_report.get("summary", ""),
            "champion_win_rate": _safe_float(((challenger_report.get("champion") or {}).get("overall_win_rate"))),
            "challenger_win_rate": _safe_float(((challenger_report.get("challenger") or {}).get("win_rate"))),
        },
        "blockers": blockers,
    }
