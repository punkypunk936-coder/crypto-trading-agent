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


def _ensure_precision_report(cfg, data_dir: Path) -> tuple[dict | None, Path]:
    report_path = data_dir / "precision_lab_report.json"
    report = _load_precision_report(report_path)
    max_age_seconds = float(
        getattr(cfg.trading, "live_promotion_report_max_age_hours", 12.0) or 12.0
    ) * 3600.0
    report_age = time.time() - _safe_float((report or {}).get("generated_at"))
    is_fresh = bool(report and report_age <= max_age_seconds)
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
    target_dir = Path(data_dir or DATA_DIR).expanduser()
    lookback = int(getattr(cfg.trading, "live_promotion_lookback_closed_trades", 40) or 40)
    min_closed = int(getattr(cfg.trading, "live_promotion_min_closed_trades", 20) or 20)
    min_wr = float(getattr(cfg.trading, "live_promotion_min_win_rate", 0.58) or 0.58)
    min_avg_pnl_pct = float(getattr(cfg.trading, "live_promotion_min_avg_pnl_pct", 0.10) or 0.10)
    min_profit_factor = float(getattr(cfg.trading, "live_promotion_min_profit_factor", 1.15) or 1.15)
    min_precision_samples = int(getattr(cfg.trading, "live_promotion_min_precision_samples", 6) or 6)
    min_precision_wr = float(getattr(cfg.trading, "live_promotion_min_precision_win_rate", 0.60) or 0.60)

    closed_rows = trade_dataset.load_closed_trades(limit=lookback)
    closed_rows = [row for row in closed_rows if str(row.get("outcome", "")).upper() in {"WIN", "LOSS", "BREAKEVEN"}]
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

    precision_report, report_path = _ensure_precision_report(cfg, target_dir)
    precision_samples = int((precision_report or {}).get("labeled_episodes", 0) or 0)
    precision_wr = _safe_float((precision_report or {}).get("overall_win_rate"))
    best_rule = dict(((precision_report or {}).get("best_rules") or [{}])[0] or {})

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
        "data_dir": str(target_dir),
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
        "blockers": blockers,
    }
