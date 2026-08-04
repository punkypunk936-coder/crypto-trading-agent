from __future__ import annotations

import time
import tempfile
from pathlib import Path
from types import SimpleNamespace

import micro_desk
import trade_dataset


def _cfg(**overrides):
    values = {
        "micro_desk_enabled": True,
        "micro_desk_history_limit": 200,
        "micro_desk_refresh_seconds": 300.0,
        "micro_desk_min_calibration_samples": 20,
        "micro_desk_max_calibrated_probability": 0.78,
        "analysis_signal_max_age_minutes": 20.0,
        "data_reliability_min_score": 58.0,
        "max_total_exposure_pct": 0.65,
        "micro_desk_max_directional_exposure_pct": 0.35,
        "micro_desk_min_net_edge_bps": 6.0,
        "micro_desk_aggressive_min_edge_bps": 35.0,
        "micro_desk_passive_haircut_threshold": 0.12,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _history_row(index: int, *, win: bool, probability: float = 0.95) -> dict:
    opened = 1_700_000_000.0 + index * 3600.0
    entry = 100.0
    exit_price = 101.0 if win else 99.0
    return {
        "trade_id": index + 1,
        "coin": "TEST",
        "direction": "LONG",
        "opened_at_ts": opened,
        "closed_at_ts": opened + 1800.0,
        "entry_price": entry,
        "exit_price": exit_price,
        "pnl_pct": 1.0 if win else -1.0,
        "pnl_usd": 1.0 if win else -1.0,
        "entry_context": {
            "instrument_type": "equity",
            "expectancy": {"probability": probability, "uncertainty": 0.20},
        },
    }


def _snapshot(*, updated_at: float | None = None, probability: float = 0.95) -> dict:
    return {
        "action": "LONG",
        "instrument_type": "equity",
        "analysis_updated_ts": updated_at if updated_at is not None else time.time(),
        "price": 100.0,
        "live_price": 100.0,
        "price_status": "OK",
        "data_reliability_score": 100.0,
        "planned_risk_pct": 1.0,
        "planned_reward_pct": 3.0,
        "trade_plan": {"risk_pct": 1.0, "reward_pct": 3.0, "risk_reward_ratio": 3.0},
        "expectancy": {"probability": probability, "uncertainty": 0.20},
        "execution_plan": {"mode": "market"},
    }


def _order(size_usd: float = 250.0):
    return SimpleNamespace(
        price=100.0,
        stop_loss=99.0,
        take_profit=103.0,
        size_usd=size_usd,
    )


def test_trade_dataset_deduplicates_csv_backfill_without_trusting_numeric_id() -> None:
    rich = _history_row(0, win=True)
    backfill = dict(rich)
    backfill.update({
        "opened_at_ts": rich["opened_at_ts"] - 20.0,
        "closed_at_ts": rich["closed_at_ts"] - 20.0,
        "entry_price": 100.00005,
        "backfilled_from_csv": True,
        "entry_context": {},
    })
    reused_id = _history_row(5, win=False)
    reused_id["trade_id"] = rich["trade_id"]

    rows = trade_dataset.deduplicate_closed_trades([backfill, rich, reused_id])

    assert len(rows) == 2
    assert sum(bool(row.get("backfilled_from_csv")) for row in rows) == 0
    assert len({row["trade_key"] for row in rows}) == 2


def test_micro_desk_shrinks_historically_overstated_probability(tmp_path: Path) -> None:
    for index in range(50):
        trade_dataset.append_closed_trade(
            _history_row(index, win=index < 20),
            data_dir=tmp_path,
        )

    desk = micro_desk.MicroDesk(_cfg(), data_dir=tmp_path)
    result = desk.assess(
        coin="TEST",
        direction="LONG",
        signal_snapshot=_snapshot(),
        order=_order(),
        portfolio_usd=10_000.0,
        execution_quality={"spread_bps": 2.0, "estimated_slippage_bps": 3.0},
    )

    assert result["raw_probability"] == 0.95
    assert result["calibrated_probability"] < 0.60
    assert result["size_multiplier"] <= 1.0
    assert desk.summary()["calibrated_brier"] < desk.summary()["raw_brier"]
    assert desk.summary()["calibration_method"] == "chronological_holdout"
    assert desk.summary()["calibration_holdout_samples"] > 0


def test_micro_desk_blocks_stale_state(tmp_path: Path) -> None:
    for index in range(25):
        trade_dataset.append_closed_trade(_history_row(index, win=index < 12), data_dir=tmp_path)
    desk = micro_desk.MicroDesk(_cfg(), data_dir=tmp_path)

    result = desk.assess(
        coin="TEST",
        direction="LONG",
        signal_snapshot=_snapshot(updated_at=time.time() - 7200),
        order=_order(),
        portfolio_usd=10_000.0,
        execution_quality={"spread_bps": 2.0, "estimated_slippage_bps": 3.0},
    )

    assert result["verdict"] == "BLOCK"
    assert "stale" in result["summary"]


def test_micro_desk_requires_passive_entry_when_confidence_is_overstated(tmp_path: Path) -> None:
    for index in range(60):
        trade_dataset.append_closed_trade(_history_row(index, win=index < 30), data_dir=tmp_path)
    desk = micro_desk.MicroDesk(_cfg(), data_dir=tmp_path)

    result = desk.assess(
        coin="TEST",
        direction="LONG",
        signal_snapshot=_snapshot(probability=0.95),
        order=_order(),
        portfolio_usd=10_000.0,
        execution_quality={"spread_bps": 2.0, "estimated_slippage_bps": 3.0},
    )

    assert result["permitted"] is True
    assert result["verdict"] == "PASSIVE_ONLY"
    assert result["net_edge_bps"] > 0
    assert result["size_multiplier"] < 1.0


if __name__ == "__main__":
    test_trade_dataset_deduplicates_csv_backfill_without_trusting_numeric_id()
    for check in (
        test_micro_desk_shrinks_historically_overstated_probability,
        test_micro_desk_blocks_stale_state,
        test_micro_desk_requires_passive_entry_when_confidence_is_overstated,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            check(Path(temp_dir))
    print("PASS micro desk truth, calibration, freshness, and passive-entry tests")
