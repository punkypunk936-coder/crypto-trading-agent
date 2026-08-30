from __future__ import annotations

from pathlib import Path

import ask_call_learning


START_TS = int(ask_call_learning._parse_ts("2026-08-20T00:00:00Z"))


def _record(**overrides) -> dict:
    row = {
        "id": "mrna-short",
        "ticker": "MRNA",
        "direction": "SHORT",
        "analysedAt": "2026-08-20T00:00:00Z",
        "horizonHours": 24,
        "venueSymbol": "xyz:MRNA",
        "current": 165.0,
        "target": 155.0,
        "invalidation": 170.0,
        "probability": 0.62,
    }
    row.update(overrides)
    return row


def _bar(ts: int, *, high: float, low: float) -> dict:
    return {"t": ts * 1000, "T": (ts + 899) * 1000, "h": high, "l": low, "c": (high + low) / 2}


def test_settlement_rewards_target_before_invalidation() -> None:
    normalized = ask_call_learning._normalize_record(_record())
    assert normalized is not None
    settled = ask_call_learning._settle_record(
        normalized,
        [_bar(START_TS, high=166.0, low=154.0)],
        START_TS + 3_000,
    )
    assert settled["outcome"] == 1
    assert settled["outcomeReason"] == "target_first"
    assert settled["resolutionSource"] == "venue_candle_path_v2"


def test_settlement_scores_same_candle_ambiguity_as_a_miss() -> None:
    normalized = ask_call_learning._normalize_record(_record())
    assert normalized is not None
    settled = ask_call_learning._settle_record(
        normalized,
        [_bar(START_TS, high=171.0, low=154.0)],
        START_TS + 3_000,
    )
    assert settled["outcome"] == 0
    assert settled["outcomeReason"] == "target_and_invalidation_same_candle"


def test_client_cannot_claim_an_unverified_win(tmp_path: Path) -> None:
    rows = ask_call_learning.upsert_forecasts(
        [{**_record(), "outcome": 1, "resolutionSource": "client_claim"}],
        data_dir=tmp_path,
        settle=False,
    )
    assert rows[0]["outcome"] is None
    assert rows[0]["resolutionSource"] == ""


def test_conditional_plan_waits_for_entry_before_it_can_fail() -> None:
    normalized = ask_call_learning._normalize_record(_record(
        id="conditional-long",
        ticker="TEST",
        direction="LONG",
        current=100.0,
        entry=105.0,
        target=110.0,
        invalidation=95.0,
    ))
    assert normalized is not None

    pending = ask_call_learning._settle_record(
        normalized,
        [_bar(START_TS, high=104.0, low=94.0)],
        START_TS + 3_000,
    )
    assert pending["outcome"] is None
    assert pending["status"] == "pending_entry"
    assert pending["activatedAt"] is None

    active = ask_call_learning._settle_record(
        pending,
        [_bar(START_TS + 3_600, high=106.0, low=100.0)],
        START_TS + 7_000,
    )
    assert active["outcome"] is None
    assert active["status"] == "active"
    assert active["activatedAt"] is not None

    resolved = ask_call_learning._settle_record(
        active,
        [_bar(START_TS + 7_200, high=111.0, low=103.0)],
        START_TS + 11_000,
    )
    assert resolved["outcome"] == 1
    assert resolved["outcomeReason"] == "target_first"


def test_untriggered_plan_expiry_does_not_become_a_fake_loss() -> None:
    normalized = ask_call_learning._normalize_record(_record(
        id="never-entered",
        ticker="TEST",
        direction="LONG",
        current=100.0,
        entry=105.0,
        target=110.0,
        invalidation=95.0,
        horizonHours=1,
    ))
    assert normalized is not None
    settled = ask_call_learning._settle_record(
        normalized,
        [
            _bar(START_TS, high=104.0, low=96.0),
            _bar(START_TS + 3_599, high=104.0, low=96.0),
        ],
        START_TS + 3_700,
    )
    assert settled["outcome"] is None
    assert settled["status"] == "expired_untriggered"
    assert settled["outcomeReason"] == "entry_never_reached"


def test_snapshot_records_and_exposes_displayed_plan_accountability(tmp_path: Path) -> None:
    snapshot = {
        "action_board": {
            "items": [{
                "coin": "AMZN",
                "asset_bucket": "equity",
                "analysis_fresh": True,
                "analysis_updated_ts": START_TS,
                "thesis_direction": "LONG",
                "probability": 0.61,
                "plain_plan": {
                    "direction": "LONG",
                    "current": 100.0,
                    "entry": 105.0,
                    "target": 115.0,
                    "invalidation": 100.0,
                    "reward_risk": 2.0,
                },
            }],
        },
    }
    enriched = ask_call_learning.refresh_snapshot_accountability(
        snapshot,
        data_dir=tmp_path,
        settle=False,
    )
    item = enriched["action_board"]["items"][0]
    assert item["plan_record_id"].startswith("displayed:AMZN:LONG:")
    assert item["plan_accuracy"]["current_status"] == "pending_entry"
    assert enriched["plan_accountability"]["records"] == 1
    assert enriched["plan_accountability"]["pending_entry"] == 1


def test_terminal_untriggered_rows_do_not_block_the_settlement_queue(monkeypatch) -> None:
    rows = []
    for index in range(ask_call_learning.MAX_SETTLEMENTS_PER_PASS):
        terminal = ask_call_learning._normalize_record(_record(id=f"terminal-{index}"))
        assert terminal is not None
        terminal.update({"status": "expired_untriggered", "outcome": None})
        rows.append(terminal)
    active = ask_call_learning._normalize_record(_record(id="still-needs-settlement"))
    assert active is not None
    rows.append(active)

    monkeypatch.setattr(
        ask_call_learning,
        "_fetch_candles",
        lambda record, now_ts: [_bar(START_TS, high=166.0, low=154.0)],
    )
    settled = ask_call_learning.settle_forecasts(rows, now_ts=START_TS + 3_000)
    assert settled[-1]["outcome"] == 1


def test_runtime_deploy_preserves_the_durable_forecast_ledger() -> None:
    deploy_script = Path("deploy_local_dashboard.sh").read_text(encoding="utf-8")
    assert '--exclude "ask_forecast_ledger.jsonl"' in deploy_script
