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
