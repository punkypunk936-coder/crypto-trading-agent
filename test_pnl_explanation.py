from dashboard.snapshot import build_dashboard_snapshot
from feature_store import build_closed_trade_feature_row
from pnl_explanation import (
    build_pnl_attribution_summary,
    explain_closed_trade,
    explain_open_position,
)


def test_open_position_explains_unrealized_loss_and_hold_decision() -> None:
    position = {
        "coin": "CRCL",
        "direction": "SHORT",
        "entry_price": 61.06,
        "current_price": 61.199,
        "size_usd": 283.1577,
        "margin_usd": 94.3859,
        "leverage": 3,
        "unrealised_pnl": -0.64,
        "stop_loss": 63.04,
        "loss_realization_hard_stop": 63.04,
        "hold_minutes": 111,
        "min_hold_minutes": 240,
        "loss_realization_guard_active": True,
        "loss_realization_guard_reason": "thesis intact at 3x; hold through volatility",
        "current_logic": "Hold until hard invalidation.",
    }
    signal = {
        "action": "SHORT",
        "structure_trend": "DOWNTREND",
        "mtf_bias": "BEARISH",
        "orderbook_breakout_state": "CONFIRMED_BEARISH_BREAKDOWN",
        "thesis": {"state": "QUALIFIED", "quality": "HIGH", "permitted": True, "conflict_points": 0},
    }

    explanation = explain_open_position(position, signal)

    assert explanation["state"] == "loss"
    assert explanation["decision"] == "HOLD_TO_HARD_INVALIDATION"
    assert explanation["thesis"]["intact"] is True
    assert explanation["reinforcement"]["provisional"] is True
    assert explanation["pnl_usd"] == -0.64
    assert "283.16 notional" in explanation["calculation"]
    assert explanation["data_quality"]["complete"] is True


def test_closed_trade_assigns_loss_to_structure_level_and_exit_trigger() -> None:
    trade = {
        "trade_id": 7,
        "coin": "CRWV",
        "direction": "SHORT",
        "entry_price": 72.55,
        "exit_price": 74.42,
        "size_usd": 284.0,
        "pnl_usd": -7.33,
        "pnl_pct": -2.58,
        "exit_reason": "stop_loss",
        "hold_minutes": 43,
    }
    record = {
        "entry_context": {
            "structure_trend": "UPTREND",
            "mtf_bias": "BULLISH",
            "orderbook_interaction": "AT_SUPPORT",
            "trade_plan": {"stop_loss": 74.4},
        },
        "exit_context": {
            "structure_trend": "UPTREND",
            "mtf_bias": "BULLISH",
            "orderbook_interaction": "AT_SUPPORT",
        },
    }

    explanation = explain_closed_trade(trade, record)

    assert explanation["primary_cause_code"] == "COUNTERTREND_OR_STRUCTURE_CONFLICT"
    assert "ENTRY_INTO_OPPOSING_LEVEL" in explanation["cause_codes"]
    assert "HARD_STOP_HIT" in explanation["cause_codes"]
    assert explanation["reinforcement"]["reward_normalized"] < 0
    assert explanation["reinforcement"]["entry_credit"] == -1.0
    assert explanation["reinforcement"]["thesis_outcome"] == "failed"


def test_summary_reconciles_gross_wins_losses_and_primary_causes() -> None:
    loss = {
        "coin": "BTC",
        "direction": "LONG",
        "entry_price": 100,
        "exit_price": 95,
        "size_usd": 200,
        "pnl_usd": -10,
        "pnl_pct": -5,
        "exit_reason": "structure_invalidation",
    }
    loss["pnl_explanation"] = explain_closed_trade(loss)
    win = {
        "coin": "ETH",
        "direction": "LONG",
        "entry_price": 100,
        "exit_price": 104,
        "size_usd": 200,
        "pnl_usd": 8,
        "pnl_pct": 4,
        "exit_reason": "take_profit",
    }
    win["pnl_explanation"] = explain_closed_trade(win)

    summary = build_pnl_attribution_summary([loss, win], [{"unrealised_pnl": -1.25}])

    assert summary["realized_pnl_usd"] == -2.0
    assert summary["tracked_pnl_usd"] == -3.25
    assert summary["gross_profit_usd"] == 8.0
    assert summary["gross_loss_usd"] == 10.0
    assert summary["wins"] == 1
    assert summary["losses"] == 1
    assert summary["top_loss_causes"][0]["code"] == "STRUCTURE_INVALIDATION"


def test_snapshot_and_feature_store_share_the_same_attribution_labels() -> None:
    state = {
        "status": "running",
        "positions": [{
            "coin": "BTC",
            "direction": "LONG",
            "entry_price": 100,
            "current_price": 98,
            "size_usd": 200,
            "margin_usd": 100,
            "leverage": 2,
            "unrealised_pnl": -4,
            "stop_loss": 94,
        }],
        "signals": {"BTC": {"action": "LONG", "structure_trend": "UPTREND", "thesis": {"state": "QUALIFIED"}}},
    }
    closed = {
        "trade_id": 1,
        "coin": "ETH",
        "direction": "SHORT",
        "entry_price": 100,
        "exit_price": 103,
        "size_usd": 100,
        "pnl_usd": -3,
        "pnl_pct": -3,
        "exit_reason": "signal_reversal",
        "closed_at": "2026-07-19 10:00",
    }
    dataset_record = {
        **closed,
        "entry_context": {"structure_trend": "UPTREND", "mtf_bias": "BULLISH"},
        "exit_context": {"structure_trend": "UPTREND", "mtf_bias": "BULLISH"},
    }

    snapshot = build_dashboard_snapshot(state, [closed], trade_dataset_records=[dataset_record])
    explanation = snapshot["trades"][0]["pnl_explanation"]
    feature_record = {**dataset_record, "pnl_explanation": explanation, "reinforcement": explanation["reinforcement"]}
    feature_row = build_closed_trade_feature_row(feature_record)

    assert snapshot["state"]["positions"][0]["pnl_explanation"]["scope"] == "open_position"
    assert snapshot["pnl_attribution"]["realized_pnl_usd"] == -3.0
    assert feature_row["labels"]["primary_cause_code"] == explanation["primary_cause_code"]
    assert feature_row["labels"]["reward_normalized"] < 0
    assert feature_row["features"]["ctx_pnl_primary_cause"] == explanation["primary_cause_code"].lower()


def run_all() -> None:
    tests = [
        test_open_position_explains_unrealized_loss_and_hold_decision,
        test_closed_trade_assigns_loss_to_structure_level_and_exit_trigger,
        test_summary_reconciles_gross_wins_losses_and_primary_causes,
        test_snapshot_and_feature_store_share_the_same_attribution_labels,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    run_all()
