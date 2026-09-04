from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from dashboard.snapshot import build_dashboard_snapshot
from weekly_outlook import build_weekly_outlook


NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


def _signal(
    instrument_type: str,
    *,
    direction: str = "LONG",
    fresh: bool = True,
    move: float = 1.0,
) -> dict:
    updated = NOW if fresh else NOW - timedelta(days=2)
    bullish = direction == "LONG"
    return {
        "instrument_type": instrument_type,
        "action": direction,
        "market_regime": "UPTREND" if bullish else "DOWNTREND",
        "structure_trend": "UPTREND" if bullish else "DOWNTREND",
        "mtf_bias": "BULLISH" if bullish else "BEARISH",
        "market_map_bias": "BULLISH" if bullish else "BEARISH",
        "crypto_market_mode": "RISK_ON" if bullish else "RISK_OFF",
        "move_pct_24h": move,
        "funding_rate": 0.00001,
        "analysis_updated_ts": updated.timestamp(),
        "analysis_updated_at": updated.isoformat(),
        "market_map_nearest_support": 77_500,
        "market_map_nearest_resistance": 78_500,
    }


def _global_context(regime: str = "RISK_ON") -> dict:
    direction = "LONG" if regime == "RISK_ON" else "SHORT"
    return {
        "active": True,
        "regime": regime,
        "cross_market": {"tactical_bias": direction, "state": "SUPPORTED"},
        "benchmarks": [
            {"symbol": "SP500", "fresh": True, "direction": direction, "support": 7_500, "resistance": 7_700},
            {"symbol": "NDX", "fresh": True, "direction": direction},
            {"symbol": "VIXINDEX", "fresh": True, "direction": "SHORT" if direction == "LONG" else "LONG"},
        ],
    }


def _ai_context(move_5d: float = 4.0, move_20d: float = 9.0) -> dict:
    return {
        "active": True,
        "sectors": [
            {"label": "AI Chips", "move_5d": move_5d, "move_20d": move_20d, "breadth_pct": 75.0},
            {"label": "Servers & Racks", "move_5d": move_5d / 2, "move_20d": move_20d / 2, "breadth_pct": 65.0},
        ],
    }


def _bullish_state() -> dict:
    signals = {
        **{f"EQ{index}": _signal("equity") for index in range(12)},
        **{symbol: _signal("crypto") for symbol in ("BTC", "ETH", "SOL", "HYPE")},
        **{f"COIN{index}": _signal("crypto") for index in range(8)},
    }
    signals["BTC"]["crypto_structure_summary"] = "Four major coins are holding bullish structure."
    return {"signals": signals, "ai_infrastructure": _ai_context()}


def test_weekly_outlook_builds_bounded_green_leans_from_aligned_fresh_evidence() -> None:
    report = build_weekly_outlook(
        _bullish_state(),
        global_context=_global_context(),
        ai_infrastructure=_ai_context(),
        now=NOW,
    )

    assert report["active"] is True
    assert report["week_id"] == "2026-W36"
    assert report["markets"]["stocks"]["next_week"]["call"] == "GREEN LEAN"
    assert report["markets"]["crypto"]["next_week"]["call"] == "GREEN LEAN"
    for market in report["markets"].values():
        for horizon in (market["next_week"], market["next_month"]):
            assert 0.32 <= horizon["green_probability"] <= 0.68
            assert 0.32 <= horizon["red_probability"] <= 0.68
            assert horizon["green_probability"] + horizon["red_probability"] == 1.0
        assert len(market["evidence"]) == 3
        assert market["invalidation"]


def test_weekly_outlook_refuses_to_call_stale_markets() -> None:
    stale_state = {
        "signals": {
            "BTC": _signal("crypto", fresh=False),
            "ETH": _signal("crypto", fresh=False),
            "AAPL": _signal("equity", fresh=False),
        }
    }
    report = build_weekly_outlook(stale_state, now=NOW)

    assert report["active"] is False
    assert report["markets"]["stocks"]["active"] is False
    assert report["markets"]["crypto"]["active"] is False
    assert report["markets"]["stocks"]["next_week"]["call"] == "NO FRESH CALL"
    assert report["markets"]["crypto"]["next_month"]["call"] == "NO FRESH CALL"


def test_weekly_outlook_neutralizes_both_crypto_horizons_when_major_coverage_is_thin() -> None:
    state = {
        "signals": {
            "BTC": _signal("crypto"),
            **{f"COIN{index}": _signal("crypto") for index in range(10)},
        }
    }
    report = build_weekly_outlook(state, now=NOW)
    crypto = report["markets"]["crypto"]

    assert crypto["active"] is False
    assert crypto["next_week"]["call"] == "NO FRESH CALL"
    assert crypto["next_month"]["call"] == "NO FRESH CALL"


def test_weekly_outlook_keeps_conflicting_evidence_mixed() -> None:
    state = _bullish_state()
    equity_symbols = [symbol for symbol in state["signals"] if symbol.startswith("EQ")]
    for symbol in equity_symbols[:6]:
        state["signals"][symbol] = _signal("equity", direction="SHORT", move=-1.0)
    report = build_weekly_outlook(
        state,
        global_context={
            **_global_context("RISK_ON"),
            "cross_market": {"tactical_bias": "SHORT", "state": "DIVERGENT"},
        },
        ai_infrastructure=_ai_context(move_5d=-2.0, move_20d=-2.0),
        now=NOW,
    )

    stock_week = report["markets"]["stocks"]["next_week"]
    assert stock_week["call"] == "MIXED"
    assert 0.44 < stock_week["green_probability"] < 0.56


def test_dashboard_snapshot_and_static_bundle_expose_weekly_note() -> None:
    state = _bullish_state()
    state.update({
        "status": "running",
        "mode": "dry_run",
        "positions": [],
        "pending_orders": [],
        "portfolio_usd": 1_000,
        "available_usd": 1_000,
    })
    snapshot = build_dashboard_snapshot(state, [])

    assert snapshot["schemaVersion"] >= 5
    assert snapshot["weekly_outlook"]["markets"]["stocks"]
    template = Path("dashboard/templates/dashboard.html").read_text()
    hosted = Path("netlify-dashboard/public/index.html").read_text()
    assert template == hosted
    assert 'id="weekly-note"' in template
    assert "function renderWeeklyOutlook" in template
    assert "latestWeeklyOutlook = data.weekly_outlook" in template
