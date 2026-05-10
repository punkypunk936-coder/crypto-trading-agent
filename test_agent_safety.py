"""
test_agent_safety.py — basic safety regressions for the live agent.

Run:
    python3 test_agent_safety.py
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd

import checkpoint as checkpoint_module
import agent as agent_module
import analog_engine as analog_engine_module
import asset_dossier as asset_dossier_module
import asset_state_machine as asset_state_machine_module
import backtest as backtest_module
import challenger_model as challenger_model_module
import data_reliability as data_reliability_module
import decision_dataset as decision_dataset_module
import decision_review_lab as decision_review_lab_module
import execution_coach as execution_coach_module
import feature_store as feature_store_module
import first_principles as first_principles_module
import hosted_state_sync as hosted_state_sync_module
import llm_referee as llm_referee_module
import main as main_module
import market_map as market_map_module
import market_universe as market_universe_module
import missed_move_lab as missed_move_lab_module
import narrative as narrative_module
import portfolio_guard as portfolio_guard_module
import performance_intelligence as performance_intelligence_module
import playbook_distiller as playbook_distiller_module
import precision_lab as precision_lab_module
import proactive_intelligence as proactive_intelligence_module
import promotion_gate as promotion_gate_module
from data import market_data as market_data_module
from indicators import equity_event_feeds as equity_event_feeds_module
from indicators import news as news_module
from indicators import orderbook_levels as orderbook_levels_module
from indicators import social_attention as social_attention_module
import trade_dataset as trade_dataset_module
import trade_logger as trade_logger_module
import trade_review as trade_review_module
import tradexyz_volume as tradexyz_volume_module
from indicators import trade_memory as trade_memory_module
from agent import TradingAgent
from config import Config
from circuit_breaker import CircuitBreakerError
from dashboard import app as dashboard_module
from dashboard.snapshot import build_dashboard_snapshot
from exchanges.base import AccountState, BaseExchange, OrderResult, LimitOrderStatus
from exchanges.dry_run import DryRunExchange
from exchanges import hyperliquid_client as hyperliquid_client_module
from exchanges import hyperliquid_markets as hyperliquid_markets_module
from exchanges import lighter_client as lighter_client_module
from risk.risk_manager import OrderRequest, OpenPosition
from strategy.aggressive_strategy import AggressiveStrategy
from strategy.order_manager import OrderManager, PendingOrder


class StubExchange(BaseExchange):
    def __init__(self, name: str, should_fill: bool):
        self.name = name
        self.should_fill = should_fill
        self.market_buy_calls = 0
        self.close_calls = 0
        self._positions = []

    def connect(self) -> bool:
        return True

    def get_account_state(self):
        return AccountState(total_equity_usd=1000.0, available_usd=1000.0, positions=list(self._positions))

    def set_leverage(self, coin: str, leverage: int) -> bool:
        return True

    def market_buy(self, coin: str, size_coin: float, slippage: float = 0.01) -> OrderResult:
        self.market_buy_calls += 1
        if self.should_fill:
            self._positions = [{
                "coin": coin,
                "size": size_coin,
                "direction": "LONG",
                "entry_price": 100.0,
            }]
            return OrderResult(success=True, filled_price=100.0, filled_size=size_coin)
        return OrderResult(success=False, error="intentional fail")

    def market_sell(self, coin: str, size_coin: float, slippage: float = 0.01) -> OrderResult:
        return self.market_buy(coin, size_coin, slippage)

    def close_position(self, coin: str) -> OrderResult:
        self.close_calls += 1
        self._positions = []
        return OrderResult(success=True, filled_price=100.0)

    def get_order_status(self, coin: str, order_id: str) -> LimitOrderStatus:
        return LimitOrderStatus(order_id=order_id, coin=coin, filled=False)

    def supported_coins(self):
        return ["BTC"]


class FailingCloseExchange(StubExchange):
    def close_position(self, coin: str) -> OrderResult:
        self.close_calls += 1
        return OrderResult(success=False, error="intentional close failure")


class Signal:
    def __init__(self, action: str):
        self.action = action
        self.score = 70.0


def build_config() -> Config:
    cfg = Config()
    cfg.trading.dry_run = True
    cfg.trading.coins = ["BTC"]
    cfg.trading.max_trade_usd = 100.0
    cfg.trading.min_trade_usd = 100.0
    cfg.trading.use_news = False
    cfg.trading.use_narrative_gate = False
    cfg.trading.precision_mode_enabled = False
    cfg.trading.enforce_active_venue_markets = False
    cfg.trading.dynamic_market_cap_watchlist_enabled = False
    cfg.trading.live_promotion_gate_enabled = False
    cfg.trading.require_notifications_for_live = False
    cfg.trading.setup_quality_guard_enabled = False
    cfg.trading.first_principles_guard_enabled = False
    cfg.trading.performance_edge_guard_enabled = False
    cfg.trading.use_social_attention = False
    return cfg


def test_checkpoint_recovery() -> None:
    cfg = build_config()
    original_manager = checkpoint_module.checkpoint_manager
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        temp_manager = checkpoint_module.CheckpointManager(db_path=str(db_path))
        checkpoint_module.checkpoint_manager = temp_manager
        agent_module.checkpoint_manager = temp_manager

        ex = DryRunExchange(starting_balance_usd=1000.0)
        ex.connect()
        live = TradingAgent(cfg, [ex])
        order = OrderRequest(
            coin="BTC",
            direction="LONG",
            size_usd=100.0,
            size_coin=0.001,
            price=70000.0,
            stop_loss=63000.0,
            take_profit=105000.0,
            leverage=2,
            approved=True,
        )
        live.risk.record_open(order, exchange=ex.name)
        live._last_portfolio_usd = 1000.0
        live._last_available_usd = 900.0
        live._save_checkpoint()

        ex_restarted = DryRunExchange(starting_balance_usd=1000.0)
        ex_restarted.connect()
        restarted = TradingAgent(cfg, [ex_restarted])
        assert "BTC" in restarted.risk.positions, "position should restore from checkpoint in dry-run"
    checkpoint_module.checkpoint_manager = original_manager
    agent_module.checkpoint_manager = original_manager


def test_checkpoint_recovery_rehydrates_dry_run_pending_limits() -> None:
    cfg = build_config()
    original_manager = checkpoint_module.checkpoint_manager
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        temp_manager = checkpoint_module.CheckpointManager(db_path=str(db_path))
        checkpoint_module.checkpoint_manager = temp_manager
        agent_module.checkpoint_manager = temp_manager
        try:
            ex = DryRunExchange(starting_balance_usd=1000.0)
            ex.connect()
            live = TradingAgent(cfg, [ex])
            result = ex.limit_buy("BTC", 0.001, 69000.0)
            live.order_mgr.register_limit_order(
                PendingOrder(
                    coin="BTC",
                    direction="LONG",
                    limit_price=69000.0,
                    size_coin=0.001,
                    size_usd=69.0,
                    stop_loss=63000.0,
                    take_profit=78000.0,
                    signal_score=66.0,
                    exchange=ex.name,
                    exchange_order_id=result.order_id,
                    reason="initial_limit",
                )
            )
            live._last_portfolio_usd = 1000.0
            live._last_available_usd = 931.0
            live._save_checkpoint()

            ex_restarted = DryRunExchange(starting_balance_usd=1000.0)
            ex_restarted.connect()
            restarted = TradingAgent(cfg, [ex_restarted])

            assert restarted.order_mgr.has_pending("BTC"), "agent pending book should restore from checkpoint"
            assert result.order_id in ex_restarted._pending_limits, "dry-run exchange should rehydrate restored pending limits"
            assert not ex_restarted.get_order_status("BTC", "missing-order").filled
        finally:
            checkpoint_module.checkpoint_manager = original_manager
            agent_module.checkpoint_manager = original_manager


def test_execute_order_stops_after_first_success() -> None:
    cfg = build_config()
    original_manager = checkpoint_module.checkpoint_manager
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        temp_manager = checkpoint_module.CheckpointManager(db_path=str(db_path))
        checkpoint_module.checkpoint_manager = temp_manager
        agent_module.checkpoint_manager = temp_manager

        first = StubExchange("first", should_fill=True)
        second = StubExchange("second", should_fill=True)
        agent = TradingAgent(cfg, [first, second])
        order = OrderRequest(
            coin="BTC",
            direction="LONG",
            size_usd=100.0,
            size_coin=1.0,
            price=100.0,
            stop_loss=90.0,
            take_profit=150.0,
            leverage=2,
            approved=True,
        )
        agent._execute_order("BTC", Signal("LONG"), order)
        assert first.market_buy_calls == 1, "first exchange should execute once"
        assert second.market_buy_calls == 0, "agent should stop after first successful execution"
    checkpoint_module.checkpoint_manager = original_manager
    agent_module.checkpoint_manager = original_manager


def test_checkpoint_recovery_skips_unsupported_state() -> None:
    cfg = build_config()
    original_manager = checkpoint_module.checkpoint_manager
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        temp_manager = checkpoint_module.CheckpointManager(db_path=str(db_path))
        checkpoint_module.checkpoint_manager = temp_manager
        agent_module.checkpoint_manager = temp_manager

        temp_manager.save(
            cycle_number=3,
            portfolio_usd=1000.0,
            available_usd=700.0,
            positions={
                "BTC": {
                    "direction": "LONG",
                    "entry_price": 70000.0,
                    "size_usd": 100.0,
                    "size_coin": 0.001,
                    "stop_loss": 63000.0,
                    "take_profit": 105000.0,
                    "trailing_stop_price": 65000.0,
                    "exchange": "DryRun (Paper Trading)",
                },
                "TAO": {
                    "direction": "SHORT",
                    "entry_price": 300.0,
                    "size_usd": 100.0,
                    "size_coin": 0.3,
                    "stop_loss": 330.0,
                    "take_profit": 150.0,
                    "trailing_stop_price": 336.0,
                    "exchange": "DryRun (Paper Trading)",
                },
            },
            pending_orders={
                "TAO": {
                    "direction": "SHORT",
                    "limit_price": 290.0,
                    "size_coin": 0.3,
                    "size_usd": 100.0,
                    "stop_loss": 330.0,
                    "take_profit": 150.0,
                    "signal_score": 25.0,
                    "exchange": "DryRun (Paper Trading)",
                    "exchange_order_id": "tao-order",
                    "cycles_waiting": 1,
                    "reason": "re_entry",
                }
            },
            reentry_watches={
                "TAO": {
                    "direction": "SHORT",
                    "entry_price": 300.0,
                    "tp_price": 150.0,
                    "reentry_price": 280.0,
                    "stop_price": 330.0,
                    "size_usd": 100.0,
                    "signal_score": 25.0,
                }
            },
            risk_manager=SimpleNamespace(daily_pnl_usd=0.0, daily_trades=0, last_trade_date=""),
            exchange_states={},
        )

        ex = DryRunExchange(starting_balance_usd=1000.0)
        ex.connect()
        restarted = TradingAgent(cfg, [ex])
        assert "BTC" in restarted.risk.positions
        assert "TAO" not in restarted.risk.positions, "unsupported recovered positions should be skipped"
        assert "TAO" not in restarted.order_mgr.pending_orders, "unsupported pending orders should be skipped"
        assert "TAO" not in restarted.order_mgr.reentry_watches, "unsupported re-entry watches should be skipped"
    checkpoint_module.checkpoint_manager = original_manager
    agent_module.checkpoint_manager = original_manager


def test_unsupported_symbols_fail_trade_universe_validation() -> None:
    cfg = build_config()
    cfg.trading.coins = ["BTC", "BRENT"]
    original_config = main_module.config
    main_module.config = cfg
    try:
        raised = False
        try:
            main_module.enforce_trade_universe()
        except ValueError:
            raised = True
        assert raised, "unsupported symbols should fail trade-universe validation"
    finally:
        main_module.config = original_config


def test_analysis_watchlist_keeps_non_tradable_assets_out_of_execution_universe() -> None:
    cfg = build_config()
    cfg.trading.analysis_coins = ["BTC", "HYPE", "SP500", "TAO"]
    original_manager = checkpoint_module.checkpoint_manager
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        temp_manager = checkpoint_module.CheckpointManager(db_path=str(db_path))
        checkpoint_module.checkpoint_manager = temp_manager
        agent_module.checkpoint_manager = temp_manager

        ex = DryRunExchange(starting_balance_usd=1000.0, supported_symbols=["BTC"])
        ex.connect()
        agent = TradingAgent(cfg, [ex])
        assert agent._tradable_coins == ["BTC"]
        assert agent._analysis_coins == ["BTC", "HYPE", "SP500", "TAO"]
    checkpoint_module.checkpoint_manager = original_manager
    agent_module.checkpoint_manager = original_manager


def test_supported_watchlist_assets_are_promoted_into_tradeable_universe() -> None:
    cfg = build_config()
    cfg.exchange.use_lighter = False
    cfg.exchange.use_hyperliquid = True
    cfg.trading.analysis_coins = ["BTC", "HYPE", "SP500", "TAO"]
    original_config = main_module.config
    main_module.config = cfg
    try:
        active = main_module.enforce_trade_universe()
        assert active == ["BTC", "HYPE", "SP500", "TAO"]
    finally:
        main_module.config = original_config


def test_inactive_hyperliquid_symbols_stay_out_of_tradeable_universe_when_fast_promotion_is_disabled() -> None:
    cfg = build_config()
    cfg.exchange.use_lighter = False
    cfg.exchange.use_hyperliquid = True
    cfg.trading.enforce_active_venue_markets = True
    cfg.trading.promote_analysis_before_activity = False
    cfg.trading.coins = ["BTC"]
    cfg.trading.analysis_coins = ["BTC", "AMZN", "MSFT"]
    original_config = main_module.config
    original_supported = main_module.get_hyperliquid_supported_coins
    original_is_active = main_module.hyperliquid_market_is_active
    main_module.config = cfg
    try:
        def fake_supported(*, include_spot=True, live_tradeable_only=False, active_only=False):
            return ["BTC", "AMZN", "MSFT"]

        def fake_is_active(coin: str, *, force_refresh: bool = False):
            return str(coin).upper() in {"BTC", "AMZN"}

        main_module.get_hyperliquid_supported_coins = fake_supported
        main_module.hyperliquid_market_is_active = fake_is_active
        active = main_module.enforce_trade_universe()
        assert active == ["BTC", "AMZN"]
    finally:
        main_module.config = original_config
        main_module.get_hyperliquid_supported_coins = original_supported
        main_module.hyperliquid_market_is_active = original_is_active


def test_inactive_supported_hyperliquid_symbols_are_armed_for_execution_by_default() -> None:
    cfg = build_config()
    cfg.exchange.use_lighter = False
    cfg.exchange.use_hyperliquid = True
    cfg.trading.enforce_active_venue_markets = True
    cfg.trading.promote_analysis_before_activity = True
    cfg.trading.coins = ["BTC"]
    cfg.trading.analysis_coins = ["BTC", "AMZN", "MSFT"]
    original_config = main_module.config
    original_supported = main_module.get_hyperliquid_supported_coins
    original_is_active = main_module.hyperliquid_market_is_active
    main_module.config = cfg
    try:
        def fake_supported(*, include_spot=True, live_tradeable_only=False, active_only=False):
            return ["BTC", "AMZN", "MSFT"]

        def fake_is_active(coin: str, *, force_refresh: bool = False):
            return str(coin).upper() in {"BTC", "AMZN"}

        main_module.get_hyperliquid_supported_coins = fake_supported
        main_module.hyperliquid_market_is_active = fake_is_active
        active = main_module.enforce_trade_universe()
        assert active == ["BTC", "AMZN", "MSFT"]
    finally:
        main_module.config = original_config
        main_module.get_hyperliquid_supported_coins = original_supported
        main_module.hyperliquid_market_is_active = original_is_active


def test_live_spot_opt_in_includes_active_equities_in_supported_universe() -> None:
    cfg = build_config()
    cfg.trading.dry_run = False
    cfg.exchange.use_lighter = False
    cfg.exchange.use_hyperliquid = True
    cfg.exchange.hl_spot_execution_enabled = True
    original_config = main_module.config
    original_supported = main_module.get_hyperliquid_supported_coins
    main_module.config = cfg
    try:
        def fake_supported(*, include_spot=True, live_tradeable_only=False, active_only=False):
            assert include_spot is True
            assert live_tradeable_only is False
            return ["BTC", "AMZN", "META"]

        main_module.get_hyperliquid_supported_coins = fake_supported
        supported = main_module.configured_supported_coins(dry_run_mode=False)
        assert supported == ["AMZN", "BTC", "META"]
    finally:
        main_module.config = original_config
        main_module.get_hyperliquid_supported_coins = original_supported


def test_lighter_promotes_growth_and_macro_symbols_into_tradeable_universe() -> None:
    cfg = build_config()
    cfg.exchange.use_lighter = True
    cfg.exchange.use_hyperliquid = False
    cfg.trading.analysis_coins = ["BTC", "HYPE", "TAO", "SP500", "XAU"]
    original_config = main_module.config
    main_module.config = cfg
    try:
        active = main_module.enforce_trade_universe()
        assert active == ["BTC", "HYPE", "TAO", "SP500", "XAU"]
    finally:
        main_module.config = original_config


def test_dynamic_trade_plan_is_attached_to_signal() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    tech = SimpleNamespace(
        valid=True,
        coin="BTC",
        price=100.0,
        rsi=38.0,
        rsi_score=74.0,
        macd_hist=1.5,
        macd_score=72.0,
        bb_score=66.0,
        ema_score=71.0,
        volume_score=1.8,
    )
    advanced = SimpleNamespace(
        valid=True,
        fib=SimpleNamespace(
            score=62.0,
            levels={"38.2%": 96.0, "50%": 98.0, "61.8%": 102.0, "78.6%": 106.0},
            nearest_level_name="50%",
            nearest_level_price=98.0,
            description="Fib support",
        ),
        msb=SimpleNamespace(
            score=74.0,
            msb_type="BULLISH_BOS",
            structure_trend="UPTREND",
            last_swing_high=109.0,
            last_swing_low=94.0,
            description="Bullish BOS",
        ),
        ob=SimpleNamespace(
            score=58.0,
            inside_bullish_ob=False,
            inside_bearish_ob=False,
            bullish_obs=[(97.5, 95.5)],
            bearish_obs=[(110.5, 108.0)],
            description="Order blocks",
        ),
        fvg=SimpleNamespace(
            score=56.0,
            bullish_fvgs=[(96.5, 97.3)],
            bearish_fvgs=[(106.5, 107.8)],
            inside_bullish_fvg=False,
            inside_bearish_fvg=False,
            description="FVGs",
        ),
        atr=SimpleNamespace(atr=2.0, atr_pct=2.0, volatility_label="high"),
    )
    regimes = SimpleNamespace(
        valid=True,
        dominant_regime="TREND",
        momentum_score=70.0,
        trend_score=76.0,
        mean_rev_score=48.0,
        volatility_score=62.0,
        absorption_score=49.0,
        catalyst_score=55.0,
    )
    candles = SimpleNamespace(valid=True, score=74.0, patterns=["Bullish Engulfing"], trend_3="UP")
    sentiment = {"signal_score": 55.0, "label": "Neutral", "raw_score": 50, "is_extreme": False}

    signal = strategy.generate_signal(
        tech=tech,
        advanced=advanced,
        sentiment=sentiment,
        current_position=None,
        regimes=regimes,
        news_signal=None,
        candle_patterns=candles,
        memory_adjustment=0.0,
        instrument_type="crypto",
        funding_oi_signal=None,
    )

    assert signal.action == "LONG"
    assert signal.stop_loss_price > 0
    assert signal.take_profit_price > signal.price
    assert signal.trade_plan["risk_reward_ratio"] >= cfg.trading.min_target_r_multiple
    assert signal.trade_plan["stop_basis"]
    assert signal.trade_plan["target_basis"]


def test_orderbook_support_blocks_weak_short_into_demand() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    tech = SimpleNamespace(
        valid=True,
        coin="BTC",
        price=100.0,
        rsi=67.0,
        rsi_score=28.0,
        macd_hist=-1.4,
        macd_score=26.0,
        bb_score=34.0,
        ema_score=31.0,
        volume_score=1.2,
    )
    advanced = SimpleNamespace(
        valid=True,
        fib=SimpleNamespace(score=38.0, levels={"38.2%": 96.0, "61.8%": 104.0}, nearest_level_name="61.8%", nearest_level_price=104.0, description="Fib resistance"),
        msb=SimpleNamespace(score=41.0, msb_type="NONE", structure_trend="RANGING", last_swing_high=104.0, last_swing_low=97.0, description="Range"),
        ob=SimpleNamespace(score=44.0, inside_bullish_ob=False, inside_bearish_ob=False, bullish_obs=[(98.0, 97.0)], bearish_obs=[(104.5, 103.5)], description="Order blocks"),
        fvg=SimpleNamespace(score=47.0, bullish_fvgs=[(97.4, 97.8)], bearish_fvgs=[(103.4, 103.8)], inside_bullish_fvg=False, inside_bearish_fvg=False, description="FVG"),
        atr=SimpleNamespace(atr=1.6, atr_pct=1.6, volatility_label="normal"),
    )
    regimes = SimpleNamespace(
        valid=True,
        dominant_regime="MIXED",
        momentum_score=40.0,
        trend_score=39.0,
        mean_rev_score=44.0,
        volatility_score=49.0,
        absorption_score=52.0,
        catalyst_score=50.0,
    )
    candles = SimpleNamespace(valid=True, score=36.0, patterns=["Bearish Engulfing"], trend_3="DOWN")
    sentiment = {"signal_score": 50.0, "label": "Neutral", "raw_score": 50, "is_extreme": False}
    orderbook_signal = SimpleNamespace(
        valid=True,
        score=64.0,
        imbalance_ratio=0.22,
        level_interaction="AT_SUPPORT",
        breakout_state="NONE",
        favor_longs=True,
        favor_shorts=False,
        block_longs=False,
        block_shorts=True,
        nearest_support=99.4,
        nearest_support_distance_pct=0.6,
        nearest_support_strength=0.95,
        nearest_resistance=103.8,
        nearest_resistance_distance_pct=3.8,
        nearest_resistance_strength=0.35,
        support_levels=[{"price": 99.4, "strength": 0.95, "source": "orderbook", "label": "bid_wall"}],
        resistance_levels=[{"price": 103.8, "strength": 0.35, "source": "daily", "label": "daily_swing_high"}],
        daily_breakout_level=104.0,
        daily_breakdown_level=97.0,
    )

    signal = strategy.generate_signal(
        tech=tech,
        advanced=advanced,
        sentiment=sentiment,
        current_position=None,
        regimes=regimes,
        news_signal=None,
        candle_patterns=candles,
        memory_adjustment=0.0,
        instrument_type="crypto",
        funding_oi_signal=None,
        orderbook_signal=orderbook_signal,
    )

    assert signal.action == "FLAT", "short should be blocked when price is sitting on strong support/demand"
    assert "support" in signal.flat_reason.lower() or "demand" in signal.flat_reason.lower()


def test_support_defense_long_promotes_defended_reclaim_setup() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    tech = SimpleNamespace(
        valid=True,
        coin="BTC",
        price=100.0,
        rsi=54.0,
        rsi_score=34.0,
        macd_hist=-0.5,
        macd_score=36.0,
        bb_score=40.0,
        ema_score=39.0,
        volume_score=1.15,
    )
    advanced = SimpleNamespace(
        valid=True,
        fib=SimpleNamespace(score=41.0, levels={"38.2%": 98.6, "61.8%": 101.6}, nearest_level_name="61.8%", nearest_level_price=101.6, description="Fib resistance nearby"),
        msb=SimpleNamespace(score=48.0, msb_type="NONE", structure_trend="RANGING", last_swing_high=102.4, last_swing_low=98.1, description="Range holding"),
        ob=SimpleNamespace(score=45.0, inside_bullish_ob=False, inside_bearish_ob=False, bullish_obs=[(99.1, 98.5)], bearish_obs=[(102.2, 101.4)], description="Order blocks"),
        fvg=SimpleNamespace(score=46.0, bullish_fvgs=[(98.8, 99.2)], bearish_fvgs=[(101.7, 102.0)], inside_bullish_fvg=False, inside_bearish_fvg=False, description="FVG"),
        atr=SimpleNamespace(atr=1.35, atr_pct=1.35, volatility_label="normal"),
    )
    regimes = SimpleNamespace(
        valid=True,
        dominant_regime="BREAKOUT",
        momentum_score=43.0,
        trend_score=44.0,
        mean_rev_score=49.0,
        volatility_score=54.0,
        absorption_score=48.0,
        catalyst_score=56.0,
    )
    candles = SimpleNamespace(valid=True, score=58.0, patterns=["Hammer"], trend_3="UP")
    sentiment = {"signal_score": 48.0, "label": "Neutral", "raw_score": 50, "is_extreme": False}
    orderbook_signal = SimpleNamespace(
        valid=True,
        score=73.2,
        imbalance_ratio=0.85,
        level_interaction="AT_SUPPORT",
        breakout_state="PROBING_BULLISH_BREAKOUT",
        favor_longs=True,
        favor_shorts=False,
        block_longs=False,
        block_shorts=True,
        nearest_support=99.96,
        nearest_support_distance_pct=0.04,
        nearest_support_strength=0.96,
        nearest_resistance=101.8,
        nearest_resistance_distance_pct=1.8,
        nearest_resistance_strength=0.44,
        support_levels=[{"price": 99.96, "strength": 0.96, "source": "orderbook", "label": "bid_wall"}],
        resistance_levels=[{"price": 101.8, "strength": 0.44, "source": "daily", "label": "daily_resistance"}],
        daily_breakout_level=101.6,
        daily_breakdown_level=98.1,
    )

    signal = strategy.generate_signal(
        tech=tech,
        advanced=advanced,
        sentiment=sentiment,
        current_position=None,
        regimes=regimes,
        news_signal=None,
        candle_patterns=candles,
        memory_adjustment=-6.0,
        instrument_type="crypto",
        funding_oi_signal=None,
        orderbook_signal=orderbook_signal,
    )

    assert signal.action == "LONG", "defended support + probing bullish breakout should promote a strict reclaim long"
    assert signal.trade_plan["stop_loss"] < signal.price
    assert signal.thesis["permitted"] is True
    assert any("support defense" in reason.lower() for reason in signal.thesis["reasons"])


def test_confirmed_breakout_can_override_neutral_supply_map_for_support_defense_long() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    tech = SimpleNamespace(
        valid=True,
        coin="ETH",
        price=100.0,
        rsi=54.0,
        rsi_score=44.0,
        macd_hist=0.2,
        macd_score=48.0,
        bb_score=46.0,
        ema_score=47.0,
        volume_score=1.05,
    )
    advanced = SimpleNamespace(
        valid=True,
        fib=SimpleNamespace(score=45.0, levels={"38.2%": 98.6, "61.8%": 101.1}, nearest_level_name="61.8%", nearest_level_price=101.1, description="Fib resistance nearby"),
        msb=SimpleNamespace(score=56.0, msb_type="NONE", structure_trend="UPTREND", last_swing_high=103.2, last_swing_low=98.2, description="Uptrend still intact"),
        ob=SimpleNamespace(score=48.0, inside_bullish_ob=False, inside_bearish_ob=False, bullish_obs=[(99.1, 98.7)], bearish_obs=[(101.7, 101.2)], description="Order blocks"),
        fvg=SimpleNamespace(score=46.0, bullish_fvgs=[(98.9, 99.3)], bearish_fvgs=[(101.3, 101.8)], inside_bullish_fvg=False, inside_bearish_fvg=False, description="FVG"),
        atr=SimpleNamespace(atr=1.2, atr_pct=1.2, volatility_label="normal"),
    )
    regimes = SimpleNamespace(
        valid=True,
        dominant_regime="BREAKOUT",
        momentum_score=55.0,
        trend_score=62.0,
        mean_rev_score=49.0,
        volatility_score=44.0,
        absorption_score=36.0,
        catalyst_score=54.0,
    )
    candles = SimpleNamespace(valid=True, score=50.0, patterns=["Spinning Top"], trend_3="UP")
    sentiment = {"signal_score": 48.0, "label": "Neutral", "raw_score": 50, "is_extreme": False}
    orderbook_signal = SimpleNamespace(
        valid=True,
        score=73.0,
        imbalance_ratio=0.16,
        imbalance_mean=0.10,
        level_interaction="AT_SUPPORT",
        breakout_state="CONFIRMED_BULLISH_BREAKOUT",
        favor_longs=True,
        favor_shorts=False,
        block_longs=False,
        block_shorts=True,
        nearest_support=99.7,
        nearest_support_distance_pct=0.3,
        nearest_support_strength=0.88,
        nearest_resistance=101.4,
        nearest_resistance_distance_pct=1.4,
        nearest_resistance_strength=0.52,
        support_levels=[{"price": 99.7, "strength": 0.88, "source": "orderbook", "label": "bid_wall"}],
        resistance_levels=[{"price": 101.4, "strength": 0.52, "source": "daily", "label": "daily_resistance"}],
        daily_breakout_level=100.9,
        daily_breakdown_level=97.8,
    )
    market_map_signal = SimpleNamespace(
        valid=True,
        bias="NEUTRAL",
        favor_longs=False,
        favor_shorts=True,
        block_longs=True,
        block_shorts=False,
        in_demand_zone=False,
        in_supply_zone=True,
        above_reclaim_levels=[],
        probing_above_reclaim_levels=[],
        below_breakdown_levels=[],
        nearest_support=99.4,
        nearest_support_distance_pct=0.6,
        nearest_resistance=101.2,
        nearest_resistance_distance_pct=1.2,
        summary="auto neutral map; price is sitting in mapped supply; price is pressing mapped resistance",
    )

    signal = strategy.generate_signal(
        tech=tech,
        advanced=advanced,
        sentiment=sentiment,
        current_position=None,
        regimes=regimes,
        news_signal=None,
        candle_patterns=candles,
        memory_adjustment=0.0,
        instrument_type="crypto",
        funding_oi_signal=None,
        orderbook_signal=orderbook_signal,
        market_map_signal=market_map_signal,
    )

    assert signal.action == "LONG"
    assert signal.thesis["permitted"] is True
    assert signal.expectancy["permitted"] is True
    assert any("support defense" in reason.lower() for reason in signal.thesis["reasons"])


def test_confirmed_breakout_can_promote_borderline_long() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    tech = SimpleNamespace(
        valid=True,
        coin="BTC",
        price=100.0,
        rsi=47.0,
        rsi_score=58.0,
        macd_hist=0.6,
        macd_score=59.0,
        bb_score=57.0,
        ema_score=58.0,
        volume_score=1.3,
    )
    advanced = SimpleNamespace(
        valid=True,
        fib=SimpleNamespace(score=56.0, levels={"38.2%": 97.0, "61.8%": 101.5, "78.6%": 104.5}, nearest_level_name="61.8%", nearest_level_price=101.5, description="Fib breakout"),
        msb=SimpleNamespace(score=60.0, msb_type="NONE", structure_trend="UPTREND", last_swing_high=104.8, last_swing_low=96.2, description="Uptrend"),
        ob=SimpleNamespace(score=54.0, inside_bullish_ob=False, inside_bearish_ob=False, bullish_obs=[(97.5, 96.5)], bearish_obs=[(105.5, 104.2)], description="Order blocks"),
        fvg=SimpleNamespace(score=53.0, bullish_fvgs=[(97.8, 98.3)], bearish_fvgs=[(104.6, 105.1)], inside_bullish_fvg=False, inside_bearish_fvg=False, description="FVG"),
        atr=SimpleNamespace(atr=1.4, atr_pct=1.4, volatility_label="normal"),
    )
    regimes = SimpleNamespace(
        valid=True,
        dominant_regime="TREND",
        momentum_score=59.0,
        trend_score=63.0,
        mean_rev_score=49.0,
        volatility_score=54.0,
        absorption_score=47.0,
        catalyst_score=52.0,
    )
    candles = SimpleNamespace(valid=True, score=55.0, patterns=["Bullish Engulfing"], trend_3="UP")
    sentiment = {"signal_score": 50.0, "label": "Neutral", "raw_score": 50, "is_extreme": False}
    orderbook_signal = SimpleNamespace(
        valid=True,
        score=71.0,
        imbalance_ratio=0.18,
        level_interaction="ABOVE_BREAKOUT",
        breakout_state="CONFIRMED_BULLISH_BREAKOUT",
        favor_longs=True,
        favor_shorts=False,
        block_longs=False,
        block_shorts=True,
        nearest_support=98.8,
        nearest_support_distance_pct=1.2,
        nearest_support_strength=0.7,
        nearest_resistance=106.0,
        nearest_resistance_distance_pct=6.0,
        nearest_resistance_strength=0.3,
        support_levels=[{"price": 98.8, "strength": 0.7, "source": "daily", "label": "prev_day_high_flip"}],
        resistance_levels=[{"price": 106.0, "strength": 0.3, "source": "round", "label": "round_level"}],
        daily_breakout_level=99.5,
        daily_breakdown_level=95.0,
    )

    signal = strategy.generate_signal(
        tech=tech,
        advanced=advanced,
        sentiment=sentiment,
        current_position=None,
        regimes=regimes,
        news_signal=None,
        candle_patterns=candles,
        memory_adjustment=0.0,
        instrument_type="crypto",
        funding_oi_signal=None,
        orderbook_signal=orderbook_signal,
    )

    assert signal.action == "LONG", "confirmed breakout + positive book context should promote a borderline long"
    assert signal.trade_plan["take_profit"] > signal.price


def test_probing_breakout_does_not_override_nearby_resistance() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    tech = SimpleNamespace(
        valid=True,
        coin="BTC",
        price=100.0,
        rsi=46.0,
        rsi_score=57.0,
        macd_hist=0.5,
        macd_score=58.0,
        bb_score=56.0,
        ema_score=57.0,
        volume_score=1.2,
    )
    advanced = SimpleNamespace(
        valid=True,
        fib=SimpleNamespace(score=55.0, levels={"38.2%": 98.0, "61.8%": 101.0, "78.6%": 103.0}, nearest_level_name="61.8%", nearest_level_price=101.0, description="Fib ceiling"),
        msb=SimpleNamespace(score=58.0, msb_type="NONE", structure_trend="UPTREND", last_swing_high=103.5, last_swing_low=97.2, description="Uptrend"),
        ob=SimpleNamespace(score=53.0, inside_bullish_ob=False, inside_bearish_ob=False, bullish_obs=[(98.4, 97.7)], bearish_obs=[(101.9, 101.2)], description="Order blocks"),
        fvg=SimpleNamespace(score=52.0, bullish_fvgs=[(98.6, 99.0)], bearish_fvgs=[(101.6, 102.0)], inside_bullish_fvg=False, inside_bearish_fvg=False, description="FVG"),
        atr=SimpleNamespace(atr=1.3, atr_pct=1.3, volatility_label="normal"),
    )
    regimes = SimpleNamespace(
        valid=True,
        dominant_regime="TREND",
        momentum_score=58.0,
        trend_score=61.0,
        mean_rev_score=48.0,
        volatility_score=53.0,
        absorption_score=46.0,
        catalyst_score=51.0,
    )
    candles = SimpleNamespace(valid=True, score=54.0, patterns=["Bullish Engulfing"], trend_3="UP")
    sentiment = {"signal_score": 50.0, "label": "Neutral", "raw_score": 50, "is_extreme": False}
    orderbook_signal = SimpleNamespace(
        valid=True,
        score=66.0,
        imbalance_ratio=0.12,
        level_interaction="BELOW_RESISTANCE",
        breakout_state="PROBING_BULLISH_BREAKOUT",
        favor_longs=False,
        favor_shorts=False,
        block_longs=True,
        block_shorts=True,
        nearest_support=98.9,
        nearest_support_distance_pct=1.1,
        nearest_support_strength=0.6,
        nearest_resistance=100.6,
        nearest_resistance_distance_pct=0.6,
        nearest_resistance_strength=0.92,
        support_levels=[{"price": 98.9, "strength": 0.6, "source": "daily", "label": "prev_day_high_flip"}],
        resistance_levels=[{"price": 100.6, "strength": 0.92, "source": "orderbook", "label": "ask_wall"}],
        daily_breakout_level=100.2,
        daily_breakdown_level=97.0,
    )

    signal = strategy.generate_signal(
        tech=tech,
        advanced=advanced,
        sentiment=sentiment,
        current_position=None,
        regimes=regimes,
        news_signal=None,
        candle_patterns=candles,
        memory_adjustment=0.0,
        instrument_type="crypto",
        funding_oi_signal=None,
        orderbook_signal=orderbook_signal,
    )

    assert signal.action == "FLAT", "probing above resistance should not override a nearby ceiling without daily confirmation"
    assert "resistance" in signal.flat_reason.lower()


def test_crypto_news_falls_back_to_google_when_cryptopanic_is_unavailable() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, json_payload=None, content: bytes = b""):
            self.status_code = status_code
            self._json_payload = json_payload or {}
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

        def json(self):
            return self._json_payload

    import requests

    original_get = news_module.requests.get
    original_cache = dict(news_module._cache)
    original_backoff = dict(news_module._source_backoff)
    original_calendar = news_module._calendar_event_headlines
    original_event_feed = news_module.equity_event_feeds.get_equity_event_feed
    try:
        news_module._cache.clear()
        news_module._source_backoff.clear()
        news_module._calendar_event_headlines = lambda *args, **kwargs: []
        news_module.equity_event_feeds.get_equity_event_feed = (
            lambda coin, **_kwargs: equity_event_feeds_module.EquityEventFeed(coin=str(coin).upper())
        )

        def fake_get(url, params=None, **kwargs):
            if "cryptopanic.com" in url:
                return _Resp(404)
            if "news.google.com" in url:
                return _Resp(
                    200,
                    content=(
                        b'<?xml version="1.0"?><rss><channel>'
                        b"<item><title>Bitcoin breakout as ETF demand rises</title></item>"
                        b"<item><title>BTC holds key support and rallies</title></item>"
                        b"</channel></rss>"
                    ),
                )
            raise AssertionError(f"unexpected url {url}")

        news_module.requests.get = fake_get
        signal = news_module.get_news_signal("BTC", auth_token="")
        assert signal.valid is True
        assert signal.article_count == 2
        assert signal.score > 50.0
        assert any("breakout" in title.lower() for title in signal.top_headlines)
    finally:
        news_module.requests.get = original_get
        news_module._calendar_event_headlines = original_calendar
        news_module.equity_event_feeds.get_equity_event_feed = original_event_feed
        news_module._cache.clear()
        news_module._cache.update(original_cache)
        news_module._source_backoff.clear()
        news_module._source_backoff.update(original_backoff)


def test_macro_news_filters_irrelevant_cross_ticker_headlines() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, content: bytes = b""):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

    import requests

    original_get = news_module.requests.get
    original_cache = dict(news_module._cache)
    original_backoff = dict(news_module._source_backoff)
    original_calendar = news_module._calendar_event_headlines
    original_event_feed = news_module.equity_event_feeds.get_equity_event_feed
    try:
        news_module._cache.clear()
        news_module._source_backoff.clear()
        news_module._calendar_event_headlines = lambda *args, **kwargs: []
        news_module.equity_event_feeds.get_equity_event_feed = (
            lambda coin, **_kwargs: equity_event_feeds_module.EquityEventFeed(coin=str(coin).upper())
        )

        def fake_get(url, params=None, **kwargs):
            if "feeds.finance.yahoo.com" in url:
                return _Resp(404)
            if "news.google.com" in url:
                return _Resp(
                    200,
                    content=(
                        b'<?xml version="1.0"?><rss><channel>'
                        b"<item><title>Tim Cook names successor as he steps down as Apple CEO</title></item>"
                        b"<item><title>Amazon stock rises as AWS growth accelerates</title></item>"
                        b"</channel></rss>"
                    ),
                )
            raise AssertionError(f"unexpected url {url}")

        news_module.requests.get = fake_get
        signal = news_module.get_news_signal("AMZN", auth_token="")
        assert signal.valid is True
        assert signal.article_count == 1
        assert signal.top_headlines == ["Amazon stock rises as AWS growth accelerates"]
        assert signal.score > 50.0
    finally:
        news_module.requests.get = original_get
        news_module._calendar_event_headlines = original_calendar
        news_module.equity_event_feeds.get_equity_event_feed = original_event_feed
        news_module._cache.clear()
        news_module._cache.update(original_cache)
        news_module._source_backoff.clear()
        news_module._source_backoff.update(original_backoff)


def test_macro_news_returns_neutral_when_no_asset_specific_headlines_exist() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, content: bytes = b""):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

    import requests

    original_get = news_module.requests.get
    original_cache = dict(news_module._cache)
    original_backoff = dict(news_module._source_backoff)
    original_calendar = news_module._calendar_event_headlines
    original_event_feed = news_module.equity_event_feeds.get_equity_event_feed
    try:
        news_module._cache.clear()
        news_module._source_backoff.clear()
        news_module._calendar_event_headlines = lambda *args, **kwargs: []
        news_module.equity_event_feeds.get_equity_event_feed = (
            lambda coin, **_kwargs: equity_event_feeds_module.EquityEventFeed(coin=str(coin).upper())
        )

        def fake_get(url, params=None, **kwargs):
            if "feeds.finance.yahoo.com" in url:
                return _Resp(404)
            if "news.google.com" in url:
                return _Resp(
                    200,
                    content=(
                        b'<?xml version="1.0"?><rss><channel>'
                        b"<item><title>Tim Cook names successor as he steps down as Apple CEO</title></item>"
                        b"<item><title>Meta boosts AI capex for next year</title></item>"
                        b"</channel></rss>"
                    ),
                )
            raise AssertionError(f"unexpected url {url}")

        news_module.requests.get = fake_get
        signal = news_module.get_news_signal("AMZN", auth_token="")
        assert signal.valid is True
        assert signal.article_count == 0
        assert signal.score == 50.0
        assert "asset-specific" in signal.error
    finally:
        news_module.requests.get = original_get
        news_module._calendar_event_headlines = original_calendar
        news_module.equity_event_feeds.get_equity_event_feed = original_event_feed
        news_module._cache.clear()
        news_module._cache.update(original_cache)
        news_module._source_backoff.clear()
        news_module._source_backoff.update(original_backoff)


def test_macro_news_recognizes_major_platform_customer_catalyst() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, content: bytes = b""):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

    import requests

    original_get = news_module.requests.get
    original_cache = dict(news_module._cache)
    original_backoff = dict(news_module._source_backoff)
    original_calendar = news_module._calendar_event_headlines
    original_event_feed = news_module.equity_event_feeds.get_equity_event_feed
    try:
        news_module._cache.clear()
        news_module._source_backoff.clear()
        news_module._calendar_event_headlines = lambda *args, **kwargs: []
        news_module.equity_event_feeds.get_equity_event_feed = (
            lambda coin, **_kwargs: equity_event_feeds_module.EquityEventFeed(coin=str(coin).upper())
        )

        def fake_get(url, params=None, **kwargs):
            if "feeds.finance.yahoo.com" in url:
                return _Resp(404)
            if "news.google.com" in url:
                return _Resp(
                    200,
                    content=(
                        b'<?xml version="1.0"?><rss><channel>'
                        b"<item><title>Anthropic commits $100 billion to AWS over next 10 years</title></item>"
                        b"</channel></rss>"
                    ),
                )
            raise AssertionError(f"unexpected url {url}")

        news_module.requests.get = fake_get
        signal = news_module.get_news_signal("AMZN", auth_token="")
        assert signal.valid is True
        assert signal.article_count == 1
        assert signal.top_headlines == ["Anthropic commits $100 billion to AWS over next 10 years"]
        assert signal.score > 60.0
        assert signal.catalyst_score >= 3.5
        assert "platform anchor" in signal.catalyst_summary
        assert "demand commitment" in signal.catalyst_summary
    finally:
        news_module.requests.get = original_get
        news_module._calendar_event_headlines = original_calendar
        news_module.equity_event_feeds.get_equity_event_feed = original_event_feed
        news_module._cache.clear()
        news_module._cache.update(original_cache)
        news_module._source_backoff.clear()
        news_module._source_backoff.update(original_backoff)


def test_macro_news_merges_pre_event_intc_catalyst_flow() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, content: bytes = b""):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

    import requests

    original_get = news_module.requests.get
    original_cache = dict(news_module._cache)
    original_backoff = dict(news_module._source_backoff)
    try:
        news_module._cache.clear()
        news_module._source_backoff.clear()

        def fake_get(url, params=None, **kwargs):
            if "feeds.finance.yahoo.com" in url:
                return _Resp(
                    200,
                    content=(
                        b'<?xml version="1.0"?><rss><channel>'
                        b"<item><title>Apple services revenue holds steady before earnings</title></item>"
                        b"</channel></rss>"
                    ),
                )
            if "news.google.com" in url:
                return _Resp(
                    200,
                    content=(
                        b'<?xml version="1.0"?><rss><channel>'
                        b"<item><title>Intel shares climb ahead of earnings as server CPU demand and Xeon backlog improve</title></item>"
                        b"</channel></rss>"
                    ),
                )
            raise AssertionError(f"unexpected url {url}")

        news_module.requests.get = fake_get
        signal = news_module.get_news_signal("INTC", auth_token="")
        assert signal.valid is True
        assert signal.article_count == 1
        assert signal.top_headlines == [
            "Intel shares climb ahead of earnings as server CPU demand and Xeon backlog improve"
        ]
        assert signal.score >= 60.0
        assert signal.catalyst_score >= 4.0
        assert "pre-event setup" in signal.catalyst_summary
        assert "earnings_event" in signal.event_tags
        assert "pre_event_setup" in signal.event_tags
    finally:
        news_module.requests.get = original_get
        news_module._cache.clear()
        news_module._cache.update(original_cache)
        news_module._source_backoff.clear()
        news_module._source_backoff.update(original_backoff)


def test_macro_news_recognizes_cerebras_pre_ipo_listing_catalyst() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, content: bytes = b""):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

    import requests

    original_get = news_module.requests.get
    original_cache = dict(news_module._cache)
    original_backoff = dict(news_module._source_backoff)
    original_event_feed = news_module.equity_event_feeds.get_equity_event_feed
    try:
        news_module._cache.clear()
        news_module._source_backoff.clear()
        news_module.equity_event_feeds.get_equity_event_feed = (
            lambda *args, **kwargs: equity_event_feeds_module.EquityEventFeed(coin="CBRS")
        )

        def fake_get(url, params=None, **kwargs):
            if "feeds.finance.yahoo.com" in url:
                return _Resp(200, content=b'<?xml version="1.0"?><rss><channel></channel></rss>')
            if "news.google.com" in url:
                return _Resp(
                    200,
                    content=(
                        b'<?xml version="1.0"?><rss><channel>'
                        b"<item><title>Cerebras Systems pre-IPO perpetual launch tracks wafer scale engine AI demand and capacity expansion</title></item>"
                        b"</channel></rss>"
                    ),
                )
            raise AssertionError(f"unexpected url {url}")

        news_module.requests.get = fake_get
        signal = news_module.get_news_signal("CBRS", auth_token="")
        assert signal.valid is True
        assert signal.article_count == 1
        assert signal.score >= 60.0
        assert signal.catalyst_score >= 4.0
        assert "pre-IPO listing" in signal.catalyst_summary
        assert "pre_ipo_listing" in signal.event_tags
        assert "ipo_event" in signal.event_tags
    finally:
        news_module.requests.get = original_get
        news_module.equity_event_feeds.get_equity_event_feed = original_event_feed
        news_module._cache.clear()
        news_module._cache.update(original_cache)
        news_module._source_backoff.clear()
        news_module._source_backoff.update(original_backoff)


def test_equity_event_feed_collects_ir_sec_options_and_analyst_revisions() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, json_payload=None, text: str = ""):
            self.status_code = status_code
            self._json_payload = json_payload or {}
            self.text = text

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"{self.status_code} boom")

        def json(self):
            return self._json_payload

    original_get = equity_event_feeds_module.requests.get
    original_cache = dict(equity_event_feeds_module._feed_cache)
    try:
        equity_event_feeds_module._feed_cache.clear()

        def fake_get(url, params=None, **kwargs):
            url_text = str(url)
            if "ir.aboutamazon.com" in url_text:
                return _Resp(
                    200,
                    text="Q1 2026 Amazon.com Inc. Earnings Conference Call April 29, 2026 02:30 PM PT",
                )
            if "data.sec.gov/submissions" in url_text:
                return _Resp(
                    200,
                    json_payload={
                        "filings": {
                            "recent": {
                                "form": ["8-K"],
                                "filingDate": ["2026-04-20"],
                                "accessionNumber": ["0001018724-26-000111"],
                                "primaryDocument": ["amzn-20260420.htm"],
                                "acceptanceDateTime": ["2026-04-20T16:03:00.000Z"],
                            }
                        }
                    },
                )
            if "query2.finance.yahoo.com/v7/finance/options" in url_text:
                return _Resp(
                    200,
                    json_payload={
                        "optionChain": {
                            "result": [{
                                "quote": {"regularMarketPrice": 100.0},
                                "expirationDates": [1777507200],
                                "options": [{
                                    "calls": [{"strike": 100.0, "bid": 2.0, "ask": 2.4, "expiration": 1777507200}],
                                    "puts": [{"strike": 100.0, "bid": 1.8, "ask": 2.2, "expiration": 1777507200}],
                                }],
                            }]
                        }
                    },
                )
            if "query2.finance.yahoo.com/v10/finance/quoteSummary" in url_text:
                return _Resp(
                    200,
                    json_payload={
                        "quoteSummary": {
                            "result": [{
                                "earningsTrend": {
                                    "trend": [{
                                        "period": "0q",
                                        "earningsEstimate": {"upLast30days": 4, "downLast30days": 1},
                                        "revenueEstimate": {"upLast30days": 3, "downLast30days": 0},
                                    }]
                                },
                                "recommendationTrend": {"trend": [{"strongBuy": 10, "buy": 20, "hold": 8, "sell": 1, "strongSell": 0}]},
                                "financialData": {
                                    "targetMeanPrice": {"raw": 118.0},
                                    "currentPrice": {"raw": 100.0},
                                },
                            }]
                        }
                    },
                )
            raise AssertionError(f"unexpected url {url}")

        equity_event_feeds_module.requests.get = fake_get
        feed = equity_event_feeds_module.get_equity_event_feed(
            "AMZN",
            calendar_events=[{
                "company": "Amazon",
                "label": "Q1 2026 earnings",
                "date": "2026-04-29",
                "timing": "after market close",
                "source": "Amazon Investor Relations",
            }],
            now=datetime(2026, 4, 26, tzinfo=timezone.utc),
        )
        assert feed.valid is True
        assert "official_ir_event" in feed.tags
        assert "sec_filing" in feed.tags
        assert "options_implied_move" in feed.tags
        assert "analyst_revision" in feed.tags
        assert feed.options_implied_move_pct == 4.2
        assert feed.analyst_revision_score > 0
        assert any("official IR" in headline for headline in feed.headlines)
    finally:
        equity_event_feeds_module.requests.get = original_get
        equity_event_feeds_module._feed_cache.clear()
        equity_event_feeds_module._feed_cache.update(original_cache)


def test_equity_event_feed_uses_nasdaq_fallback_when_yahoo_is_unavailable() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, json_payload=None):
            self.status_code = status_code
            self._json_payload = json_payload or {}
            self.text = ""

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"{self.status_code} boom")

        def json(self):
            return self._json_payload

    original_get = equity_event_feeds_module.requests.get
    original_cache = dict(equity_event_feeds_module._feed_cache)
    try:
        equity_event_feeds_module._feed_cache.clear()

        def fake_get(url, params=None, **kwargs):
            url_text = str(url)
            if "query2.finance.yahoo.com" in url_text:
                return _Resp(401)
            if "api.nasdaq.com/api/quote/AMZN/option-chain" in url_text:
                return _Resp(
                    200,
                    json_payload={
                        "data": {
                            "lastTrade": "LAST TRADE: $100.00 (AS OF APR 23, 2026)",
                            "table": {
                                "rows": [
                                    {"expirygroup": "April 30, 2026"},
                                    {
                                        "expirygroup": "",
                                        "expiryDate": "Apr 30",
                                        "strike": "100.00",
                                        "c_Bid": "2.00",
                                        "c_Ask": "2.40",
                                        "c_Last": "2.10",
                                        "p_Bid": "1.80",
                                        "p_Ask": "2.20",
                                        "p_Last": "2.00",
                                    },
                                ]
                            },
                        }
                    },
                )
            if "api.nasdaq.com/api/analyst/AMZN/earnings-forecast" in url_text:
                return _Resp(
                    200,
                    json_payload={
                        "data": {
                            "quarterlyForecast": {
                                "rows": [{
                                    "fiscalEnd": "Mar 2026",
                                    "up": 3,
                                    "down": 1,
                                    "noOfEstimates": 12,
                                }]
                            }
                        }
                    },
                )
            if "api.nasdaq.com/api/analyst/AMZN/targetprice" in url_text:
                return _Resp(
                    200,
                    json_payload={
                        "data": {
                            "consensusOverview": {
                                "priceTarget": 120.0,
                                "buy": 22,
                                "hold": 5,
                                "sell": 1,
                            }
                        }
                    },
                )
            raise AssertionError(f"unexpected url {url}")

        equity_event_feeds_module.requests.get = fake_get
        feed = equity_event_feeds_module.get_equity_event_feed(
            "AMZN",
            calendar_events=[],
            now=datetime(2026, 4, 26, tzinfo=timezone.utc),
        )
        assert feed.options_implied_move_pct == 4.2
        assert "Nasdaq" in feed.options_summary
        assert feed.analyst_revision_score > 0
        assert "Nasdaq EPS revisions" in feed.analyst_revision_summary
        assert "options_implied_move" in feed.tags
        assert "analyst_revision" in feed.tags
    finally:
        equity_event_feeds_module.requests.get = original_get
        equity_event_feeds_module._feed_cache.clear()
        equity_event_feeds_module._feed_cache.update(original_cache)


def test_macro_news_adds_upcoming_mag7_earnings_calendar_when_feeds_are_sparse() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, content: bytes = b""):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

    import requests

    original_get = news_module.requests.get
    original_cache = dict(news_module._cache)
    original_backoff = dict(news_module._source_backoff)
    original_now = news_module._utc_now
    try:
        news_module._cache.clear()
        news_module._source_backoff.clear()
        news_module._utc_now = lambda: datetime(2026, 4, 24, tzinfo=timezone.utc)

        def fake_get(url, params=None, **kwargs):
            if "feeds.finance.yahoo.com" in url or "news.google.com" in url:
                return _Resp(404)
            raise AssertionError(f"unexpected url {url}")

        news_module.requests.get = fake_get
        for coin in ("GOOGL", "META", "AMZN"):
            news_module._cache.clear()
            news_module._source_backoff.clear()
            signal = news_module.get_news_signal(coin, auth_token="")
            assert signal.valid is True
            assert signal.article_count >= 1
            assert signal.score >= 60.0
            assert signal.catalyst_score >= 4.0
            assert signal.top_headlines
            assert "calendar" in signal.top_headlines[0].lower()
            assert "calendar_event" in signal.catalyst_tags
            assert "earnings_event" in signal.event_tags
            assert "pre_event_setup" in signal.event_tags
    finally:
        news_module.requests.get = original_get
        news_module._utc_now = original_now
        news_module._cache.clear()
        news_module._cache.update(original_cache)
        news_module._source_backoff.clear()
        news_module._source_backoff.update(original_backoff)


def test_narrative_signal_boosts_major_catalyst_and_blocks_fading_it() -> None:
    news_signal = SimpleNamespace(
        valid=True,
        score=72.0,
        article_count=1,
        is_extreme=False,
        catalyst_score=4.5,
        catalyst_summary="platform anchor + demand commitment + capacity lock-in",
    )

    signal = narrative_module.get_narrative_signal("AMZN", news_signal=news_signal, now_ts=time.time())

    assert signal.headline_bias == "BULLISH"
    assert signal.score_adjustment >= 8.0
    assert signal.block_shorts is True
    assert "catalyst checklist aligned" in signal.summary


def test_market_data_reuses_stale_yahoo_candles_when_live_fetch_fails() -> None:
    import requests

    original_get = market_data_module.requests.get
    original_cache = dict(market_data_module._cache)
    try:
        now = time.time()
        stale_df = pd.DataFrame([
            {"timestamp": pd.Timestamp("2026-04-10T00:00:00Z"), "open": 6800.0, "high": 6810.0, "low": 6795.0, "close": 6805.0, "volume": 10.0, "trades": 0},
            {"timestamp": pd.Timestamp("2026-04-10T01:00:00Z"), "open": 6805.0, "high": 6825.0, "low": 6800.0, "close": 6820.0, "volume": 12.0, "trades": 0},
        ])
        market_data_module._cache["SP500_1h_yf"] = (now - 120, stale_df)

        def boom(*args, **kwargs):
            raise requests.HTTPError("429 too many requests")

        market_data_module.requests.get = boom
        df = market_data_module._fetch_candles_yahoo("SP500", "1h", 2)
        assert df is not None
        assert float(df["close"].iloc[-1]) == 6820.0
    finally:
        market_data_module.requests.get = original_get
        market_data_module._cache.clear()
        market_data_module._cache.update(original_cache)


def test_supported_hyperliquid_market_does_not_fallback_to_yahoo_when_venue_is_empty() -> None:
    original_post = market_data_module.requests.post
    original_get = market_data_module.requests.get
    original_supported = market_data_module.is_hyperliquid_supported
    original_resolve = market_data_module.resolve_hyperliquid_symbol
    original_cache = dict(market_data_module._cache)

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    try:
        market_data_module.is_hyperliquid_supported = lambda _coin: True
        market_data_module.resolve_hyperliquid_symbol = lambda _coin: "@289"
        market_data_module.requests.post = lambda *args, **kwargs: _Resp([])

        def boom(*args, **kwargs):
            raise AssertionError("Yahoo fallback should not be used for supported Hyperliquid markets")

        market_data_module.requests.get = boom
        df = market_data_module.fetch_candles("MSFT", "1h", 50)
        assert df is None, "supported Hyperliquid markets should return None when venue candles are empty"
    finally:
        market_data_module.requests.post = original_post
        market_data_module.requests.get = original_get
        market_data_module.is_hyperliquid_supported = original_supported
        market_data_module.resolve_hyperliquid_symbol = original_resolve
        market_data_module._cache.clear()
        market_data_module._cache.update(original_cache)


def test_stale_hyperliquid_candles_are_rejected_for_supported_market() -> None:
    original_post = market_data_module.requests.post
    original_get = market_data_module.requests.get
    original_supported = market_data_module.is_hyperliquid_supported
    original_resolve = market_data_module.resolve_hyperliquid_symbol
    original_cache = dict(market_data_module._cache)

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    stale_ts = int((time.time() - (72 * 3600)) * 1000)
    payload = [
        {
            "t": stale_ts,
            "o": "259.31",
            "h": "260.00",
            "l": "258.90",
            "c": "259.88",
            "v": "0.5",
            "n": 2,
        }
    ]

    try:
        market_data_module.is_hyperliquid_supported = lambda _coin: True
        market_data_module.resolve_hyperliquid_symbol = lambda _coin: "@268"
        market_data_module.requests.post = lambda *args, **kwargs: _Resp(payload)

        def boom(*args, **kwargs):
            raise AssertionError("Yahoo fallback should stay disabled in Hyperliquid-only mode")

        market_data_module.requests.get = boom
        df = market_data_module.fetch_candles("AAPL", "1h", 50)
        assert df is None, "stale Hyperliquid candles should be rejected instead of backfilled externally"
    finally:
        market_data_module.requests.post = original_post
        market_data_module.requests.get = original_get
        market_data_module.is_hyperliquid_supported = original_supported
        market_data_module.resolve_hyperliquid_symbol = original_resolve
        market_data_module._cache.clear()
        market_data_module._cache.update(original_cache)


def test_price_diagnostics_label_trade_xyz_and_flag_reference_spread() -> None:
    original_supported = market_data_module.is_hyperliquid_supported
    original_resolve = market_data_module.resolve_hyperliquid_symbol
    original_dex = market_data_module.get_hyperliquid_market_dex
    original_reference = market_data_module.get_reference_price_yahoo

    try:
        market_data_module.is_hyperliquid_supported = lambda _coin: True
        market_data_module.resolve_hyperliquid_symbol = lambda _coin: "@268"
        market_data_module.get_hyperliquid_market_dex = lambda _coin: "xyz"
        market_data_module.get_reference_price_yahoo = lambda _coin: 100.0

        diag = market_data_module.get_price_diagnostics("AAPL", venue_price=103.5, max_deviation_pct=2.0)
        assert diag["price_source"] == "Trade.xyz allMids"
        assert diag["price_source_label"] == "Trade.xyz @268"
        assert diag["reference_source"] == "Yahoo Finance"
        assert diag["price_status"] == "CHECK"
        assert diag["price_deviation_pct"] == 3.5
        assert "AAPL venue price" in diag["price_warning"]
    finally:
        market_data_module.is_hyperliquid_supported = original_supported
        market_data_module.resolve_hyperliquid_symbol = original_resolve
        market_data_module.get_hyperliquid_market_dex = original_dex
        market_data_module.get_reference_price_yahoo = original_reference


def test_reference_quote_does_not_replace_executable_price_cache() -> None:
    original_supported = market_data_module.is_hyperliquid_supported
    original_resolve = market_data_module.resolve_hyperliquid_symbol
    original_dex = market_data_module.get_hyperliquid_market_dex
    original_reference = market_data_module.get_reference_price_yahoo
    original_price_cache = dict(market_data_module._price_cache)
    original_reference_cache = dict(market_data_module._reference_price_cache)

    try:
        market_data_module._price_cache.clear()
        market_data_module._reference_price_cache.clear()
        market_data_module._cache_price("GOOGL", 350.0)
        market_data_module.is_hyperliquid_supported = lambda _coin: True
        market_data_module.resolve_hyperliquid_symbol = lambda _coin: "@266"
        market_data_module.get_hyperliquid_market_dex = lambda _coin: "xyz"
        market_data_module.get_reference_price_yahoo = lambda _coin: 348.0

        diag = market_data_module.get_price_diagnostics("GOOGL", venue_price=351.0, max_deviation_pct=2.0)
        assert diag["reference_price"] == 348.0
        assert market_data_module._get_cached_price("GOOGL") == 350.0
    finally:
        market_data_module.is_hyperliquid_supported = original_supported
        market_data_module.resolve_hyperliquid_symbol = original_resolve
        market_data_module.get_hyperliquid_market_dex = original_dex
        market_data_module.get_reference_price_yahoo = original_reference
        market_data_module._price_cache.clear()
        market_data_module._price_cache.update(original_price_cache)
        market_data_module._reference_price_cache.clear()
        market_data_module._reference_price_cache.update(original_reference_cache)


def test_data_reliability_warns_without_blocking_wide_reference_price_spread() -> None:
    cfg = build_config()
    cfg.trading.data_reliability_max_reference_deviation_pct = 2.0
    snap = {
        "execution_mode": "tradable",
        "instrument_type": "equity",
        "action": "LONG",
        "using_closed_candles": True,
        "analysis_price": 100.0,
        "live_price": 100.2,
        "price_deviation_pct": 3.25,
        "market_map_available": True,
        "orderbook_valid": True,
        "orderbook_feed_age_seconds": 1.0,
        "orderbook_feed_snapshot_count": cfg.trading.data_reliability_min_orderbook_snapshots,
    }

    reliability = data_reliability_module.assess_reliability(cfg.trading, snap)
    assert reliability["permitted"] is True
    assert reliability["blockers"] == []
    assert "venue price is +3.25% away from the reference quote" in reliability["issues"]
    assert reliability["reference_deviation_pct"] == 3.25


def test_thesis_gate_blocks_high_score_range_compression_setup() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    tech = SimpleNamespace(
        valid=True,
        coin="BTC",
        price=100.0,
        rsi=31.0,
        rsi_score=82.0,
        macd_hist=1.8,
        macd_score=78.0,
        bb_score=74.0,
        ema_score=72.0,
        volume_score=1.5,
    )
    advanced = SimpleNamespace(
        valid=True,
        fib=SimpleNamespace(score=68.0, levels={"38.2%": 98.0, "61.8%": 102.0}, nearest_level_name="61.8%", nearest_level_price=102.0, description="Fib levels"),
        msb=SimpleNamespace(score=52.0, msb_type="NONE", structure_trend="RANGING", last_swing_high=102.2, last_swing_low=97.8, description="Ranging structure"),
        ob=SimpleNamespace(score=64.0, inside_bullish_ob=False, inside_bearish_ob=False, bullish_obs=[(98.6, 97.9)], bearish_obs=[(102.4, 101.8)], description="Order blocks"),
        fvg=SimpleNamespace(score=63.0, bullish_fvgs=[(98.4, 98.9)], bearish_fvgs=[(101.6, 102.0)], inside_bullish_fvg=False, inside_bearish_fvg=False, description="FVG"),
        atr=SimpleNamespace(atr=1.2, atr_pct=1.2, volatility_label="normal"),
    )
    regimes = SimpleNamespace(
        valid=True,
        dominant_regime="TREND",
        momentum_score=70.0,
        trend_score=76.0,
        mean_rev_score=48.0,
        volatility_score=58.0,
        absorption_score=44.0,
        catalyst_score=55.0,
    )
    candles = SimpleNamespace(valid=True, score=51.0, patterns=["Doji"], trend_3="FLAT")
    sentiment = {"signal_score": 58.0, "label": "Neutral", "raw_score": 50, "is_extreme": False}
    orderbook_signal = SimpleNamespace(
        valid=True,
        score=62.0,
        imbalance_ratio=0.10,
        level_interaction="RANGE_COMPRESSION",
        breakout_state="NONE",
        favor_longs=True,
        favor_shorts=False,
        block_longs=False,
        block_shorts=True,
        nearest_support=99.1,
        nearest_support_distance_pct=0.9,
        nearest_support_strength=0.8,
        nearest_resistance=100.9,
        nearest_resistance_distance_pct=0.9,
        nearest_resistance_strength=0.82,
        support_levels=[{"price": 99.1, "strength": 0.8, "source": "orderbook", "label": "bid_wall"}],
        resistance_levels=[{"price": 100.9, "strength": 0.82, "source": "orderbook", "label": "ask_wall"}],
        daily_breakout_level=101.6,
        daily_breakdown_level=98.3,
    )

    signal = strategy.generate_signal(
        tech=tech,
        advanced=advanced,
        sentiment=sentiment,
        current_position=None,
        regimes=regimes,
        news_signal=None,
        candle_patterns=candles,
        memory_adjustment=0.0,
        instrument_type="crypto",
        funding_oi_signal=SimpleNamespace(
            valid=True,
            composite_score=58.0,
            funding_label="neutral",
            oi_change_pct=1.2,
            cvd_divergence="NONE",
        ),
        orderbook_signal=orderbook_signal,
    )

    assert signal.action == "FLAT", "range compression should block even a high-scoring directional setup"
    assert signal.thesis["state"] == "NO_TRADE"
    assert "range compression" in signal.flat_reason.lower() or "ranging" in signal.flat_reason.lower()


def test_expectancy_gate_rejects_thin_edge_setup() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    expectancy = strategy._derive_expectancy_profile(
        action="LONG",
        score=65.0,
        thesis={
            "permitted": True,
            "alignment_points": 4.0,
            "conflict_points": 0.5,
            "conviction_score": 60.0,
        },
        trade_plan={"risk_reward_ratio": 0.90},
        regimes=SimpleNamespace(valid=True, dominant_regime="MIXED"),
        orderbook_signal=SimpleNamespace(valid=True, level_interaction="RANGE_COMPRESSION", breakout_state="NONE", score=52.0),
        market_map_signal=None,
        news_signal=None,
        funding_oi_signal=None,
        narrative_signal=None,
        current_position=None,
    )

    assert expectancy["permitted"] is False
    assert expectancy["blockers"], "thin-edge setup should explain why expectancy failed"


def test_strategy_allows_pre_event_equity_starter_below_trigger() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    news_signal = SimpleNamespace(
        valid=True,
        score=66.0,
        article_count=1,
        velocity="LOW",
        catalyst_score=4.6,
        catalyst_summary="platform anchor + demand commitment + earnings event + pre-event setup",
        catalyst_tags=["platform_anchor", "demand_commitment", "earnings_event", "pre_event_setup"],
        event_tags=["earnings_event", "pre_event_setup"],
    )
    narrative_signal = SimpleNamespace(headline_bias="NEUTRAL", block_longs=False, block_shorts=False)
    orderbook_signal = SimpleNamespace(
        valid=True,
        score=49.0,
        block_longs=True,
        block_shorts=False,
        breakout_state="NONE",
        intracycle_breakout_state="NONE",
    )
    market_map_signal = SimpleNamespace(
        valid=True,
        bias="NEUTRAL",
        block_longs=True,
        block_shorts=False,
        live_above_reclaim_levels=[],
        live_below_breakdown_levels=[],
    )

    probe = strategy._conviction_probe_candidate(
        instrument_type="equity",
        action="FLAT",
        raw_score=53.0,
        news_signal=news_signal,
        narrative_signal=narrative_signal,
        orderbook_signal=orderbook_signal,
        market_map_signal=market_map_signal,
    )
    assert probe["active"] is True
    assert probe["candidate_action"] == "LONG"

    entry = strategy._build_conviction_entry(
        coin="INTC",
        instrument_type="equity",
        action="LONG",
        score=53.0,
        thesis={"alignment_points": 1.0, "conflict_points": 2.5},
        expectancy={"probability": 0.52, "score": 47.0, "uncertainty": 0.57},
        news_signal=news_signal,
        narrative_signal=narrative_signal,
        orderbook_signal=orderbook_signal,
        market_map_signal=market_map_signal,
    )
    assert entry["active"] is True
    assert entry["style"] == "EVENT_STARTER"
    assert entry["event_conviction"] is True
    assert 0.20 <= entry["size_multiplier"] <= 0.58


def test_strategy_allows_mag7_earnings_calendar_starters_when_conviction_is_shaky() -> None:
    class _Resp:
        def __init__(self, status_code: int, *, content: bytes = b""):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code} boom")

    import requests

    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)
    narrative_signal = SimpleNamespace(headline_bias="NEUTRAL", block_longs=False, block_shorts=False)
    orderbook_signal = SimpleNamespace(
        valid=True,
        score=49.0,
        block_longs=True,
        block_shorts=False,
        breakout_state="NONE",
        intracycle_breakout_state="NONE",
    )
    market_map_signal = SimpleNamespace(
        valid=True,
        bias="NEUTRAL",
        block_longs=True,
        block_shorts=False,
        live_above_reclaim_levels=[],
        live_below_breakdown_levels=[],
    )

    original_get = news_module.requests.get
    original_cache = dict(news_module._cache)
    original_backoff = dict(news_module._source_backoff)
    original_now = news_module._utc_now
    try:
        news_module._cache.clear()
        news_module._source_backoff.clear()
        news_module._utc_now = lambda: datetime(2026, 4, 24, tzinfo=timezone.utc)

        def fake_get(url, params=None, **kwargs):
            if "feeds.finance.yahoo.com" in url or "news.google.com" in url:
                return _Resp(404)
            raise AssertionError(f"unexpected url {url}")

        news_module.requests.get = fake_get
        for coin in ("GOOGL", "META", "AMZN"):
            news_module._cache.clear()
            news_module._source_backoff.clear()
            news_signal = news_module.get_news_signal(coin, auth_token="")
            shaky_pre_event_score = 46.5
            probe = strategy._conviction_probe_candidate(
                instrument_type="equity",
                action="FLAT",
                raw_score=shaky_pre_event_score,
                news_signal=news_signal,
                narrative_signal=narrative_signal,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
            assert probe["active"] is True
            assert probe["candidate_action"] == "LONG"

            entry = strategy._build_conviction_entry(
                coin=coin,
                instrument_type="equity",
                action="LONG",
                score=shaky_pre_event_score,
                thesis={"alignment_points": 1.0, "conflict_points": 3.45},
                expectancy={"probability": 0.52, "score": 47.0, "uncertainty": 0.59},
                news_signal=news_signal,
                narrative_signal=narrative_signal,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
            assert entry["active"] is True
            assert entry["style"] == "EVENT_STARTER"
            assert entry["event_conviction"] is True
            assert 0.18 <= entry["size_multiplier"] <= cfg.trading.conviction_entry_event_max_size_multiplier
    finally:
        news_module.requests.get = original_get
        news_module._utc_now = original_now
        news_module._cache.clear()
        news_module._cache.update(original_cache)
        news_module._source_backoff.clear()
        news_module._source_backoff.update(original_backoff)


def test_execution_plan_prefers_limit_entry_on_defended_support() -> None:
    cfg = build_config()
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)

    plan = strategy._build_execution_plan(
        action="LONG",
        entry_price=100.0,
        trade_plan={"risk_reward_ratio": 2.1},
        expectancy={"probability": 0.63, "score": 68.0},
        orderbook_signal=SimpleNamespace(
            valid=True,
            breakout_state="NONE",
            nearest_support=99.8,
            nearest_support_distance_pct=0.20,
            best_bid=99.92,
            best_ask=100.02,
        ),
    )

    assert plan["mode"] == "limit"
    assert plan["limit_price"] > 0
    assert "support" in plan["reason"].lower()


def test_agent_uses_completed_candles_for_conviction_but_live_price_for_execution() -> None:
    cfg = build_config()
    cfg.trading.use_orderbook_levels = False

    original_fetch = agent_module.fetch_candles
    original_compute_signals = agent_module.compute_signals
    original_compute_advanced = agent_module.compute_advanced_signals
    original_compute_regimes = agent_module.compute_regimes
    original_compute_candles = agent_module.compute_candlestick_patterns
    original_get_funding = agent_module.get_funding_oi_cvd

    captured = {}
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-04-05", periods=5, freq="1h", tz="UTC"),
        "open": [100.0, 101.0, 102.0, 103.0, 108.0],
        "high": [101.0, 102.0, 103.0, 104.0, 110.0],
        "low": [99.0, 100.0, 101.0, 102.0, 107.0],
        "close": [100.0, 101.0, 102.0, 103.0, 109.0],
        "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
        "trades": [1, 1, 1, 1, 1],
    })

    def fake_fetch_candles(*args, **kwargs):
        return df.copy()

    def fake_compute_signals(frame, coin, icfg, trading_cfg):
        captured["signal_rows"] = len(frame)
        captured["signal_last_close"] = float(frame["close"].iloc[-1])
        return SimpleNamespace(
            valid=True,
            coin=coin,
            price=float(frame["close"].iloc[-1]),
            rsi=50.0,
            rsi_score=50.0,
            macd_hist=0.0,
            macd_score=50.0,
            bb_score=50.0,
            ema_score=50.0,
            volume_score=1.0,
        )

    def fake_compute_advanced(frame, coin):
        captured["advanced_rows"] = len(frame)
        return SimpleNamespace(
            valid=True,
            fib=SimpleNamespace(score=50.0, levels={}, nearest_level_name="", nearest_level_price=0.0, description="Fib"),
            msb=SimpleNamespace(score=50.0, msb_type="NONE", structure_trend="RANGING", last_swing_high=0.0, last_swing_low=0.0, description="Range"),
            ob=SimpleNamespace(score=50.0, inside_bullish_ob=False, inside_bearish_ob=False, bullish_obs=[], bearish_obs=[], description="Order blocks"),
            fvg=SimpleNamespace(score=50.0, inside_bullish_fvg=False, inside_bearish_fvg=False, bullish_fvgs=[], bearish_fvgs=[], description="FVG"),
            atr=SimpleNamespace(atr=1.0, atr_pct=1.0, volatility_label="normal"),
        )

    def fake_compute_regimes(frame, coin):
        return SimpleNamespace(
            valid=True,
            dominant_regime="MIXED",
            momentum_score=50.0,
            trend_score=50.0,
            mean_rev_score=50.0,
            volatility_score=50.0,
            absorption_score=50.0,
            catalyst_score=50.0,
        )

    def fake_compute_candles(frame, coin):
        return SimpleNamespace(valid=True, score=50.0, patterns=[], trend_3="FLAT")

    def fake_get_funding(*args, **kwargs):
        return None

    try:
        agent_module.fetch_candles = fake_fetch_candles
        agent_module.compute_signals = fake_compute_signals
        agent_module.compute_advanced_signals = fake_compute_advanced
        agent_module.compute_regimes = fake_compute_regimes
        agent_module.compute_candlestick_patterns = fake_compute_candles
        agent_module.get_funding_oi_cvd = fake_get_funding

        agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=1000.0)])

        def fake_generate_signal(*args, **kwargs):
            tech = args[0]
            captured["strategy_price"] = tech.price
            captured["strategy_closed_price"] = getattr(tech, "closed_price", 0.0)
            return SimpleNamespace(
                action="FLAT",
                score=50.0,
                confidence="LOW",
                price=tech.price,
                reason="No trade",
                flat_reason="No trade",
                stop_loss_price=0.0,
                take_profit_price=0.0,
                trade_plan={},
                thesis={
                    "candidate_action": "FLAT",
                    "state": "NO_TRADE",
                    "permitted": False,
                    "quality": "LOW",
                    "alignment_points": 0.0,
                    "conflict_points": 0.0,
                    "conviction_score": 25.0,
                    "summary": "No trade",
                    "reasons": [],
                    "blockers": [],
                },
            )

        agent.strategy.generate_signal = fake_generate_signal
        agent._analyse_coin(
            "BTC",
            {"signal_score": 50.0, "label": "Neutral", "raw_score": 50, "is_extreme": False},
            portfolio_usd=1000.0,
        )
    finally:
        agent_module.fetch_candles = original_fetch
        agent_module.compute_signals = original_compute_signals
        agent_module.compute_advanced_signals = original_compute_advanced
        agent_module.compute_regimes = original_compute_regimes
        agent_module.compute_candlestick_patterns = original_compute_candles
        agent_module.get_funding_oi_cvd = original_get_funding

    assert captured["signal_rows"] == 4, "conviction indicators should run on completed candles only"
    assert captured["advanced_rows"] == 4, "advanced structure should also use completed candles"
    assert captured["signal_last_close"] == 103.0
    assert captured["strategy_closed_price"] == 103.0
    assert captured["strategy_price"] == 109.0, "execution context should still carry the live price"


def test_scale_in_does_not_mutate_before_fill() -> None:
    cfg = build_config()
    cfg.trading.max_trade_usd = 300.0
    cfg.trading.min_trade_usd = 25.0
    risk = __import__("risk.risk_manager", fromlist=["RiskManager"]).RiskManager(cfg.trading)
    risk.restore_position(
        OpenPosition(
            coin="BTC",
            direction="LONG",
            entry_price=100.0,
            size_usd=100.0,
            size_coin=1.0,
            stop_loss=90.0,
            take_profit=150.0,
        )
    )

    order = risk.compute_order(
        coin="BTC",
        direction="LONG",
        signal_score=92.0,
        current_price=110.0,
        stop_loss_price=99.0,
        take_profit_price=165.0,
        portfolio_usd=5000.0,
    )
    pos = risk.positions["BTC"]
    assert order.is_scale_in, "expected a scale-in order"
    assert pos.size_usd == 100.0, "compute_order must not mutate position sizing"

    order.price = 110.0
    risk.record_scale_in_fill(order, exchange="DryRun (Paper Trading)")
    pos = risk.positions["BTC"]
    assert pos.size_usd > 100.0, "scale-in fill should update tracked position"
    assert pos.entry_price > 100.0, "weighted-average entry should move toward fill price"


def test_order_sizing_scales_with_conviction_and_tempers_euphoria() -> None:
    cfg = build_config()
    cfg.trading.max_trade_usd = 2000.0
    cfg.trading.max_position_pct = 0.20
    cfg.trading.min_trade_usd = 10.0
    risk = __import__("risk.risk_manager", fromlist=["RiskManager"]).RiskManager(cfg.trading)

    low_conviction = risk.compute_order(
        coin="BTC",
        direction="LONG",
        signal_score=66.0,
        current_price=100.0,
        stop_loss_price=95.0,
        take_profit_price=112.0,
        portfolio_usd=10_000.0,
        rl_win_rate=50.0,
        rl_pattern_boost=0.0,
    )
    high_conviction = risk.compute_order(
        coin="BTC",
        direction="LONG",
        signal_score=92.0,
        current_price=100.0,
        stop_loss_price=94.0,
        take_profit_price=118.0,
        portfolio_usd=10_000.0,
        rl_win_rate=50.0,
        rl_pattern_boost=0.0,
    )
    supported_extreme = risk.compute_order(
        coin="BTC",
        direction="LONG",
        signal_score=96.0,
        current_price=100.0,
        stop_loss_price=94.0,
        take_profit_price=121.0,
        portfolio_usd=10_000.0,
        rl_win_rate=74.0,
        rl_pattern_boost=0.10,
    )
    cautious_extreme = risk.compute_order(
        coin="BTC",
        direction="LONG",
        signal_score=96.0,
        current_price=100.0,
        stop_loss_price=94.0,
        take_profit_price=121.0,
        portfolio_usd=10_000.0,
        rl_win_rate=34.0,
        rl_pattern_boost=-0.05,
    )

    assert low_conviction.approved and high_conviction.approved
    assert high_conviction.size_usd > low_conviction.size_usd, "higher conviction should allocate more capital"
    assert supported_extreme.size_usd > cautious_extreme.size_usd, "weak RL history should dampen euphoric sizing"


def test_immediate_limit_scale_in_updates_open_trade_record() -> None:
    cfg = build_config()
    cfg.trading.max_trade_usd = 300.0
    cfg.trading.min_trade_usd = 25.0
    original_manager = checkpoint_module.checkpoint_manager
    original_agent_manager = agent_module.checkpoint_manager
    original_update_open = trade_logger_module.update_open
    original_log_open = trade_logger_module.log_open

    class ImmediateLimitExchange(StubExchange):
        def __init__(self):
            super().__init__("limit-immediate", should_fill=True)

        def limit_buy(self, coin: str, size_coin: float, limit_price: float, maker_only: bool = False) -> OrderResult:
            return OrderResult(success=True, filled_price=limit_price, filled_size=size_coin)

        def limit_sell(self, coin: str, size_coin: float, limit_price: float, maker_only: bool = False) -> OrderResult:
            return OrderResult(success=True, filled_price=limit_price, filled_size=size_coin)

        def supports_limit_orders(self) -> bool:
            return True

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        temp_manager = checkpoint_module.CheckpointManager(db_path=str(db_path))
        checkpoint_module.checkpoint_manager = temp_manager
        agent_module.checkpoint_manager = temp_manager
        try:
            agent = TradingAgent(cfg, [ImmediateLimitExchange()])
            agent.risk.restore_position(
                OpenPosition(
                    coin="BTC",
                    direction="LONG",
                    entry_price=100.0,
                    size_usd=100.0,
                    size_coin=1.0,
                    stop_loss=90.0,
                    take_profit=150.0,
                )
            )

            called = {}
            trade_logger_module.update_open = lambda coin, entry_price, size_usd, stop_loss, take_profit: called.update({
                "coin": coin,
                "entry_price": entry_price,
                "size_usd": size_usd,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            })
            trade_logger_module.log_open = lambda **kwargs: called.setdefault("log_open_called", True)

            result = agent._place_limit_order(
                "BTC",
                "LONG",
                limit_price=110.0,
                size_usd=50.0,
                sl=98.0,
                tp=140.0,
                score=72.0,
                reason="initial_limit",
            )

            assert result["success"] is True and result["filled"] is True and result["pending"] is False
            assert called.get("coin") == "BTC", "scale-in fills should update the current open trade record"
            assert float(called.get("size_usd", 0.0)) > 100.0
            assert "log_open_called" not in called, "scale-in fills should not create a brand-new open trade row"
        finally:
            checkpoint_module.checkpoint_manager = original_manager
            agent_module.checkpoint_manager = original_agent_manager
            trade_logger_module.update_open = original_update_open
            trade_logger_module.log_open = original_log_open


def test_pending_limit_scale_in_updates_open_trade_record() -> None:
    cfg = build_config()
    cfg.trading.max_trade_usd = 300.0
    cfg.trading.min_trade_usd = 25.0
    original_manager = checkpoint_module.checkpoint_manager
    original_agent_manager = agent_module.checkpoint_manager
    original_update_open = trade_logger_module.update_open

    class PendingLimitExchange(StubExchange):
        def __init__(self):
            super().__init__("limit-pending", should_fill=True)

        def supports_limit_orders(self) -> bool:
            return True

        def get_order_status(self, coin: str, order_id: str) -> LimitOrderStatus:
            return LimitOrderStatus(order_id=order_id, coin=coin, filled=True, filled_price=112.0, filled_size=0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "checkpoints.db"
        temp_manager = checkpoint_module.CheckpointManager(db_path=str(db_path))
        checkpoint_module.checkpoint_manager = temp_manager
        agent_module.checkpoint_manager = temp_manager
        try:
            agent = TradingAgent(cfg, [PendingLimitExchange()])
            agent.risk.restore_position(
                OpenPosition(
                    coin="BTC",
                    direction="LONG",
                    entry_price=100.0,
                    size_usd=100.0,
                    size_coin=1.0,
                    stop_loss=90.0,
                    take_profit=150.0,
                )
            )
            agent.order_mgr.register_limit_order(
                PendingOrder(
                    coin="BTC",
                    direction="LONG",
                    limit_price=112.0,
                    size_coin=0.5,
                    size_usd=56.0,
                    stop_loss=99.0,
                    take_profit=145.0,
                    signal_score=74.0,
                    exchange="limit-pending",
                    exchange_order_id="btc-scale-order",
                    reason="initial_limit",
                )
            )

            called = {}
            trade_logger_module.update_open = lambda coin, entry_price, size_usd, stop_loss, take_profit: called.update({
                "coin": coin,
                "entry_price": entry_price,
                "size_usd": size_usd,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            })
            agent._verify_position_on_exchange = lambda *_args, **_kwargs: True

            agent._poll_pending_limits({"BTC": 112.0})

            assert called.get("coin") == "BTC", "pending scale-in fills should refresh the open trade record"
            assert float(called.get("size_usd", 0.0)) > 100.0
            assert not agent.order_mgr.has_pending("BTC"), "filled pending scale-ins should clear from the pending book"
        finally:
            checkpoint_module.checkpoint_manager = original_manager
            agent_module.checkpoint_manager = original_agent_manager
            trade_logger_module.update_open = original_update_open


def test_narrative_gate_blocks_event_risk_without_exceptional_expectancy() -> None:
    cfg = build_config()
    cfg.trading.use_narrative_gate = True
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=1000.0)])
    signal = SimpleNamespace(
        action="LONG",
        score=66.0,
        expectancy={"score": 66.0, "probability": 0.57},
    )
    narrative_signal = SimpleNamespace(
        valid=True,
        block_longs=False,
        block_shorts=False,
        event_risk_active=True,
        summary="CPI is within the narrative risk window",
    )

    assert agent._check_narrative_gate("BTC", signal, narrative_signal) is False


def test_backtest_summary_includes_baselines() -> None:
    cfg = build_config()
    original_config = backtest_module.config
    backtest_module.config = cfg
    try:
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=120, freq="1h", tz="UTC"),
            "open": [100.0 + i * 0.2 for i in range(120)],
            "high": [100.5 + i * 0.2 for i in range(120)],
            "low": [99.5 + i * 0.2 for i in range(120)],
            "close": [100.2 + i * 0.2 for i in range(120)],
            "volume": [10.0] * 120,
        })
        bt = backtest_module.Backtester("BTC", df, starting_balance=1000.0, trade_size_usd=50.0)
        summary = bt._summary()
        assert "baselines" in summary
        assert "buy_hold_return" in summary["baselines"]
    finally:
        backtest_module.config = original_config


def test_failed_close_keeps_position_open() -> None:
    cfg = build_config()
    ex = FailingCloseExchange("failing", should_fill=True)
    agent = TradingAgent(cfg, [ex])
    agent.risk.restore_position(
        OpenPosition(
            coin="BTC",
            direction="LONG",
            entry_price=100.0,
            size_usd=100.0,
            size_coin=1.0,
            stop_loss=90.0,
            take_profit=150.0,
            exchange=ex.name,
        )
    )
    agent._close_position("BTC", "manual_test", 95.0)
    assert "BTC" in agent.risk.positions, "position must remain open if close verification fails"


def test_preflight_reports_missing_live_bootstrap() -> None:
    cfg = build_config()
    cfg.trading.dry_run = True
    cfg.exchange.use_lighter = True
    cfg.exchange.use_hyperliquid = False
    cfg.exchange.lighter_l1_private_key = ""
    cfg.exchange.lighter_api_private_key = ""
    cfg.exchange.lighter_account_index = ""

    original_config = main_module.config
    original_fetch = main_module.fetch_candles
    original_price = main_module.get_current_price
    main_module.config = cfg
    main_module.fetch_candles = lambda *a, **k: __import__("pandas").DataFrame([
        {"timestamp": "2026-01-01", "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
    ])
    main_module.get_current_price = lambda *a, **k: 100.0
    try:
        rc = main_module.run_preflight(SimpleNamespace(live=True))
        assert rc == 1, "live preflight should fail when Lighter bootstrap credentials are missing"
    finally:
        main_module.config = original_config
        main_module.fetch_candles = original_fetch
        main_module.get_current_price = original_price


def test_live_config_validation_requires_notifications() -> None:
    cfg = build_config()
    cfg.trading.dry_run = False
    cfg.trading.require_notifications_for_live = True
    cfg.notifications.telegram_bot_token = ""
    cfg.notifications.telegram_chat_id = ""
    raised = False
    try:
        cfg.validate()
    except ValueError as exc:
        raised = "TELEGRAM_BOT_TOKEN" in str(exc)
    assert raised, "live validation should require Telegram notifications when the guard is enabled"


def test_live_promotion_gate_blocks_weak_metrics() -> None:
    cfg = build_config()
    cfg.trading.live_promotion_gate_enabled = True
    cfg.trading.challenger_model_enabled = False
    cfg.trading.live_promotion_min_closed_trades = 4
    cfg.trading.live_promotion_lookback_closed_trades = 4
    cfg.trading.live_promotion_min_win_rate = 0.75
    cfg.trading.live_promotion_min_avg_pnl_pct = 0.20
    cfg.trading.live_promotion_min_profit_factor = 1.50
    cfg.trading.live_promotion_min_precision_samples = 4
    cfg.trading.live_promotion_min_precision_win_rate = 0.75

    original_load = promotion_gate_module.trade_dataset.load_closed_trades
    original_report = promotion_gate_module._ensure_precision_report
    try:
        promotion_gate_module.trade_dataset.load_closed_trades = lambda limit=None: [
            {"outcome": "WIN", "pnl_pct": 0.10, "pnl_usd": 10.0},
            {"outcome": "LOSS", "pnl_pct": -0.80, "pnl_usd": -80.0},
            {"outcome": "WIN", "pnl_pct": 0.15, "pnl_usd": 15.0},
            {"outcome": "LOSS", "pnl_pct": -0.30, "pnl_usd": -30.0},
        ]
        promotion_gate_module._ensure_precision_report = lambda *_args, **_kwargs: ({
            "labeled_episodes": 4,
            "overall_win_rate": 0.50,
            "best_rules": [{"win_rate": 0.50, "samples": 4}],
        }, Path("/tmp/precision_lab_report.json"))
        gate = promotion_gate_module.evaluate_live_promotion(cfg, Path("/tmp"))
        assert gate["passed"] is False
        assert gate["blockers"], "weak metrics should block live promotion"
    finally:
        promotion_gate_module.trade_dataset.load_closed_trades = original_load
        promotion_gate_module._ensure_precision_report = original_report


def test_live_promotion_gate_passes_strong_metrics() -> None:
    cfg = build_config()
    cfg.trading.live_promotion_gate_enabled = True
    cfg.trading.challenger_model_enabled = False
    cfg.trading.live_promotion_min_closed_trades = 4
    cfg.trading.live_promotion_lookback_closed_trades = 4
    cfg.trading.live_promotion_min_win_rate = 0.50
    cfg.trading.live_promotion_min_avg_pnl_pct = 0.05
    cfg.trading.live_promotion_min_profit_factor = 1.10
    cfg.trading.live_promotion_min_precision_samples = 4
    cfg.trading.live_promotion_min_precision_win_rate = 0.60

    original_load = promotion_gate_module.trade_dataset.load_closed_trades
    original_report = promotion_gate_module._ensure_precision_report
    try:
        promotion_gate_module.trade_dataset.load_closed_trades = lambda limit=None: [
            {"outcome": "WIN", "pnl_pct": 0.60, "pnl_usd": 60.0},
            {"outcome": "WIN", "pnl_pct": 0.40, "pnl_usd": 40.0},
            {"outcome": "LOSS", "pnl_pct": -0.20, "pnl_usd": -20.0},
            {"outcome": "WIN", "pnl_pct": 0.30, "pnl_usd": 30.0},
        ]
        promotion_gate_module._ensure_precision_report = lambda *_args, **_kwargs: ({
            "labeled_episodes": 6,
            "overall_win_rate": 0.67,
            "best_rules": [{"win_rate": 0.75, "samples": 4}],
        }, Path("/tmp/precision_lab_report.json"))
        gate = promotion_gate_module.evaluate_live_promotion(cfg, Path("/tmp"))
        assert gate["passed"] is True
        assert gate["trade_metrics"]["win_rate"] >= 0.50
    finally:
        promotion_gate_module.trade_dataset.load_closed_trades = original_load
        promotion_gate_module._ensure_precision_report = original_report


def test_trade_dataset_backfills_from_csv_history() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        csv_path = data_dir / "trades_log.csv"
        csv_path.write_text(
            "\n".join([
                ",".join(trade_logger_module.HEADERS),
                "1,BTC,LONG,2026-04-01 02:03,2026-04-01 02:09,6.0,67788.0,67888.0,225.0,2,3.32,1.47,66000,69000,take_profit,71.0,WIN",
                "2,ETH,SHORT,2026-04-01 04:00,2026-04-01 04:30,30.0,2000.0,2020.0,150.0,2,-1.50,-1.00,2100,1900,conviction_lost,33.0,LOSS",
            ]),
            encoding="utf-8",
        )

        rows = trade_dataset_module.load_closed_trades(data_dir=data_dir, backfill_from_csv=True)
        assert len(rows) == 2
        assert {row["coin"] for row in rows} == {"BTC", "ETH"}
        assert (data_dir / "trade_dataset.jsonl").exists(), "backfill should create the structured dataset"


def test_live_promotion_gate_uses_csv_backfill_history() -> None:
    cfg = build_config()
    cfg.trading.live_promotion_gate_enabled = True
    cfg.trading.challenger_model_enabled = False
    cfg.trading.live_promotion_min_closed_trades = 2
    cfg.trading.live_promotion_lookback_closed_trades = 2
    cfg.trading.live_promotion_min_win_rate = 0.50
    cfg.trading.live_promotion_min_avg_pnl_pct = 0.05
    cfg.trading.live_promotion_min_profit_factor = 1.05
    cfg.trading.live_promotion_min_precision_samples = 2
    cfg.trading.live_promotion_min_precision_win_rate = 0.50

    original_resolve = promotion_gate_module.trade_dataset.resolve_richest_history_data_dir
    original_report = promotion_gate_module._ensure_precision_report
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "trades_log.csv").write_text(
                "\n".join([
                    ",".join(trade_logger_module.HEADERS),
                    "1,BTC,LONG,2026-04-01 02:03,2026-04-01 02:09,6.0,67788.0,67888.0,225.0,2,3.32,1.47,66000,69000,take_profit,71.0,WIN",
                    "2,ETH,SHORT,2026-04-01 04:00,2026-04-01 04:30,30.0,2000.0,1990.0,150.0,2,0.75,0.50,2100,1900,take_profit,67.0,WIN",
                ]),
                encoding="utf-8",
            )
            promotion_gate_module.trade_dataset.resolve_richest_history_data_dir = lambda preferred=None: data_dir
            promotion_gate_module._ensure_precision_report = lambda *_args, **_kwargs: ({
                "labeled_episodes": 4,
                "overall_win_rate": 0.75,
                "best_rules": [{"win_rate": 0.75, "samples": 4}],
            }, data_dir / "precision_lab_report.json")

            gate = promotion_gate_module.evaluate_live_promotion(cfg, data_dir)
            assert gate["passed"] is True
            assert gate["trade_metrics"]["closed_trades"] == 2
            assert gate["history_source"]["backfilled_trade_rows"] == 2
    finally:
        promotion_gate_module.trade_dataset.resolve_richest_history_data_dir = original_resolve
        promotion_gate_module._ensure_precision_report = original_report


def test_dashboard_kill_endpoint_sets_control_state() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        original_control = dashboard_module.CONTROL
        original_kill = dashboard_module.KILL
        original_state = dashboard_module.STATE
        original_log = dashboard_module.LOG
        original_snapshot = dashboard_module.SNAPSHOT
        original_remote = dict(dashboard_module._remote_state)
        try:
            dashboard_module.CONTROL = temp / "control.json"
            dashboard_module.KILL = temp / "KILL"
            dashboard_module.STATE = temp / "state.json"
            dashboard_module.LOG = temp / "trades_log.csv"
            dashboard_module.SNAPSHOT = temp / "dashboard_snapshot.json"
            dashboard_module._remote_state = {"snapshot": None}
            client = dashboard_module.app.test_client()
            resp = client.post("/api/kill", json={"reason": "test kill"})
            assert resp.status_code == 200
            state = client.get("/api/state").get_json()
            assert state["control"]["kill"]["active"] is True
            assert dashboard_module.KILL.exists(), "local kill endpoint should touch the kill file"
            assert dashboard_module.SNAPSHOT.exists(), "kill updates should keep the canonical snapshot in sync"
        finally:
            dashboard_module.CONTROL = original_control
            dashboard_module.KILL = original_kill
            dashboard_module.STATE = original_state
            dashboard_module.LOG = original_log
            dashboard_module.SNAPSHOT = original_snapshot
            dashboard_module._remote_state = original_remote


def test_dashboard_state_prefers_canonical_snapshot() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        original_control = dashboard_module.CONTROL
        original_kill = dashboard_module.KILL
        original_state = dashboard_module.STATE
        original_log = dashboard_module.LOG
        original_snapshot = dashboard_module.SNAPSHOT
        original_remote = dict(dashboard_module._remote_state)
        try:
            dashboard_module.CONTROL = temp / "control.json"
            dashboard_module.KILL = temp / "KILL"
            dashboard_module.STATE = temp / "state.json"
            dashboard_module.LOG = temp / "trades_log.csv"
            dashboard_module.SNAPSHOT = temp / "dashboard_snapshot.json"
            dashboard_module._remote_state = {"snapshot": None}

            dashboard_module.STATE.write_text(json.dumps({
                "status": "running",
                "cycle_number": 7,
                "positions": [],
                "signals": {},
            }))
            dashboard_module.LOG.write_text("coin,exit_price,pnl_usd\nBTC,0,0\n")

            snapshot = {
                "state": {
                    "status": "running",
                    "cycle_number": 99,
                    "positions": [],
                    "signals": {},
                    "positions_count": 0,
                    "decision_summary": {
                        "long_count": 0,
                        "short_count": 0,
                        "flat_count": 0,
                        "tradable_count": 0,
                        "tradable_active_count": 0,
                        "lead": None,
                    },
                },
                "trades": [{"coin": "ETH", "exit_price": "100", "pnl_usd": "12.5"}],
                "stats": {
                    "total": 1,
                    "wins": 1,
                    "losses": 0,
                    "win_rate": 100.0,
                    "total_pnl": 12.5,
                    "avg_win": 12.5,
                    "avg_loss": 0,
                    "best": 12.5,
                    "worst": 12.5,
                },
                "control": {"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
                "runtime": {"stale": False, "state_age_seconds": 3},
                "server_time": "2026-04-06 09:00:00",
            }
            dashboard_module.SNAPSHOT.write_text(json.dumps(snapshot))

            client = dashboard_module.app.test_client()
            payload = client.get("/api/state").get_json()
            assert payload["state"]["cycle_number"] == 99
            assert payload["stats"]["total"] == 1
            assert payload["trades"][0]["coin"] == "ETH"
        finally:
            dashboard_module.CONTROL = original_control
            dashboard_module.KILL = original_kill
            dashboard_module.STATE = original_state
            dashboard_module.LOG = original_log
            dashboard_module.SNAPSHOT = original_snapshot
            dashboard_module._remote_state = original_remote


def test_dashboard_refreshes_snapshot_when_state_changes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        original_control = dashboard_module.CONTROL
        original_kill = dashboard_module.KILL
        original_state = dashboard_module.STATE
        original_log = dashboard_module.LOG
        original_snapshot = dashboard_module.SNAPSHOT
        original_remote = dict(dashboard_module._remote_state)
        original_queue_refresh = dashboard_module._queue_local_snapshot_refresh
        original_local_cache = dict(dashboard_module._local_snapshot_cache)
        try:
            dashboard_module.CONTROL = temp / "control.json"
            dashboard_module.KILL = temp / "KILL"
            dashboard_module.STATE = temp / "state.json"
            dashboard_module.LOG = temp / "trades_log.csv"
            dashboard_module.SNAPSHOT = temp / "dashboard_snapshot.json"
            dashboard_module._remote_state = {"snapshot": None}
            dashboard_module._local_snapshot_cache = {
                "snapshot": None,
                "mtime_ns": None,
                "refreshing": False,
                "last_refresh_started": 0.0,
                "last_refresh_finished": 0.0,
                "last_refresh_error": "",
            }

            snapshot = {
                "state": {
                    "status": "running",
                    "cycle_number": 10,
                    "positions": [],
                    "signals": {},
                    "positions_count": 0,
                    "decision_summary": {
                        "long_count": 0,
                        "short_count": 0,
                        "flat_count": 0,
                        "tradable_count": 0,
                        "tradable_active_count": 0,
                        "lead": None,
                    },
                },
                "trades": [],
                "stats": {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0},
                "control": {"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
                "runtime": {"stale": False, "state_age_seconds": 1},
                "server_time": "2026-04-06 09:05:00",
            }
            dashboard_module.SNAPSHOT.write_text(json.dumps(snapshot))
            os.utime(dashboard_module.SNAPSHOT, (1, 1))

            dashboard_module.STATE.write_text(json.dumps({
                "status": "running",
                "cycle_number": 11,
                "positions": [],
                "signals": {},
            }))
            dashboard_module.LOG.write_text("coin,exit_price,pnl_usd\n")
            os.utime(dashboard_module.STATE, None)
            refresh_calls = {"count": 0}
            dashboard_module._queue_local_snapshot_refresh = lambda server_timestamp=None, force=False: refresh_calls.__setitem__("count", refresh_calls["count"] + 1) or True

            client = dashboard_module.app.test_client()
            payload = client.get("/api/state").get_json()
            assert payload["state"]["cycle_number"] == 10
            assert refresh_calls["count"] == 1
        finally:
            dashboard_module.CONTROL = original_control
            dashboard_module.KILL = original_kill
            dashboard_module.STATE = original_state
            dashboard_module.LOG = original_log
            dashboard_module.SNAPSHOT = original_snapshot
            dashboard_module._remote_state = original_remote
            dashboard_module._queue_local_snapshot_refresh = original_queue_refresh
            dashboard_module._local_snapshot_cache = original_local_cache


def test_dashboard_loads_prebuilt_snapshot_without_rehydrating() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        original_snapshot = dashboard_module.SNAPSHOT
        original_builder = dashboard_module.build_dashboard_snapshot
        original_local_cache = dict(dashboard_module._local_snapshot_cache)
        try:
            dashboard_module.SNAPSHOT = temp / "dashboard_snapshot.json"
            dashboard_module._local_snapshot_cache = {
                "snapshot": None,
                "mtime_ns": None,
                "refreshing": False,
                "last_refresh_started": 0.0,
                "last_refresh_finished": 0.0,
                "last_refresh_error": "",
            }
            dashboard_module.SNAPSHOT.write_text(json.dumps({
                "state": {"status": "running", "cycle_number": 77, "positions": [], "signals": {}},
                "trades": [],
                "stats": {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0, "avg_win": 0, "avg_loss": 0, "best": 0, "worst": 0},
                "control": {"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
                "runtime": {"stale": False, "state_age_seconds": 2},
                "server_time": "2026-04-23 23:16:00",
            }))
            dashboard_module.build_dashboard_snapshot = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not rehydrate prebuilt snapshot"))

            payload = dashboard_module._load_snapshot_local()
            assert payload is not None
            assert payload["state"]["cycle_number"] == 77
            assert payload["runtime"]["state_age_seconds"] == 2
        finally:
            dashboard_module.SNAPSHOT = original_snapshot
            dashboard_module.build_dashboard_snapshot = original_builder
            dashboard_module._local_snapshot_cache = original_local_cache


def test_hosted_dashboard_bundle_matches_local_template() -> None:
    local_template = Path("dashboard/templates/dashboard.html").read_text()
    hosted_bundle = Path("netlify-dashboard/public/index.html").read_text()
    assert hosted_bundle == local_template, "hosted dashboard should mirror the local dashboard UI exactly"


def test_dashboard_template_compacts_daily_view_and_hides_support_pending() -> None:
    template = Path("dashboard/templates/dashboard.html").read_text()
    assert "Desk Briefing" in template
    assert "Open Level Sheet" in template
    assert "Latest Win" in template
    assert "daily-briefing" in template
    assert "Stock Desks" in template
    assert "Mag7" in template
    assert "prob-chip" in template
    assert "Reclaim odds" in template
    assert "next_setup_reason" in template
    assert "setupStanceChipHtml" in template
    assert "Watchlist" in template
    assert "renderCallWatchlist" in template
    assert "BULLISH CALL" in template
    assert "BEARISH CALL" in template
    assert "Only the next level is shown" in template
    assert "simpleNextText" in template
    assert "simpleThesisText" in template
    assert "simpleInvalidationText" in template
    assert "Invalid if" in template
    assert "Opened because:" in template
    assert "Holding because:" in template
    assert "Proactive Desk" in template
    assert "renderProactiveDesk" in template
    assert "Morning Scout Book" in template
    assert "Starter Basket" in template
    assert "Starter Execution" in template
    assert "Forecast Calibration" in template
    assert "<strong>Lead:</strong>" in template
    assert "friction-stack" in template
    assert "catalyst-rail" in template
    assert "leadSummaryText(actionBoard.lead" in template
    assert "AbortController" in template
    assert "scheduleRefresh(" in template
    assert "setInterval(refresh, 10000);" not in template
    assert "Watching only" not in template
    assert "🧾 Latest Lesson" not in template
    assert '<div class="asset-section-title">Support Pending</div>' not in template


def test_local_dashboard_serves_hosted_bundle() -> None:
    client = dashboard_module.app.test_client()
    served = client.get("/").data
    hosted_bundle = Path("netlify-dashboard/public/index.html").read_bytes()
    assert served == hosted_bundle, "local dashboard root should serve the exact hosted UI bundle"


def test_install_launchagent_preserves_learning_datasets() -> None:
    script = Path("install_launchagent.sh").read_text(encoding="utf-8")
    for filename in (
        "decision_dataset.jsonl",
        "feature_store.jsonl",
        "trade_dataset.jsonl",
        "precision_lab_report.json",
        "playbook_distiller_report.json",
    ):
        assert f'--exclude "{filename}"' in script, f"runtime sync should preserve {filename}"


def test_market_map_signal_respects_operator_daily_levels() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = market_map_module.DAILY_MARKET_MAP_JSON
        original_daily_close = market_map_module._get_daily_close
        try:
            market_map_module.DAILY_MARKET_MAP_JSON = Path(tmpdir) / "daily_market_map.json"
            market_map_module._get_daily_close = lambda _coin, ttl_seconds=300: 72_600.0
            market_map_module.save_market_map({
                "date": "2026-04-06",
                "global_notes": "BTC daily reclaim above 71.5k and 72.5k matters",
                "coins": {
                    "BTC": {
                        "bias": "BULLISH",
                        "confidence": "HIGH",
                        "supports": [60_000.0, 68_500.0],
                        "resistances": [71_500.0, 72_500.0, 75_000.0],
                        "daily_close_long_above": [71_500.0, 72_500.0],
                        "daily_close_short_below": [67_500.0],
                        "demand_zone": {"low": 59_500.0, "high": 60_500.0},
                        "supply_zone": {"low": 74_500.0, "high": 75_500.0},
                        "notes": "Above the reclaim band, shorts should stay defensive.",
                    }
                },
            })
            signal = market_map_module.get_market_map_signal("BTC", current_price=71_900.0, closed_price=71_900.0)
            assert signal.valid is True
            assert signal.favor_longs is True
            assert signal.block_shorts is True
            assert signal.above_reclaim_levels == [71_500.0, 72_500.0]
            assert signal.live_above_reclaim_levels == [71_500.0]
            assert signal.score_adjustment > 0
            assert "daily reclaim is holding" in signal.summary.lower()
        finally:
            market_map_module.DAILY_MARKET_MAP_JSON = original_path
            market_map_module._get_daily_close = original_daily_close


def test_market_map_signal_flags_reclaim_that_slipped_back_below_live_price() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = market_map_module.DAILY_MARKET_MAP_JSON
        original_daily_close = market_map_module._get_daily_close
        try:
            market_map_module.DAILY_MARKET_MAP_JSON = Path(tmpdir) / "daily_market_map.json"
            market_map_module._get_daily_close = lambda _coin, ttl_seconds=300: 256.52
            market_map_module.save_market_map({
                "date": "2026-04-21",
                "coins": {
                    "AMZN": {
                        "bias": "BULLISH",
                        "confidence": "HIGH",
                        "supports": [250.0],
                        "resistances": [253.36, 256.5],
                        "daily_close_long_above": [253.36],
                        "daily_close_short_below": [248.0],
                    }
                },
            })
            signal = market_map_module.get_market_map_signal("AMZN", current_price=253.03, closed_price=256.52)
            assert signal.valid is True
            assert signal.above_reclaim_levels == [253.36]
            assert signal.live_above_reclaim_levels == []
            assert "slipped back below reclaim" in signal.summary.lower()
        finally:
            market_map_module.DAILY_MARKET_MAP_JSON = original_path
            market_map_module._get_daily_close = original_daily_close


def test_effective_market_map_auto_maps_tracked_assets() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_path = market_map_module.DAILY_MARKET_MAP_JSON
        original_fetch = market_map_module.fetch_candles
        original_daily_close = market_map_module._get_daily_close
        original_auto_cache = dict(market_map_module._AUTO_ENTRY_CACHE)
        try:
            market_map_module.DAILY_MARKET_MAP_JSON = Path(tmpdir) / "daily_market_map.json"
            market_map_module._AUTO_ENTRY_CACHE.clear()
            market_map_module._get_daily_close = lambda _coin, ttl_seconds=300: 112.0

            df = pd.DataFrame({
                "timestamp": pd.date_range("2026-01-01", periods=60, freq="1d", tz="UTC"),
                "open": [100 + i * 0.4 for i in range(60)],
                "high": [101 + i * 0.45 for i in range(60)],
                "low": [99 + i * 0.35 for i in range(60)],
                "close": [100 + i * 0.42 for i in range(60)],
                "volume": [1000 + i * 10 for i in range(60)],
                "trades": [0 for _ in range(60)],
            })

            market_map_module.fetch_candles = lambda coin, interval="1d", lookback=140: df.copy()

            effective = market_map_module.build_effective_market_map(["BTC"], current_prices={"BTC": 124.0})
            entry = effective["coins"]["BTC"]
            assert entry["source"] == "AUTO"
            assert entry["auto_generated"] is True
            assert entry["supports"], "auto map should synthesize support levels"
            assert entry["resistances"], "auto map should synthesize resistance levels"
            assert entry["trade_mode"], "auto map should provide a playbook"

            signal = market_map_module.get_market_map_signal("BTC", current_price=124.0, closed_price=123.4)
            assert signal.valid is True
            assert signal.source == "AUTO"
            assert "map" in signal.summary.lower()
        finally:
            market_map_module.DAILY_MARKET_MAP_JSON = original_path
            market_map_module.fetch_candles = original_fetch
            market_map_module._get_daily_close = original_daily_close
            market_map_module._AUTO_ENTRY_CACHE.clear()
            market_map_module._AUTO_ENTRY_CACHE.update(original_auto_cache)


def test_trade_review_feedback_hard_blocks_repeated_bad_thesis() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_reviews = trade_review_module.TRADE_REVIEWS_JSON
        try:
            trade_review_module.TRADE_REVIEWS_JSON = Path(tmpdir) / "trade_reviews.json"
            for trade_id in ("201", "202", "203"):
                trade_review_module.upsert_review({
                    "trade_id": trade_id,
                    "coin": "ETH",
                    "direction": "SHORT",
                    "verdict": "BAD_THESIS",
                    "thesis_quality": "WEAK",
                    "execution_quality": "OK",
                    "tags": ["faded-demand"],
                    "notes": "Shorted into demand and got squeezed.",
                })
            feedback = trade_review_module.get_directional_feedback("ETH", "SHORT")
            assert feedback["hard_block"] is True
            assert "weak ETH SHORT thesis".lower() in feedback["reason"].lower()
        finally:
            trade_review_module.TRADE_REVIEWS_JSON = original_reviews


def test_dashboard_snapshot_includes_market_map_and_trade_reviews() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "cycle_number": 88,
            "positions": [],
            "signals": {},
            "mode": "dry_run",
        },
        trades=[{
            "trade_id": "11",
            "coin": "BTC",
            "direction": "LONG",
            "opened_at": "2026-04-06 09:00",
            "closed_at": "2026-04-06 10:00",
            "entry_price": 70_000.0,
            "exit_price": 71_000.0,
            "size_usd": 100.0,
            "pnl_usd": 10.0,
            "pnl_pct": 0.10,
            "exit_reason": "take_profit",
        }],
        control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
        market_map={
            "date": "2026-04-06",
            "updated_at": "2026-04-06 08:00:00",
            "coins": {
                "BTC": {"bias": "BULLISH", "supports": [68_000.0], "resistances": [71_500.0]},
            },
        },
        trade_reviews={
            "updated_at": "2026-04-06 11:00:00",
            "reviews": {
                "11": {
                    "trade_id": "11",
                    "coin": "BTC",
                    "direction": "LONG",
                    "verdict": "GOOD_TRADE",
                    "thesis_quality": "STRONG",
                    "execution_quality": "GOOD",
                    "notes": "Clean reclaim and follow-through.",
                }
            },
        },
        server_timestamp="2026-04-06 11:05:00",
    )
    assert snapshot["market_map_summary"]["count"] == 1
    assert snapshot["review_summary"]["count"] == 1
    assert snapshot["review_summary"]["coverage_pct"] == 100.0
    assert snapshot["trades"][0]["review"]["verdict"] == "GOOD_TRADE"


def test_dashboard_snapshot_backfills_stock_desks_from_runtime_defaults() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "cycle_number": 101,
            "positions": [],
            "signals": {},
            "mode": "dry_run",
            "config": {
                "analysis_coins": ["NVDA"],
                "asset_categories": {"NVDA": "semis_memory"},
                "instrument_types": {"NVDA": "equity"},
            },
        },
        trades=[],
        control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
        market_map={
            "date": "2026-04-23",
            "updated_at": "2026-04-23 09:00:00",
            "coins": {
                "TSLA": {"bias": "BULLISH"},
                "NVDA": {"bias": "BULLISH"},
                "CRWV": {"bias": "BULLISH"},
            },
        },
        server_timestamp="2026-04-23 09:05:00",
    )
    equity_counts = snapshot["action_board"]["summary"]["bucket_counts"]["equity"]
    assert equity_counts["count"] >= 3
    equity_items = [item for item in snapshot["action_board"]["items"] if item["asset_bucket"] == "equity"]
    by_coin = {item["coin"]: item for item in equity_items}
    assert by_coin["TSLA"]["asset_category"] == "mag7"
    assert by_coin["TSLA"]["tradable"] is True
    assert by_coin["NVDA"]["asset_category"] == "mag7"
    assert by_coin["NVDA"]["tradable"] is True
    assert by_coin["NVDA"]["asset_categories"] == ["mag7", "semis_memory"]
    assert by_coin["CRWV"]["asset_category"] == "neoclouds"


def test_default_stock_categories_keep_mag7_complete() -> None:
    cfg = Config()
    mag7 = sorted(
        coin
        for coin, categories in cfg.trading.asset_category_map.items()
        if "mag7" in (categories if isinstance(categories, list) else [categories])
    )
    assert mag7 == ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
    missing = [
        coin
        for coin in hyperliquid_markets_module.TRADEXYZ_ASSET_METADATA.keys()
        if coin not in cfg.trading.analysis_coins
    ]
    assert missing == []
    missing_executable = [
        coin
        for coin in hyperliquid_markets_module.TRADEXYZ_ASSET_METADATA.keys()
        if coin not in cfg.trading.coins
    ]
    assert missing_executable == []


def test_default_crypto_category_includes_mon() -> None:
    cfg = Config()
    assert "MON" in cfg.trading.coins
    assert "MON" in cfg.trading.analysis_coins
    assert cfg.trading.instrument_types["MON"] == "crypto"
    assert cfg.trading.asset_category_map["MON"] == ["crypto"]
    assert cfg.trading.portfolio_theme_map["MON"] == "CRYPTO_HIGH_BETA"
    catalog = hyperliquid_markets_module._catalog_from_fallback()
    assert catalog["MON"]["instrument_type"] == "crypto"
    assert catalog["MON"]["market_type"] == "perp"


def test_tradexyz_pre_ipo_cerebras_defaults_to_event_theme() -> None:
    cfg = Config()
    assert "CBRS" in cfg.trading.coins
    assert "CBRS" in cfg.trading.analysis_coins
    assert cfg.trading.instrument_types["CBRS"] == "equity"
    assert cfg.trading.asset_category_map["CBRS"] == ["pre_ipo", "semis_memory", "ai_infra"]
    assert cfg.trading.portfolio_theme_map["CBRS"] == "PRE_IPO_EVENT"
    catalog = hyperliquid_markets_module._catalog_from_fallback()
    assert catalog["CBRS"]["venue_symbol"] == "xyz:CBRS"
    assert catalog["CBRS"]["market_type"] == "perp"
    assert catalog["CBRS"]["live_tradeable"] is True
    assert catalog["CBRS"]["pre_ipo"] is True


def test_tradexyz_latest_launches_are_first_class_defaults() -> None:
    cfg = Config()
    expected = {
        "EBAY": ("equity", ["consumer"], "CONSUMER_GROWTH"),
        "EWZ": ("index", ["latam_macro", "indices_macro"], "LATAM_MACRO"),
        "NIFTY": ("index", ["asia_macro", "indices_macro"], "ASIA_MACRO"),
        "KRW": ("index", ["fx_rates", "asia_macro"], "FX_RATES"),
        "ZM": ("equity", ["software", "growth"], "SOFTWARE_GROWTH"),
    }
    catalog = hyperliquid_markets_module._catalog_from_fallback()
    for coin, (instrument_type, categories, theme) in expected.items():
        assert coin in cfg.trading.coins
        assert coin in cfg.trading.analysis_coins
        assert cfg.trading.instrument_types[coin] == instrument_type
        assert cfg.trading.asset_category_map[coin] == categories
        assert cfg.trading.portfolio_theme_map[coin] == theme
        assert catalog[coin]["venue_symbol"] == f"xyz:{coin}"
        assert catalog[coin]["live_tradeable"] is True


def test_proactive_intelligence_builds_full_research_stack() -> None:
    state = {
        "portfolio_usd": 10000.0,
        "positions": [],
        "pending_orders": [],
        "config": {
            "instrument_types": {
                "INTC": "equity",
                "AMD": "equity",
                "NVDA": "equity",
                "MON": "crypto",
            },
            "asset_categories": {
                "INTC": ["semis_memory"],
                "AMD": ["semis_memory"],
                "NVDA": ["mag7", "semis_memory"],
                "MON": ["crypto"],
            },
            "portfolio_theme_map": {
                "INTC": "SEMIS_MEMORY",
                "AMD": "SEMIS_MEMORY",
                "NVDA": "MEGA_CAP_TECH",
                "MON": "CRYPTO_HIGH_BETA",
            },
        },
        "signals": {
            "INTC": {
                "action": "FLAT",
                "market_map_bias": "BULLISH",
                "score": 64.0,
                "live_price": 40.0,
                "news_event_score": 4.2,
                "news_catalyst_score": 4.0,
                "news_event_summary": "earnings setup",
                "analyst_revision_score": 1.5,
                "expectancy_probability": 0.61,
            },
            "AMD": {
                "action": "FLAT",
                "market_map_bias": "BULLISH",
                "score": 58.0,
                "live_price": 180.0,
                "news_catalyst_score": 2.5,
                "expectancy_probability": 0.56,
            },
            "NVDA": {
                "action": "FLAT",
                "market_map_bias": "BULLISH",
                "score": 57.0,
                "live_price": 120.0,
                "news_catalyst_score": 2.1,
            },
            "MON": {
                "action": "LONG",
                "score": 66.0,
                "live_price": 4.2,
                "expectancy_probability": 0.59,
            },
        },
    }
    market_map = {
        "coins": {
            "INTC": {"bias": "BULLISH", "supports": [38.0], "resistances": [42.0]},
            "AMD": {"bias": "BULLISH"},
            "NVDA": {"bias": "BULLISH"},
            "MON": {"bias": "BULLISH"},
        }
    }
    cfg = build_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        report = proactive_intelligence_module.build_and_save_report(
            state=state,
            market_map=market_map,
            config=cfg.trading,
            data_dir=Path(tmpdir),
        )
        assert report["enabled"] is True
        assert report["thesis_ledger"]["summary"]["active_count"] >= 3
        assert report["morning_scout_book"]["summary"]["top_call"]["coin"] == "INTC"
        assert report["read_through_engine"]["summary"]["impact_count"] >= 1
        assert report["starter_basket_optimizer"]["summary"]["allocation_count"] >= 1
        assert report["forecast_calibration"]["summary"]["open_count"] >= 1
        assert (Path(tmpdir) / "thesis_ledger.jsonl").exists()
        assert (Path(tmpdir) / "forecast_ledger.jsonl").exists()


def test_proactive_starter_execution_opens_capped_event_orders() -> None:
    cfg = build_config()
    cfg.trading.coins = ["GOOGL", "META", "AMZN"]
    cfg.trading.analysis_coins = ["GOOGL", "META", "AMZN"]
    cfg.trading.instrument_types.update({"GOOGL": "equity", "META": "equity", "AMZN": "equity"})
    cfg.trading.asset_category_map.update({
        "GOOGL": ["mag7"],
        "META": ["mag7"],
        "AMZN": ["mag7"],
    })
    cfg.trading.portfolio_theme_map.update({
        "GOOGL": "MEGA_CAP_TECH",
        "META": "MEGA_CAP_TECH",
        "AMZN": "MEGA_CAP_TECH",
    })
    cfg.trading.max_trade_usd = 1000.0
    cfg.trading.max_position_pct = 0.10
    cfg.trading.min_trade_usd = 100.0
    cfg.trading.event_risk_budget_min_trade_usd = 100.0
    cfg.trading.event_risk_budget_max_portfolio_pct = 0.05
    cfg.trading.event_risk_budget_max_theme_pct = 0.04
    cfg.trading.event_risk_budget_max_single_pct = 0.02
    cfg.trading.event_risk_budget_strict_caps = True
    cfg.trading.proactive_starter_execution_enabled = True
    cfg.trading.proactive_starter_execution_max_per_cycle = 3
    cfg.trading.proactive_starter_execution_min_score = 58.0
    cfg.trading.proactive_starter_execution_cooldown_minutes = 0.0
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False

    exchange = DryRunExchange(starting_balance_usd=10000.0, supported_symbols=["GOOGL", "META", "AMZN"])
    agent = TradingAgent(cfg, [exchange])
    agent.risk.positions = {}
    agent.order_mgr.pending_orders = {}
    agent._tradable_coins = ["GOOGL", "META", "AMZN"]
    agent._tradable_coin_set = set(agent._tradable_coins)
    agent._analysis_coins = list(agent._tradable_coins)
    agent._last_signals = {}
    for coin in agent._analysis_coins:
        agent._last_signals[coin] = {
            "action": "FLAT",
            "decision": "FLAT",
            "score": 64.0,
            "confidence": "MEDIUM",
            "price": 100.0,
            "live_price": 100.0,
            "analysis_price": 100.0,
            "instrument_type": "equity",
            "asset_categories": ["mag7"],
            "market_map_bias": "BULLISH",
            "news_event_score": 4.2,
            "news_catalyst_score": 4.0,
            "official_event_score": 3.0,
            "analyst_revision_score": 1.5,
            "expectancy_probability": 0.61,
            "decision_stage": "major_catalyst_watch",
            "planned_stop_loss": 92.0,
            "planned_take_profit": 118.0,
            "trade_plan": {"stop_loss": 92.0, "take_profit": 118.0, "risk_reward_ratio": 2.25},
        }

    opened_orders = []

    def fake_execute(coin, signal, order):
        opened_orders.append(order)
        agent.risk.record_open(
            order,
            exchange="UnitTest",
            metadata={"entry_context": agent._build_entry_context(coin, signal, order, entry_type="proactive_starter")},
        )
        return True

    original_data_dir = agent_module.DATA_DIR
    agent._execute_order = fake_execute
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            agent_module.DATA_DIR = Path(tmpdir)
            execution = agent._execute_proactive_starter_basket(10000.0, {"signal_score": 50.0})
        finally:
            agent_module.DATA_DIR = original_data_dir

    assert execution["summary"]["opened_count"] == 2
    assert len(opened_orders) == 2
    assert sum(order.size_usd for order in opened_orders) <= 400.01
    assert all(order.size_usd <= 200.01 for order in opened_orders)
    assert all(
        (pos.metadata.get("entry_context") or {}).get("event_risk_budget_active")
        for pos in agent.risk.positions.values()
    )


def _install_runner_position(agent: TradingAgent, coin: str = "GOOGL") -> None:
    now = time.time()
    entry_context = {
        "instrument_type": "equity",
        "score": 68.0,
        "conviction_entry_event": True,
        "event_risk_budget_active": True,
        "news_event_score": 4.1,
        "news_catalyst_score": 3.7,
        "official_event_score": 2.0,
        "planned_stop_loss": 95.0,
        "planned_take_profit": 110.0,
        "trade_plan": {"stop_loss": 95.0, "take_profit": 110.0},
    }
    agent.risk.positions = {
        coin: OpenPosition(
            coin=coin,
            direction="LONG",
            entry_price=100.0,
            size_usd=200.0,
            size_coin=2.0,
            stop_loss=95.0,
            take_profit=110.0,
            trailing_stop_price=88.0,
            opened_at=now - 90 * 60,
            exchange="UnitTest",
            metadata={"entry_context": entry_context},
        )
    }
    agent._last_signals = {
        coin: {
            "action": "FLAT",
            "score": 67.0,
            "instrument_type": "equity",
            "live_price": 111.0,
            "market_map_bias": "BULLISH",
            "market_map_reclaim_confirmed": True,
            "news_event_score": 4.1,
            "news_catalyst_score": 3.7,
            "official_event_score": 2.0,
            "conviction_entry_event": True,
            "thesis": {
                "state": "ACTIVE",
                "permitted": True,
                "conviction_score": 68.0,
                "conflict_points": 0,
            },
        }
    }


def test_thesis_runner_defers_take_profit_and_extends_target() -> None:
    cfg = build_config()
    cfg.trading.coins = ["GOOGL"]
    cfg.trading.instrument_types.update({"GOOGL": "equity"})
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    exchange = DryRunExchange(starting_balance_usd=10000.0, supported_symbols=["GOOGL"])
    agent = TradingAgent(cfg, [exchange])
    agent._tradable_coins = ["GOOGL"]
    agent._tradable_coin_set = {"GOOGL"}
    _install_runner_position(agent, "GOOGL")

    closed = []
    agent._close_position = lambda coin, reason, price: closed.append((coin, reason, price))

    agent._check_and_execute_exits({"GOOGL": 111.0}, 10000.0)

    assert closed == []
    assert "GOOGL" in agent.risk.positions
    assert agent.risk.positions["GOOGL"].take_profit > 111.0
    assert agent.risk.positions["GOOGL"].metadata["runner"]["deferred_exit_count"] == 1


def test_thesis_runner_still_honors_stop_loss() -> None:
    cfg = build_config()
    cfg.trading.coins = ["GOOGL"]
    cfg.trading.instrument_types.update({"GOOGL": "equity"})
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0, supported_symbols=["GOOGL"])])
    agent._tradable_coins = ["GOOGL"]
    agent._tradable_coin_set = {"GOOGL"}
    _install_runner_position(agent, "GOOGL")

    closed = []

    def fake_close(coin, reason, price):
        closed.append((coin, reason, price))
        agent.risk.positions.pop(coin, None)

    agent._close_position = fake_close

    agent._check_and_execute_exits({"GOOGL": 94.0}, 10000.0)

    assert closed == [("GOOGL", "stop_loss", 94.0)]
    assert "GOOGL" not in agent.risk.positions


def test_thesis_runner_blocks_time_stop_for_multi_day_event_hold() -> None:
    cfg = build_config()
    cfg.trading.coins = ["GOOGL"]
    cfg.trading.instrument_types.update({"GOOGL": "equity"})
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    cfg.trading.time_stop_minutes = 60.0
    cfg.trading.time_stop_min_tp_progress = 0.25
    cfg.trading.thesis_runner_event_min_hold_minutes = 10080.0
    cfg.trading.thesis_runner_event_max_flat_cycles = 2160
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0, supported_symbols=["GOOGL"])])
    agent._tradable_coins = ["GOOGL"]
    agent._tradable_coin_set = {"GOOGL"}
    _install_runner_position(agent, "GOOGL")
    agent.risk.positions["GOOGL"].opened_at = time.time() - 2 * 3600
    agent._last_signals["GOOGL"].update({
        "live_price": 101.0,
        "thesis": {
            "state": "NO_TRADE",
            "permitted": False,
            "conviction_score": 68.0,
            "conflict_points": 0,
        },
    })
    agent._flat_streak["GOOGL"] = 100
    signal = SimpleNamespace(
        action="FLAT",
        score=50.0,
        price=101.0,
        expectancy={"score": 48.0, "uncertainty": 0.50},
        thesis={"state": "NO_TRADE", "permitted": False, "conflict_points": 0},
    )

    assert agent._detect_position_invalidation("GOOGL", "LONG", signal) == ""
    decay = agent._assess_conviction_decay("GOOGL", "LONG", signal)
    assert decay["should_exit"] is False
    assert decay["summary"].startswith("runner hold:")


def test_thesis_runner_still_honors_hard_structure_invalidation() -> None:
    cfg = build_config()
    cfg.trading.coins = ["GOOGL"]
    cfg.trading.instrument_types.update({"GOOGL": "equity"})
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    cfg.trading.time_stop_minutes = 60.0
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0, supported_symbols=["GOOGL"])])
    agent._tradable_coins = ["GOOGL"]
    agent._tradable_coin_set = {"GOOGL"}
    _install_runner_position(agent, "GOOGL")
    agent.risk.positions["GOOGL"].opened_at = time.time() - 2 * 3600
    agent._last_signals["GOOGL"].update({
        "live_price": 101.0,
        "structure_trend": "DOWNTREND",
        "orderbook_breakout_state": "CONFIRMED_BEARISH_BREAKDOWN",
        "thesis": {
            "state": "NO_TRADE",
            "permitted": False,
            "conviction_score": 68.0,
            "conflict_points": 2,
        },
    })
    signal = SimpleNamespace(
        action="FLAT",
        score=50.0,
        price=101.0,
        expectancy={"score": 48.0, "uncertainty": 0.50},
        thesis={"state": "NO_TRADE", "permitted": False, "conflict_points": 2},
    )

    assert agent._detect_position_invalidation("GOOGL", "LONG", signal) == "structure_invalidation"


def test_stale_adverse_exit_cuts_multi_day_loser_without_killing_runners() -> None:
    cfg = build_config()
    cfg.trading.coins = ["BTC"]
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    cfg.trading.stale_adverse_min_minutes = 12 * 60
    cfg.trading.stale_adverse_max_adverse_r = 0.25
    cfg.trading.stale_adverse_max_tp_progress = 0.10
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0, supported_symbols=["BTC"])])
    agent.risk.positions = {
        "BTC": OpenPosition(
            coin="BTC",
            direction="LONG",
            entry_price=100.0,
            size_usd=200.0,
            size_coin=2.0,
            stop_loss=90.0,
            take_profit=130.0,
            trailing_stop_price=88.0,
            opened_at=time.time() - 18 * 3600,
            exchange="UnitTest",
            metadata={"entry_context": {"score": 66.0, "planned_stop_loss": 90.0, "planned_take_profit": 130.0}},
        )
    }
    agent._last_signals = {
        "BTC": {
            "action": "SHORT",
            "score": 28.0,
            "live_price": 96.0,
            "thesis": {"state": "ACTIVE", "permitted": True, "conflict_points": 0},
        }
    }
    signal = SimpleNamespace(
        action="SHORT",
        score=28.0,
        price=96.0,
        thesis={"state": "ACTIVE", "permitted": True, "conflict_points": 0},
    )

    assert agent._detect_position_invalidation("BTC", "LONG", signal) == "stale_adverse"


def test_winner_stickiness_blocks_conviction_churn_exit() -> None:
    cfg = build_config()
    cfg.trading.coins = ["BTC"]
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    cfg.trading.winner_stickiness_min_profit_pct = 0.20
    cfg.trading.winner_stickiness_min_profit_r = 0.10
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0, supported_symbols=["BTC"])])
    agent.risk.positions = {
        "BTC": OpenPosition(
            coin="BTC",
            direction="LONG",
            entry_price=100.0,
            size_usd=200.0,
            size_coin=2.0,
            stop_loss=95.0,
            take_profit=120.0,
            trailing_stop_price=88.0,
            opened_at=time.time() - 45 * 60,
            exchange="UnitTest",
            metadata={"entry_context": {"score": 66.0, "planned_stop_loss": 95.0, "planned_take_profit": 120.0}},
        )
    }
    agent._last_signals = {
        "BTC": {
            "action": "FLAT",
            "score": 50.0,
            "live_price": 101.0,
            "thesis": {"state": "NO_TRADE", "permitted": False, "conflict_points": 0},
            "expectancy": {"score": 44.0, "uncertainty": 0.50},
        }
    }
    agent._flat_streak["BTC"] = 6
    signal = SimpleNamespace(
        action="FLAT",
        score=50.0,
        price=101.0,
        expectancy={"score": 44.0, "uncertainty": 0.50},
        thesis={"state": "NO_TRADE", "permitted": False, "conflict_points": 0},
    )

    decay = agent._assess_conviction_decay("BTC", "LONG", signal)

    assert decay["should_exit"] is False
    assert decay["summary"].startswith("winner hold:")
    assert agent._last_signals["BTC"]["winner_stickiness"]["active"] is True


def test_pair_trade_book_builds_equity_long_crypto_short_overlay() -> None:
    cfg = build_config()
    cfg.trading.min_trade_usd = 100.0
    cfg.trading.event_risk_budget_min_trade_usd = 100.0
    cfg.trading.pair_trade_max_notional_pct = 0.02
    state = {
        "portfolio_usd": 10000.0,
        "positions": [],
        "pending_orders": [],
        "signals": {},
    }
    scout_book = {
        "bullish_calls": [{
            "coin": "GOOGL",
            "direction": "LONG",
            "asset_bucket": "equity",
            "theme": "MEGA_CAP_TECH",
            "scout_score": 72.0,
            "invalidation": "Invalid below 180",
        }],
        "bearish_calls": [{
            "coin": "BTC",
            "direction": "SHORT",
            "asset_bucket": "coin",
            "theme": "CRYPTO_BETA",
            "scout_score": 62.0,
            "invalidation": "Invalid above 69000",
        }],
    }

    book = proactive_intelligence_module.build_pair_trade_book(state, scout_book, config=cfg.trading)

    assert book["summary"]["pair_count"] == 1
    assert book["pairs"][0]["long_coin"] == "GOOGL"
    assert book["pairs"][0]["short_coin"] == "BTC"
    hedge = book["hedge_allocations"][0]
    assert hedge["coin"] == "BTC"
    assert hedge["direction"] == "SHORT"
    assert 100.0 <= hedge["size_usd"] <= 200.0


def test_setup_quality_guard_blocks_toxic_reversal_family() -> None:
    cfg = build_config()
    cfg.trading.setup_quality_guard_enabled = True
    cfg.trading.setup_quality_guard_min_samples = 3
    cfg.trading.setup_quality_guard_signal_reversal_loss_limit = 2
    cfg.trading.setup_quality_guard_min_long_score = 70.0
    cfg.trading.setup_quality_guard_min_probability = 0.58
    cfg.trading.setup_quality_guard_min_expected_r = 0.22
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0)])
    signal = SimpleNamespace(
        action="LONG",
        score=66.0,
        price=100.0,
        reason="marginal setup",
        flat_reason="",
        expectancy={"probability": 0.53, "expected_r": 0.08, "uncertainty": 0.40},
        thesis={"state": "ACTIVE", "permitted": True, "conflict_points": 0},
    )
    agent._last_signals["BTC"] = {
        "action": "LONG",
        "score": 66.0,
        "expectancy_probability": 0.53,
        "expectancy_expected_r": 0.08,
    }
    rows = [
        ["1", "BTC", "LONG", "2026-05-01 09:00", "2026-05-01 09:30", "30", "100", "96", "100", "2", "-4.00", "-4.00", "95", "115", "signal_reversal", "66", "LOSS"],
        ["2", "BTC", "LONG", "2026-05-01 10:00", "2026-05-01 10:30", "30", "100", "97", "100", "2", "-3.00", "-3.00", "95", "115", "signal_reversal", "67", "LOSS"],
        ["3", "BTC", "LONG", "2026-05-01 11:00", "2026-05-01 11:30", "30", "100", "100.2", "100", "2", "0.20", "0.20", "95", "115", "conviction_lost", "68", "WIN"],
    ]
    original_trades_csv = agent_module.TRADES_CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            agent_module.TRADES_CSV = Path(tmpdir) / "trades_log.csv"
            with agent_module.TRADES_CSV.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(trade_logger_module.HEADERS)
                writer.writerows(rows)
            allowed = agent._setup_quality_guard("BTC", signal, portfolio_usd=10000.0, current_position=None)
        finally:
            agent_module.TRADES_CSV = original_trades_csv

    assert allowed is False
    assert signal.action == "FLAT"
    assert "toxic" in signal.reason


def _write_north_star_trades(path: Path) -> None:
    rows = [
        ["1", "MSFT", "LONG", "2026-04-30 09:00", "2026-04-30 09:30", "30", "100", "98", "100", "2", "-2.00", "-2.00", "97", "110", "stop_loss", "64", "LOSS"],
        ["2", "MU", "LONG", "2026-04-30 10:00", "2026-04-30 10:30", "30", "100", "99", "100", "2", "-1.00", "-1.00", "97", "110", "micro_invalidation", "65", "LOSS"],
        ["3", "BABA", "LONG", "2026-04-30 11:00", "2026-04-30 11:05", "5", "100", "100.01", "100", "2", "0.01", "0.01", "97", "110", "structure_invalidation", "70", "WIN"],
        ["4", "CRWV", "LONG", "2026-04-30 12:00", "2026-04-30 12:30", "30", "100", "97", "100", "2", "-3.00", "-3.00", "97", "110", "stop_loss", "61", "LOSS"],
        ["5", "INTC", "LONG", "2026-04-30 13:00", "2026-04-30 14:00", "60", "100", "103", "100", "2", "3.00", "3.00", "97", "110", "take_profit", "75", "WIN"],
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(trade_logger_module.HEADERS)
        writer.writerows(rows)


def test_north_star_guard_blocks_marginal_recovery_entries() -> None:
    cfg = build_config()
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    cfg.trading.north_star_min_trades = 5
    cfg.trading.north_star_lookback_trades = 5
    cfg.trading.north_star_target_quality_win_rate = 0.70
    cfg.trading.north_star_recovery_min_long_score = 72.0
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0)])
    signal = SimpleNamespace(
        action="LONG",
        score=66.0,
        price=100.0,
        reason="marginal recovery setup",
        flat_reason="",
        expectancy={"probability": 0.57, "expected_r": 0.12, "uncertainty": 0.40},
        thesis={"conviction_score": 66.0, "conviction_entry": {}},
    )
    order = OrderRequest("BTC", "LONG", 100.0, 1.0, 100.0, 95.0, 115.0)
    agent._last_signals["BTC"] = {
        "action": "LONG",
        "score": 66.0,
        "expectancy_probability": 0.57,
        "expectancy_uncertainty": 0.40,
    }
    original_trades_csv = agent_module.TRADES_CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            agent_module.TRADES_CSV = Path(tmpdir) / "trades_log.csv"
            _write_north_star_trades(agent_module.TRADES_CSV)
            allowed = agent._apply_north_star_guard_to_order("BTC", signal, order, portfolio_usd=10000.0)
        finally:
            agent_module.TRADES_CSV = original_trades_csv

    assert allowed is False
    assert signal.action == "FLAT"
    assert "north-star" in signal.reason


def test_north_star_guard_allows_event_starter_but_trims_size() -> None:
    cfg = build_config()
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    cfg.trading.min_trade_usd = 100.0
    cfg.trading.north_star_min_trades = 5
    cfg.trading.north_star_lookback_trades = 5
    cfg.trading.north_star_event_size_multiplier = 0.55
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0)])
    signal = SimpleNamespace(
        action="LONG",
        score=64.0,
        price=100.0,
        reason="small pre-event starter",
        flat_reason="",
        expectancy={"probability": 0.56, "expected_r": 0.15, "uncertainty": 0.40},
        thesis={
            "conviction_score": 64.0,
            "conviction_entry": {"active": True, "event_conviction": True},
        },
    )
    order = OrderRequest("GOOGL", "LONG", 220.0, 2.2, 100.0, 95.0, 115.0)
    agent._last_signals["GOOGL"] = {
        "action": "LONG",
        "score": 64.0,
        "instrument_type": "equity",
        "news_event_score": 4.0,
        "conviction_entry_event": True,
        "expectancy_probability": 0.56,
        "expectancy_uncertainty": 0.40,
    }
    original_trades_csv = agent_module.TRADES_CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            agent_module.TRADES_CSV = Path(tmpdir) / "trades_log.csv"
            _write_north_star_trades(agent_module.TRADES_CSV)
            allowed = agent._apply_north_star_guard_to_order(
                "GOOGL",
                signal,
                order,
                portfolio_usd=10000.0,
                event_starter=True,
            )
        finally:
            agent_module.TRADES_CSV = original_trades_csv

    assert allowed is True
    assert round(order.size_usd, 2) == 121.0
    assert agent._last_signals["GOOGL"]["north_star_guard"]["active"] is True


def test_north_star_guard_cancels_stale_pending_limit_before_poll() -> None:
    cfg = build_config()
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    cfg.trading.north_star_min_trades = 5
    cfg.trading.north_star_lookback_trades = 5
    cfg.trading.north_star_target_quality_win_rate = 0.70
    cfg.trading.north_star_recovery_min_long_score = 72.0
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0)])
    result = agent._place_limit_order(
        "BTC",
        "LONG",
        100.0,
        200.0,
        95.0,
        115.0,
        66.0,
        reason="initial_limit",
        entry_context={
            "expectancy": {"probability": 0.57, "expected_r": 0.12, "uncertainty": 0.40},
            "thesis": {"conviction_score": 66.0, "conviction_entry": {}},
        },
    )
    assert result.get("pending") is True
    original_trades_csv = agent_module.TRADES_CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            agent_module.TRADES_CSV = Path(tmpdir) / "trades_log.csv"
            _write_north_star_trades(agent_module.TRADES_CSV)
            agent._enforce_north_star_on_pending_limits({"BTC": 101.0}, 10000.0)
        finally:
            agent_module.TRADES_CSV = original_trades_csv

    assert not agent.order_mgr.has_pending("BTC")


def test_north_star_guard_resizes_event_pending_limit_before_poll() -> None:
    cfg = build_config()
    cfg.trading.decision_dataset_enabled = False
    cfg.trading.feature_store_enabled = False
    cfg.trading.min_trade_usd = 100.0
    cfg.trading.north_star_min_trades = 5
    cfg.trading.north_star_lookback_trades = 5
    cfg.trading.north_star_event_size_multiplier = 0.55
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=10000.0)])
    result = agent._place_limit_order(
        "BTC",
        "LONG",
        100.0,
        220.0,
        95.0,
        115.0,
        64.0,
        reason="starter_basket",
        entry_context={
            "conviction_entry_event": True,
            "expectancy": {"probability": 0.56, "expected_r": 0.15, "uncertainty": 0.40},
            "thesis": {
                "conviction_score": 64.0,
                "conviction_entry": {"active": True, "event_conviction": True},
            },
        },
    )
    assert result.get("pending") is True
    original_trades_csv = agent_module.TRADES_CSV
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            agent_module.TRADES_CSV = Path(tmpdir) / "trades_log.csv"
            _write_north_star_trades(agent_module.TRADES_CSV)
            agent._enforce_north_star_on_pending_limits({"BTC": 101.0}, 10000.0)
        finally:
            agent_module.TRADES_CSV = original_trades_csv

    pending = agent.order_mgr.pending_orders.get("BTC")
    assert pending is not None
    assert pending.reason == "north_star_resize"
    assert round(pending.size_usd, 2) == 121.0
    assert pending.metadata.get("north_star_resized") is True


def test_dashboard_snapshot_includes_proactive_trader_report() -> None:
    snapshot = build_dashboard_snapshot(
        {"signals": {}, "positions": [], "config": {}, "cycle_number": 1},
        [],
        proactive_trader_report={
            "summary": {"active_thesis_count": 2},
            "morning_scout_book": {"summary": {"starter_count": 1}},
        },
    )
    assert snapshot["proactive_trader_report"]["summary"]["active_thesis_count"] == 2


def test_dashboard_snapshot_surfaces_exact_next_setup_blocker() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "cycle_number": 109,
            "positions": [],
            "signals": {
                "XAU": {
                    "action": "FLAT",
                    "asset_state": "OBSERVING",
                    "asset_state_label": "Observing",
                    "execution_mode": "tradable",
                    "market_map_bias": "BEARISH",
                    "market_map_block_shorts": True,
                    "live_price": 4675.50,
                    "price": 4675.50,
                    "score": 23.16,
                    "confidence": "HIGH",
                    "flat_reason": (
                        "SHORT blocked by nearby demand/support 4,675.00 (0.01% away) · "
                        "Score 23 — needs ≤35 for SHORT"
                    ),
                }
            },
            "mode": "dry_run",
            "config": {
                "coins": ["XAU"],
                "analysis_coins": ["XAU"],
                "instrument_types": {"XAU": "index"},
                "asset_categories": {"XAU": "indices_macro"},
            },
        },
        trades=[],
        control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
        market_map={
            "date": "2026-04-23",
            "updated_at": "2026-04-23 16:00:00",
            "coins": {
                "XAU": {
                    "bias": "BEARISH",
                    "daily_close_short_below": [4626.75, 4653.65],
                    "supports": [4593.65, 4626.75, 4653.65],
                    "resistances": [4700.00],
                    "summary": "auto bearish map; price is sitting in mapped demand",
                }
            },
        },
        server_timestamp="2026-04-23 16:01:00",
    )
    lead = snapshot["action_board"]["lead"]
    assert lead["coin"] == "XAU"
    assert lead["status"] == "WAIT_BREAKDOWN"
    assert "Not short yet" in lead["next_setup_reason"]
    assert "4,653.65" in lead["next_setup_reason"]
    assert "nearby demand/support 4,675.00" in lead["next_setup_reason"]


def test_dashboard_snapshot_ignores_opposite_direction_threshold_in_blocker() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "cycle_number": 110,
            "positions": [],
            "signals": {
                "UNI": {
                    "action": "FLAT",
                    "execution_mode": "tradable",
                    "market_map_bias": "BEARISH",
                    "market_map_block_shorts": True,
                    "live_price": 3.27,
                    "price": 3.27,
                    "score": 38.0,
                    "confidence": "MEDIUM",
                    "flat_reason": "Score 38 — needs ≥65 for LONG · Regime: ABSORPTION (no clear direction)",
                }
            },
            "mode": "dry_run",
            "config": {
                "coins": ["UNI"],
                "analysis_coins": ["UNI"],
                "instrument_types": {"UNI": "crypto"},
            },
        },
        trades=[],
        control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
        market_map={
            "date": "2026-04-23",
            "updated_at": "2026-04-23 16:10:00",
            "coins": {
                "UNI": {
                    "bias": "BEARISH",
                    "daily_close_short_below": [3.26],
                    "supports": [3.26],
                    "resistances": [3.40],
                }
            },
        },
        server_timestamp="2026-04-23 16:11:00",
    )
    lead = snapshot["action_board"]["lead"]
    assert lead["coin"] == "UNI"
    assert "needs ≥65 for LONG" not in lead["next_setup_reason"]
    assert "ABSORPTION" in lead["next_setup_reason"]


def test_dashboard_snapshot_normalizes_directional_breakdown_wording() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "cycle_number": 111,
            "positions": [],
            "signals": {
                "WLFI": {
                    "action": "FLAT",
                    "execution_mode": "tradable",
                    "market_map_bias": "BEARISH",
                    "market_map_block_shorts": True,
                    "live_price": 0.08,
                    "price": 0.08,
                    "score": 42.0,
                    "confidence": "HIGH",
                    "flat_reason": "LONG blocked — market is breaking down through key support (PERSISTENT_BEARISH_BREAKDOWN)",
                }
            },
            "mode": "dry_run",
            "config": {
                "coins": ["WLFI"],
                "analysis_coins": ["WLFI"],
                "instrument_types": {"WLFI": "crypto"},
            },
        },
        trades=[],
        control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
        market_map={
            "date": "2026-04-23",
            "updated_at": "2026-04-23 16:12:00",
            "coins": {
                "WLFI": {
                    "bias": "BEARISH",
                    "daily_close_short_below": [0.08],
                    "supports": [0.08],
                    "resistances": [0.09],
                }
            },
        },
        server_timestamp="2026-04-23 16:13:00",
    )
    lead = snapshot["action_board"]["lead"]
    assert lead["coin"] == "WLFI"
    assert "LONG blocked" not in lead["next_setup_reason"]
    assert "Breakdown already active" in lead["next_setup_reason"]


def test_dashboard_snapshot_includes_trade_logic_and_learning_summary() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "cycle_number": 91,
            "positions": [],
            "signals": {},
            "mode": "dry_run",
        },
        trades=[{
            "trade_id": "41",
            "coin": "BTC",
            "direction": "LONG",
            "opened_at": "2026-04-07 09:00",
            "closed_at": "2026-04-07 12:00",
            "entry_price": 70000.0,
            "exit_price": 71400.0,
            "size_usd": 100.0,
            "pnl_usd": 20.0,
            "pnl_pct": 20.0,
            "exit_reason": "take_profit",
        }],
        control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
        market_map={"coins": {"BTC": {"bias": "BULLISH", "source": "AUTO", "auto_generated": True}}},
        trade_dataset_records=[{
            "trade_id": "41",
            "coin": "BTC",
            "direction": "LONG",
            "pnl_usd": 20.0,
            "exit_reason": "take_profit",
            "thesis": {"summary": "support defense long stayed intact"},
            "trade_plan": {"risk_reward_ratio": 2.4},
            "entry_context": {
                "reason": "support-defense long with breakout pressure",
                "market_map_summary": "auto bullish map; intraday reclaim in play",
                "orderbook_interaction": "AT_SUPPORT",
                "orderbook_breakout_state": "PROBING_BULLISH_BREAKOUT",
            },
            "exit_context": {
                "thesis_summary": "follow-through held into target",
                "orderbook_interaction": "ABOVE_BREAKOUT",
            },
        }],
        server_timestamp="2026-04-07 12:05:00",
    )
    trade = snapshot["trades"][0]
    assert "support-defense long" in trade["open_logic"].lower()
    assert "target was reached" in trade["close_logic"].lower()
    assert snapshot["learning_summary"]["count"] == 1
    assert snapshot["learning_summary"]["latest"]["coin"] == "BTC"
    assert snapshot["learning_summary"]["latest"]["lesson"]


def test_dashboard_snapshot_includes_asset_dossiers_and_referee_reports() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "cycle_number": 99,
            "positions": [],
            "signals": {
                "BTC": {
                    "action": "LONG",
                    "score": 74.0,
                    "confidence": "HIGH",
                    "execution_mode": "tradable",
                    "asset_state": "WAITING_CONFIRMATION",
                    "next_unblock_reason": "Need one more clean reclaim close.",
                    "llm_referee": {"verdict": "SUPPORT", "summary": "Momentum and structure agree."},
                }
            },
            "mode": "dry_run",
            "config": {"coins": ["BTC"]},
        },
        trades=[],
        asset_dossiers={
            "summary": {"focus_assets": ["BTC"]},
            "assets": {
                "BTC": {
                    "coin": "BTC",
                    "dossier": {"current_read": "Momentum and structure agree."},
                }
            },
        },
        missed_move_report={"summary": {"missed_win_count": 3}},
        llm_referee_report={"enabled": True, "verdicts": {"BTC": {"verdict": "SUPPORT"}}},
        server_timestamp="2026-04-21 21:05:00",
    )
    assert snapshot["asset_dossiers"]["summary"]["focus_assets"] == ["BTC"]
    assert snapshot["missed_move_report"]["summary"]["missed_win_count"] == 3
    assert snapshot["llm_referee_report"]["verdicts"]["BTC"]["verdict"] == "SUPPORT"


def test_dashboard_action_board_uses_asset_state_and_next_unblock_reason() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "last_cycle": "2026-04-16 10:30:00",
            "cycle_number": 120,
            "positions": [],
            "signals": {
                "GOOGL": {
                    "action": "LONG",
                    "score": 67.0,
                    "confidence": "HIGH",
                    "execution_mode": "tradable",
                    "asset_state": "WAITING_CONFIRMATION",
                    "asset_state_label": "Waiting confirmation",
                    "next_unblock_reason": "Need 1 more confirming cycle before the entry is allowed.",
                    "decision_reason": "Long thesis is live but still confirming.",
                    "market_map_bias": "BULLISH",
                    "live_price": 336.61,
                }
            },
            "mode": "dry_run",
            "config": {"coins": ["GOOGL"]},
        },
        trades=[],
        server_timestamp="2026-04-16 10:31:00",
    )
    lead = snapshot["action_board"]["lead"]
    assert lead["coin"] == "GOOGL"
    assert lead["label"] == "Waiting confirmation"
    assert "confirming cycle" in lead["execution_note"].lower()


def test_dashboard_action_board_shows_reclaim_watch_not_wait_reclaim_after_confirmation() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "last_cycle": "2026-04-21 07:57:30",
            "cycle_number": 5889,
            "positions": [],
            "signals": {
                "AMZN": {
                    "action": "FLAT",
                    "score": 41.5,
                    "confidence": "LOW",
                    "execution_mode": "tradable",
                    "asset_state": "OBSERVING",
                    "asset_state_label": "Observing",
                    "decision_reason": "Breakout is confirmed, but the long still lacks clean continuation.",
                    "flat_reason": "Equity spot: prior reclaim slipped back below the trigger; waiting for price to hold above reclaim again",
                    "market_map_bias": "BULLISH",
                    "market_map_block_longs": True,
                    "market_map_reclaim_confirmed": True,
                    "market_map_live_reclaim": False,
                    "market_map_reclaim_lost": True,
                    "market_map_summary": "auto bullish map; daily reclaim was confirmed, but live price slipped back below reclaim; price is sitting in mapped supply; price is pressing mapped resistance",
                    "market_map_nearest_support": 250.0,
                    "market_map_nearest_resistance": 253.36,
                    "orderbook_breakout_state": "CONFIRMED_BULLISH_BREAKOUT",
                    "orderbook_intracycle_breakout_state": "PERSISTENT_BULLISH_BREAKOUT",
                    "live_price": 253.03,
                }
            },
            "mode": "dry_run",
            "config": {"coins": ["AMZN"]},
        },
        market_map={
            "coins": {
                "AMZN": {
                    "bias": "BULLISH",
                    "supports": [250.0],
                    "resistances": [253.36, 256.5],
                    "daily_close_long_above": [253.36],
                }
            }
        },
        trades=[],
        server_timestamp="2026-04-21 07:57:33",
    )
    lead = snapshot["action_board"]["lead"]
    assert lead["coin"] == "AMZN"
    assert lead["status"] == "WATCH_LONG"
    assert lead["label"] == "Bullish watch"
    assert lead["entry_status"] == "Live 253.03; reclaim 253.36; gap -0.33 (-0.13%)"
    assert lead["trigger"] == "Reclaim 253.36 (+0.33 / +0.13%)"
    assert lead["probability_pct"] == 56
    assert lead["probability_label"] == "Reclaim odds"
    assert lead["probability_text"] == "Reclaim odds 56%"
    assert "prior reclaim already printed" in lead["probability_detail"].lower()
    assert lead["risk"] == "Lose 250.00 (-3.03 / -1.20%)"
    assert lead["invalidation"] == "Invalid below 250.00 (-3.03 / -1.20%)"
    assert "slipped back below the trigger" in lead["execution_note"].lower()


def test_dashboard_action_board_uses_major_catalyst_watch_label_and_unblock_reason() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "last_cycle": "2026-04-21 07:57:30",
            "cycle_number": 5890,
            "positions": [],
            "signals": {
                "AMZN": {
                    "action": "FLAT",
                    "score": 41.5,
                    "confidence": "LOW",
                    "execution_mode": "observation_only",
                    "instrument_type": "equity",
                    "asset_state": "MAJOR_CATALYST_WATCH",
                    "asset_state_label": "Major catalyst watch",
                    "next_unblock_reason": "Major catalyst watch: hold back above 253.36 and keep the reclaim to unlock the long.",
                    "decision_reason": "Anthropic/AWS catalyst is strong, but the retake still has to confirm.",
                    "flat_reason": "Equity spot: prior reclaim slipped back below the trigger; waiting for price to hold above reclaim again",
                    "market_map_bias": "BULLISH",
                    "market_map_reclaim_confirmed": True,
                    "market_map_live_reclaim": False,
                    "market_map_reclaim_lost": True,
                    "market_map_summary": "auto bullish map; daily reclaim was confirmed, but live price slipped back below reclaim",
                    "market_map_nearest_support": 250.0,
                    "market_map_nearest_resistance": 253.36,
                    "news_catalyst_score": 3.75,
                    "news_catalyst_summary": "platform anchor + partner attached + demand commitment",
                    "live_price": 253.03,
                }
            },
            "mode": "dry_run",
            "config": {"coins": ["AMZN"]},
        },
        market_map={
            "coins": {
                "AMZN": {
                    "bias": "BULLISH",
                    "supports": [250.0],
                    "resistances": [253.36, 256.5],
                    "daily_close_long_above": [253.36],
                }
            }
        },
        trades=[],
        server_timestamp="2026-04-21 07:57:34",
    )
    lead = snapshot["action_board"]["lead"]
    assert lead["coin"] == "AMZN"
    assert lead["status"] == "WATCH_LONG"
    assert lead["label"] == "Major catalyst watch"
    assert lead["entry_status"] == "Live 253.03; reclaim 253.36; gap -0.33 (-0.13%)"
    assert lead["trigger"] == "Reclaim 253.36 (+0.33 / +0.13%)"
    assert lead["probability_pct"] == 56
    assert lead["probability_label"] == "Reclaim odds"
    assert "major catalyst still supports the move" in lead["probability_detail"].lower()
    assert lead["risk"] == "Lose 250.00 (-3.03 / -1.20%)"
    assert lead["invalidation"] == "Invalid below 250.00 (-3.03 / -1.20%)"
    assert "hold back above 253.36" in lead["execution_note"].lower()
    assert lead["mode_badge"] == "EXEC"
    assert lead["mode_label"] == "EXECUTABLE"


def test_dashboard_action_board_builds_friction_stack_catalyst_rail_and_lead_reason() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "last_cycle": "2026-04-23 10:15:00",
            "cycle_number": 6012,
            "positions": [],
            "signals": {
                "AMZN": {
                    "action": "FLAT",
                    "score": 43.5,
                    "confidence": "MEDIUM",
                    "execution_mode": "tradable",
                    "instrument_type": "equity",
                    "asset_state": "MAJOR_CATALYST_WATCH",
                    "asset_state_label": "Major catalyst watch",
                    "next_unblock_reason": "Hold back above 253.36 and keep the reclaim to unlock the long.",
                    "decision_reason": "Anthropic/AWS catalyst is strong, but the retake still has to confirm.",
                    "flat_reason": "Equity spot: prior reclaim slipped back below the trigger; waiting for price to hold above reclaim again",
                    "market_map_bias": "BULLISH",
                    "market_map_block_longs": True,
                    "market_map_reclaim_confirmed": True,
                    "market_map_live_reclaim": False,
                    "market_map_reclaim_lost": True,
                    "market_map_summary": "auto bullish map; daily reclaim was confirmed, but live price slipped back below reclaim; price is pressing mapped resistance",
                    "market_map_nearest_support": 250.0,
                    "market_map_nearest_resistance": 253.36,
                    "thesis_quality": "HIGH",
                    "thesis_summary": "Bullish daily structure is intact, but price still has to reclaim resistance cleanly.",
                    "data_reliability_quality": "HIGH",
                    "data_reliability_summary": "data quality is strong enough to trust the setup",
                    "execution_quality_score": 61.0,
                    "execution_quality_summary": "execution still wants a cleaner retake before buying",
                    "orderbook_breakout_state": "CONFIRMED_BULLISH_BREAKOUT",
                    "news_catalyst_score": 3.75,
                    "news_catalyst_summary": "platform anchor + partner attached + demand commitment",
                    "news_event_score": 3.75,
                    "news_event_summary": "earnings event + pre-event setup",
                    "news_event_tags": ["earnings_event", "pre_event_setup"],
                    "conviction_entry_active": True,
                    "conviction_entry_style": "EVENT_STARTER",
                    "conviction_entry_reason": "Pre-event starter long allowed before full confirmation.",
                    "conviction_entry_size_multiplier": 0.35,
                    "conviction_entry_event": True,
                    "narrative_event_name": "AMZN earnings",
                    "narrative_minutes_to_event": 55,
                    "narrative_event_risk_active": True,
                    "live_price": 253.03,
                }
            },
            "mode": "dry_run",
            "config": {
                "coins": ["AMZN"],
                "instrument_types": {"AMZN": "equity"},
                "asset_categories": {"AMZN": "mag7"},
            },
        },
        market_map={
            "coins": {
                "AMZN": {
                    "bias": "BULLISH",
                    "supports": [250.0],
                    "resistances": [253.36, 256.5],
                    "daily_close_long_above": [253.36],
                }
            }
        },
        trades=[],
        server_timestamp="2026-04-23 10:15:30",
    )
    lead = snapshot["action_board"]["lead"]
    assert lead["coin"] == "AMZN"
    friction_stack = lead["friction_stack"]
    assert [item["label"] for item in friction_stack] == ["Structure", "Flow", "Data", "Execution"]
    assert any(item["status"] == "clear" for item in friction_stack)
    assert any(item["status"] == "wait" for item in friction_stack)
    catalyst_rail = lead["catalyst_rail"]
    assert catalyst_rail
    assert catalyst_rail[0]["label"] == "Catalyst"
    assert any(item["label"] == "Event" for item in catalyst_rail)
    assert lead["news_event_score"] == 3.75
    assert lead["conviction_entry_active"] is True
    assert lead["conviction_entry_style"] == "EVENT_STARTER"
    assert lead["conviction_entry_size_multiplier"] == 0.35
    assert "catalyst" in lead["why_this_lead"].lower()
    assert "reclaim" in lead["why_this_lead"].lower()


def test_dashboard_action_board_calibrates_reclaim_odds_from_decision_history() -> None:
    decision_rows: list[dict] = []
    cycle = 1
    for success in [True, True, True, False, True, False, True, True, True, False, True, True]:
        decision_rows.append(
            {
                "coin": "AMZN",
                "cycle_number": cycle,
                "stage": "flat_no_trade",
                "candidate_action": "LONG",
                "final_action": "FLAT",
                "blocked": True,
                "executed": False,
                "pending_limit": False,
                "asset_state": "ARMED",
                "signal_snapshot": {
                    "instrument_type": "equity",
                    "market_map_bias": "BULLISH",
                    "market_map_reclaim_confirmed": True,
                    "market_map_live_reclaim": False,
                    "market_map_reclaim_lost": True,
                    "market_map_nearest_resistance": 100.0,
                    "market_map_nearest_support": 96.0,
                    "daily_close_long_above": [100.0],
                    "live_price": 99.82,
                    "flat_reason": "waiting for price to hold above reclaim again",
                    "news_catalyst_score": 3.6,
                },
            }
        )
        cycle += 1
        decision_rows.append(
            {
                "coin": "AMZN",
                "cycle_number": cycle,
                "stage": "signal_streak_wait" if success else "flat_no_trade",
                "candidate_action": "LONG",
                "final_action": "LONG" if success else "FLAT",
                "blocked": not success,
                "executed": False,
                "pending_limit": False,
                "asset_state": "WAITING_CONFIRMATION" if success else "OBSERVING",
                "signal_snapshot": {
                    "instrument_type": "equity",
                    "market_map_bias": "BULLISH",
                    "market_map_reclaim_confirmed": True,
                    "market_map_live_reclaim": success,
                    "market_map_reclaim_lost": not success,
                    "market_map_nearest_resistance": 100.0,
                    "daily_close_long_above": [100.0],
                    "live_price": 100.12 if success else 99.4,
                    "news_catalyst_score": 3.6,
                },
            }
        )
        cycle += 1

    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "last_cycle": "2026-04-21 07:57:30",
            "cycle_number": 6000,
            "positions": [],
            "signals": {
                "AMZN": {
                    "action": "FLAT",
                    "score": 41.5,
                    "confidence": "LOW",
                    "execution_mode": "tradable",
                    "instrument_type": "equity",
                    "asset_state": "MAJOR_CATALYST_WATCH",
                    "asset_state_label": "Major catalyst watch",
                    "next_unblock_reason": "Major catalyst watch: hold back above 253.36 and keep the reclaim to unlock the long.",
                    "decision_reason": "Anthropic/AWS catalyst is strong, but the retake still has to confirm.",
                    "flat_reason": "Equity spot: prior reclaim slipped back below the trigger; waiting for price to hold above reclaim again",
                    "market_map_bias": "BULLISH",
                    "market_map_reclaim_confirmed": True,
                    "market_map_live_reclaim": False,
                    "market_map_reclaim_lost": True,
                    "market_map_summary": "auto bullish map; daily reclaim was confirmed, but live price slipped back below reclaim",
                    "market_map_nearest_support": 250.0,
                    "market_map_nearest_resistance": 253.36,
                    "news_catalyst_score": 3.75,
                    "live_price": 253.03,
                }
            },
            "mode": "dry_run",
            "config": {"coins": ["AMZN"], "instrument_types": {"AMZN": "equity"}},
        },
        market_map={
            "coins": {
                "AMZN": {
                    "bias": "BULLISH",
                    "supports": [250.0],
                    "resistances": [253.36, 256.5],
                    "daily_close_long_above": [253.36],
                }
            }
        },
        trades=[],
        decision_dataset_records=decision_rows,
        server_timestamp="2026-04-21 07:57:34",
    )
    lead = snapshot["action_board"]["lead"]
    assert lead["coin"] == "AMZN"
    assert lead["probability_source"] == "calibrated"
    assert lead["probability_empirical_samples"] == 12
    assert lead["probability_empirical_pct"] >= 90
    assert lead["probability_pct"] >= 65
    assert "similar reclaim watches" in lead["probability_detail"].lower()


def test_dashboard_action_board_surfaces_scout_universe_summary() -> None:
    snapshot = build_dashboard_snapshot(
        state={
            "status": "online",
            "last_cycle": "2026-04-17 09:40:00",
            "cycle_number": 121,
            "positions": [],
            "signals": {
                "BTC": {
                    "action": "FLAT",
                    "score": 52.0,
                    "confidence": "LOW",
                    "execution_mode": "tradable",
                    "asset_state": "PENDING_ENTRY",
                    "asset_state_label": "Working order",
                    "next_unblock_reason": "Resting bid is live while the bot waits for a fill.",
                    "decision_reason": "Passive breakout retest order is working.",
                }
            },
            "mode": "dry_run",
            "config": {
                "coins": ["BTC"],
                "analysis_coins": ["BTC", "DOGE", "AVAX", "LINK"],
                "dynamic_analysis_coins": ["DOGE", "AVAX", "LINK"],
                "dynamic_market_cap_min_usd": 1_000_000_000.0,
            },
        },
        trades=[],
        server_timestamp="2026-04-17 09:41:00",
    )
    summary = snapshot["action_board"]["summary"]
    assert summary["pending_count"] == 1
    assert summary["scout_count"] == 3
    assert summary["scout_preview"] == ["DOGE", "AVAX", "LINK"]


def test_dashboard_snapshot_canonicalizes_inactive_control_and_empty_review_shape() -> None:
    snapshot = build_dashboard_snapshot(
        {
            "status": "running",
            "last_cycle": "2026-04-14 16:37:20",
            "cycle_number": 3865,
            "portfolio_usd": 9987.37,
            "available_usd": 3994.95,
            "positions": [],
            "signals": {},
            "pending_orders": [],
            "sentiment": {},
            "mode": "dry_run",
        },
        [
            {
                "trade_id": "57",
                "coin": "TAO",
                "direction": "LONG",
                "pnl_usd": "3.29",
                "exit_price": "100.0",
            }
        ],
        control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": "2026-04-01 07:24:06"}},
        market_map={"updated_at": "2026-04-14 16:37:20", "coins": {}},
        trade_reviews={"updated_at": "2026-04-14 16:37:20", "reviews": {}},
        trade_dataset_records=[],
        server_timestamp="2026-04-14 16:37:20",
    )

    assert snapshot["control"]["kill"]["acknowledged_at"] is None
    assert snapshot["review_summary"]["updated_at"] is None
    assert snapshot["learning_summary"]["latest"]["pnl_usd"] == 3.29


def test_dashboard_market_map_and_review_endpoints_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        original_control = dashboard_module.CONTROL
        original_kill = dashboard_module.KILL
        original_state = dashboard_module.STATE
        original_log = dashboard_module.LOG
        original_snapshot = dashboard_module.SNAPSHOT
        original_market_map = dashboard_module.MARKET_MAP
        original_reviews = dashboard_module.REVIEWS
        original_remote = dict(dashboard_module._remote_state)
        original_market_map_path = market_map_module.DAILY_MARKET_MAP_JSON
        original_review_path = trade_review_module.TRADE_REVIEWS_JSON
        original_log_file = trade_logger_module.LOG_FILE
        try:
            dashboard_module.CONTROL = temp / "control.json"
            dashboard_module.KILL = temp / "KILL"
            dashboard_module.STATE = temp / "state.json"
            dashboard_module.LOG = temp / "trades_log.csv"
            dashboard_module.SNAPSHOT = temp / "dashboard_snapshot.json"
            dashboard_module.MARKET_MAP = temp / "daily_market_map.json"
            dashboard_module.REVIEWS = temp / "trade_reviews.json"
            dashboard_module._remote_state = {
                "snapshot": build_dashboard_snapshot(
                    state={"status": "online", "cycle_number": 5, "positions": [], "signals": {}, "mode": "dry_run"},
                    trades=[{
                        "trade_id": "1",
                        "coin": "BTC",
                        "direction": "LONG",
                        "opened_at": "2026-04-06 09:00",
                        "closed_at": "2026-04-06 10:00",
                        "entry_price": 100.0,
                        "exit_price": 105.0,
                        "size_usd": 100.0,
                        "pnl_usd": 5.0,
                        "pnl_pct": 0.05,
                        "exit_reason": "take_profit",
                    }],
                    control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
                    server_timestamp="2026-04-06 10:05:00",
                )
            }
            market_map_module.DAILY_MARKET_MAP_JSON = dashboard_module.MARKET_MAP
            trade_review_module.TRADE_REVIEWS_JSON = dashboard_module.REVIEWS
            trade_logger_module.LOG_FILE = dashboard_module.LOG
            dashboard_module.LOG.write_text(
                ",".join(trade_logger_module.HEADERS) + "\n"
                + "1,BTC,LONG,2026-04-06 09:00,2026-04-06 10:00,60,100,105,100,2,5,0.05,95,110,take_profit,72,WIN\n"
            )

            client = dashboard_module.app.test_client()
            map_resp = client.post("/api/market-map", json={
                "coin": "BTC",
                "bias": "BULLISH",
                "confidence": "HIGH",
                "supports": "60000, 68500",
                "resistances": "71500, 72500",
                "daily_close_long_above": "71500, 72500",
                "demand_zone": {"low": 60000, "high": 61000},
                "notes": "Operator map says reclaim higher and respect demand below.",
            })
            assert map_resp.status_code == 200

            review_resp = client.post("/api/reviews", json={
                "trade_id": "1",
                "coin": "BTC",
                "direction": "LONG",
                "verdict": "GOOD_TRADE",
                "thesis_quality": "STRONG",
                "execution_quality": "GOOD",
                "notes": "Followed mapped reclaim and closed well.",
            })
            assert review_resp.status_code == 200

            state = client.get("/api/state").get_json()
            assert state["market_map"]["coins"]["BTC"]["bias"] == "BULLISH"
            assert state["trade_reviews"]["reviews"]["1"]["verdict"] == "GOOD_TRADE"
            assert state["review_summary"]["count"] == 1
            assert state["trades"][0]["review"]["verdict"] == "GOOD_TRADE"
        finally:
            dashboard_module.CONTROL = original_control
            dashboard_module.KILL = original_kill
            dashboard_module.STATE = original_state
            dashboard_module.LOG = original_log
            dashboard_module.SNAPSHOT = original_snapshot
            dashboard_module.MARKET_MAP = original_market_map
            dashboard_module.REVIEWS = original_reviews
            dashboard_module._remote_state = original_remote
            market_map_module.DAILY_MARKET_MAP_JSON = original_market_map_path
            trade_review_module.TRADE_REVIEWS_JSON = original_review_path
            trade_logger_module.LOG_FILE = original_log_file


def test_hosted_state_sync_can_publish_snapshot_to_git_branch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        remote = temp / "origin.git"
        subprocess_ok = __import__("subprocess")
        subprocess_ok.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)

        original_repo_dir = hosted_state_sync_module.DASHBOARD_STATE_SYNC_REPO
        original_url = os.environ.get("DASHBOARD_STATE_GIT_URL")
        original_branch = os.environ.get("DASHBOARD_STATE_GIT_BRANCH")
        original_tag = os.environ.get("DASHBOARD_STATE_GIT_TAG")
        original_enabled = os.environ.get("DASHBOARD_STATE_GIT_SYNC_ENABLED")
        try:
            hosted_state_sync_module.DASHBOARD_STATE_SYNC_REPO = temp / ".dashboard_state_sync"
            os.environ["DASHBOARD_STATE_GIT_URL"] = str(remote)
            os.environ["DASHBOARD_STATE_GIT_BRANCH"] = "codex/dashboard-state-test"
            os.environ["DASHBOARD_STATE_GIT_TAG"] = "dashboard-state-test"
            os.environ["DASHBOARD_STATE_GIT_SYNC_ENABLED"] = "1"

            ok = hosted_state_sync_module.publish_snapshot(
                {
                    "state": {"cycle_number": 321, "status": "running"},
                    "server_time": "2026-04-08 10:00:00",
                    "trades": [],
                    "control": {"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
                },
                state={"cycle_number": 321, "status": "running"},
                trades=[],
                control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
                market_map={"coins": {"BTC": {"bias": "BULLISH"}}},
                trade_reviews={"reviews": {}},
            )
            assert ok is True

            checkout = temp / "checkout"
            subprocess_ok.run(["git", "clone", "--depth", "1", "--branch", "codex/dashboard-state-test", str(remote), str(checkout)], check=True, capture_output=True, text=True)
            saved = json.loads((checkout / "dashboard" / "dashboard_snapshot.json").read_text())
            assert saved["state"]["cycle_number"] == 321
            tags = subprocess_ok.run(["git", "ls-remote", "--tags", str(remote), "dashboard-state-test"], check=True, capture_output=True, text=True)
            assert "dashboard-state-test" in tags.stdout
        finally:
            hosted_state_sync_module.DASHBOARD_STATE_SYNC_REPO = original_repo_dir
            if original_url is None:
                os.environ.pop("DASHBOARD_STATE_GIT_URL", None)
            else:
                os.environ["DASHBOARD_STATE_GIT_URL"] = original_url
            if original_branch is None:
                os.environ.pop("DASHBOARD_STATE_GIT_BRANCH", None)
            else:
                os.environ["DASHBOARD_STATE_GIT_BRANCH"] = original_branch
            if original_tag is None:
                os.environ.pop("DASHBOARD_STATE_GIT_TAG", None)
            else:
                os.environ["DASHBOARD_STATE_GIT_TAG"] = original_tag
            if original_enabled is None:
                os.environ.pop("DASHBOARD_STATE_GIT_SYNC_ENABLED", None)
            else:
                os.environ["DASHBOARD_STATE_GIT_SYNC_ENABLED"] = original_enabled


def test_hosted_state_sync_force_updates_generated_branch() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        remote = temp / "origin.git"
        subprocess_ok = __import__("subprocess")
        subprocess_ok.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)

        original_repo_dir = hosted_state_sync_module.DASHBOARD_STATE_SYNC_REPO
        original_url = os.environ.get("DASHBOARD_STATE_GIT_URL")
        original_branch = os.environ.get("DASHBOARD_STATE_GIT_BRANCH")
        original_tag = os.environ.get("DASHBOARD_STATE_GIT_TAG")
        original_enabled = os.environ.get("DASHBOARD_STATE_GIT_SYNC_ENABLED")
        try:
            hosted_state_sync_module.DASHBOARD_STATE_SYNC_REPO = temp / ".dashboard_state_sync"
            os.environ["DASHBOARD_STATE_GIT_URL"] = str(remote)
            os.environ["DASHBOARD_STATE_GIT_BRANCH"] = "codex/dashboard-state-force"
            os.environ["DASHBOARD_STATE_GIT_TAG"] = "dashboard-state-force"
            os.environ["DASHBOARD_STATE_GIT_SYNC_ENABLED"] = "1"

            assert hosted_state_sync_module.publish_snapshot(
                {"state": {"cycle_number": 100}, "server_time": "2026-04-14 16:00:00"},
                state={"cycle_number": 100},
                trades=[],
                control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
                market_map={},
                trade_reviews={},
            ) is True

            external = temp / "external"
            subprocess_ok.run(
                ["git", "clone", "--depth", "1", "--branch", "codex/dashboard-state-force", str(remote), str(external)],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess_ok.run(["git", "config", "user.name", "External Writer"], cwd=str(external), check=True, capture_output=True, text=True)
            subprocess_ok.run(["git", "config", "user.email", "external@example.com"], cwd=str(external), check=True, capture_output=True, text=True)
            injected = external / "dashboard" / "dashboard_snapshot.json"
            payload = json.loads(injected.read_text())
            payload["state"]["cycle_number"] = 999
            payload["server_time"] = "2026-04-14 16:01:00"
            injected.write_text(json.dumps(payload, indent=2) + "\n")
            subprocess_ok.run(["git", "add", "dashboard/dashboard_snapshot.json"], cwd=str(external), check=True, capture_output=True, text=True)
            subprocess_ok.run(["git", "commit", "-m", "external drift"], cwd=str(external), check=True, capture_output=True, text=True)
            subprocess_ok.run(["git", "push", "origin", "HEAD:codex/dashboard-state-force"], cwd=str(external), check=True, capture_output=True, text=True)

            assert hosted_state_sync_module.publish_snapshot(
                {"state": {"cycle_number": 101}, "server_time": "2026-04-14 16:02:00"},
                state={"cycle_number": 101},
                trades=[],
                control={"kill": {"active": False, "reason": "", "requested_at": None, "acknowledged_at": None}},
                market_map={},
                trade_reviews={},
            ) is True

            verify = temp / "verify"
            subprocess_ok.run(
                ["git", "clone", "--depth", "1", "--branch", "codex/dashboard-state-force", str(remote), str(verify)],
                check=True,
                capture_output=True,
                text=True,
            )
            saved = json.loads((verify / "dashboard" / "dashboard_snapshot.json").read_text())
            assert saved["state"]["cycle_number"] == 101
        finally:
            hosted_state_sync_module.DASHBOARD_STATE_SYNC_REPO = original_repo_dir
            if original_url is None:
                os.environ.pop("DASHBOARD_STATE_GIT_URL", None)
            else:
                os.environ["DASHBOARD_STATE_GIT_URL"] = original_url
            if original_branch is None:
                os.environ.pop("DASHBOARD_STATE_GIT_BRANCH", None)
            else:
                os.environ["DASHBOARD_STATE_GIT_BRANCH"] = original_branch
            if original_tag is None:
                os.environ.pop("DASHBOARD_STATE_GIT_TAG", None)
            else:
                os.environ["DASHBOARD_STATE_GIT_TAG"] = original_tag
            if original_enabled is None:
                os.environ.pop("DASHBOARD_STATE_GIT_SYNC_ENABLED", None)
            else:
                os.environ["DASHBOARD_STATE_GIT_SYNC_ENABLED"] = original_enabled


def test_dashboard_remote_fallback_still_publishes_git_snapshot() -> None:
    cfg = build_config()
    original_state_json = agent_module.STATE_JSON
    original_trades_csv = agent_module.TRADES_CSV
    original_snapshot_json = agent_module.DASHBOARD_SNAPSHOT_JSON
    original_control_json = agent_module.CONTROL_JSON
    original_build_effective_map = agent_module.market_map.build_effective_market_map
    original_urlopen = urllib.request.urlopen
    original_publish = hosted_state_sync_module.publish_snapshot
    original_dashboard_url = os.environ.get("DASHBOARD_URL")
    original_dashboard_token = os.environ.get("DASHBOARD_TOKEN")
    captured: dict[str, object] = {"published": False, "push_payload": None}

    class _Resp:
        def __init__(self, payload: dict, status: int = 200):
            self._payload = payload
            self.status = status

        def read(self):
            return json.dumps(self._payload).encode()

    with tempfile.TemporaryDirectory() as tmpdir:
        temp = Path(tmpdir)
        try:
            agent_module.STATE_JSON = temp / "state.json"
            agent_module.TRADES_CSV = temp / "trades.csv"
            agent_module.DASHBOARD_SNAPSHOT_JSON = temp / "dashboard_snapshot.json"
            agent_module.CONTROL_JSON = temp / "control.json"
            agent_module.market_map.build_effective_market_map = lambda *args, **kwargs: {
                "coins": {"BTC": {"bias": "BULLISH", "source": "AUTO", "auto_generated": True}}
            }

            def fake_urlopen(req, timeout=0, context=None):
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if url.endswith("/api/push"):
                    if hasattr(req, "data") and req.data:
                        captured["push_payload"] = json.loads(req.data.decode())
                    return _Resp({"ok": True, "fallback": "netlify", "storage": "fallback"})
                if url.endswith("/api/state"):
                    return _Resp({"control": {"kill": {"active": False}}})
                raise AssertionError(f"Unexpected urlopen target: {url}")

            def fake_publish(snapshot, **kwargs):
                captured["published"] = True
                captured["published_snapshot"] = snapshot
                return True

            urllib.request.urlopen = fake_urlopen
            hosted_state_sync_module.publish_snapshot = fake_publish
            os.environ["DASHBOARD_URL"] = "https://example.test"
            os.environ["DASHBOARD_TOKEN"] = "secret"

            agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=1000.0)])
            agent._last_power_status = {"source": "Battery Power", "available": True}
            agent._last_signals = {
                "BTC": {
                    "action": "FLAT",
                    "score": 50.0,
                    "confidence": "LOW",
                    "live_price": 100.0,
                    "analysis_price": 99.0,
                    "decision_reason": "No trade",
                    "thesis_summary": "No trade",
                }
            }
            agent._write_state(
                portfolio_usd=1000.0,
                sentiment={"signal_score": 50.0, "label": "Neutral", "raw_score": 50, "is_extreme": False},
            )
        finally:
            agent_module.STATE_JSON = original_state_json
            agent_module.TRADES_CSV = original_trades_csv
            agent_module.DASHBOARD_SNAPSHOT_JSON = original_snapshot_json
            agent_module.CONTROL_JSON = original_control_json
            agent_module.market_map.build_effective_market_map = original_build_effective_map
            urllib.request.urlopen = original_urlopen
            hosted_state_sync_module.publish_snapshot = original_publish
            if original_dashboard_url is None:
                os.environ.pop("DASHBOARD_URL", None)
            else:
                os.environ["DASHBOARD_URL"] = original_dashboard_url
            if original_dashboard_token is None:
                os.environ.pop("DASHBOARD_TOKEN", None)
            else:
                os.environ["DASHBOARD_TOKEN"] = original_dashboard_token

    assert captured["published"] is True
    push_payload = captured["push_payload"]
    assert isinstance(push_payload, dict)
    assert push_payload["market_map"]["coins"]["BTC"]["bias"] == "BULLISH"


def test_trade_memory_records_richer_loss_reasoning() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_memory_file = trade_memory_module.MEMORY_FILE
        try:
            trade_memory_module.MEMORY_FILE = Path(tmpdir) / "trade_memory.json"
            memory = trade_memory_module.TradeMemory()
            memory.record_trade(
                coin="BTC",
                direction="LONG",
                signal_score=66.0,
                entry_price=100.0,
                exit_price=94.0,
                exit_reason="stop_loss",
                hold_minutes=20.0,
                trend_context="DOWN",
                market_regime="RANGING",
                dominant_regime="ABSORPTION",
                volatility_label="extreme",
                entry_context={
                    "confidence": "LOW",
                    "mtf_bias": "DOWN",
                    "news_score": 35.0,
                    "candle_trend": "DOWN",
                    "foc_score": 40.0,
                },
            )
            stats = memory.get_stats()["BTC"]
            assert stats["latest_failure_summary"], "loss reasoning summary should be recorded"
            assert "Higher timeframe trend was against the long." in stats["root_causes"]
            assert "LOW_CONFIDENCE_ENTRY" in stats["failure_modes"]
            assert "HTF_CONFLICT" in stats["failure_modes"]
        finally:
            trade_memory_module.MEMORY_FILE = original_memory_file


def test_trade_memory_directional_pause_and_guard() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_memory_file = trade_memory_module.MEMORY_FILE
        try:
            trade_memory_module.MEMORY_FILE = Path(tmpdir) / "trade_memory.json"
            memory = trade_memory_module.TradeMemory()
            for _ in range(2):
                memory.record_trade(
                    coin="ETH",
                    direction="LONG",
                    signal_score=67.0,
                    entry_price=100.0,
                    exit_price=95.0,
                    exit_reason="stop_loss",
                    hold_minutes=25.0,
                    trend_context="DOWN",
                    market_regime="RANGING",
                    dominant_regime="ABSORPTION",
                    volatility_label="NORMAL",
                    entry_context={
                        "confidence": "LOW",
                        "mtf_bias": "DOWN",
                        "news_score": 40.0,
                        "candle_trend": "DOWN",
                        "foc_score": 41.0,
                    },
                )
            guard = memory.get_directional_guard("ETH", "LONG")
            assert guard["pause_cycles"] > 0, "repeated same-direction losses should trigger a directional pause"
            assert guard["threshold_boost"] > 0, "repeated loss causes should tighten the threshold"
            assert guard["reasons"], "guard should explain why the threshold was tightened"
        finally:
            trade_memory_module.MEMORY_FILE = original_memory_file


def test_mtf_fail_closed_blocks_when_confirmation_is_unavailable() -> None:
    agent = object.__new__(TradingAgent)
    agent.cfg = build_config()
    agent.cfg.trading.strict_confirmation_fail_closed = True
    agent._last_signals = {"BTC": {}}

    class BrokenCircuit:
        def call(self, *_args, **_kwargs):
            raise CircuitBreakerError("mtf unavailable")

    agent._mtf_circuit = BrokenCircuit()
    allowed = agent._check_mtf_safe("BTC", Signal("LONG"))
    assert allowed is False, "strict confirmation mode should fail closed when MTF is unavailable"
    assert agent._last_signals["BTC"]["mtf_bias"] == "UNAVAILABLE"


def test_execution_quality_gate_blocks_thin_unstable_orderbooks() -> None:
    agent = object.__new__(TradingAgent)
    agent.cfg = build_config()
    agent._orderbook_history = {
        "BTC": [{
            "breakout_state": "NONE",
            "level_interaction": "BETWEEN_LEVELS",
            "support": 99.0,
            "resistance": 101.0,
            "spread_bps": 22.0,
            "bid_notional": 500.0,
            "ask_notional": 500.0,
        }]
    }
    order = OrderRequest(
        coin="BTC",
        direction="LONG",
        size_usd=100.0,
        size_coin=1.0,
        price=100.0,
        stop_loss=96.0,
        take_profit=108.0,
    )
    orderbook_signal = SimpleNamespace(
        valid=True,
        spread_bps=22.0,
        bid_notional=500.0,
        ask_notional=500.0,
        breakout_state="NONE",
        level_interaction="BETWEEN_LEVELS",
        nearest_support=99.0,
        nearest_resistance=101.0,
        block_longs=False,
        block_shorts=False,
    )
    quality = agent._assess_execution_quality("BTC", "LONG", order, orderbook_signal)
    assert quality["permitted"] is False, "thin or unstable books should block market entries"
    assert quality["blockers"], "the gate should explain what failed"


def test_execution_quality_can_fall_back_to_passive_rescue_limit() -> None:
    agent = object.__new__(TradingAgent)
    agent.cfg = build_config()
    agent._orderbook_history = {
        "BTC": [{
            "breakout_state": "CONFIRMED_BULLISH_BREAKOUT",
            "level_interaction": "ABOVE_RESISTANCE",
            "support": 99.0,
            "resistance": 101.0,
            "spread_bps": 20.0,
            "bid_notional": 25000.0,
            "ask_notional": 22000.0,
        }]
    }
    order = OrderRequest(
        coin="BTC",
        direction="LONG",
        size_usd=100.0,
        size_coin=1.0,
        price=100.0,
        stop_loss=96.0,
        take_profit=108.0,
    )
    orderbook_signal = SimpleNamespace(
        valid=True,
        spread_bps=20.0,
        bid_notional=25000.0,
        ask_notional=22000.0,
        breakout_state="CONFIRMED_BULLISH_BREAKOUT",
        level_interaction="ABOVE_RESISTANCE",
        nearest_support=99.0,
        nearest_resistance=101.0,
        block_longs=False,
        block_shorts=False,
        best_bid=99.95,
        best_ask=100.15,
    )
    quality = agent._assess_execution_quality("BTC", "LONG", order, orderbook_signal)
    assert quality["permitted"] is False, "market sweep should still be blocked"
    assert quality["prefer_passive_entry"] is True, "the bot should downgrade to a passive maker entry"
    assert quality["passive_limit_price"] == 99.95


def test_pending_limit_can_reprice_when_book_drifts() -> None:
    cfg = build_config()
    ex = DryRunExchange(starting_balance_usd=1000.0)
    ex.connect()
    agent = TradingAgent(cfg, [ex])
    agent._last_signals["BTC"] = {
        "action": "LONG",
        "decision": "LONG",
        "score": 68.0,
        "confidence": "HIGH",
        "decision_reason": "resting buy",
    }
    pending = PendingOrder(
        coin="BTC",
        direction="LONG",
        limit_price=99.0,
        size_coin=1.0,
        size_usd=99.0,
        stop_loss=96.0,
        take_profit=108.0,
        signal_score=68.0,
        exchange=ex.name,
        exchange_order_id="abc",
        cycles_waiting=3,
        reason="initial_limit",
        metadata={"entry_context": {"confidence": "HIGH", "instrument_type": "crypto", "trade_plan": {}}},
    )
    agent.order_mgr.pending_orders["BTC"] = pending

    original_get_orderbook = agent_module.get_orderbook_levels
    original_cancel = agent._cancel_pending_limit
    original_place = agent._place_limit_order
    original_record = agent._record_decision_snapshot
    called: dict[str, object] = {}
    try:
        agent_module.get_orderbook_levels = lambda *args, **kwargs: SimpleNamespace(
            valid=True,
            best_bid=100.4,
            best_ask=100.6,
            breakout_state="NONE",
            level_interaction="AT_SUPPORT",
            block_longs=False,
            block_shorts=False,
        )
        agent._cancel_pending_limit = lambda pending, reason: called.setdefault("cancel_reason", reason) or True

        def _stub_place(*args, **kwargs):
            called["repriced_to"] = args[2]
            called["reprice_count"] = kwargs.get("extra_metadata", {}).get("reprice_count")
            return {"success": True, "pending": True, "filled": False}

        agent._place_limit_order = _stub_place
        agent._record_decision_snapshot = lambda *args, **kwargs: None
        agent._manage_pending_limits({"BTC": 100.5}, 1000.0)
        assert round(float(called.get("repriced_to", 0.0)), 4) == 100.4
        assert int(called.get("reprice_count", 0) or 0) == 1
    finally:
        agent_module.get_orderbook_levels = original_get_orderbook
        agent._cancel_pending_limit = original_cancel
        agent._place_limit_order = original_place
        agent._record_decision_snapshot = original_record


def test_pending_limit_can_escalate_to_market_on_clean_breakout() -> None:
    cfg = build_config()
    cfg.trading.execution_pending_reprice_enabled = False
    ex = DryRunExchange(starting_balance_usd=1000.0)
    ex.connect()
    agent = TradingAgent(cfg, [ex])
    agent._last_signals["BTC"] = {
        "action": "LONG",
        "decision": "LONG",
        "score": 72.0,
        "confidence": "HIGH",
        "decision_reason": "breakout stalking",
    }
    pending = PendingOrder(
        coin="BTC",
        direction="LONG",
        limit_price=99.0,
        size_coin=1.0,
        size_usd=99.0,
        stop_loss=96.0,
        take_profit=110.0,
        signal_score=72.0,
        exchange=ex.name,
        exchange_order_id="abc",
        cycles_waiting=4,
        reason="initial_limit",
        metadata={"entry_context": {"confidence": "HIGH", "instrument_type": "crypto", "trade_plan": {}}},
    )
    agent.order_mgr.pending_orders["BTC"] = pending

    original_get_orderbook = agent_module.get_orderbook_levels
    original_quality = agent._assess_execution_quality
    original_escalate = agent._escalate_pending_limit_to_market
    original_record = agent._record_decision_snapshot
    called: dict[str, object] = {}
    try:
        agent_module.get_orderbook_levels = lambda *args, **kwargs: SimpleNamespace(
            valid=True,
            best_bid=100.9,
            best_ask=101.0,
            breakout_state="CONFIRMED_BULLISH_BREAKOUT",
            level_interaction="ABOVE_BREAKOUT",
            block_longs=False,
            block_shorts=False,
        )
        agent._assess_execution_quality = lambda *args, **kwargs: {
            "permitted": True,
            "spread_bps": 4.0,
            "estimated_slippage_bps": 6.0,
            "score": 84.0,
        }

        def _stub_escalate(pending, *, live_price, reason, orderbook_signal=None):
            called["live_price"] = live_price
            called["reason"] = reason
            return True

        agent._escalate_pending_limit_to_market = _stub_escalate
        agent._record_decision_snapshot = lambda *args, **kwargs: None
        agent._manage_pending_limits({"BTC": 101.0}, 1000.0)
        assert called["live_price"] == 101.0
        assert "breakout is running away" in str(called["reason"])
    finally:
        agent_module.get_orderbook_levels = original_get_orderbook
        agent._assess_execution_quality = original_quality
        agent._escalate_pending_limit_to_market = original_escalate
        agent._record_decision_snapshot = original_record


def test_asset_state_machine_reports_confirmation_wait_clearly() -> None:
    lifecycle = asset_state_machine_module.build_asset_state(
        {
            "action": "LONG",
            "execution_mode": "tradable",
            "streak_confirmation_remaining": 1,
            "thesis_candidate_action": "LONG",
        },
        stage="signal_streak_wait",
    )
    assert lifecycle["state"] == "WAITING_CONFIRMATION"
    assert "confirming cycle" in lifecycle["next_unblock_reason"]


def test_asset_state_machine_promotes_major_catalyst_reclaim_watch() -> None:
    lifecycle = asset_state_machine_module.build_asset_state(
        {
            "action": "FLAT",
            "execution_mode": "observation_only",
            "instrument_type": "equity",
            "news_score": 62.77,
            "news_catalyst_score": 3.75,
            "news_catalyst_summary": "platform anchor + partner attached + demand commitment",
            "market_map_bias": "BULLISH",
            "market_map_reclaim_confirmed": True,
            "market_map_live_reclaim": False,
            "market_map_reclaim_lost": True,
            "market_map_nearest_resistance": 253.36,
            "thesis_candidate_action": "LONG",
            "thesis_permitted": False,
        },
        stage="observation_only",
    )
    assert lifecycle["state"] == "MAJOR_CATALYST_WATCH"
    assert lifecycle["label"] == "Major catalyst watch"
    assert "253.36" in lifecycle["next_unblock_reason"]


def test_data_reliability_blocks_stale_incoherent_setup() -> None:
    cfg = build_config()
    cfg.trading.use_news = True
    reliability = data_reliability_module.assess_reliability(
        cfg.trading,
        {
            "execution_mode": "tradable",
            "instrument_type": "crypto",
            "action": "LONG",
            "using_closed_candles": False,
            "analysis_price": 100.0,
            "live_price": 101.5,
            "market_map_available": False,
            "news_articles": 0,
            "orderbook_valid": False,
            "orderbook_feed_snapshot_count": 0,
        },
    )
    assert reliability["permitted"] is False
    assert reliability["blockers"], "unreliable data should explicitly block the setup"


def test_portfolio_guard_blocks_theme_stacking_and_trims_secondary_exposure() -> None:
    cfg = build_config()
    open_positions = [
        OpenPosition("HYPE", "LONG", 100.0, 90.0, 0.9, 95.0, 108.0),
        OpenPosition("TAO", "LONG", 100.0, 80.0, 0.8, 95.0, 108.0),
    ]
    blocked = portfolio_guard_module.assess_correlation(
        cfg.trading,
        coin="SOL",
        direction="LONG",
        instrument_type="crypto",
        portfolio_usd=1000.0,
        proposed_size_usd=100.0,
        open_positions=open_positions,
        pending_orders=[],
    )
    assert blocked["permitted"] is False
    assert blocked["theme"] == "CRYPTO_HIGH_BETA"

    trimmed = portfolio_guard_module.assess_correlation(
        cfg.trading,
        coin="GOOGL",
        direction="LONG",
        instrument_type="equity",
        portfolio_usd=1000.0,
        proposed_size_usd=100.0,
        open_positions=[OpenPosition("META", "LONG", 100.0, 60.0, 0.6, 95.0, 108.0)],
        pending_orders=[],
    )
    assert trimmed["permitted"] is True
    assert trimmed["size_multiplier"] < 1.0

    earnings_starter_blocked = portfolio_guard_module.assess_correlation(
        cfg.trading,
        coin="META",
        direction="LONG",
        instrument_type="equity",
        portfolio_usd=10000.0,
        proposed_size_usd=250.0,
        open_positions=[OpenPosition("GOOGL", "LONG", 339.0, 175.0, 0.5, 335.0, 350.0)],
        pending_orders=[
            PendingOrder(
                coin="AMZN",
                direction="LONG",
                limit_price=255.0,
                size_coin=1.0,
                size_usd=255.0,
                stop_loss=250.0,
                take_profit=270.0,
                signal_score=66.0,
            )
        ],
    )
    assert earnings_starter_blocked["permitted"] is False

    earnings_starter_allowed = portfolio_guard_module.assess_correlation(
        cfg.trading,
        coin="META",
        direction="LONG",
        instrument_type="equity",
        portfolio_usd=10000.0,
        proposed_size_usd=250.0,
        open_positions=[OpenPosition("GOOGL", "LONG", 339.0, 175.0, 0.5, 335.0, 350.0)],
        pending_orders=[
            PendingOrder(
                coin="AMZN",
                direction="LONG",
                limit_price=255.0,
                size_coin=1.0,
                size_usd=255.0,
                stop_loss=250.0,
                take_profit=270.0,
                signal_score=66.0,
            )
        ],
        event_starter=True,
    )
    assert earnings_starter_allowed["permitted"] is True
    assert earnings_starter_allowed["size_multiplier"] < trimmed["size_multiplier"]
    assert "extra scout slot" in earnings_starter_allowed["summary"]

    crowded_earnings_starter_allowed = portfolio_guard_module.assess_correlation(
        cfg.trading,
        coin="META",
        direction="LONG",
        instrument_type="equity",
        portfolio_usd=10000.0,
        proposed_size_usd=250.0,
        open_positions=[
            OpenPosition("GOOGL", "LONG", 339.0, 175.0, 0.5, 335.0, 350.0),
            OpenPosition("MSFT", "LONG", 514.0, 317.0, 0.62, 505.0, 535.0),
        ],
        pending_orders=[
            PendingOrder(
                coin="AMZN",
                direction="LONG",
                limit_price=255.0,
                size_coin=1.0,
                size_usd=255.0,
                stop_loss=250.0,
                take_profit=270.0,
                signal_score=66.0,
            )
        ],
        event_starter=True,
    )
    assert crowded_earnings_starter_allowed["permitted"] is True
    assert crowded_earnings_starter_allowed["size_multiplier"] < earnings_starter_allowed["size_multiplier"]


def test_event_risk_budget_trims_and_blocks_crowded_pre_event_starters() -> None:
    cfg = build_config()
    cfg.trading.event_risk_budget_max_portfolio_pct = 0.05
    cfg.trading.event_risk_budget_max_theme_pct = 0.04
    cfg.trading.event_risk_budget_max_single_pct = 0.02
    event_metadata = {
        "entry_context": {
            "instrument_type": "equity",
            "conviction_entry_event": True,
            "news_event_score": 4.5,
            "news_event_tags": ["official_ir_event", "analyst_revision"],
        }
    }
    existing = [
        OpenPosition(
            "GOOGL",
            "LONG",
            180.0,
            200.0,
            1.1,
            172.0,
            196.0,
            metadata=event_metadata,
        )
    ]
    trimmed = portfolio_guard_module.assess_correlation(
        cfg.trading,
        coin="META",
        direction="LONG",
        instrument_type="equity",
        portfolio_usd=10000.0,
        proposed_size_usd=600.0,
        open_positions=existing,
        pending_orders=[],
        event_starter=True,
    )
    assert trimmed["permitted"] is True
    assert trimmed["event_budget"]["active"] is True
    assert trimmed["event_budget_size_multiplier"] <= 0.334
    assert trimmed["size_multiplier"] < 1.0
    assert "event risk" in " ".join(trimmed["warnings"]).lower()

    full = portfolio_guard_module.assess_correlation(
        cfg.trading,
        coin="AMZN",
        direction="LONG",
        instrument_type="equity",
        portfolio_usd=10000.0,
        proposed_size_usd=150.0,
        open_positions=[
            OpenPosition(
                "GOOGL",
                "LONG",
                180.0,
                250.0,
                1.3,
                172.0,
                196.0,
                metadata=event_metadata,
            ),
            OpenPosition(
                "META",
                "LONG",
                640.0,
                250.0,
                0.4,
                620.0,
                700.0,
                metadata=event_metadata,
            ),
        ],
        pending_orders=[],
        event_starter=True,
    )
    assert full["permitted"] is False
    assert "event risk" in full["summary"].lower()


def test_pre_ipo_symbols_automatically_use_tighter_event_budget() -> None:
    cfg = build_config()
    cfg.trading.min_trade_usd = 50.0
    cfg.trading.event_risk_budget_max_single_pct = 0.02
    cfg.trading.pre_ipo_event_risk_budget_max_single_pct = 0.0125
    cfg.trading.pre_ipo_event_risk_budget_max_theme_pct = 0.025
    result = portfolio_guard_module.assess_correlation(
        cfg.trading,
        coin="CBRS",
        direction="LONG",
        instrument_type="equity",
        portfolio_usd=10000.0,
        proposed_size_usd=600.0,
        open_positions=[],
        pending_orders=[],
        event_starter=False,
    )
    assert result["permitted"] is True
    assert result["theme"] == "PRE_IPO_EVENT"
    assert result["event_budget"]["active"] is True
    assert result["event_budget"]["pre_ipo"] is True
    assert result["event_budget"]["single_trade_cap_pct"] == 1.25
    assert result["event_budget_size_multiplier"] <= 0.209
    assert result["size_multiplier"] < result["event_budget_size_multiplier"]
    assert "pre-ipo" in " ".join(result["warnings"]).lower()


def test_background_orderbook_feed_enriches_signal_with_persistence_history() -> None:
    class FakeFeed:
        def __init__(self, snapshots):
            self._snapshots = snapshots

        def get_recent_snapshots(self, _coin, *, max_age_seconds=45.0, limit=None):
            items = list(self._snapshots)
            if limit:
                items = items[-limit:]
            return items

    def make_snapshot(ts_offset: float, ref_price: float, imbalance: float) -> orderbook_levels_module.OrderBookFeedSnapshot:
        return orderbook_levels_module.OrderBookFeedSnapshot(
            coin="BTC",
            ts=time.time() - ts_offset,
            valid=True,
            ref_price=ref_price,
            last_trade_price=ref_price,
            price_decimals=2,
            book={
                "bids": [
                    {"price": "99.90", "remaining_base_amount": "4"},
                    {"price": "99.00", "remaining_base_amount": "25"},
                    {"price": "98.50", "remaining_base_amount": "4"},
                ],
                "asks": [
                    {"price": "100.10", "remaining_base_amount": "4"},
                    {"price": "101.00", "remaining_base_amount": "10"},
                    {"price": "101.50", "remaining_base_amount": "3"},
                ],
            },
            bid_notional=3100.0,
            ask_notional=1900.0,
            best_bid=99.90,
            best_ask=100.10,
            spread_bps=20.0,
            imbalance_ratio=imbalance,
            strongest_bid_wall=99.0,
            strongest_bid_wall_notional=2475.0,
            strongest_ask_wall=101.0,
            strongest_ask_wall_notional=1010.0,
        )

    snapshots = [
        make_snapshot(6.0, 100.00, 0.12),
        make_snapshot(3.0, 100.05, 0.18),
        make_snapshot(1.0, 100.08, 0.24),
    ]

    original_feed = orderbook_levels_module._BACKGROUND_ORDERBOOK_FEED
    original_daily_levels = orderbook_levels_module._daily_levels
    try:
        orderbook_levels_module._BACKGROUND_ORDERBOOK_FEED = FakeFeed(snapshots)
        orderbook_levels_module._daily_levels = lambda *_args, **_kwargs: (
            [(99.0, 0.85, "daily", "daily_swing_low")],
            [(101.0, 0.80, "daily", "daily_swing_high")],
            98.7,
        )
        signal = orderbook_levels_module.get_orderbook_levels(
            "BTC",
            current_price=100.08,
            cache_ttl_seconds=0,
        )
        assert signal.valid is True
        assert signal.feed_snapshot_count == 3, "feed-backed signal should expose history depth"
        assert signal.support_wall_persistence >= 3, "repeated bid wall should persist across feed samples"
        assert signal.imbalance_mean > 0.15, "history should smooth into a bullish mean imbalance"
        assert signal.score > 55.0, "persistent supportive flow should raise the composite orderbook score"
    finally:
        orderbook_levels_module._BACKGROUND_ORDERBOOK_FEED = original_feed
        orderbook_levels_module._daily_levels = original_daily_levels


def test_background_orderbook_feed_detects_intracycle_breakout_between_agent_cycles() -> None:
    class FakeFeed:
        def __init__(self, snapshots):
            self._snapshots = snapshots

        def get_recent_snapshots(self, _coin, *, max_age_seconds=45.0, limit=None):
            items = list(self._snapshots)
            if limit:
                items = items[-limit:]
            return items

    def make_snapshot(ts_offset: float, ref_price: float) -> orderbook_levels_module.OrderBookFeedSnapshot:
        return orderbook_levels_module.OrderBookFeedSnapshot(
            coin="BTC",
            ts=time.time() - ts_offset,
            valid=True,
            ref_price=ref_price,
            last_trade_price=ref_price,
            price_decimals=2,
            book={
                "bids": [
                    {"price": "100.10", "remaining_base_amount": "8"},
                    {"price": "99.80", "remaining_base_amount": "5"},
                ],
                "asks": [
                    {"price": "100.55", "remaining_base_amount": "5"},
                    {"price": "101.20", "remaining_base_amount": "10"},
                ],
            },
            bid_notional=3600.0,
            ask_notional=2100.0,
            best_bid=100.10,
            best_ask=100.55,
            spread_bps=44.9,
            imbalance_ratio=0.20,
            strongest_bid_wall=100.10,
            strongest_bid_wall_notional=800.0,
            strongest_ask_wall=101.20,
            strongest_ask_wall_notional=1012.0,
        )

    snapshots = [
        make_snapshot(8.0, 100.18),
        make_snapshot(5.0, 100.28),
        make_snapshot(2.0, 100.42),
    ]

    original_feed = orderbook_levels_module._BACKGROUND_ORDERBOOK_FEED
    original_daily_levels = orderbook_levels_module._daily_levels
    try:
        orderbook_levels_module._BACKGROUND_ORDERBOOK_FEED = FakeFeed(snapshots)
        orderbook_levels_module._daily_levels = lambda *_args, **_kwargs: (
            [(98.5, 0.70, "daily", "daily_swing_low")],
            [(100.0, 0.82, "daily", "daily_swing_high"), (101.5, 0.75, "daily", "20d_high")],
            99.6,
        )
        signal = orderbook_levels_module.get_orderbook_levels(
            "BTC",
            current_price=100.42,
            cache_ttl_seconds=0,
            feed_breakout_samples=2,
        )
        assert signal.breakout_state == "PERSISTENT_BULLISH_BREAKOUT", "feed should detect a breakout persisting between agent cycles"
        assert signal.intracycle_breakout_state == "PERSISTENT_BULLISH_BREAKOUT"
        assert signal.favor_longs is True
    finally:
        orderbook_levels_module._BACKGROUND_ORDERBOOK_FEED = original_feed
        orderbook_levels_module._daily_levels = original_daily_levels


def test_agent_bootstraps_background_orderbook_feed() -> None:
    agent = object.__new__(TradingAgent)
    agent.cfg = build_config()
    agent._analysis_coins = ["BTC", "ETH", "SP500"]

    calls = []
    original_configure = agent_module.configure_background_orderbook_feed
    original_prime = agent_module.prime_background_orderbook_feed
    original_start = agent_module.start_background_orderbook_feed
    try:
        agent_module.configure_background_orderbook_feed = lambda coins, **kwargs: calls.append(("configure", list(coins), kwargs))
        agent_module.prime_background_orderbook_feed = lambda: calls.append(("prime",))
        agent_module.start_background_orderbook_feed = lambda: calls.append(("start",))
        agent._start_background_orderbook_feed()
        assert calls[0][0] == "configure"
        assert calls[0][1] == ["BTC", "ETH", "SP500"], "agent should warm the feed for the full analysis universe"
        assert ("prime",) in calls
        assert ("start",) in calls
    finally:
        agent_module.configure_background_orderbook_feed = original_configure
        agent_module.prime_background_orderbook_feed = original_prime
        agent_module.start_background_orderbook_feed = original_start


def test_lighter_read_auth_headers_gracefully_fallback_without_credentials() -> None:
    original_env = {key: os.environ.get(key) for key in (
        "LIGHTER_L1_PRIVATE_KEY",
        "LIGHTER_PRIVATE_KEY",
        "LIGHTER_API_PRIVATE_KEY",
        "LIGHTER_ACCOUNT_INDEX",
        "LIGHTER_API_KEY_INDEX",
        "LIGHTER_API_BASE_URL",
    )}
    original_cache = dict(lighter_client_module._READ_AUTH_CACHE)
    original_missing_logged = lighter_client_module._READ_AUTH_MISSING_LOGGED
    try:
        for key in original_env:
            os.environ.pop(key, None)
        lighter_client_module._READ_AUTH_CACHE.update(
            {"token": "", "expires_at": 0.0, "account_index": None, "api_key_index": None, "api_base_url": ""}
        )
        lighter_client_module._READ_AUTH_MISSING_LOGGED = False
        headers = asyncio.run(lighter_client_module.get_lighter_read_auth_headers())
        assert headers == {}, "missing Lighter credentials should keep read auth optional"
    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        lighter_client_module._READ_AUTH_CACHE.clear()
        lighter_client_module._READ_AUTH_CACHE.update(original_cache)
        lighter_client_module._READ_AUTH_MISSING_LOGGED = original_missing_logged


def test_market_data_attaches_lighter_auth_header_to_read_requests() -> None:
    captured: dict = {}

    class FakeResponse:
        async def text(self) -> str:
            return '{"trades":[{"price":"123.45"}]}'

    class FakeApiClient:
        def __init__(self, _config):
            pass

        async def close(self) -> None:
            return None

    class FakeOrderApi:
        def __init__(self, _client):
            pass

        async def recent_trades_without_preload_content(self, **kwargs):
            captured.update(kwargs)
            return FakeResponse()

    fake_lighter = ModuleType("lighter")
    fake_lighter.Configuration = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_lighter.ApiClient = FakeApiClient
    fake_lighter.OrderApi = FakeOrderApi

    original_lighter = sys.modules.get("lighter")
    original_auth = market_data_module.get_lighter_read_auth_headers
    try:
        sys.modules["lighter"] = fake_lighter

        async def fake_auth_headers(**_kwargs):
            return {"Authorization": "signed-test-token"}

        market_data_module.get_lighter_read_auth_headers = fake_auth_headers
        payload = market_data_module._run_async(
            market_data_module._lighter_api_get(
                "recent_trades_without_preload_content",
                market_id=1,
                limit=1,
            )
        )
        assert payload["trades"][0]["price"] == "123.45"
        assert captured.get("_headers", {}).get("Authorization") == "signed-test-token"
    finally:
        market_data_module.get_lighter_read_auth_headers = original_auth
        if original_lighter is None:
            sys.modules.pop("lighter", None)
        else:
            sys.modules["lighter"] = original_lighter


def test_orderbook_reader_uses_hyperliquid_l2_snapshot() -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "coin": "@268",
                "time": 1234567890,
                "levels": [
                    [{"px": "100.0", "sz": "2.5", "n": 1}],
                    [{"px": "101.0", "sz": "3.0", "n": 2}],
                ],
            }

    original_post = orderbook_levels_module.requests.post
    original_supported = orderbook_levels_module.is_hyperliquid_supported
    original_resolve = orderbook_levels_module.resolve_hyperliquid_symbol
    original_dex = orderbook_levels_module.get_hyperliquid_market_dex
    try:
        orderbook_levels_module.is_hyperliquid_supported = lambda _coin: True
        orderbook_levels_module.resolve_hyperliquid_symbol = lambda _coin: "@268"
        orderbook_levels_module.get_hyperliquid_market_dex = lambda _coin: "xyz"

        def fake_post(url, json=None, timeout=0):
            captured["url"] = url
            captured["json"] = json
            captured["timeout"] = timeout
            return FakeResponse()

        orderbook_levels_module.requests.post = fake_post
        snapshot = orderbook_levels_module._fetch_live_snapshot("AAPL", depth_limit=25)
        assert snapshot.valid is True
        assert snapshot.best_bid == 100.0
        assert snapshot.best_ask == 101.0
        assert snapshot.bid_notional == 250.0
        assert snapshot.ask_notional == 303.0
        assert captured["json"] == {"type": "l2Book", "coin": "@268", "dex": "xyz"}
    finally:
        orderbook_levels_module.requests.post = original_post
        orderbook_levels_module.is_hyperliquid_supported = original_supported
        orderbook_levels_module.resolve_hyperliquid_symbol = original_resolve
        orderbook_levels_module.get_hyperliquid_market_dex = original_dex


def test_dry_run_blocks_short_on_long_only_spot_symbol() -> None:
    ex = DryRunExchange(
        starting_balance_usd=1000.0,
        supported_symbols=["AAPL"],
        shortable_map={"AAPL": False},
    )
    ex.connect()
    result = ex.market_sell("AAPL", 1.0)
    assert result.success is False
    assert "long-only" in result.error.lower()


def test_hyperliquid_limit_order_returns_resting_order_id() -> None:
    original_spec = hyperliquid_client_module.get_hyperliquid_market_spec
    original_resolve = hyperliquid_client_module.resolve_hyperliquid_symbol
    try:
        hyperliquid_client_module.get_hyperliquid_market_spec = lambda _coin: {
            "venue_symbol": "BTC",
            "market_type": "perp",
        }
        hyperliquid_client_module.resolve_hyperliquid_symbol = lambda _coin: "BTC"

        class FakeExchange:
            def order(self, **kwargs):
                assert kwargs["order_type"] == {"limit": {"tif": "Alo"}}
                return {
                    "status": "ok",
                    "response": {
                        "data": {
                            "statuses": [
                                {"resting": {"oid": 123456}}
                            ]
                        }
                    },
                }

        client = hyperliquid_client_module.HyperliquidClient("pk", "0xabc")
        client._connected = True
        client._exchange = FakeExchange()

        result = client.limit_buy("BTC", 1.25, 100.0, maker_only=True)
        assert result.success is True
        assert result.order_id == "123456"
    finally:
        hyperliquid_client_module.get_hyperliquid_market_spec = original_spec
        hyperliquid_client_module.resolve_hyperliquid_symbol = original_resolve


def test_hyperliquid_market_catalog_expands_unknown_live_perps() -> None:
    original_perps = hyperliquid_markets_module._fetch_perp_names
    original_spots = hyperliquid_markets_module._fetch_spot_pairs
    original_cache = dict(hyperliquid_markets_module._CATALOG_CACHE)
    try:
        hyperliquid_markets_module._CATALOG_CACHE["ts"] = 0.0
        hyperliquid_markets_module._CATALOG_CACHE["catalog"] = {}
        hyperliquid_markets_module._fetch_perp_names = lambda dex="": {"BTC", "XRP"} if not dex else {"xyz:INTC"}
        hyperliquid_markets_module._fetch_spot_pairs = lambda: {}
        catalog = hyperliquid_markets_module.get_hyperliquid_market_catalog(force_refresh=True)
        assert "XRP" in catalog
        assert catalog["XRP"]["market_type"] == "perp"
        assert catalog["XRP"]["instrument_type"] == "crypto"
        assert catalog["XRP"]["shortable"] is True
        assert "INTC" in catalog
        assert catalog["INTC"]["venue_symbol"] == "xyz:INTC"
        assert catalog["INTC"]["dex"] == "xyz"
        assert catalog["INTC"]["instrument_type"] == "equity"
    finally:
        hyperliquid_markets_module._fetch_perp_names = original_perps
        hyperliquid_markets_module._fetch_spot_pairs = original_spots
        hyperliquid_markets_module._CATALOG_CACHE.clear()
        hyperliquid_markets_module._CATALOG_CACHE.update(original_cache)


def test_hyperliquid_market_catalog_discovers_unknown_tradexyz_equities() -> None:
    original_perps = hyperliquid_markets_module._fetch_perp_names
    original_spots = hyperliquid_markets_module._fetch_spot_pairs
    original_cache = dict(hyperliquid_markets_module._CATALOG_CACHE)
    try:
        hyperliquid_markets_module._CATALOG_CACHE["ts"] = 0.0
        hyperliquid_markets_module._CATALOG_CACHE["catalog"] = {}
        hyperliquid_markets_module._fetch_perp_names = (
            lambda dex="": {"BTC"} if not dex else {"xyz:NVDA", "xyz:CRWV", "xyz:EWY", "xyz:NEWIPO"}
        )
        hyperliquid_markets_module._fetch_spot_pairs = lambda: {}
        catalog = hyperliquid_markets_module.get_hyperliquid_market_catalog(force_refresh=True)
        assert catalog["NVDA"]["venue_symbol"] == "xyz:NVDA"
        assert catalog["NVDA"]["instrument_type"] == "equity"
        assert catalog["NVDA"]["dex"] == "xyz"
        assert catalog["CRWV"]["display_name"] == "CoreWeave"
        assert catalog["EWY"]["instrument_type"] == "index"
        assert catalog["NEWIPO"]["venue_symbol"] == "xyz:NEWIPO"
        assert catalog["NEWIPO"]["instrument_type"] == "equity"
        assert catalog["NEWIPO"]["categories"] == ["other_stocks"]
    finally:
        hyperliquid_markets_module._fetch_perp_names = original_perps
        hyperliquid_markets_module._fetch_spot_pairs = original_spots
        hyperliquid_markets_module._CATALOG_CACHE.clear()
        hyperliquid_markets_module._CATALOG_CACHE.update(original_cache)


def test_hyperliquid_market_catalog_enables_full_tradexyz_catalog_and_prefers_perps() -> None:
    original_perps = hyperliquid_markets_module._fetch_perp_names
    original_spots = hyperliquid_markets_module._fetch_spot_pairs
    original_cache = dict(hyperliquid_markets_module._CATALOG_CACHE)
    try:
        hyperliquid_markets_module._CATALOG_CACHE["ts"] = 0.0
        hyperliquid_markets_module._CATALOG_CACHE["catalog"] = {}
        tradexyz_symbols = {
            f"xyz:{coin}"
            for coin in hyperliquid_markets_module.TRADEXYZ_ASSET_METADATA.keys()
        }
        spot_pairs = {
            str(spec["pair_name"]).upper(): {"venue_symbol": spec["fallback_symbol"]}
            for spec in hyperliquid_markets_module._SPOT_MARKETS.values()
        }
        hyperliquid_markets_module._fetch_perp_names = (
            lambda dex="": {"BTC", "ETH", "SPX"} if not dex else tradexyz_symbols
        )
        hyperliquid_markets_module._fetch_spot_pairs = lambda: spot_pairs
        catalog = hyperliquid_markets_module.get_hyperliquid_market_catalog(force_refresh=True)

        missing = [
            coin
            for coin in hyperliquid_markets_module.TRADEXYZ_ASSET_METADATA.keys()
            if coin not in catalog
        ]
        assert missing == []
        for coin in hyperliquid_markets_module.TRADEXYZ_ASSET_METADATA.keys():
            assert catalog[coin]["market_type"] == "perp"
            assert catalog[coin]["dex"] == "xyz"
            assert catalog[coin]["shortable"] is True
            assert catalog[coin]["live_tradeable"] is True
        assert catalog["TSLA"]["venue_symbol"] == "xyz:TSLA"
        assert catalog["TSLA"]["instrument_type"] == "equity"
        assert catalog["AAPL"]["market_type"] == "perp"
        assert catalog["CBRS"]["venue_symbol"] == "xyz:CBRS"
        assert catalog["CBRS"]["categories"] == ["pre_ipo", "semis_memory", "ai_infra"]
        assert catalog["CBRS"]["pre_ipo"] is True
        assert catalog["NVDA"]["categories"] == ["mag7", "semis_memory"]
    finally:
        hyperliquid_markets_module._fetch_perp_names = original_perps
        hyperliquid_markets_module._fetch_spot_pairs = original_spots
        hyperliquid_markets_module._CATALOG_CACHE.clear()
        hyperliquid_markets_module._CATALOG_CACHE.update(original_cache)


def test_apply_dynamic_analysis_universe_auto_adds_supported_stocks() -> None:
    cfg = build_config()
    cfg.trading.dry_run = False
    cfg.exchange.use_hyperliquid = True
    cfg.exchange.use_lighter = False
    cfg.exchange.hl_spot_execution_enabled = True
    cfg.trading.analysis_coins = ["BTC"]
    cfg.trading.dynamic_market_cap_watchlist_enabled = False
    original_config = main_module.config
    original_supported = main_module.get_hyperliquid_supported_coins
    original_catalog = main_module.get_hyperliquid_market_catalog
    main_module.config = cfg
    try:
        main_module.get_hyperliquid_supported_coins = lambda **kwargs: [
            "BTC", "TSLA", "NVDA", "EWY", "CBRS", "EWZ", "KRW", "ZM",
        ]
        main_module.get_hyperliquid_market_catalog = lambda force_refresh=False: {
            "BTC": {"instrument_type": "crypto"},
            "CBRS": {"instrument_type": "equity", "categories": ["pre_ipo", "semis_memory", "ai_infra"]},
            "EWZ": {"instrument_type": "index", "categories": ["latam_macro", "indices_macro"]},
            "KRW": {"instrument_type": "index", "categories": ["fx_rates", "asia_macro"]},
            "TSLA": {"instrument_type": "equity"},
            "NVDA": {"instrument_type": "equity"},
            "EWY": {"instrument_type": "index"},
            "ZM": {"instrument_type": "equity", "categories": ["software", "growth"]},
        }
        dynamic = main_module.apply_dynamic_analysis_universe()
        assert dynamic == []
        assert main_module.config.trading.analysis_coins == [
            "BTC", "CBRS", "EWY", "EWZ", "KRW", "NVDA", "TSLA", "ZM",
        ]
        assert main_module.config.trading.instrument_types["CBRS"] == "equity"
        assert main_module.config.trading.instrument_types["NVDA"] == "equity"
        assert main_module.config.trading.instrument_types["EWY"] == "index"
        assert main_module.config.trading.asset_category_map["EWZ"] == ["latam_macro", "indices_macro"]
        assert main_module.config.trading.portfolio_theme_map["EWZ"] == "LATAM_MACRO"
        assert main_module.config.trading.asset_category_map["KRW"] == ["fx_rates", "asia_macro"]
        assert main_module.config.trading.portfolio_theme_map["KRW"] == "FX_RATES"
        assert main_module.config.trading.asset_category_map["ZM"] == ["software", "growth"]
        assert main_module.config.trading.portfolio_theme_map["ZM"] == "SOFTWARE_GROWTH"
    finally:
        main_module.config = original_config
        main_module.get_hyperliquid_supported_coins = original_supported
        main_module.get_hyperliquid_market_catalog = original_catalog


def test_apply_dynamic_analysis_universe_gates_tradexyz_equities_by_market_cap() -> None:
    cfg = build_config()
    cfg.trading.dry_run = True
    cfg.exchange.use_hyperliquid = True
    cfg.trading.dynamic_market_cap_watchlist_enabled = True
    cfg.trading.dynamic_market_cap_gate_tradexyz_enabled = True
    cfg.trading.coins = ["BTC", "H100", "EBAY", "NIFTY"]
    cfg.trading.analysis_coins = ["BTC", "BIRD", "H100", "EBAY", "NIFTY"]
    original_config = main_module.config
    original_supported = main_module.get_hyperliquid_supported_coins
    original_catalog = main_module.get_hyperliquid_market_catalog
    original_builder = main_module.build_hyperliquid_market_cap_watchlist
    main_module.config = cfg
    try:
        main_module.get_hyperliquid_supported_coins = lambda **kwargs: ["BTC", "BIRD", "H100", "EBAY", "NIFTY"]
        main_module.get_hyperliquid_market_catalog = lambda force_refresh=False: {
            "BTC": {"instrument_type": "crypto", "market_type": "perp", "venue_symbol": "BTC"},
            "BIRD": {"instrument_type": "equity", "market_type": "perp", "venue_symbol": "xyz:BIRD", "dex": "xyz"},
            "H100": {"instrument_type": "equity", "market_type": "perp", "venue_symbol": "xyz:H100", "dex": "xyz"},
            "EBAY": {"instrument_type": "equity", "market_type": "perp", "venue_symbol": "xyz:EBAY", "dex": "xyz", "categories": ["consumer"]},
            "NIFTY": {"instrument_type": "index", "market_type": "perp", "venue_symbol": "xyz:NIFTY", "dex": "xyz", "categories": ["asia_macro", "indices_macro"]},
        }
        main_module.build_hyperliquid_market_cap_watchlist = lambda **kwargs: {
            "coins": ["EBAY"],
            "records": [{"coin": "EBAY", "market_cap_usd": 45_000_000_000.0}],
        }

        dynamic = main_module.apply_dynamic_analysis_universe()

        assert dynamic == ["EBAY"]
        assert "EBAY" in main_module.config.trading.analysis_coins
        assert "EBAY" in main_module.config.trading.coins
        assert "NIFTY" in main_module.config.trading.analysis_coins
        assert "NIFTY" in main_module.config.trading.coins
        assert "BIRD" not in main_module.config.trading.analysis_coins
        assert "H100" not in main_module.config.trading.analysis_coins
        assert "H100" not in main_module.config.trading.coins
    finally:
        main_module.config = original_config
        main_module.get_hyperliquid_supported_coins = original_supported
        main_module.get_hyperliquid_market_catalog = original_catalog
        main_module.build_hyperliquid_market_cap_watchlist = original_builder


def test_agent_runtime_tradexyz_listing_sync_onboards_new_symbols() -> None:
    cfg = build_config()
    cfg.trading.coins = ["BTC"]
    cfg.trading.analysis_coins = ["BTC"]
    cfg.trading.dynamic_analysis_coins = []
    cfg.trading.auto_promote_analysis_coins = True
    cfg.trading.promote_analysis_before_activity = True
    cfg.trading.tradexyz_listing_auto_sync_enabled = True
    cfg.trading.tradexyz_listing_sync_interval_cycles = 1
    exchange = DryRunExchange(starting_balance_usd=1000.0, supported_symbols=["BTC"])
    agent = TradingAgent(cfg, [exchange])

    original_catalog = agent_module.get_hyperliquid_market_catalog
    original_active = agent_module.hyperliquid_market_is_active
    original_supported = agent_module.is_hyperliquid_supported
    try:
        agent_module.get_hyperliquid_market_catalog = lambda force_refresh=False: {
            "BTC": {"instrument_type": "crypto", "paper_tradeable": True},
            "NEWIPO": {
                "coin": "NEWIPO",
                "venue_symbol": "xyz:NEWIPO",
                "dex": "xyz",
                "market_type": "perp",
                "instrument_type": "equity",
                "categories": ["pre_ipo", "ai_infra"],
                "shortable": True,
                "paper_tradeable": True,
                "live_tradeable": True,
            },
        }
        agent_module.hyperliquid_market_is_active = lambda coin, force_refresh=False: False
        agent_module.is_hyperliquid_supported = lambda coin: str(coin).upper() in {"BTC", "NEWIPO"}

        added = agent._maybe_sync_tradexyz_listing_universe(force=True)

        assert added == ["NEWIPO"]
        assert "NEWIPO" in agent._analysis_coins
        assert "NEWIPO" in agent._tradable_coins
        assert "NEWIPO" in agent._tradable_coin_set
        assert "NEWIPO" in cfg.trading.analysis_coins
        assert "NEWIPO" in cfg.trading.coins
        assert cfg.trading.dynamic_analysis_coins == ["NEWIPO"]
        assert cfg.trading.instrument_types["NEWIPO"] == "equity"
        assert cfg.trading.asset_category_map["NEWIPO"] == ["pre_ipo", "ai_infra"]
        assert cfg.trading.portfolio_theme_map["NEWIPO"] == "PRE_IPO_EVENT"
        assert "NEWIPO" in exchange.supported_coins()
        assert "NEWIPO" in agent._price_circuits
        assert agent._last_listing_sync_report["added_execution"] == ["NEWIPO"]
    finally:
        agent_module.get_hyperliquid_market_catalog = original_catalog
        agent_module.hyperliquid_market_is_active = original_active
        agent_module.is_hyperliquid_supported = original_supported


def test_agent_runtime_tradexyz_listing_sync_skips_low_cap_equities() -> None:
    cfg = build_config()
    cfg.trading.coins = ["BTC"]
    cfg.trading.analysis_coins = ["BTC"]
    cfg.trading.dynamic_analysis_coins = []
    cfg.trading.auto_promote_analysis_coins = True
    cfg.trading.promote_analysis_before_activity = True
    cfg.trading.dynamic_market_cap_watchlist_enabled = True
    cfg.trading.dynamic_market_cap_gate_tradexyz_enabled = True
    cfg.trading.tradexyz_listing_auto_sync_enabled = True
    cfg.trading.tradexyz_listing_sync_interval_cycles = 1
    exchange = DryRunExchange(starting_balance_usd=1000.0, supported_symbols=["BTC"])
    agent = TradingAgent(cfg, [exchange])

    original_catalog = agent_module.get_hyperliquid_market_catalog
    original_builder = agent_module.build_hyperliquid_market_cap_watchlist
    try:
        agent_module.get_hyperliquid_market_catalog = lambda force_refresh=False: {
            "BTC": {"instrument_type": "crypto", "paper_tradeable": True},
            "NEWIPO": {
                "coin": "NEWIPO",
                "venue_symbol": "xyz:NEWIPO",
                "dex": "xyz",
                "market_type": "perp",
                "instrument_type": "equity",
                "categories": ["pre_ipo", "ai_infra"],
                "shortable": True,
                "paper_tradeable": True,
                "live_tradeable": True,
            },
            "BABY": {
                "coin": "BABY",
                "venue_symbol": "xyz:BABY",
                "dex": "xyz",
                "market_type": "perp",
                "instrument_type": "equity",
                "categories": ["growth"],
                "shortable": True,
                "paper_tradeable": True,
                "live_tradeable": True,
            },
        }
        agent_module.build_hyperliquid_market_cap_watchlist = lambda **kwargs: {
            "coins": ["NEWIPO"],
            "records": [{"coin": "NEWIPO", "market_cap_usd": 5_000_000_000.0}],
        }

        added = agent._maybe_sync_tradexyz_listing_universe(force=True)

        assert added == ["NEWIPO"]
        assert "NEWIPO" in agent._analysis_coins
        assert "BABY" not in agent._analysis_coins
        assert "BABY" not in agent._tradable_coin_set
    finally:
        agent_module.get_hyperliquid_market_catalog = original_catalog
        agent_module.build_hyperliquid_market_cap_watchlist = original_builder


def test_market_universe_filters_hyperliquid_large_caps_into_scout_watchlist() -> None:
    original_fetch = market_universe_module._fetch_coingecko_market_caps
    original_equity_fetch = market_universe_module._fetch_equity_market_caps
    original_catalog = market_universe_module.get_hyperliquid_market_catalog
    original_active = market_universe_module.hyperliquid_market_is_active
    try:
        market_universe_module._fetch_coingecko_market_caps = lambda pages: [
            {"symbol": "btc", "name": "Bitcoin", "market_cap": 1_000_000_000_000, "market_cap_rank": 1, "price_change_percentage_24h": 2.0},
            {"symbol": "xrp", "name": "XRP", "market_cap": 50_000_000_000, "market_cap_rank": 4, "price_change_percentage_24h": 3.0},
            {"symbol": "doge", "name": "Dogecoin", "market_cap": 15_000_000_000, "market_cap_rank": 8, "price_change_percentage_24h": 1.0},
            {"symbol": "abc", "name": "Abc", "market_cap": 500_000_000, "market_cap_rank": 999, "price_change_percentage_24h": 0.0},
        ]
        market_universe_module._fetch_equity_market_caps = lambda symbols: {}
        market_universe_module.get_hyperliquid_market_catalog = lambda force_refresh=False: {
            "BTC": {"market_type": "perp", "instrument_type": "crypto", "venue_symbol": "BTC"},
            "XRP": {"market_type": "perp", "instrument_type": "crypto", "venue_symbol": "XRP"},
            "DOGE": {"market_type": "perp", "instrument_type": "crypto", "venue_symbol": "DOGE"},
            "SP500": {"market_type": "perp", "instrument_type": "index", "venue_symbol": "SPX"},
        }
        market_universe_module.hyperliquid_market_is_active = lambda coin: coin != "DOGE"
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = market_universe_module.build_hyperliquid_market_cap_watchlist(
                min_market_cap_usd=1_000_000_000.0,
                active_only=True,
                max_coins=10,
                cache_path=Path(tmpdir) / "market_cap_universe.json",
                force_refresh=True,
            )
        assert payload["coins"] == ["BTC", "XRP"]
    finally:
        market_universe_module._fetch_coingecko_market_caps = original_fetch
        market_universe_module._fetch_equity_market_caps = original_equity_fetch
        market_universe_module.get_hyperliquid_market_catalog = original_catalog
        market_universe_module.hyperliquid_market_is_active = original_active


def test_market_universe_includes_large_cap_hyperliquid_equities() -> None:
    original_fetch = market_universe_module._fetch_coingecko_market_caps
    original_equity_fetch = market_universe_module._fetch_equity_market_caps
    original_catalog = market_universe_module.get_hyperliquid_market_catalog
    original_active = market_universe_module.hyperliquid_market_is_active
    try:
        market_universe_module._fetch_coingecko_market_caps = lambda pages: []
        market_universe_module._fetch_equity_market_caps = lambda symbols: {
            "INTC": {"name": "Intel", "market_cap": 90_000_000_000, "market_cap_rank": None, "price_change_percentage_24h": 1.2},
            "HIMS": {"name": "Hims & Hers", "market_cap": 14_000_000_000, "market_cap_rank": None, "price_change_percentage_24h": 3.4},
        }
        market_universe_module.get_hyperliquid_market_catalog = lambda force_refresh=False: {
            "INTC": {"market_type": "perp", "instrument_type": "equity", "venue_symbol": "xyz:INTC"},
            "HIMS": {"market_type": "perp", "instrument_type": "equity", "venue_symbol": "xyz:HIMS"},
            "BTC": {"market_type": "perp", "instrument_type": "crypto", "venue_symbol": "BTC"},
        }
        market_universe_module.hyperliquid_market_is_active = lambda coin: coin != "HIMS"
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = market_universe_module.build_hyperliquid_market_cap_watchlist(
                min_market_cap_usd=1_000_000_000.0,
                active_only=True,
                max_coins=10,
                cache_path=Path(tmpdir) / "market_cap_universe.json",
                force_refresh=True,
            )
        assert payload["coins"] == ["INTC"]
        assert payload["records"][0]["venue_symbol"] == "xyz:INTC"
    finally:
        market_universe_module._fetch_coingecko_market_caps = original_fetch
        market_universe_module._fetch_equity_market_caps = original_equity_fetch
        market_universe_module.get_hyperliquid_market_catalog = original_catalog
        market_universe_module.hyperliquid_market_is_active = original_active


def test_reentry_watch_inherits_dynamic_trade_plan() -> None:
    manager = OrderManager()
    manager.schedule_reentry(
        coin="BTC",
        direction="LONG",
        entry_price=80.0,
        tp_price=120.0,
        size_usd=100.0,
        signal_score=70.0,
        trade_plan={"risk_pct": 2.0, "risk_reward_ratio": 2.5},
        entry_context={"trade_plan": {"risk_pct": 2.0, "risk_reward_ratio": 2.5}},
    )
    actions = manager.tick({"BTC": 100.0})
    assert actions and actions[0]["type"] == "place_limit"
    assert round(actions[0]["sl"], 2) == 98.00
    assert round(actions[0]["tp"], 2) == 105.00


def test_trade_logger_normalizes_legacy_headerless_log() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_log = Path(tmpdir) / "trades_log.csv"
        row = [
            "1", "BTC", "LONG", "2026-01-01 00:00", "2026-01-01 01:00", "60",
            "100", "105", "100", "2", "5", "5", "95", "110", "take_profit", "72", "WIN",
        ]
        temp_log.write_text(",".join(row) + "\n")

        original_log = trade_logger_module.LOG_FILE
        try:
            trade_logger_module.LOG_FILE = temp_log
            rows = trade_logger_module.read_closed_trades()
            assert len(rows) == 1
            assert rows[0]["coin"] == "BTC"
            assert temp_log.read_text().splitlines()[0].startswith("trade_id,coin,direction")
        finally:
            trade_logger_module.LOG_FILE = original_log


def test_trade_memory_can_hard_block_failing_direction_family() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        original_memory_file = trade_memory_module.MEMORY_FILE
        try:
            trade_memory_module.MEMORY_FILE = Path(tmpdir) / "trade_memory.json"
            memory = trade_memory_module.TradeMemory()
            for _ in range(4):
                memory.record_trade(
                    coin="ETH",
                    direction="SHORT",
                    signal_score=32.0,
                    entry_price=100.0,
                    exit_price=105.0,
                    exit_reason="stop_loss",
                    hold_minutes=30.0,
                    trend_context="UP",
                    market_regime="TRENDING",
                    dominant_regime="TREND",
                    volatility_label="NORMAL",
                    entry_context={"mtf_bias": "BULLISH", "confidence": "LOW"},
                )
            guard = memory.get_directional_guard("ETH", "SHORT")
            assert guard["hard_block"] is True, "repeated low-win-rate failures should embargo the direction"
            assert guard["hard_block_reason"], "hard block should explain why the setup family is embargoed"
        finally:
            trade_memory_module.MEMORY_FILE = original_memory_file


def test_decision_dataset_and_feature_store_capture_flat_decisions() -> None:
    original_decision_path = decision_dataset_module.DECISION_DATASET_JSONL
    original_feature_path = feature_store_module.FEATURE_STORE_JSONL
    with tempfile.TemporaryDirectory() as tmpdir:
        decision_dataset_module.DECISION_DATASET_JSONL = Path(tmpdir) / "decision_dataset.jsonl"
        feature_store_module.FEATURE_STORE_JSONL = Path(tmpdir) / "feature_store.jsonl"
        try:
            record = {
                "cycle_number": 42,
                "coin": "BTC",
                "stage": "guardrails_flat",
                "candidate_action": "LONG",
                "final_action": "FLAT",
                "decision_reason": "waiting for reclaim confirmation",
                "has_position": False,
                "current_position": "",
                "tradable": True,
                "execution_mode": "tradable",
                "executed": False,
                "blocked": True,
                "pending_limit": False,
                "signal_snapshot": {
                    "action": "FLAT",
                    "decision": "FLAT",
                    "score": 61.5,
                    "confidence": "MEDIUM",
                    "decision_reason": "waiting for reclaim confirmation",
                    "flat_reason": "waiting for reclaim confirmation",
                    "instrument_type": "crypto",
                    "candle_score": 58.0,
                    "news_score": 54.0,
                    "market_regime": "TREND",
                    "dominant_regime": "TREND",
                    "orderbook_score": 66.0,
                    "market_map_bias": "BULLISH",
                    "planned_risk_pct": 1.2,
                    "planned_reward_pct": 2.8,
                    "planned_risk_reward_ratio": 2.33,
                    "thesis_candidate_action": "LONG",
                    "thesis_state": "NO_TRADE",
                    "thesis_permitted": False,
                    "thesis_quality": "MEDIUM",
                    "thesis_alignment_points": 4.0,
                    "thesis_conflict_points": 2.0,
                    "thesis_conviction_score": 61.5,
                    "expectancy_probability": 0.56,
                    "expectancy_expected_r": 0.22,
                    "expectancy_uncertainty": 0.34,
                    "expectancy_score": 60.0,
                },
            }
            decision_dataset_module.append_decision(record)
            feature_store_module.append_decision_feature_row(record)

            decisions = decision_dataset_module.load_decisions()
            feature_rows = feature_store_module.load_feature_rows(row_type="decision")
            assert len(decisions) == 1, "decision dataset should capture flat cycle decisions"
            assert decisions[0]["final_action"] == "FLAT"
            assert len(feature_rows) == 1, "feature store should mirror decision rows"
            assert feature_rows[0]["features"]["score"] == 61.5
            assert feature_rows[0]["features"]["ctx_stage"] == "guardrails_flat"
            assert feature_rows[0]["labels"]["blocked"] is True
        finally:
            decision_dataset_module.DECISION_DATASET_JSONL = original_decision_path
            feature_store_module.FEATURE_STORE_JSONL = original_feature_path


def test_feature_store_captures_asset_state_and_guard_features() -> None:
    record = {
        "cycle_number": 99,
        "coin": "BTC",
        "stage": "signal_streak_wait",
        "candidate_action": "LONG",
        "final_action": "FLAT",
        "blocked": True,
        "executed": False,
        "pending_limit": False,
        "signal_snapshot": {
            "action": "FLAT",
            "decision": "FLAT",
            "score": 63.0,
            "confidence": "HIGH",
            "asset_state": "WAITING_CONFIRMATION",
            "decision_stage": "signal_streak_wait",
            "next_unblock_reason": "Need 1 more confirming cycle before entry.",
            "data_reliability_score": 71.0,
            "data_reliability_quality": "MEDIUM",
            "data_reliability_summary": "microstructure history is still thin",
            "data_reliability": {"permitted": True},
            "portfolio_theme": "CRYPTO_BETA",
            "portfolio_guard_summary": "CRYPTO_BETA already has a same-direction live idea; trimming size",
            "portfolio_guard_size_multiplier": 0.65,
            "portfolio_guard": {
                "permitted": True,
                "same_direction_exposure_pct": 11.4,
                "total_theme_exposure_pct": 12.0,
            },
        },
    }
    row = feature_store_module.build_decision_feature_row(record)
    assert row["features"]["asset_state"] == "waiting_confirmation"
    assert row["features"]["decision_stage"] == "signal_streak_wait"
    assert row["features"]["data_reliability_score"] == 71.0
    assert row["features"]["portfolio_guard_size_multiplier"] == 0.65


def test_record_decision_snapshot_promotes_major_catalyst_watch_stage() -> None:
    cfg = build_config()
    exchange = DryRunExchange(starting_balance_usd=1000.0)
    exchange.connect()
    agent = TradingAgent(cfg, [exchange])
    agent._cycle = 777
    agent._last_signals["AMZN"] = {
        "action": "FLAT",
        "decision": "FLAT",
        "score": 41.5,
        "confidence": "LOW",
        "reason": "Breakout is confirmed, but the long still lacks clean continuation.",
        "flat_reason": "Equity spot: prior reclaim slipped back below the trigger; waiting for price to hold above reclaim again",
        "decision_reason": "Breakout is confirmed, but the long still lacks clean continuation.",
        "execution_mode": "observation_only",
        "instrument_type": "equity",
        "news_score": 62.77,
        "news_catalyst_score": 3.75,
        "news_catalyst_summary": "platform anchor + partner attached + demand commitment",
        "market_map_bias": "BULLISH",
        "market_map_reclaim_confirmed": True,
        "market_map_live_reclaim": False,
        "market_map_reclaim_lost": True,
        "market_map_nearest_resistance": 253.36,
        "market_map_nearest_support": 250.0,
        "thesis_candidate_action": "LONG",
        "thesis_permitted": False,
        "asset_state": "OBSERVING",
        "asset_state_label": "Observing",
        "next_unblock_reason": "",
    }

    captured: dict[str, dict] = {}
    original_append_decision = decision_dataset_module.append_decision
    original_append_feature = feature_store_module.append_decision_feature_row
    try:
        decision_dataset_module.append_decision = lambda record: captured.setdefault("decision", dict(record))
        feature_store_module.append_decision_feature_row = lambda record: captured.setdefault("feature", dict(record))
        agent._record_decision_snapshot(
            "AMZN",
            portfolio_usd=10_000.0,
            stage="observation_only",
        )
    finally:
        decision_dataset_module.append_decision = original_append_decision
        feature_store_module.append_decision_feature_row = original_append_feature

    record = captured["decision"]
    assert record["stage"] == "major_catalyst_watch"
    assert record["asset_state"] == "MAJOR_CATALYST_WATCH"
    assert record["signal_snapshot"]["asset_state"] == "MAJOR_CATALYST_WATCH"
    assert "hold back above 253.36" in record["signal_snapshot"]["next_unblock_reason"].lower()


def test_decision_review_lab_flags_missed_winner() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "decision_dataset.jsonl").write_text(
            json.dumps(
                {
                    "decision_id": "1",
                    "coin": "BTC",
                    "stage": "data_reliability_block",
                    "candidate_action": "LONG",
                    "final_action": "FLAT",
                    "blocked": True,
                    "executed": False,
                    "pending_limit": False,
                    "recorded_at_ts": time.time(),
                    "signal_snapshot": {
                        "planned_risk_pct": 1.2,
                        "planned_reward_pct": 2.4,
                        "planned_risk_reward_ratio": 2.0,
                        "expectancy_probability": 0.62,
                        "expectancy_uncertainty": 0.22,
                        "expectancy_score": 68.0,
                        "confidence": "HIGH",
                        "thesis_quality": "HIGH",
                        "orderbook_breakout_state": "CONFIRMED_BULLISH_BREAKOUT",
                        "orderbook_interaction": "ABOVE_RESISTANCE",
                        "dominant_regime": "TREND",
                        "instrument_type": "crypto",
                    },
                }
            ) + "\n",
            encoding="utf-8",
        )
        original_label = decision_review_lab_module.precision_lab._label_episode
        try:
            decision_review_lab_module.precision_lab._label_episode = lambda row, **_kwargs: {**row, "outcome": 1}
            report = decision_review_lab_module.build_report(
                data_dir=data_dir,
                target_r=0.25,
                horizon_minutes=720,
                interval="5m",
                dedupe_minutes=30,
            )
            assert report["classifications"]["MISSED_WIN"] == 1
        finally:
            decision_review_lab_module.precision_lab._label_episode = original_label


def test_missed_move_lab_surfaces_recent_blockers() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "decision_dataset.jsonl").write_text(
            json.dumps(
                {
                    "decision_id": "1",
                    "coin": "AMZN",
                    "stage": "precision_cadence_block",
                    "candidate_action": "LONG",
                    "final_action": "FLAT",
                    "blocked": True,
                    "executed": False,
                    "pending_limit": False,
                    "decision_reason": "same family is cooling down for another ~120m",
                    "recorded_at_ts": time.time(),
                    "signal_snapshot": {
                        "planned_risk_pct": 1.1,
                        "planned_reward_pct": 3.0,
                        "planned_risk_reward_ratio": 2.7,
                        "expectancy_probability": 0.64,
                        "expectancy_uncertainty": 0.21,
                        "expectancy_score": 73.0,
                        "confidence": "HIGH",
                        "thesis_quality": "HIGH",
                        "orderbook_breakout_state": "PERSISTENT_BULLISH_BREAKOUT",
                        "orderbook_interaction": "ABOVE_RESISTANCE",
                        "dominant_regime": "TREND",
                        "instrument_type": "equity",
                        "flat_reason": "same family is cooling down for another ~120m",
                    },
                }
            ) + "\n",
            encoding="utf-8",
        )
        original_label = missed_move_lab_module.precision_lab._label_episode
        try:
            missed_move_lab_module.precision_lab._label_episode = lambda row, **_kwargs: {**row, "outcome": 1}
            report = missed_move_lab_module.build_report(
                data_dir=data_dir,
                target_r=0.25,
                horizon_minutes=720,
                interval="5m",
                dedupe_minutes=30,
            )
            assert report["summary"]["missed_win_count"] == 1
            assert report["top_missed_assets"][0]["coin"] == "AMZN"
            assert "cooling down" in report["recent_missed_moves"][0]["summary"]
        finally:
            missed_move_lab_module.precision_lab._label_episode = original_label


def test_missed_move_lab_builds_daily_top_mover_replay() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        now = time.time()

        def row(decision_id: str, ts: float, price: float, *, blocked: bool, stage: str) -> dict:
            return {
                "decision_id": decision_id,
                "coin": "INTC",
                "stage": stage,
                "candidate_action": "LONG",
                "final_action": "FLAT" if blocked else "LONG",
                "blocked": blocked,
                "executed": False,
                "pending_limit": False,
                "decision_reason": "waiting for post-event confirmation" if blocked else "tracking move",
                "recorded_at_ts": ts,
                "signal_snapshot": {
                    "live_price": price,
                    "planned_risk_pct": 1.0,
                    "planned_reward_pct": 3.0,
                    "planned_risk_reward_ratio": 3.0,
                    "expectancy_probability": 0.58,
                    "expectancy_uncertainty": 0.34,
                    "expectancy_score": 62.0,
                    "confidence": "MEDIUM",
                    "thesis_quality": "MEDIUM",
                    "instrument_type": "equity",
                    "news_catalyst_score": 4.2,
                    "news_event_score": 4.0,
                    "conviction_entry_event": False,
                    "flat_reason": "waiting for post-event confirmation" if blocked else "",
                },
            }

        (data_dir / "decision_dataset.jsonl").write_text(
            "\n".join(
                json.dumps(item)
                for item in [
                    row("1", now - 6 * 3600, 20.0, blocked=True, stage="precision_cadence_block"),
                    row("2", now - 60, 25.0, blocked=False, stage="analysis"),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        original_label = missed_move_lab_module.precision_lab._label_episode
        try:
            missed_move_lab_module.precision_lab._label_episode = lambda row, **_kwargs: {**row, "outcome": 1}
            report = missed_move_lab_module.build_report(
                data_dir=data_dir,
                target_r=0.25,
                horizon_minutes=720,
                interval="5m",
                dedupe_minutes=30,
            )
            replay = report["daily_top_mover_replay"]
            assert replay["top_movers"][0]["coin"] == "INTC"
            assert replay["top_movers"][0]["move_pct"] == 25.0
            assert replay["missed_top_movers"][0]["coin"] == "INTC"
            assert "confirmation" in replay["missed_top_movers"][0]["summary"]
        finally:
            missed_move_lab_module.precision_lab._label_episode = original_label


def test_challenger_model_reports_when_shadow_is_ready() -> None:
    cfg = build_config()
    cfg.trading.challenger_min_labeled_decisions = 4
    cfg.trading.challenger_min_win_rate_edge = 0.05
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        (data_dir / "precision_lab_report.json").write_text(
            json.dumps(
                {
                    "overall_win_rate": 0.56,
                    "labeled_episodes": 12,
                    "best_rules": [{"win_rate": 0.68, "samples": 6}],
                }
            ),
            encoding="utf-8",
        )
        (data_dir / "decision_review_report.json").write_text(
            json.dumps(
                {
                    "labeled_episodes": 8,
                    "classifications": {
                        "MISSED_WIN": 1,
                        "CORRECT_PASS": 4,
                        "GOOD_TRADE": 2,
                        "BAD_TRADE": 1,
                    },
                    "missed_families": [{"family": "BTC:LONG", "misses": 1}],
                }
            ),
            encoding="utf-8",
        )
        report = challenger_model_module.build_report(cfg, data_dir=data_dir)
        assert report["shadow_ready"] is True
        assert report["promote"] is True
        assert report["status"] == "CHALLENGER_READY"


def test_asset_dossier_builds_focus_assets_and_referee_context() -> None:
    state = {
        "last_cycle": "2026-04-21 20:00:00",
        "signals": {
            "BTC": {
                "action": "LONG",
                "execution_mode": "tradable",
                "asset_state": "WAITING_CONFIRMATION",
                "instrument_type": "crypto",
                "confidence": "HIGH",
                "score": 78.0,
                "expectancy_probability": 0.66,
                "expectancy_expected_r": 0.42,
                "live_price": 71000.0,
                "decision_reason": "Support-defense long is setting up cleanly.",
                "next_unblock_reason": "Need one more clean close above reclaim.",
                "market_map_bias": "BULLISH",
                "market_map_summary": "daily reclaim is holding",
                "narrative_summary": "ETF flow and risk appetite still support the move.",
                "analog_summary": "close winners mostly worked when reclaim held.",
            }
        },
        "positions": [],
        "config": {
            "coins": ["BTC"],
            "analysis_coins": ["BTC", "AMZN"],
            "instrument_types": {"BTC": "crypto", "AMZN": "equity"},
        },
    }
    market_map = {
        "coins": {
            "BTC": {
                "bias": "BULLISH",
                "supports": [68500],
                "resistances": [72500],
                "daily_close_long_above": [71500],
                "trade_mode": "Buy defended reclaims.",
            }
        }
    }
    report = asset_dossier_module.build_report(
        state=state,
        trades=[{"coin": "BTC", "agent_lesson": "Respect demand when reclaim holds."}],
        market_map=market_map,
        missed_move_report={"summary": {"missed_win_count": 2}, "top_missed_assets": [{"coin": "BTC", "misses": 2}]},
        llm_referee_report={"enabled": True, "verdicts": {"BTC": {"verdict": "SUPPORT", "summary": "Continuation still looks healthy."}}},
    )
    assert report["summary"]["focus_assets"][0] == "BTC"
    assert report["assets"]["BTC"]["dossier"]["playbook"] == "Buy defended reclaims."
    assert report["assets"]["BTC"]["missed_move_context"]["miss_count"] == 2
    assert report["assets"]["BTC"]["llm_referee"]["verdict"] == "SUPPORT"


def test_llm_referee_returns_disabled_without_api_key() -> None:
    cfg = build_config()
    cfg.trading.llm_referee_enabled = True
    cfg.trading.llm_referee_api_key = ""
    referee = llm_referee_module.LLMReferee(cfg.trading)
    result = referee.review_setup("BTC", {"action": "LONG"})
    assert result["enabled"] is False
    assert result["verdict"] == "DISABLED"


def test_llm_referee_parses_structured_openai_verdict() -> None:
    cfg = build_config()
    cfg.trading.llm_referee_enabled = True
    cfg.trading.llm_referee_api_key = "test-key"
    cfg.trading.llm_referee_model = "gpt-5.4"
    referee = llm_referee_module.LLMReferee(cfg.trading)
    original_post = llm_referee_module._post_responses_request
    try:
        llm_referee_module._post_responses_request = lambda **_kwargs: {
            "output_text": json.dumps(
                {
                    "verdict": "BLOCK",
                    "confidence": "HIGH",
                    "sentiment_bias": "MIXED",
                    "summary": "Narrative and structure are still fighting each other.",
                    "why_now": "Price reclaimed, but the catalyst read is still messy.",
                    "principal_risk": "failed reclaim",
                    "invalidation_focus": "lose 71,500",
                    "next_unblock": "wait for a cleaner daily close",
                    "execution_style": "wait",
                }
            )
        }
        result = referee.review_setup(
            "BTC",
            {
                "action": "LONG",
                "asset_state": "WAITING_CONFIRMATION",
                "score": 74.0,
                "expectancy_probability": 0.68,
                "expectancy_expected_r": 0.44,
                "confidence": "HIGH",
                "live_price": 71800.0,
            },
        )
        assert result["used"] is True
        assert result["verdict"] == "BLOCK"
        assert "messy" in result["why_now"].lower()
    finally:
        llm_referee_module._post_responses_request = original_post


def test_historical_analog_engine_blends_supportive_history() -> None:
    cfg = build_config()
    cfg.trading.analog_min_samples = 3
    cfg.trading.analog_hard_block_min_samples = 4
    cfg.trading.analog_similarity_floor = 0.50
    engine = analog_engine_module.HistoricalAnalogEngine(cfg.trading)

    original_loader = analog_engine_module.trade_dataset.load_closed_trades
    try:
        def _record(trade_id: str, pnl_pct: float, captured_r: float) -> dict:
            return {
                "trade_id": trade_id,
                "coin": "BTC",
                "direction": "LONG",
                "signal_score": 71.0,
                "pnl_pct": pnl_pct,
                "pnl_usd": pnl_pct * 10.0,
                "hold_minutes": 180.0,
                "outcome": "WIN" if pnl_pct > 0 else "LOSS",
                "exit_reason": "take_profit" if pnl_pct > 0 else "stop_loss",
                "entry_context": {
                    "score": 71.0,
                    "confidence": "HIGH",
                    "instrument_type": "crypto",
                    "mtf_bias": "BULLISH",
                    "candle_score": 66.0,
                    "candle_trend": "UP",
                    "news_score": 57.0,
                    "news_velocity": "MEDIUM",
                    "memory_adj": 0.5,
                    "rl_total_trades": 8,
                    "rl_win_rate": 62.0,
                    "rl_pattern_boost": 1.0,
                    "market_regime": "TREND",
                    "dominant_regime": "TREND",
                    "volatility_label": "NORMAL",
                    "foc_score": 58.0,
                    "funding_label": "NORMAL",
                    "orderbook_score": 72.0,
                    "market_map_bias": "BULLISH",
                    "market_map_summary": "reclaim held",
                    "planned_risk_pct": 1.1,
                    "planned_reward_pct": 2.9,
                    "planned_risk_reward_ratio": 2.64,
                    "planned_stop_atr_multiple": 1.2,
                    "planned_target_atr_multiple": 2.8,
                    "planned_target_r_multiple": 2.3,
                    "stop_basis": "support",
                    "target_basis": "resistance",
                    "price_action_summary": "breakout retest",
                    "expectancy_probability": 0.62,
                    "expectancy_expected_r": 0.42,
                    "expectancy_uncertainty": 0.21,
                    "expectancy_score": 71.0,
                    "execution_mode": "tradable",
                },
                "thesis": {
                    "candidate_action": "LONG",
                    "state": "TRADEABLE",
                    "permitted": True,
                    "quality": "HIGH",
                    "alignment_points": 6.0,
                    "conflict_points": 1.0,
                    "conviction_score": 71.0,
                    "summary": "trend and orderbook aligned",
                },
                "trade_plan": {
                    "risk_pct": 1.1,
                    "reward_pct": 2.9,
                    "risk_reward_ratio": 2.64,
                    "stop_atr_multiple": 1.2,
                    "target_atr_multiple": 2.8,
                    "target_r_multiple": 2.3,
                    "stop_basis": "support",
                    "target_basis": "resistance",
                    "price_action_summary": "breakout retest",
                },
                "execution_quality": {
                    "score": 77.0,
                    "summary": "good depth",
                    "estimated_slippage_bps": 5.0,
                    "persistence_cycles": 3,
                },
                "plan_outcome": {
                    "captured_r_multiple": captured_r,
                },
            }

        analog_engine_module.trade_dataset.load_closed_trades = lambda limit=None: [
            _record("a1", 2.6, 1.4),
            _record("a2", 3.4, 1.8),
            _record("a3", 1.9, 1.1),
            _record("a4", 2.2, 1.3),
        ]

        current_signal = {
            "action": "LONG",
            "score": 70.0,
            "confidence": "HIGH",
            "instrument_type": "crypto",
            "candle_score": 65.0,
            "news_score": 56.0,
            "market_regime": "TREND",
            "dominant_regime": "TREND",
            "orderbook_score": 70.0,
            "orderbook_imbalance": 0.18,
            "market_map_bias": "BULLISH",
            "planned_risk_pct": 1.0,
            "planned_reward_pct": 2.8,
            "planned_risk_reward_ratio": 2.6,
            "planned_stop_atr_multiple": 1.2,
            "planned_target_atr_multiple": 2.7,
            "planned_target_r_multiple": 2.2,
            "stop_basis": "support",
            "target_basis": "resistance",
            "thesis_candidate_action": "LONG",
            "thesis_state": "TRADEABLE",
            "thesis_quality": "HIGH",
            "thesis_alignment_points": 6.0,
            "thesis_conflict_points": 1.0,
            "thesis_conviction_score": 70.0,
            "expectancy_probability": 0.60,
            "expectancy_expected_r": 0.35,
            "expectancy_uncertainty": 0.24,
            "expectancy_score": 69.0,
            "execution_mode": "tradable",
            "execution_quality_score": 75.0,
            "estimated_slippage_bps": 6.0,
        }

        analog = engine.evaluate("BTC", "LONG", current_signal)
        assert analog["supportive"] is True, "strong winning analogs should support similar live setups"
        assert analog["score_adjustment"] > 0.0
        blended = engine.blend_expectancy(
            {
                "permitted": True,
                "probability": 0.60,
                "expected_r": 0.35,
                "uncertainty": 0.24,
                "score": 69.0,
                "summary": "base expectancy is acceptable",
                "reasons": [],
                "blockers": [],
            },
            analog,
        )
        assert blended["probability"] > 0.60
        assert blended["score"] > 69.0
        assert blended["permitted"] is True
    finally:
        analog_engine_module.trade_dataset.load_closed_trades = original_loader


def test_agent_apply_analog_context_can_flatten_bad_setup() -> None:
    cfg = build_config()
    exchange = DryRunExchange(starting_balance_usd=1000.0)
    exchange.connect()
    live = TradingAgent(cfg, [exchange])
    live._last_signals["BTC"] = {
        "action": "LONG",
        "decision": "LONG",
        "score": 68.0,
        "confidence": "HIGH",
        "decision_reason": "breakout retest",
        "flat_reason": "",
        "instrument_type": "crypto",
        "planned_risk_pct": 1.0,
        "planned_reward_pct": 2.5,
        "planned_risk_reward_ratio": 2.5,
        "thesis_candidate_action": "LONG",
        "thesis_state": "TRADEABLE",
        "thesis_quality": "HIGH",
        "thesis_alignment_points": 6.0,
        "thesis_conflict_points": 0.0,
        "thesis_conviction_score": 68.0,
        "expectancy_probability": 0.61,
        "expectancy_expected_r": 0.33,
        "expectancy_uncertainty": 0.22,
        "expectancy_score": 68.0,
        "execution_mode": "tradable",
    }
    signal = SimpleNamespace(
        action="LONG",
        score=68.0,
        confidence="HIGH",
        reason="breakout retest",
        flat_reason="",
        stop_loss_price=98.0,
        take_profit_price=105.0,
        trade_plan={},
        execution_plan={},
        thesis={
            "candidate_action": "LONG",
            "state": "TRADEABLE",
            "permitted": True,
            "quality": "HIGH",
            "alignment_points": 6.0,
            "conflict_points": 0.0,
            "conviction_score": 68.0,
            "summary": "trend and levels aligned",
            "reasons": [],
            "blockers": [],
        },
        expectancy={
            "permitted": True,
            "probability": 0.61,
            "expected_r": 0.33,
            "uncertainty": 0.22,
            "score": 68.0,
            "summary": "good baseline expectancy",
            "reasons": [],
            "blockers": [],
        },
    )

    class StubAnalogEngine:
        def evaluate(self, coin, action, signal_snapshot):
            return {
                "enabled": True,
                "verdict": "HARD_BLOCK",
                "sample_size": 9,
                "avg_similarity": 0.81,
                "reliability": 0.73,
                "win_rate": 0.22,
                "avg_pnl_pct": -2.1,
                "avg_captured_r": -0.9,
                "supportive": False,
                "adverse": True,
                "hard_block": True,
                "score_adjustment": -4.0,
                "probability_adjustment": -0.06,
                "expected_r_adjustment": -0.12,
                "uncertainty_adjustment": 0.08,
                "summary": "historical analog engine hard-blocked the setup",
                "top_matches": [],
            }

        def blend_expectancy(self, expectancy, analog, same_direction_position=False):
            payload = dict(expectancy)
            payload.update({
                "permitted": False,
                "score": 64.0,
                "probability": 0.55,
                "expected_r": 0.21,
                "uncertainty": 0.30,
                "summary": "historical analog engine hard-blocked the setup",
                "blockers": ["historical analog engine hard-blocked the setup"],
                "reasons": [],
            })
            return payload

    live._analog_engine = StubAnalogEngine()
    analog = live._apply_analog_context("BTC", signal, None)
    assert analog["hard_block"] is True
    assert signal.action == "FLAT", "hard-blocked analogs should flatten the live setup"
    assert "hard-blocked" in signal.flat_reason
    assert live._last_signals["BTC"]["analog_hard_block"] is True
    assert live._last_signals["BTC"]["analog_verdict"] == "HARD_BLOCK"


def test_precision_analog_guard_allows_event_starter_when_history_is_not_adverse() -> None:
    cfg = build_config()
    cfg.trading.precision_mode_enabled = True
    exchange = DryRunExchange(starting_balance_usd=1000.0)
    live = TradingAgent(cfg, [exchange])
    live._last_signals["GOOGL"] = {
        "analog_sample_size": 5,
        "analog_reliability": 0.69,
        "analog_win_rate": 0.62,
        "analog_hard_block": True,
        "analog_adverse": False,
        "analog_summary": "5 analogs with 62% win rate and positive average R",
    }
    signal = SimpleNamespace(
        action="LONG",
        score=65.1,
        confidence="HIGH",
        flat_reason="",
        reason="pre-event catalyst flow",
        thesis={
            "quality": "MEDIUM",
            "support_defense_long": True,
            "confirmed_breakout": True,
            "conviction_entry": {
                "active": True,
                "bypass_precision": True,
                "event_conviction": True,
                "style": "EVENT_STARTER",
            },
        },
        expectancy={
            "probability": 0.52,
            "expected_r": 0.28,
            "uncertainty": 0.57,
            "score": 47.0,
        },
        trade_plan={"risk_reward_ratio": 2.1},
    )
    live._apply_precision_analog_guard(
        "GOOGL",
        signal,
        orderbook_signal=SimpleNamespace(valid=True, score=67.0, breakout_state="CONFIRMED_BULLISH_BREAKOUT"),
        market_map_signal=SimpleNamespace(valid=True, favor_longs=True),
    )
    assert signal.action == "LONG"
    assert signal.flat_reason == ""


def test_analog_uncertainty_gate_allows_event_starter_when_not_hard_adverse() -> None:
    cfg = build_config()
    cfg.trading.precision_mode_enabled = True
    live = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=1000.0)])
    live._last_signals["META"] = {"coin": "META", "action": "LONG"}

    class StubAnalogEngine:
        def evaluate(self, coin, action, signal_snapshot):
            return {
                "enabled": True,
                "verdict": "MIXED",
                "sample_size": 5,
                "avg_similarity": 0.74,
                "reliability": 0.64,
                "win_rate": 0.55,
                "avg_pnl_pct": 0.1,
                "avg_captured_r": 0.05,
                "supportive": False,
                "adverse": False,
                "hard_block": False,
                "score_adjustment": 0.0,
                "probability_adjustment": 0.0,
                "expected_r_adjustment": 0.0,
                "uncertainty_adjustment": 0.08,
                "summary": "mixed analogs increase uncertainty",
                "top_matches": [],
            }

        def blend_expectancy(self, expectancy, analog, same_direction_position=False):
            payload = dict(expectancy)
            payload.update({
                "permitted": False,
                "score": 47.0,
                "probability": 0.89,
                "expected_r": 0.30,
                "uncertainty": 0.51,
                "summary": "uncertainty 0.51 is above 0.42",
                "blockers": ["uncertainty 0.51 is above 0.42"],
                "reasons": [],
            })
            return payload

    signal = SimpleNamespace(
        action="LONG",
        score=51.1,
        confidence="LOW",
        flat_reason="",
        reason="pre-event catalyst flow",
        thesis={
            "candidate_action": "LONG",
            "conviction_entry": {
                "active": True,
                "bypass_precision": True,
                "event_conviction": True,
                "style": "EVENT_STARTER",
            },
        },
        expectancy={
            "probability": 0.89,
            "expected_r": 0.22,
            "uncertainty": 0.43,
            "score": 48.0,
            "permitted": True,
        },
        trade_plan={"risk_reward_ratio": 2.0},
    )

    live._analog_engine = StubAnalogEngine()
    live._apply_analog_context("META", signal, None)

    assert signal.action == "LONG"
    assert signal.flat_reason == ""
    assert signal.expectancy["permitted"] is True
    assert signal.expectancy["blockers"] == []


def test_precision_mode_blocks_embargoed_coin_direction() -> None:
    cfg = build_config()
    cfg.trading.precision_mode_enabled = True
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)
    allowed, reason = strategy._passes_precision_mode(
        coin="SP500",
        action="SHORT",
        confidence="HIGH",
        thesis={
            "quality": "HIGH",
            "confirmed_breakout": True,
            "persistent_breakout": False,
            "support_defense_long": False,
        },
        expectancy={
            "probability": 0.94,
            "expected_r": 0.62,
            "uncertainty": 0.18,
        },
        trade_plan={"risk_reward_ratio": 2.4},
        orderbook_signal=SimpleNamespace(valid=True, score=24.0),
        market_map_signal=SimpleNamespace(valid=True, favor_shorts=True),
    )
    assert allowed is False, "embargoed coin-direction families should never pass precision mode"
    assert "embargoed" in reason


def test_precision_mode_allows_elite_breakout_long() -> None:
    cfg = build_config()
    cfg.trading.precision_mode_enabled = True
    strategy = AggressiveStrategy(cfg.trading, cfg.indicators)
    allowed, reason = strategy._passes_precision_mode(
        coin="BTC",
        action="LONG",
        confidence="HIGH",
        thesis={
            "quality": "HIGH",
            "confirmed_breakout": True,
            "persistent_breakout": False,
            "support_defense_long": False,
        },
        expectancy={
            "probability": 0.93,
            "expected_r": 0.54,
            "uncertainty": 0.18,
        },
        trade_plan={"risk_reward_ratio": 2.2},
        orderbook_signal=SimpleNamespace(valid=True, score=74.0),
        market_map_signal=SimpleNamespace(valid=True, favor_longs=True),
    )
    assert allowed is True, f"elite breakout long should pass precision mode, got: {reason}"


def test_precision_lab_collapses_repeated_rows_and_flags_toxic_families() -> None:
    now = time.time() - 7200
    rows = [
        {
            "decision_id": "spx-1",
            "coin": "SP500",
            "recorded_at_ts": now,
            "final_action": "SHORT",
            "signal_snapshot": {
                "planned_risk_pct": 1.0,
                "planned_reward_pct": 2.5,
                "planned_risk_reward_ratio": 2.5,
                "expectancy_probability": 0.94,
                "expectancy_uncertainty": 0.18,
                "expectancy_score": 72.0,
                "orderbook_score": 24.0,
                "confidence": "HIGH",
                "thesis_quality": "HIGH",
                "orderbook_breakout_state": "confirmed_bearish_breakdown",
                "orderbook_interaction": "at_resistance",
                "dominant_regime": "momentum",
                "instrument_type": "index",
            },
        },
        {
            "decision_id": "spx-dup",
            "coin": "SP500",
            "recorded_at_ts": now + 300,
            "final_action": "SHORT",
            "signal_snapshot": {
                "planned_risk_pct": 1.0,
                "planned_reward_pct": 2.5,
                "planned_risk_reward_ratio": 2.5,
                "expectancy_probability": 0.94,
                "expectancy_uncertainty": 0.18,
                "expectancy_score": 72.0,
                "orderbook_score": 24.0,
                "confidence": "HIGH",
                "thesis_quality": "HIGH",
                "orderbook_breakout_state": "confirmed_bearish_breakdown",
                "orderbook_interaction": "at_resistance",
                "dominant_regime": "momentum",
                "instrument_type": "index",
            },
        },
        {
            "decision_id": "spx-2",
            "coin": "SP500",
            "recorded_at_ts": now + 2400,
            "final_action": "SHORT",
            "signal_snapshot": {
                "planned_risk_pct": 1.0,
                "planned_reward_pct": 2.5,
                "planned_risk_reward_ratio": 2.5,
                "expectancy_probability": 0.93,
                "expectancy_uncertainty": 0.18,
                "expectancy_score": 71.0,
                "orderbook_score": 26.0,
                "confidence": "HIGH",
                "thesis_quality": "HIGH",
                "orderbook_breakout_state": "confirmed_bearish_breakdown",
                "orderbook_interaction": "at_resistance",
                "dominant_regime": "momentum",
                "instrument_type": "index",
            },
        },
        {
            "decision_id": "spx-3",
            "coin": "SP500",
            "recorded_at_ts": now + 4800,
            "final_action": "SHORT",
            "signal_snapshot": {
                "planned_risk_pct": 1.0,
                "planned_reward_pct": 2.5,
                "planned_risk_reward_ratio": 2.5,
                "expectancy_probability": 0.92,
                "expectancy_uncertainty": 0.18,
                "expectancy_score": 70.0,
                "orderbook_score": 25.0,
                "confidence": "HIGH",
                "thesis_quality": "HIGH",
                "orderbook_breakout_state": "confirmed_bearish_breakdown",
                "orderbook_interaction": "at_resistance",
                "dominant_regime": "momentum",
                "instrument_type": "index",
            },
        },
        {
            "decision_id": "btc-1",
            "coin": "BTC",
            "recorded_at_ts": now + 6000,
            "final_action": "LONG",
            "signal_snapshot": {
                "planned_risk_pct": 0.5,
                "planned_reward_pct": 1.5,
                "planned_risk_reward_ratio": 3.0,
                "expectancy_probability": 0.95,
                "expectancy_uncertainty": 0.14,
                "expectancy_score": 78.0,
                "orderbook_score": 76.0,
                "confidence": "HIGH",
                "thesis_quality": "MEDIUM",
                "orderbook_breakout_state": "confirmed_bullish_breakout",
                "orderbook_interaction": "at_support",
                "dominant_regime": "breakout",
                "instrument_type": "crypto",
            },
        },
        {
            "decision_id": "btc-dup",
            "coin": "BTC",
            "recorded_at_ts": now + 6180,
            "final_action": "LONG",
            "signal_snapshot": {
                "planned_risk_pct": 0.5,
                "planned_reward_pct": 1.5,
                "planned_risk_reward_ratio": 3.0,
                "expectancy_probability": 0.95,
                "expectancy_uncertainty": 0.14,
                "expectancy_score": 78.0,
                "orderbook_score": 76.0,
                "confidence": "HIGH",
                "thesis_quality": "MEDIUM",
                "orderbook_breakout_state": "confirmed_bullish_breakout",
                "orderbook_interaction": "at_support",
                "dominant_regime": "breakout",
                "instrument_type": "crypto",
            },
        },
    ]

    original_fetch = precision_lab_module.fetch_candles
    try:
        def fake_fetch(coin: str, interval: str = "5m", lookback: int = 100):
            timestamps = pd.to_datetime(
                [
                    now - 300,
                    now,
                    now + 300,
                    now + 2400,
                    now + 2700,
                    now + 4800,
                    now + 5100,
                    now + 6000,
                    now + 6300,
                ],
                unit="s",
                utc=True,
            )
            if coin == "SP500":
                return pd.DataFrame({
                    "timestamp": timestamps,
                    "open": [100.0] * len(timestamps),
                    "high": [101.6] * len(timestamps),
                    "low": [99.8] * len(timestamps),
                    "close": [101.2] * len(timestamps),
                    "volume": [1.0] * len(timestamps),
                })
            return pd.DataFrame({
                "timestamp": timestamps,
                "open": [100.0] * len(timestamps),
                "high": [100.5] * len(timestamps),
                "low": [99.85] * len(timestamps),
                "close": [100.3] * len(timestamps),
                "volume": [1.0] * len(timestamps),
            })

        precision_lab_module.fetch_candles = fake_fetch

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            dataset_path = data_dir / "decision_dataset.jsonl"
            dataset_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            report = precision_lab_module.build_report(
                data_dir=data_dir,
                target_r=0.25,
                horizon_minutes=180,
                interval="5m",
                dedupe_minutes=30,
            )

        assert report["episodes"] == 4, "repeated cycles should collapse into setup episodes"
        assert report["labeled_episodes"] == 4
        toxic = {item["family"] for item in report["toxic_families"]}
        assert "SP500:SHORT" in toxic, "repeated losing setup family should be flagged as toxic"
    finally:
        precision_lab_module.fetch_candles = original_fetch


def test_precision_entry_cadence_blocks_repeat_activity() -> None:
    cfg = build_config()
    cfg.trading.precision_mode_enabled = True
    exchange = DryRunExchange(starting_balance_usd=1000.0)
    exchange.connect()
    live = TradingAgent(cfg, [exchange])
    now = time.time()
    live._precision_entry_history = [
        {
            "ts": now - 600,
            "coin": "BTC",
            "action": "LONG",
            "family": "BTC:LONG:BREAKOUT_CONTINUATION",
            "mode": "market",
        }
    ]
    signal = SimpleNamespace(
        action="LONG",
        thesis={"archetype": "BREAKOUT_CONTINUATION"},
    )
    allowed, reason = live._check_precision_entry_cadence("BTC", signal)
    assert allowed is False, "repeat coin activity should be blocked while precision cooldown is active"
    assert "cooldown" in reason.lower()


def test_tradexyz_volume_summary_rolls_up_wallet_activity() -> None:
    fills = [
        {"coin": "xyz:GOOGL", "px": "100", "sz": "2", "side": "B", "time": 1_710_000_000_000},
        {"coin": "xyz:GOOGL", "px": "105", "sz": "1", "side": "A", "time": 1_710_100_000_000},
        {"coin": "xyz:AMD", "px": "150", "sz": "3", "side": "B", "time": 1_710_200_000_000},
    ]
    summary = tradexyz_volume_module.summarize_tradexyz_fills(
        "0x1111111111111111111111111111111111111111",
        fills,
        universe=[{"name": "xyz:GOOGL"}, {"name": "xyz:AMD"}],
        coverage={"request_count": 3, "start_time": "2024-01-01T00:00:00+00:00", "end_time": "2024-02-01T00:00:00+00:00"},
        identity={"role": "subAccount", "query_scope": "strict_address", "notes": ["strict"]},
    )

    assert summary["summary"]["total_volume_usd"] == 755.0
    assert summary["summary"]["market_count"] == 2
    assert summary["summary"]["fill_count"] == 3
    assert summary["summary"]["buy_volume_usd"] == 650.0
    assert summary["summary"]["sell_volume_usd"] == 105.0
    assert summary["markets"][0]["coin"] == "xyz:AMD"
    assert summary["markets"][0]["volume_usd"] == 450.0
    assert summary["identity"]["role"] == "subAccount"


def test_tradexyz_identity_inspection_rejects_agent_wallets() -> None:
    original_post = tradexyz_volume_module._post_info
    try:
        def fake_post(payload):
            if payload["type"] == "userRole":
                return {"role": "agent", "data": {"user": "0x2222222222222222222222222222222222222222"}}
            if payload["type"] == "userAbstraction":
                return "default"
            if payload["type"] == "userDexAbstraction":
                return None
            if payload["type"] == "subAccounts":
                return None
            raise AssertionError(f"unexpected payload: {payload}")

        tradexyz_volume_module._post_info = fake_post
        try:
            tradexyz_volume_module.inspect_wallet_identity("0x1111111111111111111111111111111111111111")
            raise AssertionError("agent wallets should be rejected")
        except ValueError as exc:
            assert "agent/api wallet" in str(exc).lower()
    finally:
        tradexyz_volume_module._post_info = original_post


def test_tradexyz_identity_inspection_keeps_strict_subaccount_scope() -> None:
    original_post = tradexyz_volume_module._post_info
    try:
        def fake_post(payload):
            if payload["type"] == "userRole":
                return {"role": "subAccount", "data": {"master": "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}
            if payload["type"] == "userAbstraction":
                return "dexAbstraction"
            if payload["type"] == "userDexAbstraction":
                return {"linkedUser": "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
            if payload["type"] == "subAccounts":
                return [{"subAccountUser": "0xcccccccccccccccccccccccccccccccccccccccc"}]
            raise AssertionError(f"unexpected payload: {payload}")

        tradexyz_volume_module._post_info = fake_post
        identity = tradexyz_volume_module.inspect_wallet_identity("0x1111111111111111111111111111111111111111")
        assert identity["query_scope"] == "strict_address"
        assert identity["role"] == "subAccount"
        assert "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in json.dumps(identity)
        assert any("does not intentionally roll up the master" in note for note in identity["notes"])
        assert any("dex-abstraction" in note.lower() for note in identity["notes"])
    finally:
        tradexyz_volume_module._post_info = original_post


def test_dashboard_tradexyz_volume_endpoint_returns_checker_payload() -> None:
    original_fetch = dashboard_module.tradexyz_volume.fetch_tradexyz_volume
    try:
        dashboard_module.tradexyz_volume.fetch_tradexyz_volume = lambda wallet: {
            "wallet": wallet,
            "identity": {"role": "user", "query_scope": "strict_address", "notes": []},
            "summary": {"total_volume_usd": 1234.56, "fill_count": 5, "market_count": 2, "tracked_markets": 10},
            "coverage": {"request_count": 4, "start_time": "2024-01-01T00:00:00+00:00", "end_time": "2024-02-01T00:00:00+00:00"},
            "markets": [{"coin": "xyz:GOOGL", "volume_usd": 1000.0, "fills": 3}],
            "checked_at": "2026-04-20T00:00:00+00:00",
            "dex": "xyz",
        }
        client = dashboard_module.app.test_client()
        response = client.get("/api/tradexyz-volume?wallet=0x1111111111111111111111111111111111111111")
        payload = response.get_json()
        assert response.status_code == 200
        assert payload["ok"] is True
        assert payload["summary"]["total_volume_usd"] == 1234.56
        assert payload["markets"][0]["coin"] == "xyz:GOOGL"
        assert payload["identity"]["query_scope"] == "strict_address"
    finally:
        dashboard_module.tradexyz_volume.fetch_tradexyz_volume = original_fetch


def test_execution_coach_prefers_passive_retest_entry() -> None:
    cfg = build_config()
    quality = {"permitted": True, "score": 82.0, "summary": "spread 3.2bps, depth 18.0x"}
    signal_snapshot = {
        "action": "LONG",
        "price": 100.06,
        "live_price": 100.06,
        "execution_plan": {
            "mode": "limit",
            "entry_price": 100.0,
            "limit_price": 100.0,
            "reason": "buying the defended retest near support",
        },
        "expectancy": {"probability": 0.68, "score": 71.0},
        "trade_plan": {"risk_reward_ratio": 2.4},
        "thesis": {"support_defense_long": True},
    }
    order = SimpleNamespace(price=100.0)
    coached = execution_coach_module.decide_execution(
        cfg.trading,
        coin="BTC",
        signal_snapshot=signal_snapshot,
        order=order,
        execution_quality=quality,
    )
    assert coached["verdict"] == "PASSIVE"
    assert coached["execution_plan"]["mode"] in {"limit", "maker_limit"}


def test_execution_coach_skips_stretched_nonurgent_entry() -> None:
    cfg = build_config()
    quality = {"permitted": True, "score": 78.0, "summary": "spread 5.0bps, depth 12.0x"}
    signal_snapshot = {
        "action": "LONG",
        "price": 101.2,
        "live_price": 101.2,
        "execution_plan": {
            "mode": "market",
            "entry_price": 100.0,
            "reason": "default aggressive entry",
        },
        "expectancy": {"probability": 0.59, "score": 61.0},
        "trade_plan": {"risk_reward_ratio": 2.0},
        "thesis": {},
        "orderbook_breakout_state": "NONE",
    }
    order = SimpleNamespace(price=101.2)
    coached = execution_coach_module.decide_execution(
        cfg.trading,
        coin="BTC",
        signal_snapshot=signal_snapshot,
        order=order,
        execution_quality=quality,
    )
    assert coached["verdict"] == "SKIP"
    assert "stretched" in coached["summary"].lower()


def test_playbook_distiller_rewrites_best_and_worst_families() -> None:
    cfg = build_config()
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        now = time.time()

        def append_trade(*, direction: str, regime: str, pnl_pct: float, idx: int) -> None:
            trade_dataset_module.append_closed_trade(
                {
                    "trade_id": idx + 1,
                    "coin": "BTC",
                    "direction": direction,
                    "opened_at_ts": now - 3600 * (idx + 2),
                    "closed_at_ts": now - 1800 * idx,
                    "hold_minutes": 120.0,
                    "entry_price": 100.0,
                    "exit_price": 100.0 * (1 + pnl_pct / 100.0),
                    "size_usd": 100.0,
                    "size_coin": 1.0,
                    "pnl_usd": round(pnl_pct, 4),
                    "pnl_pct": pnl_pct,
                    "exit_reason": "test",
                    "outcome": "WIN" if pnl_pct > 0 else "LOSS",
                    "signal_score": 70.0,
                    "entry_context": {
                        "instrument_type": "crypto",
                        "dominant_regime": regime,
                        "execution_coach_verdict": "PASSIVE" if direction == "LONG" else "CHASE",
                        "execution_plan": {"mode": "limit" if direction == "LONG" else "market"},
                        "reason": f"{direction} {regime}",
                    },
                    "trade_plan": {"risk_reward_ratio": 2.2},
                    "plan_outcome": {"captured_r_multiple": 1.4},
                },
                data_dir=data_dir,
            )

        append_trade(direction="LONG", regime="TREND", pnl_pct=4.2, idx=0)
        append_trade(direction="LONG", regime="TREND", pnl_pct=3.8, idx=1)
        append_trade(direction="LONG", regime="TREND", pnl_pct=2.9, idx=2)
        append_trade(direction="SHORT", regime="RANGING", pnl_pct=-2.1, idx=3)
        append_trade(direction="SHORT", regime="RANGING", pnl_pct=-1.7, idx=4)
        append_trade(direction="SHORT", regime="RANGING", pnl_pct=-3.0, idx=5)

        report = playbook_distiller_module.build_report(cfg, data_dir=data_dir)
        asset = report["assets"]["BTC"]
        assert asset["best_family"]["direction"] == "LONG"
        assert asset["best_family"]["regime"] == "TREND"
        assert asset["avoid_family"]["direction"] == "SHORT"
        assert asset["avoid_family"]["regime"] == "RANGING"
        assert "Lean into BTC LONG" in asset["playbook"]


def test_dashboard_snapshot_includes_playbook_distiller_report() -> None:
    snapshot = build_dashboard_snapshot(
        {"signals": {}, "positions": [], "config": {}, "cycle_number": 1},
        [],
        playbook_distiller_report={"summary": {"working_family_count": 2}, "assets": {"BTC": {"playbook": "Test"}}},
    )
    assert snapshot["playbook_distiller_report"]["summary"]["working_family_count"] == 2
    assert snapshot["playbook_distiller_report"]["assets"]["BTC"]["playbook"] == "Test"


def test_first_principles_view_sequences_fundamentals_before_price() -> None:
    cfg = build_config()
    cfg.trading.asset_category_map = {"GOOGL": ["mag7", "ai_infra"]}
    view = first_principles_module.build_first_principles_view(
        "GOOGL",
        {
            "action": "LONG",
            "score": 66.0,
            "instrument_type": "equity",
            "official_event_score": 3.0,
            "news_catalyst_score": 2.5,
            "analyst_revision_score": 1.2,
            "social_attention_score": 64.0,
            "social_attention_mentions": 5,
            "expectancy_probability": 0.57,
            "planned_stop_loss": 175.0,
        },
        cfg.trading,
    )
    assert [step["step"] for step in view["sequence"]] == ["Fundamentals", "Attention", "Flows", "Price"]
    assert view["direction"] == "LONG"
    assert view["fundamental_score"] >= 62
    assert "175" in view["wrong_if"]


def test_social_attention_scans_configured_public_feeds() -> None:
    original_fetch = social_attention_module._fetch_source_text
    social_attention_module._CACHE.clear()
    try:
        social_attention_module._fetch_source_text = lambda url, timeout: "$GOOGL breakout long. AI demand strong. $GOOGL upgrade."
        cfg = SimpleNamespace(
            use_social_attention=True,
            social_attention_sources=["https://example.test/feed"],
            social_attention_max_sources=2,
            social_attention_timeout_seconds=1.0,
            social_attention_cache_seconds=1,
        )
        signal = social_attention_module.get_social_attention_signal("GOOGL", cfg)
        assert signal.valid is True
        assert signal.mentions >= 2
        assert signal.score > 50
        assert signal.sentiment == "BULLISH"
    finally:
        social_attention_module._fetch_source_text = original_fetch
        social_attention_module._CACHE.clear()


def test_performance_edges_summarize_what_pays_and_hurts() -> None:
    rows = [
        {"trade_id": "1", "coin": "GOOGL", "direction": "LONG", "signal_score": "74", "duration_mins": "1800", "pnl_usd": "8", "result": "WIN"},
        {"trade_id": "2", "coin": "GOOGL", "direction": "LONG", "signal_score": "72", "duration_mins": "2200", "pnl_usd": "6", "result": "WIN"},
        {"trade_id": "3", "coin": "META", "direction": "LONG", "signal_score": "71", "duration_mins": "1600", "pnl_usd": "4", "result": "WIN"},
        {"trade_id": "4", "coin": "BTC", "direction": "LONG", "signal_score": "66", "duration_mins": "20", "pnl_usd": "-2", "result": "LOSS"},
        {"trade_id": "5", "coin": "ETH", "direction": "LONG", "signal_score": "67", "duration_mins": "30", "pnl_usd": "-3", "result": "LOSS"},
        {"trade_id": "6", "coin": "SOL", "direction": "LONG", "signal_score": "68", "duration_mins": "25", "pnl_usd": "-1", "result": "LOSS"},
    ]
    report = performance_intelligence_module.build_performance_edges(rows, min_samples=3)
    assert report["summary"]["total_closed"] == 6
    assert report["working_edges"]
    assert report["failing_edges"]
    assert "Lean into" in " ".join(report["lessons"])
    assert "Avoid" in " ".join(report["lessons"])


def test_dashboard_snapshot_includes_performance_edges() -> None:
    snapshot = build_dashboard_snapshot(
        {
            "signals": {},
            "positions": [],
            "config": {},
            "cycle_number": 1,
            "performance_edges": {"summary": {"top_lesson": "Lean into LONG score 70-75"}},
        },
        [],
    )
    assert snapshot["performance_edges"]["summary"]["top_lesson"].startswith("Lean into")


def test_first_principles_guard_blocks_marginal_price_only_entry() -> None:
    cfg = build_config()
    cfg.trading.first_principles_guard_enabled = True
    cfg.trading.performance_edge_guard_enabled = False
    cfg.trading.decision_dataset_enabled = False
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=1000.0)])
    signal = SimpleNamespace(
        action="LONG",
        score=66.0,
        confidence="MEDIUM",
        reason="marginal price-only long",
        flat_reason="",
        stop_loss_price=95.0,
        take_profit_price=120.0,
        trade_plan={},
        thesis={"state": "ACTIVE", "permitted": True, "conviction_score": 66.0},
        expectancy={"probability": 0.51, "expected_r": 0.05, "uncertainty": 0.40, "score": 54.0},
        execution_plan={},
    )
    agent._last_signals["BTC"] = {
        "action": "LONG",
        "score": 66.0,
        "instrument_type": "crypto",
        "planned_stop_loss": 95.0,
        "expectancy_probability": 0.51,
        "news_event_score": 0.0,
        "news_catalyst_score": 0.0,
        "social_attention_score": 50.0,
    }
    assert agent._first_principles_entry_guard("BTC", signal, portfolio_usd=1000.0, current_position=None) is False
    assert signal.action == "FLAT"
    assert "first-principles guard" in signal.reason


def test_first_principles_guard_allows_marginal_event_attention_starter() -> None:
    cfg = build_config()
    cfg.trading.first_principles_guard_enabled = True
    cfg.trading.performance_edge_guard_enabled = False
    cfg.trading.decision_dataset_enabled = False
    agent = TradingAgent(cfg, [DryRunExchange(starting_balance_usd=1000.0)])
    signal = SimpleNamespace(
        action="LONG",
        score=66.0,
        confidence="MEDIUM",
        reason="small event starter",
        flat_reason="",
        stop_loss_price=175.0,
        take_profit_price=220.0,
        trade_plan={},
        thesis={
            "state": "CONVICTION_ENTRY",
            "permitted": True,
            "conviction_score": 66.0,
            "conviction_entry": {"active": True, "event_conviction": True},
        },
        expectancy={"probability": 0.55, "expected_r": 0.18, "uncertainty": 0.36, "score": 58.0},
        execution_plan={},
    )
    agent._last_signals["GOOGL"] = {
        "action": "LONG",
        "score": 66.0,
        "instrument_type": "equity",
        "asset_categories": ["mag7", "ai_infra"],
        "planned_stop_loss": 175.0,
        "expectancy_probability": 0.55,
        "official_event_score": 3.0,
        "news_catalyst_score": 2.6,
        "analyst_revision_score": 1.0,
        "social_attention_score": 64.0,
        "social_attention_mentions": 4,
        "conviction_entry": {"active": True, "event_conviction": True},
        "conviction_entry_event": True,
    }
    assert agent._first_principles_entry_guard("GOOGL", signal, portfolio_usd=1000.0, current_position=None) is True


def run_all() -> None:
    test_checkpoint_recovery()
    print("PASS checkpoint recovery")
    test_checkpoint_recovery_rehydrates_dry_run_pending_limits()
    print("PASS dry-run pending limit recovery")
    test_execute_order_stops_after_first_success()
    print("PASS single-exchange execution")
    test_checkpoint_recovery_skips_unsupported_state()
    print("PASS recovery trade-universe filtering")
    test_unsupported_symbols_fail_trade_universe_validation()
    print("PASS unsupported symbols validation")
    test_analysis_watchlist_keeps_non_tradable_assets_out_of_execution_universe()
    print("PASS analysis watchlist separation")
    test_supported_watchlist_assets_are_promoted_into_tradeable_universe()
    print("PASS watchlist promotion")
    test_inactive_hyperliquid_symbols_stay_out_of_tradeable_universe_when_fast_promotion_is_disabled()
    print("PASS active market filtering override")
    test_inactive_supported_hyperliquid_symbols_are_armed_for_execution_by_default()
    print("PASS default execution arming")
    test_live_spot_opt_in_includes_active_equities_in_supported_universe()
    print("PASS live spot opt-in universe")
    test_lighter_promotes_growth_and_macro_symbols_into_tradeable_universe()
    print("PASS lighter universe expansion")
    test_dynamic_trade_plan_is_attached_to_signal()
    print("PASS dynamic trade planning")
    test_orderbook_support_blocks_weak_short_into_demand()
    print("PASS orderbook support guard")
    test_support_defense_long_promotes_defended_reclaim_setup()
    print("PASS support-defense long promotion")
    test_confirmed_breakout_can_override_neutral_supply_map_for_support_defense_long()
    print("PASS support-defense breakout map override")
    test_confirmed_breakout_can_promote_borderline_long()
    print("PASS confirmed breakout promotion")
    test_probing_breakout_does_not_override_nearby_resistance()
    print("PASS probing breakout guard")
    test_crypto_news_falls_back_to_google_when_cryptopanic_is_unavailable()
    print("PASS crypto news fallback")
    test_macro_news_filters_irrelevant_cross_ticker_headlines()
    print("PASS macro news relevance filter")
    test_macro_news_returns_neutral_when_no_asset_specific_headlines_exist()
    print("PASS macro news neutral fallback")
    test_macro_news_recognizes_major_platform_customer_catalyst()
    print("PASS macro news catalyst checklist")
    test_macro_news_merges_pre_event_intc_catalyst_flow()
    print("PASS macro news pre-event INTC catalyst")
    test_macro_news_recognizes_cerebras_pre_ipo_listing_catalyst()
    print("PASS macro news Cerebras pre-IPO catalyst")
    test_equity_event_feed_collects_ir_sec_options_and_analyst_revisions()
    print("PASS equity event feed bundle")
    test_equity_event_feed_uses_nasdaq_fallback_when_yahoo_is_unavailable()
    print("PASS equity event feed Nasdaq fallback")
    test_macro_news_adds_upcoming_mag7_earnings_calendar_when_feeds_are_sparse()
    print("PASS macro news MAG7 earnings calendar")
    test_narrative_signal_boosts_major_catalyst_and_blocks_fading_it()
    print("PASS narrative catalyst boost")
    test_market_data_reuses_stale_yahoo_candles_when_live_fetch_fails()
    print("PASS stale market data fallback")
    test_supported_hyperliquid_market_does_not_fallback_to_yahoo_when_venue_is_empty()
    print("PASS Hyperliquid-only candle path")
    test_stale_hyperliquid_candles_are_rejected_for_supported_market()
    print("PASS stale Hyperliquid candle rejection")
    test_price_diagnostics_label_trade_xyz_and_flag_reference_spread()
    print("PASS price diagnostics source label")
    test_reference_quote_does_not_replace_executable_price_cache()
    print("PASS reference quote cache isolation")
    test_thesis_gate_blocks_high_score_range_compression_setup()
    print("PASS thesis no-trade gate")
    test_expectancy_gate_rejects_thin_edge_setup()
    print("PASS expectancy gate")
    test_strategy_allows_pre_event_equity_starter_below_trigger()
    print("PASS pre-event equity starter")
    test_strategy_allows_mag7_earnings_calendar_starters_when_conviction_is_shaky()
    print("PASS MAG7 earnings calendar starters")
    test_execution_plan_prefers_limit_entry_on_defended_support()
    print("PASS planned limit execution")
    test_agent_uses_completed_candles_for_conviction_but_live_price_for_execution()
    print("PASS completed-candle conviction split")
    test_scale_in_does_not_mutate_before_fill()
    print("PASS scale-in accounting")
    test_immediate_limit_scale_in_updates_open_trade_record()
    print("PASS immediate limit scale-in logging")
    test_pending_limit_scale_in_updates_open_trade_record()
    print("PASS pending limit scale-in logging")
    test_order_sizing_scales_with_conviction_and_tempers_euphoria()
    print("PASS conviction sizing")
    test_narrative_gate_blocks_event_risk_without_exceptional_expectancy()
    print("PASS narrative event gate")
    test_backtest_summary_includes_baselines()
    print("PASS backtest baselines")
    test_failed_close_keeps_position_open()
    print("PASS close failure safety")
    test_preflight_reports_missing_live_bootstrap()
    print("PASS preflight live bootstrap check")
    test_live_config_validation_requires_notifications()
    print("PASS live notification validation")
    test_live_promotion_gate_blocks_weak_metrics()
    print("PASS live promotion gate block")
    test_live_promotion_gate_passes_strong_metrics()
    print("PASS live promotion gate pass")
    test_dashboard_kill_endpoint_sets_control_state()
    print("PASS dashboard kill endpoint")
    test_dashboard_state_prefers_canonical_snapshot()
    print("PASS dashboard canonical snapshot")
    test_dashboard_refreshes_snapshot_when_state_changes()
    print("PASS dashboard snapshot refresh")
    test_dashboard_loads_prebuilt_snapshot_without_rehydrating()
    print("PASS dashboard prebuilt snapshot fast path")
    test_hosted_dashboard_bundle_matches_local_template()
    print("PASS hosted dashboard bundle sync")
    test_dashboard_template_compacts_daily_view_and_hides_support_pending()
    print("PASS dashboard compact daily view")
    test_local_dashboard_serves_hosted_bundle()
    print("PASS local dashboard hosted bundle")
    test_install_launchagent_preserves_learning_datasets()
    print("PASS runtime sync learning preservation")
    test_market_map_signal_respects_operator_daily_levels()
    print("PASS daily market map signal")
    test_effective_market_map_auto_maps_tracked_assets()
    print("PASS auto market map coverage")
    test_trade_review_feedback_hard_blocks_repeated_bad_thesis()
    print("PASS operator review hard block")
    test_dashboard_snapshot_includes_market_map_and_trade_reviews()
    print("PASS dashboard market-map snapshot")
    test_dashboard_snapshot_backfills_stock_desks_from_runtime_defaults()
    print("PASS dashboard stock desk backfill")
    test_default_stock_categories_keep_mag7_complete()
    print("PASS default stock category completeness")
    test_default_crypto_category_includes_mon()
    print("PASS default crypto category includes MON")
    test_tradexyz_pre_ipo_cerebras_defaults_to_event_theme()
    print("PASS TradeXYZ Cerebras pre-IPO defaults")
    test_tradexyz_latest_launches_are_first_class_defaults()
    print("PASS TradeXYZ latest listing defaults")
    test_proactive_intelligence_builds_full_research_stack()
    print("PASS proactive intelligence research stack")
    test_proactive_starter_execution_opens_capped_event_orders()
    print("PASS proactive starter execution caps")
    test_thesis_runner_defers_take_profit_and_extends_target()
    print("PASS thesis runner TP deferral")
    test_thesis_runner_still_honors_stop_loss()
    print("PASS thesis runner stop-loss integrity")
    test_thesis_runner_blocks_time_stop_for_multi_day_event_hold()
    print("PASS thesis runner multi-day time-stop bypass")
    test_thesis_runner_still_honors_hard_structure_invalidation()
    print("PASS thesis runner hard invalidation integrity")
    test_stale_adverse_exit_cuts_multi_day_loser_without_killing_runners()
    print("PASS stale adverse loser invalidation")
    test_winner_stickiness_blocks_conviction_churn_exit()
    print("PASS winner stickiness churn guard")
    test_pair_trade_book_builds_equity_long_crypto_short_overlay()
    print("PASS pair trade overlay book")
    test_setup_quality_guard_blocks_toxic_reversal_family()
    print("PASS setup quality toxic-family guard")
    test_north_star_guard_blocks_marginal_recovery_entries()
    print("PASS north-star marginal entry block")
    test_north_star_guard_allows_event_starter_but_trims_size()
    print("PASS north-star event starter trim")
    test_north_star_guard_cancels_stale_pending_limit_before_poll()
    print("PASS north-star pending stale limit cancel")
    test_north_star_guard_resizes_event_pending_limit_before_poll()
    print("PASS north-star pending event limit resize")
    test_dashboard_snapshot_includes_proactive_trader_report()
    print("PASS proactive dashboard snapshot")
    test_dashboard_snapshot_surfaces_exact_next_setup_blocker()
    print("PASS dashboard exact next setup blocker")
    test_dashboard_snapshot_ignores_opposite_direction_threshold_in_blocker()
    print("PASS dashboard directional blocker selection")
    test_dashboard_snapshot_normalizes_directional_breakdown_wording()
    print("PASS dashboard directional blocker wording")
    test_dashboard_snapshot_includes_trade_logic_and_learning_summary()
    print("PASS dashboard learning summary")
    test_dashboard_snapshot_includes_asset_dossiers_and_referee_reports()
    print("PASS dashboard intelligence snapshot")
    test_dashboard_action_board_uses_asset_state_and_next_unblock_reason()
    print("PASS dashboard asset-state action board")
    test_dashboard_action_board_calibrates_reclaim_odds_from_decision_history()
    print("PASS dashboard calibrated reclaim odds")
    test_dashboard_action_board_uses_major_catalyst_watch_label_and_unblock_reason()
    print("PASS dashboard major catalyst watch")
    test_dashboard_action_board_builds_friction_stack_catalyst_rail_and_lead_reason()
    print("PASS dashboard friction stack and catalyst rail")
    test_dashboard_snapshot_canonicalizes_inactive_control_and_empty_review_shape()
    print("PASS dashboard canonical state shape")
    test_dashboard_market_map_and_review_endpoints_roundtrip()
    print("PASS dashboard market-map/review endpoints")
    test_hosted_state_sync_can_publish_snapshot_to_git_branch()
    print("PASS hosted dashboard git fallback sync")
    test_hosted_state_sync_force_updates_generated_branch()
    print("PASS hosted dashboard git force sync")
    test_dashboard_remote_fallback_still_publishes_git_snapshot()
    print("PASS dashboard fallback publishes git snapshot")
    test_trade_memory_records_richer_loss_reasoning()
    print("PASS richer RL loss reasoning")
    test_trade_memory_directional_pause_and_guard()
    print("PASS directional RL guardrails")
    test_mtf_fail_closed_blocks_when_confirmation_is_unavailable()
    print("PASS fail-closed MTF safety")
    test_execution_quality_gate_blocks_thin_unstable_orderbooks()
    print("PASS execution-quality gate")
    test_execution_quality_can_fall_back_to_passive_rescue_limit()
    print("PASS passive execution rescue")
    test_pending_limit_can_reprice_when_book_drifts()
    print("PASS pending limit repricing")
    test_pending_limit_can_escalate_to_market_on_clean_breakout()
    print("PASS pending limit escalation")
    test_asset_state_machine_reports_confirmation_wait_clearly()
    print("PASS asset state machine")
    test_asset_state_machine_promotes_major_catalyst_reclaim_watch()
    print("PASS major catalyst asset state")
    test_data_reliability_blocks_stale_incoherent_setup()
    print("PASS data reliability gate")
    test_data_reliability_warns_without_blocking_wide_reference_price_spread()
    print("PASS reference spread reliability warning")
    test_portfolio_guard_blocks_theme_stacking_and_trims_secondary_exposure()
    print("PASS portfolio correlation guard")
    test_event_risk_budget_trims_and_blocks_crowded_pre_event_starters()
    print("PASS event risk budget")
    test_pre_ipo_symbols_automatically_use_tighter_event_budget()
    print("PASS pre-IPO event budget caps")
    test_background_orderbook_feed_enriches_signal_with_persistence_history()
    print("PASS background orderbook persistence history")
    test_background_orderbook_feed_detects_intracycle_breakout_between_agent_cycles()
    print("PASS background orderbook breakout detection")
    test_agent_bootstraps_background_orderbook_feed()
    print("PASS background orderbook feed bootstrap")
    test_lighter_read_auth_headers_gracefully_fallback_without_credentials()
    print("PASS lighter read auth fallback")
    test_market_data_attaches_lighter_auth_header_to_read_requests()
    print("PASS lighter market data auth headers")
    test_orderbook_reader_uses_hyperliquid_l2_snapshot()
    print("PASS Hyperliquid orderbook reader")
    test_dry_run_blocks_short_on_long_only_spot_symbol()
    print("PASS long-only spot short block")
    test_hyperliquid_limit_order_returns_resting_order_id()
    print("PASS Hyperliquid limit order support")
    test_hyperliquid_market_catalog_expands_unknown_live_perps()
    print("PASS Hyperliquid market catalog expansion")
    test_hyperliquid_market_catalog_discovers_unknown_tradexyz_equities()
    print("PASS Hyperliquid Trade.xyz stock discovery")
    test_hyperliquid_market_catalog_enables_full_tradexyz_catalog_and_prefers_perps()
    print("PASS full Trade.xyz executable catalog")
    test_apply_dynamic_analysis_universe_auto_adds_supported_stocks()
    print("PASS supported stock auto-promotion")
    test_apply_dynamic_analysis_universe_gates_tradexyz_equities_by_market_cap()
    print("PASS Trade.xyz market-cap execution gate")
    test_agent_runtime_tradexyz_listing_sync_onboards_new_symbols()
    print("PASS runtime TradeXYZ listing sync")
    test_agent_runtime_tradexyz_listing_sync_skips_low_cap_equities()
    print("PASS runtime TradeXYZ low-cap skip")
    test_market_universe_filters_hyperliquid_large_caps_into_scout_watchlist()
    print("PASS market-cap scout universe")
    test_reentry_watch_inherits_dynamic_trade_plan()
    print("PASS re-entry dynamic trade plan")
    test_trade_logger_normalizes_legacy_headerless_log()
    print("PASS legacy trade log normalization")
    test_trade_memory_can_hard_block_failing_direction_family()
    print("PASS directional hard embargo")
    test_decision_dataset_and_feature_store_capture_flat_decisions()
    print("PASS decision dataset + feature store")
    test_feature_store_captures_asset_state_and_guard_features()
    print("PASS feature store state features")
    test_record_decision_snapshot_promotes_major_catalyst_watch_stage()
    print("PASS major catalyst decision stage")
    test_decision_review_lab_flags_missed_winner()
    print("PASS decision review lab")
    test_missed_move_lab_surfaces_recent_blockers()
    print("PASS missed move lab")
    test_missed_move_lab_builds_daily_top_mover_replay()
    print("PASS missed move daily top-mover replay")
    test_challenger_model_reports_when_shadow_is_ready()
    print("PASS challenger model")
    test_asset_dossier_builds_focus_assets_and_referee_context()
    print("PASS asset dossier")
    test_execution_coach_prefers_passive_retest_entry()
    print("PASS execution coach passive retest")
    test_execution_coach_skips_stretched_nonurgent_entry()
    print("PASS execution coach stretched skip")
    test_playbook_distiller_rewrites_best_and_worst_families()
    print("PASS playbook distiller")
    test_dashboard_snapshot_includes_playbook_distiller_report()
    print("PASS dashboard playbook distiller snapshot")
    test_first_principles_view_sequences_fundamentals_before_price()
    print("PASS first-principles reasoning sequence")
    test_social_attention_scans_configured_public_feeds()
    print("PASS social attention feed scan")
    test_performance_edges_summarize_what_pays_and_hurts()
    print("PASS performance edge summary")
    test_dashboard_snapshot_includes_performance_edges()
    print("PASS dashboard performance edge snapshot")
    test_first_principles_guard_blocks_marginal_price_only_entry()
    print("PASS first-principles marginal block")
    test_first_principles_guard_allows_marginal_event_attention_starter()
    print("PASS first-principles event starter")
    test_llm_referee_returns_disabled_without_api_key()
    print("PASS LLM referee disabled fallback")
    test_llm_referee_parses_structured_openai_verdict()
    print("PASS LLM referee structured output")
    test_historical_analog_engine_blends_supportive_history()
    print("PASS historical analog engine")
    test_agent_apply_analog_context_can_flatten_bad_setup()
    print("PASS analog integration flattening")
    test_precision_analog_guard_allows_event_starter_when_history_is_not_adverse()
    print("PASS event starter analog bypass")
    test_analog_uncertainty_gate_allows_event_starter_when_not_hard_adverse()
    print("PASS event starter analog uncertainty bypass")
    test_precision_mode_blocks_embargoed_coin_direction()
    print("PASS precision-mode embargo")
    test_precision_mode_allows_elite_breakout_long()
    print("PASS precision-mode elite breakout")
    test_precision_lab_collapses_repeated_rows_and_flags_toxic_families()
    print("PASS precision lab replay")
    test_precision_entry_cadence_blocks_repeat_activity()
    print("PASS precision cadence throttling")
    test_tradexyz_volume_summary_rolls_up_wallet_activity()
    print("PASS Trade.xyz volume summary")
    test_tradexyz_identity_inspection_rejects_agent_wallets()
    print("PASS Trade.xyz agent wallet rejection")
    test_tradexyz_identity_inspection_keeps_strict_subaccount_scope()
    print("PASS Trade.xyz strict sub-account identity")
    test_dashboard_tradexyz_volume_endpoint_returns_checker_payload()
    print("PASS Trade.xyz volume endpoint")


if __name__ == "__main__":
    run_all()
