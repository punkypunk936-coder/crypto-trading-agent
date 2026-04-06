"""
test_agent_safety.py — basic safety regressions for the live agent.

Run:
    python3 test_agent_safety.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import checkpoint as checkpoint_module
import agent as agent_module
import main as main_module
import trade_logger as trade_logger_module
from indicators import trade_memory as trade_memory_module
from agent import TradingAgent
from config import Config
from dashboard import app as dashboard_module
from exchanges.base import AccountState, BaseExchange, OrderResult, LimitOrderStatus
from exchanges.dry_run import DryRunExchange
from risk.risk_manager import OrderRequest, OpenPosition
from strategy.aggressive_strategy import AggressiveStrategy


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
    cfg.trading.coins = ["BTC", "HYPE"]
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
        try:
            dashboard_module.CONTROL = temp / "control.json"
            dashboard_module.KILL = temp / "KILL"
            dashboard_module.STATE = temp / "state.json"
            dashboard_module.LOG = temp / "trades_log.csv"
            dashboard_module.SNAPSHOT = temp / "dashboard_snapshot.json"
            dashboard_module._remote_state = {"snapshot": None}

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

            client = dashboard_module.app.test_client()
            payload = client.get("/api/state").get_json()
            assert payload["state"]["cycle_number"] == 11
        finally:
            dashboard_module.CONTROL = original_control
            dashboard_module.KILL = original_kill
            dashboard_module.STATE = original_state
            dashboard_module.LOG = original_log
            dashboard_module.SNAPSHOT = original_snapshot
            dashboard_module._remote_state = original_remote


def test_hosted_dashboard_bundle_matches_local_template() -> None:
    local_template = Path("dashboard/templates/dashboard.html").read_text()
    hosted_bundle = Path("netlify-dashboard/public/index.html").read_text()
    assert hosted_bundle == local_template, "hosted dashboard should mirror the local dashboard UI exactly"


def test_local_dashboard_serves_hosted_bundle() -> None:
    client = dashboard_module.app.test_client()
    served = client.get("/").data
    hosted_bundle = Path("netlify-dashboard/public/index.html").read_bytes()
    assert served == hosted_bundle, "local dashboard root should serve the exact hosted UI bundle"


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


def run_all() -> None:
    test_checkpoint_recovery()
    print("PASS checkpoint recovery")
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
    test_dynamic_trade_plan_is_attached_to_signal()
    print("PASS dynamic trade planning")
    test_orderbook_support_blocks_weak_short_into_demand()
    print("PASS orderbook support guard")
    test_confirmed_breakout_can_promote_borderline_long()
    print("PASS confirmed breakout promotion")
    test_probing_breakout_does_not_override_nearby_resistance()
    print("PASS probing breakout guard")
    test_thesis_gate_blocks_high_score_range_compression_setup()
    print("PASS thesis no-trade gate")
    test_agent_uses_completed_candles_for_conviction_but_live_price_for_execution()
    print("PASS completed-candle conviction split")
    test_scale_in_does_not_mutate_before_fill()
    print("PASS scale-in accounting")
    test_order_sizing_scales_with_conviction_and_tempers_euphoria()
    print("PASS conviction sizing")
    test_failed_close_keeps_position_open()
    print("PASS close failure safety")
    test_preflight_reports_missing_live_bootstrap()
    print("PASS preflight live bootstrap check")
    test_dashboard_kill_endpoint_sets_control_state()
    print("PASS dashboard kill endpoint")
    test_dashboard_state_prefers_canonical_snapshot()
    print("PASS dashboard canonical snapshot")
    test_dashboard_refreshes_snapshot_when_state_changes()
    print("PASS dashboard snapshot refresh")
    test_hosted_dashboard_bundle_matches_local_template()
    print("PASS hosted dashboard bundle sync")
    test_local_dashboard_serves_hosted_bundle()
    print("PASS local dashboard hosted bundle")
    test_trade_memory_records_richer_loss_reasoning()
    print("PASS richer RL loss reasoning")
    test_trade_memory_directional_pause_and_guard()
    print("PASS directional RL guardrails")


if __name__ == "__main__":
    run_all()
