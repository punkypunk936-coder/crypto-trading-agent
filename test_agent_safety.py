"""
test_agent_safety.py — basic safety regressions for the live agent.

Run:
    python3 test_agent_safety.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import checkpoint as checkpoint_module
import agent as agent_module
from agent import TradingAgent
from config import Config
from exchanges.base import AccountState, BaseExchange, OrderResult, LimitOrderStatus
from exchanges.dry_run import DryRunExchange
from risk.risk_manager import OrderRequest


class StubExchange(BaseExchange):
    def __init__(self, name: str, should_fill: bool):
        self.name = name
        self.should_fill = should_fill
        self.market_buy_calls = 0

    def connect(self) -> bool:
        return True

    def get_account_state(self):
        return AccountState(total_equity_usd=1000.0, available_usd=1000.0, positions=[])

    def set_leverage(self, coin: str, leverage: int) -> bool:
        return True

    def market_buy(self, coin: str, size_coin: float, slippage: float = 0.01) -> OrderResult:
        self.market_buy_calls += 1
        if self.should_fill:
            return OrderResult(success=True, filled_price=100.0, filled_size=size_coin)
        return OrderResult(success=False, error="intentional fail")

    def market_sell(self, coin: str, size_coin: float, slippage: float = 0.01) -> OrderResult:
        return self.market_buy(coin, size_coin, slippage)

    def close_position(self, coin: str) -> OrderResult:
        return OrderResult(success=True, filled_price=100.0)

    def get_order_status(self, coin: str, order_id: str) -> LimitOrderStatus:
        return LimitOrderStatus(order_id=order_id, coin=coin, filled=False)


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


def run_all() -> None:
    test_checkpoint_recovery()
    print("PASS checkpoint recovery")
    test_execute_order_stops_after_first_success()
    print("PASS single-exchange execution")


if __name__ == "__main__":
    run_all()
