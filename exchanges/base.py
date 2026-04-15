"""
exchanges/base.py
Abstract base class that every exchange connector must implement.
This lets the agent swap exchanges without changing any other code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class AccountState:
    total_equity_usd: float        # Total account value in USD
    available_usd: float           # Free margin / available balance
    positions: List[dict]          # Raw position list from the exchange


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    filled_price: float = 0.0
    filled_size: float = 0.0
    error: str = ""


@dataclass
class LimitOrderStatus:
    """Status of a previously placed limit order."""
    order_id: str
    coin: str
    filled: bool = False
    cancelled: bool = False
    filled_price: float = 0.0
    filled_size: float = 0.0


class BaseExchange(ABC):
    name: str = "base"

    @abstractmethod
    def connect(self) -> bool:
        """Initialise the connection. Return True if successful."""

    @abstractmethod
    def get_account_state(self) -> Optional[AccountState]:
        """Return current equity and open positions."""

    @abstractmethod
    def set_leverage(self, coin: str, leverage: int) -> bool:
        """Set leverage for a coin. Return True if successful."""

    @abstractmethod
    def market_buy(self, coin: str, size_coin: float,
                   slippage: float = 0.01) -> OrderResult:
        """Place a market BUY order."""

    @abstractmethod
    def market_sell(self, coin: str, size_coin: float,
                    slippage: float = 0.01) -> OrderResult:
        """Place a market SELL (short) order."""

    @abstractmethod
    def close_position(self, coin: str) -> OrderResult:
        """Close the full position for a coin at market price."""

    # ── Limit order support (optional — subclasses that support it override) ──

    def limit_buy(self, coin: str, size_coin: float,
                  limit_price: float, maker_only: bool = False) -> OrderResult:
        """
        Place a limit BUY order.
        Default falls back to market_buy (safe for exchanges without limit support).
        Override in subclasses that support limit orders natively.
        """
        return self.market_buy(coin, size_coin)

    def limit_sell(self, coin: str, size_coin: float,
                   limit_price: float, maker_only: bool = False) -> OrderResult:
        """
        Place a limit SELL order.
        Default falls back to market_sell.
        """
        return self.market_sell(coin, size_coin)

    def cancel_order(self, coin: str, order_id: str) -> bool:
        """
        Cancel a pending limit order.
        Returns True if cancel was successful (or order already gone).
        Default is a no-op (market orders can't be cancelled).
        """
        return True

    def get_order_status(self, coin: str, order_id: str) -> LimitOrderStatus:
        """
        Poll the status of a pending limit order.
        Default: assume filled (safe fallback for market-only exchanges).
        """
        return LimitOrderStatus(order_id=order_id, coin=coin, filled=True)

    def supports_limit_orders(self) -> bool:
        """Advertise whether this exchange has real limit order support."""
        return False

    def is_dry_run(self) -> bool:
        return False

    def supported_coins(self) -> List[str]:
        """Return the symbols this exchange can safely trade."""
        return []
