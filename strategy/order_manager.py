"""
strategy/order_manager.py — Limit order placement + re-entry after TP.

Philosophy
──────────
• Market orders for initial entries (fast fill, avoids slippage drift).
• Limit orders for re-entry after a Take-Profit close — we wait for the price
  to pull back to a fib retracement level before buying back in.
• Cancel stale pending orders if price moves decisively away.
• All state is held in PendingOrder objects; RiskManager stores them.

Re-entry logic
──────────────
  After a TP close (LONG example):
    1. Mark the coin as "watching for re-entry"
    2. Compute fib retracement from entry → TP:
         0.382 → shallow dip  (aggressive re-entry)
         0.500 → mid dip      (default)
         0.618 → deep dip     (conservative, better R:R)
    3. Post a limit buy at the chosen fib level
    4. If price never reaches it within MAX_REENTRY_WAIT_CYCLES, cancel + forget
    5. If price drops through stop-loss of TP level, cancel (trend reversed)

Order states
────────────
  PENDING   → order placed on exchange, not filled
  FILLED    → exchange confirmed fill
  CANCELLED → manually cancelled (stale / reversal)
  EXPIRED   → timed out

"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

from logger import get_logger

log = get_logger("order_manager")

# How many agent cycles to wait before cancelling a stale limit order
MAX_REENTRY_WAIT_CYCLES = 15   # ~30 min at 120s/cycle

# Which fib level to re-enter at (0.382 / 0.500 / 0.618)
REENTRY_FIB_LEVEL = 0.500


class OrderState(str, Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED   = "EXPIRED"


@dataclass
class PendingOrder:
    """Represents a limit order waiting to be filled."""
    coin: str
    direction: str           # "LONG" or "SHORT"
    limit_price: float
    size_coin: float
    size_usd: float
    stop_loss: float
    take_profit: float
    signal_score: float
    exchange: str = ""
    leverage: int = 1
    margin_usd: float = 0.0
    exchange_order_id: Optional[str] = None   # filled by exchange after placement
    state: OrderState = OrderState.PENDING
    cycles_waiting: int = 0
    reprice_count: int = 0
    placed_at: float = field(default_factory=time.time)
    reason: str = "re_entry"   # "re_entry" | "initial_limit" | etc.
    metadata: dict = field(default_factory=dict)

    def age_cycles(self) -> int:
        return self.cycles_waiting

    def is_expired(self) -> bool:
        return self.cycles_waiting >= MAX_REENTRY_WAIT_CYCLES


@dataclass
class ReEntryWatch:
    """Tracks a coin awaiting a re-entry limit order after TP."""
    coin: str
    direction: str          # direction of the closed TP trade
    entry_price: float      # entry of the closed trade
    tp_price: float         # the TP level that was hit
    reentry_price: float    # limit price to re-enter at (fib level)
    stop_price: float       # abort watch if price crosses this (trend reversal)
    size_usd: float
    signal_score: float
    trade_plan: dict = field(default_factory=dict)
    entry_context: dict = field(default_factory=dict)
    cycles: int = 0
    max_cycles: int = MAX_REENTRY_WAIT_CYCLES


class OrderManager:
    """
    Manages limit orders and re-entry watches across all coins.
    The agent calls tick() each cycle to advance state machines.
    """

    def __init__(self):
        # coin → PendingOrder (one active limit order per coin max)
        self.pending_orders: Dict[str, PendingOrder] = {}
        # coin → ReEntryWatch (one active re-entry watch per coin max)
        self.reentry_watches: Dict[str, ReEntryWatch] = {}
        # History of filled/cancelled orders this session
        self.order_history: List[PendingOrder] = []

    # ── Re-entry scheduling ───────────────────────────────────────────────

    def schedule_reentry(
        self,
        coin: str,
        direction: str,
        entry_price: float,
        tp_price: float,
        size_usd: float,
        signal_score: float,
        trade_plan: Optional[dict] = None,
        entry_context: Optional[dict] = None,
    ) -> ReEntryWatch:
        """
        Called immediately after a TP close.
        Calculates the fib retracement level and starts watching.
        """
        # Fib retracement: price pulls back from TP toward entry
        move = abs(tp_price - entry_price)
        if direction == "LONG":
            # TP was above entry; re-enter on a dip back down
            reentry_price = tp_price - move * REENTRY_FIB_LEVEL
            # Abort watch if price drops below entry (original stop area)
            stop_price    = entry_price * 0.97
        else:  # SHORT
            # TP was below entry; re-enter on a bounce back up
            reentry_price = tp_price + move * REENTRY_FIB_LEVEL
            stop_price    = entry_price * 1.03

        watch = ReEntryWatch(
            coin          = coin,
            direction     = direction,
            entry_price   = entry_price,
            tp_price      = tp_price,
            reentry_price = reentry_price,
            stop_price    = stop_price,
            size_usd      = size_usd,
            signal_score  = signal_score,
            trade_plan    = dict(trade_plan or {}),
            entry_context = dict(entry_context or {}),
        )
        self.reentry_watches[coin] = watch
        log.info(
            f"[{coin}] Re-entry watch: {direction} @ ${reentry_price:.2f} "
            f"(TP was ${tp_price:.2f}, fib {REENTRY_FIB_LEVEL:.3f}). "
            f"Abort if price crosses ${stop_price:.2f}. "
            f"Max {MAX_REENTRY_WAIT_CYCLES} cycles."
        )
        return watch

    def cancel_reentry_watch(self, coin: str, reason: str = "manual"):
        """Remove a re-entry watch (e.g. signal reversed, new trade opened)."""
        if coin in self.reentry_watches:
            log.info(f"[{coin}] Re-entry watch cancelled: {reason}")
            del self.reentry_watches[coin]

    # ── Limit order lifecycle ─────────────────────────────────────────────

    def register_limit_order(self, order: PendingOrder):
        """Called after the exchange accepts a limit order."""
        self.pending_orders[order.coin] = order
        log.info(
            f"[{order.coin}] Limit {order.direction} registered: "
            f"{order.size_coin:.6f} @ ${order.limit_price:.2f} "
            f"(exchange={order.exchange or 'unknown'} id={order.exchange_order_id})"
        )

    def restore_pending_order(self, order: PendingOrder):
        """Restore a pending order after restart."""
        self.pending_orders[order.coin] = order
        log.info(
            f"[{order.coin}] Restored pending {order.direction} @ ${order.limit_price:.2f} "
            f"(exchange={order.exchange or 'unknown'} id={order.exchange_order_id})"
        )

    def restore_watch(self, watch: ReEntryWatch):
        """Restore a re-entry watch after restart."""
        self.reentry_watches[watch.coin] = watch
        log.info(
            f"[{watch.coin}] Restored re-entry watch {watch.direction} "
            f"@ ${watch.reentry_price:.2f}"
        )

    def mark_filled(self, coin: str, filled_price: float):
        """Called when exchange reports fill."""
        order = self.pending_orders.get(coin)
        if order:
            order.state = OrderState.FILLED
            self.order_history.append(order)
            del self.pending_orders[coin]
            log.info(f"[{coin}] Limit order FILLED @ ${filled_price:.2f}")

    def mark_cancelled(self, coin: str):
        """Called when order is cancelled."""
        order = self.pending_orders.get(coin)
        if order:
            order.state = OrderState.CANCELLED
            self.order_history.append(order)
            del self.pending_orders[coin]
            log.info(f"[{coin}] Limit order CANCELLED")

    # ── Per-cycle tick ────────────────────────────────────────────────────

    def tick(self, current_prices: Dict[str, float]) -> List[dict]:
        """
        Called once per agent cycle.
        Returns a list of action dicts for the agent to act on:
          {"type": "place_limit", "coin": ..., "direction": ...,
           "price": ..., "size_usd": ..., "sl": ..., "tp": ..., "score": ...}
          {"type": "cancel_limit", "coin": ..., "order_id": ...}

        Also advances re-entry watches and expires stale pending orders.
        """
        actions = []

        # 1. Advance re-entry watches
        for coin, watch in list(self.reentry_watches.items()):
            price = current_prices.get(coin)
            if price is None:
                continue

            watch.cycles += 1

            # Abort: price moved through the abort level (trend reversed)
            abort_triggered = (
                (watch.direction == "LONG"  and price < watch.stop_price) or
                (watch.direction == "SHORT" and price > watch.stop_price)
            )
            if abort_triggered:
                log.info(
                    f"[{coin}] Re-entry watch aborted: price ${price:.2f} "
                    f"crossed abort level ${watch.stop_price:.2f}"
                )
                del self.reentry_watches[coin]
                continue

            # Expired
            if watch.cycles >= watch.max_cycles:
                log.info(f"[{coin}] Re-entry watch expired after {watch.cycles} cycles")
                del self.reentry_watches[coin]
                continue

            # Trigger: price reached reentry level → request a limit order
            triggered = (
                (watch.direction == "LONG"  and price <= watch.reentry_price) or
                (watch.direction == "SHORT" and price >= watch.reentry_price)
            )
            if triggered and coin not in self.pending_orders:
                log.info(
                    f"[{coin}] Re-entry triggered: price ${price:.2f} "
                    f"reached limit level ${watch.reentry_price:.2f}"
                )
                # Calculate SL/TP relative to the limit price
                plan = dict(watch.trade_plan or {})
                risk_pct = float(plan.get("risk_pct", 0.0) or 0.0) / 100.0
                rr_ratio = float(plan.get("risk_reward_ratio", 0.0) or 0.0)
                if watch.direction == "LONG":
                    sl = watch.reentry_price * (1 - risk_pct) if risk_pct > 0 else watch.reentry_price * 0.90
                    tp = watch.reentry_price + (watch.reentry_price - sl) * rr_ratio if rr_ratio > 0 else watch.reentry_price * 1.50
                else:
                    sl = watch.reentry_price * (1 + risk_pct) if risk_pct > 0 else watch.reentry_price * 1.10
                    tp = watch.reentry_price - (sl - watch.reentry_price) * rr_ratio if rr_ratio > 0 else watch.reentry_price * 0.50

                actions.append({
                    "type":      "place_limit",
                    "coin":      coin,
                    "direction": watch.direction,
                    "price":     watch.reentry_price,
                    "size_usd":  watch.size_usd,
                    "sl":        sl,
                    "tp":        tp,
                    "score":     watch.signal_score,
                    "reason":    "re_entry",
                    "leverage":  int((watch.entry_context or {}).get("leverage", 1) or 1),
                    "margin_usd": float((watch.entry_context or {}).get("margin_usd", 0.0) or 0.0),
                    "entry_context": dict(watch.entry_context or {}),
                    "trade_plan": dict(watch.trade_plan or {}),
                })
                # Remove watch after triggering (one re-entry per TP)
                del self.reentry_watches[coin]

        # 2. Expire stale pending limit orders
        for coin, order in list(self.pending_orders.items()):
            order.cycles_waiting += 1
            if order.is_expired():
                log.info(
                    f"[{coin}] Limit order expired after "
                    f"{order.cycles_waiting} cycles (never filled)"
                )
                actions.append({
                    "type":     "cancel_limit",
                    "coin":     coin,
                    "order_id": order.exchange_order_id,
                    "exchange": order.exchange,
                })
                order.state = OrderState.EXPIRED
                self.order_history.append(order)
                del self.pending_orders[coin]

        return actions

    # ── Query helpers ─────────────────────────────────────────────────────

    def has_pending(self, coin: str) -> bool:
        return coin in self.pending_orders

    def has_watch(self, coin: str) -> bool:
        return coin in self.reentry_watches

    def summary(self) -> str:
        lines = ["  [OrderManager]"]
        if self.pending_orders:
            for coin, o in self.pending_orders.items():
                lines.append(
                    f"    Pending {o.direction} {coin} @ ${o.limit_price:.2f} "
                    f"(cycle {o.cycles_waiting}/{MAX_REENTRY_WAIT_CYCLES})"
                )
        if self.reentry_watches:
            for coin, w in self.reentry_watches.items():
                lines.append(
                    f"    Watching {w.direction} {coin} for re-entry @ ${w.reentry_price:.2f} "
                    f"(cycle {w.cycles}/{w.max_cycles})"
                )
        if not self.pending_orders and not self.reentry_watches:
            lines.append("    No pending orders or re-entry watches.")
        return "\n".join(lines)
