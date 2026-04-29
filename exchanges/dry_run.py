"""
exchanges/dry_run.py
Paper-trading exchange — simulates all orders without sending anything real.
Useful for safely testing the strategy before going live.

Now supports simulated limit orders:
  • limit_buy / limit_sell register a pending order (not filled immediately)
  • get_order_status checks if current price has crossed the limit price
  • cancel_order removes a pending limit order
"""

import time
from typing import Optional, Dict, List
from logger import get_logger
from exchanges.base import BaseExchange, AccountState, OrderResult, LimitOrderStatus
from data.market_data import get_current_price

log = get_logger("dry_run")


class DryRunExchange(BaseExchange):
    name = "DryRun (Paper Trading)"

    def __init__(self, starting_balance_usd: float = 10_000.0,
                 supported_symbols: Optional[List[str]] = None,
                 shortable_map: Optional[Dict[str, bool]] = None):
        self.balance     = starting_balance_usd
        self.positions: Dict[str, dict] = {}    # coin → {size, direction, entry_price…}
        self.order_count = 0
        self._supported_symbols = list(supported_symbols or ["BTC", "ETH", "SOL"])
        self._shortable_map = {
            str(symbol).upper(): bool(value)
            for symbol, value in (shortable_map or {}).items()
        }
        # Pending limit orders: order_id → {coin, direction, limit_price, size_coin, size_usd}
        self._pending_limits: Dict[str, dict] = {}

    def connect(self) -> bool:
        log.info(f"[DRY RUN] Paper trading active. Balance: ${self.balance:,.2f}")
        return True

    def is_dry_run(self) -> bool:
        return True

    def supports_limit_orders(self) -> bool:
        return True

    def supported_coins(self) -> List[str]:
        return list(self._supported_symbols)

    # ── Account state ─────────────────────────────────────

    def get_account_state(self) -> Optional[AccountState]:
        positions = []
        total_equity = self.balance
        for coin, p in self.positions.items():
            price = get_current_price(coin) or p["entry_price"]
            if p["direction"] == "LONG":
                pnl = (price - p["entry_price"]) / p["entry_price"] * p["size_usd"]
            else:
                pnl = (p["entry_price"] - price) / p["entry_price"] * p["size_usd"]
            total_equity += pnl
            positions.append({
                "coin":           coin,
                "size":           p["size_coin"],
                "direction":      p["direction"],
                "entry_price":    p["entry_price"],
                "unrealised_pnl": pnl,
            })
        return AccountState(
            total_equity_usd = total_equity,
            available_usd    = self.balance,
            positions        = positions,
        )

    def set_leverage(self, coin: str, leverage: int) -> bool:
        log.info(f"[DRY RUN] [{coin}] Leverage set to {leverage}× (simulated)")
        return True

    # ── Market orders ─────────────────────────────────────

    def market_buy(self, coin: str, size_coin: float,
                   slippage: float = 0.01) -> OrderResult:
        return self._simulate_open(coin, "LONG", size_coin)

    def market_sell(self, coin: str, size_coin: float,
                    slippage: float = 0.01) -> OrderResult:
        return self._simulate_open(coin, "SHORT", size_coin)

    # ── Limit orders ──────────────────────────────────────

    def limit_buy(self, coin: str, size_coin: float,
                  limit_price: float, maker_only: bool = False) -> OrderResult:
        """Register a pending limit BUY; simulates fill when price ≤ limit_price."""
        return self._register_limit(coin, "LONG", size_coin, limit_price)

    def limit_sell(self, coin: str, size_coin: float,
                   limit_price: float, maker_only: bool = False) -> OrderResult:
        """Register a pending limit SELL; simulates fill when price ≥ limit_price."""
        return self._register_limit(coin, "SHORT", size_coin, limit_price)

    def _register_limit(self, coin: str, direction: str, size_coin: float,
                        limit_price: float) -> OrderResult:
        if coin not in self._supported_symbols:
            return OrderResult(success=False, error=f"{coin} is not supported in this dry-run venue")
        if direction == "SHORT" and not self._shortable_map.get(coin.upper(), True):
            return OrderResult(success=False, error=f"{coin} is long-only in this dry-run venue")
        self.order_count += 1
        oid = f"DRY-LMT-{self.order_count:04d}"
        size_usd = size_coin * limit_price
        self._pending_limits[oid] = {
            "coin":        coin,
            "direction":   direction,
            "limit_price": limit_price,
            "size_coin":   size_coin,
            "size_usd":    size_usd,
            "placed_at":   time.time(),
        }
        log.info(
            f"[DRY RUN] [{coin}] Limit {direction} registered: "
            f"{size_coin:.6f} @ ${limit_price:.2f}  (id={oid})"
        )
        return OrderResult(
            success      = True,
            order_id     = oid,
            filled_price = 0.0,   # not filled yet
            filled_size  = 0.0,
        )

    def get_order_status(self, coin: str, order_id: str) -> LimitOrderStatus:
        """Check if a limit order should be considered filled by comparing to live price."""
        pending = self._pending_limits.get(order_id)
        if not pending:
            # A paper exchange restart can lose in-memory limit state while the
            # agent checkpoint still has the pending order. Do not fake a fill
            # unless this exchange can actually simulate it.
            return LimitOrderStatus(order_id=order_id, coin=coin, filled=False)

        if pending["coin"] != coin:
            return LimitOrderStatus(order_id=order_id, coin=coin)

        price = get_current_price(coin)
        if price is None:
            return LimitOrderStatus(order_id=order_id, coin=coin)

        direction   = pending["direction"]
        limit_price = pending["limit_price"]

        # LONG fill: price dropped to or below the limit (good buy)
        # SHORT fill: price rose to or above the limit (good sell)
        should_fill = (
            (direction == "LONG"  and price <= limit_price) or
            (direction == "SHORT" and price >= limit_price)
        )

        if should_fill:
            # Simulate the fill
            size_coin = pending["size_coin"]
            size_usd  = pending["size_usd"]
            self.positions[coin] = {
                "direction":   direction,
                "entry_price": price,
                "size_coin":   size_coin,
                "size_usd":    size_usd,
                "opened_at":   time.time(),
            }
            self.balance -= size_usd
            del self._pending_limits[order_id]
            log.info(
                f"[DRY RUN] [{coin}] Limit {direction} FILLED @ ${price:.2f} "
                f"(limit was ${limit_price:.2f})"
            )
            return LimitOrderStatus(
                order_id     = order_id,
                coin         = coin,
                filled       = True,
                filled_price = price,
                filled_size  = size_coin,
            )

        return LimitOrderStatus(order_id=order_id, coin=coin, filled=False)

    def restore_limit_order(
        self,
        *,
        order_id: str,
        coin: str,
        direction: str,
        size_coin: float,
        limit_price: float,
        size_usd: float = 0.0,
    ) -> bool:
        """Rehydrate a pending paper limit order from the agent checkpoint."""
        coin = str(coin or "").upper()
        direction = str(direction or "").upper()
        if not order_id or coin not in self._supported_symbols or direction not in {"LONG", "SHORT"}:
            return False
        self._pending_limits[order_id] = {
            "coin": coin,
            "direction": direction,
            "limit_price": float(limit_price or 0.0),
            "size_coin": float(size_coin or 0.0),
            "size_usd": float(size_usd or 0.0) or float(size_coin or 0.0) * float(limit_price or 0.0),
            "placed_at": time.time(),
        }
        try:
            suffix = int(str(order_id).rsplit("-", 1)[-1])
            self.order_count = max(self.order_count, suffix)
        except Exception:
            pass
        log.info(
            f"[DRY RUN] [{coin}] Restored pending limit {direction}: "
            f"{float(size_coin or 0.0):.6f} @ ${float(limit_price or 0.0):.2f} (id={order_id})"
        )
        return True

    def cancel_order(self, coin: str, order_id: str) -> bool:
        """Cancel a pending limit order."""
        if order_id in self._pending_limits:
            log.info(f"[DRY RUN] [{coin}] Limit order {order_id} cancelled")
            del self._pending_limits[order_id]
            return True
        return True   # already gone

    # ── Internal helpers ──────────────────────────────────

    def _simulate_open(self, coin: str, direction: str, size_coin: float) -> OrderResult:
        if coin not in self._supported_symbols:
            return OrderResult(success=False, error=f"{coin} is not supported in this dry-run venue")
        if direction == "SHORT" and not self._shortable_map.get(coin.upper(), True):
            return OrderResult(success=False, error=f"{coin} is long-only in this dry-run venue")
        price = get_current_price(coin)
        if not price:
            return OrderResult(success=False, error=f"Could not get price for {coin}")

        size_usd = size_coin * price
        self.order_count += 1
        oid = f"DRY-MKT-{self.order_count:04d}"

        self.positions[coin] = {
            "direction":   direction,
            "entry_price": price,
            "size_coin":   size_coin,
            "size_usd":    size_usd,
            "opened_at":   time.time(),
        }
        self.balance -= size_usd    # margin reservation

        log.info(
            f"[DRY RUN] [{coin}] Simulated {direction}: "
            f"{size_coin:.6f} coins @ ${price:.2f}  (${size_usd:.2f})"
        )
        return OrderResult(
            success      = True,
            order_id     = oid,
            filled_price = price,
            filled_size  = size_coin,
        )

    def close_position(self, coin: str) -> OrderResult:
        pos = self.positions.pop(coin, None)
        if not pos:
            return OrderResult(success=True)
        price = get_current_price(coin) or pos["entry_price"]
        if pos["direction"] == "LONG":
            pnl = (price - pos["entry_price"]) / pos["entry_price"] * pos["size_usd"]
        else:
            pnl = (pos["entry_price"] - price) / pos["entry_price"] * pos["size_usd"]
        self.balance += pos["size_usd"] + pnl
        pnl_pct = pnl / pos["size_usd"] * 100
        log.info(
            f"[DRY RUN] [{coin}] Position closed @ ${price:.2f} "
            f"PnL: {pnl_pct:+.2f}% (${pnl:+.2f}) | New balance: ${self.balance:,.2f}"
        )
        return OrderResult(success=True, filled_price=price)
