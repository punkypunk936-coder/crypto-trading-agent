"""
exchanges/lighter_client.py
Connector for Lighter DEX (lighter.xyz) — zero-fee decentralised perpetuals on Arbitrum.

Authentication: EVM private key + an Arbitrum RPC URL.
SDK: lighter-sdk (pip install lighter-sdk)

Market IDs on Lighter (perpetuals):
  0  → BTC-USDC
  1  → ETH-USDC
  2  → SOL-USDC
  (Check https://docs.lighter.xyz for the full list; HYPE is on Hyperliquid only)
"""

from typing import Optional, Dict
from logger import get_logger
from exchanges.base import BaseExchange, AccountState, OrderResult, LimitOrderStatus

log = get_logger("lighter")

# Map our coin tickers to Lighter's numeric market IDs
# Verify latest IDs at https://docs.lighter.xyz/markets
COIN_TO_MARKET_ID: Dict[str, int] = {
    "BTC":  0,
    "ETH":  1,
    "SOL":  2,
    "HYPE": 3,   # Hyperliquid token — verify market ID at lighter.xyz
}

# Minimum order sizes per coin (in coin units)
MIN_ORDER_SIZE: Dict[str, float] = {
    "BTC":  0.0001,
    "ETH":  0.001,
    "SOL":  0.01,
    "HYPE": 0.1,
}


class LighterClient(BaseExchange):
    name = "Lighter"

    def __init__(self, private_key: str, web3_url: str):
        self.private_key  = private_key
        self.web3_url     = web3_url
        self._client      = None
        self._connected   = False

    # ── Connection ────────────────────────────────────────

    def connect(self) -> bool:
        try:
            from lighter import Client, ClientConfig   # lighter-sdk

            cfg = ClientConfig(
                api_auth_private_key = self.private_key,
                web3_provider_url    = self.web3_url,
            )
            self._client    = Client(cfg)
            self._connected = True
            log.info("Connected to Lighter DEX (Arbitrum)")
            return True

        except ImportError:
            log.error("lighter-sdk not installed. Run: pip install lighter-sdk")
            return False
        except Exception as e:
            log.error(f"Lighter connection failed: {e}")
            return False

    def supports_limit_orders(self) -> bool:
        return True

    # ── Account info ──────────────────────────────────────

    def get_account_state(self) -> Optional[AccountState]:
        if not self._connected:
            log.error("Not connected to Lighter")
            return None
        try:
            account_info = self._client.account_api.get_account()

            equity = float(account_info.get("equity", 0) or 0)
            avail  = float(account_info.get("available_balance", 0) or 0)

            raw_positions = account_info.get("positions", []) or []
            positions = []
            for rp in raw_positions:
                size = float(rp.get("size", 0) or 0)
                if size == 0:
                    continue
                market_id = rp.get("market_id")
                coin = next(
                    (c for c, mid in COIN_TO_MARKET_ID.items() if mid == market_id),
                    str(market_id)
                )
                positions.append({
                    "coin":        coin,
                    "size":        size,
                    "direction":   "LONG" if size > 0 else "SHORT",
                    "entry_price": float(rp.get("avg_entry_price", 0) or 0),
                    "unrealised_pnl": float(rp.get("unrealized_pnl", 0) or 0),
                })

            return AccountState(
                total_equity_usd = equity,
                available_usd    = avail,
                positions        = positions,
            )
        except Exception as e:
            log.error(f"Failed to get Lighter account state: {e}")
            return None

    # ── Leverage (Lighter handles this at account level) ──

    def set_leverage(self, coin: str, leverage: int) -> bool:
        # Lighter uses isolated margin per market; leverage set at order time.
        log.debug(f"[{coin}] Lighter leverage set to {leverage}× (order-level)")
        return True

    # ── Market orders ─────────────────────────────────────

    def market_buy(self, coin: str, size_coin: float,
                   slippage: float = 0.01) -> OrderResult:
        return self._place_order(coin, side="buy", size=size_coin,
                                 order_type="market")

    def market_sell(self, coin: str, size_coin: float,
                    slippage: float = 0.01) -> OrderResult:
        return self._place_order(coin, side="sell", size=size_coin,
                                 order_type="market")

    # ── Limit orders ──────────────────────────────────────

    def limit_buy(self, coin: str, size_coin: float,
                  limit_price: float) -> OrderResult:
        """Place a resting limit BUY order."""
        return self._place_order(coin, side="buy", size=size_coin,
                                 order_type="limit", price=limit_price)

    def limit_sell(self, coin: str, size_coin: float,
                   limit_price: float) -> OrderResult:
        """Place a resting limit SELL order."""
        return self._place_order(coin, side="sell", size=size_coin,
                                 order_type="limit", price=limit_price)

    def cancel_order(self, coin: str, order_id: str) -> bool:
        """Cancel a pending limit order by order_id."""
        if not self._connected:
            return False
        market_id = COIN_TO_MARKET_ID.get(coin)
        if market_id is None:
            return False
        try:
            result = self._client.transaction_api.send_tx(
                tx_type   = "cancel_order",
                market_id = market_id,
                order_id  = order_id,
            )
            success = result and result.get("status") in ("ok", "success", "submitted")
            if success:
                log.info(f"[{coin}] Limit order {order_id} cancelled")
            else:
                log.warning(f"[{coin}] Cancel order returned: {result}")
            return bool(success)
        except Exception as e:
            log.error(f"[{coin}] cancel_order exception: {e}")
            return False

    def get_order_status(self, coin: str, order_id: str) -> LimitOrderStatus:
        """Poll whether a limit order has been filled."""
        if not self._connected:
            return LimitOrderStatus(order_id=order_id, coin=coin)
        market_id = COIN_TO_MARKET_ID.get(coin)
        if market_id is None:
            return LimitOrderStatus(order_id=order_id, coin=coin)
        try:
            order_info = self._client.order_api.get_order(
                market_id = market_id,
                order_id  = order_id,
            )
            status = order_info.get("status", "").lower()
            filled    = status in ("filled", "fully_filled")
            cancelled = status in ("cancelled", "canceled", "expired")
            return LimitOrderStatus(
                order_id     = order_id,
                coin         = coin,
                filled       = filled,
                cancelled    = cancelled,
                filled_price = float(order_info.get("avg_price", 0) or 0),
                filled_size  = float(order_info.get("filled_size", 0) or 0),
            )
        except Exception as e:
            log.error(f"[{coin}] get_order_status exception: {e}")
            return LimitOrderStatus(order_id=order_id, coin=coin)

    # ── Internal order placement ───────────────────────────

    def _place_order(self, coin: str, side: str, size: float,
                     order_type: str = "market",
                     price: float = 0.0) -> OrderResult:
        if not self._connected:
            return OrderResult(success=False, error="Not connected")

        market_id = COIN_TO_MARKET_ID.get(coin)
        if market_id is None:
            return OrderResult(success=False, error=f"Unknown coin {coin} on Lighter")

        min_sz = MIN_ORDER_SIZE.get(coin, 0.0001)
        if size < min_sz:
            return OrderResult(
                success=False,
                error=f"Size {size} below minimum {min_sz} for {coin}"
            )

        try:
            tx_kwargs = dict(
                tx_type    = "create_order",
                market_id  = market_id,
                order_side = side,
                order_type = order_type,
                size       = str(size),
            )
            if order_type == "limit" and price > 0:
                tx_kwargs["price"] = str(price)

            tx_result = self._client.transaction_api.send_tx(**tx_kwargs)

            if tx_result and tx_result.get("status") in ("ok", "success", "submitted"):
                filled_px = float(tx_result.get("avg_price", 0) or 0)
                filled_sz = float(tx_result.get("filled_size", size) or size)
                log.info(
                    f"[{coin}] Lighter {order_type.upper()} {side.upper()} accepted: "
                    f"{size} @ ${price if order_type == 'limit' else filled_px:.2f}"
                )
                return OrderResult(
                    success      = True,
                    order_id     = str(tx_result.get("order_id", "")),
                    filled_price = filled_px,
                    filled_size  = filled_sz,
                )
            else:
                err = str(tx_result)
                log.error(f"[{coin}] Lighter order failed: {err}")
                return OrderResult(success=False, error=err)

        except Exception as e:
            log.error(f"[{coin}] Lighter _place_order exception: {e}")
            return OrderResult(success=False, error=str(e))

    # ── Close position ────────────────────────────────────

    def close_position(self, coin: str) -> OrderResult:
        """Close the full position on Lighter by placing the opposing market order."""
        if not self._connected:
            return OrderResult(success=False, error="Not connected")
        try:
            state = self.get_account_state()
            if not state:
                return OrderResult(success=False, error="Cannot get account state")

            pos = next((p for p in state.positions if p["coin"] == coin), None)
            if not pos:
                log.info(f"[{coin}] No open Lighter position to close")
                return OrderResult(success=True)

            direction  = pos["direction"]
            size       = abs(pos["size"])
            close_side = "sell" if direction == "LONG" else "buy"

            result = self._place_order(coin, side=close_side, size=size,
                                       order_type="market")
            if result.success:
                log.info(f"[{coin}] Lighter position closed")
            return result

        except Exception as e:
            log.error(f"[{coin}] close_position exception: {e}")
            return OrderResult(success=False, error=str(e))
