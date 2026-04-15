"""
exchanges/hyperliquid_client.py
Connector for Hyperliquid.

Authentication: EVM private key (no centralised API key needed).
SDK: hyperliquid-python-sdk (pip install hyperliquid-python-sdk)

Execution semantics:
  - perps are fully supported live
  - spot equities are wired into paper-trading + analysis now
  - live spot execution remains disabled by default until the agent has a
    dedicated spot inventory/close path
"""

from typing import Optional

from logger import get_logger
from exchanges.base import BaseExchange, AccountState, OrderResult
from exchanges.hyperliquid_markets import (
    get_hyperliquid_market_spec,
    get_hyperliquid_supported_coins,
    hyperliquid_market_type,
    resolve_hyperliquid_symbol,
)

log = get_logger("hyperliquid")


class HyperliquidClient(BaseExchange):
    name = "Hyperliquid"

    def __init__(
        self,
        private_key: str,
        account_address: str,
        mainnet: bool = True,
        allow_spot_execution: bool = False,
    ):
        self.private_key     = private_key
        self.account_address = account_address
        self.mainnet         = mainnet
        self.allow_spot_execution = bool(allow_spot_execution)
        self._info           = None
        self._exchange       = None
        self._connected      = False

    # ── Connection ────────────────────────────────────────

    def connect(self) -> bool:
        try:
            import eth_account
            from hyperliquid.info     import Info
            from hyperliquid.exchange import Exchange
            from hyperliquid.utils    import constants

            account = eth_account.Account.from_key(self.private_key)
            url     = (constants.MAINNET_API_URL if self.mainnet
                       else constants.TESTNET_API_URL)

            self._info     = Info(url, skip_ws=True)
            self._exchange = Exchange(account, url,
                                      account_address=self.account_address)
            self._connected = True
            log.info(f"Connected to Hyperliquid {'mainnet' if self.mainnet else 'testnet'} "
                     f"as {self.account_address[:10]}…")
            return True

        except ImportError:
            log.error("hyperliquid-python-sdk not installed. "
                      "Run: pip install hyperliquid-python-sdk")
            return False
        except Exception as e:
            log.error(f"Hyperliquid connection failed: {e}")
            return False

    # ── Account info ──────────────────────────────────────

    def get_account_state(self) -> Optional[AccountState]:
        if not self._connected:
            log.error("Not connected to Hyperliquid")
            return None
        try:
            state  = self._info.user_state(self.account_address)
            equity = float(state["marginSummary"]["accountValue"])
            avail  = float(state["marginSummary"]["totalRawUsd"])
            positions = []
            for p in state.get("assetPositions", []):
                pos = p.get("position", {})
                szi = float(pos.get("szi", 0))
                if szi == 0:
                    continue
                positions.append({
                    "coin":        pos.get("coin"),
                    "size":        szi,
                    "direction":   "LONG" if szi > 0 else "SHORT",
                    "entry_price": float(pos.get("entryPx", 0)),
                    "unrealised_pnl": float(pos.get("unrealizedPnl", 0)),
                    "leverage":    pos.get("leverage", {}).get("value", 1),
                })
            return AccountState(
                total_equity_usd = equity,
                available_usd    = avail,
                positions        = positions,
            )
        except Exception as e:
            log.error(f"Failed to get Hyperliquid account state: {e}")
            return None

    # ── Leverage ──────────────────────────────────────────

    def set_leverage(self, coin: str, leverage: int) -> bool:
        if not self._connected:
            return False
        if hyperliquid_market_type(coin) != "perp":
            log.debug("[%s] Spot market detected — leverage update skipped", coin)
            return True
        try:
            venue_symbol = resolve_hyperliquid_symbol(coin)
            result = self._exchange.update_leverage(
                leverage=leverage, coin=venue_symbol, is_cross=True
            )
            if result.get("status") == "ok":
                log.info(f"[{coin}] Leverage set to {leverage}×")
                return True
            log.warning(f"[{coin}] Leverage update returned: {result}")
            return False
        except Exception as e:
            log.error(f"[{coin}] Failed to set leverage: {e}")
            return False

    # ── Order placement ───────────────────────────────────

    def market_buy(self, coin: str, size_coin: float,
                   slippage: float = 0.01) -> OrderResult:
        return self._market_order(coin, is_buy=True,
                                  size_coin=size_coin, slippage=slippage)

    def market_sell(self, coin: str, size_coin: float,
                    slippage: float = 0.01) -> OrderResult:
        return self._market_order(coin, is_buy=False,
                                  size_coin=size_coin, slippage=slippage)

    def _market_order(self, coin: str, is_buy: bool,
                      size_coin: float, slippage: float) -> OrderResult:
        if not self._connected:
            return OrderResult(success=False, error="Not connected")
        spec = get_hyperliquid_market_spec(coin)
        if not spec:
            return OrderResult(success=False, error=f"{coin} is not supported on Hyperliquid")
        if spec.get("market_type") == "spot":
            if not self.allow_spot_execution:
                return OrderResult(
                    success=False,
                    error=f"{coin} spot execution is disabled in the live Hyperliquid client",
                )
            if not is_buy:
                return OrderResult(
                    success=False,
                    error=f"{coin} is a long-only Hyperliquid spot market; short sells are blocked",
                )
        try:
            venue_symbol = str(spec.get("venue_symbol") or resolve_hyperliquid_symbol(coin)).upper()
            result = self._exchange.market_open(
                name     = venue_symbol,
                is_buy   = is_buy,
                sz       = size_coin,
                slippage = slippage,
            )
            if result.get("status") == "ok":
                fill = result.get("response", {}).get("data", {})
                filled_px = 0.0
                filled_sz = 0.0
                statuses  = fill.get("statuses", [])
                if statuses:
                    s = statuses[0]
                    filled_px = float(s.get("filled", {}).get("avgPx", 0) or 0)
                    filled_sz = float(s.get("filled", {}).get("totalSz", 0) or 0)
                log.info(
                    f"[{coin}] {'BUY' if is_buy else 'SELL'} filled: "
                    f"{filled_sz} @ ${filled_px:.2f}"
                )
                return OrderResult(
                    success      = True,
                    filled_price = filled_px,
                    filled_size  = filled_sz,
                )
            else:
                err = str(result)
                log.error(f"[{coin}] Order failed: {err}")
                return OrderResult(success=False, error=err)
        except Exception as e:
            log.error(f"[{coin}] market_order exception: {e}")
            return OrderResult(success=False, error=str(e))

    def close_position(self, coin: str) -> OrderResult:
        if not self._connected:
            return OrderResult(success=False, error="Not connected")
        spec = get_hyperliquid_market_spec(coin)
        if not spec:
            return OrderResult(success=False, error=f"{coin} is not supported on Hyperliquid")
        if spec.get("market_type") == "spot":
            return OrderResult(
                success=False,
                error=f"{coin} spot close handling is not enabled in the live Hyperliquid client yet",
            )
        try:
            venue_symbol = str(spec.get("venue_symbol") or resolve_hyperliquid_symbol(coin)).upper()
            result = self._exchange.market_close(venue_symbol)
            if result.get("status") == "ok":
                log.info(f"[{coin}] Position closed successfully")
                return OrderResult(success=True)
            err = str(result)
            log.error(f"[{coin}] Close failed: {err}")
            return OrderResult(success=False, error=err)
        except Exception as e:
            log.error(f"[{coin}] close_position exception: {e}")
            return OrderResult(success=False, error=str(e))

    @classmethod
    def supported_coins_for_mode(cls, *, include_spot: bool = False) -> list[str]:
        return get_hyperliquid_supported_coins(
            include_spot=include_spot,
            live_tradeable_only=not include_spot,
        )

    def supported_coins(self) -> list[str]:
        return self.supported_coins_for_mode(include_spot=self.allow_spot_execution)
