"""
exchanges/hyperliquid_client.py
Connector for Hyperliquid perpetuals DEX.

Authentication: EVM private key (no centralised API key needed).
SDK: hyperliquid-python-sdk (pip install hyperliquid-python-sdk)

Hyperliquid markets use coin tickers directly (e.g. "BTC", "ETH", "SOL").
All positions are USD-margined perpetuals.
"""

import time
from typing import Optional

import requests
from logger import get_logger
from exchanges.base import BaseExchange, AccountState, OrderResult

log = get_logger("hyperliquid")


class HyperliquidClient(BaseExchange):
    name = "Hyperliquid"
    _SUPPORTED_CACHE: list[str] | None = None
    _SUPPORTED_CACHE_TS: float = 0.0
    _SUPPORTED_CACHE_TTL: float = 300.0

    def __init__(self, private_key: str, account_address: str, mainnet: bool = True):
        self.private_key     = private_key
        self.account_address = account_address
        self.mainnet         = mainnet
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
        try:
            result = self._exchange.update_leverage(
                leverage=leverage, coin=coin, is_cross=True
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
        try:
            result = self._exchange.market_open(
                coin     = coin,
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
        try:
            result = self._exchange.market_close(coin)
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
    def supported_coins(cls):
        now = time.time()
        if cls._SUPPORTED_CACHE and (now - cls._SUPPORTED_CACHE_TS) < cls._SUPPORTED_CACHE_TTL:
            return list(cls._SUPPORTED_CACHE)

        fallback = ["BTC", "ETH", "SOL", "HYPE", "TAO"]
        try:
            resp = requests.post("https://api.hyperliquid.xyz/info", json={"type": "meta"}, timeout=5)
            resp.raise_for_status()
            universe = resp.json().get("universe", []) or []
            configured = {"BTC", "ETH", "SOL", "HYPE", "TAO", "SP500", "XAU"}
            supported = sorted(
                {
                    str(item.get("name") or "").upper()
                    for item in universe
                    if str(item.get("name") or "").upper() in configured
                }
            )
            if supported:
                cls._SUPPORTED_CACHE = supported
                cls._SUPPORTED_CACHE_TS = now
                return list(supported)
        except Exception as exc:
            log.warning(f"Failed to refresh Hyperliquid supported coins from live meta: {exc}")

        cls._SUPPORTED_CACHE = fallback
        cls._SUPPORTED_CACHE_TS = now
        return list(fallback)
