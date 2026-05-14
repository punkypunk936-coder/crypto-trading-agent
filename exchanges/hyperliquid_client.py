"""
exchanges/hyperliquid_client.py
Connector for Hyperliquid.

Authentication: EVM private key (no centralised API key needed).
SDK: hyperliquid-python-sdk (pip install hyperliquid-python-sdk)

Execution semantics:
  - perps are fully supported live
  - spot equities are supported as an opt-in long-only live lane
  - active-market gating still decides which live spot names are eligible
"""

from typing import Any, Optional

from logger import get_logger
from exchanges.base import BaseExchange, AccountState, LimitOrderStatus, OrderResult
from exchanges.hyperliquid_markets import (
    get_hyperliquid_market_dex,
    get_hyperliquid_market_spec,
    get_hyperliquid_supported_coins,
    get_hyperliquid_supported_dexes,
    hyperliquid_market_type,
    resolve_hyperliquid_internal_coin,
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

    def _active_user_address(self) -> str:
        return self.account_address or getattr(getattr(self, "_exchange", None), "wallet", None).address

    @staticmethod
    def _extract_order_result_payload(result: Any) -> dict:
        response = dict((result or {}).get("response", {}) or {})
        return dict(response.get("data", {}) or {})

    @staticmethod
    def _extract_oid(data: dict) -> str:
        statuses = list(data.get("statuses", []) or [])
        for status in statuses:
            if not isinstance(status, dict):
                continue
            resting = status.get("resting")
            if isinstance(resting, dict) and resting.get("oid") is not None:
                return str(resting.get("oid"))
            filled = status.get("filled")
            if isinstance(filled, dict) and filled.get("oid") is not None:
                return str(filled.get("oid"))
        return ""

    @staticmethod
    def _extract_fill(data: dict) -> tuple[float, float]:
        statuses = list(data.get("statuses", []) or [])
        for status in statuses:
            if not isinstance(status, dict):
                continue
            filled = status.get("filled")
            if isinstance(filled, dict):
                try:
                    return (
                        float(filled.get("avgPx", 0) or 0.0),
                        float(filled.get("totalSz", 0) or 0.0),
                    )
                except Exception:
                    return (0.0, 0.0)
        return (0.0, 0.0)

    @staticmethod
    def _extract_error(data: dict, fallback: str = "") -> str:
        statuses = list(data.get("statuses", []) or [])
        for status in statuses:
            if not isinstance(status, dict):
                continue
            if status.get("error"):
                return str(status.get("error"))
        return fallback

    def _spot_position_size(self, coin: str) -> float:
        if not self._connected:
            return 0.0
        spec = get_hyperliquid_market_spec(coin) or {}
        pair_name = str(spec.get("pair_name") or "").upper()
        base_symbol = pair_name.split("/")[0] if "/" in pair_name else str(coin or "").upper()
        try:
            state = self._info.spot_user_state(self._active_user_address())
        except Exception as exc:
            log.error("[%s] Failed to read Hyperliquid spot state: %s", coin, exc)
            return 0.0

        balances = list((state or {}).get("balances", []) or [])
        for balance in balances:
            if str(balance.get("coin") or "").upper() != base_symbol:
                continue
            for key in ("total", "hold", "available", "balance"):
                try:
                    value = float(balance.get(key, 0) or 0.0)
                except Exception:
                    value = 0.0
                if value > 0:
                    return value
        return 0.0

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
            perp_dexs = get_hyperliquid_supported_dexes() or None

            self._info     = Info(url, skip_ws=True, perp_dexs=perp_dexs)
            self._exchange = Exchange(
                account,
                url,
                account_address=self.account_address,
                perp_dexs=perp_dexs,
            )
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
            dexs = [""] + list(get_hyperliquid_supported_dexes())
            perp_states = []
            equity = 0.0
            avail = 0.0
            for dex in dexs:
                try:
                    state = self._info.user_state(self.account_address, dex)
                except Exception as exc:
                    label = dex or "native"
                    log.warning("Failed to read Hyperliquid %s perp state: %s", label, exc)
                    continue
                perp_states.append(state)
                margin = dict((state or {}).get("marginSummary") or {})
                try:
                    equity += float(margin.get("accountValue", 0) or 0.0)
                except Exception:
                    pass
                try:
                    avail += float(margin.get("totalRawUsd", 0) or 0.0)
                except Exception:
                    pass
            positions = []
            for state in perp_states:
                for p in state.get("assetPositions", []):
                    pos = p.get("position", {})
                    szi = float(pos.get("szi", 0))
                    if szi == 0:
                        continue
                    venue_coin = str(pos.get("coin") or "")
                    positions.append({
                        "coin":        resolve_hyperliquid_internal_coin(venue_coin),
                        "size":        szi,
                        "direction":   "LONG" if szi > 0 else "SHORT",
                        "entry_price": float(pos.get("entryPx", 0)),
                        "unrealised_pnl": float(pos.get("unrealizedPnl", 0)),
                        "leverage":    pos.get("leverage", {}).get("value", 1),
                    })
            if self.allow_spot_execution:
                try:
                    spot_state = self._info.spot_user_state(self._active_user_address())
                    balances = list((spot_state or {}).get("balances", []) or [])
                    for coin in self.supported_coins():
                        spec = get_hyperliquid_market_spec(coin) or {}
                        if spec.get("market_type") != "spot":
                            continue
                        pair_name = str(spec.get("pair_name") or "").upper()
                        base_symbol = pair_name.split("/")[0] if "/" in pair_name else str(coin).upper()
                        for balance in balances:
                            if str(balance.get("coin") or "").upper() != base_symbol:
                                continue
                            total = float(balance.get("total", 0) or 0.0)
                            if total <= 0:
                                continue
                            entry_ntl = float(balance.get("entryNtl", 0) or 0.0)
                            entry_price = (entry_ntl / total) if entry_ntl and total else 0.0
                            positions.append({
                                "coin": coin,
                                "size": total,
                                "direction": "LONG",
                                "entry_price": entry_price,
                                "unrealised_pnl": 0.0,
                                "leverage": 1,
                            })
                except Exception as exc:
                    log.warning("Failed to read Hyperliquid spot balances: %s", exc)
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
                leverage=leverage, name=venue_symbol, is_cross=True
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

    def limit_buy(self, coin: str, size_coin: float,
                  limit_price: float, maker_only: bool = False) -> OrderResult:
        return self._limit_order(
            coin,
            is_buy=True,
            size_coin=size_coin,
            limit_price=limit_price,
            maker_only=maker_only,
        )

    def limit_sell(self, coin: str, size_coin: float,
                   limit_price: float, maker_only: bool = False) -> OrderResult:
        return self._limit_order(
            coin,
            is_buy=False,
            size_coin=size_coin,
            limit_price=limit_price,
            maker_only=maker_only,
        )

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
            venue_symbol = str(spec.get("venue_symbol") or resolve_hyperliquid_symbol(coin)).strip()
            result = self._exchange.market_open(
                name     = venue_symbol,
                is_buy   = is_buy,
                sz       = size_coin,
                slippage = slippage,
            )
            if result.get("status") == "ok":
                fill = self._extract_order_result_payload(result)
                filled_px, filled_sz = self._extract_fill(fill)
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

    def _limit_order(
        self,
        coin: str,
        *,
        is_buy: bool,
        size_coin: float,
        limit_price: float,
        maker_only: bool = False,
    ) -> OrderResult:
        if not self._connected:
            return OrderResult(success=False, error="Not connected")
        spec = get_hyperliquid_market_spec(coin)
        if not spec:
            return OrderResult(success=False, error=f"{coin} is not supported on Hyperliquid")
        if spec.get("market_type") == "spot":
            if not self.allow_spot_execution:
                return OrderResult(success=False, error=f"{coin} spot execution is disabled in the live Hyperliquid client")
            if not is_buy:
                return OrderResult(success=False, error=f"{coin} is a long-only Hyperliquid spot market; short sells are blocked")
        try:
            venue_symbol = str(spec.get("venue_symbol") or resolve_hyperliquid_symbol(coin)).strip()
            tif = "Alo" if maker_only else "Gtc"
            result = self._exchange.order(
                name=venue_symbol,
                is_buy=is_buy,
                sz=size_coin,
                limit_px=limit_price,
                order_type={"limit": {"tif": tif}},
                reduce_only=False,
            )
            if result.get("status") != "ok":
                err = str(result)
                log.error("[%s] Limit order failed: %s", coin, err)
                return OrderResult(success=False, error=err)

            data = self._extract_order_result_payload(result)
            error = self._extract_error(data)
            if error:
                log.error("[%s] Limit order rejected: %s", coin, error)
                return OrderResult(success=False, error=error)

            order_id = self._extract_oid(data)
            filled_px, filled_sz = self._extract_fill(data)
            if filled_sz > 0:
                log.info("[%s] Limit order immediately filled: %s @ $%.2f", coin, filled_sz, filled_px)
            else:
                log.info("[%s] Limit order accepted (%s) id=%s @ $%.4f", coin, tif, order_id or "n/a", limit_price)
            return OrderResult(
                success=True,
                order_id=order_id,
                filled_price=filled_px,
                filled_size=filled_sz,
            )
        except Exception as exc:
            log.error("[%s] limit_order exception: %s", coin, exc)
            return OrderResult(success=False, error=str(exc))

    def cancel_order(self, coin: str, order_id: str) -> bool:
        if not self._connected:
            return False
        if not order_id:
            return True
        try:
            venue_symbol = resolve_hyperliquid_symbol(coin)
            result = self._exchange.cancel(venue_symbol, int(order_id))
            if result.get("status") == "ok":
                log.info("[%s] Hyperliquid limit order %s cancelled", coin, order_id)
                return True
            log.warning("[%s] Hyperliquid cancel returned: %s", coin, result)
            return False
        except Exception as exc:
            log.error("[%s] cancel_order exception: %s", coin, exc)
            return False

    def get_order_status(self, coin: str, order_id: str) -> LimitOrderStatus:
        if not self._connected or not order_id:
            return LimitOrderStatus(order_id=order_id, coin=coin)
        try:
            oid = int(order_id)
            raw = self._info.query_order_by_oid(self._active_user_address(), oid)
            status_text = str((raw or {}).get("status") or "").lower()
            if any(token in status_text for token in ("cancel", "reject", "expire")):
                return LimitOrderStatus(order_id=order_id, coin=coin, cancelled=True)
            if any(token in status_text for token in ("filled", "closed")):
                return LimitOrderStatus(order_id=order_id, coin=coin, filled=True)

            open_orders = self._info.open_orders(
                self._active_user_address(),
                get_hyperliquid_market_dex(coin) or "",
            )
            for open_order in open_orders or []:
                if str(open_order.get("oid") or "") == str(order_id):
                    return LimitOrderStatus(order_id=order_id, coin=coin, filled=False)

            if status_text and "open" not in status_text:
                return LimitOrderStatus(order_id=order_id, coin=coin, filled=True)
            return LimitOrderStatus(order_id=order_id, coin=coin, filled=False)
        except Exception as exc:
            log.error("[%s] get_order_status exception: %s", coin, exc)
            return LimitOrderStatus(order_id=order_id, coin=coin, filled=False)

    def close_position(self, coin: str) -> OrderResult:
        if not self._connected:
            return OrderResult(success=False, error="Not connected")
        spec = get_hyperliquid_market_spec(coin)
        if not spec:
            return OrderResult(success=False, error=f"{coin} is not supported on Hyperliquid")
        if spec.get("market_type") == "spot":
            if not self.allow_spot_execution:
                return OrderResult(success=False, error=f"{coin} spot execution is disabled in the live Hyperliquid client")
            size = self._spot_position_size(coin)
            if size <= 0:
                return OrderResult(success=False, error=f"No open spot balance found for {coin}")
            try:
                venue_symbol = str(spec.get("venue_symbol") or resolve_hyperliquid_symbol(coin)).strip()
                result = self._exchange.market_open(
                    name=venue_symbol,
                    is_buy=False,
                    sz=size,
                    slippage=0.02,
                )
                if result.get("status") == "ok":
                    data = self._extract_order_result_payload(result)
                    filled_px, filled_sz = self._extract_fill(data)
                    log.info(f"[{coin}] Spot position closed: {filled_sz} @ ${filled_px:.2f}")
                    return OrderResult(success=True, filled_price=filled_px, filled_size=filled_sz)
                err = str(result)
                log.error(f"[{coin}] Spot close failed: {err}")
                return OrderResult(success=False, error=err)
            except Exception as exc:
                log.error(f"[{coin}] close_position exception: {exc}")
                return OrderResult(success=False, error=str(exc))
        try:
            venue_symbol = str(spec.get("venue_symbol") or resolve_hyperliquid_symbol(coin)).strip()
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

    def supports_limit_orders(self) -> bool:
        return True
