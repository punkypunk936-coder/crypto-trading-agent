"""
exchanges/lighter_client.py
Connector for Lighter perpetuals using the current lighter-sdk.

Current Lighter SDK requirements:
  - an L1 wallet private key (used for one-time bootstrap / account lookup)
  - a Lighter API private key
  - a Lighter account index

The older Client / ClientConfig interface no longer exists in lighter-sdk 1.x.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

from logger import get_logger
from exchanges.base import BaseExchange, AccountState, OrderResult, LimitOrderStatus

log = get_logger("lighter")

DEFAULT_LIGHTER_API_BASE_URL = "https://mainnet.zklighter.elliot.ai"

# Map our coin tickers to Lighter's numeric perpetual market IDs.
# Verified against live Lighter market metadata on April 8, 2026.
COIN_TO_MARKET_ID: Dict[str, int] = {
    "ETH": 0,
    "BTC": 1,
    "SOL": 2,
    "TAO": 13,
    "HYPE": 24,
    "SP500": 42,   # Lighter venue symbol is SPX; we keep SP500 as the internal alias.
    "XAU": 92,
}

MIN_ORDER_SIZE: Dict[str, float] = {
    "BTC": 0.0002,
    "ETH": 0.005,
    "SOL": 0.05,
    "TAO": 0.05,
    "HYPE": 0.50,
    "SP500": 5.0,
    "XAU": 0.003,
}


def _set_ca_env(cert_path: str) -> None:
    os.environ.setdefault("SSL_CERT_FILE", cert_path)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
    os.environ.setdefault("CURL_CA_BUNDLE", cert_path)


def _run(coro):
    return asyncio.run(coro)


def _float(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


def derive_l1_address(l1_private_key: str) -> str:
    from eth_account import Account

    return Account.from_key(l1_private_key).address


async def _async_account_api_call(api_base_url: str, ssl_ca_cert: str, method_name: str, **kwargs):
    import lighter

    api_client = lighter.ApiClient(lighter.Configuration(host=api_base_url, ssl_ca_cert=ssl_ca_cert))
    try:
        method = getattr(lighter.AccountApi(api_client), method_name)
        return await method(**kwargs)
    finally:
        await api_client.close()


async def _async_order_api_call(api_base_url: str, ssl_ca_cert: str, method_name: str, **kwargs):
    import lighter

    api_client = lighter.ApiClient(lighter.Configuration(host=api_base_url, ssl_ca_cert=ssl_ca_cert))
    try:
        method = getattr(lighter.OrderApi(api_client), method_name)
        return await method(**kwargs)
    finally:
        await api_client.close()


def _is_missing_account_error(exc: Exception) -> bool:
    status = getattr(exc, "status", None)
    body = str(getattr(exc, "body", "") or "").lower()
    reason = str(getattr(exc, "reason", "") or "").lower()
    text = str(exc).lower()
    return status == 400 and "account not found" in f"{body} {reason} {text}"


def _missing_account_message(address: str) -> str:
    return (
        f"No Lighter account found for wallet {address}. "
        "Open lighter.xyz with this wallet first and create/fund the account, "
        "then run: python3 main.py --lighter-bootstrap"
    )


def bootstrap_lighter_api(
    *,
    l1_private_key: str,
    api_base_url: str = DEFAULT_LIGHTER_API_BASE_URL,
    api_key_index: int = 1,
) -> Dict[str, Any]:
    """
    Create and register a Lighter API key for the wallet's main account.

    This only works if the wallet already has a Lighter account.
    """
    try:
        import certifi
        import lighter
        from lighter.signer_client import SignerClient, create_api_key
    except ImportError as exc:
        return {"ok": False, "error": f"lighter-sdk dependency missing: {exc}"}

    if not l1_private_key:
        return {"ok": False, "error": "LIGHTER_L1_PRIVATE_KEY is missing."}

    _set_ca_env(certifi.where())
    address = derive_l1_address(l1_private_key)
    try:
        accounts = _run(
            _async_account_api_call(
                api_base_url,
                certifi.where(),
                "accounts_by_l1_address",
                l1_address=address,
            )
        )
        sub_accounts = list(getattr(accounts, "sub_accounts", []) or [])
        if not sub_accounts:
            return {
                "ok": False,
                "address": address,
                "error": _missing_account_message(address),
            }

        account_index = int(getattr(sub_accounts[0], "index"))
        api_private_key, api_public_key, error = create_api_key()
        if error:
            return {"ok": False, "address": address, "account_index": account_index, "error": error}

        signer = SignerClient(
            url=api_base_url,
            account_index=account_index,
            api_private_keys={api_key_index: api_private_key},
        )
        response, error = _run(
            signer.change_api_key(
                eth_private_key=l1_private_key,
                new_pubkey=api_public_key,
                api_key_index=api_key_index,
            )
        )
        if error:
            return {"ok": False, "address": address, "account_index": account_index, "error": error}

        return {
            "ok": True,
            "address": address,
            "account_index": account_index,
            "api_key_index": api_key_index,
            "api_private_key": api_private_key,
            "response": response.to_dict() if hasattr(response, "to_dict") else str(response),
        }
    except Exception as exc:
        if _is_missing_account_error(exc):
            return {"ok": False, "address": address, "error": _missing_account_message(address)}
        return {"ok": False, "address": address, "error": str(exc)}
    finally:
        if "signer" in locals():
            _run(signer.close())


class LighterClient(BaseExchange):
    name = "Lighter"

    @staticmethod
    def supported_market_symbols() -> List[str]:
        return list(COIN_TO_MARKET_ID.keys())

    def __init__(
        self,
        private_key: str = "",
        web3_url: str = "",
        *,
        l1_private_key: str = "",
        api_private_key: str = "",
        account_index: str | int = "",
        api_key_index: int = 1,
        api_base_url: str = DEFAULT_LIGHTER_API_BASE_URL,
    ):
        # `private_key` is kept as a legacy alias for the L1 wallet key.
        self.l1_private_key = l1_private_key or private_key
        self.api_private_key = api_private_key
        self.account_index = int(account_index) if str(account_index or "").strip() else None
        self.api_key_index = int(api_key_index)
        self.api_base_url = api_base_url or DEFAULT_LIGHTER_API_BASE_URL
        self.web3_url = web3_url

        self._client = None
        self._connected = False
        self._l1_address = ""
        self._ssl_ca_cert = ""

    def _resolve_account_index(self) -> Optional[int]:
        import certifi

        if self.account_index is not None:
            return self.account_index
        if not self.l1_private_key:
            return None

        self._ssl_ca_cert = certifi.where()
        _set_ca_env(self._ssl_ca_cert)
        self._l1_address = derive_l1_address(self.l1_private_key)
        try:
            accounts = _run(
                _async_account_api_call(
                    self.api_base_url,
                    self._ssl_ca_cert,
                    "accounts_by_l1_address",
                    l1_address=self._l1_address,
                )
            )
            sub_accounts = list(getattr(accounts, "sub_accounts", []) or [])
            if not sub_accounts:
                return None
            self.account_index = int(getattr(sub_accounts[0], "index"))
            return self.account_index
        except Exception as exc:
            if _is_missing_account_error(exc):
                return None
            raise

    def _auth_token(self) -> str:
        auth, error = self._client.create_auth_token_with_expiry(api_key_index=self.api_key_index)
        if error:
            raise RuntimeError(error)
        return auth

    def _find_order(self, coin: str, order_ref: str, include_inactive: bool = True):
        if not self._connected:
            return None
        market_id = COIN_TO_MARKET_ID.get(coin)
        if market_id is None:
            return None

        auth = self._auth_token()
        active = _run(
            _async_order_api_call(
                self.api_base_url,
                self._ssl_ca_cert,
                "account_active_orders",
                account_index=self.account_index,
                market_id=market_id,
                auth=auth,
            )
        )
        orders = list(getattr(active, "orders", []) or [])
        if include_inactive:
            inactive = _run(
                _async_order_api_call(
                    self.api_base_url,
                    self._ssl_ca_cert,
                    "account_inactive_orders",
                    account_index=self.account_index,
                    limit=50,
                    market_id=market_id,
                    auth=auth,
                )
            )
            orders.extend(list(getattr(inactive, "orders", []) or []))

        for order in orders:
            refs = {
                str(getattr(order, "order_index", "")),
                str(getattr(order, "order_id", "")),
                str(getattr(order, "client_order_index", "")),
                str(getattr(order, "client_order_id", "")),
            }
            if str(order_ref) in refs:
                return order
        return None

    def _predicted_price(self, market_id: int, size: float, is_ask: bool) -> float:
        try:
            px, _ = _run(self._client.get_potential_execution_price(market_id, size, is_ask))
            return _float(px)
        except Exception:
            return 0.0

    def connect(self) -> bool:
        try:
            import certifi
            from lighter.signer_client import SignerClient

            self._ssl_ca_cert = certifi.where()
            _set_ca_env(self._ssl_ca_cert)

            if not self.l1_private_key:
                log.error("LIGHTER_L1_PRIVATE_KEY is missing. Add it to .env.")
                return False
            if not self.api_private_key:
                log.error("LIGHTER_API_PRIVATE_KEY is missing. Run: python3 main.py --lighter-bootstrap")
                return False

            self._l1_address = derive_l1_address(self.l1_private_key)
            resolved_index = self._resolve_account_index()
            if resolved_index is None:
                log.error(_missing_account_message(self._l1_address))
                return False

            self._client = SignerClient(
                url=self.api_base_url,
                account_index=resolved_index,
                api_private_keys={self.api_key_index: self.api_private_key},
            )
            err = self._client.check_client()
            if err:
                log.error("Lighter API key check failed: %s", err)
                log.error("Run: python3 main.py --lighter-bootstrap")
                return False

            self._connected = True
            log.info("Connected to Lighter DEX")
            log.info("Lighter account index: %s", self.account_index)
            return True

        except ImportError as exc:
            log.error("lighter-sdk import failed: %s", exc)
            log.error("Use the project venv and install requirements first.")
            return False
        except Exception as exc:
            log.error("Lighter connection failed: %s", exc)
            return False

    def supports_limit_orders(self) -> bool:
        return True

    def supported_coins(self) -> List[str]:
        return self.supported_market_symbols()

    def get_account_state(self) -> Optional[AccountState]:
        if not self._connected:
            log.error("Not connected to Lighter")
            return None
        try:
            response = _run(
                _async_account_api_call(
                    self.api_base_url,
                    self._ssl_ca_cert,
                    "account",
                    by="index",
                    value=str(self.account_index),
                )
            )
            accounts = list(getattr(response, "accounts", []) or [])
            if not accounts:
                raise RuntimeError("Empty account payload from Lighter")

            account = accounts[0]
            positions: List[dict] = []
            for pos in list(getattr(account, "positions", []) or []):
                size = _float(getattr(pos, "position", 0))
                sign = int(getattr(pos, "sign", 0) or 0)
                signed_size = abs(size) if sign >= 0 else -abs(size)
                if signed_size == 0:
                    continue
                symbol = str(getattr(pos, "symbol", "") or "")
                coin = symbol.split("-", 1)[0].upper() if symbol else next(
                    (c for c, mid in COIN_TO_MARKET_ID.items() if mid == getattr(pos, "market_id", None)),
                    str(getattr(pos, "market_id", "")),
                )
                positions.append(
                    {
                        "coin": coin,
                        "size": signed_size,
                        "direction": "LONG" if signed_size > 0 else "SHORT",
                        "entry_price": _float(getattr(pos, "avg_entry_price", 0)),
                        "unrealised_pnl": _float(getattr(pos, "unrealized_pnl", 0)),
                    }
                )

            total_equity = _float(getattr(account, "total_asset_value", 0)) or _float(getattr(account, "collateral", 0))
            available = _float(getattr(account, "available_balance", 0))
            return AccountState(total_equity_usd=total_equity, available_usd=available, positions=positions)
        except Exception as exc:
            log.error("Failed to get Lighter account state: %s", exc)
            return None

    def set_leverage(self, coin: str, leverage: int) -> bool:
        log.debug("[%s] Lighter leverage set to %sx (strategy-level only)", coin, leverage)
        return True

    def market_buy(self, coin: str, size_coin: float, slippage: float = 0.01) -> OrderResult:
        return self._place_order(coin, side="buy", size=size_coin, order_type="market", slippage=slippage)

    def market_sell(self, coin: str, size_coin: float, slippage: float = 0.01) -> OrderResult:
        return self._place_order(coin, side="sell", size=size_coin, order_type="market", slippage=slippage)

    def limit_buy(self, coin: str, size_coin: float, limit_price: float) -> OrderResult:
        return self._place_order(coin, side="buy", size=size_coin, order_type="limit", price=limit_price)

    def limit_sell(self, coin: str, size_coin: float, limit_price: float) -> OrderResult:
        return self._place_order(coin, side="sell", size=size_coin, order_type="limit", price=limit_price)

    def cancel_order(self, coin: str, order_id: str) -> bool:
        if not self._connected:
            return False
        market_id = COIN_TO_MARKET_ID.get(coin)
        if market_id is None:
            return False
        try:
            order = self._find_order(coin, order_id, include_inactive=False)
            order_index = int(getattr(order, "order_index", order_id))
            _, response, error = _run(
                self._client.cancel_order(market_id, order_index, api_key_index=self.api_key_index)
            )
            if error:
                log.error("[%s] cancel_order failed: %s", coin, error)
                return False
            log.info("[%s] Lighter limit order %s cancelled", coin, order_index)
            return bool(getattr(response, "tx_hash", ""))
        except Exception as exc:
            log.error("[%s] cancel_order exception: %s", coin, exc)
            return False

    def get_order_status(self, coin: str, order_id: str) -> LimitOrderStatus:
        if not self._connected:
            return LimitOrderStatus(order_id=order_id, coin=coin)
        try:
            order = self._find_order(coin, order_id, include_inactive=True)
            if not order:
                return LimitOrderStatus(order_id=order_id, coin=coin)
            status = str(getattr(order, "status", "") or "").lower()
            filled_size = _float(getattr(order, "filled_base_amount", 0))
            filled_quote = _float(getattr(order, "filled_quote_amount", 0))
            filled_price = filled_quote / filled_size if filled_size else _float(getattr(order, "price", 0))
            return LimitOrderStatus(
                order_id=str(getattr(order, "order_index", order_id)),
                coin=coin,
                filled="filled" in status,
                cancelled=("cancel" in status) or ("expired" in status),
                filled_price=filled_price,
                filled_size=filled_size,
            )
        except Exception as exc:
            log.error("[%s] get_order_status exception: %s", coin, exc)
            return LimitOrderStatus(order_id=order_id, coin=coin)

    def _place_order(
        self,
        coin: str,
        side: str,
        size: float,
        order_type: str = "market",
        price: float = 0.0,
        slippage: float = 0.01,
        reduce_only: bool = False,
    ) -> OrderResult:
        if not self._connected:
            return OrderResult(success=False, error="Not connected")

        market_id = COIN_TO_MARKET_ID.get(coin)
        if market_id is None:
            return OrderResult(success=False, error=f"{coin} is not supported on Lighter")

        min_sz = MIN_ORDER_SIZE.get(coin, 0.0001)
        if size < min_sz:
            return OrderResult(success=False, error=f"Size {size} below minimum {min_sz} for {coin}")

        try:
            is_ask = side == "sell"
            client_order_index = int(time.time() * 1000)
            predicted_price = self._predicted_price(market_id, size, is_ask)

            if order_type == "market":
                _, response, error = _run(
                    self._client.create_market_order_limited_slippage(
                        market_id,
                        client_order_index,
                        size,
                        max(0.001, slippage),
                        is_ask,
                        reduce_only=reduce_only,
                        api_key_index=self.api_key_index,
                    )
                )
            else:
                _, response, error = _run(
                    self._client.create_order(
                        market_id,
                        client_order_index,
                        size,
                        price,
                        is_ask,
                        self._client.ORDER_TYPE_LIMIT,
                        self._client.ORDER_TIME_IN_FORCE_GOOD_TILL_TIME,
                        reduce_only=reduce_only,
                        api_key_index=self.api_key_index,
                    )
                )

            if error:
                log.error("[%s] Lighter %s %s failed: %s", coin, order_type.upper(), side.upper(), error)
                return OrderResult(success=False, error=str(error))

            order = self._find_order(coin, str(client_order_index), include_inactive=True)
            order_ref = str(getattr(order, "order_index", client_order_index))
            filled_size = _float(getattr(order, "filled_base_amount", 0)) or size
            if order is not None:
                filled_quote = _float(getattr(order, "filled_quote_amount", 0))
                filled_price = filled_quote / filled_size if filled_quote and filled_size else _float(getattr(order, "price", 0))
            else:
                filled_price = price if order_type == "limit" and price > 0 else predicted_price

            log.info(
                "[%s] Lighter %s %s accepted: size=%s order_ref=%s tx=%s",
                coin,
                order_type.upper(),
                side.upper(),
                size,
                order_ref,
                getattr(response, "tx_hash", ""),
            )
            return OrderResult(
                success=True,
                order_id=order_ref,
                filled_price=filled_price,
                filled_size=filled_size,
            )
        except Exception as exc:
            log.error("[%s] Lighter _place_order exception: %s", coin, exc)
            return OrderResult(success=False, error=str(exc))

    def close_position(self, coin: str) -> OrderResult:
        if not self._connected:
            return OrderResult(success=False, error="Not connected")
        try:
            state = self.get_account_state()
            if not state:
                return OrderResult(success=False, error="Cannot get account state")

            pos = next((p for p in state.positions if p["coin"] == coin), None)
            if not pos:
                log.info("[%s] No open Lighter position to close", coin)
                return OrderResult(success=True)

            close_side = "sell" if pos["direction"] == "LONG" else "buy"
            result = self._place_order(
                coin,
                side=close_side,
                size=abs(pos["size"]),
                order_type="market",
                reduce_only=True,
                slippage=0.01,
            )
            if result.success:
                log.info("[%s] Lighter position closed", coin)
            return result
        except Exception as exc:
            log.error("[%s] close_position exception: %s", coin, exc)
            return OrderResult(success=False, error=str(exc))
