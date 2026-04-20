from __future__ import annotations

import json
import math
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from re import findall
from typing import Any


HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
TRADEXYZ_DEX = "xyz"
TRADEXYZ_PREFIX = "xyz:"
DEFAULT_START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
WINDOW_MS = 45 * 24 * 60 * 60 * 1000
MIN_SPLIT_WINDOW_MS = 6 * 60 * 60 * 1000
FILL_RESPONSE_CAP = 2000
MAX_REQUESTS = 160
REQUEST_TIMEOUT_SECONDS = 20


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _iso_from_ms(timestamp_ms: int | float | None) -> str | None:
    if not timestamp_ms:
        return None
    try:
        return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _safe_float(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except Exception:
        return 0.0
    return number if math.isfinite(number) else 0.0


def _extract_addresses(payload: Any) -> set[str]:
    matches: set[str] = set()
    if isinstance(payload, str):
        for address in findall(r"0x[a-fA-F0-9]{40}", payload):
            matches.add(address.lower())
        return matches
    if isinstance(payload, dict):
        for value in payload.values():
            matches.update(_extract_addresses(value))
        return matches
    if isinstance(payload, (list, tuple, set)):
        for value in payload:
            matches.update(_extract_addresses(value))
    return matches


def _validate_wallet(wallet: str) -> str:
    text = str(wallet or "").strip()
    if len(text) != 42 or not text.startswith("0x"):
        raise ValueError("Wallet must be a 42-character EVM address.")
    try:
        int(text[2:], 16)
    except ValueError as exc:
        raise ValueError("Wallet must be a valid hexadecimal EVM address.") from exc
    return text.lower()


def _post_info(payload: dict[str, Any]) -> Any:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        HYPERLIQUID_INFO_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS, context=_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Hyperliquid request failed ({exc.code}): {detail or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Hyperliquid request failed: {exc.reason}") from exc


def inspect_wallet_identity(wallet: str) -> dict[str, Any]:
    safe_wallet = _validate_wallet(wallet)
    role_payload = _post_info({"type": "userRole", "user": safe_wallet}) or {}
    role = str((role_payload or {}).get("role") or "user").strip() or "user"
    role_data = dict((role_payload or {}).get("data") or {})
    abstraction_mode = _post_info({"type": "userAbstraction", "user": safe_wallet})
    dex_abstraction = _post_info({"type": "userDexAbstraction", "user": safe_wallet})
    sub_accounts = _post_info({"type": "subAccounts", "user": safe_wallet}) or []
    linked_addresses = set()
    linked_addresses.update(_extract_addresses(role_data))
    linked_addresses.update(_extract_addresses(dex_abstraction))
    linked_addresses.update(_extract_addresses(sub_accounts))
    linked_addresses.discard(safe_wallet)

    notes: list[str] = []
    role_lower = role.lower()
    if role_lower == "agent":
        linked_user = str(role_data.get("user") or "").strip().lower()
        raise ValueError(
            "This address is a Hyperliquid agent/API wallet. Use the actual user or sub-account address instead"
            + (f" (linked user: {linked_user})" if linked_user else ".")
        )
    if role_lower == "subaccount":
        master = str(role_data.get("master") or "").strip().lower()
        notes.append(
            "This address is a Hyperliquid sub-account. The checker stays pinned to this exact sub-account and does not intentionally roll up the master."
            + (f" Master: {master}." if master else "")
        )
    else:
        notes.append("This lookup is strict to the exact address you entered. It does not intentionally merge linked users, sub-accounts, or agents.")
    abstraction_text = str(abstraction_mode or "default").strip()
    if abstraction_text and abstraction_text.lower() != "default":
        notes.append(
            f"Hyperliquid reports this address in {abstraction_text} abstraction mode. Linked abstraction addresses can surface the same Trade.xyz history at the protocol layer."
        )
    if dex_abstraction not in (None, {}, [], "", "default"):
        notes.append(
            "A Hyperliquid dex-abstraction link exists for this address. If two linked addresses show the same Trade.xyz activity, that linkage is coming from Hyperliquid rather than from this checker."
        )
    if linked_addresses:
        notes.append("Linked Hyperliquid addresses detected: " + ", ".join(sorted(linked_addresses)) + ".")
    return {
        "requested_address": safe_wallet,
        "queried_address": safe_wallet,
        "query_scope": "strict_address",
        "role": role,
        "role_details": role_data,
        "abstraction_mode": abstraction_text or "default",
        "dex_abstraction": dex_abstraction,
        "linked_addresses": sorted(linked_addresses),
        "notes": notes,
    }


def load_tradexyz_universe() -> list[dict[str, Any]]:
    payload = {"type": "meta", "dex": TRADEXYZ_DEX}
    data = _post_info(payload)
    universe = list((data or {}).get("universe") or [])
    return [dict(item or {}) for item in universe if isinstance(item, dict)]


def _fetch_user_fills_window(wallet: str, start_time_ms: int, end_time_ms: int) -> list[dict[str, Any]]:
    payload = {
        "type": "userFillsByTime",
        "user": wallet,
        "startTime": int(start_time_ms),
        "endTime": int(end_time_ms),
        "aggregateByTime": True,
        "dex": TRADEXYZ_DEX,
    }
    data = _post_info(payload)
    if not isinstance(data, list):
        raise RuntimeError("Unexpected Hyperliquid fills payload.")
    return [dict(item or {}) for item in data if isinstance(item, dict)]


def _is_tradexyz_fill(fill: dict[str, Any], xyz_markets: set[str]) -> bool:
    coin = str(fill.get("coin") or "").strip()
    return coin in xyz_markets or coin.startswith(TRADEXYZ_PREFIX)


def _fill_key(fill: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(fill.get("hash") or ""),
        str(fill.get("coin") or ""),
        str(fill.get("oid") or ""),
        str(fill.get("time") or ""),
        str(fill.get("px") or ""),
        str(fill.get("sz") or ""),
        str(fill.get("side") or ""),
    )


def collect_tradexyz_fills(
    wallet: str,
    xyz_markets: set[str],
    *,
    start_time_ms: int = DEFAULT_START_MS,
    end_time_ms: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    end_ms = int(end_time_ms or time.time() * 1000)
    start_ms = int(start_time_ms or DEFAULT_START_MS)
    if end_ms <= start_ms:
        raise ValueError("End time must be after start time.")

    windows: list[tuple[int, int]] = []
    cursor = start_ms
    while cursor < end_ms:
        window_end = min(cursor + WINDOW_MS - 1, end_ms)
        windows.append((cursor, window_end))
        cursor = window_end + 1

    stack = list(reversed(windows))
    request_count = 0
    split_count = 0
    truncated_windows = 0
    seen: set[tuple[Any, ...]] = set()
    matched: list[dict[str, Any]] = []

    while stack:
        start, end = stack.pop()
        request_count += 1
        if request_count > MAX_REQUESTS:
            raise RuntimeError("Trade.xyz volume lookup exceeded the safe request budget. Narrow the range and try again.")

        fills = _fetch_user_fills_window(wallet, start, end)
        if len(fills) >= FILL_RESPONSE_CAP and (end - start) > MIN_SPLIT_WINDOW_MS:
            midpoint = start + ((end - start) // 2)
            split_count += 1
            stack.append((midpoint + 1, end))
            stack.append((start, midpoint))
            continue
        if len(fills) >= FILL_RESPONSE_CAP:
            truncated_windows += 1

        for fill in fills:
            if not _is_tradexyz_fill(fill, xyz_markets):
                continue
            key = _fill_key(fill)
            if key in seen:
                continue
            seen.add(key)
            matched.append(fill)

    matched.sort(key=lambda item: int(item.get("time") or 0), reverse=True)
    coverage = {
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "start_time": _iso_from_ms(start_ms),
        "end_time": _iso_from_ms(end_ms),
        "request_count": request_count,
        "split_count": split_count,
        "truncated_window_count": truncated_windows,
        "possible_truncation": truncated_windows > 0,
    }
    return matched, coverage


def summarize_tradexyz_fills(
    wallet: str,
    fills: list[dict[str, Any]],
    *,
    universe: list[dict[str, Any]] | None = None,
    coverage: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    universe = list(universe or [])
    coverage = dict(coverage or {})
    identity = dict(identity or {})
    markets: dict[str, dict[str, Any]] = {}
    total_volume = 0.0
    buy_volume = 0.0
    sell_volume = 0.0
    first_fill_ms: int | None = None
    last_fill_ms: int | None = None

    for fill in fills:
        coin = str(fill.get("coin") or "").strip()
        px = _safe_float(fill.get("px"))
        sz = abs(_safe_float(fill.get("sz")))
        if not coin or px <= 0 or sz <= 0:
            continue
        notional = abs(px * sz)
        total_volume += notional
        fill_time = int(fill.get("time") or 0)
        if fill_time > 0:
            first_fill_ms = fill_time if first_fill_ms is None else min(first_fill_ms, fill_time)
            last_fill_ms = fill_time if last_fill_ms is None else max(last_fill_ms, fill_time)

        side = str(fill.get("side") or "").upper()
        if side == "B":
            buy_volume += notional
        elif side == "A":
            sell_volume += notional

        entry = markets.setdefault(
            coin,
            {
                "coin": coin,
                "volume_usd": 0.0,
                "buy_volume_usd": 0.0,
                "sell_volume_usd": 0.0,
                "fills": 0,
                "first_fill_at": None,
                "last_fill_at": None,
            },
        )
        entry["volume_usd"] += notional
        if side == "B":
            entry["buy_volume_usd"] += notional
        elif side == "A":
            entry["sell_volume_usd"] += notional
        entry["fills"] += 1
        if fill_time > 0:
            iso = _iso_from_ms(fill_time)
            if entry["first_fill_at"] is None or fill_time < int(datetime.fromisoformat(entry["first_fill_at"]).timestamp() * 1000):
                entry["first_fill_at"] = iso
            if entry["last_fill_at"] is None or fill_time > int(datetime.fromisoformat(entry["last_fill_at"]).timestamp() * 1000):
                entry["last_fill_at"] = iso

    market_rows = sorted(
        [
            {
                **row,
                "volume_usd": round(row["volume_usd"], 2),
                "buy_volume_usd": round(row["buy_volume_usd"], 2),
                "sell_volume_usd": round(row["sell_volume_usd"], 2),
            }
            for row in markets.values()
        ],
        key=lambda row: row["volume_usd"],
        reverse=True,
    )

    preview = []
    for fill in fills[:12]:
        px = _safe_float(fill.get("px"))
        sz = abs(_safe_float(fill.get("sz")))
        preview.append(
            {
                "coin": str(fill.get("coin") or ""),
                "time": _iso_from_ms(fill.get("time")),
                "side": str(fill.get("side") or "").upper(),
                "notional_usd": round(abs(px * sz), 2),
                "price": px,
                "size": sz,
            }
        )

    return {
        "wallet": wallet,
        "dex": TRADEXYZ_DEX,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "identity": identity,
        "summary": {
            "total_volume_usd": round(total_volume, 2),
            "buy_volume_usd": round(buy_volume, 2),
            "sell_volume_usd": round(sell_volume, 2),
            "fill_count": len(fills),
            "market_count": len(market_rows),
            "first_fill_at": _iso_from_ms(first_fill_ms),
            "last_fill_at": _iso_from_ms(last_fill_ms),
            "tracked_markets": len(universe),
        },
        "coverage": coverage,
        "markets": market_rows,
        "fills_preview": preview,
        "tracked_markets": [str(item.get("name") or "") for item in universe if str(item.get("name") or "").strip()],
    }


def fetch_tradexyz_volume(
    wallet: str,
    *,
    start_time_ms: int = DEFAULT_START_MS,
    end_time_ms: int | None = None,
) -> dict[str, Any]:
    identity = inspect_wallet_identity(wallet)
    safe_wallet = identity["queried_address"]
    universe = load_tradexyz_universe()
    xyz_markets = {
        str(item.get("name") or "").strip()
        for item in universe
        if str(item.get("name") or "").strip()
    }
    fills, coverage = collect_tradexyz_fills(
        safe_wallet,
        xyz_markets,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )
    return summarize_tradexyz_fills(
        safe_wallet,
        fills,
        universe=universe,
        coverage=coverage,
        identity=identity,
    )
