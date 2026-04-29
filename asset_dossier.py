"""
asset_dossier.py — living asset intelligence for the trading agent.

Each dossier is a compact research snapshot that answers:
  • what this asset is doing now
  • what unlocks the next trade
  • what invalidates the idea
  • what the bot recently learned on this symbol
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterable

from logger import get_logger
from paths import ASSET_DOSSIERS_JSON

log = get_logger("asset_dossier")


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _pick_level(values: Any, *, prefer: str = "min", fallback: Any = None) -> float | None:
    numbers: list[float] = []
    for value in list(values or []):
        number = _safe_float(value)
        if number > 0:
            numbers.append(number)
    fallback_number = _safe_float(fallback)
    if fallback_number > 0:
        numbers.append(fallback_number)
    if not numbers:
        return None
    return min(numbers) if prefer == "min" else max(numbers)


def _asset_bucket(instrument_type: str) -> str:
    return "coin" if str(instrument_type or "crypto").lower() == "crypto" else "equity"


def _primary_reason(*values: Any) -> str:
    for raw in values:
        text = str(raw or "").replace("|", "·")
        for part in [piece.strip() for piece in text.split("·")]:
            lowered = part.lower()
            if not part:
                continue
            if lowered.startswith("score ") or lowered.startswith("map:") or lowered.startswith("breakout state:"):
                continue
            return part
    return ""


def _instrument_type_for_coin(coin: str, signal: dict | None, state: dict | None) -> str:
    signal_type = _safe_str((signal or {}).get("instrument_type")).lower()
    if signal_type:
        return signal_type
    instrument_types = dict(((state or {}).get("config") or {}).get("instrument_types") or {})
    return _safe_str(instrument_types.get(str(coin or "").upper()), "crypto").lower()


def _latest_trade_by_coin(trades: Iterable[dict] | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for trade in list(trades or []):
        if not isinstance(trade, dict):
            continue
        coin = _safe_str(trade.get("coin")).upper()
        if not coin:
            continue
        previous = out.get(coin)
        if previous is None:
            out[coin] = dict(trade)
            continue
        if _safe_str(trade.get("closed_at")) >= _safe_str(previous.get("closed_at")):
            out[coin] = dict(trade)
    return out


def _missed_by_coin(missed_move_report: dict | None) -> dict[str, dict]:
    by_coin: dict[str, dict] = {}
    report = dict(missed_move_report or {})
    for item in list(report.get("top_missed_assets") or []):
        coin = _safe_str((item or {}).get("coin")).upper()
        if coin:
            by_coin[coin] = {"misses": int((item or {}).get("misses") or 0)}
    for item in list(report.get("recent_missed_moves") or []):
        coin = _safe_str((item or {}).get("coin")).upper()
        if not coin:
            continue
        slot = by_coin.setdefault(coin, {"misses": 0})
        slot.setdefault("latest", dict(item or {}))
    replay = dict(report.get("daily_top_mover_replay") or {})
    for item in list(replay.get("missed_top_movers") or []):
        coin = _safe_str((item or {}).get("coin")).upper()
        if not coin:
            continue
        slot = by_coin.setdefault(coin, {"misses": 0})
        slot["daily_top_mover_replay"] = dict(item or {})
    return by_coin


def _playbook_distiller_asset(playbook_distiller_report: dict | None, coin: str) -> dict:
    report = dict(playbook_distiller_report or {})
    return dict((report.get("assets") or {}).get(str(coin or "").upper()) or {})


def _playbook(entry: dict, signal: dict, instrument_type: str, distilled_asset: dict | None = None) -> str:
    distilled_asset = dict(distilled_asset or {})
    distilled = _safe_str(distilled_asset.get("playbook"))
    if distilled:
        return distilled
    manual = _safe_str(entry.get("trade_mode"))
    if manual:
        return manual
    bias = _safe_str(signal.get("market_map_bias") or entry.get("bias"), "NEUTRAL").upper()
    if bias == "BULLISH":
        return "Buy reclaims and disciplined pullbacks; do not chase stretched moves."
    if bias == "BEARISH":
        return "Fade failed reclaims and clean breakdowns; avoid shorting into demand."
    if instrument_type == "equity":
        return "Wait for cleaner session structure, catalyst alignment, and cleaner fills."
    return "Wait for structure, order-flow, and invalidation to agree before acting."


def build_report(
    *,
    state: dict,
    trades: Iterable[dict] | None,
    market_map: dict | None,
    missed_move_report: dict | None = None,
    llm_referee_report: dict | None = None,
    playbook_distiller_report: dict | None = None,
) -> dict:
    safe_state = dict(state or {})
    signals = dict(safe_state.get("signals") or {})
    positions = list(safe_state.get("positions") or [])
    position_map = {str(item.get("coin") or "").upper(): dict(item or {}) for item in positions}
    config = dict(safe_state.get("config") or {})
    tracked = set()
    tracked.update(str(coin or "").upper() for coin in config.get("coins") or [])
    tracked.update(str(coin or "").upper() for coin in config.get("analysis_coins") or [])
    tracked.update(str(coin or "").upper() for coin in config.get("dynamic_analysis_coins") or [])
    tracked.update(str(coin or "").upper() for coin in signals.keys())
    tracked.update(str(coin or "").upper() for coin in position_map.keys())
    tracked.update(str(coin or "").upper() for coin in dict((market_map or {}).get("coins") or {}).keys())

    map_entries = dict((market_map or {}).get("coins") or {})
    latest_trades = _latest_trade_by_coin(trades)
    missed = _missed_by_coin(missed_move_report)
    llm_verdicts = dict((llm_referee_report or {}).get("verdicts") or {})

    assets: dict[str, dict] = {}
    focus_rank: list[tuple[tuple[int, int, float, float], str]] = []
    tradable_count = 0

    for coin in sorted(item for item in tracked if item):
        sig = dict(signals.get(coin) or {})
        pos = dict(position_map.get(coin) or {})
        entry = dict(map_entries.get(coin) or {})
        instrument_type = _instrument_type_for_coin(coin, sig, safe_state)
        asset_bucket = _asset_bucket(instrument_type)
        action = _safe_str(sig.get("action"), "FLAT").upper()
        bias = _safe_str(sig.get("market_map_bias") or entry.get("bias"), "NEUTRAL").upper()
        asset_state = _safe_str(sig.get("asset_state"), "OBSERVING").upper()
        tradable = (_safe_str(sig.get("execution_mode"), "observation_only") == "tradable") or bool(pos)
        if tradable:
            tradable_count += 1

        support = _pick_level(entry.get("supports"), prefer="max", fallback=sig.get("market_map_nearest_support"))
        resistance = _pick_level(entry.get("resistances"), prefer="min", fallback=sig.get("market_map_nearest_resistance"))
        long_trigger = _pick_level(entry.get("daily_close_long_above"), prefer="min", fallback=resistance)
        short_trigger = _pick_level(entry.get("daily_close_short_below"), prefer="max", fallback=support)
        latest_trade = latest_trades.get(coin, {})
        missed_coin = dict(missed.get(coin) or {})
        llm = dict(llm_verdicts.get(coin) or {})
        distilled_asset = _playbook_distiller_asset(playbook_distiller_report, coin)
        live_price = _safe_float(sig.get("live_price") or sig.get("price"))
        score = round(_safe_float(sig.get("score") or 50.0), 1)
        probability = round(_safe_float(sig.get("expectancy_probability") or 0.50), 4)
        expected_r = round(_safe_float(sig.get("expectancy_expected_r") or 0.0), 3)

        now_view = _primary_reason(
            sig.get("decision_reason"),
            sig.get("thesis_summary"),
            sig.get("market_map_summary"),
            sig.get("analog_summary"),
        ) or "No clean edge right now."
        next_unblock = _primary_reason(sig.get("next_unblock_reason"), llm.get("next_unblock")) or (
            f"Hold above {long_trigger:,.2f}" if action == "LONG" and long_trigger else
            f"Break below {short_trigger:,.2f}" if action == "SHORT" and short_trigger else
            "Wait for structure and execution quality to align."
        )
        invalidation = _primary_reason(
            llm.get("invalidation_focus"),
            (
                f"Lose {support:,.2f}" if support and (action == "LONG" or bias == "BULLISH")
                else f"Reclaim {resistance:,.2f}" if resistance and (action == "SHORT" or bias == "BEARISH")
                else ""
            ),
        ) or "No clear invalidation level recorded yet."

        assets[coin] = {
            "coin": coin,
            "instrument_type": instrument_type,
            "asset_bucket": asset_bucket,
            "tradable": tradable,
            "execution_mode": _safe_str(sig.get("execution_mode"), "observation_only"),
            "asset_state": asset_state,
            "bias": bias,
            "action": action,
            "confidence": _safe_str(sig.get("confidence"), "LOW").upper(),
            "score": score,
            "expectancy_probability": probability,
            "expectancy_expected_r": expected_r,
            "live_price": round(live_price, 6) if live_price > 0 else 0.0,
            "levels": {
                "support": support,
                "resistance": resistance,
                "long_trigger": long_trigger,
                "short_trigger": short_trigger,
            },
            "dossier": {
                "current_read": now_view,
                "next_unblock": next_unblock,
                "invalidation": invalidation,
                "playbook": _playbook(entry, sig, instrument_type, distilled_asset),
                "narrative": _safe_str(sig.get("narrative_summary")) or "No special narrative risk flagged.",
                "analog": _safe_str(sig.get("analog_summary")) or "No strong analog edge recorded yet.",
                "recent_lesson": _safe_str(latest_trade.get("agent_lesson")) or "No symbol-specific lesson recorded yet.",
                "recent_trade_logic": _safe_str(latest_trade.get("open_logic")),
            },
            "missed_move_context": {
                "miss_count": int(missed_coin.get("misses") or 0),
                "latest": dict(missed_coin.get("latest") or {}),
                "daily_top_mover_replay": dict(missed_coin.get("daily_top_mover_replay") or {}),
            },
            "playbook_distiller": distilled_asset,
            "llm_referee": llm,
        }

        rank = (
            1 if pos else 0,
            1 if action in {"LONG", "SHORT"} else 0,
            abs(score - 50.0),
            probability,
        )
        focus_rank.append((rank, coin))

    focus_assets = [coin for _rank, coin in sorted(focus_rank, reverse=True)[:8]]
    report = {
        "updated_at": int(time.time()),
        "last_cycle": safe_state.get("last_cycle"),
        "count": len(assets),
        "tradable_count": tradable_count,
        "summary": {
            "focus_assets": focus_assets,
            "coin_count": sum(1 for item in assets.values() if item.get("asset_bucket") == "coin"),
            "equity_count": sum(1 for item in assets.values() if item.get("asset_bucket") == "equity"),
            "llm_enabled": bool((llm_referee_report or {}).get("enabled", False)),
            "missed_win_count": int(((missed_move_report or {}).get("summary") or {}).get("missed_win_count", 0) or 0),
        },
        "assets": assets,
    }
    return report


def build_and_save_report(
    *,
    state: dict,
    trades: Iterable[dict] | None,
    market_map: dict | None,
    missed_move_report: dict | None = None,
    llm_referee_report: dict | None = None,
    playbook_distiller_report: dict | None = None,
) -> dict:
    report = build_report(
        state=state,
        trades=trades,
        market_map=market_map,
        missed_move_report=missed_move_report,
        llm_referee_report=llm_referee_report,
        playbook_distiller_report=playbook_distiller_report,
    )
    try:
        ASSET_DOSSIERS_JSON.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        log.debug("asset_dossiers.json write failed: %s", exc)
    return report
