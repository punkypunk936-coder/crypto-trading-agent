"""Compact, serializable price-action context for open-position management."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if pd.notna(result) else default


def _timestamp(value: Any) -> str:
    if value is None:
        return ""
    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.notna(parsed):
            return parsed.isoformat()
    except Exception:
        pass
    return str(value)


def build_position_price_action(
    frame: pd.DataFrame | None,
    *,
    interval: str = "1h",
    max_bars: int = 48,
) -> dict:
    """Build chart bars and a volatility-normalized dip-resilience profile."""
    empty = {
        "valid": False,
        "interval": interval,
        "bars": [],
        "bar_count": 0,
        "pullback_pct": 0.0,
        "pullback_atr": 0.0,
        "rebound_pct": 0.0,
        "rebound_atr": 0.0,
        "recovery_fraction": 0.0,
        "bullish_lower_wicks": 0,
        "dip_absorption_active": False,
        "v_reversal_active": False,
        "volatility_normal": False,
        "structural_damage": False,
        "resilience_score": 0.0,
        "status": "NO_DATA",
        "summary": "Price-action history is not available yet.",
    }
    required = {"open", "high", "low", "close"}
    if frame is None or frame.empty or not required.issubset(frame.columns):
        return empty

    safe = frame.tail(max(12, int(max_bars or 48))).copy().reset_index(drop=True)
    if len(safe) < 6:
        return empty

    bars = []
    for index, row in safe.iterrows():
        timestamp_value = row.get("timestamp") if "timestamp" in safe.columns else index
        bars.append({
            "time": _timestamp(timestamp_value),
            "open": round(_number(row.get("open")), 6),
            "high": round(_number(row.get("high")), 6),
            "low": round(_number(row.get("low")), 6),
            "close": round(_number(row.get("close")), 6),
            "volume": round(_number(row.get("volume")), 4),
        })

    highs = safe["high"].astype(float).reset_index(drop=True)
    lows = safe["low"].astype(float).reset_index(drop=True)
    closes = safe["close"].astype(float).reset_index(drop=True)
    opens = safe["open"].astype(float).reset_index(drop=True)
    previous_closes = closes.shift(1)
    true_ranges = pd.concat(
        [
            highs - lows,
            (highs - previous_closes).abs(),
            (lows - previous_closes).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = max(_number(true_ranges.tail(min(14, len(true_ranges))).mean()), 1e-9)

    running_peak = highs.cummax()
    drawdowns = (lows - running_peak) / running_peak.replace(0, pd.NA)
    trough_index = int(drawdowns.fillna(0.0).idxmin())
    peak_index = int(highs.iloc[: trough_index + 1].idxmax())
    peak = _number(highs.iloc[peak_index])
    trough = _number(lows.iloc[trough_index])
    current = _number(closes.iloc[-1])
    pullback_pct = ((trough - peak) / peak * 100.0) if peak > 0 else 0.0
    rebound_pct = ((current - trough) / trough * 100.0) if trough > 0 else 0.0
    pullback_atr = max(0.0, (peak - trough) / atr)
    rebound_atr = max(0.0, (current - trough) / atr)
    recovery_fraction = (
        max(0.0, min(1.25, (current - trough) / (peak - trough)))
        if peak > trough
        else 1.0
    )

    bullish_lower_wicks = 0
    for index in range(max(0, len(safe) - 8), len(safe)):
        full_range = _number(highs.iloc[index] - lows.iloc[index])
        if full_range <= 0:
            continue
        lower_wick = min(_number(opens.iloc[index]), _number(closes.iloc[index])) - _number(lows.iloc[index])
        close_location = (_number(closes.iloc[index]) - _number(lows.iloc[index])) / full_range
        if lower_wick / full_range >= 0.34 and close_location >= 0.55:
            bullish_lower_wicks += 1

    dip_event = bool(pullback_pct <= -1.0 or pullback_atr >= 0.75)
    recent_upturn = bool(len(closes) >= 3 and current > _number(closes.iloc[-3]))
    v_reversal_active = bool(
        dip_event
        and trough_index > peak_index
        and recovery_fraction >= 0.50
        and rebound_atr >= 0.75
    )
    dip_absorption_active = bool(
        v_reversal_active
        or bullish_lower_wicks >= 2
        or (dip_event and recovery_fraction >= 0.35 and recent_upturn)
    )
    structural_damage = bool(
        dip_event
        and pullback_atr >= 3.0
        and recovery_fraction < 0.25
        and not recent_upturn
    )
    volatility_normal = bool(not structural_damage or dip_absorption_active)
    resilience_score = 50.0
    resilience_score += min(28.0, recovery_fraction * 32.0)
    resilience_score += min(12.0, bullish_lower_wicks * 4.0)
    if v_reversal_active:
        resilience_score += 10.0
    if structural_damage:
        resilience_score -= 35.0
    resilience_score = max(0.0, min(100.0, resilience_score))

    if structural_damage:
        status = "STRUCTURAL_DAMAGE"
        summary = (
            f"Selloff is {pullback_atr:.1f} ATR and only {recovery_fraction * 100:.0f}% reclaimed; "
            "price has not shown enough absorption yet."
        )
    elif v_reversal_active:
        status = "V_REVERSAL"
        summary = (
            f"Dip reached {pullback_atr:.1f} ATR and has reclaimed {recovery_fraction * 100:.0f}%; "
            "the tape is showing a V-shaped recovery."
        )
    elif dip_absorption_active:
        status = "DIP_ABSORPTION"
        summary = (
            f"Dip is {pullback_atr:.1f} ATR with {recovery_fraction * 100:.0f}% reclaimed and "
            f"{bullish_lower_wicks} recent buyer-absorption wick(s)."
        )
    elif dip_event:
        status = "NORMAL_PULLBACK"
        summary = (
            f"Pullback is {pullback_atr:.1f} ATR with {recovery_fraction * 100:.0f}% reclaimed; "
            "it remains inside the normal volatility envelope."
        )
    else:
        status = "TREND_ALIGNED"
        summary = "No material pullback is active; price remains near the recent range high."

    return {
        "valid": True,
        "interval": interval,
        "bars": bars,
        "bar_count": len(bars),
        "current_price": round(current, 6),
        "recent_peak": round(peak, 6),
        "recent_trough": round(trough, 6),
        "atr": round(atr, 6),
        "pullback_pct": round(pullback_pct, 3),
        "pullback_atr": round(pullback_atr, 3),
        "rebound_pct": round(rebound_pct, 3),
        "rebound_atr": round(rebound_atr, 3),
        "recovery_fraction": round(recovery_fraction, 4),
        "bullish_lower_wicks": bullish_lower_wicks,
        "dip_absorption_active": dip_absorption_active,
        "v_reversal_active": v_reversal_active,
        "volatility_normal": volatility_normal,
        "structural_damage": structural_damage,
        "resilience_score": round(resilience_score, 1),
        "status": status,
        "summary": summary,
    }
