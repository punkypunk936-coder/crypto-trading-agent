"""
indicators/candlestick_patterns.py — Pure OHLCV candlestick pattern recognition.

No API calls, no screenshots — works directly from the price candle data the
agent already fetches every cycle.

Patterns detected
─────────────────
Bullish (boost LONG signals):
  • Hammer           — long lower wick, small body at top (reversal after drop)
  • Bullish Engulfing — green candle body swallows previous red candle body
  • Morning Star     — 3-candle bottom reversal pattern
  • Three Soldiers   — 3 consecutive solid green candles
  • Bullish Marubozu — large green candle, tiny wicks (strong buying)
  • Bullish Pin Bar   — strong rejection of lower prices (long lower wick)

Bearish (boost SHORT signals):
  • Shooting Star     — long upper wick, small body at bottom (reversal after rise)
  • Bearish Engulfing — red candle body swallows previous green candle body
  • Evening Star     — 3-candle top reversal pattern
  • Three Crows      — 3 consecutive solid red candles
  • Bearish Marubozu — large red candle, tiny wicks (strong selling)
  • Bearish Pin Bar   — strong rejection of higher prices (long upper wick)

Neutral / indecision:
  • Doji              — open ≈ close (market uncertainty)
  • Spinning Top      — small body, long wicks both sides

Output
──────
PatternSignal.score  — 0 (max bearish) … 50 (neutral) … 100 (max bullish)
PatternSignal.patterns — list of pattern names found on the last 3 candles
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
import numpy as np

from logger import get_logger

log = get_logger("candles")


# ── How much each pattern moves the neutral score (50) ───────────────────────

PATTERN_SCORES: dict = {
    # Bullish patterns  → positive delta
    "Hammer":            +22,
    "Bullish Engulfing": +28,
    "Morning Star":      +30,
    "Three Soldiers":    +25,
    "Bullish Marubozu":  +18,
    "Bullish Pin Bar":   +15,
    # Bearish patterns  → negative delta
    "Shooting Star":     -22,
    "Bearish Engulfing": -28,
    "Evening Star":      -30,
    "Three Crows":       -25,
    "Bearish Marubozu":  -18,
    "Bearish Pin Bar":   -15,
    # Neutral
    "Doji":               0,
    "Spinning Top":       0,
}


@dataclass
class PatternSignal:
    coin: str
    score: float              # 0–100 (50 = no clear pattern)
    patterns: List[str]       # names of patterns found
    last_candle_bullish: bool # True if last closed candle was green
    body_size_pct: float      # last candle body / price (as %)
    trend_3: str              # "UP" | "DOWN" | "FLAT" (last 3 candles)
    valid: bool = True
    error: str  = ""


def compute_candlestick_patterns(df: pd.DataFrame, coin: str) -> PatternSignal:
    """
    Main entry point. Pass the standard OHLCV DataFrame (columns: open, high,
    low, close, volume). Needs at least 5 rows; returns neutral on error.
    """
    try:
        if df is None or len(df) < 5:
            return PatternSignal(coin=coin, score=50.0, patterns=[],
                                 last_candle_bullish=True, body_size_pct=0.0,
                                 trend_3="FLAT", valid=False,
                                 error="Insufficient candle data")

        # Normalise column names to lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)

        n = len(df)
        patterns: List[str] = []

        # ── Last 3 candle indices ─────────────────────────────────────
        i0 = n - 3   # three candles ago
        i1 = n - 2   # two candles ago
        i2 = n - 1   # most recent closed candle

        c2, o2, h2, l2 = c[i2], o[i2], h[i2], l[i2]
        c1, o1, h1, l1 = c[i1], o[i1], h[i1], l[i1]
        c0, o0, h0, l0 = c[i0], o[i0], h[i0], l[i0]

        # Helpers
        def body(ci, oi):       return abs(ci - oi)
        def upper_wick(ci, oi, hi): return hi - max(ci, oi)
        def lower_wick(ci, oi, li): return min(ci, oi) - li
        def candle_range(hi, li): return hi - li if hi != li else 1e-9
        def is_green(ci, oi):   return ci > oi
        def is_red(ci, oi):     return ci < oi

        r2 = candle_range(h2, l2)
        r1 = candle_range(h1, l1)
        r0 = candle_range(h0, l0)

        b2 = body(c2, o2)
        b1 = body(c1, o1)
        b0 = body(c0, o0)

        # ── Single-candle patterns (most recent) ─────────────────────

        # Doji: body < 5% of range
        if r2 > 0 and b2 / r2 < 0.05:
            patterns.append("Doji")

        # Spinning Top: body < 30% of range, wicks on both sides
        elif r2 > 0 and b2 / r2 < 0.30:
            uw = upper_wick(c2, o2, h2)
            lw = lower_wick(c2, o2, l2)
            if uw > b2 * 0.5 and lw > b2 * 0.5:
                patterns.append("Spinning Top")

        # Hammer (bullish): small body at top, lower wick ≥ 2× body, tiny upper wick
        # Best after a downtrend — check that prev candle was down
        elif (is_green(c2, o2) or b2 / r2 < 0.35) and r2 > 0:
            lw = lower_wick(c2, o2, l2)
            uw = upper_wick(c2, o2, h2)
            if lw >= 2.0 * b2 and uw <= 0.3 * b2 and is_red(c1, o1):
                patterns.append("Hammer")

        # Shooting Star (bearish): small body at bottom, upper wick ≥ 2× body
        # Best after an uptrend
        elif r2 > 0:
            uw = upper_wick(c2, o2, h2)
            lw = lower_wick(c2, o2, l2)
            if uw >= 2.0 * b2 and lw <= 0.3 * b2 and is_green(c1, o1):
                patterns.append("Shooting Star")

        # Bullish Marubozu: large green candle, tiny wicks
        if is_green(c2, o2) and r2 > 0 and b2 / r2 > 0.90:
            patterns.append("Bullish Marubozu")

        # Bearish Marubozu: large red candle, tiny wicks
        elif is_red(c2, o2) and r2 > 0 and b2 / r2 > 0.90:
            patterns.append("Bearish Marubozu")

        # Bullish Pin Bar: close in upper 30% of range, long lower wick
        if r2 > 0:
            pos = (c2 - l2) / r2   # 0=bottom, 1=top
            lw = lower_wick(c2, o2, l2)
            uw = upper_wick(c2, o2, h2)
            if pos > 0.70 and lw > 2 * uw and lw > r2 * 0.40:
                if "Hammer" not in patterns:
                    patterns.append("Bullish Pin Bar")

        # Bearish Pin Bar: close in lower 30% of range, long upper wick
        if r2 > 0:
            pos = (c2 - l2) / r2
            uw = upper_wick(c2, o2, h2)
            lw = lower_wick(c2, o2, l2)
            if pos < 0.30 and uw > 2 * lw and uw > r2 * 0.40:
                if "Shooting Star" not in patterns:
                    patterns.append("Bearish Pin Bar")

        # ── Two-candle patterns ──────────────────────────────────────

        # Bullish Engulfing: red then green, green body > red body
        if is_red(c1, o1) and is_green(c2, o2):
            if c2 > o1 and o2 < c1:   # green open < red close, green close > red open
                patterns.append("Bullish Engulfing")

        # Bearish Engulfing: green then red, red body > green body
        if is_green(c1, o1) and is_red(c2, o2):
            if o2 > c1 and c2 < o1:   # red open > green close, red close < green open
                patterns.append("Bearish Engulfing")

        # ── Three-candle patterns ────────────────────────────────────

        # Morning Star (bullish reversal): red → small/doji → green
        if is_red(c0, o0) and is_green(c2, o2):
            small_middle = b1 < b0 * 0.5 and b1 < b2 * 0.5
            if small_middle and c2 > (c0 + o0) / 2:
                patterns.append("Morning Star")

        # Evening Star (bearish reversal): green → small/doji → red
        if is_green(c0, o0) and is_red(c2, o2):
            small_middle = b1 < b0 * 0.5 and b1 < b2 * 0.5
            if small_middle and c2 < (c0 + o0) / 2:
                patterns.append("Evening Star")

        # Three White Soldiers: 3 consecutive green candles, each closing higher
        if (is_green(c0, o0) and is_green(c1, o1) and is_green(c2, o2)
                and c1 > c0 and c2 > c1
                and b0 > r0 * 0.5 and b1 > r1 * 0.5 and b2 > r2 * 0.5):
            patterns.append("Three Soldiers")

        # Three Black Crows: 3 consecutive red candles, each closing lower
        if (is_red(c0, o0) and is_red(c1, o1) and is_red(c2, o2)
                and c1 < c0 and c2 < c1
                and b0 > r0 * 0.5 and b1 > r1 * 0.5 and b2 > r2 * 0.5):
            patterns.append("Three Crows")

        # ── Score from detected patterns ─────────────────────────────
        delta = sum(PATTERN_SCORES.get(p, 0) for p in patterns)

        # Cap at ±40 (gives score 10–90 range max)
        delta = max(-40, min(40, delta))
        score = 50.0 + delta
        score = round(max(0.0, min(100.0, score)), 2)

        # ── Trend of last 3 candles ───────────────────────────────────
        if c2 > c0 * 1.002:
            trend_3 = "UP"
        elif c2 < c0 * 0.998:
            trend_3 = "DOWN"
        else:
            trend_3 = "FLAT"

        body_size_pct = round(b2 / c2 * 100, 3) if c2 > 0 else 0.0

        if patterns:
            log.info(
                f"[{coin}] Candles: {patterns} | score={score:.0f}/100 | "
                f"trend3={trend_3} | last={'🟢' if is_green(c2, o2) else '🔴'}"
            )
        else:
            log.debug(
                f"[{coin}] Candles: no pattern | score=50 | trend3={trend_3}"
            )

        return PatternSignal(
            coin              = coin,
            score             = score,
            patterns          = patterns,
            last_candle_bullish = bool(is_green(c2, o2)),
            body_size_pct     = body_size_pct,
            trend_3           = trend_3,
            valid             = True,
        )

    except Exception as e:
        log.warning(f"[{coin}] Candlestick pattern error: {e}")
        return PatternSignal(coin=coin, score=50.0, patterns=[],
                             last_candle_bullish=True, body_size_pct=0.0,
                             trend_3="FLAT", valid=False, error=str(e))


# ── Quick test: python -m indicators.candlestick_patterns ────────────────────
if __name__ == "__main__":
    import sys, json
    from data.market_data import fetch_candles
    coin = sys.argv[1].upper() if len(sys.argv) > 1 else "BTC"
    df = fetch_candles(coin=coin, interval="1h", lookback=50)
    sig = compute_candlestick_patterns(df, coin)
    print(f"\n{coin} Candlestick Analysis")
    print(f"  Score   : {sig.score}/100")
    print(f"  Patterns: {sig.patterns or ['None']}")
    print(f"  Trend 3h: {sig.trend_3}")
    print(f"  Last candle: {'Green' if sig.last_candle_bullish else 'Red'}")
