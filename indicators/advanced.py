"""
indicators/advanced.py
Smart-money and structure-based indicators for perpetuals trading.

Indicators implemented:
  1. Fibonacci Retracement   — key S/R levels from swing H/L
  2. Market Structure Break  — Higher/Lower high/low identification + CHoCH/BOS
  3. Order Blocks            — institutional buy/sell zones
  4. Fair Value Gap (FVG)    — 3-candle imbalance / liquidity voids
  5. ATR                     — Average True Range (volatility context)

All return a normalised signal_score (0–100):
  > 60  → bullish bias
  < 40  → bearish bias
  40–60 → neutral / no edge
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from logger import get_logger

log = get_logger("advanced")


# ─────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────

@dataclass
class FibResult:
    score: float = 50.0
    fib_position: float = 0.5          # 0 = at swing low, 1 = at swing high
    swing_high: float = 0.0
    swing_low: float = 0.0
    levels: dict = field(default_factory=dict)
    nearest_level_name: str = ""
    nearest_level_price: float = 0.0
    description: str = ""


@dataclass
class MSBResult:
    score: float = 50.0
    msb_type: str = "NONE"            # BULLISH_BOS, BEARISH_BOS, BULLISH_CHOCH, BEARISH_CHOCH, NONE
    structure_trend: str = "RANGING"  # UPTREND, DOWNTREND, RANGING
    last_swing_high: float = 0.0
    last_swing_low: float = 0.0
    description: str = ""


@dataclass
class OrderBlockResult:
    score: float = 50.0
    inside_bullish_ob: bool = False
    inside_bearish_ob: bool = False
    bullish_obs: List[Tuple[float, float]] = field(default_factory=list)  # (high, low)
    bearish_obs: List[Tuple[float, float]] = field(default_factory=list)
    description: str = ""


@dataclass
class FVGResult:
    score: float = 50.0
    bullish_fvgs: List[Tuple[float, float]] = field(default_factory=list)  # (bottom, top)
    bearish_fvgs: List[Tuple[float, float]] = field(default_factory=list)
    inside_bullish_fvg: bool = False
    inside_bearish_fvg: bool = False
    description: str = ""


@dataclass
class ATRResult:
    atr: float = 0.0
    atr_pct: float = 0.0            # ATR as % of price
    volatility_label: str = "normal"  # "low", "normal", "high", "extreme"


# ─────────────────────────────────────────────────────────────
# 1. FIBONACCI RETRACEMENT
# ─────────────────────────────────────────────────────────────

def compute_fibonacci(df: pd.DataFrame, lookback: int = 60) -> FibResult:
    """
    Identify the dominant swing high and swing low over `lookback` candles,
    then position current price within the Fibonacci grid.

    Key levels: 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
    Golden ratio zones: 61.8% & 78.6% → strongest reversal magnets.
    """
    result = FibResult()
    if len(df) < lookback:
        return result

    window   = df.iloc[-lookback:]
    current  = float(df["close"].iloc[-1])
    sh       = float(window["high"].max())
    sl       = float(window["low"].min())
    fib_range= sh - sl

    result.swing_high = sh
    result.swing_low  = sl

    if fib_range <= 0:
        result.description = "Flat range — no fib signal"
        return result

    # Key Fibonacci levels
    fib_pcts = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
    fib_names = ["0%", "23.6%", "38.2%", "50%", "61.8%", "78.6%", "100%"]
    levels = {name: sl + pct * fib_range
              for name, pct in zip(fib_names, fib_pcts)}
    result.levels = levels

    # Where is price in the fib range? (0=at swing low, 1=at swing high)
    fib_pos = (current - sl) / fib_range
    result.fib_position = fib_pos

    # Find nearest fib level
    dists = {name: abs(current - price) for name, price in levels.items()}
    nearest = min(dists, key=dists.get)
    result.nearest_level_name  = nearest
    result.nearest_level_price = levels[nearest]
    proximity_pct = dists[nearest] / current * 100  # % distance from nearest level

    # ── Scoring logic ──────────────────────────────────────
    # Lower in the range → bullish (potential support / reversal up)
    # Higher in the range → bearish (potential resistance / reversal down)
    # Strong reversal zones: 61.8% (golden ratio), 78.6%, 38.2%
    NEAR = 0.8  # within 0.8% is "touching" a level

    if fib_pos <= 0.236:
        score = 78   # Near swing low → strong bullish zone
        desc  = f"Near swing low (fib 0–23.6%)"
    elif fib_pos <= 0.382:
        score = 68   # 23.6%–38.2% → bullish
        desc  = f"Bullish fib zone (23.6–38.2%)"
    elif fib_pos <= 0.500:
        score = 58   # 38.2%–50% → slight bullish
        desc  = f"Fib 38.2–50% (mild bullish)"
    elif fib_pos <= 0.618:
        score = 44   # 50%–61.8% → slight bearish
        desc  = f"Fib 50–61.8% (mild bearish)"
    elif fib_pos <= 0.786:
        score = 34   # 61.8%–78.6% → bearish
        desc  = f"Bearish fib zone (61.8–78.6%)"
    else:
        score = 24   # Near swing high → bearish
        desc  = f"Near swing high (fib 78.6–100%)"

    # Bonus: price touching a golden ratio level strongly amplifies the signal
    if proximity_pct < NEAR:
        level_pct = levels.get(nearest, current)
        if nearest in ["61.8%", "78.6%"]:
            if fib_pos < 0.618:   # price below 61.8 → bouncing up from key level
                score = min(score + 12, 88)
                desc += f" ← touching {nearest} (golden ratio support)"
            else:
                score = max(score - 12, 12)
                desc += f" ← at {nearest} (golden ratio resistance)"
        elif nearest in ["38.2%", "50%"]:
            score += 5 if fib_pos < 0.5 else -5
            desc  += f" ← at {nearest}"

    result.score       = round(score, 2)
    result.description = desc
    log.debug(f"Fibonacci: pos={fib_pos:.3f} score={score:.0f} | {desc}")
    return result


# ─────────────────────────────────────────────────────────────
# 2. MARKET STRUCTURE BREAK (MSB / BOS / CHoCH)
# ─────────────────────────────────────────────────────────────

def _find_swing_points(highs: pd.Series, lows: pd.Series,
                       strength: int = 3) -> Tuple[List, List]:
    """
    Identify swing highs and swing lows using a simple pivot approach.
    `strength` = number of candles each side that must be lower/higher.
    Returns lists of (index, price) tuples.
    """
    n = len(highs)
    swing_highs = []
    swing_lows  = []
    for i in range(strength, n - strength):
        h = highs.iloc[i]
        l = lows.iloc[i]
        is_sh = all(h >= highs.iloc[i - j] and h >= highs.iloc[i + j]
                    for j in range(1, strength + 1))
        is_sl = all(l <= lows.iloc[i - j] and l <= lows.iloc[i + j]
                    for j in range(1, strength + 1))
        if is_sh:
            swing_highs.append((i, h))
        if is_sl:
            swing_lows.append((i, l))
    return swing_highs, swing_lows


def compute_msb(df: pd.DataFrame, lookback: int = 50, strength: int = 3) -> MSBResult:
    """
    Detect Market Structure Breaks:

    BOS  (Break of Structure)     — continuation of existing trend
    CHoCH (Change of Character)   — reversal of existing trend

    Bullish BOS   : in uptrend, price breaks above last swing high
    Bearish BOS   : in downtrend, price breaks below last swing low
    Bullish CHoCH : in downtrend, price breaks above last Lower High → reversal
    Bearish CHoCH : in uptrend, price breaks below last Higher Low → reversal
    """
    result = MSBResult()
    window_size = min(lookback, len(df))
    if window_size < 20:
        return result

    window = df.iloc[-window_size:]
    highs  = window["high"]
    lows   = window["low"]
    closes = window["close"]
    current_close = float(closes.iloc[-1])
    current_high  = float(highs.iloc[-1])
    current_low   = float(lows.iloc[-1])

    swing_highs, swing_lows = _find_swing_points(highs, lows, strength)

    if not swing_highs or not swing_lows:
        result.description = "Not enough swing points to determine structure"
        return result

    # Last meaningful swing high and low (before current candle)
    last_sh_idx, last_sh = swing_highs[-1]
    last_sl_idx, last_sl = swing_lows[-1]
    result.last_swing_high = last_sh
    result.last_swing_low  = last_sl

    # ── Determine market structure trend ──────────────────
    # Need at least 2 swing highs and lows to judge trend
    trend = "RANGING"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        sh_rising = swing_highs[-1][1] > swing_highs[-2][1]
        sh_falling= swing_highs[-1][1] < swing_highs[-2][1]
        sl_rising = swing_lows[-1][1]  > swing_lows[-2][1]
        sl_falling= swing_lows[-1][1]  < swing_lows[-2][1]

        if sh_rising and sl_rising:
            trend = "UPTREND"
        elif sh_falling and sl_falling:
            trend = "DOWNTREND"
        else:
            trend = "RANGING"

    result.structure_trend = trend

    # ── Detect BOS / CHoCH ────────────────────────────────
    msb_type = "NONE"
    score    = 50.0

    if trend == "UPTREND":
        if current_high > last_sh:
            msb_type = "BULLISH_BOS"
            score    = 72      # continuation — high confidence long
            desc = f"Bullish BOS: broke above swing high {last_sh:.2f}"
        elif current_low < swing_lows[-1][1]:
            msb_type = "BEARISH_CHOCH"
            score    = 22      # reversal — strong short signal
            desc = f"Bearish CHoCH: uptrend broken, HL violated at {swing_lows[-1][1]:.2f}"
        else:
            score = 60
            desc  = "Uptrend intact (no break yet)"

    elif trend == "DOWNTREND":
        if current_low < last_sl:
            msb_type = "BEARISH_BOS"
            score    = 28      # continuation short
            desc = f"Bearish BOS: broke below swing low {last_sl:.2f}"
        elif current_high > swing_highs[-1][1]:
            msb_type = "BULLISH_CHOCH"
            score    = 78      # reversal — strong long signal
            desc = f"Bullish CHoCH: downtrend broken, LH violated at {swing_highs[-1][1]:.2f}"
        else:
            score = 40
            desc  = "Downtrend intact (no break yet)"

    else:  # RANGING
        # In range: score leans toward recent price action momentum
        mid = (last_sh + last_sl) / 2
        if current_close > mid:
            score = 55
            desc  = f"Ranging market, price above midpoint ({mid:.2f})"
        else:
            score = 45
            desc  = f"Ranging market, price below midpoint ({mid:.2f})"
        msb_type = "NONE"

    result.score       = round(score, 2)
    result.msb_type    = msb_type
    result.description = desc
    log.debug(f"MSB: trend={trend} type={msb_type} score={score:.0f} | {desc}")
    return result


# ─────────────────────────────────────────────────────────────
# 3. ORDER BLOCKS
# ─────────────────────────────────────────────────────────────

def compute_order_blocks(df: pd.DataFrame, lookback: int = 40,
                         impulse_candles: int = 3,
                         min_impulse_pct: float = 0.005) -> OrderBlockResult:
    """
    Order Blocks — institutional accumulation/distribution zones.

    Bullish OB : last BEARISH (red) candle before a strong bullish impulse move.
                 Price returning to this zone = potential long entry.
    Bearish OB : last BULLISH (green) candle before a strong bearish impulse move.
                 Price returning to this zone = potential short entry.

    Impulse = `impulse_candles` consecutive same-direction candles
              each moving at least `min_impulse_pct` of price.
    """
    result  = OrderBlockResult()
    n = len(df)
    if n < lookback + impulse_candles + 2:
        return result

    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    start  = max(0, n - lookback - impulse_candles)
    end    = n - impulse_candles - 1

    bullish_obs: List[Tuple[float, float]] = []
    bearish_obs: List[Tuple[float, float]] = []

    for i in range(start, end):
        # Check for bullish impulse starting at i+1
        bull_impulse = True
        for k in range(1, impulse_candles + 1):
            if i + k >= n:
                bull_impulse = False
                break
            c, o = closes[i + k], opens[i + k]
            if c <= o or (c - o) / max(o, 1e-9) < min_impulse_pct:
                bull_impulse = False
                break
            if k > 1 and closes[i + k] < closes[i + k - 1]:
                bull_impulse = False
                break

        if bull_impulse and closes[i] < opens[i]:  # OB candle is bearish
            bullish_obs.append((highs[i], lows[i]))

        # Check for bearish impulse starting at i+1
        bear_impulse = True
        for k in range(1, impulse_candles + 1):
            if i + k >= n:
                bear_impulse = False
                break
            c, o = closes[i + k], opens[i + k]
            if c >= o or (o - c) / max(o, 1e-9) < min_impulse_pct:
                bear_impulse = False
                break
            if k > 1 and closes[i + k] > closes[i + k - 1]:
                bear_impulse = False
                break

        if bear_impulse and closes[i] > opens[i]:  # OB candle is bullish
            bearish_obs.append((highs[i], lows[i]))

    current_price = float(closes[-1])
    result.bullish_obs = bullish_obs[-3:] if bullish_obs else []
    result.bearish_obs = bearish_obs[-3:] if bearish_obs else []

    score = 50.0
    inside_bull = False
    inside_bear = False

    # Check if price is currently inside a recent OB zone
    for ob_high, ob_low in result.bullish_obs:
        if ob_low <= current_price <= ob_high:
            inside_bull = True
            score += 18    # Strong bullish bias — price in OB support zone
            break

    for ob_high, ob_low in result.bearish_obs:
        if ob_low <= current_price <= ob_high:
            inside_bear = True
            score -= 18    # Strong bearish bias — price in OB resistance zone
            break

    # Approaching an OB (within 1% of zone)
    if not inside_bull and result.bullish_obs:
        ob_h, ob_l = result.bullish_obs[-1]
        if current_price > ob_l and current_price < ob_h * 1.01:
            score += 8

    if not inside_bear and result.bearish_obs:
        ob_h, ob_l = result.bearish_obs[-1]
        if current_price < ob_h and current_price > ob_l * 0.99:
            score -= 8

    score = max(0.0, min(100.0, score))

    result.score             = round(score, 2)
    result.inside_bullish_ob = inside_bull
    result.inside_bearish_ob = inside_bear

    parts = []
    if inside_bull:
        parts.append("Inside bullish OB zone (long bias)")
    if inside_bear:
        parts.append("Inside bearish OB zone (short bias)")
    if not parts:
        parts.append(f"{len(bullish_obs)} bull OBs, {len(bearish_obs)} bear OBs identified")
    result.description = " | ".join(parts)

    log.debug(f"OrderBlocks: score={score:.0f} bullOB={inside_bull} bearOB={inside_bear}")
    return result


# ─────────────────────────────────────────────────────────────
# 4. FAIR VALUE GAP (FVG)
# ─────────────────────────────────────────────────────────────

def compute_fvg(df: pd.DataFrame, lookback: int = 30) -> FVGResult:
    """
    Fair Value Gap — 3-candle imbalance pattern (liquidity void).

    Bullish FVG : candle[i].high  < candle[i+2].low
                  (price gapped up — gap acts as support when revisited)
    Bearish FVG : candle[i].low   > candle[i+2].high
                  (price gapped down — gap acts as resistance when revisited)

    When price trades back into an open FVG, it often fills and reverses.
    """
    result = FVGResult()
    n = len(df)
    if n < 10:
        return result

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    start = max(0, n - lookback)
    current_price = float(closes[-1])

    bullish_fvgs: List[Tuple[float, float]] = []   # (bottom, top) of gap
    bearish_fvgs: List[Tuple[float, float]] = []

    for i in range(start, n - 2):
        # Bullish FVG: gap between candle i high and candle i+2 low
        if lows[i + 2] > highs[i]:
            bottom = highs[i]
            top    = lows[i + 2]
            # Only keep unfilled gaps (price hasn't traded back inside)
            filled = any(lows[j] <= top and highs[j] >= bottom
                         for j in range(i + 3, n))
            if not filled:
                bullish_fvgs.append((bottom, top))

        # Bearish FVG: gap between candle i low and candle i+2 high
        if highs[i + 2] < lows[i]:
            bottom = highs[i + 2]
            top    = lows[i]
            filled = any(lows[j] <= top and highs[j] >= bottom
                         for j in range(i + 3, n))
            if not filled:
                bearish_fvgs.append((bottom, top))

    result.bullish_fvgs = bullish_fvgs[-4:] if bullish_fvgs else []
    result.bearish_fvgs = bearish_fvgs[-4:] if bearish_fvgs else []

    score = 50.0
    inside_bull_fvg = False
    inside_bear_fvg = False

    # Is price inside a bullish FVG? (bouncing off gap support)
    for fvg_bottom, fvg_top in result.bullish_fvgs:
        if fvg_bottom <= current_price <= fvg_top:
            inside_bull_fvg = True
            score += 14
            break
        # Approaching bullish FVG from above (about to fill)
        elif current_price > fvg_bottom and current_price < fvg_top * 1.005:
            score += 6

    # Is price inside a bearish FVG? (hitting gap resistance)
    for fvg_top, fvg_bottom in [(t, b) for b, t in result.bearish_fvgs]:
        if fvg_bottom <= current_price <= fvg_top:
            inside_bear_fvg = True
            score -= 14
            break
        elif current_price < fvg_top and current_price > fvg_bottom * 0.995:
            score -= 6

    score = max(0.0, min(100.0, score))

    result.score             = round(score, 2)
    result.inside_bullish_fvg= inside_bull_fvg
    result.inside_bearish_fvg= inside_bear_fvg

    parts = []
    if inside_bull_fvg:
        parts.append("Inside bullish FVG (gap support)")
    if inside_bear_fvg:
        parts.append("Inside bearish FVG (gap resistance)")
    if not parts:
        parts.append(f"{len(bullish_fvgs)} bull FVGs, {len(bearish_fvgs)} bear FVGs open")
    result.description = " | ".join(parts)

    log.debug(f"FVG: score={score:.0f} bullFVG={inside_bull_fvg} bearFVG={inside_bear_fvg}")
    return result


# ─────────────────────────────────────────────────────────────
# 5. ATR (Average True Range)
# ─────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> ATRResult:
    """
    ATR for volatility context.
    Used by the strategy to optionally scale stop-loss distances
    and to filter out entries during extreme volatility.
    """
    result = ATRResult()
    if len(df) < period + 1:
        return result

    highs  = df["high"]
    lows   = df["low"]
    closes = df["close"]

    prev_close = closes.shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows  - prev_close).abs(),
    ], axis=1).max(axis=1)

    atr = float(tr.ewm(span=period, adjust=False).mean().iloc[-1])
    price = float(closes.iloc[-1])
    atr_pct = atr / price * 100 if price > 0 else 0

    result.atr     = atr
    result.atr_pct = atr_pct

    if atr_pct < 0.5:
        result.volatility_label = "low"
    elif atr_pct < 1.5:
        result.volatility_label = "normal"
    elif atr_pct < 3.0:
        result.volatility_label = "high"
    else:
        result.volatility_label = "extreme"

    log.debug(f"ATR: {atr:.4f} ({atr_pct:.2f}%) → {result.volatility_label} volatility")
    return result


# ─────────────────────────────────────────────────────────────
# Combined advanced signal summary (convenience function)
# ─────────────────────────────────────────────────────────────

@dataclass
class AdvancedSignals:
    coin: str
    fib:   FibResult        = field(default_factory=FibResult)
    msb:   MSBResult        = field(default_factory=MSBResult)
    ob:    OrderBlockResult = field(default_factory=OrderBlockResult)
    fvg:   FVGResult        = field(default_factory=FVGResult)
    atr:   ATRResult        = field(default_factory=ATRResult)
    valid: bool             = True


def compute_advanced_signals(df: pd.DataFrame, coin: str) -> AdvancedSignals:
    """Run all advanced indicators on the given OHLCV DataFrame."""
    out = AdvancedSignals(coin=coin)
    if df is None or len(df) < 30:
        out.valid = False
        return out
    out.fib = compute_fibonacci(df)
    out.msb = compute_msb(df)
    out.ob  = compute_order_blocks(df)
    out.fvg = compute_fvg(df)
    out.atr = compute_atr(df)
    return out
