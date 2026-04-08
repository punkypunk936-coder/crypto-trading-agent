"""
indicators/mtf.py — Multi-Timeframe Analysis (MTF)

Instead of trusting a 1H candle alone, we read the same indicators
across three timeframes and only trade when they agree.

Timeframes
──────────
  1H  — entry timing (current, already computed by the agent)
  4H  — medium-term momentum / trend direction
  12H — macro bias (is the bigger picture bullish or bearish?)

Logic
─────
  Each timeframe produces a bias: BULLISH | BEARISH | NEUTRAL
  The agent uses this to gate entries:

  • 12H BEARISH + 4H BEARISH → only allow SHORT trades on 1H, block LONGs
  • 12H BULLISH + 4H BULLISH → only allow LONG trades on 1H, block SHORTs
  • Disagreement          → no filter (trust 1H signal as-is)
  • One strong + one neutral → soft filter (reduce score by 10pts)

This prevents trading against the larger trend — the most common mistake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from data.market_data import fetch_candles, completed_candle_frame
from logger import get_logger

log = get_logger("mtf")

# How many candles to load per timeframe
LOOKBACK = 100


@dataclass
class TimeframeBias:
    """Bias reading for one timeframe."""
    tf: str               # "1h", "4h", "12h"
    bias: str             # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float       # 0-100 (how strongly biased)
    ema_trend: str        # "UP", "DOWN", "FLAT"
    rsi: float            # current RSI
    above_ema200: bool    # price above 200 EMA


@dataclass
class MTFAnalysis:
    """Combined multi-timeframe result for one coin."""
    coin: str
    bias_1h:  Optional[TimeframeBias]
    bias_4h:  Optional[TimeframeBias]
    bias_12h: Optional[TimeframeBias]
    combined_bias: str       # "BULLISH", "BEARISH", "NEUTRAL"
    score_adjustment: float  # pts to add/subtract from composite score
    allow_long: bool         # is a LONG entry allowed?
    allow_short: bool        # is a SHORT entry allowed?
    reason: str
    valid: bool = True


def compute_mtf(coin: str, candle_interval_1h: str = "1h") -> MTFAnalysis:
    """
    Fetch candles for all three timeframes and compute biases.
    Falls back gracefully if a timeframe is unavailable.
    """
    tf_map = {
        "1h":  "1h",
        "4h":  "4h",
        "12h": "12h",
    }

    biases = {}
    for key, tf in tf_map.items():
        df = fetch_candles(coin=coin, interval=tf, lookback=LOOKBACK)
        closed_df = completed_candle_frame(df)
        if closed_df is not None and len(closed_df) >= 20:
            biases[key] = _compute_bias(closed_df, tf)
        else:
            log.debug(f"[{coin}] No {tf} candles available — skipping this TF")
            biases[key] = None

    return _combine(coin, biases.get("1h"), biases.get("4h"), biases.get("12h"))


def _rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _compute_bias(df: pd.DataFrame, tf: str) -> TimeframeBias:
    """Compute bias for a single timeframe dataframe."""
    closes = df["close"].astype(float)
    highs  = df["high"].astype(float)
    lows   = df["low"].astype(float)
    price  = float(closes.iloc[-1])

    # EMAs
    ema20  = _ema(closes, 20)
    ema50  = _ema(closes, 50)
    ema200 = _ema(closes, 200) if len(closes) >= 200 else _ema(closes, len(closes) // 2)

    ema20_cur  = float(ema20.iloc[-1])
    ema50_cur  = float(ema50.iloc[-1])
    ema200_cur = float(ema200.iloc[-1])

    # EMA trend: compare short vs long EMA
    if ema20_cur > ema50_cur * 1.005:
        ema_trend = "UP"
    elif ema20_cur < ema50_cur * 0.995:
        ema_trend = "DOWN"
    else:
        ema_trend = "FLAT"

    above_ema200 = price > ema200_cur
    rsi_val      = _rsi(closes)

    # Score: bull signals count up, bear signals count down
    bull = 0
    bear = 0

    if price > ema20_cur:   bull += 1
    else:                   bear += 1
    if price > ema50_cur:   bull += 1
    else:                   bear += 1
    if above_ema200:        bull += 1
    else:                   bear += 1
    if ema_trend == "UP":   bull += 2
    elif ema_trend == "DOWN":bear+= 2
    if rsi_val > 55:        bull += 1
    elif rsi_val < 45:      bear += 1

    # Recent candle structure: higher highs / lower lows
    recent_highs = highs.iloc[-5:]
    recent_lows  = lows.iloc[-5:]
    if recent_highs.iloc[-1] > recent_highs.iloc[0]: bull += 1
    else:                                             bear += 1
    if recent_lows.iloc[-1] > recent_lows.iloc[0]:   bull += 1
    else:                                             bear += 1

    total = bull + bear
    if total == 0:
        strength = 50.0
        bias = "NEUTRAL"
    else:
        bull_pct = bull / total * 100
        if bull_pct >= 65:
            bias = "BULLISH"
        elif bull_pct <= 35:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"
        strength = bull_pct if bias == "BULLISH" else (100 - bull_pct) if bias == "BEARISH" else 50.0

    return TimeframeBias(
        tf=tf, bias=bias, strength=round(strength, 1),
        ema_trend=ema_trend, rsi=round(rsi_val, 1),
        above_ema200=above_ema200,
    )


def _combine(coin: str, b1h, b4h, b12h) -> MTFAnalysis:
    """Combine timeframe biases into a trading decision."""
    allow_long  = True
    allow_short = True
    score_adj   = 0.0
    parts       = []

    # 12H macro bias — strongest filter
    if b12h:
        parts.append(f"12H={b12h.bias}({b12h.strength:.0f})")
        if b12h.bias == "BEARISH":
            allow_long  = False
            score_adj  -= 8
            parts.append("12H macro bearish: blocking LONGs")
        elif b12h.bias == "BULLISH":
            allow_short = False
            score_adj  += 8
            parts.append("12H macro bullish: blocking SHORTs")

    # 4H medium bias — secondary filter
    if b4h:
        parts.append(f"4H={b4h.bias}({b4h.strength:.0f})")
        if b4h.bias == "BEARISH":
            if allow_long:   # only block if 12H didn't already
                score_adj -= 5
            if b12h and b12h.bias == "BEARISH":
                allow_long = False   # double bearish → hard block
        elif b4h.bias == "BULLISH":
            if allow_short:
                score_adj += 5
            if b12h and b12h.bias == "BULLISH":
                allow_short = False  # double bullish → hard block

    # 1H
    if b1h:
        parts.append(f"1H={b1h.bias}({b1h.strength:.0f})")

    # Determine combined bias
    biases = [b for b in [b12h, b4h, b1h] if b is not None]
    bull_count = sum(1 for b in biases if b.bias == "BULLISH")
    bear_count = sum(1 for b in biases if b.bias == "BEARISH")
    if bull_count > bear_count:
        combined = "BULLISH"
    elif bear_count > bull_count:
        combined = "BEARISH"
    else:
        combined = "NEUTRAL"

    log.info(
        f"[{coin}] MTF: {' | '.join(parts)} → combined={combined} "
        f"adj={score_adj:+.0f} long={'✅' if allow_long else '🚫'} "
        f"short={'✅' if allow_short else '🚫'}"
    )

    return MTFAnalysis(
        coin           = coin,
        bias_1h        = b1h,
        bias_4h        = b4h,
        bias_12h       = b12h,
        combined_bias  = combined,
        score_adjustment = score_adj,
        allow_long     = allow_long,
        allow_short    = allow_short,
        reason         = " | ".join(parts),
        valid          = True,
    )
