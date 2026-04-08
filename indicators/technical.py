"""
indicators/technical.py
Computes RSI, MACD, Bollinger Bands, EMA crossover, and Volume signals
from a pandas DataFrame of OHLCV candles.

All maths done with numpy/pandas — no heavy TA-lib dependency.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from logger import get_logger

log = get_logger("technical")


@dataclass
class TechnicalSignals:
    """Holds every indicator reading for a single coin at a point in time."""
    coin: str

    # Raw indicator values
    rsi: float               = 50.0
    macd: float              = 0.0
    macd_signal: float       = 0.0
    macd_hist: float         = 0.0
    bb_upper: float          = 0.0
    bb_lower: float          = 0.0
    bb_mid: float            = 0.0
    bb_pct: float            = 0.5    # 0 = at lower band, 1 = at upper band
    ema_fast: float          = 0.0
    ema_slow: float          = 0.0
    price: float             = 0.0
    volume: float            = 0.0
    volume_ma: float         = 0.0

    # Derived signal scores (0–100 each; 50 = neutral)
    rsi_score: float         = 50.0
    macd_score: float        = 50.0
    bb_score: float          = 50.0
    ema_score: float         = 50.0
    volume_score: float      = 50.0   # volume boost applied inside strategy

    # True if data was sufficient to compute signals
    valid: bool              = True
    reason: str              = ""


# ── Core calculation helpers ──────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def _macd(closes: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = _ema(closes, fast)
    ema_slow   = _ema(closes, slow)
    macd_line  = ema_fast - ema_slow
    signal_line= _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(closes: pd.Series, period=20, std_mult=2.0):
    mid   = closes.rolling(period).mean()
    std   = closes.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


# ── Score converters ─────────────────────────────────────

def _rsi_to_score(rsi_val: float,
                  buy_thresh: float = 42.0,
                  sell_thresh: float = 58.0) -> float:
    """
    RSI → 0-100 score:
      RSI < buy_thresh  → score > 50 (bullish)
      RSI > sell_thresh → score < 50 (bearish)
    """
    if rsi_val <= buy_thresh:
        # Linearly scale: RSI=0 → score=100, RSI=buy_thresh → score=70
        return 70 + (buy_thresh - rsi_val) / buy_thresh * 30
    elif rsi_val >= sell_thresh:
        # RSI=sell_thresh → score=30, RSI=100 → score=0
        return 30 - (rsi_val - sell_thresh) / (100 - sell_thresh) * 30
    else:
        # Neutral zone: linear interpolation from 70→30
        span  = sell_thresh - buy_thresh
        ratio = (rsi_val - buy_thresh) / span
        return 70 - ratio * 40


def _macd_to_score(hist: float, prev_hist: float) -> float:
    """
    MACD histogram → 0-100 score.
    Positive & rising histogram → bullish (score > 50).
    Negative & falling → bearish (score < 50).
    """
    if hist > 0 and hist >= prev_hist:
        return min(85, 55 + abs(hist) * 5)
    elif hist > 0 and hist < prev_hist:
        return 55.0
    elif hist < 0 and hist <= prev_hist:
        return max(15, 45 - abs(hist) * 5)
    else:
        return 45.0


def _bb_to_score(bb_pct: float) -> float:
    """
    Bollinger Band %B → 0-100 score.
    Below lower band (bb_pct < 0)  → buy signal → score approaches 90
    Above upper band (bb_pct > 1)  → sell signal → score approaches 10
    """
    if bb_pct < 0:
        return min(90, 65 + abs(bb_pct) * 40)
    elif bb_pct > 1:
        return max(10, 35 - (bb_pct - 1) * 40)
    else:
        return 65 - bb_pct * 30   # 65 at lower band, 35 at upper


def _ema_to_score(ema_fast: float, ema_slow: float) -> float:
    """
    EMA crossover → 0-100 score.
    Fast > Slow → bullish. Spread magnitude boosts confidence.
    """
    if ema_slow == 0:
        return 50.0
    spread = (ema_fast - ema_slow) / ema_slow * 100  # % spread
    if spread > 0:
        return min(85, 55 + spread * 10)
    else:
        return max(15, 45 + spread * 10)


def _volume_score(volume: float, volume_ma: float) -> float:
    """High volume → amplifies other signals (returned as a multiplier context)."""
    if volume_ma == 0:
        return 1.0
    ratio = volume / volume_ma
    # Return a ratio used as a confidence multiplier elsewhere
    return min(ratio, 2.5)


# ── Main entry point ─────────────────────────────────────

def compute_signals(
    df: pd.DataFrame,
    coin: str,
    cfg,          # IndicatorConfig
    trading_cfg,  # TradingConfig
) -> TechnicalSignals:
    """
    Given an OHLCV DataFrame, compute all technical indicators
    and return a populated TechnicalSignals object.
    """
    signals = TechnicalSignals(coin=coin)

    MIN_ROWS = max(cfg.macd_slow + cfg.macd_signal, cfg.bb_period) + 5
    if df is None or len(df) < MIN_ROWS:
        signals.valid  = False
        signals.reason = f"Insufficient data ({0 if df is None else len(df)} rows, need {MIN_ROWS})"
        log.warning(f"[{coin}] {signals.reason}")
        return signals

    closes  = df["close"]
    volumes = df["volume"]

    # ── RSI ──────────────────────────────────────────────
    rsi_series = _rsi(closes, cfg.rsi_period)
    rsi_val    = float(rsi_series.iloc[-1])
    signals.rsi       = rsi_val
    signals.rsi_score = _rsi_to_score(
        rsi_val,
        trading_cfg.rsi_long_threshold,
        trading_cfg.rsi_short_threshold,
    )

    # ── MACD ─────────────────────────────────────────────
    macd_line, signal_line, histogram = _macd(
        closes, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal
    )
    signals.macd        = float(macd_line.iloc[-1])
    signals.macd_signal = float(signal_line.iloc[-1])
    signals.macd_hist   = float(histogram.iloc[-1])
    prev_hist           = float(histogram.iloc[-2]) if len(histogram) > 1 else 0.0
    signals.macd_score  = _macd_to_score(signals.macd_hist, prev_hist)

    # ── Bollinger Bands ───────────────────────────────────
    bb_upper, bb_mid, bb_lower = _bollinger(closes, cfg.bb_period, cfg.bb_std)
    price = float(closes.iloc[-1])
    band_range = float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1])
    bb_pct = ((price - float(bb_lower.iloc[-1])) / band_range
              if band_range > 0 else 0.5)
    signals.bb_upper  = float(bb_upper.iloc[-1])
    signals.bb_lower  = float(bb_lower.iloc[-1])
    signals.bb_mid    = float(bb_mid.iloc[-1])
    signals.bb_pct    = bb_pct
    signals.bb_score  = _bb_to_score(bb_pct)

    # ── EMA Crossover ─────────────────────────────────────
    ema_fast_s = _ema(closes, cfg.ema_fast)
    ema_slow_s = _ema(closes, cfg.ema_slow)
    signals.ema_fast  = float(ema_fast_s.iloc[-1])
    signals.ema_slow  = float(ema_slow_s.iloc[-1])
    signals.ema_score = _ema_to_score(signals.ema_fast, signals.ema_slow)

    # ── Volume ───────────────────────────────────────────
    vol_ma = volumes.rolling(cfg.volume_ma_period).mean()
    signals.volume    = float(volumes.iloc[-1])
    signals.volume_ma = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) else float(volumes.mean())
    signals.volume_score = _volume_score(signals.volume, signals.volume_ma)

    # ── Price ────────────────────────────────────────────
    signals.price = price

    log.debug(
        f"[{coin}] RSI={rsi_val:.1f}({signals.rsi_score:.0f}) "
        f"MACD_hist={signals.macd_hist:.4f}({signals.macd_score:.0f}) "
        f"BB%={bb_pct:.2f}({signals.bb_score:.0f}) "
        f"EMA({signals.ema_score:.0f}) Vol×{signals.volume_score:.2f}"
    )

    return signals
