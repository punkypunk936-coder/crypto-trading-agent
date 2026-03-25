"""
indicators/regimes.py
Six market regime detectors — each returns a signal_score (0-100).

  1. Momentum          — it moved fast & hard, keep riding it
  2. Trend             — it's been going one way, keep going that way
  3. Mean Reversion    — it moved too far, it'll come back
  4. Volatility Expansion — it's been silent, it's about to explode
  5. Absorption        — someone tried to move it and it didn't, reversal coming
  6. Catalyst Absorption — big volume, tiny move = smart money absorbing the headline

These are intentionally at ODDS with each other. The strategy layer decides
which regime is dominant and weights accordingly. All return 0-100:
  > 60 = bullish bias
  < 40 = bearish bias
  40-60 = neutral / no edge
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from logger import get_logger

log = get_logger("regimes")


@dataclass
class RegimeSignals:
    coin: str

    # Individual regime scores (0-100)
    momentum_score:    float = 50.0
    trend_score:       float = 50.0
    mean_rev_score:    float = 50.0
    volatility_score:  float = 50.0
    absorption_score:  float = 50.0
    catalyst_score:    float = 50.0

    # Dominant regime detected
    dominant_regime:   str   = "MIXED"   # MOMENTUM, TREND, MEAN_REV, BREAKOUT, ABSORPTION, MIXED

    # Descriptions for logging
    momentum_desc:    str = ""
    trend_desc:       str = ""
    mean_rev_desc:    str = ""
    volatility_desc:  str = ""
    absorption_desc:  str = ""
    catalyst_desc:    str = ""

    valid: bool = True


# ─────────────────────────────────────────────────────────────
# 1. MOMENTUM
# "It moved fast and hard — it's going to keep moving"
# ─────────────────────────────────────────────────────────────

def _momentum(df: pd.DataFrame, fast: int = 5, slow: int = 20) -> tuple:
    closes = df["close"]
    n = len(closes)
    if n < slow + 2:
        return 50.0, "insufficient data"

    # Rate of change
    roc_fast = (float(closes.iloc[-1]) - float(closes.iloc[-fast]))  / float(closes.iloc[-fast])
    roc_slow = (float(closes.iloc[-1]) - float(closes.iloc[-slow]))  / float(closes.iloc[-slow])

    # Acceleration: is momentum speeding up?
    roc_fast_prev = (float(closes.iloc[-2]) - float(closes.iloc[-fast - 1])) / float(closes.iloc[-fast - 1])
    accelerating  = roc_fast > roc_fast_prev

    # Score: strong positive ROC → bullish (score approaches 85)
    #        strong negative ROC → bearish (score approaches 15)
    base = 50.0 + (roc_fast * 500)        # 1% ROC → +5 pts
    base = max(10.0, min(90.0, base))

    if accelerating and roc_fast > 0:
        base = min(base + 8, 90)
        desc = f"Bullish momentum accelerating (ROC {roc_fast*100:+.2f}%)"
    elif accelerating and roc_fast < 0:
        base = max(base - 8, 10)
        desc = f"Bearish momentum accelerating (ROC {roc_fast*100:+.2f}%)"
    else:
        desc = f"Momentum {roc_fast*100:+.2f}% (decelerating)"

    return round(base, 2), desc


# ─────────────────────────────────────────────────────────────
# 2. TREND STRENGTH (ADX)
# "It's been going one way — keep going that way"
# ─────────────────────────────────────────────────────────────

def _adx(highs: pd.Series, lows: pd.Series, closes: pd.Series,
         period: int = 14) -> tuple:
    """Returns (adx, di_plus, di_minus) using Wilder smoothing."""
    h = highs.values
    l = lows.values
    c = closes.values
    n = len(c)
    if n < period + 2:
        return 20.0, 0.0, 0.0

    tr_arr, dmp_arr, dmm_arr = [], [], []
    for i in range(1, n):
        tr  = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        dmp = max(h[i] - h[i-1], 0) if (h[i] - h[i-1]) > (l[i-1] - l[i]) else 0
        dmm = max(l[i-1] - l[i], 0) if (l[i-1] - l[i]) > (h[i] - h[i-1]) else 0
        tr_arr.append(tr); dmp_arr.append(dmp); dmm_arr.append(dmm)

    tr_s  = pd.Series(tr_arr).ewm(span=period, adjust=False).mean()
    dmp_s = pd.Series(dmp_arr).ewm(span=period, adjust=False).mean()
    dmm_s = pd.Series(dmm_arr).ewm(span=period, adjust=False).mean()

    di_p  = 100 * dmp_s / tr_s.replace(0, np.nan)
    di_m  = 100 * dmm_s / tr_s.replace(0, np.nan)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m).replace(0, np.nan)
    adx   = dx.ewm(span=period, adjust=False).mean()

    return (float(adx.iloc[-1]),
            float(di_p.iloc[-1]),
            float(di_m.iloc[-1]))


def _trend(df: pd.DataFrame) -> tuple:
    adx, di_p, di_m = _adx(df["high"], df["low"], df["close"])

    trending = adx > 22
    bull     = di_p > di_m

    if not trending:
        score = 50.0
        desc  = f"Ranging market (ADX={adx:.1f} < 22) — trend signals weak"
    elif bull:
        # Scale: ADX 22→80 maps to score 60→88
        score = min(88, 60 + (adx - 22) * 0.47)
        desc  = f"Uptrend (ADX={adx:.1f} DI+={di_p:.1f} DI-={di_m:.1f})"
    else:
        score = max(12, 40 - (adx - 22) * 0.47)
        desc  = f"Downtrend (ADX={adx:.1f} DI+={di_p:.1f} DI-={di_m:.1f})"

    return round(score, 2), desc


# ─────────────────────────────────────────────────────────────
# 3. MEAN REVERSION (Z-score)
# "It moved a shitload in one direction — it'll come back"
# ─────────────────────────────────────────────────────────────

def _mean_reversion(df: pd.DataFrame, period: int = 20) -> tuple:
    closes = df["close"]
    if len(closes) < period + 2:
        return 50.0, "insufficient data"

    mean   = closes.rolling(period).mean()
    std    = closes.rolling(period).std()
    z      = ((closes - mean) / std).iloc[-1]
    price  = float(closes.iloc[-1])

    # Extreme z-score → high mean reversion probability
    # z > +2.0 = very overbought → short bias
    # z < -2.0 = very oversold   → long bias
    if z <= -2.5:
        score = 88; desc = f"Extremely oversold (z={z:.2f}) — strong mean rev long"
    elif z <= -2.0:
        score = 78; desc = f"Oversold (z={z:.2f}) — mean rev long signal"
    elif z <= -1.5:
        score = 65; desc = f"Mildly oversold (z={z:.2f})"
    elif z >= 2.5:
        score = 12; desc = f"Extremely overbought (z={z:.2f}) — strong mean rev short"
    elif z >= 2.0:
        score = 22; desc = f"Overbought (z={z:.2f}) — mean rev short signal"
    elif z >= 1.5:
        score = 35; desc = f"Mildly overbought (z={z:.2f})"
    else:
        score = 50; desc = f"Within normal range (z={z:.2f})"

    return round(score, 2), desc


# ─────────────────────────────────────────────────────────────
# 4. VOLATILITY EXPANSION (Bollinger Squeeze)
# "It's been going nowhere — it's about to go somewhere"
# ─────────────────────────────────────────────────────────────

def _volatility_expansion(df: pd.DataFrame,
                           bb_period: int = 20,
                           bb_std: float  = 2.0,
                           squeeze_lookback: int = 10) -> tuple:
    closes = df["close"]
    highs  = df["high"]
    lows   = df["low"]
    if len(closes) < bb_period + squeeze_lookback + 2:
        return 50.0, "insufficient data"

    # Bollinger Bands
    mid   = closes.rolling(bb_period).mean()
    std   = closes.rolling(bb_period).std()
    bb_up = mid + bb_std * std
    bb_lo = mid - bb_std * std
    bb_w  = (bb_up - bb_lo) / mid           # band width as % of price

    # Keltner Channel (ATR-based)
    tr = pd.concat([highs - lows,
                    (highs - closes.shift()).abs(),
                    (lows  - closes.shift()).abs()], axis=1).max(axis=1)
    atr    = tr.ewm(span=bb_period, adjust=False).mean()
    kc_up  = mid + 1.5 * atr
    kc_lo  = mid - 1.5 * atr

    # Squeeze = BB inside KC
    squeeze = (bb_up.iloc[-1] < kc_up.iloc[-1]) and (bb_lo.iloc[-1] > kc_lo.iloc[-1])
    squeeze_ended_recently = False
    for i in range(2, squeeze_lookback + 1):
        was_sq = (bb_up.iloc[-i] < kc_up.iloc[-i]) and (bb_lo.iloc[-i] > kc_lo.iloc[-i])
        if was_sq and not squeeze:
            squeeze_ended_recently = True
            break

    # Current direction of the breakout
    price    = float(closes.iloc[-1])
    prev     = float(closes.iloc[-squeeze_lookback])
    momentum = price - prev

    if squeeze:
        # Still in squeeze — neutral but "ready to pop"
        score = 52.0
        desc  = f"BB Squeeze active — volatility coiled, breakout imminent"
    elif squeeze_ended_recently:
        # Just broke out — ride the direction
        if momentum > 0:
            score = 74.0
            desc  = f"Volatility expansion LONG — squeeze just fired upward"
        else:
            score = 26.0
            desc  = f"Volatility expansion SHORT — squeeze just fired downward"
    else:
        # No squeeze, normal volatility
        bw_now  = float(bb_w.iloc[-1])
        bw_prev = float(bb_w.iloc[-squeeze_lookback])
        if bw_now > bw_prev * 1.3:
            score = 55 if momentum > 0 else 45
            desc  = f"Volatility expanding (BW {bw_now:.3f})"
        else:
            score = 50.0
            desc  = f"Normal volatility (BW {bw_now:.3f})"

    return round(score, 2), desc


# ─────────────────────────────────────────────────────────────
# 5. ABSORPTION
# "Someone really tried to move it and it barely budged"
# ─────────────────────────────────────────────────────────────

def _absorption(df: pd.DataFrame, lookback: int = 5) -> tuple:
    if len(df) < lookback + 2:
        return 50.0, "insufficient data"

    recent  = df.iloc[-lookback:]
    score   = 50.0
    desc    = "No absorption pattern detected"

    for i in range(len(recent) - 1, max(len(recent) - 4, 0), -1):
        c = recent.iloc[i]
        body      = abs(float(c["close"]) - float(c["open"]))
        full_range= float(c["high"]) - float(c["low"])
        if full_range == 0:
            continue
        body_ratio = body / full_range       # small body = large wicks

        upper_wick = float(c["high"])  - max(float(c["open"]), float(c["close"]))
        lower_wick = min(float(c["open"]), float(c["close"])) - float(c["low"])

        # Large lower wick (buyers absorbed sellers) → bullish
        if lower_wick > full_range * 0.55 and body_ratio < 0.35:
            score = 76.0
            desc  = f"Bullish absorption — large lower wick ({lower_wick/full_range*100:.0f}% of range)"
            break

        # Large upper wick (sellers absorbed buyers) → bearish
        if upper_wick > full_range * 0.55 and body_ratio < 0.35:
            score = 24.0
            desc  = f"Bearish absorption — large upper wick ({upper_wick/full_range*100:.0f}% of range)"
            break

        # Doji-like: both wicks large = indecision / absorption both ways
        if upper_wick > full_range * 0.3 and lower_wick > full_range * 0.3:
            score = 50.0
            desc  = "Two-sided absorption (doji) — indecision"
            break

    return round(score, 2), desc


# ─────────────────────────────────────────────────────────────
# 6. CATALYST ABSORPTION
# "Big volume, tiny move — smart money eating the crowd"
# ─────────────────────────────────────────────────────────────

def _catalyst_absorption(df: pd.DataFrame,
                          vol_lookback: int = 20,
                          vol_threshold: float = 2.2) -> tuple:
    if len(df) < vol_lookback + 2:
        return 50.0, "insufficient data"

    closes  = df["close"].values
    volumes = df["volume"].values
    opens   = df["open"].values

    vol_avg  = np.mean(volumes[-vol_lookback:-1])
    last_vol = volumes[-1]
    vol_ratio= last_vol / vol_avg if vol_avg > 0 else 1.0

    price_move = abs(closes[-1] - opens[-1]) / opens[-1] if opens[-1] > 0 else 0

    if vol_ratio < vol_threshold:
        return 50.0, f"No volume anomaly (vol ratio {vol_ratio:.1f}x)"

    # Big volume, big move → not absorption, just a normal momentum candle
    if price_move > 0.015:   # > 1.5% body move
        bias  = "BULLISH" if closes[-1] > opens[-1] else "BEARISH"
        score = 68.0 if closes[-1] > opens[-1] else 32.0
        desc  = f"High vol ({vol_ratio:.1f}x) + big move ({price_move*100:.1f}%) — {bias} catalyst"
        return round(score, 2), desc

    # Big volume, SMALL move → absorption
    # Who won? Check which direction the volume was leaning
    if closes[-1] >= opens[-1]:
        # Attempted bearish move but price went up = buyers won
        score = 72.0
        desc  = (f"Catalyst absorption BULLISH — "
                 f"{vol_ratio:.1f}x vol but only {price_move*100:.2f}% move "
                 f"(buyers absorbed sellers)")
    else:
        # Attempted bullish move but price went down = sellers won
        score = 28.0
        desc  = (f"Catalyst absorption BEARISH — "
                 f"{vol_ratio:.1f}x vol but only {price_move*100:.2f}% move "
                 f"(sellers absorbed buyers)")

    return round(score, 2), desc


# ─────────────────────────────────────────────────────────────
# Dominant regime detection
# ─────────────────────────────────────────────────────────────

def _dominant_regime(sig: RegimeSignals) -> str:
    """
    Figure out which regime is most active this candle.
    Used by the strategy to know which signals to weight higher.
    """
    # Volatility expansion overrides everything if squeeze just fired
    if "squeeze just fired" in sig.volatility_desc:
        return "BREAKOUT"

    # Absorption overrides if detected
    if "absorption" in sig.absorption_desc.lower() and "no absorption" not in sig.absorption_desc.lower():
        return "ABSORPTION"
    if "catalyst absorption" in sig.catalyst_desc.lower():
        return "ABSORPTION"

    # Extreme mean reversion overrides trending (when z > 2.5)
    if sig.mean_rev_score >= 85 or sig.mean_rev_score <= 15:
        return "MEAN_REV"

    # Strong trend
    trend_dist = abs(sig.trend_score - 50)
    mom_dist   = abs(sig.momentum_score - 50)

    if trend_dist > 15 and mom_dist > 12:
        return "MOMENTUM"
    if trend_dist > 12:
        return "TREND"

    return "MIXED"


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

def compute_regimes(df: pd.DataFrame, coin: str) -> RegimeSignals:
    sig = RegimeSignals(coin=coin)

    if df is None or len(df) < 30:
        sig.valid = False
        return sig

    sig.momentum_score,   sig.momentum_desc   = _momentum(df)
    sig.trend_score,      sig.trend_desc       = _trend(df)
    sig.mean_rev_score,   sig.mean_rev_desc    = _mean_reversion(df)
    sig.volatility_score, sig.volatility_desc  = _volatility_expansion(df)
    sig.absorption_score, sig.absorption_desc  = _absorption(df)
    sig.catalyst_score,   sig.catalyst_desc    = _catalyst_absorption(df)
    sig.dominant_regime                         = _dominant_regime(sig)

    log.debug(
        f"[{coin}] Regimes: "
        f"Mom={sig.momentum_score:.0f} "
        f"Trend={sig.trend_score:.0f} "
        f"MRev={sig.mean_rev_score:.0f} "
        f"Vol={sig.volatility_score:.0f} "
        f"Abs={sig.absorption_score:.0f} "
        f"Cat={sig.catalyst_score:.0f} "
        f"→ {sig.dominant_regime}"
    )
    return sig
