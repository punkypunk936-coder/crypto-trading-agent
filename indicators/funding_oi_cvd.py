"""
indicators/funding_oi_cvd.py — Funding Rate, Open Interest Delta, and CVD.

Three high-signal indicators specific to perpetuals trading that most retail
algos completely ignore. Together they reveal WHO is driving price and whether
the move has conviction or is just leverage-driven noise.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. FUNDING RATE  (perp-specific, real edge)
   ─────────────────────────────────────────
   • Positive funding → longs pay shorts → market over-leveraged long
     → contrarian signal: lean SHORT or fade longs
   • Negative funding → shorts pay longs → over-leveraged short
     → contrarian signal: lean LONG or fade shorts
   • Extreme funding (>0.1% per 8h) → forced liquidation risk
   • Near-zero funding → balanced positioning, neutral
   Signal output: score 0-100 where 50 = neutral
     >70 = bullish (negative funding, shorts squeezable)
     <30 = bearish (positive funding, longs vulnerable)

2. OPEN INTEREST DELTA  (conviction check)
   ──────────────────────────────────────
   • Rising price + Rising OI  → new longs entering → BULLISH (real demand)
   • Rising price + Falling OI → short squeeze only  → CAUTION (less durable)
   • Falling price + Rising OI → new shorts entering → BEARISH (real selling)
   • Falling price + Falling OI→ long liquidation    → potential BOTTOM
   Signal: confirms or questions the price move's credibility

3. CVD — CUMULATIVE VOLUME DELTA  (order flow truth)
   ────────────────────────────────────────────────
   • Green candle close > open → buying pressure → positive delta
   • Red candle close < open   → selling pressure → negative delta
   • CVD = cumulative sum of (buy_vol - sell_vol)
   • Divergence: price UP + CVD DOWN → distribution → BEARISH
   • Divergence: price DOWN + CVD UP → accumulation → BULLISH
   Signal: divergence from price is one of the strongest leading indicators
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import requests

from logger import get_logger

log = get_logger("funding_oi_cvd")

# ── Hyperliquid API ──────────────────────────────────────────────────────────
HL_INFO_URL = "https://api.hyperliquid.xyz/info"
_CACHE: dict = {}
CACHE_TTL   = 180   # seconds — funding/OI refresh every 3 min


@dataclass
class FundingOISignal:
    valid: bool
    coin: str

    # Funding rate
    funding_rate: float         = 0.0   # current 8h rate, e.g. 0.0001 = 0.01%
    funding_annualised: float   = 0.0   # annualised: rate * 3 * 365
    funding_signal: float       = 50.0  # 0-100, >50 bullish (neg funding)
    funding_label: str          = "NEUTRAL"

    # OI delta
    oi_current: float           = 0.0
    oi_prev: float              = 0.0
    oi_change_pct: float        = 0.0   # % change
    oi_signal: float            = 50.0  # 0-100
    oi_price_divergence: str    = "ALIGNED"  # "ALIGNED" | "DIVERGING"

    # CVD
    cvd_current: float          = 0.0   # cumulative over lookback
    cvd_slope: float            = 0.0   # recent 10-candle slope
    cvd_signal: float           = 50.0  # 0-100
    cvd_divergence: str         = "NONE"  # "BULLISH" | "BEARISH" | "NONE"

    # Composite
    composite_score: float      = 50.0  # 0-100 weighted combination
    summary: str                = ""


def get_funding_oi_cvd(coin: str, df: pd.DataFrame) -> FundingOISignal:
    """
    Main entry point. Fetches funding + OI from Hyperliquid, computes CVD
    from the OHLCV dataframe. Returns a FundingOISignal.

    Falls back gracefully to neutral (score=50) on any error.
    """
    result = FundingOISignal(valid=False, coin=coin)
    try:
        # ── 1. Funding rate + OI from Hyperliquid ──────────────────────────
        _fetch_funding_and_oi(result, coin)

        # ── 2. CVD from candle data ─────────────────────────────────────────
        _compute_cvd(result, df)

        # ── 3. Composite score ──────────────────────────────────────────────
        _composite(result)

        result.valid = True

    except Exception as e:
        log.warning(f"[{coin}] FundingOI failed: {e}")
        result.composite_score = 50.0
        result.valid = False

    return result


# ── Funding + OI ─────────────────────────────────────────────────────────────

def _fetch_funding_and_oi(result: FundingOISignal, coin: str) -> None:
    """Fetch current funding rate and open interest from Hyperliquid."""
    cache_key = f"meta_{coin}"
    now = time.time()

    if cache_key in _CACHE and (now - _CACHE[cache_key]["ts"]) < CACHE_TTL:
        data = _CACHE[cache_key]["data"]
    else:
        try:
            resp = requests.post(HL_INFO_URL, json={"type": "metaAndAssetCtxs"}, timeout=5)
            resp.raise_for_status()
            payload = resp.json()
            # payload = [meta, assetCtxs]
            meta      = payload[0]
            asset_ctxs = payload[1]

            # Find the coin index
            universe = meta.get("universe", [])
            coin_idx = next(
                (i for i, u in enumerate(universe) if u.get("name") == coin),
                None
            )

            if coin_idx is None:
                log.debug(f"[{coin}] Not found in Hyperliquid universe — using neutral")
                _CACHE[cache_key] = {"ts": now, "data": None}
                return

            ctx = asset_ctxs[coin_idx]
            _CACHE[cache_key] = {"ts": now, "data": ctx}
            data = ctx

        except Exception as e:
            log.debug(f"[{coin}] HL meta fetch failed: {e}")
            _CACHE[cache_key] = {"ts": now, "data": None}
            return

    if not data:
        return

    # Funding rate (string like "0.0001235")
    try:
        fr = float(data.get("funding", 0))
        result.funding_rate       = fr
        result.funding_annualised = fr * 3 * 365 * 100  # % per year

        # Signal: negative funding = shorts paying longs = bullish setup
        # Map: -0.10% per 8h (very negative) → 100, +0.10% (very positive) → 0
        # Typical range: -0.05% to +0.05%
        clamp = max(-0.001, min(0.001, fr))  # clamp to ±0.1%
        result.funding_signal = 50.0 - (clamp / 0.001) * 40.0  # 10 to 90

        if fr > 0.0005:         # >0.05% per 8h — very positive, longs over-leveraged
            result.funding_label = "LONGS_CROWDED"
        elif fr > 0.0002:
            result.funding_label = "SLIGHTLY_LONG_BIASED"
        elif fr < -0.0005:      # <-0.05% — very negative, shorts over-leveraged
            result.funding_label = "SHORTS_CROWDED"
        elif fr < -0.0002:
            result.funding_label = "SLIGHTLY_SHORT_BIASED"
        else:
            result.funding_label = "BALANCED"

    except (ValueError, TypeError):
        pass

    # Open Interest
    try:
        oi = float(data.get("openInterest", 0))
        result.oi_current = oi
        # We'll compare with cached previous to get delta
        oi_hist_key = f"oi_hist_{coin}"
        if oi_hist_key in _CACHE:
            result.oi_prev = _CACHE[oi_hist_key].get("oi", oi)
            if result.oi_prev > 0:
                result.oi_change_pct = (oi - result.oi_prev) / result.oi_prev * 100
        _CACHE[oi_hist_key] = {"oi": oi, "ts": now}

        # OI signal
        chg = result.oi_change_pct
        if chg > 3:       # OI rising fast → new positions entering → amplify trend
            result.oi_signal = 65.0
        elif chg > 1:
            result.oi_signal = 57.0
        elif chg < -3:    # OI falling fast → liquidations / unwinding
            result.oi_signal = 40.0
        elif chg < -1:
            result.oi_signal = 45.0
        else:
            result.oi_signal = 50.0

    except (ValueError, TypeError):
        pass


# ── CVD ──────────────────────────────────────────────────────────────────────

def _compute_cvd(result: FundingOISignal, df: pd.DataFrame) -> None:
    """Compute Cumulative Volume Delta from OHLCV candles."""
    if df is None or len(df) < 10:
        return

    try:
        close  = df["close"].values
        open_  = df["open"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(df))

        # Delta per candle: green candle → buy pressure, red → sell pressure
        # Simple approximation: sign(close-open) × volume
        body_pct = (close - open_) / np.where(open_ > 0, open_, 1)
        delta    = body_pct * volume

        # Cumulative delta
        cvd      = np.cumsum(delta)
        result.cvd_current = float(cvd[-1])

        # Slope of last 10 candles (is CVD accelerating up or down?)
        if len(cvd) >= 10:
            recent_cvd = cvd[-10:]
            x = np.arange(len(recent_cvd), dtype=float)
            slope = float(np.polyfit(x, recent_cvd, 1)[0])
            result.cvd_slope = slope

        # Divergence detection: compare last 20 candles
        n = min(20, len(df))
        price_change = close[-1] - close[-n] if n > 1 else 0
        cvd_change   = cvd[-1] - cvd[-n] if n > 1 else 0

        # Normalise direction for comparison
        price_up = price_change > 0
        cvd_up   = cvd_change > 0

        # Divergence: price and CVD moving in OPPOSITE directions
        if price_up and not cvd_up:
            result.cvd_divergence = "BEARISH"    # price rising but sellers dominate
            result.cvd_signal = 30.0
        elif not price_up and cvd_up:
            result.cvd_divergence = "BULLISH"    # price falling but buyers absorbing
            result.cvd_signal = 70.0
        else:
            # Aligned — amplify the direction
            result.cvd_divergence = "NONE"
            if price_up:
                result.cvd_signal = 60.0 if cvd_up else 50.0
            else:
                result.cvd_signal = 40.0 if not cvd_up else 50.0

    except Exception as e:
        log.debug(f"[{result.coin}] CVD compute error: {e}")


# ── Composite score ───────────────────────────────────────────────────────────

def _composite(result: FundingOISignal) -> None:
    """
    Combine funding, OI, and CVD into one composite score (0-100).
    Weights: CVD 40% | Funding 35% | OI 25%
    CVD is weighted most — it's the most immediate order-flow signal.
    """
    weights = {
        "cvd":     0.40,
        "funding": 0.35,
        "oi":      0.25,
    }
    composite = (
        result.cvd_signal     * weights["cvd"] +
        result.funding_signal * weights["funding"] +
        result.oi_signal      * weights["oi"]
    )
    result.composite_score = round(composite, 1)

    # Build summary string
    parts = []
    if result.funding_label != "BALANCED":
        parts.append(f"funding={result.funding_label}({result.funding_rate*100:+.4f}%/8h)")
    if abs(result.oi_change_pct) > 0.5:
        parts.append(f"OI{result.oi_change_pct:+.1f}%")
    if result.cvd_divergence != "NONE":
        parts.append(f"CVD={result.cvd_divergence}_divergence")

    result.summary = " | ".join(parts) if parts else "balanced positioning"

    log.debug(
        f"[{result.coin}] FundingOI: score={composite:.1f} "
        f"funding={result.funding_signal:.0f} oi={result.oi_signal:.0f} "
        f"cvd={result.cvd_signal:.0f} ({result.cvd_divergence})"
    )
