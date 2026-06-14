"""
indicators/sentiment.py
Fetches the Crypto Fear & Greed Index from Alternative.me (free, no key needed).

The raw Fear & Greed read is useful, but it is not enough on its own. In a
broad crypto breakdown, "fear" is often confirmation that majors are breaking,
not a clean dip-buy. The live signal below blends F&G with BTC/ETH/SOL market
structure so the agent can tell panic-with-reclaim apart from risk-off trend.
"""

import time
import requests
from logger import get_logger

log = get_logger("sentiment")

FNG_URL = "https://api.alternative.me/fng/?limit=1"
CACHE_TTL = 3600   # Re-fetch at most once per hour (index updates daily)
STRUCTURE_CACHE_TTL = 900

_cache = {"ts": 0.0, "score": 50, "label": "Neutral"}
_structure_cache = {"ts": 0.0, "snapshot": {}}
MAJOR_CRYPTO_ASSETS = ("BTC", "ETH", "SOL")


def get_fear_greed_score() -> dict:
    """
    Returns a dict with:
        score (int 0-100): raw F&G index value
        label (str)      : 'Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'
        signal_score (float 0-100): converted to our scoring system
              low F&G (fear) → high signal_score (bullish bias)
              high F&G (greed) → low signal_score (bearish bias)
    """
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL:
        log.debug(f"Sentiment cache hit: {_cache['label']} ({_cache['score']})")
        return _build_result(_cache["score"], _cache["label"])

    try:
        resp = requests.get(FNG_URL, timeout=8)
        resp.raise_for_status()
        data  = resp.json()["data"][0]
        score = int(data["value"])
        label = data["value_classification"]
        _cache.update({"ts": now, "score": score, "label": label})
        log.info(f"Fear & Greed Index: {label} ({score}/100)")
        return _build_result(score, label)
    except Exception as e:
        log.warning(f"Could not fetch F&G index ({e}). Using last known value.")
        return _build_result(_cache["score"], _cache["label"])


def _build_result(fng_score: int, label: str) -> dict:
    """
    Convert raw Fear & Greed score (0=fear, 100=greed)
    into our signal score (0=bearish, 100=bullish), then blend it with
    major-crypto structure. This keeps the agent from blindly buying fear
    while BTC/ETH/SOL are breaking down together.
    """
    contrarian_score = 100.0 - fng_score

    # Boost extremes: extreme fear / greed are stronger signals
    if fng_score <= 20:          # Extreme Fear → very bullish
        contrarian_score = min(90, contrarian_score + 10)
    elif fng_score >= 80:        # Extreme Greed → very bearish
        contrarian_score = max(10, contrarian_score - 10)

    structure = _major_structure_snapshot()
    structural_score = float(structure.get("structural_score", 50.0) or 50.0)
    market_mode = str(structure.get("market_mode", "UNKNOWN") or "UNKNOWN").upper()
    directional_bias = str(structure.get("directional_bias", "NEUTRAL") or "NEUTRAL").upper()

    if market_mode == "DRAWDOWN":
        signal_score = (contrarian_score * 0.25) + (structural_score * 0.75)
        if directional_bias == "BEARISH":
            signal_score = min(signal_score, 42.0)
    elif market_mode == "RISK_OFF":
        signal_score = (contrarian_score * 0.35) + (structural_score * 0.65)
        if directional_bias == "BEARISH":
            signal_score = min(signal_score, 48.0)
    elif market_mode == "RISK_ON":
        signal_score = (contrarian_score * 0.45) + (structural_score * 0.55)
    else:
        signal_score = (contrarian_score * 0.65) + (structural_score * 0.35)
    signal_score = max(0.0, min(100.0, signal_score))

    result = {
        "raw_score":     fng_score,
        "label":         label,
        "signal_score":  signal_score,
        "fng_contrarian_score": round(contrarian_score, 2),
        "structural_score": round(structural_score, 2),
        "market_mode": market_mode,
        "directional_bias": directional_bias,
        "risk_off": bool(structure.get("risk_off", False)),
        "major_breakdown_count": int(structure.get("major_breakdown_count", 0) or 0),
        "major_reclaim_count": int(structure.get("major_reclaim_count", 0) or 0),
        "majors": structure.get("majors", {}),
        "structure_summary": structure.get("summary", ""),
        "is_extreme":    fng_score <= 20 or fng_score >= 80,
    }
    result["summary"] = sentiment_summary(result)
    return result


def _pct_change(old: float, new: float) -> float:
    try:
        old = float(old or 0.0)
        new = float(new or 0.0)
        if old <= 0:
            return 0.0
        return (new - old) / old * 100.0
    except Exception:
        return 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _asset_structure(coin: str) -> dict:
    try:
        from data.market_data import completed_candle_frame, fetch_candles

        df = fetch_candles(coin, "1h", 190)
        frame = completed_candle_frame(df, min_rows=48)
    except Exception as exc:
        log.debug(f"[{coin}] Major-structure sentiment skipped: {exc}")
        frame = None
    if frame is None or getattr(frame, "empty", True) or len(frame) < 30:
        return {
            "coin": coin,
            "valid": False,
            "score": 50.0,
            "summary": "not enough candle history",
        }

    closes = frame["close"].astype(float)
    close = float(closes.iloc[-1])
    ret_24h = _pct_change(float(closes.iloc[-25]), close) if len(closes) >= 25 else 0.0
    ret_72h = _pct_change(float(closes.iloc[-73]), close) if len(closes) >= 73 else ret_24h
    ret_7d = _pct_change(float(closes.iloc[-169]), close) if len(closes) >= 169 else ret_72h
    sma20 = float(closes.tail(20).mean())
    sma72 = float(closes.tail(min(72, len(closes))).mean())
    prior_window = closes.iloc[max(0, len(closes) - 73): max(1, len(closes) - 1)]
    prior_low = float(prior_window.min()) if len(prior_window) else close
    prior_high = float(prior_window.max()) if len(prior_window) else close
    breaking = bool(
        (close < prior_low * 0.995 and ret_24h < -1.0)
        or (ret_24h <= -3.5 and close < sma20 and ret_72h < -4.0)
        or (ret_7d <= -8.0 and close < sma72)
    )
    reclaiming = bool(
        (close > prior_high * 1.005 and ret_24h > 1.0)
        or (ret_24h >= 3.5 and close > sma20 and ret_72h > 4.0)
        or (ret_7d >= 8.0 and close > sma72)
    )

    score = 50.0
    score += _clamp(ret_24h * 2.0, -18.0, 18.0)
    score += _clamp(ret_72h * 0.9, -14.0, 14.0)
    score += _clamp(ret_7d * 0.45, -12.0, 12.0)
    score += 8.0 if close > sma20 else -8.0
    score += 6.0 if close > sma72 else -6.0
    if breaking:
        score -= 12.0
    if reclaiming:
        score += 12.0
    score = _clamp(score, 0.0, 100.0)

    if breaking:
        state = "BREAKING"
    elif reclaiming:
        state = "RECLAIMING"
    elif score <= 42:
        state = "WEAK"
    elif score >= 58:
        state = "STRONG"
    else:
        state = "MIXED"
    return {
        "coin": coin,
        "valid": True,
        "score": round(score, 2),
        "state": state,
        "close": round(close, 6),
        "return_24h_pct": round(ret_24h, 3),
        "return_72h_pct": round(ret_72h, 3),
        "return_7d_pct": round(ret_7d, 3),
        "below_sma20": bool(close < sma20),
        "below_sma72": bool(close < sma72),
        "breaking": breaking,
        "reclaiming": reclaiming,
        "summary": f"{coin} {state.lower()} ({ret_24h:+.1f}% 24h, {ret_7d:+.1f}% 7d)",
    }


def _major_structure_snapshot() -> dict:
    now = time.time()
    cached = _structure_cache.get("snapshot") or {}
    if cached and now - float(_structure_cache.get("ts", 0.0) or 0.0) < STRUCTURE_CACHE_TTL:
        return dict(cached)

    majors = {coin: _asset_structure(coin) for coin in MAJOR_CRYPTO_ASSETS}
    valid = [item for item in majors.values() if item.get("valid")]
    if not valid:
        snapshot = {
            "market_mode": "UNKNOWN",
            "directional_bias": "NEUTRAL",
            "risk_off": False,
            "structural_score": 50.0,
            "major_breakdown_count": 0,
            "major_reclaim_count": 0,
            "majors": majors,
            "summary": "Major crypto structure unavailable; treating sentiment as neutral.",
        }
        _structure_cache.update({"ts": now, "snapshot": snapshot})
        return dict(snapshot)

    avg_score = sum(float(item.get("score", 50.0) or 50.0) for item in valid) / max(len(valid), 1)
    breakdown_count = sum(1 for item in valid if item.get("breaking") or float(item.get("score", 50.0) or 50.0) <= 38.0)
    reclaim_count = sum(1 for item in valid if item.get("reclaiming") or float(item.get("score", 50.0) or 50.0) >= 62.0)

    if breakdown_count >= 2 and avg_score <= 44.0:
        mode = "DRAWDOWN"
        bias = "BEARISH"
        risk_off = True
    elif breakdown_count >= 2 or avg_score <= 42.0:
        mode = "RISK_OFF"
        bias = "BEARISH"
        risk_off = True
    elif reclaim_count >= 2 and avg_score >= 56.0:
        mode = "RISK_ON"
        bias = "BULLISH"
        risk_off = False
    else:
        mode = "BALANCED"
        bias = "NEUTRAL"
        risk_off = False

    snapshot = {
        "market_mode": mode,
        "directional_bias": bias,
        "risk_off": risk_off,
        "structural_score": round(avg_score, 2),
        "major_breakdown_count": breakdown_count,
        "major_reclaim_count": reclaim_count,
        "majors": majors,
        "summary": (
            f"Crypto majors {mode.lower()}: {breakdown_count} breaking, "
            f"{reclaim_count} reclaiming, structure {avg_score:.0f}/100."
        ),
    }
    _structure_cache.update({"ts": now, "snapshot": snapshot})
    return dict(snapshot)


def sentiment_summary(result: dict) -> str:
    s = result["signal_score"]
    risk_off = bool(result.get("risk_off", False))
    directional_bias = str(result.get("directional_bias", "NEUTRAL") or "NEUTRAL").upper()
    if risk_off and directional_bias == "BEARISH" and s <= 48:
        bias = "BEARISH RISK-OFF"
    elif s >= 70:
        bias = "BULLISH"
    elif s <= 30:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"
    mode = str(result.get("market_mode", "UNKNOWN") or "UNKNOWN")
    structure = result.get("structure_summary") or ""
    suffix = f" | {structure}" if structure else ""
    return (f"Sentiment: {result['label']} (F&G={result['raw_score']}) + "
            f"crypto mode {mode} → {bias} signal ({s:.0f}/100){suffix}")
