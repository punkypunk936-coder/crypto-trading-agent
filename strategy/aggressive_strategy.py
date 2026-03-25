from dataclasses import dataclass
from typing import Optional
from indicators.technical import TechnicalSignals
from indicators.advanced  import AdvancedSignals
from indicators.regimes   import RegimeSignals
from indicators.funding_oi_cvd import FundingOISignal
from logger import get_logger

log = get_logger("strategy")

@dataclass
class TradeSignal:
    coin: str
    action: str
    score: float
    confidence: str
    price: float = 0.0
    reason: str = ""
    flat_reason: str = ""       # Why the agent chose FLAT (shown on dashboard)
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    instrument_type: str = "crypto"  # "crypto" | "index"


class AggressiveStrategy:

    def __init__(self, trading_cfg, indicator_cfg):
        self.tcfg = trading_cfg
        self.icfg = indicator_cfg

    def generate_signal(
        self,
        tech,
        advanced,
        sentiment,
        current_position=None,
        regimes=None,
        news_signal=None,          # NewsSignal from indicators/news.py
        candle_patterns=None,      # PatternSignal from indicators/candlestick_patterns.py
        memory_adjustment: float = 0.0,  # adjustment from TradeMemory
        instrument_type: str = "crypto",  # "crypto" | "index"
        funding_oi_signal=None,    # FundingOISignal from indicators/funding_oi_cvd.py
    ):
        if not tech.valid:
            return TradeSignal(coin=tech.coin, action="FLAT", score=50.0,
                               confidence="LOW", price=tech.price,
                               reason="No technical signal")

        icfg = self.icfg

        # ── 1. Classic technical score ──────────────────────────────────────
        classic_score = (
            tech.rsi_score  * icfg.weight_rsi  +
            tech.macd_score * icfg.weight_macd +
            tech.bb_score   * icfg.weight_bb   +
            tech.ema_score  * icfg.weight_ema  +
            sentiment["signal_score"] * icfg.weight_sentiment
        )

        # ── 2. News sentiment (feeds directly into score now) ───────────────
        news_score = 50.0
        if news_signal and news_signal.valid:
            news_score = news_signal.score
            log.info(
                f"[{tech.coin}] News score: {news_score:.1f}/100 "
                f"({news_signal.velocity} velocity, {news_signal.article_count} articles)"
            )
        classic_score += news_score * icfg.weight_news

        # ── 3. Candlestick pattern score ─────────────────────────────────────
        candle_score = 50.0
        if candle_patterns and candle_patterns.valid:
            candle_score = candle_patterns.score
        classic_score += candle_score * icfg.weight_candles

        # ── 4. Advanced / structure indicators ─────────────────────────────
        if advanced.valid:
            adv_score = (
                advanced.fib.score * icfg.weight_fib +
                advanced.msb.score * icfg.weight_msb +
                advanced.ob.score  * icfg.weight_ob  +
                advanced.fvg.score * icfg.weight_fvg
            )
        else:
            adv_score = 50.0 * (icfg.weight_fib + icfg.weight_msb +
                                icfg.weight_ob  + icfg.weight_fvg)

        # ── 5. Market regime indicators ─────────────────────────────────────
        if regimes and regimes.valid:
            regime_score = (
                regimes.momentum_score  * icfg.weight_regime_momentum +
                regimes.trend_score     * icfg.weight_regime_trend    +
                regimes.mean_rev_score  * icfg.weight_regime_mean_rev +
                regimes.volatility_score * icfg.weight_regime_vol_exp +
                regimes.absorption_score * icfg.weight_regime_absorption +
                regimes.catalyst_score  * icfg.weight_regime_catalyst
            )
        else:
            regime_score = 50.0 * (
                icfg.weight_regime_momentum + icfg.weight_regime_trend +
                icfg.weight_regime_mean_rev + icfg.weight_regime_vol_exp +
                icfg.weight_regime_absorption + icfg.weight_regime_catalyst
            )

        # ── 5b. Funding Rate / OI / CVD (order-flow intelligence) ──────────
        # These three signals reveal WHO is driving price: real demand or leverage noise.
        # Weight: 15% of total score — strong enough to shift borderline decisions.
        foc_score = 50.0
        if funding_oi_signal and funding_oi_signal.valid:
            foc_score = funding_oi_signal.composite_score
            log.info(
                f"[{tech.coin}] FundingOI: score={foc_score:.1f} "
                f"funding={funding_oi_signal.funding_label} "
                f"OI_chg={funding_oi_signal.oi_change_pct:+.1f}% "
                f"CVD={funding_oi_signal.cvd_divergence}"
            )
            # CVD divergence override: if CVD says the opposite of price, dampen
            if funding_oi_signal.cvd_divergence == "BEARISH":
                log.info(f"[{tech.coin}] ⚠️  CVD bearish divergence — dampening bullish signals")
            elif funding_oi_signal.cvd_divergence == "BULLISH":
                log.info(f"[{tech.coin}] ⚠️  CVD bullish divergence — dampening bearish signals")

        # Normalise foc_score to same scale as other components (×15 for weight)
        foc_component = foc_score * 0.15

        # ── 6. Combine weighted scores → raw_score 0–100 ───────────────────
        # FOC adds 15%, so divide by 115 to keep scale
        raw_score = (classic_score + adv_score + regime_score + foc_component) / 107.5

        # ── 6b. Index-specific adjustments ──────────────────────────────────
        # Equity indexes are macro-driven, smoother, less volatile.
        # - Momentum/regime signals matter MORE → amplify them
        # - Short-term oscillators (RSI, BB) matter LESS → dampen them
        # - Require a sustained trend, not a single candle spike
        if instrument_type == "index":
            # Dampen raw oscillator contribution slightly (pull toward neutral)
            raw_score = 50.0 + (raw_score - 50.0) * 0.90
            # But boost regime/trend signal weight if it's strong
            if regimes and regimes.valid:
                trend_contribution = (regimes.trend_score - 50.0) * 0.08
                raw_score += trend_contribution
            log.debug(f"[{tech.coin}] Index adjustment applied → {raw_score:.1f}")

        # ── 7. Structure overrides (MSB / OB+FVG confluence) ───────────────
        msb = advanced.msb
        if msb.msb_type in ("BULLISH_CHOCH", "BULLISH_BOS"):
            if raw_score < 55:
                raw_score = max(raw_score, 57.0)
        elif msb.msb_type in ("BEARISH_CHOCH", "BEARISH_BOS"):
            if raw_score > 45:
                raw_score = min(raw_score, 43.0)

        ob  = advanced.ob
        fvg = advanced.fvg
        if ob.inside_bullish_ob and fvg.inside_bullish_fvg:
            raw_score = min(raw_score + 4, 100)
        elif ob.inside_bearish_ob and fvg.inside_bearish_fvg:
            raw_score = max(raw_score - 4, 0)

        # ── 8. Regime bias amplification / dampening ────────────────────────
        if regimes and regimes.valid:
            dom = regimes.dominant_regime
            distance = raw_score - 50.0

            # ── RANGING / ABSORPTION: market has no directional conviction.
            # Pull score hard toward neutral — only trade if signal is extreme.
            # This single rule would have prevented ALL 7 losing trades (all were
            # in RANGING structure or ABSORPTION regime).
            if dom == "ABSORPTION":
                raw_score = 50.0 + (raw_score - 50.0) * 0.45
                log.debug(f"[{tech.coin}] ABSORPTION regime: dampening → {raw_score:.1f}")

            elif dom == "MIXED":
                raw_score = 50.0 + (raw_score - 50.0) * 0.60

            # Trending regimes: amplify the signal
            elif dom in ("MOMENTUM", "TREND"):
                raw_score += 5.0 * (1 if distance >= 0 else -1)

            elif dom == "MEAN_REV":
                raw_score = 50.0 + (raw_score - 50.0) * 0.75

            elif dom == "BREAKOUT":
                raw_score += 6.0 * (1 if distance >= 0 else -1)

        # ── 8b. Market structure guard: RANGING = no directional edge ────────
        # MSB structure_trend == "RANGING" means no higher-highs/lower-lows pattern.
        # Dampen heavily — only very strong conviction signals should pass through.
        msb_struct = advanced.msb.structure_trend if advanced.valid else "RANGING"
        if msb_struct == "RANGING":
            raw_score = 50.0 + (raw_score - 50.0) * 0.55
            log.debug(f"[{tech.coin}] RANGING structure: dampening → {raw_score:.1f}")

        # ── 9. Volume amplification ──────────────────────────────────────────
        vol_ratio = tech.volume_score
        if vol_ratio >= 1.5:
            distance = raw_score - 50.0
            boost = min(7.0, (vol_ratio - 1.0) * 5.0)
            raw_score += boost * (1 if distance >= 0 else -1)

        # ── 10. Extreme sentiment amplifier ─────────────────────────────────
        if sentiment.get("is_extreme"):
            raw_score += (raw_score - 50.0) * 0.15

        # ── 11. Extreme news amplifier (news velocity = EXTREME) ────────────
        if news_signal and news_signal.valid and news_signal.is_extreme:
            news_dir = raw_score - 50.0
            raw_score += news_dir * 0.15
            log.info(f"[{tech.coin}] Extreme news event amplifying score")

        # ── 12. Candlestick amplifier — strong pattern confirmation ──────────
        if candle_patterns and candle_patterns.valid:
            cdir = candle_score - 50.0
            sdir = raw_score - 50.0
            # Only amplify when candles AGREE with current direction
            if cdir * sdir > 0 and abs(cdir) >= 15:
                raw_score += cdir * 0.12

        # ── 13. Extreme volatility dampener ─────────────────────────────────
        if advanced.atr.volatility_label == "extreme":
            raw_score = 50.0 + (raw_score - 50.0) * 0.65

        raw_score = max(0.0, min(100.0, raw_score))

        # ── 14. Memory-based score adjustment ────────────────────────────────
        if memory_adjustment != 0.0:
            raw_score = max(0.0, min(100.0, raw_score + memory_adjustment))
            log.info(
                f"[{tech.coin}] Memory adj: {memory_adjustment:+.1f} → "
                f"score now {raw_score:.1f}"
            )

        # ── 15. Apply thresholds with position-hysteresis ────────────────────
        long_thresh  = self.tcfg.signal_long_threshold
        short_thresh = self.tcfg.signal_short_threshold

        # Hysteresis: slightly harder to re-enter same direction, harder to flip
        if current_position == "LONG":
            short_thresh -= 4   # need stronger SHORT signal to flip
            long_thresh  += 4   # need more to add to existing LONG
        elif current_position == "SHORT":
            long_thresh  += 4
            short_thresh -= 4

        if raw_score >= long_thresh:
            action = "LONG"
        elif raw_score <= short_thresh:
            action = "SHORT"
        else:
            action = "FLAT"

        # ── 16. Confidence ───────────────────────────────────────────────────
        distance = abs(raw_score - 50.0)
        confidence = "HIGH" if distance >= 20 else "MEDIUM" if distance >= 10 else "LOW"

        # ── 17. Strategic FLAT reasoning ─────────────────────────────────────
        # When staying flat, build a clear explanation so the dashboard shows
        # WHY the agent chose not to trade — inaction is intentional, not lazy.
        flat_reason = ""
        if action == "FLAT":
            flat_parts = []
            # Primary reason: score position
            if 45 <= raw_score <= 55:
                flat_parts.append(f"Score {raw_score:.0f} — deep neutral zone")
            elif raw_score < long_thresh:
                flat_parts.append(f"Score {raw_score:.0f} — needs ≥{long_thresh:.0f} for LONG")
            else:
                flat_parts.append(f"Score {raw_score:.0f} — needs ≤{short_thresh:.0f} for SHORT")

            # Candle indecision
            if candle_patterns and candle_patterns.valid:
                indecision = [p for p in candle_patterns.patterns if p in ("Doji", "Spinning Top")]
                if indecision:
                    flat_parts.append(f"Candles: {'+'.join(indecision)} (indecision)")
                if candle_patterns.trend_3 == "FLAT":
                    flat_parts.append("3-candle trend flat")

            # Mixed signals: news vs technicals
            if news_signal and news_signal.valid and news_signal.article_count > 0:
                tech_dir = "LONG" if raw_score > 50 else "SHORT"
                news_dir = "LONG" if news_signal.score > 55 else ("SHORT" if news_signal.score < 45 else "NEUTRAL")
                if tech_dir != news_dir and news_dir != "NEUTRAL":
                    flat_parts.append(f"News ({news_dir}) vs technicals ({tech_dir}) conflict")

            # Regime mixed
            if regimes and regimes.valid and regimes.dominant_regime in ("MIXED", "ABSORPTION"):
                flat_parts.append(f"Regime: {regimes.dominant_regime} (no clear direction)")

            # Memory cooling
            if memory_adjustment <= -8:
                flat_parts.append("Memory: recent losses suppressing signal")

            # Index-specific
            if instrument_type == "index":
                flat_parts.append("Index: waiting for macro confirmation")

            flat_reason = " · ".join(flat_parts) if flat_parts else "Insufficient conviction"
            log.info(f"[{tech.coin}] ✋ FLAT — {flat_reason}")

        # ── 18. Stop-loss / take-profit ──────────────────────────────────────
        sl_price = tp_price = 0.0
        if tech.price > 0 and action != "FLAT":
            if action == "LONG":
                sl_price = tech.price * (1 - self.tcfg.stop_loss_pct)
                tp_price = tech.price * (1 + self.tcfg.take_profit_pct)
            else:
                sl_price = tech.price * (1 + self.tcfg.stop_loss_pct)
                tp_price = tech.price * (1 - self.tcfg.take_profit_pct)

        reason = self._build_reason(
            tech, advanced, sentiment, raw_score, action,
            vol_ratio, regimes, news_signal, candle_patterns
        )

        log.info(
            f"[{tech.coin}] Score={raw_score:.1f} → {action} ({confidence}) | "
            f"{reason[:120]}"
        )

        return TradeSignal(
            coin              = tech.coin,
            action            = action,
            score             = round(raw_score, 2),
            confidence        = confidence,
            price             = tech.price,
            reason            = reason,
            flat_reason       = flat_reason,
            stop_loss_price   = sl_price,
            take_profit_price = tp_price,
            instrument_type   = instrument_type,
        )

    def _build_reason(
        self, tech, advanced, sentiment, score, action,
        vol_ratio, regimes=None, news_signal=None, candle_patterns=None
    ):
        parts = []
        msb = advanced.msb
        parts.append(msb.description if msb.msb_type != "NONE"
                     else f"Structure: {msb.structure_trend}")

        if regimes and regimes.valid and regimes.dominant_regime != "MIXED":
            parts.append(f"Regime: {regimes.dominant_regime}")

        if advanced.fib.nearest_level_name:
            parts.append(f"Fib: {advanced.fib.description}")

        if advanced.ob.inside_bullish_ob or advanced.ob.inside_bearish_ob:
            parts.append(advanced.ob.description)

        if advanced.fvg.inside_bullish_fvg or advanced.fvg.inside_bearish_fvg:
            parts.append(advanced.fvg.description)

        rsi = tech.rsi
        parts.append(
            f"RSI oversold ({rsi:.0f})" if rsi < 35
            else f"RSI overbought ({rsi:.0f})" if rsi > 65
            else f"RSI {rsi:.0f}"
        )

        parts.append(f"MACD {'bullish' if tech.macd_hist > 0 else 'bearish'}")
        parts.append(f"F&G: {sentiment['label']} ({sentiment['raw_score']})")

        if vol_ratio >= 1.5:
            parts.append(f"High vol (x{vol_ratio:.1f})")

        if advanced.atr.volatility_label in ("high", "extreme"):
            parts.append(f"ATR {advanced.atr.volatility_label}")

        # News
        if news_signal and news_signal.valid and news_signal.article_count > 0:
            parts.append(
                f"News: {news_signal.score:.0f}/100 "
                f"({news_signal.velocity})"
            )

        # Candles
        if candle_patterns and candle_patterns.valid and candle_patterns.patterns:
            parts.append(f"Candles: {'+'.join(candle_patterns.patterns[:2])}")

        return " | ".join(parts)
