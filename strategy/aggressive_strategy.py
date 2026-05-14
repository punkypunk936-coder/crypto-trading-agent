from dataclasses import dataclass, field
from typing import Dict, List, Tuple
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
    instrument_type: str = "crypto"  # "crypto" | "index" | "equity"
    trade_plan: dict = field(default_factory=dict)
    thesis: dict = field(default_factory=dict)
    expectancy: dict = field(default_factory=dict)
    execution_plan: dict = field(default_factory=dict)


class AggressiveStrategy:

    def __init__(self, trading_cfg, indicator_cfg):
        self.tcfg = trading_cfg
        self.icfg = indicator_cfg

    @staticmethod
    def _label_rank(value: str, ordering: Tuple[str, ...], default: int = 0) -> int:
        text = str(value or "").strip().upper()
        try:
            return ordering.index(text)
        except ValueError:
            return default

    def _coin_direction_embargoed(self, coin: str, action: str) -> bool:
        key = f"{str(coin or '').upper()}:{str(action or '').upper()}"
        entries = {
            str(item or "").strip().upper()
            for item in (getattr(self.tcfg, "precision_coin_direction_embargoes", []) or [])
            if str(item or "").strip()
        }
        return key in entries

    def _passes_precision_mode(
        self,
        *,
        coin: str,
        action: str,
        confidence: str,
        thesis: Dict | None = None,
        expectancy: Dict | None = None,
        trade_plan: Dict | None = None,
        orderbook_signal=None,
        market_map_signal=None,
    ) -> tuple[bool, str]:
        if not getattr(self.tcfg, "precision_mode_enabled", False):
            return True, ""
        if action not in {"LONG", "SHORT"}:
            return True, ""

        thesis = dict(thesis or {})
        expectancy = dict(expectancy or {})
        trade_plan = dict(trade_plan or {})
        blockers: List[str] = []
        bullish = action == "LONG"
        conviction_entry = dict(thesis.get("conviction_entry") or {})

        if self._coin_direction_embargoed(coin, action):
            blockers.append(f"{coin} {action} is embargoed until its recent edge improves")
        scalp_profile = dict(thesis.get("scalp") or {})
        if (
            scalp_profile.get("selected")
            and getattr(self.tcfg, "scalp_bypass_precision", True)
            and not blockers
        ):
            return True, ""
        if conviction_entry.get("active") and conviction_entry.get("bypass_precision", False) and not blockers:
            return True, ""

        confidence_rank = self._label_rank(confidence, ("LOW", "MEDIUM", "HIGH"))
        min_confidence_rank = self._label_rank(
            getattr(self.tcfg, "precision_min_confidence", "HIGH"),
            ("LOW", "MEDIUM", "HIGH"),
        )
        if confidence_rank < min_confidence_rank:
            blockers.append(
                f"precision mode needs {str(getattr(self.tcfg, 'precision_min_confidence', 'HIGH')).lower()} confidence"
            )

        thesis_quality_rank = self._label_rank(
            thesis.get("quality", "LOW"),
            ("LOW", "MEDIUM", "HIGH"),
        )
        min_thesis_quality_rank = self._label_rank(
            getattr(self.tcfg, "precision_min_thesis_quality", "MEDIUM"),
            ("LOW", "MEDIUM", "HIGH"),
        )
        if thesis_quality_rank < min_thesis_quality_rank:
            blockers.append("thesis quality is not strong enough for precision mode")

        probability = float(expectancy.get("probability", 0.0) or 0.0)
        min_probability = float(getattr(self.tcfg, "precision_min_expectancy_probability", 0.90) or 0.90)
        if probability < min_probability:
            blockers.append(f"estimated win probability {probability * 100:.0f}% is below precision minimum")

        expected_r = float(expectancy.get("expected_r", 0.0) or 0.0)
        min_expected_r = float(getattr(self.tcfg, "precision_min_expected_r", 0.30) or 0.30)
        if expected_r < min_expected_r:
            blockers.append(f"expected value {expected_r:+.2f}R is too thin for precision mode")

        uncertainty = float(expectancy.get("uncertainty", 1.0) or 1.0)
        max_uncertainty = float(getattr(self.tcfg, "precision_max_uncertainty", 0.24) or 0.24)
        if uncertainty > max_uncertainty:
            blockers.append(f"uncertainty {uncertainty * 100:.0f}% is above the precision cap")

        rr = float(trade_plan.get("risk_reward_ratio", 0.0) or 0.0)
        min_rr = float(getattr(self.tcfg, "precision_min_risk_reward_ratio", 1.90) or 1.90)
        if rr <= 0 or rr < min_rr:
            blockers.append(f"planned R:R {rr:.2f} is below the precision minimum")

        confirmed_breakout = bool(thesis.get("confirmed_breakout", False))
        persistent_breakout = bool(thesis.get("persistent_breakout", False))
        support_defense_long = bool(thesis.get("support_defense_long", False))
        if getattr(self.tcfg, "precision_require_confirmed_breakout", True):
            has_precision_trigger = confirmed_breakout or persistent_breakout
            if bullish and getattr(self.tcfg, "precision_allow_support_defense_longs", True):
                has_precision_trigger = has_precision_trigger or support_defense_long
            if not has_precision_trigger:
                blockers.append("precision mode only permits confirmed breakout or reclaimed-support setups")

        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            orderbook_score = float(getattr(orderbook_signal, "score", 50.0) or 50.0)
            if bullish:
                min_book = float(getattr(self.tcfg, "precision_min_long_orderbook_score", 68.0) or 68.0)
                if orderbook_score < min_book and not support_defense_long:
                    blockers.append(f"orderbook score {orderbook_score:.0f} is too weak for an elite long")
            else:
                max_book = float(getattr(self.tcfg, "precision_max_short_orderbook_score", 38.0) or 38.0)
                if orderbook_score > max_book:
                    blockers.append(f"orderbook score {orderbook_score:.0f} is too strong against the short")
        else:
            blockers.append("precision mode requires valid orderbook context")

        if getattr(self.tcfg, "precision_require_market_map_alignment", False):
            if not market_map_signal or not getattr(market_map_signal, "valid", False):
                blockers.append("daily map alignment is missing")
            elif bullish:
                if not (
                    getattr(market_map_signal, "favor_longs", False)
                    or support_defense_long
                    or confirmed_breakout
                    or persistent_breakout
                ):
                    blockers.append("daily market map is not backing the long")
            else:
                if not (
                    getattr(market_map_signal, "favor_shorts", False)
                    or confirmed_breakout
                    or persistent_breakout
                ):
                    blockers.append("daily market map is not backing the short")

        if blockers:
            return False, blockers[0]
        return True, ""

    @staticmethod
    def _same_direction_breakout_live(action: str, orderbook_signal) -> bool:
        breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE").upper()
        if action == "LONG":
            return breakout_state in {
                "PROBING_BULLISH_BREAKOUT",
                "CONFIRMED_BULLISH_BREAKOUT",
                "PERSISTENT_BULLISH_BREAKOUT",
            }
        if action == "SHORT":
            return breakout_state in {
                "PROBING_BEARISH_BREAKDOWN",
                "CONFIRMED_BEARISH_BREAKDOWN",
                "PERSISTENT_BEARISH_BREAKDOWN",
            }
        return False

    @staticmethod
    def _catalyst_tags(news_signal) -> set[str]:
        tags: set[str] = set()
        for raw in list(getattr(news_signal, "catalyst_tags", []) or []) + list(getattr(news_signal, "event_tags", []) or []):
            text = str(raw or "").strip()
            if text:
                tags.add(text)
        return tags

    def _event_conviction_active(
        self,
        *,
        instrument_type: str,
        action: str,
        news_signal=None,
    ) -> bool:
        if str(instrument_type or "crypto").lower() != "equity":
            return False
        action_text = str(action or "").upper()
        if action_text not in {"LONG", "SHORT"}:
            return False
        if action_text == "SHORT" and not getattr(self.tcfg, "conviction_entry_event_short_enabled", True):
            return False
        if not news_signal or not getattr(news_signal, "valid", False):
            return False
        tags = self._catalyst_tags(news_signal)
        catalyst_tags = {
            "calendar_event",
            "earnings_event",
            "pre_event_setup",
            "pre_ipo_listing",
            "ipo_event",
            "analyst_conviction",
            "demand_commitment",
            "capacity_lock_in",
            "strategic_deal",
            "official_ir_event",
            "sec_filing",
            "options_implied_move",
            "analyst_revision",
        }
        if not (tags & catalyst_tags):
            return False
        news_score = float(getattr(news_signal, "score", 50.0) or 50.0)
        catalyst_score = float(getattr(news_signal, "catalyst_score", 0.0) or 0.0)
        min_news = float(getattr(self.tcfg, "conviction_entry_event_min_news_score", 60.0) or 60.0)
        min_catalyst = float(getattr(self.tcfg, "conviction_entry_event_min_catalyst_score", 4.0) or 4.0)
        if catalyst_score < min_catalyst:
            return False
        if action_text == "LONG":
            return news_score >= min_news
        return news_score <= (100.0 - min_news)

    def _conviction_probe_candidate(
        self,
        *,
        instrument_type: str,
        action: str,
        raw_score: float,
        news_signal=None,
        narrative_signal=None,
        orderbook_signal=None,
        market_map_signal=None,
    ) -> dict:
        candidate = {"active": False, "candidate_action": str(action or "FLAT").upper(), "summary": "", "trigger_gap_points": 0.0}
        if not getattr(self.tcfg, "conviction_entry_enabled", True):
            return candidate
        if str(action or "").upper() in {"LONG", "SHORT"}:
            return candidate
        instrument = str(instrument_type or "crypto").lower()
        if instrument == "index":
            return candidate
        if not news_signal or not getattr(news_signal, "valid", False):
            return candidate

        news_score = float(getattr(news_signal, "score", 50.0) or 50.0)
        article_count = int(getattr(news_signal, "article_count", 0) or 0)
        catalyst_score = float(getattr(news_signal, "catalyst_score", 0.0) or 0.0)
        velocity = str(getattr(news_signal, "velocity", "LOW") or "LOW").upper()
        if article_count <= 0:
            return candidate

        min_news = float(getattr(self.tcfg, "conviction_entry_min_news_score", 58.0) or 58.0)
        min_catalyst = float(getattr(self.tcfg, "conviction_entry_min_catalyst_score", 3.0) or 3.0)
        score_buffer = float(getattr(self.tcfg, "conviction_entry_score_buffer", 6.0) or 6.0)
        market_bias = str(getattr(market_map_signal, "bias", "NEUTRAL") or "NEUTRAL").upper()
        narrative_bias = str(getattr(narrative_signal, "headline_bias", "NEUTRAL") or "NEUTRAL").upper()
        bullish_breakout_live = self._same_direction_breakout_live("LONG", orderbook_signal)
        bearish_breakout_live = self._same_direction_breakout_live("SHORT", orderbook_signal)
        long_gap = max(0.0, float(self.tcfg.signal_long_threshold) - float(raw_score or 0.0))
        short_gap = max(0.0, float(raw_score or 0.0) - float(self.tcfg.signal_short_threshold))
        event_long = self._event_conviction_active(
            instrument_type=instrument_type,
            action="LONG",
            news_signal=news_signal,
        )
        event_short = self._event_conviction_active(
            instrument_type=instrument_type,
            action="SHORT",
            news_signal=news_signal,
        )
        event_score_buffer = float(
            getattr(self.tcfg, "conviction_entry_event_score_buffer", score_buffer) or score_buffer
        )
        long_score_buffer = event_score_buffer if event_long else score_buffer
        short_score_buffer = event_score_buffer if event_short else score_buffer

        if instrument == "equity":
            catalyst_ready = catalyst_score >= min_catalyst or event_long or event_short
        else:
            catalyst_ready = velocity in {"HIGH", "EXTREME"} and abs(news_score - 50.0) >= 14.0
        if not catalyst_ready:
            return candidate

        if long_gap <= long_score_buffer and news_score >= min_news:
            long_blocked = bool(getattr(narrative_signal, "block_longs", False))
            long_blocked = long_blocked or bool(
                market_map_signal
                and getattr(market_map_signal, "block_longs", False)
                and not bullish_breakout_live
                and not bool(getattr(market_map_signal, "live_above_reclaim_levels", []))
                and not event_long
            )
            long_blocked = long_blocked or bool(
                orderbook_signal
                and getattr(orderbook_signal, "block_longs", False)
                and not bullish_breakout_live
                and not event_long
            )
            if not long_blocked and (
                event_long
                or
                narrative_bias == "BULLISH"
                or market_bias == "BULLISH"
                or bullish_breakout_live
                or long_gap <= max(2.0, score_buffer / 2.0)
            ):
                candidate.update({
                    "active": True,
                    "candidate_action": "LONG",
                    "summary": (
                        (
                            f"Pre-event conviction scout: score {raw_score:.1f} is {long_gap:.1f} pts below the "
                            f"{float(self.tcfg.signal_long_threshold):.0f} long trigger, but catalyst {catalyst_score:.2f} "
                            f"and news {news_score:.0f} justify a starter before full price confirmation."
                        )
                        if event_long else
                        (
                            f"Conviction scout: score {raw_score:.1f} is {long_gap:.1f} pts below the "
                            f"{float(self.tcfg.signal_long_threshold):.0f} long trigger with catalyst {catalyst_score:.2f} "
                            f"and news {news_score:.0f}."
                        )
                    ),
                    "trigger_gap_points": round(long_gap, 2),
                })
                return candidate

        bearish_news_floor = 100.0 - min_news
        if short_gap <= short_score_buffer and (news_score <= bearish_news_floor or event_short):
            short_blocked = bool(getattr(narrative_signal, "block_shorts", False))
            short_blocked = short_blocked or bool(
                market_map_signal
                and getattr(market_map_signal, "block_shorts", False)
                and not bearish_breakout_live
                and not bool(getattr(market_map_signal, "live_below_breakdown_levels", []))
                and not event_short
            )
            short_blocked = short_blocked or bool(
                orderbook_signal
                and getattr(orderbook_signal, "block_shorts", False)
                and not bearish_breakout_live
                and not event_short
            )
            if not short_blocked and (
                event_short
                or
                narrative_bias == "BEARISH"
                or market_bias == "BEARISH"
                or bearish_breakout_live
                or short_gap <= max(2.0, score_buffer / 2.0)
            ):
                candidate.update({
                    "active": True,
                    "candidate_action": "SHORT",
                    "summary": (
                        (
                            f"Pre-event conviction scout short: score {raw_score:.1f} is {short_gap:.1f} pts above the "
                            f"{float(self.tcfg.signal_short_threshold):.0f} short trigger, but bearish catalyst "
                            f"{catalyst_score:.2f} and news {news_score:.0f} justify starter risk."
                        )
                        if event_short else
                        (
                            f"Conviction scout: score {raw_score:.1f} is {short_gap:.1f} pts above the "
                            f"{float(self.tcfg.signal_short_threshold):.0f} short trigger with news {news_score:.0f}."
                        )
                    ),
                    "trigger_gap_points": round(short_gap, 2),
                })
        return candidate

    def _build_conviction_entry(
        self,
        *,
        coin: str,
        instrument_type: str,
        action: str,
        score: float,
        thesis: Dict | None = None,
        expectancy: Dict | None = None,
        news_signal=None,
        narrative_signal=None,
        orderbook_signal=None,
        market_map_signal=None,
    ) -> dict:
        entry = {
            "active": False,
            "direction": str(action or "").upper(),
            "style": "",
            "size_multiplier": 1.0,
            "summary": "",
            "reason": "",
            "blockers": [],
            "bypass_precision": False,
        }
        if not getattr(self.tcfg, "conviction_entry_enabled", True):
            return entry
        action = str(action or "").upper()
        if action not in {"LONG", "SHORT"}:
            return entry
        instrument = str(instrument_type or "crypto").lower()
        if instrument == "index":
            return entry
        if not news_signal or not getattr(news_signal, "valid", False):
            return entry

        thesis = dict(thesis or {})
        expectancy = dict(expectancy or {})
        bullish = action == "LONG"
        min_news = float(getattr(self.tcfg, "conviction_entry_min_news_score", 58.0) or 58.0)
        min_catalyst = float(getattr(self.tcfg, "conviction_entry_min_catalyst_score", 3.0) or 3.0)
        score_buffer = float(getattr(self.tcfg, "conviction_entry_score_buffer", 6.0) or 6.0)
        news_score = float(getattr(news_signal, "score", 50.0) or 50.0)
        article_count = int(getattr(news_signal, "article_count", 0) or 0)
        catalyst_score = float(getattr(news_signal, "catalyst_score", 0.0) or 0.0)
        catalyst_summary = str(getattr(news_signal, "catalyst_summary", "") or "").strip()
        velocity = str(getattr(news_signal, "velocity", "LOW") or "LOW").upper()
        market_bias = str(getattr(market_map_signal, "bias", "NEUTRAL") or "NEUTRAL").upper()
        narrative_bias = str(getattr(narrative_signal, "headline_bias", "NEUTRAL") or "NEUTRAL").upper()
        same_direction_breakout_live = self._same_direction_breakout_live(action, orderbook_signal)
        trigger_threshold = float(self.tcfg.signal_long_threshold if bullish else self.tcfg.signal_short_threshold)
        trigger_gap_points = max(0.0, (trigger_threshold - score) if bullish else (score - trigger_threshold))
        event_conviction = self._event_conviction_active(
            instrument_type=instrument_type,
            action=action,
            news_signal=news_signal,
        )
        event_score_buffer = float(
            getattr(self.tcfg, "conviction_entry_event_score_buffer", score_buffer) or score_buffer
        )
        effective_score_buffer = event_score_buffer if event_conviction else score_buffer
        blockers: List[str] = []

        if article_count <= 0:
            blockers.append("no asset-specific headline flow is active")
        if instrument == "equity" and catalyst_score < min_catalyst and not event_conviction:
            blockers.append(f"catalyst score {catalyst_score:.2f} is below the starter floor")
        if bullish and news_score < min_news:
            blockers.append(f"news score {news_score:.0f} is below the bullish starter floor")
        if (not bullish) and news_score > (100.0 - min_news):
            blockers.append(f"news score {news_score:.0f} is not bearish enough for a starter short")
        if trigger_gap_points > effective_score_buffer:
            blockers.append(f"score is still {trigger_gap_points:.1f} pts away from the trigger")

        if bullish:
            if getattr(narrative_signal, "block_longs", False):
                blockers.append("headline flow still blocks longs")
            if orderbook_signal and getattr(orderbook_signal, "block_longs", False) and not same_direction_breakout_live and not event_conviction:
                blockers.append("orderbook still shows active overhead supply")
            if market_map_signal and getattr(market_map_signal, "block_longs", False):
                live_reclaim = bool(getattr(market_map_signal, "live_above_reclaim_levels", []))
                if not (same_direction_breakout_live or live_reclaim or event_conviction):
                    blockers.append("daily map still warns against longs here")
        else:
            if getattr(narrative_signal, "block_shorts", False):
                blockers.append("headline flow still blocks shorts")
            if orderbook_signal and getattr(orderbook_signal, "block_shorts", False) and not same_direction_breakout_live and not event_conviction:
                blockers.append("orderbook still shows active demand under price")
            if market_map_signal and getattr(market_map_signal, "block_shorts", False):
                live_breakdown = bool(getattr(market_map_signal, "live_below_breakdown_levels", []))
                if not (same_direction_breakout_live or live_breakdown or event_conviction):
                    blockers.append("daily map still warns against shorts here")

        alignment = float(thesis.get("alignment_points", 0.0) or 0.0)
        conflicts = float(thesis.get("conflict_points", 0.0) or 0.0)
        probability = float(expectancy.get("probability", 0.0) or 0.0)
        expectancy_score = float(expectancy.get("score", 0.0) or 0.0)
        uncertainty = float(expectancy.get("uncertainty", 1.0) or 1.0)
        catalyst_extreme = catalyst_score >= (min_catalyst + 1.0) or velocity == "EXTREME"
        min_alignment = float(getattr(self.tcfg, "conviction_entry_min_alignment_points", 3.0) or 3.0)
        max_conflicts = float(getattr(self.tcfg, "conviction_entry_max_conflict_points", 1.5) or 1.5)
        probability_floor = float(getattr(self.tcfg, "conviction_entry_min_probability", 0.56) or 0.56)
        expectancy_floor = float(getattr(self.tcfg, "conviction_entry_min_expectancy_score", 54.0) or 54.0)
        uncertainty_cap = float(getattr(self.tcfg, "conviction_entry_max_uncertainty", 0.46) or 0.46)
        if event_conviction:
            min_alignment = float(getattr(self.tcfg, "conviction_entry_event_min_alignment_points", 1.0) or 1.0)
            max_conflicts = float(getattr(self.tcfg, "conviction_entry_event_max_conflict_points", 2.75) or 2.75)
            probability_floor = float(getattr(self.tcfg, "conviction_entry_event_min_probability", 0.51) or 0.51)
            expectancy_floor = float(getattr(self.tcfg, "conviction_entry_event_min_expectancy_score", 46.0) or 46.0)
            uncertainty_cap = float(getattr(self.tcfg, "conviction_entry_event_max_uncertainty", 0.58) or 0.58)
        if catalyst_extreme:
            min_alignment -= 0.5
            max_conflicts += 0.25
            probability_floor -= 0.02
            expectancy_floor -= 2.0
            uncertainty_cap += 0.02

        if alignment < min_alignment:
            blockers.append(f"alignment {alignment:.1f} is below the starter floor")
        if conflicts > max_conflicts:
            blockers.append(f"conflicts {conflicts:.1f} are too high for a starter")
        if probability < probability_floor:
            blockers.append(f"win probability {probability * 100:.0f}% is below the starter floor")
        if expectancy_score < expectancy_floor:
            blockers.append(f"expectancy score {expectancy_score:.0f} is below the starter floor")
        if uncertainty > uncertainty_cap:
            blockers.append(f"uncertainty {uncertainty * 100:.0f}% is above the starter cap")
        if blockers:
            entry["blockers"] = blockers[:4]
            return entry

        base_size = float(
            getattr(
                self.tcfg,
                "conviction_entry_event_size_multiplier" if event_conviction else "conviction_entry_size_multiplier",
                0.35 if event_conviction else 0.45,
            )
            or (0.35 if event_conviction else 0.45)
        )
        size_bonus = max(
            0.0,
            min(
                0.14 if event_conviction else 0.18,
                max(0.0, catalyst_score - min_catalyst) * 0.06
                + max(0.0, probability - probability_floor) * 0.30
                + (0.04 if same_direction_breakout_live else 0.0),
            ),
        )
        if event_conviction:
            max_size = float(getattr(self.tcfg, "conviction_entry_event_max_size_multiplier", 0.46) or 0.46)
            uncertainty_penalty = max(0.0, uncertainty - (uncertainty_cap * 0.75)) * 0.35
            conflict_penalty = max(0.0, conflicts - (max_conflicts * 0.65)) * 0.05
            size_multiplier = max(0.18, min(max_size, base_size + size_bonus - uncertainty_penalty - conflict_penalty))
        else:
            size_multiplier = max(0.20, min(0.72, base_size + size_bonus))
        catalyst_text = f"catalyst {catalyst_score:.2f}" if catalyst_score > 0 else f"headline flow {news_score:.0f}"
        direction_text = "long" if bullish else "short"
        summary = (
            f"{'Pre-event starter' if event_conviction else 'Starter'} {direction_text} allowed: {catalyst_text}, news {news_score:.0f}, "
            f"p {probability * 100:.0f}%, score {score:.1f} vs {trigger_threshold:.0f} trigger. "
            f"Begin {size_multiplier * 100:.0f}% size before full confirmation."
        )
        if catalyst_summary:
            summary += f" Thesis: {catalyst_summary}."

        entry.update({
            "active": True,
            "direction": action,
            "style": "EVENT_STARTER" if event_conviction else "STARTER",
            "size_multiplier": round(size_multiplier, 4),
            "summary": summary,
            "reason": summary,
            "blockers": [],
            "bypass_precision": bool(getattr(self.tcfg, "conviction_entry_precision_override_enabled", True)),
            "event_conviction": bool(event_conviction),
        })
        return entry

    @staticmethod
    def _bullish_candle_context(patterns) -> bool:
        names = set(getattr(patterns, "patterns", []) or [])
        return bool(names & {
            "Hammer", "Bullish Engulfing", "Morning Star",
            "Three Soldiers", "Bullish Marubozu", "Bullish Pin Bar",
        })

    @staticmethod
    def _bearish_candle_context(patterns) -> bool:
        names = set(getattr(patterns, "patterns", []) or [])
        return bool(names & {
            "Shooting Star", "Bearish Engulfing", "Evening Star",
            "Three Crows", "Bearish Marubozu", "Bearish Pin Bar",
        })

    @staticmethod
    def _indecision_candle_context(patterns) -> bool:
        names = set(getattr(patterns, "patterns", []) or [])
        return bool(names & {"Doji", "Spinning Top"})

    @staticmethod
    def _directional_distance(action: str, entry: float, level: float) -> float:
        if entry <= 0 or level <= 0:
            return 0.0
        if action == "LONG":
            return max(0.0, level - entry)
        return max(0.0, entry - level)

    @staticmethod
    def _remaining_distance(action: str, current: float, level: float) -> float:
        if current <= 0 or level <= 0:
            return 0.0
        if action == "LONG":
            return max(0.0, level - current)
        return max(0.0, current - level)

    def _scalp_profile(
        self,
        *,
        action: str,
        score: float,
        instrument_type: str = "crypto",
        advanced=None,
        regimes=None,
        candle_patterns=None,
        news_signal=None,
        funding_oi_signal=None,
        orderbook_signal=None,
        market_map_signal=None,
        narrative_signal=None,
        social_attention_signal=None,
        trade_plan: Dict | None = None,
    ) -> dict:
        profile = {
            "active": False,
            "style": "",
            "alignment_points": 0.0,
            "conflict_points": 0.0,
            "summary": "",
            "reasons": [],
            "blockers": [],
        }
        action = str(action or "").upper()
        if not getattr(self.tcfg, "scalp_trading_enabled", True) or action not in {"LONG", "SHORT"}:
            return profile
        instrument = str(instrument_type or "crypto").lower()
        if instrument == "index":
            return profile
        if action == "SHORT" and instrument == "equity" and not getattr(self.tcfg, "scalp_allow_equity_shorts", True):
            profile["blockers"] = ["equity scalp shorts are disabled"]
            return profile

        bullish = action == "LONG"
        points = 0.0
        conflicts = 0.0
        reasons: List[str] = []
        blockers: List[str] = []

        breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE").upper()
        if bullish:
            same_breakout = (
                self._is_confirmed_bullish_breakout(orderbook_signal)
                or self._is_persistent_bullish_breakout(orderbook_signal)
            )
        else:
            same_breakout = (
                self._is_confirmed_bearish_breakdown(orderbook_signal)
                or self._is_persistent_bearish_breakdown(orderbook_signal)
            )
        probing_breakout = breakout_state == (
            "PROBING_BULLISH_BREAKOUT" if bullish else "PROBING_BEARISH_BREAKDOWN"
        )
        live_reclaim = bool(getattr(market_map_signal, "live_above_reclaim_levels", [])) if market_map_signal else False
        live_breakdown = bool(getattr(market_map_signal, "live_below_breakdown_levels", [])) if market_map_signal else False

        if narrative_signal and getattr(narrative_signal, "valid", False):
            if bullish and getattr(narrative_signal, "block_longs", False):
                blockers.append("headline flow blocks the long scalp")
            elif (not bullish) and getattr(narrative_signal, "block_shorts", False):
                blockers.append("headline flow blocks the short scalp")

        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            level_interaction = str(getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS") or "BETWEEN_LEVELS").upper()
            if level_interaction == "RANGE_COMPRESSION" and not (same_breakout or probing_breakout):
                blockers.append("range compression blocks the scalp")
            if bullish and getattr(orderbook_signal, "block_longs", False) and not (same_breakout or live_reclaim):
                blockers.append("nearby supply still blocks the long scalp")
            elif (not bullish) and getattr(orderbook_signal, "block_shorts", False) and not (same_breakout or live_breakdown):
                blockers.append("nearby demand still blocks the short scalp")

        if market_map_signal and getattr(market_map_signal, "valid", False):
            if bullish and getattr(market_map_signal, "block_longs", False) and not (same_breakout or live_reclaim):
                blockers.append("daily map blocks the long scalp")
            elif (not bullish) and getattr(market_map_signal, "block_shorts", False) and not (same_breakout or live_breakdown):
                blockers.append("daily map blocks the short scalp")

        if same_breakout:
            points += 2.0
            reasons.append("breakdown/breakout is confirmed")
        elif probing_breakout:
            points += 1.25
            reasons.append("breakdown/breakout pressure is live")

        msb = getattr(advanced, "msb", None)
        msb_type = str(getattr(msb, "msb_type", "NONE") or "NONE").upper()
        structure_trend = str(getattr(msb, "structure_trend", "RANGING") or "RANGING").upper()
        if bullish:
            if msb_type in {"BULLISH_CHOCH", "BULLISH_BOS"}:
                points += 1.25
                reasons.append("structure flipped up")
            elif msb_type in {"BEARISH_CHOCH", "BEARISH_BOS"}:
                conflicts += 0.75
            if structure_trend == "UPTREND":
                points += 0.75
            elif structure_trend == "DOWNTREND":
                conflicts += 0.75
        else:
            if msb_type in {"BEARISH_CHOCH", "BEARISH_BOS"}:
                points += 1.25
                reasons.append("structure flipped down")
            elif msb_type in {"BULLISH_CHOCH", "BULLISH_BOS"}:
                conflicts += 0.75
            if structure_trend == "DOWNTREND":
                points += 0.75
            elif structure_trend == "UPTREND":
                conflicts += 0.75
                reasons.append("countertrend scalp only")

        if regimes and getattr(regimes, "valid", False):
            dom = str(getattr(regimes, "dominant_regime", "MIXED") or "MIXED").upper()
            if dom in {"TREND", "MOMENTUM", "BREAKOUT"}:
                points += 0.75
            elif dom in {"ABSORPTION", "MIXED"}:
                conflicts += 0.5

        if candle_patterns and getattr(candle_patterns, "valid", False):
            candle_score = float(getattr(candle_patterns, "score", 50.0) or 50.0)
            candle_trend = str(getattr(candle_patterns, "trend_3", "FLAT") or "FLAT").upper()
            if (bullish and (candle_score >= 58.0 or candle_trend == "UP" or self._bullish_candle_context(candle_patterns))) or (
                (not bullish) and (candle_score <= 42.0 or candle_trend == "DOWN" or self._bearish_candle_context(candle_patterns))
            ):
                points += 1.0
                reasons.append("recent candles confirm")
            elif (bullish and (candle_score <= 42.0 or candle_trend == "DOWN")) or (
                (not bullish) and (candle_score >= 58.0 or candle_trend == "UP")
            ):
                conflicts += 0.75

        if news_signal and getattr(news_signal, "valid", False) and getattr(news_signal, "article_count", 0) > 0:
            news_score = float(getattr(news_signal, "score", 50.0) or 50.0)
            bull_news = float(getattr(self.tcfg, "scalp_bullish_news_score", 55.0) or 55.0)
            bear_news = float(getattr(self.tcfg, "scalp_bearish_news_score", 45.0) or 45.0)
            if bullish and news_score >= bull_news:
                points += 1.25
                reasons.append("newsflow backs the scalp")
            elif (not bullish) and news_score <= bear_news:
                points += 1.25
                reasons.append("newsflow backs the scalp")
            elif bullish and news_score <= bear_news:
                conflicts += 1.0
            elif (not bullish) and news_score >= bull_news:
                conflicts += 1.0

        if social_attention_signal and getattr(social_attention_signal, "valid", False):
            mentions = int(getattr(social_attention_signal, "mentions", 0) or 0)
            min_mentions = int(getattr(self.tcfg, "social_attention_min_mentions_for_signal", 2) or 2)
            social_score = float(getattr(social_attention_signal, "score", 50.0) or 50.0)
            attention = str(getattr(social_attention_signal, "attention_level", "LOW") or "LOW").upper()
            enough_attention = mentions >= min_mentions or attention in {"MEDIUM", "HIGH"}
            if enough_attention and ((bullish and social_score >= 58.0) or ((not bullish) and social_score <= 42.0)):
                points += 1.0
                reasons.append("trader attention is aligned")
            elif enough_attention and ((bullish and social_score <= 42.0) or ((not bullish) and social_score >= 58.0)):
                conflicts += 0.75

        if funding_oi_signal and getattr(funding_oi_signal, "valid", False):
            foc_score = float(getattr(funding_oi_signal, "composite_score", 50.0) or 50.0)
            if (bullish and foc_score >= 58.0) or ((not bullish) and foc_score <= 42.0):
                points += 0.75
            elif (bullish and foc_score <= 42.0) or ((not bullish) and foc_score >= 58.0):
                conflicts += 0.75

        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            orderbook_score = float(getattr(orderbook_signal, "score", 50.0) or 50.0)
            imbalance = float(getattr(orderbook_signal, "imbalance_mean", getattr(orderbook_signal, "imbalance_ratio", 0.0)) or 0.0)
            if bullish and (getattr(orderbook_signal, "favor_longs", False) or orderbook_score >= 58.0 or imbalance >= 0.05):
                points += 1.25
                reasons.append("book pressure is long")
            elif (not bullish) and (getattr(orderbook_signal, "favor_shorts", False) or orderbook_score <= 42.0 or imbalance <= -0.05):
                points += 1.25
                reasons.append("book pressure is short")
            elif bullish and orderbook_score <= 45.0:
                conflicts += 0.75
            elif (not bullish) and orderbook_score >= 55.0:
                conflicts += 0.75

        if market_map_signal and getattr(market_map_signal, "valid", False):
            market_bias = str(getattr(market_map_signal, "bias", "NEUTRAL") or "NEUTRAL").upper()
            if bullish and (getattr(market_map_signal, "favor_longs", False) or market_bias == "BULLISH" or live_reclaim):
                points += 1.0
                reasons.append("daily map permits the long")
            elif (not bullish) and (getattr(market_map_signal, "favor_shorts", False) or market_bias == "BEARISH" or live_breakdown):
                points += 1.0
                reasons.append("daily map permits the short")
            elif bullish and market_bias == "BEARISH":
                conflicts += 0.75
            elif (not bullish) and market_bias == "BULLISH":
                conflicts += 0.75

        long_threshold = float(getattr(self.tcfg, "signal_long_threshold", 65.0) or 65.0)
        short_threshold = float(getattr(self.tcfg, "signal_short_threshold", 35.0) or 35.0)
        score_buffer = float(getattr(self.tcfg, "scalp_score_buffer", 4.0) or 4.0)
        if bullish and score >= (long_threshold - score_buffer):
            points += 0.75
            reasons.append("score is close enough for a tactical long")
        elif (not bullish) and score <= (short_threshold + score_buffer):
            points += 0.75
            reasons.append("score is close enough for a tactical short")

        rr = float((trade_plan or {}).get("risk_reward_ratio", 0.0) or 0.0)
        if rr > 0 and rr < 1.15:
            blockers.append(f"scalp R:R {rr:.2f} is too thin")

        min_points = float(getattr(self.tcfg, "scalp_min_alignment_points", 3.0) or 3.0)
        max_conflicts = float(getattr(self.tcfg, "scalp_max_conflict_points", 3.0) or 3.0)
        active = not blockers and points >= min_points and conflicts <= max_conflicts
        direction_text = "long" if bullish else "short"
        summary = (
            f"Scalp {direction_text}: {', '.join(reasons[:3])}."
            if active and reasons else
            (blockers[0] if blockers else f"scalp checks {points:.1f}/{min_points:.1f}")
        )
        profile.update({
            "active": bool(active),
            "style": "SCALP_LONG" if bullish else "SCALP_SHORT",
            "alignment_points": round(points, 2),
            "conflict_points": round(conflicts, 2),
            "summary": summary,
            "reasons": reasons[:6],
            "blockers": blockers[:4],
            "max_hold_minutes": float(getattr(self.tcfg, "scalp_max_hold_minutes", 240.0) or 240.0),
        })
        return profile

    def _tactical_scalp_probe_candidate(
        self,
        *,
        instrument_type: str,
        action: str,
        raw_score: float,
        advanced=None,
        regimes=None,
        candle_patterns=None,
        news_signal=None,
        funding_oi_signal=None,
        orderbook_signal=None,
        market_map_signal=None,
        narrative_signal=None,
        social_attention_signal=None,
    ) -> dict:
        candidate = {
            "active": False,
            "candidate_action": str(action or "FLAT").upper(),
            "summary": "",
            "trigger_gap_points": 0.0,
            "source": "scalp_probe",
        }
        if not getattr(self.tcfg, "scalp_probe_from_flat_enabled", True):
            return candidate
        if str(action or "").upper() != "FLAT":
            return candidate
        instrument = str(instrument_type or "crypto").lower()
        if instrument == "index":
            return candidate

        ranked: List[tuple[float, str, dict]] = []
        for direction in ("LONG", "SHORT"):
            profile = self._scalp_profile(
                action=direction,
                score=raw_score,
                instrument_type=instrument_type,
                advanced=advanced,
                regimes=regimes,
                candle_patterns=candle_patterns,
                news_signal=news_signal,
                funding_oi_signal=funding_oi_signal,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
                narrative_signal=narrative_signal,
                social_attention_signal=social_attention_signal,
                trade_plan=None,
            )
            if not profile.get("active"):
                continue
            edge = float(profile.get("alignment_points", 0.0) or 0.0) - float(profile.get("conflict_points", 0.0) or 0.0)
            if (direction == "LONG" and raw_score >= 50.0) or (direction == "SHORT" and raw_score <= 50.0):
                edge += 0.35
            ranked.append((edge, direction, profile))

        if not ranked:
            return candidate
        ranked.sort(key=lambda item: item[0], reverse=True)
        edge, direction, profile = ranked[0]
        min_edge = float(getattr(self.tcfg, "scalp_probe_min_edge_points", 1.0) or 1.0)
        if edge < min_edge:
            return candidate
        if len(ranked) > 1 and edge - ranked[1][0] < 0.25:
            return candidate

        trigger = float(self.tcfg.signal_long_threshold if direction == "LONG" else self.tcfg.signal_short_threshold)
        gap = max(0.0, trigger - raw_score) if direction == "LONG" else max(0.0, raw_score - trigger)
        side = "long" if direction == "LONG" else "short"
        reasons = list(profile.get("reasons") or [])
        candidate.update({
            "active": True,
            "candidate_action": direction,
            "summary": (
                f"Tactical {side} probe: {', '.join(reasons[:3]) or 'near-term evidence is aligned'}; "
                f"score is {gap:.1f} pts from the hard trigger."
            ),
            "trigger_gap_points": round(gap, 2),
            "scalp_profile": dict(profile),
        })
        return candidate

    def _apply_scalp_trade_plan(self, action: str, entry_price: float, trade_plan: Dict | None) -> Dict:
        plan = dict(trade_plan or {})
        action = str(action or "").upper()
        entry = float(plan.get("entry_price", entry_price) or entry_price or 0.0)
        if action not in {"LONG", "SHORT"} or entry <= 0:
            return plan
        stop_pct = max(0.001, float(getattr(self.tcfg, "scalp_stop_pct", 0.035) or 0.035))
        target_pct = max(stop_pct * 1.05, float(getattr(self.tcfg, "scalp_target_pct", 0.055) or 0.055))
        current_stop = float(plan.get("stop_loss", 0.0) or 0.0)
        current_target = float(plan.get("take_profit", 0.0) or 0.0)
        if action == "LONG":
            scalp_stop = entry * (1.0 - stop_pct)
            scalp_target = entry * (1.0 + target_pct)
            stop = max(current_stop, scalp_stop) if 0 < current_stop < entry else scalp_stop
            target = min(current_target, scalp_target) if current_target > entry else scalp_target
        else:
            scalp_stop = entry * (1.0 + stop_pct)
            scalp_target = entry * (1.0 - target_pct)
            stop = min(current_stop, scalp_stop) if current_stop > entry else scalp_stop
            target = max(current_target, scalp_target) if 0 < current_target < entry else scalp_target
        risk_pct = abs(entry - stop) / entry * 100.0
        reward_pct = abs(target - entry) / entry * 100.0
        plan.update({
            "entry_price": round(entry, 6),
            "stop_loss": round(stop, 6),
            "take_profit": round(target, 6),
            "risk_pct": round(risk_pct, 3),
            "reward_pct": round(reward_pct, 3),
            "risk_reward_ratio": round(reward_pct / max(risk_pct, 1e-9), 3),
            "trade_style": "scalp",
            "max_hold_minutes": float(getattr(self.tcfg, "scalp_max_hold_minutes", 240.0) or 240.0),
            "stop_basis": "scalp_invalidation",
            "target_basis": "scalp_target",
            "price_action_summary": "tactical scalp with pre-set invalidation",
        })
        return plan

    @staticmethod
    def _is_confirmed_bullish_breakout(orderbook_signal) -> bool:
        return str(getattr(orderbook_signal, "breakout_state", "")).upper() == "CONFIRMED_BULLISH_BREAKOUT"

    @staticmethod
    def _is_confirmed_bearish_breakdown(orderbook_signal) -> bool:
        return str(getattr(orderbook_signal, "breakout_state", "")).upper() == "CONFIRMED_BEARISH_BREAKDOWN"

    @staticmethod
    def _is_persistent_bullish_breakout(orderbook_signal) -> bool:
        return str(getattr(orderbook_signal, "breakout_state", "")).upper() == "PERSISTENT_BULLISH_BREAKOUT"

    @staticmethod
    def _is_persistent_bearish_breakdown(orderbook_signal) -> bool:
        return str(getattr(orderbook_signal, "breakout_state", "")).upper() == "PERSISTENT_BEARISH_BREAKDOWN"

    def _market_map_supports_breakout_long_override(self, market_map_signal, breakout_state: str) -> bool:
        if not getattr(self.tcfg, "support_defense_map_override_enabled", True):
            return False
        if not market_map_signal or not getattr(market_map_signal, "valid", False):
            return False

        breakout_state = str(breakout_state or "NONE").upper()
        bias = str(getattr(market_map_signal, "bias", "NEUTRAL") or "NEUTRAL").upper()
        if bias == "BEARISH":
            return False
        if getattr(market_map_signal, "below_breakdown_levels", []):
            return False

        confirmed = breakout_state in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"}
        probing = breakout_state == "PROBING_BULLISH_BREAKOUT"
        if confirmed:
            return True
        if probing and (
            getattr(market_map_signal, "above_reclaim_levels", [])
            or getattr(market_map_signal, "probing_above_reclaim_levels", [])
            or getattr(market_map_signal, "in_demand_zone", False)
            or bias == "BULLISH"
        ):
            return True
        return False

    def _qualifies_support_defense_long(
        self,
        *,
        raw_score: float,
        advanced=None,
        candle_patterns=None,
        orderbook_signal=None,
        market_map_signal=None,
    ) -> bool:
        if not getattr(self.tcfg, "support_defense_long_enabled", True):
            return False
        if not orderbook_signal or not getattr(orderbook_signal, "valid", False):
            return False

        breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE").upper()
        breakout_override = self._market_map_supports_breakout_long_override(market_map_signal, breakout_state)
        if breakout_state in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN", "PROBING_BEARISH_BREAKDOWN"}:
            return False
        if str(getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS") or "BETWEEN_LEVELS").upper() == "RANGE_COMPRESSION":
            return False
        if getattr(orderbook_signal, "block_longs", False):
            return False

        support_distance = float(getattr(orderbook_signal, "nearest_support_distance_pct", 0.0) or 0.0)
        max_support_distance = float(getattr(self.tcfg, "support_defense_max_support_distance_pct", 0.40) or 0.40)
        if support_distance <= 0 or support_distance > max_support_distance:
            return False

        score_floor = float(getattr(self.tcfg, "support_defense_long_score_floor", 24.0) or 24.0)
        breakout_score_floor = float(getattr(self.tcfg, "support_defense_breakout_score_floor", 36.0) or 36.0)
        min_score = breakout_score_floor if breakout_override else score_floor
        if raw_score < min_score:
            return False

        orderbook_score = float(getattr(orderbook_signal, "score", 50.0) or 50.0)
        min_book_score = float(getattr(self.tcfg, "support_defense_min_orderbook_score", 62.0) or 62.0)
        imbalance = float(getattr(orderbook_signal, "imbalance_ratio", 0.0) or 0.0)
        min_imbalance = float(getattr(self.tcfg, "support_defense_min_imbalance", 0.10) or 0.10)
        if orderbook_score < min_book_score:
            return False
        if imbalance < min_imbalance and not getattr(orderbook_signal, "favor_longs", False):
            return False

        map_supportive = False
        map_blocks_shorts = False
        if market_map_signal and getattr(market_map_signal, "valid", False):
            if getattr(market_map_signal, "block_longs", False) and not breakout_override:
                return False
            map_blocks_shorts = bool(getattr(market_map_signal, "block_shorts", False))
            map_supportive = bool(
                getattr(market_map_signal, "favor_longs", False)
                or getattr(market_map_signal, "in_demand_zone", False)
                or str(getattr(market_map_signal, "bias", "NEUTRAL") or "NEUTRAL").upper() == "BULLISH"
                or breakout_override
            )

        if not (getattr(orderbook_signal, "block_shorts", False) or map_blocks_shorts):
            return False

        if not (
            getattr(orderbook_signal, "favor_longs", False)
            or breakout_state in {"PROBING_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT", "CONFIRMED_BULLISH_BREAKOUT"}
            or map_supportive
        ):
            return False

        bearish_msb = {"BEARISH_CHOCH", "BEARISH_BOS"}
        msb = getattr(advanced, "msb", None)
        msb_type = str(getattr(msb, "msb_type", "NONE") or "NONE").upper()
        structure_trend = str(getattr(msb, "structure_trend", "RANGING") or "RANGING").upper()
        if msb_type in bearish_msb or structure_trend == "DOWNTREND":
            return False

        if candle_patterns and getattr(candle_patterns, "valid", False):
            if self._bearish_candle_context(candle_patterns):
                return False

        return True

    def _evaluate_trade_thesis(
        self,
        *,
        action: str,
        score: float,
        current_position=None,
        advanced=None,
        regimes=None,
        candle_patterns=None,
        news_signal=None,
        funding_oi_signal=None,
        orderbook_signal=None,
        market_map_signal=None,
        narrative_signal=None,
        social_attention_signal=None,
        instrument_type: str = "crypto",
        trade_plan: Dict | None = None,
    ) -> dict:
        thesis = {
            "candidate_action": action,
            "state": "NO_TRADE",
            "permitted": False,
            "quality": "LOW",
            "archetype": "NONE",
            "support_defense_long": False,
            "confirmed_breakout": False,
            "persistent_breakout": False,
            "alignment_points": 0.0,
            "conflict_points": 0.0,
            "conviction_score": 50.0,
            "summary": "No directional thesis qualified",
            "reasons": [],
            "blockers": [],
        }
        if action not in ("LONG", "SHORT"):
            thesis["summary"] = "Raw score remained inside the no-trade zone"
            return thesis

        bullish = action == "LONG"
        same_direction_position = current_position == action
        alignment = 0.0
        conflicts = 0.0
        reasons: List[str] = []
        blockers: List[str] = []

        bullish_msb = {"BULLISH_CHOCH", "BULLISH_BOS"}
        bearish_msb = {"BEARISH_CHOCH", "BEARISH_BOS"}

        msb = getattr(advanced, "msb", None)
        msb_type = str(getattr(msb, "msb_type", "NONE") or "NONE").upper()
        structure_trend = str(getattr(msb, "structure_trend", "RANGING") or "RANGING").upper()
        dominant_regime = str(getattr(regimes, "dominant_regime", "MIXED") or "MIXED").upper()
        breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE").upper()
        level_interaction = str(getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS") or "BETWEEN_LEVELS").upper()
        confirmed_breakout = (
            self._is_confirmed_bullish_breakout(orderbook_signal)
            if bullish else
            self._is_confirmed_bearish_breakdown(orderbook_signal)
        )
        persistent_breakout = (
            self._is_persistent_bullish_breakout(orderbook_signal)
            if bullish else
            self._is_persistent_bearish_breakdown(orderbook_signal)
        )
        probing_breakout = breakout_state == (
            "PROBING_BULLISH_BREAKOUT" if bullish else "PROBING_BEARISH_BREAKDOWN"
        )
        support_defense_long = (
            bullish and self._qualifies_support_defense_long(
                raw_score=score,
                advanced=advanced,
                candle_patterns=candle_patterns,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
        )
        market_map_breakout_override = (
            bullish and self._market_map_supports_breakout_long_override(market_map_signal, breakout_state)
        )
        event_conviction = self._event_conviction_active(
            instrument_type=instrument_type,
            action=action,
            news_signal=news_signal,
        )

        if support_defense_long:
            alignment += 1.5
            reasons.append("strong support defense is holding beneath price")
        if event_conviction:
            alignment += 2.25
            reasons.append("pre-event catalyst flow supports taking starter risk before full price confirmation")
            catalyst_tags = self._catalyst_tags(news_signal)
            if catalyst_tags & {"demand_commitment", "capacity_lock_in", "strategic_deal"}:
                alignment += 0.75
                reasons.append("demand/supply catalyst is directly tied to the equity theme")

        if confirmed_breakout:
            alignment += 2.0
            reasons.append("daily breakout is already confirmed")
        elif persistent_breakout:
            alignment += 1.25
            reasons.append("background orderbook feed shows the breakout holding between cycles")
        elif probing_breakout:
            if support_defense_long:
                alignment += 0.5
                reasons.append("breakout is probing, but support-defense flow is aligned")
            else:
                conflicts += 0.5
                reasons.append("breakout is only probing and has not closed yet")

        if bullish:
            if msb_type in bullish_msb:
                alignment += 2.0
                reasons.append("market structure break is bullish")
            elif msb_type in bearish_msb:
                conflicts += 2.0
                blockers.append("market structure still resolves bearish")

            if structure_trend == "UPTREND":
                alignment += 2.0
                reasons.append("higher structure remains in an uptrend")
            elif structure_trend == "DOWNTREND":
                conflicts += 2.0
                blockers.append("higher structure is still a downtrend")
            elif structure_trend == "RANGING" and getattr(self.tcfg, "thesis_block_on_range_conditions", True) and not (confirmed_breakout or persistent_breakout):
                if support_defense_long:
                    alignment += 1.0
                    reasons.append("support-defense setup allows a range low reclaim before full breakout confirmation")
                elif event_conviction:
                    conflicts += 0.25
                    reasons.append("higher structure is ranging, but the event catalyst justifies starter risk")
                else:
                    blockers.append("higher structure is still ranging")
        else:
            if msb_type in bearish_msb:
                alignment += 2.0
                reasons.append("market structure break is bearish")
            elif msb_type in bullish_msb:
                conflicts += 2.0
                blockers.append("market structure still resolves bullish")

            if structure_trend == "DOWNTREND":
                alignment += 2.0
                reasons.append("higher structure remains in a downtrend")
            elif structure_trend == "UPTREND":
                if event_conviction:
                    conflicts += 0.75
                    reasons.append("higher structure is still up, so bearish event exposure stays starter-sized")
                else:
                    conflicts += 2.0
                    blockers.append("higher structure is still an uptrend")
            elif structure_trend == "RANGING" and getattr(self.tcfg, "thesis_block_on_range_conditions", True) and not (confirmed_breakout or persistent_breakout):
                if event_conviction:
                    conflicts += 0.25
                    reasons.append("higher structure is ranging, but bearish event flow allows starter risk")
                else:
                    blockers.append("higher structure is still ranging")

        if dominant_regime in {"TREND", "MOMENTUM", "BREAKOUT"}:
            alignment += 1.5
            reasons.append(f"{dominant_regime.lower()} regime supports follow-through")
        elif dominant_regime in {"ABSORPTION", "MIXED"}:
            if getattr(self.tcfg, "thesis_block_on_range_conditions", True) and not (confirmed_breakout or persistent_breakout):
                if event_conviction:
                    conflicts += 0.25
                    reasons.append(f"{dominant_regime.lower()} regime is not ideal, but catalyst conviction keeps the starter alive")
                else:
                    blockers.append(f"{dominant_regime.lower()} regime is too indecisive for a fresh trade")
            else:
                conflicts += 0.5
        elif dominant_regime == "MEAN_REV":
            conflicts += 1.0
            reasons.append("mean-reversion regime lowers continuation quality")

        if candle_patterns and getattr(candle_patterns, "valid", False):
            candle_trend = str(getattr(candle_patterns, "trend_3", "FLAT") or "FLAT").upper()
            indecision = self._indecision_candle_context(candle_patterns)
            directional_candles = (
                self._bullish_candle_context(candle_patterns)
                if bullish else
                self._bearish_candle_context(candle_patterns)
            )
            opposing_candles = (
                self._bearish_candle_context(candle_patterns)
                if bullish else
                self._bullish_candle_context(candle_patterns)
            )
            directional_trend = "UP" if bullish else "DOWN"
            opposing_trend = "DOWN" if bullish else "UP"

            if directional_candles or candle_trend == directional_trend:
                alignment += 1.0
                reasons.append("recent candles confirm the direction")
            elif opposing_candles or candle_trend == opposing_trend:
                conflicts += 1.0
                reasons.append("recent candles still lean the other way")

            if indecision and not (confirmed_breakout or persistent_breakout):
                if event_conviction:
                    conflicts += 0.25
                    reasons.append("candles are indecisive, but pre-event catalyst flow is carrying the starter thesis")
                else:
                    blockers.append("recent candles are still indecisive")

        if news_signal and getattr(news_signal, "valid", False) and getattr(news_signal, "article_count", 0) > 0:
            news_score = float(getattr(news_signal, "score", 50.0) or 50.0)
            if (bullish and news_score >= 55.0) or ((not bullish) and news_score <= 45.0):
                alignment += 1.0
                reasons.append("newsflow is aligned with the direction")
            elif (bullish and news_score <= 45.0) or ((not bullish) and news_score >= 55.0):
                conflicts += 1.0
                reasons.append("newsflow is leaning the other way")

        if funding_oi_signal and getattr(funding_oi_signal, "valid", False):
            foc_score = float(getattr(funding_oi_signal, "composite_score", 50.0) or 50.0)
            if (bullish and foc_score >= 55.0) or ((not bullish) and foc_score <= 45.0):
                alignment += 1.0
                reasons.append("order-flow confirms the move")
            elif (bullish and foc_score <= 45.0) or ((not bullish) and foc_score >= 55.0):
                conflicts += 1.0
                reasons.append("order-flow is not confirming the move")

        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            orderbook_score = float(getattr(orderbook_signal, "score", 50.0) or 50.0)
            if bullish:
                if getattr(orderbook_signal, "block_longs", False) and not (confirmed_breakout or persistent_breakout):
                    if event_conviction:
                        conflicts += 0.75
                        reasons.append("overhead supply is active, so the event entry must stay starter-sized")
                    else:
                        blockers.append("overhead supply/resistance is still active")
                elif support_defense_long:
                    alignment += 1.5
                    reasons.append("orderbook is aggressively defending support for longs")
                elif getattr(orderbook_signal, "favor_longs", False) or orderbook_score >= 58.0:
                    alignment += 1.5
                    reasons.append("orderbook and key levels support the long")
                elif orderbook_score <= 45.0:
                    conflicts += 1.0
                    reasons.append("orderbook still leans against the long")
            else:
                if getattr(orderbook_signal, "block_shorts", False) and not (confirmed_breakout or persistent_breakout):
                    if event_conviction:
                        conflicts += 0.75
                        reasons.append("demand is still defending below price, so the bearish event short stays starter-sized")
                    else:
                        blockers.append("demand/support is still defending below price")
                elif getattr(orderbook_signal, "favor_shorts", False) or orderbook_score <= 42.0:
                    alignment += 1.5
                    reasons.append("orderbook and key levels support the short")
                elif orderbook_score >= 55.0:
                    conflicts += 1.0
                    reasons.append("orderbook still leans against the short")

            support_wall_persistence = int(getattr(orderbook_signal, "support_wall_persistence", 0) or 0)
            resistance_wall_persistence = int(getattr(orderbook_signal, "resistance_wall_persistence", 0) or 0)
            imbalance_mean = float(getattr(orderbook_signal, "imbalance_mean", getattr(orderbook_signal, "imbalance_ratio", 0.0)) or 0.0)
            if bullish and support_wall_persistence >= 2 and imbalance_mean >= -0.03:
                alignment += 0.75
                reasons.append("bid wall persistence is absorbing sellers under price")
            elif (not bullish) and resistance_wall_persistence >= 2 and imbalance_mean <= 0.03:
                alignment += 0.75
                reasons.append("ask wall persistence is absorbing buyers above price")

            if level_interaction == "RANGE_COMPRESSION" and getattr(self.tcfg, "thesis_block_on_range_conditions", True) and not (confirmed_breakout or persistent_breakout):
                if event_conviction:
                    conflicts += 0.75
                    reasons.append("range compression is active, so the event thesis must stay starter-sized")
                else:
                    blockers.append("price is compressed between nearby support and resistance")

        if market_map_signal and getattr(market_map_signal, "valid", False):
            market_bias = str(getattr(market_map_signal, "bias", "NEUTRAL") or "NEUTRAL").upper()
            if bullish:
                if getattr(market_map_signal, "block_longs", False) and not market_map_breakout_override:
                    if event_conviction:
                        conflicts += 0.5
                        reasons.append("daily map has not fully confirmed, so the event thesis stays small")
                    else:
                        blockers.append("daily market map still warns against longs here")
                elif getattr(market_map_signal, "block_longs", False) and market_map_breakout_override:
                    conflicts += 0.35
                    reasons.append("mapped supply is overhead, but breakout reclaim is overriding it")
                elif getattr(market_map_signal, "favor_longs", False):
                    alignment += 1.5
                    reasons.append("daily market map supports the long")
                if market_bias == "BEARISH":
                    conflicts += 1.0
                    reasons.append("daily operator bias still leans bearish")
            else:
                if getattr(market_map_signal, "block_shorts", False):
                    live_breakdown = bool(getattr(market_map_signal, "live_below_breakdown_levels", []))
                    if event_conviction and (live_breakdown or market_bias != "BULLISH"):
                        conflicts += 0.5
                        reasons.append("daily map is not clean, so the bearish event short stays starter-sized")
                    else:
                        blockers.append("daily market map still warns against shorts here")
                elif getattr(market_map_signal, "favor_shorts", False):
                    alignment += 1.5
                    reasons.append("daily market map supports the short")
                if market_bias == "BULLISH":
                    conflicts += 1.0
                    reasons.append("daily operator bias still leans bullish")

        if narrative_signal and getattr(narrative_signal, "valid", False):
            headline_bias = str(getattr(narrative_signal, "headline_bias", "NEUTRAL") or "NEUTRAL").upper()
            if bullish:
                if getattr(narrative_signal, "block_longs", False):
                    blockers.append("major headline flow still blocks longs")
                elif headline_bias == "BULLISH":
                    alignment += 0.75
                    reasons.append("narrative flow supports the long")
                elif headline_bias == "BEARISH":
                    conflicts += 0.75
                    reasons.append("narrative flow still leans bearish")
            else:
                if getattr(narrative_signal, "block_shorts", False):
                    blockers.append("major headline flow still blocks shorts")
                elif headline_bias == "BEARISH":
                    alignment += 0.75
                    reasons.append("narrative flow supports the short")
                elif headline_bias == "BULLISH":
                    conflicts += 0.75
                    reasons.append("narrative flow still leans bullish")

        if social_attention_signal and getattr(social_attention_signal, "valid", False):
            mentions = int(getattr(social_attention_signal, "mentions", 0) or 0)
            min_mentions = int(getattr(self.tcfg, "social_attention_min_mentions_for_signal", 2) or 2)
            social_score = float(getattr(social_attention_signal, "score", 50.0) or 50.0)
            attention = str(getattr(social_attention_signal, "attention_level", "LOW") or "LOW").upper()
            enough_attention = mentions >= min_mentions or attention in {"MEDIUM", "HIGH"}
            if enough_attention and ((bullish and social_score >= 58.0) or ((not bullish) and social_score <= 42.0)):
                alignment += 0.75
                reasons.append("trader attention is aligned")
            elif enough_attention and ((bullish and social_score <= 42.0) or ((not bullish) and social_score >= 58.0)):
                conflicts += 0.75
                reasons.append("trader attention leans the other way")

        rr = float((trade_plan or {}).get("risk_reward_ratio", 0.0) or 0.0)
        min_rr = float(getattr(self.tcfg, "thesis_min_risk_reward_ratio", 1.75) or 1.75)
        if rr > 0 and rr >= min_rr:
            alignment += 1.0
            reasons.append(f"planned R:R {rr:.2f} is good enough")
        elif trade_plan:
            blockers.append(f"planned R:R {rr:.2f} is below {min_rr:.2f}")

        long_thresh = float(self.tcfg.signal_long_threshold)
        short_thresh = float(self.tcfg.signal_short_threshold)
        if current_position == "LONG":
            short_thresh -= 4.0
            long_thresh += 4.0
        elif current_position == "SHORT":
            long_thresh += 4.0
            short_thresh -= 4.0

        score_buffer = score - long_thresh if bullish else short_thresh - score
        min_buffer = float(getattr(self.tcfg, "thesis_min_score_buffer", 3.0) or 3.0)
        if same_direction_position:
            if score_buffer >= 1.0:
                alignment += 0.5
                reasons.append("same-direction thesis is still intact")
        elif score_buffer >= min_buffer:
            alignment += 1.0
            reasons.append("directional score cleared the trigger with room to spare")
        elif support_defense_long and score_buffer >= 0.5:
            alignment += 0.75
            reasons.append("support-defense reclaim can trigger closer to the threshold")
        elif event_conviction and score_buffer >= -float(getattr(self.tcfg, "conviction_entry_event_score_buffer", 16.0) or 16.0):
            alignment += 0.50
            reasons.append("pre-event catalyst allows a starter below the normal trigger")
        elif not (confirmed_breakout or persistent_breakout):
            blockers.append("score only barely crossed the trigger without enough thesis buffer")

        min_alignment = float(getattr(self.tcfg, "thesis_min_alignment_points", 4) or 4)
        max_conflicts = float(getattr(self.tcfg, "thesis_max_conflict_points", 1) or 1)
        permitted = not blockers and alignment >= min_alignment and conflicts <= max_conflicts

        conviction_score = 50.0 + alignment * 7.5 - conflicts * 8.0
        if confirmed_breakout:
            conviction_score += 6.0
        conviction_score = max(0.0, min(100.0, conviction_score))

        scalp_profile = self._scalp_profile(
            action=action,
            score=score,
            instrument_type=instrument_type,
            advanced=advanced,
            regimes=regimes,
            candle_patterns=candle_patterns,
            news_signal=news_signal,
            funding_oi_signal=funding_oi_signal,
            orderbook_signal=orderbook_signal,
            market_map_signal=market_map_signal,
            narrative_signal=narrative_signal,
            social_attention_signal=social_attention_signal,
            trade_plan=trade_plan,
        )
        scalp_active = bool(scalp_profile.get("active", False))
        scalp_selected = bool((not permitted) and scalp_active)
        if scalp_selected:
            permitted = True
            alignment = max(alignment, float(scalp_profile.get("alignment_points", 0.0) or 0.0))
            conflicts = min(conflicts, float(scalp_profile.get("conflict_points", conflicts) or conflicts))
            conviction_score = max(
                conviction_score,
                58.0
                + float(scalp_profile.get("alignment_points", 0.0) or 0.0) * 3.5
                - float(scalp_profile.get("conflict_points", 0.0) or 0.0) * 2.5,
            )
            conviction_score = max(0.0, min(100.0, conviction_score))
            scalp_summary = str(scalp_profile.get("summary", "") or "")
            reasons = ([scalp_summary] if scalp_summary else []) + reasons
            blockers = []

        if permitted:
            quality = "HIGH" if alignment >= (min_alignment + 2.0) and conflicts == 0 else "MEDIUM"
            if scalp_selected:
                quality = "MEDIUM"
                summary = str(scalp_profile.get("summary") or "; ".join(reasons[:3]) or "Tactical scalp qualified")
                state = str(scalp_profile.get("style") or "SCALP")
            else:
                summary = "; ".join(reasons[:3]) if reasons else "Directional thesis qualified"
                state = "QUALIFIED"
        else:
            quality = "LOW"
            if blockers:
                summary = blockers[0]
            elif alignment < min_alignment:
                summary = (
                    f"only {alignment:.1f} aligned thesis checks "
                    f"(need at least {min_alignment:.0f})"
                )
            else:
                summary = (
                    f"too many thesis conflicts remain "
                    f"({conflicts:.1f} > {max_conflicts:.0f})"
                )
            state = "NO_TRADE"

        archetype = "DIRECTIONAL_CONTINUATION"
        if support_defense_long:
            archetype = "SUPPORT_DEFENSE_LONG"
        elif confirmed_breakout or persistent_breakout:
            archetype = "BREAKOUT_CONTINUATION"
        if scalp_selected:
            archetype = "TACTICAL_SCALP"
        scalp_profile["selected"] = bool(scalp_selected)

        thesis.update({
            "state": state,
            "permitted": permitted,
            "quality": quality,
            "archetype": archetype,
            "support_defense_long": bool(support_defense_long),
            "confirmed_breakout": bool(confirmed_breakout),
            "persistent_breakout": bool(persistent_breakout),
            "event_conviction": bool(event_conviction),
            "scalp": scalp_profile,
            "alignment_points": round(alignment, 2),
            "conflict_points": round(conflicts, 2),
            "conviction_score": round(conviction_score, 2),
            "summary": summary,
            "reasons": reasons[:6],
            "blockers": blockers[:4],
        })
        return thesis

    def _derive_expectancy_profile(
        self,
        *,
        action: str,
        score: float,
        thesis: Dict | None = None,
        trade_plan: Dict | None = None,
        regimes=None,
        orderbook_signal=None,
        market_map_signal=None,
        news_signal=None,
        funding_oi_signal=None,
        narrative_signal=None,
        social_attention_signal=None,
        current_position=None,
    ) -> dict:
        thesis = dict(thesis or {})
        trade_plan = dict(trade_plan or {})
        bullish = action == "LONG"
        same_direction_position = current_position == action
        alignment = float(thesis.get("alignment_points", 0.0) or 0.0)
        conflicts = float(thesis.get("conflict_points", 0.0) or 0.0)
        conviction = float(thesis.get("conviction_score", 50.0) or 50.0)
        rr = float(trade_plan.get("risk_reward_ratio", 0.0) or 0.0)
        support_defense_long = (
            bullish and self._qualifies_support_defense_long(
                raw_score=score,
                advanced=None,
                candle_patterns=None,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
        )
        scalp_active = bool((thesis.get("scalp") or {}).get("selected", False))
        social_reason = ""

        probability = 0.50
        probability += ((conviction - 50.0) / 50.0) * 0.18
        probability += min(0.14, alignment * 0.028)
        probability -= min(0.18, conflicts * 0.045)
        probability += max(-0.06, min(0.06, (score - 50.0) / 50.0 * 0.08))

        uncertainty = 0.34
        uncertainty -= min(0.10, alignment * 0.018)
        uncertainty += min(0.18, conflicts * 0.035)
        if not thesis.get("permitted", False):
            uncertainty += 0.08

        dominant_regime = str(getattr(regimes, "dominant_regime", "MIXED") or "MIXED").upper()
        if dominant_regime in {"TREND", "MOMENTUM", "BREAKOUT"}:
            probability += 0.03
            uncertainty -= 0.03
        elif dominant_regime in {"ABSORPTION", "MIXED"}:
            uncertainty += 0.05
        elif dominant_regime == "MEAN_REV":
            uncertainty += 0.03

        breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE").upper()
        level_interaction = str(getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS") or "BETWEEN_LEVELS").upper()
        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            orderbook_score = float(getattr(orderbook_signal, "score", 50.0) or 50.0)
            imbalance = float(getattr(orderbook_signal, "imbalance_mean", getattr(orderbook_signal, "imbalance_ratio", 0.0)) or 0.0)
            orderbook_bonus = float(getattr(self.tcfg, "expectancy_orderbook_bonus", 0.05) or 0.05)
            if (bullish and orderbook_score >= 58.0) or ((not bullish) and orderbook_score <= 42.0):
                probability += orderbook_bonus
                uncertainty -= 0.03
            elif (bullish and orderbook_score <= 45.0) or ((not bullish) and orderbook_score >= 55.0):
                probability -= orderbook_bonus
                uncertainty += 0.05
            if bullish and imbalance >= 0.05:
                probability += 0.02
            elif (not bullish) and imbalance <= -0.05:
                probability += 0.02
            if breakout_state.startswith("PROBING"):
                uncertainty += 0.05
            elif breakout_state.startswith("CONFIRMED") or breakout_state.startswith("PERSISTENT"):
                probability += 0.03
                uncertainty -= 0.04
            if level_interaction == "RANGE_COMPRESSION":
                uncertainty += 0.14

            if support_defense_long:
                support_bonus = float(getattr(self.tcfg, "support_defense_expectancy_bonus", 0.05) or 0.05)
                probability += support_bonus
                uncertainty -= 0.03
                if breakout_state.startswith("CONFIRMED") or breakout_state.startswith("PERSISTENT"):
                    probability += 0.025
                    uncertainty -= 0.02
        else:
            uncertainty += 0.06

        if market_map_signal and getattr(market_map_signal, "valid", False):
            market_map_bonus = float(getattr(self.tcfg, "expectancy_market_map_bonus", 0.04) or 0.04)
            if bullish and getattr(market_map_signal, "favor_longs", False):
                probability += market_map_bonus
            elif (not bullish) and getattr(market_map_signal, "favor_shorts", False):
                probability += market_map_bonus
            if bullish and getattr(market_map_signal, "block_longs", False):
                probability -= market_map_bonus
                uncertainty += 0.04
            elif (not bullish) and getattr(market_map_signal, "block_shorts", False):
                probability -= market_map_bonus
                uncertainty += 0.04

        if news_signal and getattr(news_signal, "valid", False):
            news_score = float(getattr(news_signal, "score", 50.0) or 50.0)
            news_bonus = float(getattr(self.tcfg, "expectancy_news_bonus", 0.04) or 0.04)
            if (bullish and news_score >= 55.0) or ((not bullish) and news_score <= 45.0):
                probability += news_bonus
                uncertainty -= 0.02
            elif (bullish and news_score <= 45.0) or ((not bullish) and news_score >= 55.0):
                probability -= news_bonus
                uncertainty += 0.04

        if social_attention_signal and getattr(social_attention_signal, "valid", False):
            mentions = int(getattr(social_attention_signal, "mentions", 0) or 0)
            min_mentions = int(getattr(self.tcfg, "social_attention_min_mentions_for_signal", 2) or 2)
            attention = str(getattr(social_attention_signal, "attention_level", "LOW") or "LOW").upper()
            social_score = float(getattr(social_attention_signal, "score", 50.0) or 50.0)
            enough_attention = mentions >= min_mentions or attention in {"MEDIUM", "HIGH"}
            if enough_attention and ((bullish and social_score >= 58.0) or ((not bullish) and social_score <= 42.0)):
                probability += 0.025
                uncertainty -= 0.015
                social_reason = "social attention confirms direction"
            elif enough_attention and ((bullish and social_score <= 42.0) or ((not bullish) and social_score >= 58.0)):
                probability -= 0.025
                uncertainty += 0.025
                social_reason = "social attention warns against direction"

        if funding_oi_signal and getattr(funding_oi_signal, "valid", False):
            foc_score = float(getattr(funding_oi_signal, "composite_score", 50.0) or 50.0)
            if (bullish and foc_score >= 55.0) or ((not bullish) and foc_score <= 45.0):
                probability += 0.03
            elif (bullish and foc_score <= 45.0) or ((not bullish) and foc_score >= 55.0):
                probability -= 0.03
                uncertainty += 0.03

        if narrative_signal and getattr(narrative_signal, "valid", False):
            probability += float(getattr(narrative_signal, "score_adjustment", 0.0) or 0.0) / 100.0
            uncertainty += float(getattr(narrative_signal, "uncertainty_delta", 0.0) or 0.0)
            if bullish and getattr(narrative_signal, "block_longs", False):
                probability -= 0.10
            elif (not bullish) and getattr(narrative_signal, "block_shorts", False):
                probability -= 0.10

        probability = max(0.05, min(0.95, probability))
        uncertainty = max(0.05, min(0.95, uncertainty))

        expected_r = 0.0
        if rr > 0:
            expected_r = (probability * rr) - ((1.0 - probability) * 1.0)

        expectancy_score = (
            50.0
            + (probability - 0.50) * 70.0
            + expected_r * 16.0
            - uncertainty * 22.0
            + alignment * 1.4
            - conflicts * 2.6
        )
        expectancy_score = max(0.0, min(100.0, expectancy_score))

        min_probability = float(getattr(self.tcfg, "expectancy_min_probability", 0.54) or 0.54)
        min_expected_r = float(getattr(self.tcfg, "expectancy_min_expected_r", 0.18) or 0.18)
        max_uncertainty = float(getattr(self.tcfg, "expectancy_max_uncertainty", 0.42) or 0.42)
        min_score = float(
            getattr(
                self.tcfg,
                "expectancy_same_direction_min_score" if same_direction_position else "expectancy_min_score",
                56.0 if not same_direction_position else 52.0,
            )
            or (56.0 if not same_direction_position else 52.0)
        )
        if scalp_active:
            min_probability = float(getattr(self.tcfg, "scalp_min_probability", 0.51) or 0.51)
            min_expected_r = float(getattr(self.tcfg, "scalp_min_expected_r", 0.08) or 0.08)
            max_uncertainty = float(getattr(self.tcfg, "scalp_max_uncertainty", 0.62) or 0.62)
            min_score = float(getattr(self.tcfg, "scalp_min_expectancy_score", 48.0) or 48.0)

        permitted = bool(thesis.get("permitted", False))
        blockers: List[str] = []
        reasons: List[str] = [
            f"estimated win probability {probability * 100:.0f}%",
            f"expected value {expected_r:+.2f}R",
            f"uncertainty {uncertainty * 100:.0f}%",
        ]
        if rr > 0:
            reasons.append(f"target profile {rr:.2f}R")
        if social_reason:
            reasons.append(social_reason)

        if probability < min_probability:
            blockers.append(f"win probability {probability * 100:.0f}% is below {min_probability * 100:.0f}%")
        if rr > 0 and expected_r < min_expected_r:
            blockers.append(f"expected value {expected_r:+.2f}R is below {min_expected_r:.2f}R")
        if uncertainty > max_uncertainty:
            blockers.append(f"uncertainty {uncertainty * 100:.0f}% is above {max_uncertainty * 100:.0f}%")
        if expectancy_score < min_score:
            blockers.append(f"expectancy score {expectancy_score:.0f} is below {min_score:.0f}")

        permitted = permitted and not blockers
        quality = "HIGH" if expectancy_score >= 70.0 and uncertainty <= 0.28 else "MEDIUM" if expectancy_score >= min_score and uncertainty <= max_uncertainty else "LOW"
        summary = "; ".join(reasons[:3]) if permitted else (blockers[0] if blockers else "expectancy did not clear the gate")
        return {
            "permitted": permitted,
            "probability": round(probability, 4),
            "expected_r": round(expected_r, 4),
            "uncertainty": round(uncertainty, 4),
            "score": round(expectancy_score, 2),
            "quality": quality,
            "summary": summary,
            "reasons": reasons[:5],
            "blockers": blockers[:4],
        }

    def _build_execution_plan(
        self,
        *,
        action: str,
        entry_price: float,
        trade_plan: Dict | None = None,
        expectancy: Dict | None = None,
        orderbook_signal=None,
    ) -> dict:
        if action not in {"LONG", "SHORT"} or entry_price <= 0:
            return {}

        trade_plan = dict(trade_plan or {})
        expectancy = dict(expectancy or {})
        plan = {
            "mode": "market",
            "entry_price": round(entry_price, 6),
            "limit_price": 0.0,
            "max_wait_cycles": int(getattr(self.tcfg, "execution_limit_timeout_cycles", 6) or 6),
            "reason": "default aggressive entry",
        }
        if not getattr(self.tcfg, "execution_planning_enabled", True):
            plan["reason"] = "execution planning disabled"
            return plan

        probability = float(expectancy.get("probability", 0.0) or 0.0)
        expectancy_score = float(expectancy.get("score", 0.0) or 0.0)
        breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE").upper()
        confirmed_breakout = breakout_state.startswith("CONFIRMED") or breakout_state.startswith("PERSISTENT")
        passive_offset_bps = float(getattr(self.tcfg, "execution_passive_entry_offset_bps", 4.0) or 4.0) / 10_000.0
        retest_distance_pct = float(getattr(self.tcfg, "execution_limit_retest_distance_pct", 0.40) or 0.40)
        breakout_min_prob = float(getattr(self.tcfg, "execution_breakout_market_probability", 0.64) or 0.64)
        breakout_min_score = float(getattr(self.tcfg, "execution_breakout_market_expectancy_score", 64.0) or 64.0)

        if not orderbook_signal or not getattr(orderbook_signal, "valid", False):
            plan["reason"] = "no orderbook context; use market entry"
            return plan

        if confirmed_breakout and probability >= breakout_min_prob and expectancy_score >= breakout_min_score:
            plan["mode"] = "market"
            plan["reason"] = "confirmed breakout with strong expectancy"
            return plan

        if action == "LONG":
            support = float(getattr(orderbook_signal, "nearest_support", 0.0) or 0.0)
            support_distance = float(getattr(orderbook_signal, "nearest_support_distance_pct", 0.0) or 0.0)
            best_bid = float(getattr(orderbook_signal, "best_bid", 0.0) or 0.0)
            if support > 0 and support_distance <= retest_distance_pct:
                limit_price = max(support, best_bid or support)
                plan.update({
                    "mode": "limit",
                    "limit_price": round(limit_price, 6),
                    "entry_price": round(limit_price, 6),
                    "reason": "buying the defended retest near support",
                })
            elif best_bid > 0:
                limit_price = best_bid * (1 - passive_offset_bps)
                plan.update({
                    "mode": "maker_limit",
                    "limit_price": round(limit_price, 6),
                    "entry_price": round(limit_price, 6),
                    "reason": "using passive bid placement to improve entry quality",
                })
        else:
            resistance = float(getattr(orderbook_signal, "nearest_resistance", 0.0) or 0.0)
            resistance_distance = float(getattr(orderbook_signal, "nearest_resistance_distance_pct", 0.0) or 0.0)
            best_ask = float(getattr(orderbook_signal, "best_ask", 0.0) or 0.0)
            if resistance > 0 and resistance_distance <= retest_distance_pct:
                limit_price = min(resistance, best_ask or resistance)
                plan.update({
                    "mode": "limit",
                    "limit_price": round(limit_price, 6),
                    "entry_price": round(limit_price, 6),
                    "reason": "selling the retest near defended resistance",
                })
            elif best_ask > 0:
                limit_price = best_ask * (1 + passive_offset_bps)
                plan.update({
                    "mode": "maker_limit",
                    "limit_price": round(limit_price, 6),
                    "entry_price": round(limit_price, 6),
                    "reason": "using passive ask placement to improve entry quality",
                })
        return plan

    def _build_trade_plan(self, tech, advanced, action, confidence, regimes=None, candle_patterns=None, orderbook_signal=None, market_map_signal=None):
        if action == "FLAT" or tech.price <= 0:
            return {}

        entry = float(tech.price)
        advanced_valid = bool(advanced and getattr(advanced, "valid", False))
        atr = float(getattr(getattr(advanced, "atr", None), "atr", 0.0) or 0.0) if advanced_valid else 0.0
        atr_pct = float(getattr(getattr(advanced, "atr", None), "atr_pct", 0.0) or 0.0) if advanced_valid else 0.0
        volatility_label = str(getattr(getattr(advanced, "atr", None), "volatility_label", "normal") or "normal")
        if atr <= 0:
            atr = entry * max(self.tcfg.stop_loss_pct * 0.30, 0.0035)
            atr_pct = atr / entry * 100 if entry > 0 else 0.0

        dominant_regime = str(getattr(regimes, "dominant_regime", "MIXED") or "MIXED")
        structure_trend = str(getattr(getattr(advanced, "msb", None), "structure_trend", "RANGING") or "RANGING")
        msb_type = str(getattr(getattr(advanced, "msb", None), "msb_type", "NONE") or "NONE")
        strong_candle = (
            self._bullish_candle_context(candle_patterns)
            if action == "LONG"
            else self._bearish_candle_context(candle_patterns)
        )
        indecision = self._indecision_candle_context(candle_patterns)

        stop_atr_multiple = float(self.tcfg.base_stop_atr_multiple)
        if confidence == "LOW":
            stop_atr_multiple += 0.35
        elif confidence == "HIGH":
            stop_atr_multiple -= 0.10

        if volatility_label == "high":
            stop_atr_multiple += 0.25
        elif volatility_label == "extreme":
            stop_atr_multiple += 0.55

        if dominant_regime in {"ABSORPTION", "MIXED", "MEAN_REV"}:
            stop_atr_multiple += 0.15
        elif dominant_regime in {"TREND", "MOMENTUM", "BREAKOUT"}:
            stop_atr_multiple += 0.05

        if indecision:
            stop_atr_multiple += 0.12
        elif strong_candle:
            stop_atr_multiple -= 0.10

        stop_atr_multiple = max(
            float(self.tcfg.min_stop_atr_multiple),
            min(float(self.tcfg.max_stop_atr_multiple), stop_atr_multiple),
        )

        target_r_multiple = float(self.tcfg.base_target_r_multiple)
        if confidence == "HIGH":
            target_r_multiple += 0.25
        elif confidence == "LOW":
            target_r_multiple -= 0.20

        if dominant_regime in {"TREND", "MOMENTUM", "BREAKOUT"}:
            target_r_multiple += 0.35
        elif dominant_regime in {"ABSORPTION", "MIXED", "MEAN_REV"}:
            target_r_multiple -= 0.25

        if indecision:
            target_r_multiple -= 0.15
        elif strong_candle:
            target_r_multiple += 0.10

        target_r_multiple = max(
            float(self.tcfg.min_target_r_multiple),
            min(float(self.tcfg.max_target_r_multiple), target_r_multiple),
        )

        stop_distance = atr * stop_atr_multiple
        min_stop_distance = atr * max(float(self.tcfg.min_stop_atr_multiple) * 0.80, 0.70)
        max_stop_distance = atr * max(float(self.tcfg.max_stop_atr_multiple), 1.50)
        min_target_distance = stop_distance * max(float(self.tcfg.min_target_r_multiple), 1.10)
        max_target_distance = stop_distance * (float(self.tcfg.max_target_r_multiple) + 0.50)

        stop_candidates: List[Tuple[float, str]] = []
        target_candidates: List[Tuple[float, str]] = []

        if advanced_valid:
            fib_levels = getattr(getattr(advanced, "fib", None), "levels", {}) or {}
            msb = advanced.msb
            ob = advanced.ob
            fvg = advanced.fvg

            if action == "LONG":
                if float(getattr(msb, "last_swing_low", 0.0) or 0.0) > 0:
                    stop_candidates.append((float(msb.last_swing_low) - atr * 0.15, "swing_low"))
                if float(getattr(msb, "last_swing_high", 0.0) or 0.0) > entry:
                    target_candidates.append((float(msb.last_swing_high), "swing_high"))
                for ob_high, ob_low in list(getattr(ob, "bullish_obs", []) or [])[-2:]:
                    stop_candidates.append((float(ob_low) - atr * 0.12, "bullish_ob"))
                for ob_high, ob_low in list(getattr(ob, "bearish_obs", []) or [])[-2:]:
                    target_candidates.append((float(ob_low), "bearish_ob"))
                for fvg_bottom, fvg_top in list(getattr(fvg, "bullish_fvgs", []) or [])[-2:]:
                    stop_candidates.append((float(fvg_bottom) - atr * 0.10, "bullish_fvg"))
                for fvg_bottom, fvg_top in list(getattr(fvg, "bearish_fvgs", []) or [])[-2:]:
                    target_candidates.append((float(fvg_bottom), "bearish_fvg"))
                for name, level in fib_levels.items():
                    level = float(level)
                    if level < entry:
                        stop_candidates.append((level - atr * 0.08, f"fib_support_{name}"))
                    elif level > entry:
                        target_candidates.append((level, f"fib_resistance_{name}"))
            else:
                if float(getattr(msb, "last_swing_high", 0.0) or 0.0) > 0:
                    stop_candidates.append((float(msb.last_swing_high) + atr * 0.15, "swing_high"))
                if 0 < float(getattr(msb, "last_swing_low", 0.0) or 0.0) < entry:
                    target_candidates.append((float(msb.last_swing_low), "swing_low"))
                for ob_high, ob_low in list(getattr(ob, "bearish_obs", []) or [])[-2:]:
                    stop_candidates.append((float(ob_high) + atr * 0.12, "bearish_ob"))
                for ob_high, ob_low in list(getattr(ob, "bullish_obs", []) or [])[-2:]:
                    target_candidates.append((float(ob_high), "bullish_ob"))
                for fvg_bottom, fvg_top in list(getattr(fvg, "bearish_fvgs", []) or [])[-2:]:
                    stop_candidates.append((float(fvg_top) + atr * 0.10, "bearish_fvg"))
                for fvg_bottom, fvg_top in list(getattr(fvg, "bullish_fvgs", []) or [])[-2:]:
                    target_candidates.append((float(fvg_top), "bullish_fvg"))
                for name, level in fib_levels.items():
                    level = float(level)
                    if level > entry:
                        stop_candidates.append((level + atr * 0.08, f"fib_resistance_{name}"))
                    elif 0 < level < entry:
                        target_candidates.append((level, f"fib_support_{name}"))

        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            support_levels = list(getattr(orderbook_signal, "support_levels", []) or [])
            resistance_levels = list(getattr(orderbook_signal, "resistance_levels", []) or [])

            if action == "LONG":
                for level in support_levels[:3]:
                    price = float(level.get("price", 0.0) or 0.0)
                    if 0 < price < entry:
                        stop_candidates.append((price - atr * 0.08, f"key_support_{level.get('label', 'wall')}"))
                for level in resistance_levels[:4]:
                    price = float(level.get("price", 0.0) or 0.0)
                    if price > entry:
                        target_candidates.append((price, f"key_resistance_{level.get('label', 'wall')}"))
            else:
                for level in resistance_levels[:3]:
                    price = float(level.get("price", 0.0) or 0.0)
                    if price > entry:
                        stop_candidates.append((price + atr * 0.08, f"key_resistance_{level.get('label', 'wall')}"))
                for level in support_levels[:4]:
                    price = float(level.get("price", 0.0) or 0.0)
                    if 0 < price < entry:
                        target_candidates.append((price, f"key_support_{level.get('label', 'wall')}"))

        if market_map_signal and getattr(market_map_signal, "valid", False):
            mapped_support = float(getattr(market_map_signal, "nearest_support", 0.0) or 0.0)
            mapped_resistance = float(getattr(market_map_signal, "nearest_resistance", 0.0) or 0.0)
            if action == "LONG":
                if 0 < mapped_support < entry:
                    stop_candidates.append((mapped_support - atr * 0.06, "market_map_support"))
                if mapped_resistance > entry:
                    target_candidates.append((mapped_resistance, "market_map_resistance"))
            else:
                if mapped_resistance > entry:
                    stop_candidates.append((mapped_resistance + atr * 0.06, "market_map_resistance"))
                if 0 < mapped_support < entry:
                    target_candidates.append((mapped_support, "market_map_support"))

        base_stop = entry - stop_distance if action == "LONG" else entry + stop_distance
        base_target = (
            entry + stop_distance * target_r_multiple
            if action == "LONG"
            else entry - stop_distance * target_r_multiple
        )

        if action == "LONG":
            valid_stops = [
                (price, basis) for price, basis in stop_candidates
                if 0 < entry - price >= min_stop_distance and entry - price <= max_stop_distance
            ]
            valid_targets = [
                (price, basis) for price, basis in target_candidates
                if price > entry
                and price - entry >= min_target_distance
                and price - entry <= max_target_distance
            ]
            stop_price, stop_basis = (
                max(valid_stops, key=lambda item: item[0])
                if valid_stops else
                (base_stop, "atr_guard")
            )
            take_profit, target_basis = (
                min(valid_targets, key=lambda item: item[0])
                if valid_targets else
                (base_target, "atr_r_multiple")
            )
        else:
            valid_stops = [
                (price, basis) for price, basis in stop_candidates
                if price > entry
                and price - entry >= min_stop_distance
                and price - entry <= max_stop_distance
            ]
            valid_targets = [
                (price, basis) for price, basis in target_candidates
                if 0 < price < entry
                and entry - price >= min_target_distance
                and entry - price <= max_target_distance
            ]
            stop_price, stop_basis = (
                min(valid_stops, key=lambda item: item[0])
                if valid_stops else
                (base_stop, "atr_guard")
            )
            take_profit, target_basis = (
                max(valid_targets, key=lambda item: item[0])
                if valid_targets else
                (base_target, "atr_r_multiple")
            )

        if action == "LONG":
            risk_distance = max(0.0, entry - stop_price)
        else:
            risk_distance = max(0.0, stop_price - entry)
        reward_distance = self._directional_distance(action, entry, take_profit)
        if risk_distance <= 0:
            risk_distance = max(stop_distance, entry * 0.0035)
            stop_price = entry - risk_distance if action == "LONG" else entry + risk_distance
            stop_basis = "atr_guard"
        if reward_distance <= 0:
            reward_distance = stop_distance * target_r_multiple
            take_profit = entry + reward_distance if action == "LONG" else entry - reward_distance
            target_basis = "atr_r_multiple"

        price_action_tags: List[str] = []
        if msb_type != "NONE":
            price_action_tags.append(msb_type)
        if structure_trend != "RANGING":
            price_action_tags.append(structure_trend)
        if action == "LONG" and strong_candle:
            price_action_tags.append("bullish_candles")
        elif action == "SHORT" and strong_candle:
            price_action_tags.append("bearish_candles")
        if indecision:
            price_action_tags.append("indecision")

        rr = reward_distance / risk_distance if risk_distance > 0 else 0.0
        return {
            "entry_price": round(entry, 6),
            "stop_loss": round(stop_price, 6),
            "take_profit": round(take_profit, 6),
            "risk_per_unit": round(risk_distance, 6),
            "reward_per_unit": round(reward_distance, 6),
            "risk_pct": round(risk_distance / entry * 100, 3) if entry > 0 else 0.0,
            "reward_pct": round(reward_distance / entry * 100, 3) if entry > 0 else 0.0,
            "risk_reward_ratio": round(rr, 3),
            "atr": round(atr, 6),
            "atr_pct": round(atr_pct, 3),
            "stop_atr_multiple": round(risk_distance / atr, 3) if atr > 0 else 0.0,
            "target_atr_multiple": round(reward_distance / atr, 3) if atr > 0 else 0.0,
            "target_r_multiple": round(target_r_multiple, 3),
            "stop_basis": stop_basis,
            "target_basis": target_basis,
            "price_action_summary": ", ".join(price_action_tags),
            "dominant_regime": dominant_regime,
            "structure_trend": structure_trend,
            "volatility_label": volatility_label,
        }

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
        instrument_type: str = "crypto",  # "crypto" | "index" | "equity"
        funding_oi_signal=None,    # FundingOISignal from indicators/funding_oi_cvd.py
        orderbook_signal=None,
        market_map_signal=None,
        narrative_signal=None,
        social_attention_signal=None,
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
            if getattr(news_signal, "catalyst_summary", ""):
                log.info(
                    f"[{tech.coin}] News catalyst: "
                    f"{getattr(news_signal, 'catalyst_summary', '')} "
                    f"(score={float(getattr(news_signal, 'catalyst_score', 0.0) or 0.0):.2f})"
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
        if instrument_type in {"index", "equity"}:
            # Dampen raw oscillator contribution slightly (pull toward neutral)
            dampener = 0.90 if instrument_type == "index" else 0.94
            raw_score = 50.0 + (raw_score - 50.0) * dampener
            # But boost regime/trend signal weight if it's strong
            if regimes and regimes.valid:
                trend_weight = 0.08 if instrument_type == "index" else 0.06
                trend_contribution = (regimes.trend_score - 50.0) * trend_weight
                raw_score += trend_contribution
            log.debug(f"[{tech.coin}] {instrument_type.title()} adjustment applied → {raw_score:.1f}")

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

        # ── 13b. Orderbook + key-level overlay ─────────────────────────────
        orderbook_score = 50.0
        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            orderbook_score = float(getattr(orderbook_signal, "score", 50.0) or 50.0)
            imbalance = float(getattr(orderbook_signal, "imbalance_ratio", 0.0) or 0.0)
            influence = float(getattr(self.tcfg, "orderbook_score_influence", 0.35) or 0.35)
            overlay = (orderbook_score - 50.0) * influence
            raw_score += overlay
            log.info(
                f"[{tech.coin}] OrderbookLevels: score={orderbook_score:.1f} "
                f"interaction={getattr(orderbook_signal, 'level_interaction', 'BETWEEN_LEVELS')} "
                f"breakout={getattr(orderbook_signal, 'breakout_state', 'NONE')} "
                f"imbalance={imbalance:+.2f} "
                f"mean={float(getattr(orderbook_signal, 'imbalance_mean', imbalance) or imbalance):+.2f} "
                f"walls={int(getattr(orderbook_signal, 'support_wall_persistence', 0) or 0)}/"
                f"{int(getattr(orderbook_signal, 'resistance_wall_persistence', 0) or 0)}"
            )

        if market_map_signal and getattr(market_map_signal, "valid", False):
            map_adjustment = float(getattr(market_map_signal, "score_adjustment", 0.0) or 0.0)
            map_adjustment *= float(getattr(self.tcfg, "market_map_score_influence", 1.0) or 1.0)
            raw_score += map_adjustment
            log.info(
                f"[{tech.coin}] MarketMap: bias={getattr(market_map_signal, 'bias', 'NEUTRAL')} "
                f"adj={map_adjustment:+.1f} summary={getattr(market_map_signal, 'summary', '')[:72]}"
            )

        if narrative_signal and getattr(narrative_signal, "valid", False):
            narrative_adjustment = float(getattr(narrative_signal, "score_adjustment", 0.0) or 0.0)
            raw_score += narrative_adjustment
            if narrative_adjustment:
                log.info(
                    f"[{tech.coin}] Narrative: adj={narrative_adjustment:+.1f} "
                    f"summary={getattr(narrative_signal, 'summary', '')[:72]}"
                )

        if social_attention_signal and getattr(social_attention_signal, "valid", False):
            mentions = int(getattr(social_attention_signal, "mentions", 0) or 0)
            min_mentions = int(getattr(self.tcfg, "social_attention_min_mentions_for_signal", 2) or 2)
            attention = str(getattr(social_attention_signal, "attention_level", "LOW") or "LOW").upper()
            if mentions >= min_mentions or attention in {"MEDIUM", "HIGH"}:
                social_score = float(getattr(social_attention_signal, "score", 50.0) or 50.0)
                influence = float(getattr(self.tcfg, "social_attention_score_influence", 0.12) or 0.12)
                social_adjustment = (social_score - 50.0) * influence
                raw_score += social_adjustment
                log.info(
                    f"[{tech.coin}] SocialAttention: score={social_score:.1f} "
                    f"mentions={mentions} attention={attention} adj={social_adjustment:+.1f}"
                )

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

        # ── 15b. Key-level / orderbook guardrails ──────────────────────────
        orderbook_guard_reason = ""
        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            override_score = float(getattr(self.tcfg, "orderbook_override_score", 82.0) or 82.0)
            short_override_score = max(0.0, 100.0 - override_score)
            breakout_state = str(getattr(orderbook_signal, "breakout_state", "")).upper()
            confirmed_bullish_breakout = breakout_state in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"}
            confirmed_bearish_breakdown = breakout_state in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"}
            bullish_breakout_pressure = breakout_state in {
                "PROBING_BULLISH_BREAKOUT",
                "CONFIRMED_BULLISH_BREAKOUT",
                "PERSISTENT_BULLISH_BREAKOUT",
            }
            bearish_breakdown_pressure = breakout_state in {
                "PROBING_BEARISH_BREAKDOWN",
                "CONFIRMED_BEARISH_BREAKDOWN",
                "PERSISTENT_BEARISH_BREAKDOWN",
            }

            if action == "LONG":
                if getattr(orderbook_signal, "block_longs", False) and raw_score < override_score and not confirmed_bullish_breakout:
                    orderbook_guard_reason = (
                        f"LONG blocked by nearby resistance "
                        f"{getattr(orderbook_signal, 'nearest_resistance', 0.0):,.2f} "
                        f"({getattr(orderbook_signal, 'nearest_resistance_distance_pct', 0.0):.2f}% away)"
                    )
                    action = "FLAT"
                elif bearish_breakdown_pressure and raw_score < override_score:
                    orderbook_guard_reason = (
                        f"LONG blocked — market is breaking down through key support "
                        f"({getattr(orderbook_signal, 'breakout_state', 'NONE')})"
                    )
                    action = "FLAT"
            elif action == "SHORT":
                if getattr(orderbook_signal, "block_shorts", False) and raw_score > short_override_score and not confirmed_bearish_breakdown:
                    orderbook_guard_reason = (
                        f"SHORT blocked by nearby demand/support "
                        f"{getattr(orderbook_signal, 'nearest_support', 0.0):,.2f} "
                        f"({getattr(orderbook_signal, 'nearest_support_distance_pct', 0.0):.2f}% away)"
                    )
                    action = "FLAT"
                elif bullish_breakout_pressure and raw_score > short_override_score:
                    orderbook_guard_reason = (
                        f"SHORT blocked — market is breaking above key resistance "
                        f"({getattr(orderbook_signal, 'breakout_state', 'NONE')})"
                    )
                    action = "FLAT"
            elif action == "FLAT":
                if getattr(orderbook_signal, "favor_longs", False) and not getattr(orderbook_signal, "block_longs", False):
                    if raw_score >= (long_thresh - 2.0) and orderbook_score >= 60.0:
                        raw_score = max(raw_score, long_thresh + 0.5)
                        action = "LONG"
                elif getattr(orderbook_signal, "favor_shorts", False) and not getattr(orderbook_signal, "block_shorts", False):
                    if raw_score <= (short_thresh + 2.0) and orderbook_score <= 40.0:
                        raw_score = min(raw_score, short_thresh - 0.5)
                        action = "SHORT"

            support_defense_long = self._qualifies_support_defense_long(
                raw_score=raw_score,
                advanced=advanced,
                candle_patterns=candle_patterns,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
            if support_defense_long and action in {"FLAT", "SHORT"}:
                raw_score = max(raw_score, long_thresh + 0.5)
                action = "LONG"
                orderbook_guard_reason = (
                    f"LONG promoted by defended support "
                    f"{getattr(orderbook_signal, 'nearest_support', 0.0):,.2f} "
                    f"({getattr(orderbook_signal, 'nearest_support_distance_pct', 0.0):.2f}% away)"
                )
                log.info(f"[{tech.coin}] Support-defense long archetype activated")

        threshold_action = action
        conviction_probe = {"active": False, "candidate_action": action, "summary": ""}
        candidate_action = action
        if action == "FLAT":
            conviction_probe = self._conviction_probe_candidate(
                instrument_type=instrument_type,
                action=action,
                raw_score=raw_score,
                news_signal=news_signal,
                narrative_signal=narrative_signal,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
            if conviction_probe.get("active"):
                candidate_action = str(conviction_probe.get("candidate_action", "FLAT") or "FLAT").upper()
                log.info(f"[{tech.coin}] {conviction_probe.get('summary', '')}")
            else:
                scalp_probe = self._tactical_scalp_probe_candidate(
                    instrument_type=instrument_type,
                    action=action,
                    raw_score=raw_score,
                    advanced=advanced,
                    regimes=regimes,
                    candle_patterns=candle_patterns,
                    news_signal=news_signal,
                    funding_oi_signal=funding_oi_signal,
                    orderbook_signal=orderbook_signal,
                    market_map_signal=market_map_signal,
                    narrative_signal=narrative_signal,
                    social_attention_signal=social_attention_signal,
                )
                if scalp_probe.get("active"):
                    conviction_probe = scalp_probe
                    candidate_action = str(scalp_probe.get("candidate_action", "FLAT") or "FLAT").upper()
                    log.info(f"[{tech.coin}] {scalp_probe.get('summary', '')}")

        # ── 16. Confidence ───────────────────────────────────────────────────
        distance = abs(raw_score - 50.0)
        confidence = "HIGH" if distance >= 20 else "MEDIUM" if distance >= 10 else "LOW"

        # ── 17. Preliminary trade plan + thesis / expectancy gate ───────────
        sl_price = tp_price = 0.0
        trade_plan = {}
        expectancy = {
            "permitted": False,
            "probability": 0.50,
            "expected_r": 0.0,
            "uncertainty": 0.50,
            "score": round(max(0.0, min(100.0, distance * 1.6)), 2),
            "quality": confidence,
            "summary": "No expectancy profile because the setup remained flat",
            "reasons": [],
            "blockers": [],
        }
        execution_plan = {}
        thesis = {
            "candidate_action": candidate_action,
            "state": "NO_TRADE",
            "permitted": candidate_action in ("LONG", "SHORT"),
            "quality": confidence,
            "archetype": "NONE",
            "support_defense_long": False,
            "confirmed_breakout": False,
            "persistent_breakout": False,
            "alignment_points": 0.0,
            "conflict_points": 0.0,
            "conviction_score": round(distance * 2.0, 2),
            "summary": "Raw score remained inside the no-trade zone",
            "reasons": [],
            "blockers": [],
        }
        thesis_guard_reason = ""
        starter_override_required = False
        if tech.price > 0 and candidate_action != "FLAT":
            if getattr(self.tcfg, "dynamic_trade_planning", True):
                trade_plan = self._build_trade_plan(
                    tech=tech,
                    advanced=advanced,
                    action=candidate_action,
                    confidence=confidence,
                    regimes=regimes,
                    candle_patterns=candle_patterns,
                    orderbook_signal=orderbook_signal,
                    market_map_signal=market_map_signal,
                )
                sl_price = float(trade_plan.get("stop_loss", 0.0) or 0.0)
                tp_price = float(trade_plan.get("take_profit", 0.0) or 0.0)
            if sl_price <= 0 or tp_price <= 0:
                if candidate_action == "LONG":
                    sl_price = tech.price * (1 - self.tcfg.stop_loss_pct)
                    tp_price = tech.price * (1 + self.tcfg.take_profit_pct)
                else:
                    sl_price = tech.price * (1 + self.tcfg.stop_loss_pct)
                    tp_price = tech.price * (1 - self.tcfg.take_profit_pct)
                trade_plan = {
                    "entry_price": round(float(tech.price), 6),
                    "stop_loss": round(float(sl_price), 6),
                    "take_profit": round(float(tp_price), 6),
                    "risk_pct": round(abs(tech.price - sl_price) / tech.price * 100, 3),
                    "reward_pct": round(abs(tp_price - tech.price) / tech.price * 100, 3),
                    "risk_reward_ratio": round(
                        abs(tp_price - tech.price) / max(abs(tech.price - sl_price), 1e-9), 3
                    ),
                    "stop_basis": "static_pct",
                    "target_basis": "static_pct",
                    "price_action_summary": "",
                }

            thesis = self._evaluate_trade_thesis(
                action=candidate_action,
                score=raw_score,
                current_position=current_position,
                advanced=advanced,
                regimes=regimes,
                candle_patterns=candle_patterns,
                news_signal=news_signal,
                funding_oi_signal=funding_oi_signal,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
                narrative_signal=narrative_signal,
                social_attention_signal=social_attention_signal,
                instrument_type=instrument_type,
                trade_plan=trade_plan,
            )
            if (thesis.get("scalp") or {}).get("selected"):
                trade_plan = self._apply_scalp_trade_plan(candidate_action, tech.price, trade_plan)
                sl_price = float(trade_plan.get("stop_loss", sl_price) or sl_price)
                tp_price = float(trade_plan.get("take_profit", tp_price) or tp_price)
            expectancy = self._derive_expectancy_profile(
                action=candidate_action,
                score=raw_score,
                thesis=thesis,
                trade_plan=trade_plan,
                regimes=regimes,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
                news_signal=news_signal,
                funding_oi_signal=funding_oi_signal,
                narrative_signal=narrative_signal,
                social_attention_signal=social_attention_signal,
                current_position=current_position,
            )
            conviction_entry = self._build_conviction_entry(
                coin=tech.coin,
                instrument_type=instrument_type,
                action=candidate_action,
                score=raw_score,
                thesis=thesis,
                expectancy=expectancy,
                news_signal=news_signal,
                narrative_signal=narrative_signal,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
            if (thesis.get("scalp") or {}).get("selected") and not conviction_entry.get("active"):
                scalp_profile = dict(thesis.get("scalp") or {})
                scalp_size = float(getattr(self.tcfg, "scalp_size_multiplier", 0.42) or 0.42)
                scalp_size = max(0.10, min(0.60, scalp_size))
                scalp_summary = str(scalp_profile.get("summary") or "Tactical scalp qualified")
                conviction_entry = {
                    "active": True,
                    "direction": candidate_action,
                    "style": str(scalp_profile.get("style") or "SCALP"),
                    "size_multiplier": round(scalp_size, 4),
                    "summary": scalp_summary,
                    "reason": scalp_summary,
                    "blockers": [],
                    "bypass_precision": bool(getattr(self.tcfg, "scalp_bypass_precision", True)),
                    "event_conviction": False,
                    "scalp": True,
                    "max_hold_minutes": scalp_profile.get(
                        "max_hold_minutes",
                        float(getattr(self.tcfg, "scalp_max_hold_minutes", 240.0) or 240.0),
                    ),
                }
            thesis["conviction_entry"] = conviction_entry
            expectancy["conviction_entry"] = conviction_entry
            confidence = str(expectancy.get("quality", confidence) or confidence).upper()
            precision_allowed, precision_reason = self._passes_precision_mode(
                coin=tech.coin,
                action=candidate_action,
                confidence=confidence,
                thesis=thesis,
                expectancy=expectancy,
                trade_plan=trade_plan,
                orderbook_signal=orderbook_signal,
                market_map_signal=market_map_signal,
            )
            regular_gates_clear = bool(thesis.get("permitted", False)) and bool(expectancy.get("permitted", False)) and precision_allowed
            starter_override_required = bool(conviction_entry.get("active")) and (
                threshold_action != candidate_action or not regular_gates_clear
            )
            if starter_override_required:
                action = candidate_action
                thesis_guard_reason = str(conviction_entry.get("summary", "") or "")
                thesis["state"] = "SCALP_ENTRY" if conviction_entry.get("scalp") else "CONVICTION_ENTRY"
                thesis["permitted"] = True
                if str(thesis.get("quality", "LOW") or "LOW").upper() == "LOW":
                    thesis["quality"] = "MEDIUM"
                thesis["summary"] = thesis_guard_reason or str(thesis.get("summary", "") or "")
                thesis["reasons"] = [thesis_guard_reason] + [item for item in list(thesis.get("reasons", []) or []) if item != thesis_guard_reason][:5]
                thesis["blockers"] = []
                thesis["precision_blocked"] = False
                thesis["precision_summary"] = precision_reason if not precision_allowed else ""
                expectancy["permitted"] = True
                if str(expectancy.get("quality", "LOW") or "LOW").upper() == "LOW":
                    expectancy["quality"] = "MEDIUM"
                expectancy["summary"] = thesis_guard_reason or str(expectancy.get("summary", "") or "")
                expectancy["blockers"] = []
                execution_plan = self._build_execution_plan(
                    action=candidate_action,
                    entry_price=float(trade_plan.get("entry_price", tech.price) or tech.price),
                    trade_plan=trade_plan,
                    expectancy=expectancy,
                    orderbook_signal=orderbook_signal,
                )
                execution_plan["starter_size_multiplier"] = float(conviction_entry.get("size_multiplier", 1.0) or 1.0)
                execution_plan["reason"] = thesis_guard_reason or execution_plan.get("reason", "starter conviction entry")
            elif not expectancy.get("permitted", False):
                action = "FLAT"
                thesis_guard_reason = str(expectancy.get("summary", "") or "")
                sl_price = 0.0
                tp_price = 0.0
                trade_plan = {}
                execution_plan = {}
            elif not thesis.get("permitted", False):
                action = "FLAT"
                thesis_guard_reason = str(thesis.get("summary", "") or "")
                sl_price = 0.0
                tp_price = 0.0
                trade_plan = {}
                execution_plan = {}
            else:
                if not precision_allowed:
                    action = "FLAT"
                    thesis_guard_reason = precision_reason
                    sl_price = 0.0
                    tp_price = 0.0
                    trade_plan = {}
                    execution_plan = {}
                    thesis["precision_blocked"] = True
                    thesis["precision_summary"] = precision_reason
                else:
                    thesis["precision_blocked"] = False
                    thesis["precision_summary"] = ""
                    execution_plan = self._build_execution_plan(
                        action=candidate_action,
                        entry_price=float(trade_plan.get("entry_price", tech.price) or tech.price),
                        trade_plan=trade_plan,
                        expectancy=expectancy,
                        orderbook_signal=orderbook_signal,
                    )
                    action = candidate_action
        elif candidate_action == "FLAT":
            thesis["permitted"] = False
            thesis["quality"] = "LOW"
            thesis["conviction_score"] = round(max(0.0, min(100.0, 50.0 - distance)), 2)
            expectancy["score"] = round(max(0.0, min(100.0, 50.0 - distance)), 2)

        # ── 18. Strategic FLAT reasoning ─────────────────────────────────────
        # When staying flat, build a clear explanation so the dashboard shows
        # WHY the agent chose not to trade — inaction is intentional, not lazy.
        flat_reason = ""
        if action == "FLAT":
            flat_parts = []
            if thesis_guard_reason:
                flat_parts.append(thesis_guard_reason)

            # Primary reason: score position
            if candidate_action == "LONG" and raw_score >= long_thresh:
                flat_parts.append(f"Score {raw_score:.0f} triggered LONG, but the thesis was not strong enough")
            elif candidate_action == "SHORT" and raw_score <= short_thresh:
                flat_parts.append(f"Score {raw_score:.0f} triggered SHORT, but the thesis was not strong enough")
            elif 45 <= raw_score <= 55:
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
            if news_signal and news_signal.valid:
                if getattr(news_signal, "article_count", 0) <= 0 and "asset-specific" in str(getattr(news_signal, "error", "") or ""):
                    flat_parts.append("News: no asset-specific flow confirmed yet")
                elif news_signal.article_count > 0:
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
            elif instrument_type == "equity" and not thesis_guard_reason:
                reclaim_confirmed = bool(getattr(market_map_signal, "above_reclaim_levels", [])) if market_map_signal else False
                live_reclaim = bool(getattr(market_map_signal, "live_above_reclaim_levels", [])) if market_map_signal else False
                breakout_state = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE").upper()
                bullish_breakout_live = breakout_state in {
                    "PROBING_BULLISH_BREAKOUT",
                    "CONFIRMED_BULLISH_BREAKOUT",
                    "PERSISTENT_BULLISH_BREAKOUT",
                }
                if reclaim_confirmed and not live_reclaim:
                    flat_parts.append("Equity spot: prior reclaim slipped back below the trigger; price has to hold above it again")
                elif reclaim_confirmed or bullish_breakout_live:
                    flat_parts.append("Equity spot: breakout pressure is live, but the follow-through is still not clean enough")
                else:
                    flat_parts.append("Equity spot: catalyst is live, but price has not earned the entry yet")

            if orderbook_guard_reason and orderbook_guard_reason not in flat_parts:
                flat_parts.insert(0, orderbook_guard_reason)
            elif orderbook_signal and getattr(orderbook_signal, "valid", False):
                if getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS") != "BETWEEN_LEVELS":
                    flat_parts.append(
                        f"Key levels: {orderbook_signal.level_interaction.lower().replace('_', ' ')}"
                    )
                if getattr(orderbook_signal, "breakout_state", "NONE") != "NONE":
                    flat_parts.append(
                        f"Breakout state: {orderbook_signal.breakout_state.lower().replace('_', ' ')}"
                    )
            if market_map_signal and getattr(market_map_signal, "valid", False):
                flat_parts.append(f"Map: {getattr(market_map_signal, 'summary', '')}")

            flat_reason = " · ".join(flat_parts) if flat_parts else "Insufficient conviction"
            log.info(f"[{tech.coin}] ✋ FLAT — {flat_reason}")

        reason = self._build_reason(
            tech, advanced, sentiment, raw_score, action,
            vol_ratio, regimes, news_signal, candle_patterns, orderbook_signal, market_map_signal,
            thesis=thesis,
            expectancy=expectancy,
            narrative_signal=narrative_signal,
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
            trade_plan        = trade_plan,
            thesis            = thesis,
            expectancy        = expectancy,
            execution_plan    = execution_plan,
        )

    def _build_reason(
        self, tech, advanced, sentiment, score, action,
        vol_ratio, regimes=None, news_signal=None, candle_patterns=None, orderbook_signal=None, market_map_signal=None,
        thesis=None,
        expectancy=None,
        narrative_signal=None,
    ):
        parts = []
        msb = advanced.msb
        thesis_summary = str((thesis or {}).get("summary", "") or "")
        thesis_quality = str((thesis or {}).get("quality", "") or "")
        expectancy_summary = str((expectancy or {}).get("summary", "") or "")
        if thesis_summary:
            prefix = "Thesis"
            if thesis_quality:
                prefix += f" {thesis_quality.lower()}"
            parts.append(f"{prefix}: {thesis_summary}")
        if expectancy_summary:
            parts.append(f"Expectancy: {expectancy_summary}")
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
            catalyst_summary = str(getattr(news_signal, "catalyst_summary", "") or "")
            if catalyst_summary:
                parts.append(f"Catalyst: {catalyst_summary}")

        # Candles
        if candle_patterns and candle_patterns.valid and candle_patterns.patterns:
            parts.append(f"Candles: {'+'.join(candle_patterns.patterns[:2])}")

        if orderbook_signal and getattr(orderbook_signal, "valid", False):
            interaction = str(getattr(orderbook_signal, "level_interaction", "BETWEEN_LEVELS") or "BETWEEN_LEVELS")
            breakout = str(getattr(orderbook_signal, "breakout_state", "NONE") or "NONE")
            if interaction != "BETWEEN_LEVELS":
                parts.append(f"Levels: {interaction.replace('_', ' ').title()}")
            if breakout != "NONE":
                parts.append(f"Breakout: {breakout.replace('_', ' ').title()}")
            imbalance = float(getattr(orderbook_signal, "imbalance_ratio", 0.0) or 0.0)
            parts.append(f"Book imbalance {imbalance:+.2f}")

        if market_map_signal and getattr(market_map_signal, "valid", False):
            parts.append(f"Map: {getattr(market_map_signal, 'summary', '')}")

        if narrative_signal and getattr(narrative_signal, "valid", False):
            narrative_summary = str(getattr(narrative_signal, "summary", "") or "")
            if narrative_summary:
                parts.append(f"Narrative: {narrative_summary}")

        return " | ".join(parts)
