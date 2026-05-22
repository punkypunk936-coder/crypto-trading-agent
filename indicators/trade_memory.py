"""
indicators/trade_memory.py — Trade introspection & reinforcement learning.

The agent reviews its own closed trades in detail, learns which conditions
produced wins vs losses, and proactively adjusts future behaviour:

  WHAT IT LEARNS
  ──────────────
  • Win/loss rate by coin + direction
  • Which market REGIMES are profitable (TRENDING vs RANGING)
  • Which dominant REGIMES work (MOMENTUM, ABSORPTION, MIXED, TREND)
  • Which signal SCORE bands have edge (65-70 vs 80+)
  • Time-of-day patterns (UTC hour buckets)
  • Entry failure modes (counter-trend, weak signal, too fast)

  WHAT IT DOES WITH THE LEARNING
  ────────────────────────────────
  • get_score_adjustment(): dampens or boosts raw signal score
  • get_pattern_boost():    tells risk manager to size up/down
  • Cooldown after 3 consecutive losses (suppresses that coin)
  • Persists all data to JSON across restarts
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from logger import get_logger

log = get_logger("trade_memory")

from paths import TRADE_MEMORY as MEMORY_FILE
MAX_TRADES_PER_COIN    = 50    # remember more history per coin
COOLDOWN_CYCLES        = 6    # 6-cycle cooldown after consecutive losses (was 4)
DIRECTION_PAUSE_CYCLES = 4    # short pause for one coin+direction after repeated bad setups
DIRECTION_EMBARGO_MIN_TRADES = 4
DIRECTION_EMBARGO_MAX_WIN_RATE = 0.25
MIN_HOLD_THRESHOLD     = 180  # flag reversals faster: 3h not 4h (was 240)
CONSECUTIVE_LOSS_LIMIT = 3    # still triggers at 3 losses in a row


@dataclass
class TradeRecord:
    coin: str
    direction: str
    signal_score: float
    entry_price: float
    exit_price: float
    pnl_pct: float
    exit_reason: str
    hold_minutes: float
    trend_context: str
    timestamp: float
    market_regime: str       = "RANGING"
    dominant_regime: str     = "MIXED"
    hour_utc: int            = 0
    volatility_label: str    = "NORMAL"
    failure_modes: List[str] = field(default_factory=list)
    root_causes: List[str]   = field(default_factory=list)
    failure_summary: str     = ""
    entry_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_win(self) -> bool:
        return self.pnl_pct > 0

    @property
    def score_bucket(self) -> str:
        c = abs(self.signal_score - 50.0)
        if c >= 35: return "EXTREME(90+)"
        if c >= 25: return "HIGH(80-89)"
        if c >= 15: return "MEDIUM(70-79)"
        return "BASE(60-69)"


class TradeMemory:
    """Persistent trade history with proactive reinforcement-based adjustments."""

    def __init__(self):
        self._trades: Dict[str, List[TradeRecord]] = defaultdict(list)
        self._cooldown: Dict[str, int] = {}
        self._directional_pause: Dict[str, int] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_trade(
        self,
        coin: str,
        direction: str,
        signal_score: float,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        hold_minutes: float        = 0.0,
        trend_context: str         = "FLAT",
        market_regime: str         = "RANGING",
        dominant_regime: str       = "MIXED",
        volatility_label: str      = "NORMAL",
        entry_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a closed trade with full context, then run post-mortem."""
        if entry_price <= 0:
            return

        pnl_pct = (
            (exit_price - entry_price) / entry_price * 100
            if direction == "LONG"
            else (entry_price - exit_price) / entry_price * 100
        )

        failure_analysis = self._analyse_failure(
            direction, signal_score, pnl_pct, exit_reason,
            hold_minutes, trend_context, market_regime, dominant_regime,
            volatility_label=volatility_label,
            entry_context=entry_context or {},
        )

        record = TradeRecord(
            coin            = coin,
            direction       = direction,
            signal_score    = signal_score,
            entry_price     = entry_price,
            exit_price      = exit_price,
            pnl_pct         = round(pnl_pct, 3),
            exit_reason     = exit_reason,
            hold_minutes    = round(hold_minutes, 1),
            trend_context   = trend_context,
            timestamp       = time.time(),
            market_regime   = market_regime,
            dominant_regime = dominant_regime,
            hour_utc        = datetime.now(timezone.utc).hour,
            volatility_label= volatility_label,
            failure_modes   = failure_analysis["failure_modes"],
            root_causes     = failure_analysis["root_causes"],
            failure_summary = failure_analysis["summary"],
            entry_context   = dict(entry_context or {}),
        )

        self._trades[coin].append(record)
        if len(self._trades[coin]) > MAX_TRADES_PER_COIN:
            self._trades[coin] = self._trades[coin][-MAX_TRADES_PER_COIN:]

        self._maybe_trigger_cooldown(coin)
        self._maybe_trigger_direction_pause(coin, direction)

        result_str = "WIN  ✅" if record.is_win else f"LOSS ❌ {record.failure_summary or record.failure_modes}"
        log.info(
            f"[{coin}] Brain: {direction} {pnl_pct:+.2f}% — {result_str} "
            f"(hold={hold_minutes:.0f}m score={signal_score:.1f} "
            f"regime={market_regime}/{dominant_regime})"
        )

        self._save()
        self._log_pattern_insights(coin)

    def get_score_adjustment(
        self, coin: str, direction: str, signal_score: float,
        market_regime: str = "RANGING", dominant_regime: str = "MIXED",
    ) -> float:
        """
        Returns a score delta (−25 … +15) to overlay on the raw signal score.
        Negative = be more cautious. Positive = confidence boost.

        Deliberately aggressive — the agent should learn hard from its mistakes
        and be rewarded clearly for its wins. Weak adjustments = no learning.
        """
        adj = 0.0

        if self._cooldown.get(coin, 0) > 0:
            adj -= 20.0   # was -15 — make cooldown really sting
            log.info(f"[{coin}] Cooldown active ({self._cooldown[coin]}c) adj={adj:+.0f}")

        recent = self._trades.get(coin, [])
        if not recent:
            return round(adj, 2)

        same_dir = [t for t in recent[-15:] if t.direction == direction]  # look at more history
        if not same_dir:
            return round(adj, 2)

        # ── Consecutive loss penalty (checks last 2 as well for faster response) ──
        last2 = same_dir[-2:]
        last3 = same_dir[-3:]
        if len(last2) >= 2 and all(not t.is_win for t in last2):
            adj -= 6.0    # two in a row: early warning
        if len(last3) >= 3 and all(not t.is_win for t in last3):
            adj -= 15.0   # was -10 — 3 in a row should really suppress this setup

        # ── Win-rate in same score band ──────────────────────────────────────
        score_lo, score_hi = signal_score - 8, signal_score + 8
        similar = [t for t in same_dir if score_lo <= t.signal_score <= score_hi]
        if len(similar) >= 2:   # was 3 — react sooner
            wr = sum(1 for t in similar if t.is_win) / len(similar)
            if wr < 0.30:
                adj -= 10.0   # was -7
            elif wr < 0.45:
                adj -= 5.0    # new: moderate penalty for below-average band
            elif wr >= 0.65:
                adj += 8.0    # was +5
            elif wr >= 0.55:
                adj += 4.0    # new: reward even modestly profitable bands

        # ── Regime-specific learning ──────────────────────────────────────────
        regime_trades = [t for t in same_dir if t.market_regime == market_regime]
        if len(regime_trades) >= 2:   # was 3
            rwr = sum(1 for t in regime_trades if t.is_win) / len(regime_trades)
            if rwr < 0.30:
                adj -= 10.0   # was -5 — bad regime should really hurt
            elif rwr < 0.45:
                adj -= 5.0
            elif rwr >= 0.70:
                adj += 6.0    # was +3
            elif rwr >= 0.55:
                adj += 3.0

        # ── Dominant regime ───────────────────────────────────────────────────
        dom_trades = [t for t in same_dir if t.dominant_regime == dominant_regime]
        if len(dom_trades) >= 2:
            dwr = sum(1 for t in dom_trades if t.is_win) / len(dom_trades)
            if dwr < 0.30:
                adj -= 8.0    # was -4
            elif dwr < 0.45:
                adj -= 4.0
            elif dwr >= 0.70:
                adj += 5.0    # was +3
            elif dwr >= 0.55:
                adj += 2.0

        # ── Failure mode recurrence penalties ─────────────────────────────────
        recent_modes = [m for t in same_dir[-4:] for m in t.failure_modes]
        if recent_modes.count("REVERSED_TOO_FAST") >= 2:
            adj -= 6.0    # was -4
        if recent_modes.count("REVERSED_TOO_FAST") >= 3:
            adj -= 4.0    # extra layer for persistent reversals
        if recent_modes.count("COUNTER_TREND") >= 2:
            adj -= 8.0    # was -5 — fighting trend is expensive
        if recent_modes.count("RANGING_TRAP") >= 2:
            adj -= 9.0    # was -6
        if recent_modes.count("ABSORPTION_TRAP") >= 2:
            adj -= 7.0    # new: absorptions are brutal
        if recent_modes.count("WEAK_SIGNAL") >= 3:
            adj -= 5.0    # new: consistently entering on weak signals
        if recent_modes.count("LOW_CONFIDENCE_ENTRY") >= 2:
            adj -= 4.0
        if recent_modes.count("HTF_CONFLICT") >= 2:
            adj -= 6.0
        if recent_modes.count("NEWS_CONFLICT") >= 2:
            adj -= 4.0
        if recent_modes.count("ORDER_FLOW_CONFLICT") >= 2:
            adj -= 5.0
        if recent_modes.count("ENTRY_TIMING_POOR") >= 2:
            adj -= 4.0
        if recent_modes.count("VOLATILITY_SHAKEOUT") >= 2:
            adj -= 3.0

        # ── Recent run bonus / penalty ────────────────────────────────────────
        wins_last5 = sum(1 for t in same_dir[-5:] if t.is_win)
        if wins_last5 >= 4:
            adj += 10.0   # was +6 — hot streak deserves a real boost
        elif wins_last5 >= 3:
            adj += 5.0    # new: solid run
        elif wins_last5 == 0 and len(same_dir) >= 3:
            adj -= 8.0    # was -5 — 0-for-5 should really dampen this
        elif wins_last5 <= 1 and len(same_dir) >= 4:
            adj -= 4.0    # new: 1-for-5 also worrying

        adj_final = round(max(-25.0, min(15.0, adj)), 2)   # wider range: was -18/+10
        if adj_final != 0:
            log.info(f"[{coin}] RL score adjustment: {direction} adj={adj_final:+.1f} "
                     f"(regime={market_regime}/{dominant_regime})")
        return adj_final

    def get_pattern_boost(
        self, coin: str, direction: str, signal_score: float,
        market_regime: str = "TRENDING", dominant_regime: str = "MOMENTUM",
    ) -> float:
        """
        Sizing multiplier boost (0.0 … 0.45) for the risk manager.
        Rewards historically winning pattern contexts aggressively.
        0.15 = 15% larger position. "Double down hard on what works."
        """
        boost = 0.0
        recent = self._trades.get(coin, [])
        if len(recent) < 3:   # was 4 — reward good patterns sooner
            return 0.0

        same_dir = [t for t in recent[-15:] if t.direction == direction]
        if len(same_dir) < 2:   # was 3 — unlock boost sooner
            return 0.0

        # ── Does this exact context have a strong track record? ──────────────
        matching = [
            t for t in same_dir
            if t.market_regime == market_regime
            and t.dominant_regime == dominant_regime
            and abs(t.signal_score - signal_score) <= 12  # wider window (was 10)
        ]
        if len(matching) >= 1:   # even 1 matching win in exact context earns small boost
            pattern_wr = sum(1 for t in matching if t.is_win) / len(matching)
            if pattern_wr >= 0.80:
                boost += 0.25   # was 0.20 — very strong pattern match
                log.info(
                    f"[{coin}] 🔥 Pattern boost: {direction} {market_regime}/{dominant_regime} "
                    f"WR={pattern_wr*100:.0f}% ({len(matching)} trades) → +{boost*100:.0f}% size"
                )
            elif pattern_wr >= 0.65:
                boost += 0.15   # was 0.10
                log.info(
                    f"[{coin}] Pattern boost: {direction} {market_regime}/{dominant_regime} "
                    f"WR={pattern_wr*100:.0f}% → +{boost*100:.0f}% size"
                )
            elif pattern_wr >= 0.50:
                boost += 0.07   # new: even a slight edge gets rewarded

        # ── Streak bonus: 3 consecutive same-direction wins ──────────────────
        if len(same_dir) >= 3 and all(t.is_win for t in same_dir[-3:]):
            boost += 0.15   # was 0.10 — hot streak is real signal
            log.info(f"[{coin}] 🔥 Streak bonus: 3 consecutive {direction} wins → +15%")
        elif len(same_dir) >= 2 and all(t.is_win for t in same_dir[-2:]):
            boost += 0.07   # new: 2 in a row is worth noticing

        # ── Overall win rate bonus ────────────────────────────────────────────
        wr = sum(1 for t in same_dir if t.is_win) / len(same_dir)
        if wr >= 0.75 and len(same_dir) >= 4:    # was 5 — unlock sooner
            boost += 0.15   # was 0.10
        elif wr >= 0.60 and len(same_dir) >= 4:
            boost += 0.08   # new: 60% WR also earns a small reward

        return round(min(boost, 0.45), 2)   # was 0.30 — allow up to 45% boost

    def get_directional_guard(self, coin: str, direction: str) -> Dict[str, Any]:
        """
        Return per-coin, per-direction guardrails based on recurring failure patterns.

        threshold_boost:
          extra points required before a setup is allowed.
          LONG uses score >= long_threshold + threshold_boost
          SHORT uses score <= short_threshold - threshold_boost

        pause_cycles:
          temporarily block this exact coin+direction pair after repeated losses
          with overlapping causes.
        """
        recent = self._trades.get(coin, [])
        same_dir = [t for t in recent[-15:] if t.direction == direction]
        if not same_dir:
            return {"threshold_boost": 0.0, "pause_cycles": 0, "reasons": [], "hard_block": False, "hard_block_reason": ""}

        key = self._pause_key(coin, direction)
        threshold_boost = 0.0
        reasons: List[str] = []
        recent_modes = [m for t in same_dir[-4:] for m in t.failure_modes]

        def tighten(boost: float, label: str, mode: str, minimum: int = 2) -> None:
            nonlocal threshold_boost
            if recent_modes.count(mode) >= minimum:
                threshold_boost += boost
                reasons.append(label)

        tighten(6.0, "HTF conflict", "HTF_CONFLICT")
        tighten(5.0, "counter-trend entries", "COUNTER_TREND")
        tighten(4.0, "low-confidence entries", "LOW_CONFIDENCE_ENTRY")
        tighten(4.0, "order-flow conflict", "ORDER_FLOW_CONFLICT")
        tighten(4.0, "ranging trap", "RANGING_TRAP")
        tighten(4.0, "absorption trap", "ABSORPTION_TRAP")
        tighten(3.0, "news conflict", "NEWS_CONFLICT")
        tighten(3.0, "poor timing", "ENTRY_TIMING_POOR")
        tighten(3.0, "candle conflict", "CANDLE_CONFLICT")
        tighten(3.0, "marginal edge", "MARGINAL_EDGE")
        tighten(4.0, "low expectancy", "LOW_EXPECTANCY")
        tighten(4.0, "high uncertainty", "HIGH_UNCERTAINTY")
        tighten(5.0, "narrative event risk", "NARRATIVE_EVENT_RISK")

        pause_cycles = self._directional_pause.get(key, 0)
        same_dir_wr = sum(1 for t in same_dir if t.is_win) / len(same_dir) if same_dir else 0.0
        recent_three_losses = len(same_dir) >= 3 and all(not t.is_win for t in same_dir[-3:])
        hard_block = (
            len(same_dir) >= DIRECTION_EMBARGO_MIN_TRADES
            and same_dir_wr <= DIRECTION_EMBARGO_MAX_WIN_RATE
            and recent_three_losses
        )
        hard_block_reason = ""
        if hard_block:
            hard_block_reason = (
                f"embargoed after {len(same_dir)} {direction} trades with "
                f"{same_dir_wr*100:.0f}% win rate"
            )
        return {
            "threshold_boost": round(min(threshold_boost, 12.0), 2),
            "pause_cycles": pause_cycles,
            "reasons": reasons[:3],
            "hard_block": hard_block,
            "hard_block_reason": hard_block_reason,
        }

    def tick_cooldowns(self) -> None:
        expired = []
        for coin, remaining in self._cooldown.items():
            if remaining <= 1:
                expired.append(coin)
                log.info(f"[{coin}] Cooldown expired — normal trading resumed")
            else:
                self._cooldown[coin] = remaining - 1
        for coin in expired:
            del self._cooldown[coin]
        pause_expired = []
        for key, remaining in self._directional_pause.items():
            if remaining <= 1:
                pause_expired.append(key)
                log.info(f"[{key}] Directional pause expired — setup eligible again")
            else:
                self._directional_pause[key] = remaining - 1
        for key in pause_expired:
            del self._directional_pause[key]

    def summary(self) -> str:
        parts = []
        for coin, trades in self._trades.items():
            wins  = sum(1 for t in trades if t.is_win)
            total = len(trades)
            avg   = sum(t.pnl_pct for t in trades) / total if total else 0
            cd    = f" COOLDOWN({self._cooldown[coin]})" if coin in self._cooldown else ""
            directional = [
                f"{key.split(':', 1)[1]}-PAUSE({remaining})"
                for key, remaining in self._directional_pause.items()
                if key.startswith(f"{coin}:")
            ]
            dp = f" {' '.join(directional)}" if directional else ""
            parts.append(f"{coin}: {wins}/{total} wins avg={avg:+.2f}%{cd}{dp}")
        return "Memory: " + " | ".join(parts) if parts else "Memory: no history yet"

    def get_stats(self) -> dict:
        result = {}
        for coin, trades in self._trades.items():
            wins  = sum(1 for t in trades if t.is_win)
            total = len(trades)
            if total == 0:
                continue

            # Regime breakdown
            regime_stats: Dict[str, dict] = {}
            for t in trades:
                key = f"{t.market_regime}/{t.dominant_regime}"
                if key not in regime_stats:
                    regime_stats[key] = {"wins": 0, "total": 0}
                regime_stats[key]["total"] += 1
                if t.is_win:
                    regime_stats[key]["wins"] += 1

            best_regime  = max(regime_stats, key=lambda k: regime_stats[k]["wins"] / max(regime_stats[k]["total"],1)) if regime_stats else "N/A"
            worst_regime = min(regime_stats, key=lambda k: regime_stats[k]["wins"] / max(regime_stats[k]["total"],1)) if regime_stats else "N/A"

            all_modes = [m for t in trades for m in t.failure_modes]
            mode_counts: Dict[str, int] = {}
            for m in all_modes:
                mode_counts[m] = mode_counts.get(m, 0) + 1
            all_root_causes = [m for t in trades for m in getattr(t, "root_causes", [])]
            root_cause_counts: Dict[str, int] = {}
            for m in all_root_causes:
                root_cause_counts[m] = root_cause_counts.get(m, 0) + 1
            latest_loss = next((t for t in reversed(trades) if not t.is_win), None)

            band_wins: Dict[str, int]  = {}
            band_total: Dict[str, int] = {}
            for t in trades:
                b = t.score_bucket
                band_total[b] = band_total.get(b, 0) + 1
                if t.is_win:
                    band_wins[b] = band_wins.get(b, 0) + 1
            trigger_trades = [
                t for t in trades
                if bool((t.entry_context or {}).get("trigger_entry"))
                or "trigger" in str((t.entry_context or {}).get("entry_type", "")).lower()
            ]
            trigger_wins = sum(1 for t in trigger_trades if t.is_win)

            result[coin] = {
                "total":        total,
                "wins":         wins,
                "win_rate":     round(wins / total * 100, 1),
                "avg_pnl_pct":  round(sum(t.pnl_pct for t in trades) / total, 2),
                "best_pnl":     round(max(t.pnl_pct for t in trades), 2),
                "worst_pnl":    round(min(t.pnl_pct for t in trades), 2),
                "cooldown":     self._cooldown.get(coin, 0),
                "last_5_results": [("WIN" if t.is_win else "LOSS") for t in trades[-5:]],
                "last_5_pnl":   [round(t.pnl_pct, 2) for t in trades[-5:]],
                "best_regime":  best_regime,
                "worst_regime": worst_regime,
                "regime_stats": {
                    k: {"wr": round(v["wins"]/max(v["total"],1)*100, 0), "n": v["total"]}
                    for k, v in regime_stats.items()
                },
                "failure_modes": mode_counts,
                "root_causes": root_cause_counts,
                "latest_failure_summary": getattr(latest_loss, "failure_summary", ""),
                "directional_pause": {
                    direction: self._directional_pause.get(self._pause_key(coin, direction), 0)
                    for direction in ("LONG", "SHORT")
                },
                "directional_guards": {
                    direction: self.get_directional_guard(coin, direction)
                    for direction in ("LONG", "SHORT")
                },
                "score_band_wr": {
                    b: round(band_wins.get(b, 0) / band_total[b] * 100, 0)
                    for b in band_total
                },
                "trigger_entry": {
                    "total": len(trigger_trades),
                    "wins": trigger_wins,
                    "win_rate": round(trigger_wins / max(len(trigger_trades), 1) * 100, 1),
                    "avg_pnl_pct": round(sum(t.pnl_pct for t in trigger_trades) / max(len(trigger_trades), 1), 2),
                },
                "long_trades":  sum(1 for t in trades if t.direction == "LONG"),
                "short_trades": sum(1 for t in trades if t.direction == "SHORT"),
                "long_wr":      round(sum(1 for t in trades if t.direction == "LONG" and t.is_win) /
                                      max(sum(1 for t in trades if t.direction == "LONG"), 1) * 100, 1),
                "short_wr":     round(sum(1 for t in trades if t.direction == "SHORT" and t.is_win) /
                                      max(sum(1 for t in trades if t.direction == "SHORT"), 1) * 100, 1),
            }
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _analyse_failure(
        self, direction, signal_score, pnl_pct, exit_reason,
        hold_minutes, trend_context, market_regime, dominant_regime,
        volatility_label: str = "NORMAL",
        entry_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if pnl_pct >= 0:
            return {"failure_modes": [], "root_causes": [], "summary": ""}

        ctx = dict(entry_context or {})
        modes: List[str] = []
        root_causes: List[str] = []

        def add(code: str, explanation: str) -> None:
            if code not in modes:
                modes.append(code)
            if explanation and explanation not in root_causes:
                root_causes.append(explanation)

        if exit_reason == "signal_reversal" and hold_minutes < MIN_HOLD_THRESHOLD:
            add("REVERSED_TOO_FAST", "Market reversed too quickly after entry.")
        if direction == "LONG" and trend_context == "DOWN":
            add("COUNTER_TREND", "Higher timeframe trend was against the long.")
        elif direction == "SHORT" and trend_context == "UP":
            add("COUNTER_TREND", "Higher timeframe trend was against the short.")
        if direction == "LONG" and signal_score < 70:
            add("WEAK_SIGNAL", "Entry was taken without strong enough long conviction.")
        elif direction == "SHORT" and signal_score > 30:
            add("WEAK_SIGNAL", "Entry was taken without strong enough short conviction.")
        if exit_reason == "stop_loss":
            add("STOP_HIT", "The stop loss was hit before the move developed.")
        if market_regime == "RANGING":
            add("RANGING_TRAP", "The market was ranging, so follow-through was weak.")
        if dominant_regime == "ABSORPTION":
            add("ABSORPTION_TRAP", "Absorption regime likely trapped directional follow-through.")
        if exit_reason == "conviction_lost":
            add("THESIS_DISSOLVED", "The thesis weakened after entry and conviction faded.")

        if hold_minutes <= 45 and exit_reason in {"stop_loss", "signal_reversal"}:
            add("ENTRY_TIMING_POOR", "Entry timing was poor and invalidated quickly.")

        conviction = str(ctx.get("confidence", "")).upper()
        if conviction == "LOW":
            add("LOW_CONFIDENCE_ENTRY", "The trade was opened on a low-confidence signal.")

        entry_type = str(ctx.get("entry_type", "") or "").lower()
        trigger_context = dict(ctx.get("trigger_watch") or {})
        trigger_entry = bool(ctx.get("trigger_entry")) or "trigger" in entry_type
        if trigger_entry:
            try:
                trigger_chase_pct = float(
                    ctx.get("entry_trigger_chase_pct", trigger_context.get("trigger_chase_pct", 0.0)) or 0.0
                )
            except Exception:
                trigger_chase_pct = 0.0
            add("TRIGGER_ENTRY_FAILED", "The mapped trigger entry did not produce profitable follow-through.")
            if trigger_chase_pct > 1.25:
                add("TRIGGER_CHASED_TOO_FAR", "The entry chased too far above the original trigger.")
            if exit_reason == "stop_loss" and hold_minutes <= 120:
                add("BAD_TRIGGER_LEVEL", "The trigger level failed quickly and needs stricter calibration.")

        try:
            expectancy_score = float(ctx.get("expectancy_score", 50.0))
        except Exception:
            expectancy_score = 50.0
        try:
            expectancy_r = float(ctx.get("expectancy_expected_r", 0.0))
        except Exception:
            expectancy_r = 0.0
        try:
            expectancy_uncertainty = float(ctx.get("expectancy_uncertainty", 0.50))
        except Exception:
            expectancy_uncertainty = 0.50
        if expectancy_score < 56.0:
            add("LOW_EXPECTANCY", "The setup did not clear a strong expectancy threshold.")
        if expectancy_r < 0.15:
            add("MARGINAL_EDGE", "Expected edge was too thin relative to the risk taken.")
        if expectancy_uncertainty >= 0.42:
            add("HIGH_UNCERTAINTY", "The setup carried too much uncertainty into entry.")

        mtf_bias = str(ctx.get("mtf_bias", "")).upper()
        if direction == "LONG" and mtf_bias == "DOWN":
            add("HTF_CONFLICT", "Higher timeframe bias disagreed with the long setup.")
        elif direction == "SHORT" and mtf_bias == "UP":
            add("HTF_CONFLICT", "Higher timeframe bias disagreed with the short setup.")

        try:
            news_score = float(ctx.get("news_score", 50.0))
        except Exception:
            news_score = 50.0
        if direction == "LONG" and news_score < 45:
            add("NEWS_CONFLICT", "Newsflow leaned against the long.")
        elif direction == "SHORT" and news_score > 55:
            add("NEWS_CONFLICT", "Newsflow leaned against the short.")

        candle_trend = str(ctx.get("candle_trend", "")).upper()
        if direction == "LONG" and candle_trend == "DOWN":
            add("CANDLE_CONFLICT", "Short-term candle trend was still down into the entry.")
        elif direction == "SHORT" and candle_trend == "UP":
            add("CANDLE_CONFLICT", "Short-term candle trend was still up into the entry.")

        try:
            foc_score = float(ctx.get("foc_score", 50.0))
        except Exception:
            foc_score = 50.0
        if direction == "LONG" and foc_score < 45:
            add("ORDER_FLOW_CONFLICT", "Funding/OI/CVD context did not confirm the long.")
        elif direction == "SHORT" and foc_score > 55:
            add("ORDER_FLOW_CONFLICT", "Funding/OI/CVD context did not confirm the short.")

        try:
            orderbook_score = float(ctx.get("orderbook_score", 50.0))
        except Exception:
            orderbook_score = 50.0
        orderbook_interaction = str(ctx.get("orderbook_interaction", "")).upper()
        orderbook_breakout = str(ctx.get("orderbook_breakout_state", "")).upper()
        if direction == "LONG" and orderbook_score < 45:
            add("KEY_LEVEL_CONFLICT", "Orderbook/key-level context leaned against the long.")
        elif direction == "SHORT" and orderbook_score > 55:
            add("KEY_LEVEL_CONFLICT", "Orderbook/key-level context leaned against the short.")

        if exit_reason == "stop_loss" and direction == "LONG" and orderbook_interaction in {"AT_RESISTANCE", "BELOW_RESISTANCE"}:
            add("LONGED_INTO_SUPPLY", "The long was opened too close to overhead supply/resistance.")
        elif exit_reason == "stop_loss" and direction == "SHORT" and orderbook_interaction in {"AT_SUPPORT", "ABOVE_SUPPORT"}:
            add("SHORTED_INTO_DEMAND", "The short was opened too close to a defended demand/support zone.")

        if direction == "SHORT" and orderbook_breakout in {"CONFIRMED_BULLISH_BREAKOUT", "PERSISTENT_BULLISH_BREAKOUT"}:
            add("FADED_CONFIRMED_BREAKOUT", "The short faded a confirmed bullish breakout through key resistance.")
        elif direction == "LONG" and orderbook_breakout in {"CONFIRMED_BEARISH_BREAKDOWN", "PERSISTENT_BEARISH_BREAKDOWN"}:
            add("FADED_CONFIRMED_BREAKDOWN", "The long faded a confirmed bearish breakdown through key support.")

        if bool(ctx.get("narrative_event_risk_active", False)):
            add("NARRATIVE_EVENT_RISK", "The trade was opened inside a high-impact narrative event window.")

        try:
            planned_stop_atr = float(ctx.get("planned_stop_atr_multiple", 0.0))
        except Exception:
            planned_stop_atr = 0.0
        try:
            planned_rr = float(ctx.get("planned_risk_reward_ratio", 0.0))
        except Exception:
            planned_rr = 0.0
        try:
            tp_progress = float(ctx.get("tp_progress_ratio", 0.0))
        except Exception:
            tp_progress = 0.0
        try:
            remaining_to_tp_pct = float(ctx.get("remaining_to_tp_pct", 0.0))
        except Exception:
            remaining_to_tp_pct = 0.0
        try:
            remaining_to_sl_pct = float(ctx.get("remaining_to_sl_pct", 0.0))
        except Exception:
            remaining_to_sl_pct = 0.0

        if exit_reason == "stop_loss" and 0 < planned_stop_atr < 1.10:
            add("STOP_TOO_TIGHT", "The planned stop sat inside normal ATR noise for this setup.")
        if exit_reason == "conviction_lost" and tp_progress >= 0.65:
            add("TARGET_TOO_FAR", "Price travelled far enough to validate the thesis, but the target was too ambitious.")
        if exit_reason in {"signal_reversal", "conviction_lost"} and remaining_to_tp_pct > 0 and remaining_to_sl_pct > 0:
            if remaining_to_tp_pct <= max(remaining_to_sl_pct * 0.70, 0.35):
                add("EXITED_NEAR_TARGET", "The trade closed after price had already pushed close to its target.")
        if exit_reason == "stop_loss" and planned_rr >= 2.80:
            add("TARGET_PROFILE_STRETCHED", "The setup aimed for a very stretched reward multiple relative to risk.")

        if str(volatility_label).lower() == "extreme" and exit_reason == "stop_loss":
            add("VOLATILITY_SHAKEOUT", "Extreme volatility likely shook out the trade.")

        if abs(signal_score - 50.0) < 18:
            add("MARGINAL_EDGE", "The setup was too close to neutral to have a durable edge.")

        if not root_causes:
            root_causes.append("The setup lost edge after entry for reasons not yet classified.")

        summary = "; ".join(root_causes[:3])
        return {"failure_modes": modes, "root_causes": root_causes, "summary": summary}

    def _maybe_trigger_cooldown(self, coin: str) -> None:
        recent = self._trades.get(coin, [])
        if len(recent) < CONSECUTIVE_LOSS_LIMIT:
            return
        last_n = recent[-CONSECUTIVE_LOSS_LIMIT:]
        if all(not t.is_win for t in last_n):
            self._cooldown[coin] = COOLDOWN_CYCLES
            log.warning(f"[{coin}] {CONSECUTIVE_LOSS_LIMIT} consecutive losses → COOLDOWN {COOLDOWN_CYCLES} cycles")

    def _maybe_trigger_direction_pause(self, coin: str, direction: str) -> None:
        same_dir = [t for t in self._trades.get(coin, []) if t.direction == direction]
        if len(same_dir) < 2:
            return
        last_two = same_dir[-2:]
        if not all(not t.is_win for t in last_two):
            return
        overlapping_modes = set(last_two[0].failure_modes) & set(last_two[1].failure_modes)
        if not overlapping_modes:
            return
        key = self._pause_key(coin, direction)
        current = self._directional_pause.get(key, 0)
        self._directional_pause[key] = max(current, DIRECTION_PAUSE_CYCLES)
        log.warning(
            f"[{coin}] {direction} paused for {DIRECTION_PAUSE_CYCLES} cycles "
            f"after repeated failure pattern(s): {', '.join(sorted(overlapping_modes))}"
        )

    def _log_pattern_insights(self, coin: str) -> None:
        trades = self._trades.get(coin, [])
        if len(trades) < 4:
            return
        regime_wins: Dict[str, list] = {}
        for t in trades:
            key = f"{t.market_regime}/{t.dominant_regime}"
            regime_wins.setdefault(key, []).append(t.is_win)
        if not regime_wins:
            return
        best  = max(regime_wins.items(), key=lambda x: sum(x[1]) / max(len(x[1]),1))
        worst = min(regime_wins.items(), key=lambda x: sum(x[1]) / max(len(x[1]),1))
        log.info(
            f"[{coin}] Pattern: best={best[0]}({sum(best[1])}/{len(best[1])}) "
            f"worst={worst[0]}({sum(worst[1])}/{len(worst[1])})"
        )

    def _save(self) -> None:
        try:
            data = {
                "trades": {
                    coin: [asdict(t) for t in trades]
                    for coin, trades in self._trades.items()
                },
                "cooldown": self._cooldown,
                "directional_pause": self._directional_pause,
                "saved_at": datetime.utcnow().isoformat(),
            }
            MEMORY_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log.warning(f"Trade memory save failed: {e}")

    def _load(self) -> None:
        if not MEMORY_FILE.exists():
            log.info("Trade memory: starting fresh")
            return
        try:
            data = json.loads(MEMORY_FILE.read_text())
            for coin, records in data.get("trades", {}).items():
                loaded = []
                for r in records:
                    r.setdefault("market_regime",    "RANGING")
                    r.setdefault("dominant_regime",  "MIXED")
                    r.setdefault("hour_utc",         0)
                    r.setdefault("volatility_label", "NORMAL")
                    r.setdefault("root_causes",      [])
                    r.setdefault("failure_summary",  "")
                    r.setdefault("entry_context",    {})
                    if (not r.get("root_causes") or not r.get("failure_summary")) and float(r.get("pnl_pct", 0) or 0) < 0:
                        analysis = self._analyse_failure(
                            direction=r.get("direction", ""),
                            signal_score=float(r.get("signal_score", 50.0) or 50.0),
                            pnl_pct=float(r.get("pnl_pct", 0.0) or 0.0),
                            exit_reason=r.get("exit_reason", ""),
                            hold_minutes=float(r.get("hold_minutes", 0.0) or 0.0),
                            trend_context=r.get("trend_context", "FLAT"),
                            market_regime=r.get("market_regime", "RANGING"),
                            dominant_regime=r.get("dominant_regime", "MIXED"),
                            volatility_label=r.get("volatility_label", "NORMAL"),
                            entry_context=r.get("entry_context", {}) or {},
                        )
                        if not r.get("failure_modes"):
                            r["failure_modes"] = analysis["failure_modes"]
                        if not r.get("root_causes"):
                            r["root_causes"] = analysis["root_causes"]
                        if not r.get("failure_summary"):
                            r["failure_summary"] = analysis["summary"]
                    loaded.append(TradeRecord(**r))
                self._trades[coin] = loaded
            self._cooldown = data.get("cooldown", {})
            self._directional_pause = data.get("directional_pause", {})
            n = sum(len(v) for v in self._trades.values())
            log.info(f"Trade memory: loaded {n} trades across {len(self._trades)} coins")
        except Exception as e:
            log.warning(f"Trade memory load failed: {e} — starting fresh")

    @staticmethod
    def _pause_key(coin: str, direction: str) -> str:
        return f"{coin.upper()}:{direction.upper()}"


trade_memory = TradeMemory()
