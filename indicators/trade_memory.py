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
from typing import List, Optional, Dict
from datetime import datetime, timezone

from logger import get_logger

log = get_logger("trade_memory")

from paths import TRADE_MEMORY as MEMORY_FILE
MAX_TRADES_PER_COIN  = 30
COOLDOWN_CYCLES      = 4
MIN_HOLD_THRESHOLD   = 240     # minutes
CONSECUTIVE_LOSS_LIMIT = 3


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
    ) -> None:
        """Record a closed trade with full context, then run post-mortem."""
        if entry_price <= 0:
            return

        pnl_pct = (
            (exit_price - entry_price) / entry_price * 100
            if direction == "LONG"
            else (entry_price - exit_price) / entry_price * 100
        )

        failure_modes = self._analyse_failure(
            direction, signal_score, pnl_pct, exit_reason,
            hold_minutes, trend_context, market_regime, dominant_regime,
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
            failure_modes   = failure_modes,
        )

        self._trades[coin].append(record)
        if len(self._trades[coin]) > MAX_TRADES_PER_COIN:
            self._trades[coin] = self._trades[coin][-MAX_TRADES_PER_COIN:]

        self._maybe_trigger_cooldown(coin)

        result_str = "WIN  ✅" if record.is_win else f"LOSS ❌ {failure_modes}"
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
        Returns a score delta (−18 … +10) to overlay on the raw signal score.
        Negative = be more cautious. Positive = slight confidence boost.
        """
        adj = 0.0

        if self._cooldown.get(coin, 0) > 0:
            adj -= 15.0
            log.info(f"[{coin}] Cooldown active ({self._cooldown[coin]}c) adj={adj:+.0f}")

        recent = self._trades.get(coin, [])
        if not recent:
            return round(adj, 2)

        same_dir = [t for t in recent[-10:] if t.direction == direction]
        if not same_dir:
            return round(adj, 2)

        # Consecutive loss penalty
        last3 = same_dir[-3:]
        if len(last3) >= 3 and all(not t.is_win for t in last3):
            adj -= 10.0

        # Win-rate in same score band
        score_lo, score_hi = signal_score - 8, signal_score + 8
        similar = [t for t in same_dir if score_lo <= t.signal_score <= score_hi]
        if len(similar) >= 3:
            wr = sum(1 for t in similar if t.is_win) / len(similar)
            if wr < 0.30:
                adj -= 7.0
            elif wr >= 0.65:
                adj += 5.0

        # Regime-specific learning
        regime_trades = [t for t in same_dir if t.market_regime == market_regime]
        if len(regime_trades) >= 3:
            rwr = sum(1 for t in regime_trades if t.is_win) / len(regime_trades)
            if rwr < 0.35:
                adj -= 5.0
            elif rwr >= 0.70:
                adj += 3.0

        # Dominant regime (ABSORPTION kills us)
        dom_trades = [t for t in same_dir if t.dominant_regime == dominant_regime]
        if len(dom_trades) >= 3:
            dwr = sum(1 for t in dom_trades if t.is_win) / len(dom_trades)
            if dwr < 0.35:
                adj -= 4.0
            elif dwr >= 0.70:
                adj += 3.0

        # Failure mode recurrence penalties
        recent_modes = [m for t in same_dir[-4:] for m in t.failure_modes]
        if recent_modes.count("REVERSED_TOO_FAST") >= 2:
            adj -= 4.0
        if recent_modes.count("COUNTER_TREND") >= 2:
            adj -= 5.0
        if recent_modes.count("RANGING_TRAP") >= 2:
            adj -= 6.0

        # Recent run bonus/penalty
        wins_last5 = sum(1 for t in same_dir[-5:] if t.is_win)
        if wins_last5 >= 4:
            adj += 6.0
        elif wins_last5 == 0 and len(same_dir) >= 3:
            adj -= 5.0

        return round(max(-18.0, min(10.0, adj)), 2)

    def get_pattern_boost(
        self, coin: str, direction: str, signal_score: float,
        market_regime: str = "TRENDING", dominant_regime: str = "MOMENTUM",
    ) -> float:
        """
        Sizing multiplier boost (0.0 … 0.30) for the risk manager.
        Rewards historically winning pattern contexts.
        0.10 = 10% larger position. This is "double down on what works".
        """
        boost = 0.0
        recent = self._trades.get(coin, [])
        if len(recent) < 4:
            return 0.0

        same_dir = [t for t in recent[-12:] if t.direction == direction]
        if len(same_dir) < 3:
            return 0.0

        # Does this exact context have a strong track record?
        matching = [
            t for t in same_dir
            if t.market_regime == market_regime
            and t.dominant_regime == dominant_regime
            and abs(t.signal_score - signal_score) <= 10
        ]
        if len(matching) >= 2:
            pattern_wr = sum(1 for t in matching if t.is_win) / len(matching)
            if pattern_wr >= 0.75:
                boost += 0.20
                log.info(
                    f"[{coin}] Pattern boost: {direction} {market_regime}/{dominant_regime} "
                    f"WR={pattern_wr*100:.0f}% → +{boost*100:.0f}% size"
                )
            elif pattern_wr >= 0.60:
                boost += 0.10

        # Streak bonus: 3 consecutive same-direction wins
        if len(same_dir) >= 3 and all(t.is_win for t in same_dir[-3:]):
            boost += 0.10

        # Overall win rate bonus
        wr = sum(1 for t in same_dir if t.is_win) / len(same_dir)
        if wr >= 0.75 and len(same_dir) >= 5:
            boost += 0.10

        return round(min(boost, 0.30), 2)

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

    def summary(self) -> str:
        parts = []
        for coin, trades in self._trades.items():
            wins  = sum(1 for t in trades if t.is_win)
            total = len(trades)
            avg   = sum(t.pnl_pct for t in trades) / total if total else 0
            cd    = f" COOLDOWN({self._cooldown[coin]})" if coin in self._cooldown else ""
            parts.append(f"{coin}: {wins}/{total} wins avg={avg:+.2f}%{cd}")
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

            band_wins: Dict[str, int]  = {}
            band_total: Dict[str, int] = {}
            for t in trades:
                b = t.score_bucket
                band_total[b] = band_total.get(b, 0) + 1
                if t.is_win:
                    band_wins[b] = band_wins.get(b, 0) + 1

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
                "score_band_wr": {
                    b: round(band_wins.get(b, 0) / band_total[b] * 100, 0)
                    for b in band_total
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
    ) -> List[str]:
        if pnl_pct >= 0:
            return []
        modes = []
        if exit_reason == "signal_reversal" and hold_minutes < MIN_HOLD_THRESHOLD:
            modes.append("REVERSED_TOO_FAST")
        if direction == "LONG" and trend_context == "DOWN":
            modes.append("COUNTER_TREND")
        elif direction == "SHORT" and trend_context == "UP":
            modes.append("COUNTER_TREND")
        if direction == "LONG" and signal_score < 70:
            modes.append("WEAK_SIGNAL")
        elif direction == "SHORT" and signal_score > 30:
            modes.append("WEAK_SIGNAL")
        if exit_reason == "stop_loss":
            modes.append("STOP_HIT")
        if market_regime == "RANGING":
            modes.append("RANGING_TRAP")
        if dominant_regime == "ABSORPTION":
            modes.append("ABSORPTION_TRAP")
        if exit_reason == "conviction_lost":
            modes.append("THESIS_DISSOLVED")
        return modes

    def _maybe_trigger_cooldown(self, coin: str) -> None:
        recent = self._trades.get(coin, [])
        if len(recent) < CONSECUTIVE_LOSS_LIMIT:
            return
        last_n = recent[-CONSECUTIVE_LOSS_LIMIT:]
        if all(not t.is_win for t in last_n):
            self._cooldown[coin] = COOLDOWN_CYCLES
            log.warning(f"[{coin}] {CONSECUTIVE_LOSS_LIMIT} consecutive losses → COOLDOWN {COOLDOWN_CYCLES} cycles")

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
                    loaded.append(TradeRecord(**r))
                self._trades[coin] = loaded
            self._cooldown = data.get("cooldown", {})
            n = sum(len(v) for v in self._trades.values())
            log.info(f"Trade memory: loaded {n} trades across {len(self._trades)} coins")
        except Exception as e:
            log.warning(f"Trade memory load failed: {e} — starting fresh")


trade_memory = TradeMemory()
