"""
analog_engine.py — historical analog retrieval for live trade context.

This is the first practical AI layer in the stack:
  • retrieve the closest historical executed trades
  • estimate whether the current setup resembles past winners or losers
  • feed a conservative probability / expectancy adjustment back into the agent
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import feature_store
import trade_dataset
from logger import get_logger
from paths import TRADE_DATASET_JSONL

log = get_logger("analog_engine")

NUMERIC_KEYS = (
    "score",
    "candle_score",
    "news_score",
    "memory_adj",
    "rl_pattern_boost",
    "atr_pct",
    "foc_score",
    "orderbook_score",
    "orderbook_imbalance",
    "orderbook_imbalance_mean",
    "orderbook_imbalance_trend",
    "orderbook_imbalance_volatility",
    "orderbook_support_distance_pct",
    "orderbook_resistance_distance_pct",
    "orderbook_support_strength",
    "orderbook_resistance_strength",
    "orderbook_support_wall_persistence",
    "orderbook_resistance_wall_persistence",
    "market_map_score_adjustment",
    "planned_risk_pct",
    "planned_reward_pct",
    "planned_risk_reward_ratio",
    "planned_stop_atr_multiple",
    "planned_target_atr_multiple",
    "planned_target_r_multiple",
    "thesis_alignment_points",
    "thesis_conflict_points",
    "thesis_conviction_score",
    "expectancy_probability",
    "expectancy_expected_r",
    "expectancy_uncertainty",
    "expectancy_score",
    "execution_quality_score",
    "estimated_slippage_bps",
)

CATEGORICAL_KEYS = (
    "instrument_type",
    "candle_trend",
    "news_velocity",
    "market_regime",
    "dominant_regime",
    "volatility_label",
    "msb_type",
    "structure_trend",
    "funding_label",
    "cvd_divergence",
    "orderbook_interaction",
    "orderbook_breakout_state",
    "market_map_bias",
    "stop_basis",
    "target_basis",
    "thesis_candidate_action",
    "thesis_state",
    "thesis_quality",
    "execution_mode",
)

SCALE_HINTS = {
    "score": 100.0,
    "candle_score": 100.0,
    "news_score": 100.0,
    "rl_pattern_boost": 20.0,
    "memory_adj": 20.0,
    "atr_pct": 10.0,
    "foc_score": 100.0,
    "orderbook_score": 100.0,
    "orderbook_imbalance": 1.0,
    "orderbook_imbalance_mean": 1.0,
    "orderbook_imbalance_trend": 1.0,
    "orderbook_imbalance_volatility": 1.0,
    "orderbook_support_distance_pct": 5.0,
    "orderbook_resistance_distance_pct": 5.0,
    "orderbook_support_strength": 1.0,
    "orderbook_resistance_strength": 1.0,
    "orderbook_support_wall_persistence": 12.0,
    "orderbook_resistance_wall_persistence": 12.0,
    "market_map_score_adjustment": 20.0,
    "planned_risk_pct": 12.0,
    "planned_reward_pct": 30.0,
    "planned_risk_reward_ratio": 4.0,
    "planned_stop_atr_multiple": 4.0,
    "planned_target_atr_multiple": 8.0,
    "planned_target_r_multiple": 4.0,
    "thesis_alignment_points": 8.0,
    "thesis_conflict_points": 6.0,
    "thesis_conviction_score": 100.0,
    "expectancy_probability": 1.0,
    "expectancy_expected_r": 2.0,
    "expectancy_uncertainty": 1.0,
    "expectancy_score": 100.0,
    "execution_quality_score": 100.0,
    "estimated_slippage_bps": 40.0,
}


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


class HistoricalAnalogEngine:
    def __init__(self, trading_cfg):
        self.cfg = trading_cfg
        self._cache_mtime: float | None = None
        self._cache_rows: list[dict] = []

    def _load_trade_rows(self) -> list[dict]:
        dataset_path = TRADE_DATASET_JSONL
        try:
            mtime = dataset_path.stat().st_mtime if dataset_path.exists() else None
        except Exception:
            mtime = None
        if mtime == self._cache_mtime and self._cache_rows:
            return list(self._cache_rows)

        rows = trade_dataset.load_closed_trades(limit=getattr(self.cfg, "analog_history_limit", 1500))
        feature_rows = [feature_store.build_closed_trade_feature_row(row) for row in rows if isinstance(row, dict)]
        self._cache_rows = feature_rows
        self._cache_mtime = mtime
        return list(feature_rows)

    def _numeric_similarity(self, key: str, current: float, previous: float) -> float:
        scale = float(SCALE_HINTS.get(key, max(abs(current), abs(previous), 1.0)) or 1.0)
        diff = abs(current - previous) / scale
        return _clamp(1.0 - diff, 0.0, 1.0)

    def _similarity(self, current: Mapping[str, Any], previous: Mapping[str, Any]) -> float:
        numeric_scores: list[float] = []
        for key in NUMERIC_KEYS:
            if key not in current or key not in previous:
                continue
            numeric_scores.append(self._numeric_similarity(key, _safe_float(current.get(key)), _safe_float(previous.get(key))))

        categorical_scores: list[float] = []
        for key in CATEGORICAL_KEYS:
            cur_val = str(current.get(key, "") or "").strip().lower()
            prev_val = str(previous.get(key, "") or "").strip().lower()
            if not cur_val or not prev_val:
                continue
            categorical_scores.append(1.0 if cur_val == prev_val else 0.0)

        if not numeric_scores and not categorical_scores:
            return 0.0

        numeric_component = sum(numeric_scores) / len(numeric_scores) if numeric_scores else 0.5
        categorical_component = sum(categorical_scores) / len(categorical_scores) if categorical_scores else 0.5
        return _clamp(numeric_component * 0.72 + categorical_component * 0.28, 0.0, 1.0)

    def evaluate(self, coin: str, action: str, signal_snapshot: Mapping[str, Any]) -> dict:
        action = str(action or "").upper()
        if not getattr(self.cfg, "analog_engine_enabled", True):
            return {"enabled": False, "summary": "analog engine disabled", "sample_size": 0}
        if action not in {"LONG", "SHORT"}:
            return {"enabled": True, "summary": "no directional analog lookup for flat setup", "sample_size": 0}

        current_features = feature_store.build_signal_feature_map(signal_snapshot)
        history = self._load_trade_rows()
        candidates: list[dict] = []
        same_coin_bonus = float(getattr(self.cfg, "analog_same_coin_bonus", 0.08) or 0.08)
        same_instrument_bonus = float(getattr(self.cfg, "analog_same_instrument_bonus", 0.04) or 0.04)
        similarity_floor = float(getattr(self.cfg, "analog_similarity_floor", 0.58) or 0.58)

        for row in history:
            direction = str(row.get("direction") or "").upper()
            if direction != action:
                continue

            row_features = dict(row.get("features") or {})
            similarity = self._similarity(current_features, row_features)
            if str(row.get("coin") or "").upper() == str(coin or "").upper():
                similarity += same_coin_bonus
            if row_features.get("instrument_type") == current_features.get("instrument_type"):
                similarity += same_instrument_bonus
            similarity = _clamp(similarity, 0.0, 1.0)
            if similarity < similarity_floor:
                continue

            labels = dict(row.get("labels") or {})
            candidates.append({
                "coin": row.get("coin"),
                "trade_id": row.get("trade_id"),
                "similarity": round(similarity, 4),
                "outcome": labels.get("outcome", "UNKNOWN"),
                "pnl_pct": _safe_float(labels.get("pnl_pct")),
                "captured_r_multiple": _safe_float(labels.get("captured_r_multiple")),
                "exit_reason": labels.get("exit_reason", ""),
            })

        if not candidates:
            return {
                "enabled": True,
                "verdict": "INSUFFICIENT",
                "sample_size": 0,
                "avg_similarity": 0.0,
                "reliability": 0.0,
                "summary": "no close historical analogs yet",
                "supportive": False,
                "adverse": False,
                "hard_block": False,
                "top_matches": [],
                "score_adjustment": 0.0,
                "probability_adjustment": 0.0,
                "expected_r_adjustment": 0.0,
                "uncertainty_adjustment": 0.0,
            }

        candidates.sort(key=lambda item: (-item["similarity"], item["coin"] or "", item["trade_id"] or ""))
        top_n = max(1, int(getattr(self.cfg, "analog_max_examples", 5) or 5))
        top = candidates[:top_n]
        weights = [max(item["similarity"], 0.01) ** 2 for item in top]
        weight_sum = sum(weights) or 1.0
        avg_similarity = sum(weight * item["similarity"] for weight, item in zip(weights, top)) / weight_sum
        win_score = sum(
            weight * (
                1.0 if item["outcome"] == "WIN" else 0.5 if item["outcome"] == "BREAKEVEN" else 0.0
            )
            for weight, item in zip(weights, top)
        ) / weight_sum
        avg_pnl_pct = sum(weight * item["pnl_pct"] for weight, item in zip(weights, top)) / weight_sum
        avg_captured_r = sum(weight * item["captured_r_multiple"] for weight, item in zip(weights, top)) / weight_sum
        same_coin_share = sum(
            weight for weight, item in zip(weights, top) if str(item["coin"] or "").upper() == str(coin or "").upper()
        ) / weight_sum

        min_samples = int(getattr(self.cfg, "analog_min_samples", 5) or 5)
        hard_block_min_samples = int(getattr(self.cfg, "analog_hard_block_min_samples", 8) or 8)
        min_reliability = float(getattr(self.cfg, "analog_min_reliability", 0.42) or 0.42)
        reliability = _clamp(avg_similarity * min(1.0, len(top) / float(max(min_samples, 1))), 0.0, 1.0)
        supportive_wr = float(getattr(self.cfg, "analog_supportive_win_rate", 0.57) or 0.57)
        adverse_wr = float(getattr(self.cfg, "analog_adverse_win_rate", 0.43) or 0.43)
        hard_block_wr = float(getattr(self.cfg, "analog_hard_block_win_rate", 0.35) or 0.35)
        positive_expected_r = float(getattr(self.cfg, "analog_positive_expected_r", 0.18) or 0.18)
        negative_expected_r = float(getattr(self.cfg, "analog_negative_expected_r", -0.10) or -0.10)

        supportive = (
            len(top) >= min_samples
            and reliability >= min_reliability
            and win_score >= supportive_wr
            and avg_captured_r >= positive_expected_r
            and avg_pnl_pct >= 0.0
        )
        adverse = (
            len(top) >= min_samples
            and reliability >= min_reliability
            and (win_score <= adverse_wr or avg_captured_r <= negative_expected_r or avg_pnl_pct < 0.0)
        )
        hard_block = (
            len(top) >= hard_block_min_samples
            and reliability >= min_reliability + 0.08
            and win_score <= hard_block_wr
            and avg_captured_r <= negative_expected_r
            and avg_pnl_pct < 0.0
        )

        score_cap = float(getattr(self.cfg, "analog_score_adjustment_cap", 4.0) or 4.0)
        prob_cap = float(getattr(self.cfg, "analog_probability_adjustment_cap", 0.06) or 0.06)
        expected_r_cap = float(getattr(self.cfg, "analog_expected_r_adjustment_cap", 0.12) or 0.12)
        uncertainty_cap = float(getattr(self.cfg, "analog_uncertainty_adjustment_cap", 0.08) or 0.08)

        score_adjustment = 0.0
        probability_adjustment = 0.0
        expected_r_adjustment = 0.0
        uncertainty_adjustment = 0.0
        verdict = "INSUFFICIENT"

        if supportive:
            verdict = "SUPPORTIVE"
            magnitude = reliability * (0.5 + max(0.0, win_score - supportive_wr) * 3.5)
            score_adjustment = min(score_cap, magnitude * score_cap)
            probability_adjustment = min(prob_cap, magnitude * prob_cap)
            expected_r_adjustment = min(expected_r_cap, magnitude * expected_r_cap + max(0.0, avg_captured_r) * 0.05)
            uncertainty_adjustment = -min(uncertainty_cap, magnitude * uncertainty_cap)
        elif adverse:
            verdict = "ADVERSE"
            magnitude = reliability * (0.5 + max(0.0, adverse_wr - win_score) * 3.5)
            score_adjustment = -min(score_cap, magnitude * score_cap)
            probability_adjustment = -min(prob_cap, magnitude * prob_cap)
            expected_r_adjustment = -min(expected_r_cap, magnitude * expected_r_cap + max(0.0, -avg_captured_r) * 0.05)
            uncertainty_adjustment = min(uncertainty_cap, magnitude * uncertainty_cap)

        if hard_block:
            verdict = "HARD_BLOCK"
            score_adjustment = -score_cap
            probability_adjustment = -prob_cap
            expected_r_adjustment = -expected_r_cap
            uncertainty_adjustment = uncertainty_cap

        top_matches = [
            {
                "coin": item["coin"],
                "trade_id": item["trade_id"],
                "similarity": item["similarity"],
                "outcome": item["outcome"],
                "pnl_pct": round(item["pnl_pct"], 4),
                "captured_r_multiple": round(item["captured_r_multiple"], 4),
                "exit_reason": item["exit_reason"],
            }
            for item in top[:3]
        ]
        summary = (
            f"{len(top)} analogs • win {win_score * 100:.0f}% • avg pnl {avg_pnl_pct:+.2f}% "
            f"• avg R {avg_captured_r:+.2f} • reliability {reliability:.2f}"
        )
        return {
            "enabled": True,
            "verdict": verdict,
            "sample_size": len(top),
            "avg_similarity": round(avg_similarity, 4),
            "reliability": round(reliability, 4),
            "win_rate": round(win_score, 4),
            "avg_pnl_pct": round(avg_pnl_pct, 4),
            "avg_captured_r": round(avg_captured_r, 4),
            "same_coin_share": round(same_coin_share, 4),
            "supportive": supportive,
            "adverse": adverse,
            "hard_block": hard_block,
            "score_adjustment": round(score_adjustment, 4),
            "probability_adjustment": round(probability_adjustment, 4),
            "expected_r_adjustment": round(expected_r_adjustment, 4),
            "uncertainty_adjustment": round(uncertainty_adjustment, 4),
            "top_matches": top_matches,
            "summary": summary,
        }

    def blend_expectancy(self, expectancy: Mapping[str, Any], analog: Mapping[str, Any], *, same_direction_position: bool = False) -> dict:
        blended = dict(expectancy or {})
        if not analog or not analog.get("enabled", False):
            return blended

        blended["probability"] = round(
            _clamp(_safe_float(blended.get("probability"), 0.50) + _safe_float(analog.get("probability_adjustment")), 0.0, 1.0),
            4,
        )
        blended["expected_r"] = round(
            _safe_float(blended.get("expected_r")) + _safe_float(analog.get("expected_r_adjustment")),
            4,
        )
        blended["uncertainty"] = round(
            _clamp(_safe_float(blended.get("uncertainty"), 0.50) + _safe_float(analog.get("uncertainty_adjustment")), 0.0, 1.0),
            4,
        )
        blended["score"] = round(
            _clamp(_safe_float(blended.get("score"), 50.0) + _safe_float(analog.get("score_adjustment")), 0.0, 100.0),
            2,
        )
        blended["analog_summary"] = str(analog.get("summary", "") or "")
        reasons = list(blended.get("reasons") or [])
        blockers = list(blended.get("blockers") or [])

        verdict = str(analog.get("verdict", "INSUFFICIENT") or "INSUFFICIENT").upper()
        if verdict == "SUPPORTIVE":
            reasons.append(f"historical analogs support this {analog.get('sample_size', 0)}-sample setup")
        elif verdict in {"ADVERSE", "HARD_BLOCK"}:
            blockers.append(f"historical analogs oppose this setup ({analog.get('sample_size', 0)} close matches)")

        blended["reasons"] = reasons
        blended["blockers"] = blockers
        blended["permitted"] = bool(blended.get("permitted", True))

        min_probability = float(getattr(self.cfg, "expectancy_min_probability", 0.54) or 0.54)
        min_expected_r = float(getattr(self.cfg, "expectancy_min_expected_r", 0.18) or 0.18)
        max_uncertainty = float(getattr(self.cfg, "expectancy_max_uncertainty", 0.42) or 0.42)
        min_score = float(
            getattr(
                self.cfg,
                "expectancy_same_direction_min_score" if same_direction_position else "expectancy_min_score",
                56.0,
            ) or 56.0
        )

        gate_failures: list[str] = []
        if blended["score"] < min_score:
            gate_failures.append(f"expectancy score {blended['score']:.0f} is below {min_score:.0f}")
        if blended["probability"] < min_probability:
            gate_failures.append(f"win probability {blended['probability'] * 100:.0f}% is below {min_probability * 100:.0f}%")
        if blended["expected_r"] < min_expected_r:
            gate_failures.append(f"expected R {blended['expected_r']:.2f} is below {min_expected_r:.2f}")
        if blended["uncertainty"] > max_uncertainty:
            gate_failures.append(f"uncertainty {blended['uncertainty']:.2f} is above {max_uncertainty:.2f}")
        if analog.get("hard_block", False):
            gate_failures.insert(0, "historical analog engine hard-blocked the setup")

        if gate_failures:
            blended["permitted"] = False
            for reason in gate_failures:
                if reason not in blended["blockers"]:
                    blended["blockers"].append(reason)
            blended["summary"] = gate_failures[0]
        elif verdict == "SUPPORTIVE":
            blended["summary"] = blended.get("summary") or str(analog.get("summary", "") or "historical analogs support the setup")

        return blended
