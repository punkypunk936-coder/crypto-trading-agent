"""A single accountable admission layer for trade probability, cost, and inventory.

The strategy may generate many narratives and guards. This module answers the
last economic question before an order is funded: does the empirically
calibrated edge still exist after uncertainty, execution cost, and book risk?
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping

import trade_dataset


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _entry_expectancy(row: Mapping[str, Any]) -> dict:
    context = dict(row.get("entry_context") or {})
    expectancy = dict(context.get("expectancy") or {})
    if "probability" not in expectancy and context.get("expectancy_probability") is not None:
        expectancy["probability"] = context.get("expectancy_probability")
    return expectancy


def _instrument_type(row: Mapping[str, Any]) -> str:
    context = dict(row.get("entry_context") or {})
    return _safe_str(context.get("instrument_type"), "unknown").lower()


class MicroDesk:
    """Calibrate candidate edge and constrain its expression at book level."""

    def __init__(self, cfg, *, data_dir: Path, report_path: Path | None = None):
        self.cfg = cfg
        self.data_dir = Path(data_dir).expanduser()
        self.report_path = Path(report_path).expanduser() if report_path else self.data_dir / "micro_desk_report.json"
        self._model: dict[str, Any] = {}
        self._last_refresh_ts = 0.0
        self._recent: deque[dict] = deque(maxlen=24)
        self._verdict_counts: Counter[str] = Counter()
        self.refresh(force=True)

    @staticmethod
    def _bucket(probability: float) -> str:
        value = _clamp(probability, 0.0, 0.999999)
        if value < 0.60:
            return "50-60"
        if value < 0.70:
            return "60-70"
        if value < 0.80:
            return "70-80"
        if value < 0.90:
            return "80-90"
        return "90-100"

    @staticmethod
    def _posterior(wins: int, samples: int, prior: float, strength: float) -> float:
        if samples <= 0:
            return prior
        return (float(wins) + max(0.0, strength) * prior) / (float(samples) + max(0.0, strength))

    def _history_rows(self) -> tuple[list[dict], Counter[str]]:
        limit = max(100, int(getattr(self.cfg, "micro_desk_history_limit", 1600) or 1600))
        raw = trade_dataset.load_closed_trades(
            limit=limit * 3,
            data_dir=self.data_dir,
            backfill_from_csv=True,
            deduplicate=True,
        )
        rows = list(raw[-limit:])
        price_logs: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            coin = _safe_str(row.get("coin")).upper()
            for field in ("entry_price", "exit_price"):
                price = _safe_float(row.get(field))
                if coin and price > 0:
                    price_logs[coin].append(math.log(price))
        medians = {
            coin: statistics.median(values)
            for coin, values in price_logs.items()
            if values
        }

        clean: list[dict] = []
        rejected: Counter[str] = Counter()
        max_scale_ratio = max(3.0, _safe_float(getattr(self.cfg, "micro_desk_max_price_scale_ratio", 20.0), 20.0))
        max_log_gap = math.log(max_scale_ratio)
        for row in rows:
            coin = _safe_str(row.get("coin")).upper()
            direction = _safe_str(row.get("direction")).upper()
            entry = _safe_float(row.get("entry_price"))
            exit_price = _safe_float(row.get("exit_price"))
            opened = _safe_float(row.get("opened_at_ts"))
            closed = _safe_float(row.get("closed_at_ts"))
            pnl_pct = _safe_float(row.get("pnl_pct"))
            if not coin or direction not in {"LONG", "SHORT"} or entry <= 0 or exit_price <= 0:
                rejected["invalid_identity_or_price"] += 1
                continue
            if opened <= 0 or closed <= 0 or closed < opened:
                rejected["invalid_timestamps"] += 1
                continue
            median_log = medians.get(coin)
            if median_log is not None and (
                abs(math.log(entry) - median_log) > max_log_gap
                or abs(math.log(exit_price) - median_log) > max_log_gap
            ):
                rejected["price_scale_mismatch"] += 1
                continue
            implied_pct = ((exit_price / entry) - 1.0) * 100.0 * (1.0 if direction == "LONG" else -1.0)
            tolerance = max(0.10, abs(pnl_pct) * 0.03)
            if abs(implied_pct - pnl_pct) > tolerance:
                rejected["pnl_geometry_mismatch"] += 1
                continue
            enriched = dict(row)
            enriched["trade_key"] = trade_dataset.trade_record_key(enriched)
            enriched["instrument_type"] = _instrument_type(enriched)
            clean.append(enriched)
        return clean, rejected

    def _calibrate_from_model(
        self,
        raw_probability: float,
        *,
        coin: str,
        direction: str,
        instrument_type: str,
        model: Mapping[str, Any] | None = None,
    ) -> dict:
        model = dict(model or self._model)
        raw = _clamp(raw_probability, 0.25, 0.99)
        global_stats = dict(model.get("global") or {})
        global_rate = _safe_float(global_stats.get("posterior"), 0.50)
        bucket_name = self._bucket(raw)
        bucket = dict((model.get("buckets") or {}).get(bucket_name) or {})
        bucket_rate = _safe_float(bucket.get("posterior"), global_rate)
        family_key = f"{coin.upper()}:{direction.upper()}"
        family = dict((model.get("families") or {}).get(family_key) or {})
        family_rate = _safe_float(family.get("posterior"), bucket_rate)
        instrument_key = f"{instrument_type.lower()}:{direction.upper()}"
        instrument = dict((model.get("instruments") or {}).get(instrument_key) or {})
        instrument_rate = _safe_float(instrument.get("posterior"), global_rate)

        samples = int(global_stats.get("samples") or 0)
        raw_weight = 0.10 if samples >= 50 else 0.35
        family_weight = min(0.30, int(family.get("samples") or 0) / 80.0)
        bucket_weight = min(0.45, int(bucket.get("samples") or 0) / 120.0)
        instrument_weight = min(0.20, int(instrument.get("samples") or 0) / 120.0)
        global_weight = max(0.0, 1.0 - raw_weight - family_weight - bucket_weight - instrument_weight)
        total_weight = raw_weight + family_weight + bucket_weight + instrument_weight + global_weight
        calibrated = (
            raw * raw_weight
            + family_rate * family_weight
            + bucket_rate * bucket_weight
            + instrument_rate * instrument_weight
            + global_rate * global_weight
        ) / max(total_weight, 1e-9)
        max_probability = _clamp(
            _safe_float(getattr(self.cfg, "micro_desk_max_calibrated_probability", 0.78), 0.78),
            0.51,
            0.95,
        )
        calibrated = _clamp(calibrated, 0.20, max_probability)
        return {
            "raw_probability": round(raw, 4),
            "calibrated_probability": round(calibrated, 4),
            "probability_haircut": round(raw - calibrated, 4),
            "bucket": bucket_name,
            "bucket_samples": int(bucket.get("samples") or 0),
            "family": family_key,
            "family_samples": int(family.get("samples") or 0),
            "instrument_family": instrument_key,
            "instrument_samples": int(instrument.get("samples") or 0),
            "global_samples": samples,
            "global_win_rate": round(global_rate, 4),
        }

    def _fit_model(self, rows: list[dict]) -> dict:
        wins = sum(row["win"] for row in rows)
        samples = len(rows)
        global_rate = self._posterior(wins, samples, 0.50, 12.0)

        def build_stats(key_fn, prior: float, strength: float) -> dict:
            groups: dict[str, list[dict]] = defaultdict(list)
            for row in rows:
                groups[key_fn(row)].append(row)
            return {
                key: {
                    "samples": len(items),
                    "wins": sum(item["win"] for item in items),
                    "win_rate": round(sum(item["win"] for item in items) / max(1, len(items)), 4),
                    "posterior": round(
                        self._posterior(sum(item["win"] for item in items), len(items), prior, strength),
                        4,
                    ),
                }
                for key, items in groups.items()
            }

        return {
            "global": {
                "samples": samples,
                "wins": wins,
                "win_rate": round(wins / max(1, samples), 4),
                "posterior": round(global_rate, 4),
            },
            "buckets": build_stats(lambda row: self._bucket(row["probability"]), global_rate, 14.0),
            "families": build_stats(lambda row: f"{row['coin']}:{row['direction']}", global_rate, 20.0),
            "instruments": build_stats(
                lambda row: f"{row['instrument_type']}:{row['direction']}",
                global_rate,
                24.0,
            ),
        }

    def refresh(self, *, force: bool = False) -> dict:
        refresh_seconds = max(30.0, _safe_float(getattr(self.cfg, "micro_desk_refresh_seconds", 300.0), 300.0))
        if not force and self._model and (time.time() - self._last_refresh_ts) < refresh_seconds:
            return self.summary()

        clean, rejected = self._history_rows()
        calibration_rows: list[dict] = []
        for row in clean:
            probability = _safe_float(_entry_expectancy(row).get("probability"), -1.0)
            if probability < 0.25 or probability > 1.0:
                continue
            calibration_rows.append({
                "probability": _clamp(probability, 0.25, 0.99),
                "win": 1 if _safe_float(row.get("pnl_usd")) > 0 else 0,
                "coin": _safe_str(row.get("coin")).upper(),
                "direction": _safe_str(row.get("direction")).upper(),
                "instrument_type": _safe_str(row.get("instrument_type"), "unknown").lower(),
            })

        samples = len(calibration_rows)
        fitted = self._fit_model(calibration_rows)
        self._model = {
            "generated_at": int(time.time()),
            "clean_history_trades": len(clean),
            "rejected_history_rows": sum(rejected.values()),
            "rejection_reasons": dict(rejected),
            **fitted,
        }

        raw_brier = 0.0
        calibrated_brier = 0.0
        calibration_gap = 0.0
        holdout_count = max(1, min(max(20, int(samples * 0.20)), samples // 3)) if samples >= 30 else 0
        holdout_rows = calibration_rows[-holdout_count:] if holdout_count else []
        training_rows = calibration_rows[:-holdout_count] if holdout_count else []
        validation_model = self._fit_model(training_rows) if training_rows and holdout_rows else {}
        for row in holdout_rows:
            calibrated = self._calibrate_from_model(
                row["probability"],
                coin=row["coin"],
                direction=row["direction"],
                instrument_type=row["instrument_type"],
                model=validation_model,
            )["calibrated_probability"]
            raw_brier += (row["probability"] - row["win"]) ** 2
            calibrated_brier += (calibrated - row["win"]) ** 2
            calibration_gap += abs(row["probability"] - row["win"])
        if holdout_count:
            self._model["calibration"] = {
                "method": "chronological_holdout",
                "training_samples": len(training_rows),
                "holdout_samples": holdout_count,
                "raw_brier": round(raw_brier / holdout_count, 4),
                "calibrated_brier": round(calibrated_brier / holdout_count, 4),
                "raw_absolute_error": round(calibration_gap / holdout_count, 4),
                "improvement_pct": round(
                    max(0.0, (raw_brier - calibrated_brier) / max(raw_brier, 1e-9) * 100.0),
                    2,
                ),
            }
        else:
            self._model["calibration"] = {}
        self._last_refresh_ts = time.time()
        self._write_report()
        return self.summary()

    @staticmethod
    def _position_size(position: Any) -> float:
        if isinstance(position, Mapping):
            return _safe_float(position.get("size_usd"))
        return _safe_float(getattr(position, "size_usd", 0.0))

    @staticmethod
    def _position_direction(position: Any) -> str:
        if isinstance(position, Mapping):
            return _safe_str(position.get("direction")).upper()
        return _safe_str(getattr(position, "direction", "")).upper()

    def assess(
        self,
        *,
        coin: str,
        direction: str,
        signal_snapshot: Mapping[str, Any],
        order: Any,
        portfolio_usd: float,
        open_positions: Iterable[Any] | None = None,
        execution_quality: Mapping[str, Any] | None = None,
    ) -> dict:
        self.refresh()
        snap = dict(signal_snapshot or {})
        expectancy = dict(snap.get("expectancy") or {})
        plan = dict(snap.get("trade_plan") or {})
        execution_plan = dict(snap.get("execution_plan") or {})
        quality = dict(execution_quality or {})
        direction = _safe_str(direction).upper()
        instrument = _safe_str(snap.get("instrument_type"), "unknown").lower()
        raw_probability = _safe_float(
            expectancy.get("probability", snap.get("expectancy_probability", 0.50)),
            0.50,
        )
        calibration = self._calibrate_from_model(
            raw_probability,
            coin=coin,
            direction=direction,
            instrument_type=instrument,
        )
        probability = _safe_float(calibration["calibrated_probability"], 0.50)
        uncertainty = _clamp(
            _safe_float(expectancy.get("uncertainty", snap.get("expectancy_uncertainty", 0.50)), 0.50),
            0.0,
            1.0,
        )
        price = _safe_float(getattr(order, "price", 0.0) or snap.get("live_price") or snap.get("price"))
        live_price = _safe_float(snap.get("live_price") or snap.get("price") or price)
        risk_pct = _safe_float(snap.get("planned_risk_pct", plan.get("risk_pct")))
        reward_pct = _safe_float(snap.get("planned_reward_pct", plan.get("reward_pct")))
        stop = _safe_float(getattr(order, "stop_loss", 0.0) or plan.get("stop_loss"))
        target = _safe_float(getattr(order, "take_profit", 0.0) or plan.get("take_profit"))
        if risk_pct <= 0 and price > 0 and stop > 0:
            risk_pct = abs(price - stop) / price * 100.0
        if reward_pct <= 0 and price > 0 and target > 0:
            reward_pct = abs(target - price) / price * 100.0

        blockers: list[str] = []
        warnings: list[str] = []
        now = time.time()
        analysis_ts = _safe_float(snap.get("analysis_updated_ts"))
        max_age = max(60.0, _safe_float(getattr(self.cfg, "analysis_signal_max_age_minutes", 20.0), 20.0) * 60.0)
        signal_age = max(0.0, now - analysis_ts) if analysis_ts > 0 else float("inf")
        if analysis_ts <= 0 or signal_age > max_age:
            blockers.append("analysis state is stale")
        if direction not in {"LONG", "SHORT"}:
            blockers.append("no directional candidate")
        if price <= 0 or live_price <= 0:
            blockers.append("live execution price is unavailable")
        price_drift_bps = abs(price - live_price) / max(live_price, 1e-9) * 10_000.0 if price > 0 and live_price > 0 else 0.0
        max_drift = max(1.0, _safe_float(getattr(self.cfg, "micro_desk_max_price_drift_bps", 30.0), 30.0))
        if price_drift_bps > max_drift:
            blockers.append(f"order price drifted {price_drift_bps:.1f}bps from the live mark")
        if _safe_str(snap.get("price_status"), "OK").upper() in {"INVALID", "STALE", "MISMATCH"}:
            blockers.append("price validation is not clean")
        reliability = _safe_float(snap.get("data_reliability_score"), 100.0)
        min_reliability = _safe_float(getattr(self.cfg, "data_reliability_min_score", 58.0), 58.0)
        if reliability < min_reliability:
            blockers.append(f"data reliability {reliability:.0f} is below {min_reliability:.0f}")
        if risk_pct <= 0 or reward_pct <= 0:
            blockers.append("trade plan has no valid risk and reward geometry")

        risk_bps = max(0.0, risk_pct * 100.0)
        reward_bps = max(0.0, reward_pct * 100.0)
        gross_edge_bps = probability * reward_bps - (1.0 - probability) * risk_bps
        mode = _safe_str(execution_plan.get("mode"), "market").lower()
        passive = mode in {"limit", "maker_limit"}
        maker_fee = max(0.0, _safe_float(getattr(self.cfg, "micro_desk_maker_fee_bps", 1.5), 1.5))
        taker_fee = max(0.0, _safe_float(getattr(self.cfg, "micro_desk_taker_fee_bps", 4.5), 4.5))
        spread_bps = max(0.0, _safe_float(quality.get("spread_bps")))
        slippage_bps = max(0.0, _safe_float(quality.get("estimated_slippage_bps")))
        cost_buffer = max(0.0, _safe_float(getattr(self.cfg, "micro_desk_cost_buffer_bps", 4.0), 4.0))
        if quality:
            market_friction = (spread_bps * 0.5) + slippage_bps
            passive_friction = min(slippage_bps * 0.25, 3.0)
        else:
            market_friction = max(8.0, cost_buffer)
            passive_friction = max(3.0, cost_buffer * 0.5)
            warnings.append("execution-cost estimate is using a conservative fallback")
        all_in_cost_bps = (
            maker_fee + taker_fee + passive_friction + cost_buffer
            if passive
            else taker_fee * 2.0 + market_friction + cost_buffer
        )
        uncertainty_charge_bps = uncertainty * risk_bps * max(
            0.0,
            _safe_float(getattr(self.cfg, "micro_desk_uncertainty_charge", 0.12), 0.12),
        )
        net_edge_bps = gross_edge_bps - all_in_cost_bps - uncertainty_charge_bps

        open_positions = list(open_positions or [])
        gross_exposure = sum(self._position_size(position) for position in open_positions)
        signed_exposure = sum(
            self._position_size(position) * (1.0 if self._position_direction(position) == "LONG" else -1.0)
            for position in open_positions
        )
        proposed_size = max(0.0, _safe_float(getattr(order, "size_usd", 0.0)))
        proposed_sign = 1.0 if direction == "LONG" else -1.0
        max_gross_pct = max(0.01, _safe_float(getattr(self.cfg, "max_total_exposure_pct", 0.65), 0.65))
        next_gross_pct = (gross_exposure + proposed_size) / max(portfolio_usd, 1e-9)
        next_net_pct = abs(signed_exposure + proposed_size * proposed_sign) / max(portfolio_usd, 1e-9)
        inventory_multiplier = 1.0
        if next_gross_pct > max_gross_pct:
            blockers.append(f"gross inventory would reach {next_gross_pct * 100:.1f}%")
        elif next_gross_pct > max_gross_pct * 0.75:
            inventory_multiplier = min(
                inventory_multiplier,
                max(0.25, (max_gross_pct - next_gross_pct) / max(max_gross_pct * 0.25, 1e-9) + 0.25),
            )
            warnings.append("book is near its gross inventory budget")
        max_net_pct = max(0.05, _safe_float(getattr(self.cfg, "micro_desk_max_directional_exposure_pct", 0.35), 0.35))
        if next_net_pct > max_net_pct:
            inventory_multiplier = min(inventory_multiplier, max(0.20, max_net_pct / max(next_net_pct, 1e-9)))
            warnings.append("same-direction inventory is crowded")

        rr = reward_bps / max(risk_bps, 1e-9)
        kelly = max(0.0, (probability * rr - (1.0 - probability)) / max(rr, 1e-9))
        min_edge = max(0.0, _safe_float(getattr(self.cfg, "micro_desk_min_net_edge_bps", 6.0), 6.0))
        if net_edge_bps < min_edge:
            blockers.append(f"net edge {net_edge_bps:.1f}bps does not clear {min_edge:.1f}bps")

        edge_scale = _clamp(
            net_edge_bps / max(1.0, _safe_float(getattr(self.cfg, "micro_desk_full_size_edge_bps", 60.0), 60.0)),
            0.20,
            1.0,
        )
        confidence_scale = 1.0
        haircut = _safe_float(calibration.get("probability_haircut"))
        if haircut >= 0.25:
            confidence_scale = 0.35
        elif haircut >= 0.15:
            confidence_scale = 0.55
        elif haircut >= 0.08:
            confidence_scale = 0.75
        kelly_scale = _clamp(
            kelly / max(0.05, _safe_float(getattr(self.cfg, "micro_desk_full_size_kelly", 0.20), 0.20)),
            0.20,
            1.0,
        )
        size_multiplier = min(1.0, edge_scale, confidence_scale, kelly_scale, inventory_multiplier)

        global_samples = int(calibration.get("global_samples") or 0)
        min_samples = max(1, int(getattr(self.cfg, "micro_desk_min_calibration_samples", 40) or 40))
        passive_only = False
        if global_samples < min_samples:
            passive_only = True
            warnings.append("calibration history is still thin")
        if haircut >= _safe_float(getattr(self.cfg, "micro_desk_passive_haircut_threshold", 0.12), 0.12):
            passive_only = True
            warnings.append("model confidence was materially overstated")
        aggressive_edge = max(min_edge, _safe_float(getattr(self.cfg, "micro_desk_aggressive_min_edge_bps", 35.0), 35.0))
        if not passive and net_edge_bps < aggressive_edge:
            passive_only = True
            warnings.append("edge is positive but too thin to cross the spread")

        permitted = not blockers
        verdict = "BLOCK" if not permitted else ("PASSIVE_ONLY" if passive_only else "APPROVE")
        summary = (
            "; ".join(blockers[:2])
            if blockers
            else (
                f"{calibration['raw_probability'] * 100:.0f}% raw -> {probability * 100:.0f}% calibrated; "
                f"edge {net_edge_bps:+.1f}bps after {all_in_cost_bps:.1f}bps cost"
                + ("; maker order required" if passive_only else "; execution may proceed")
            )
        )
        result = {
            "enabled": bool(getattr(self.cfg, "micro_desk_enabled", True)),
            "permitted": permitted,
            "verdict": verdict,
            "summary": summary,
            "blockers": blockers,
            "warnings": warnings,
            **calibration,
            "uncertainty": round(uncertainty, 4),
            "risk_bps": round(risk_bps, 2),
            "reward_bps": round(reward_bps, 2),
            "gross_edge_bps": round(gross_edge_bps, 2),
            "all_in_cost_bps": round(all_in_cost_bps, 2),
            "uncertainty_charge_bps": round(uncertainty_charge_bps, 2),
            "net_edge_bps": round(net_edge_bps, 2),
            "kelly_fraction": round(kelly, 4),
            "size_multiplier": round(size_multiplier, 4),
            "inventory_multiplier": round(inventory_multiplier, 4),
            "gross_exposure_pct": round(gross_exposure / max(portfolio_usd, 1e-9), 4),
            "next_gross_exposure_pct": round(next_gross_pct, 4),
            "next_directional_exposure_pct": round(next_net_pct, 4),
            "signal_age_seconds": None if not math.isfinite(signal_age) else round(signal_age, 2),
            "price_drift_bps": round(price_drift_bps, 2),
            "passive_required": passive_only,
            "evaluated_at": int(now),
        }
        if not bool(getattr(self.cfg, "micro_desk_enabled", True)):
            result.update({
                "permitted": True,
                "verdict": "DISABLED",
                "summary": "Micro desk is disabled; upstream policy owns the order.",
                "blockers": [],
                "size_multiplier": 1.0,
                "passive_required": False,
            })
        self._recent.appendleft({
            "coin": coin.upper(),
            "direction": direction,
            "verdict": result["verdict"],
            "raw_probability": result["raw_probability"],
            "calibrated_probability": result["calibrated_probability"],
            "net_edge_bps": result["net_edge_bps"],
            "all_in_cost_bps": result["all_in_cost_bps"],
            "size_multiplier": result["size_multiplier"],
            "summary": result["summary"],
            "evaluated_at": result["evaluated_at"],
        })
        self._verdict_counts[result["verdict"]] += 1
        return result

    def summary(self) -> dict:
        model = dict(self._model or {})
        global_stats = dict(model.get("global") or {})
        calibration = dict(model.get("calibration") or {})
        raw_brier = _safe_float(calibration.get("raw_brier"))
        calibrated_brier = _safe_float(calibration.get("calibrated_brier"))
        headline = (
            f"{int(global_stats.get('samples') or 0)} clean calibrated trades; "
            f"holdout Brier {raw_brier:.3f} -> {calibrated_brier:.3f}."
            if global_stats
            else "Waiting for clean closed-trade history."
        )
        return {
            "enabled": bool(getattr(self.cfg, "micro_desk_enabled", True)),
            "generated_at": int(time.time()),
            "headline": headline,
            "clean_history_trades": int(model.get("clean_history_trades") or 0),
            "rejected_history_rows": int(model.get("rejected_history_rows") or 0),
            "rejection_reasons": dict(model.get("rejection_reasons") or {}),
            "calibration_samples": int(global_stats.get("samples") or 0),
            "empirical_win_rate": _safe_float(global_stats.get("win_rate")),
            "posterior_win_rate": _safe_float(global_stats.get("posterior")),
            "raw_brier": raw_brier,
            "calibrated_brier": calibrated_brier,
            "calibration_improvement_pct": _safe_float(calibration.get("improvement_pct")),
            "calibration_method": _safe_str(calibration.get("method")),
            "calibration_holdout_samples": int(calibration.get("holdout_samples") or 0),
            "verdict_counts": dict(self._verdict_counts),
            "latest_decisions": list(self._recent)[:8],
            "principle": "Fund only positive edge after cost and uncertainty; never enlarge the strategy order.",
        }

    def _write_report(self) -> None:
        try:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            payload = self.summary()
            payload["model"] = self._model
            self.report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            return
