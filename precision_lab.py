"""
precision_lab.py — replay recent decision history to find elite setup families.

This tool does not promise a fixed win rate. It helps us answer a more honest
question: given the agent's recent directional decisions, which setup families
actually showed clean follow-through, and which ones should be embargoed?

Usage
─────
  python3 precision_lab.py
  python3 precision_lab.py --data-dir "/Users/.../crypto_trading_agent_runtime"
  python3 precision_lab.py --target-r 0.25 --horizon-minutes 720
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from data.market_data import fetch_candles

INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _safe_str(value: Any, default: str = "") -> str:
    text = str(value or default).strip()
    return text if text else default


def _confidence_rank(value: str) -> int:
    ordering = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    return ordering.get(_safe_str(value).upper(), 0)


def _default_data_dir() -> Path:
    repo_dir = Path(__file__).resolve().parent
    runtime_dir = Path.home() / "Library" / "Application Support" / "crypto_trading_agent_runtime"
    candidates = [repo_dir, runtime_dir]
    scored: list[tuple[int, Path]] = []
    for path in candidates:
        score = 0
        for name in ("decision_dataset.jsonl", "feature_store.jsonl"):
            file_path = path / name
            if file_path.exists():
                score += int(file_path.stat().st_size)
        scored.append((score, path))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _load_directional_decisions(data_dir: Path, *, final_only: bool = True) -> list[dict]:
    dataset_path = data_dir / "decision_dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"decision dataset not found at {dataset_path}")

    rows: list[dict] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue

            snap = dict(record.get("signal_snapshot") or {})
            action_key = "final_action" if final_only else "candidate_action"
            action = _safe_str(record.get(action_key, snap.get("action", "FLAT"))).upper()
            if action not in {"LONG", "SHORT"}:
                continue

            risk_pct = _safe_float(snap.get("planned_risk_pct")) / 100.0
            if risk_pct <= 0:
                continue

            rows.append({
                "decision_id": record.get("decision_id"),
                "cycle_number": record.get("cycle_number"),
                "coin": _safe_str(record.get("coin")).upper(),
                "action": action,
                "ts": _safe_float(record.get("recorded_at_ts")),
                "risk_pct": risk_pct,
                "reward_pct": _safe_float(snap.get("planned_reward_pct")) / 100.0,
                "rr": _safe_float(snap.get("planned_risk_reward_ratio")),
                "prob": _safe_float(snap.get("expectancy_probability"), 0.50),
                "unc": _safe_float(snap.get("expectancy_uncertainty"), 0.50),
                "score": _safe_float(snap.get("expectancy_score"), 50.0),
                "orderbook_score": _safe_float(snap.get("orderbook_score"), 50.0),
                "confidence": _safe_str(snap.get("confidence"), "LOW").upper(),
                "thesis_quality": _safe_str(snap.get("thesis_quality"), "LOW").upper(),
                "breakout": _safe_str(snap.get("orderbook_breakout_state"), "NONE").lower(),
                "interaction": _safe_str(snap.get("orderbook_interaction"), "between_levels").lower(),
                "regime": _safe_str(snap.get("dominant_regime"), "mixed").lower(),
                "instrument_type": _safe_str(snap.get("instrument_type"), "crypto").lower(),
                "support_defense_long": bool((snap.get("thesis") or {}).get("support_defense_long", False)),
            })

    rows.sort(key=lambda item: item["ts"])
    return rows


def _collapse_episodes(rows: list[dict], *, dedupe_minutes: int = 30) -> list[dict]:
    episodes: list[dict] = []
    max_gap = dedupe_minutes * 60
    for row in rows:
        if not episodes:
            episodes.append(dict(row))
            continue
        last = episodes[-1]
        same_family = (
            row["coin"] == last["coin"]
            and row["action"] == last["action"]
            and row["breakout"] == last["breakout"]
            and row["interaction"] == last["interaction"]
            and row["regime"] == last["regime"]
        )
        if same_family and (row["ts"] - last["ts"]) <= max_gap:
            continue
        episodes.append(dict(row))
    return episodes


_future_cache: dict[tuple[str, str, int], pd.DataFrame | None] = {}


def _future_window(coin: str, ts: float, *, interval: str, horizon_minutes: int) -> pd.DataFrame | None:
    now = time.time()
    interval_seconds = INTERVAL_SECONDS[interval]
    age_seconds = max(0.0, now - ts)
    bars_needed = int(math.ceil(age_seconds / interval_seconds) + math.ceil((horizon_minutes * 60) / interval_seconds) + 30)
    lookback = max(200, min(4000, bars_needed))
    cache_key = (coin, interval, lookback)
    if cache_key not in _future_cache:
        _future_cache[cache_key] = fetch_candles(coin, interval=interval, lookback=lookback)

    df = _future_cache[cache_key]
    if df is None or df.empty or "timestamp" not in df.columns:
        return None

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    start = pd.to_datetime(ts, unit="s", utc=True)
    end = start + pd.Timedelta(minutes=horizon_minutes)
    window = work[(work["timestamp"] >= start) & (work["timestamp"] <= end)].reset_index(drop=True)
    return window if not window.empty else None


def _label_episode(episode: dict, *, interval: str, horizon_minutes: int, target_r: float) -> dict | None:
    future = _future_window(episode["coin"], episode["ts"], interval=interval, horizon_minutes=horizon_minutes)
    if future is None or future.empty:
        return None

    first = future.iloc[0]
    try:
        base = float(first["open"] or first["close"])
    except Exception:
        return None
    if base <= 0:
        return None

    risk_pct = float(episode["risk_pct"] or 0.0)
    if risk_pct <= 0:
        return None

    target_move = risk_pct * float(target_r)
    stop_move = risk_pct
    direction = episode["action"]

    max_favorable = 0.0
    max_adverse = 0.0
    outcome = 0
    hit_reason = "expired"

    for candle in future.itertuples(index=False):
        high = float(getattr(candle, "high", 0.0) or 0.0)
        low = float(getattr(candle, "low", 0.0) or 0.0)
        if direction == "LONG":
            favorable = max(0.0, (high - base) / base)
            adverse = max(0.0, (base - low) / base)
        else:
            favorable = max(0.0, (base - low) / base)
            adverse = max(0.0, (high - base) / base)

        max_favorable = max(max_favorable, favorable)
        max_adverse = max(max_adverse, adverse)

        if adverse >= stop_move and favorable >= target_move:
            outcome = 0
            hit_reason = "both_same_candle_stop_first"
            break
        if adverse >= stop_move:
            outcome = 0
            hit_reason = "stop_hit"
            break
        if favorable >= target_move:
            outcome = 1
            hit_reason = "target_hit"
            break

    labeled = dict(episode)
    labeled.update({
        "outcome": outcome,
        "hit_reason": hit_reason,
        "max_favorable_pct": round(max_favorable * 100.0, 4),
        "max_adverse_pct": round(max_adverse * 100.0, 4),
        "target_r": float(target_r),
        "horizon_minutes": int(horizon_minutes),
    })
    return labeled


def _summarize_families(rows: list[dict]) -> list[dict]:
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        buckets[(row["coin"], row["action"])].append(row)

    summary: list[dict] = []
    for (coin, action), items in sorted(buckets.items()):
        wins = sum(int(item["outcome"]) for item in items)
        total = len(items)
        summary.append({
            "family": f"{coin}:{action}",
            "coin": coin,
            "action": action,
            "samples": total,
            "wins": wins,
            "win_rate": round(wins / total, 4) if total else 0.0,
            "avg_prob": round(sum(item["prob"] for item in items) / total, 4),
            "avg_uncertainty": round(sum(item["unc"] for item in items) / total, 4),
            "avg_orderbook_score": round(sum(item["orderbook_score"] for item in items) / total, 4),
        })
    return summary


def _sweep_rules(rows: list[dict]) -> list[dict]:
    results: list[dict] = []
    for min_prob in (0.90, 0.92, 0.94):
        for max_unc in (0.20, 0.24, 0.28):
            for min_conf in ("MEDIUM", "HIGH"):
                for min_thesis in ("MEDIUM", "HIGH"):
                    for require_breakout in (True,):
                        filtered = []
                        for row in rows:
                            if row["prob"] < min_prob:
                                continue
                            if row["unc"] > max_unc:
                                continue
                            if _confidence_rank(row["confidence"]) < _confidence_rank(min_conf):
                                continue
                            if _confidence_rank(row["thesis_quality"]) < _confidence_rank(min_thesis):
                                continue
                            if require_breakout and row["breakout"] not in {
                                "confirmed_bullish_breakout",
                                "persistent_bullish_breakout",
                                "confirmed_bearish_breakdown",
                                "persistent_bearish_breakdown",
                            }:
                                continue
                            filtered.append(row)

                        if len(filtered) < 3:
                            continue
                        wins = sum(int(item["outcome"]) for item in filtered)
                        total = len(filtered)
                        results.append({
                            "samples": total,
                            "wins": wins,
                            "win_rate": round(wins / total, 4),
                            "min_prob": min_prob,
                            "max_unc": max_unc,
                            "min_confidence": min_conf,
                            "min_thesis_quality": min_thesis,
                            "require_breakout": require_breakout,
                        })
    results.sort(key=lambda item: (-item["win_rate"], -item["samples"], -item["min_prob"], item["max_unc"]))
    return results[:8]


def build_report(
    *,
    data_dir: Path,
    target_r: float,
    horizon_minutes: int,
    interval: str,
    dedupe_minutes: int,
) -> dict:
    decisions = _load_directional_decisions(data_dir)
    episodes = _collapse_episodes(decisions, dedupe_minutes=dedupe_minutes)
    labeled = [
        labeled_row
        for labeled_row in (
            _label_episode(row, interval=interval, horizon_minutes=horizon_minutes, target_r=target_r)
            for row in episodes
        )
        if labeled_row is not None
    ]

    family_summary = _summarize_families(labeled)
    toxic_families = [
        item for item in family_summary
        if item["samples"] >= 3 and item["win_rate"] < 0.35
    ]
    promising_families = [
        item for item in family_summary
        if item["samples"] >= 2 and item["win_rate"] >= 0.60
    ]

    return {
        "data_dir": str(data_dir),
        "generated_at": int(time.time()),
        "target_r": target_r,
        "horizon_minutes": horizon_minutes,
        "interval": interval,
        "decision_rows": len(decisions),
        "episodes": len(episodes),
        "labeled_episodes": len(labeled),
        "overall_win_rate": round(
            (sum(int(row["outcome"]) for row in labeled) / len(labeled)) if labeled else 0.0,
            4,
        ),
        "coin_counts": dict(Counter(row["coin"] for row in labeled)),
        "family_summary": family_summary,
        "promising_families": promising_families,
        "toxic_families": toxic_families,
        "best_rules": _sweep_rules(labeled),
    }


def _print_report(report: dict) -> None:
    print(f"Data dir:        {report['data_dir']}")
    print(f"Directional rows: {report['decision_rows']}")
    print(f"Episodes:         {report['episodes']}")
    print(f"Labeled episodes: {report['labeled_episodes']}")
    print(f"Overall WR:       {report['overall_win_rate'] * 100:.1f}%")
    print(f"Coins:            {report['coin_counts']}")
    print("")
    print("Toxic families")
    for item in report["toxic_families"][:6]:
        print(
            f"  - {item['family']}: {item['wins']}/{item['samples']} "
            f"({item['win_rate'] * 100:.1f}% WR)"
        )
    print("")
    print("Promising families")
    for item in report["promising_families"][:6]:
        print(
            f"  - {item['family']}: {item['wins']}/{item['samples']} "
            f"({item['win_rate'] * 100:.1f}% WR)"
        )
    print("")
    print("Best rule presets")
    for item in report["best_rules"][:5]:
        print(
            f"  - WR {item['win_rate'] * 100:.1f}% on {item['samples']} samples | "
            f"p>={item['min_prob']:.2f}, unc<={item['max_unc']:.2f}, "
            f"conf>={item['min_confidence']}, thesis>={item['min_thesis_quality']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay recent decision history and score elite setup families.")
    parser.add_argument("--data-dir", default="", help="Directory containing decision_dataset.jsonl")
    parser.add_argument("--interval", default="5m", choices=sorted(INTERVAL_SECONDS), help="Replay interval")
    parser.add_argument("--horizon-minutes", type=int, default=720, help="How far forward to replay each episode")
    parser.add_argument("--target-r", type=float, default=0.25, help="Target multiple to count as a win")
    parser.add_argument("--dedupe-minutes", type=int, default=30, help="Collapse repeated setup cycles into one episode")
    parser.add_argument("--output", default="", help="Optional JSON path for the report")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else _default_data_dir()
    report = build_report(
        data_dir=data_dir,
        target_r=float(args.target_r),
        horizon_minutes=int(args.horizon_minutes),
        interval=str(args.interval),
        dedupe_minutes=int(args.dedupe_minutes),
    )
    _print_report(report)

    output_path = Path(args.output).expanduser() if args.output else (data_dir / "precision_lab_report.json")
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print("")
    print(f"Saved report: {output_path}")


if __name__ == "__main__":
    main()
