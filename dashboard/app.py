"""
dashboard/app.py — Trading agent dashboard.

Works in two modes:
  LOCAL:  reads state.json + trades_log.csv written by the agent
  REMOTE: receives state via POST /api/push (agent pushes each cycle)

Start locally:   python3 dashboard/app.py
Deploy (Railway): set PORT env var, agent pushes to your Railway URL
"""

import json
import os
import threading
from datetime import datetime

import decision_dataset
from flask import Flask, render_template, jsonify, request, abort, send_file
import market_map as market_map_store
import trade_dataset
import trade_logger
import trade_review as trade_review_store
import tradexyz_volume
from paths import (
    ASSET_DOSSIERS_JSON,
    CHALLENGER_MODEL_JSON,
    CODE_ROOT,
    CONTROL_JSON,
    DAILY_MARKET_MAP_JSON,
    DASHBOARD_SNAPSHOT_JSON,
    DECISION_REVIEW_REPORT_JSON,
    KILL_FILE,
    LLM_REFEREE_REPORT_JSON,
    MISSED_MOVE_REPORT_JSON,
    PLAYBOOK_DISTILLER_REPORT_JSON,
    STATE_JSON,
    TRADE_REVIEWS_JSON,
    TRADES_CSV,
)
from dashboard.snapshot import (
    build_dashboard_snapshot,
    default_control,
    default_state,
    normalize_control,
)

STATE    = STATE_JSON
LOG      = TRADES_CSV
CONTROL  = CONTROL_JSON
SNAPSHOT = DASHBOARD_SNAPSHOT_JSON
KILL     = KILL_FILE
MARKET_MAP = DAILY_MARKET_MAP_JSON
REVIEWS = TRADE_REVIEWS_JSON
DECISION_REVIEW = DECISION_REVIEW_REPORT_JSON
CHALLENGER_REPORT = CHALLENGER_MODEL_JSON
MISSED_MOVE_REPORT = MISSED_MOVE_REPORT_JSON
ASSET_DOSSIERS = ASSET_DOSSIERS_JSON
LLM_REFEREE_REPORT = LLM_REFEREE_REPORT_JSON
PLAYBOOK_DISTILLER_REPORT = PLAYBOOK_DISTILLER_REPORT_JSON
HOSTED_INDEX = CODE_ROOT / "netlify-dashboard" / "public" / "index.html"

# Secret token for push endpoint (set DASHBOARD_TOKEN env var for security)
PUSH_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")

app = Flask(__name__, template_folder="templates", static_folder="static")

# In-memory store for remote-pushed snapshot
_remote_state = {"snapshot": None}
_lock = threading.Lock()

def _load_state_local() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return default_state()


def _load_trades_local() -> list:
    try:
        return trade_logger.read_closed_trades()
    except Exception:
        return []


def _load_market_map_local() -> dict:
    try:
        return market_map_store.load_market_map()
    except Exception:
        return market_map_store.default_market_map()


def _load_trade_dataset_local() -> list:
    try:
        history_dir = trade_dataset.resolve_richest_history_data_dir()
        return trade_dataset.load_closed_trades(limit=250, data_dir=history_dir)
    except Exception:
        return []


def _load_decision_dataset_local() -> list:
    try:
        history_dir = decision_dataset.resolve_richest_decision_data_dir()
        return decision_dataset.load_decisions(limit=25000, data_dir=history_dir)
    except Exception:
        return []


def _load_trade_reviews_local() -> dict:
    try:
        return trade_review_store.load_reviews()
    except Exception:
        return trade_review_store.default_reviews()


def _load_decision_review_local() -> dict:
    if DECISION_REVIEW.exists():
        try:
            return json.loads(DECISION_REVIEW.read_text())
        except Exception:
            pass
    return {}


def _load_challenger_report_local() -> dict:
    if CHALLENGER_REPORT.exists():
        try:
            return json.loads(CHALLENGER_REPORT.read_text())
        except Exception:
            pass
    return {}


def _load_missed_move_report_local() -> dict:
    if MISSED_MOVE_REPORT.exists():
        try:
            return json.loads(MISSED_MOVE_REPORT.read_text())
        except Exception:
            pass
    return {}


def _load_asset_dossiers_local() -> dict:
    if ASSET_DOSSIERS.exists():
        try:
            return json.loads(ASSET_DOSSIERS.read_text())
        except Exception:
            pass
    return {}


def _load_llm_referee_report_local() -> dict:
    if LLM_REFEREE_REPORT.exists():
        try:
            return json.loads(LLM_REFEREE_REPORT.read_text())
        except Exception:
            pass
    return {}


def _load_playbook_distiller_report_local() -> dict:
    if PLAYBOOK_DISTILLER_REPORT.exists():
        try:
            return json.loads(PLAYBOOK_DISTILLER_REPORT.read_text())
        except Exception:
            pass
    return {}


def _load_control_local() -> dict:
    if CONTROL.exists():
        try:
            return normalize_control(json.loads(CONTROL.read_text()))
        except Exception:
            pass
    return default_control()


def _save_control_local(control: dict) -> None:
    CONTROL.write_text(json.dumps(normalize_control(control), indent=2))


def _load_snapshot_local() -> dict | None:
    if not SNAPSHOT.exists():
        return None
    try:
        payload = json.loads(SNAPSHOT.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict) or "state" not in payload:
        return None
    return payload


def _save_snapshot_local(snapshot: dict) -> None:
    SNAPSHOT.write_text(json.dumps(snapshot, indent=2))


def _snapshot_needs_refresh() -> bool:
    if not SNAPSHOT.exists():
        return True
    try:
        snapshot_mtime = SNAPSHOT.stat().st_mtime
    except Exception:
        return True
    for path in (STATE, LOG, CONTROL, MARKET_MAP, REVIEWS, DECISION_REVIEW, CHALLENGER_REPORT, MISSED_MOVE_REPORT, ASSET_DOSSIERS, LLM_REFEREE_REPORT, PLAYBOOK_DISTILLER_REPORT):
        try:
            if path.exists() and path.stat().st_mtime > snapshot_mtime:
                return True
        except Exception:
            continue
    return False


def _build_local_snapshot(server_timestamp: str | None = None) -> dict:
    state = _load_state_local()
    tracked_coins = market_map_store.tracked_coins_from_state(state)
    effective_market_map = market_map_store.build_effective_market_map(
        tracked_coins,
        base_map=_load_market_map_local(),
    )
    return build_dashboard_snapshot(
        state,
        _load_trades_local(),
        _load_control_local(),
        market_map=effective_market_map,
        trade_reviews=_load_trade_reviews_local(),
        trade_dataset_records=_load_trade_dataset_local(),
        decision_dataset_records=_load_decision_dataset_local(),
        decision_review_report=_load_decision_review_local(),
        challenger_report=_load_challenger_report_local(),
        missed_move_report=_load_missed_move_report_local(),
        asset_dossiers=_load_asset_dossiers_local(),
        llm_referee_report=_load_llm_referee_report_local(),
        playbook_distiller_report=_load_playbook_distiller_report_local(),
        server_timestamp=server_timestamp,
    )


def _hydrate_snapshot_payload(data: dict, *, server_timestamp: str | None = None) -> dict:
    snapshot = data.get("snapshot")
    if isinstance(snapshot, dict) and "state" in snapshot:
        payload = dict(snapshot)
        if server_timestamp:
            payload["server_time"] = server_timestamp
        return payload
    return build_dashboard_snapshot(
        data.get("state"),
        data.get("trades", []),
        data.get("control"),
        data.get("market_map"),
        data.get("trade_reviews"),
        decision_dataset_records=data.get("decision_dataset_records"),
        decision_review_report=data.get("decision_review_report"),
        challenger_report=data.get("challenger_report"),
        missed_move_report=data.get("missed_move_report"),
        asset_dossiers=data.get("asset_dossiers"),
        llm_referee_report=data.get("llm_referee_report"),
        playbook_distiller_report=data.get("playbook_distiller_report"),
        server_timestamp=server_timestamp,
    )


def _set_kill_state(snapshot: dict, *, active: bool, reason: str, requested_at: str) -> dict:
    updated = dict(snapshot or {})
    control = normalize_control(updated.get("control"))
    control["kill"] = {
        "active": active,
        "reason": reason if active else "",
        "requested_at": requested_at if active else None,
        "acknowledged_at": requested_at if not active else control["kill"].get("acknowledged_at"),
    }
    updated["control"] = control
    updated["server_time"] = requested_at
    return updated


@app.route("/")
def index():
    if HOSTED_INDEX.exists():
        return send_file(HOSTED_INDEX)
    return render_template("dashboard.html")


@app.route("/tradexyz-volume")
def tradexyz_volume_page():
    return render_template("tradexyz_volume.html")


@app.route("/api/state")
def api_state():
    # If a remote snapshot has been pushed, serve that exact payload.
    with _lock:
        if _remote_state["snapshot"] is not None:
            return jsonify(_remote_state["snapshot"])

    snapshot = _load_snapshot_local()
    if snapshot is None or _snapshot_needs_refresh():
        snapshot = _build_local_snapshot()
        _save_snapshot_local(snapshot)
    return jsonify(snapshot)


@app.route("/api/tradexyz-volume")
def api_tradexyz_volume():
    wallet = str(request.args.get("wallet", "")).strip()
    if not wallet:
        return jsonify({"ok": False, "error": "Missing wallet address."}), 400
    try:
        payload = tradexyz_volume.fetch_tradexyz_volume(wallet)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 502
    return jsonify({"ok": True, **payload})


@app.route("/api/push", methods=["POST"])
def api_push():
    """Agent pushes state here each cycle (for remote/Railway deployment)."""
    # Token check
    if PUSH_TOKEN:
        token = request.headers.get("X-Token", "")
        if token != PUSH_TOKEN:
            abort(403, "Invalid token")

    data = request.get_json(silent=True)
    if not data or ("snapshot" not in data and "state" not in data):
        abort(400, "Missing snapshot/state in payload")

    snapshot = _hydrate_snapshot_payload(data)

    with _lock:
        _remote_state["snapshot"] = snapshot

    state = snapshot.get("state") or {}
    return jsonify({"ok": True, "cycle": state.get("cycle_number", 0)})


@app.route("/api/kill", methods=["POST"])
def api_kill():
    data = request.get_json(silent=True) or {}
    active = bool(data.get("active", True))
    reason = str(data.get("reason", "Dashboard kill switch activated")).strip() or "Dashboard kill switch activated"
    requested_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with _lock:
        if _remote_state["snapshot"] is not None:
            snapshot = _set_kill_state(
                _remote_state["snapshot"],
                active=active,
                reason=reason,
                requested_at=requested_at,
            )
            _remote_state["snapshot"] = snapshot
            return jsonify({"ok": True, "control": snapshot["control"]})

    control = _load_control_local()
    control["kill"] = {
        "active": active,
        "reason": reason if active else "",
        "requested_at": requested_at if active else None,
        "acknowledged_at": requested_at if not active else control["kill"].get("acknowledged_at"),
    }
    _save_control_local(control)
    snapshot = _load_snapshot_local() or _build_local_snapshot(server_timestamp=requested_at)
    snapshot = _set_kill_state(snapshot, active=active, reason=reason, requested_at=requested_at)
    _save_snapshot_local(snapshot)
    if active:
        KILL.write_text(reason)
    else:
        KILL.unlink(missing_ok=True)
    return jsonify({"ok": True, "control": control})


@app.route("/api/market-map", methods=["GET", "POST"])
def api_market_map():
    if request.method == "GET":
        state = (_remote_state.get("snapshot") or {}).get("state") if isinstance(_remote_state.get("snapshot"), dict) else None
        if not isinstance(state, dict):
            state = _load_state_local()
        tracked_coins = market_map_store.tracked_coins_from_state(state)
        return jsonify(
            market_map_store.build_effective_market_map(
                tracked_coins,
                base_map=_load_market_map_local(),
            )
        )

    data = request.get_json(silent=True) or {}
    if data.get("delete") and data.get("coin"):
        payload = market_map_store.delete_market_map_entry(str(data.get("coin")))
    elif data.get("coin"):
        payload = market_map_store.upsert_market_map_entry(str(data.get("coin")), data)
    else:
        payload = market_map_store.save_market_map(data)
    with _lock:
        if _remote_state["snapshot"] is not None:
            state = (_remote_state["snapshot"] or {}).get("state") or {}
            tracked_coins = market_map_store.tracked_coins_from_state(state)
            effective_market_map = market_map_store.build_effective_market_map(
                tracked_coins,
                base_map=payload,
            )
            _remote_state["snapshot"]["market_map"] = effective_market_map
            _remote_state["snapshot"]["market_map_summary"] = market_map_store.review_summary(effective_market_map)
        else:
            _save_snapshot_local(_build_local_snapshot())
    return jsonify({"ok": True, "market_map": payload})


@app.route("/api/reviews", methods=["GET", "POST"])
def api_reviews():
    if request.method == "GET":
        return jsonify(_load_trade_reviews_local())

    data = request.get_json(silent=True) or {}
    payload = trade_review_store.upsert_review(data)
    review_summary = trade_review_store.review_summary(_load_trades_local())
    with _lock:
        if _remote_state["snapshot"] is not None:
            trades = _remote_state["snapshot"].get("trades") or []
            _remote_state["snapshot"]["trade_reviews"] = payload
            _remote_state["snapshot"]["review_summary"] = review_summary
            _remote_state["snapshot"]["trades"] = trade_review_store.merge_reviews_into_trades(trades)
        else:
            _save_snapshot_local(_build_local_snapshot())
    return jsonify({
        "ok": True,
        "trade_reviews": payload,
        "review_summary": review_summary,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  Trading Agent Dashboard")
    print(f"  Open in your browser: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
