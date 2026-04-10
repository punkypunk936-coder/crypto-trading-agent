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
import csv
import threading
from datetime import datetime

from flask import Flask, render_template, jsonify, request, abort, send_file
from paths import (
    CODE_ROOT,
    CONTROL_JSON,
    DASHBOARD_SNAPSHOT_JSON,
    KILL_FILE,
    STATE_JSON,
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
    if not LOG.exists():
        return []
    try:
        with open(LOG, newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


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
    for path in (STATE, LOG, CONTROL):
        try:
            if path.exists() and path.stat().st_mtime > snapshot_mtime:
                return True
        except Exception:
            continue
    return False


def _build_local_snapshot(server_timestamp: str | None = None) -> dict:
    return build_dashboard_snapshot(
        _load_state_local(),
        _load_trades_local(),
        _load_control_local(),
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


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  Trading Agent Dashboard")
    print(f"  Open in your browser: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
