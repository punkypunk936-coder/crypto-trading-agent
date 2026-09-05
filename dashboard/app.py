"""
dashboard/app.py — Trading agent dashboard.

Works in two modes:
  LOCAL:  reads state.json + trades_log.csv written by the agent
  REMOTE: receives state via POST /api/push (agent pushes each cycle)

Start locally:   python3 dashboard/app.py
Deploy (Railway): set PORT env var, agent pushes to your Railway URL
"""

import base64
import gzip
import io
import json
import hmac
import os
import re
import shutil
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import decision_dataset
import ask_call_learning
import asia_session
import global_market_context
import earnings_session
import us_market_context
from flask import Flask, render_template, jsonify, request, abort, send_file, Response
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
    EARNINGS_SESSION_JSON,
    KILL_FILE,
    LLM_REFEREE_REPORT_JSON,
    MISSED_MOVE_REPORT_JSON,
    POLICY_HEALTH_REPORT_JSON,
    PLAYBOOK_DISTILLER_REPORT_JSON,
    PROACTIVE_TRADER_REPORT_JSON,
    STATE_JSON,
    TRADE_REVIEWS_JSON,
    TRADES_CSV,
)
from dashboard.snapshot import (
    DASHBOARD_SCHEMA_VERSION,
    augment_state,
    build_dashboard_snapshot,
    default_control,
    default_state,
    normalize_control,
)
from tradexyz_profile import build_xyz_section

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
POLICY_HEALTH_REPORT = POLICY_HEALTH_REPORT_JSON
PROACTIVE_TRADER_REPORT = PROACTIVE_TRADER_REPORT_JSON
HOSTED_INDEX = CODE_ROOT / "netlify-dashboard" / "public" / "index.html"
HOSTED_ASK_CONFIDENCE = (
    CODE_ROOT / "netlify-dashboard" / "public" / "ask-confidence.js"
)
SNAPSHOT_REFRESH_GRACE_SECONDS = max(
    0.0,
    float(os.environ.get("DASHBOARD_SNAPSHOT_REFRESH_GRACE_SECONDS", "20") or 20),
)

# Secret token for push endpoint (set DASHBOARD_TOKEN env var for security)
PUSH_TOKEN = os.environ.get("DASHBOARD_TOKEN", "")
DASHBOARD_BASIC_AUTH = os.environ.get("DASHBOARD_BASIC_AUTH", "")
PUBLIC_READ_ONLY = str(os.environ.get("DASHBOARD_PUBLIC_READ_ONLY", "")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_PUSH_BYTES = max(
    1_000_000,
    int(os.environ.get("DASHBOARD_MAX_PUSH_BYTES", "12000000") or 12_000_000),
)
MAX_DECOMPRESSED_PUSH_BYTES = max(
    10_000_000,
    int(os.environ.get("DASHBOARD_MAX_DECOMPRESSED_PUSH_BYTES", "64000000") or 64_000_000),
)
PUSH_CHUNKS_DIR = SNAPSHOT.parent / "push_chunks"
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_PUSH_BYTES

# In-memory store for remote-pushed snapshot
_remote_state = {"snapshot": None}
_local_snapshot_cache = {
    "snapshot": None,
    "mtime_ns": None,
    "refreshing": False,
    "last_refresh_started": 0.0,
    "last_refresh_finished": 0.0,
    "last_refresh_error": "",
}
_lock = threading.Lock()


@app.before_request
def _require_dashboard_basic_auth():
    push_paths = {"/api/push", "/api/push-chunk"}
    if PUBLIC_READ_ONLY and request.method not in {"GET", "HEAD", "OPTIONS"} and request.path not in push_paths:
        abort(403, "Public dashboard is read-only")
    if not DASHBOARD_BASIC_AUTH or request.path in push_paths:
        return None
    auth = request.authorization
    supplied = f"{auth.username}:{auth.password}" if auth else ""
    if hmac.compare_digest(supplied, DASHBOARD_BASIC_AUTH):
        return None
    return Response(
        "Authentication required",
        401,
        {"WWW-Authenticate": 'Basic realm="Trading Dashboard"'},
    )


def _require_push_token() -> None:
    if not PUSH_TOKEN:
        return
    supplied = request.headers.get("X-Token", "")
    if not hmac.compare_digest(supplied, PUSH_TOKEN):
        abort(403, "Invalid token")


def _decode_push_payload(data: object) -> dict:
    if not isinstance(data, dict):
        abort(400, "JSON object required")
    if data.get("encoding") != "gzip-base64":
        return data

    encoded = data.get("payload")
    if not isinstance(encoded, str) or not encoded:
        abort(400, "Missing compressed payload")
    try:
        compressed = base64.b64decode(encoded, validate=True)
        with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as archive:
            raw = archive.read(MAX_DECOMPRESSED_PUSH_BYTES + 1)
        if len(raw) > MAX_DECOMPRESSED_PUSH_BYTES:
            abort(413, "Decompressed dashboard payload is too large")
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        if hasattr(exc, "code"):
            raise
        abort(400, f"Invalid compressed payload: {exc}")
    if not isinstance(decoded, dict):
        abort(400, "Decoded payload must be a JSON object")
    return decoded

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
        return decision_dataset.load_decisions(limit=5000, data_dir=history_dir)
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


def _load_policy_health_report_local() -> dict:
    if POLICY_HEALTH_REPORT.exists():
        try:
            return json.loads(POLICY_HEALTH_REPORT.read_text())
        except Exception:
            pass
    return {}


def _load_proactive_trader_report_local() -> dict:
    if PROACTIVE_TRADER_REPORT.exists():
        try:
            return json.loads(PROACTIVE_TRADER_REPORT.read_text())
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


def _snapshot_is_prebuilt(payload: dict) -> bool:
    return bool(
        isinstance(payload, dict)
        and isinstance(payload.get("state"), dict)
        and (
            "stats" in payload
            or "runtime" in payload
            or "action_board" in payload
        )
    )


def _cache_local_snapshot(snapshot: dict | None, *, mtime_ns: int | None = None) -> dict | None:
    if not isinstance(snapshot, dict) or "state" not in snapshot:
        return None
    cached = dict(snapshot)
    cached["control"] = normalize_control(cached.get("control"))
    with _lock:
        _local_snapshot_cache["snapshot"] = cached
        _local_snapshot_cache["mtime_ns"] = mtime_ns
    return cached


def _load_snapshot_local() -> dict | None:
    if not SNAPSHOT.exists():
        return None
    try:
        mtime_ns = SNAPSHOT.stat().st_mtime_ns
    except Exception:
        return None
    with _lock:
        cached_snapshot = _local_snapshot_cache.get("snapshot")
        cached_mtime_ns = _local_snapshot_cache.get("mtime_ns")
    if isinstance(cached_snapshot, dict) and cached_mtime_ns == mtime_ns:
        return cached_snapshot
    try:
        payload = json.loads(SNAPSHOT.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict) or "state" not in payload:
        return None
    if _snapshot_is_prebuilt(payload) and "daily_radar" in payload:
        shaped_state = augment_state(payload.get("state") or {})
        xyz_rows = list((payload.get("xyz") or {}).get("items") or [])
        xyz_needs_upgrade = "xyz" not in payload or any(
            isinstance(row, dict) and not str(row.get("structural_thesis") or "").strip()
            for row in xyz_rows
        )
        if xyz_needs_upgrade:
            try:
                payload["xyz"] = build_xyz_section(shaped_state, payload.get("action_board") or {})
            except Exception:
                payload["xyz"] = {"title": "xyz", "summary": {}, "items": [], "segments": []}
        if "asia_session" not in payload:
            payload["asia_session"] = asia_session.build_asia_session(shaped_state)
        try:
            payload["us_market_context"] = us_market_context.build_us_market_context(
                shaped_state,
                market_map=payload.get("market_map") or {},
                asia_context=payload.get("asia_session") or {},
            )
        except Exception:
            payload.setdefault("us_market_context", {"active": False, "benchmarks": []})
        try:
            payload["global_market_context"] = global_market_context.build_global_market_context(
                shaped_state,
                market_map=payload.get("market_map") or {},
                asia_context=payload.get("asia_session") or {},
                us_context=payload.get("us_market_context") or {},
            )
        except Exception:
            payload.setdefault(
                "global_market_context",
                payload.get("us_market_context") or {"active": False, "benchmarks": []},
            )
        if "earnings_session" not in payload:
            payload["earnings_session"] = earnings_session.build_earnings_session(
                shaped_state,
                payload.get("daily_radar") or {},
                ledger_path=EARNINGS_SESSION_JSON,
            )
        if xyz_needs_upgrade:
            try:
                _save_snapshot_local(payload)
                return payload
            except Exception:
                pass
        return _cache_local_snapshot(payload, mtime_ns=mtime_ns)
    hydrated = _hydrate_snapshot_payload({"snapshot": payload}, server_timestamp=payload.get("server_time"))
    return _cache_local_snapshot(hydrated, mtime_ns=mtime_ns)


def _save_snapshot_local(snapshot: dict) -> None:
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=SNAPSHOT.parent,
            prefix=f".{SNAPSHOT.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(snapshot, handle, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
            tmp_path = Path(handle.name)
        tmp_path.replace(SNAPSHOT)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    try:
        mtime_ns = SNAPSHOT.stat().st_mtime_ns
    except Exception:
        mtime_ns = None
    _cache_local_snapshot(snapshot, mtime_ns=mtime_ns)


def _state_revision(state: dict | None) -> tuple[int, str]:
    safe_state = dict(state or {})
    try:
        cycle = int(safe_state.get("cycle_number") or 0)
    except (TypeError, ValueError):
        cycle = 0
    return cycle, str(safe_state.get("last_cycle") or "")


def _snapshot_revision(snapshot: dict | None) -> tuple[int, int, str]:
    safe_snapshot = dict(snapshot or {})
    try:
        version = int(safe_snapshot.get("version") or 0)
    except (TypeError, ValueError):
        version = 0
    cycle, last_cycle = _state_revision(safe_snapshot.get("state"))
    updated_at = str(safe_snapshot.get("updatedAt") or safe_snapshot.get("server_time") or last_cycle)
    return version, cycle, updated_at


def _snapshot_needs_refresh(snapshot: dict | None = None) -> bool:
    if isinstance(snapshot, dict):
        try:
            snapshot_schema = int(snapshot.get("schemaVersion") or 0)
        except (TypeError, ValueError):
            snapshot_schema = 0
        if snapshot_schema < DASHBOARD_SCHEMA_VERSION:
            return True
    if not SNAPSHOT.exists():
        return True
    try:
        snapshot_mtime = SNAPSHOT.stat().st_mtime
    except Exception:
        return True
    newer_source_mtimes = []
    for path in (STATE, LOG, CONTROL, MARKET_MAP, REVIEWS, DECISION_REVIEW, CHALLENGER_REPORT, MISSED_MOVE_REPORT, ASSET_DOSSIERS, LLM_REFEREE_REPORT, PLAYBOOK_DISTILLER_REPORT, POLICY_HEALTH_REPORT, PROACTIVE_TRADER_REPORT):
        try:
            source_mtime = path.stat().st_mtime if path.exists() else 0.0
            if source_mtime > snapshot_mtime:
                newer_source_mtimes.append(source_mtime)
        except Exception:
            continue
    if newer_source_mtimes:
        newest_source_age = max(0.0, time.time() - max(newer_source_mtimes))
        if newest_source_age >= SNAPSHOT_REFRESH_GRACE_SECONDS:
            return True
        return False
    # A refresh worker and the agent can finish in the opposite order. File
    # mtimes alone cannot detect an older snapshot written after newer state.
    if isinstance(snapshot, dict) and STATE.exists():
        current_state = _load_state_local()
        if _state_revision(current_state) > _state_revision(snapshot.get("state")):
            return True
    return False


def _build_local_snapshot(server_timestamp: str | None = None) -> dict:
    state = augment_state(_load_state_local())
    tracked_coins = market_map_store.tracked_coins_from_state(state)
    effective_market_map = market_map_store.build_effective_market_map(
        tracked_coins,
        base_map=_load_market_map_local(),
    )
    snapshot = build_dashboard_snapshot(
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
        policy_health_report=_load_policy_health_report_local(),
        proactive_trader_report=_load_proactive_trader_report_local(),
        earnings_ledger_path=EARNINGS_SESSION_JSON,
        server_timestamp=server_timestamp,
    )
    return ask_call_learning.refresh_snapshot_accountability(snapshot, settle=False)


def _refresh_local_snapshot_worker(server_timestamp: str | None = None) -> None:
    try:
        snapshot = _build_local_snapshot(server_timestamp=server_timestamp)
        _save_snapshot_local(snapshot)
        error = ""
    except Exception as exc:
        error = str(exc)
    finally:
        with _lock:
            _local_snapshot_cache["refreshing"] = False
            _local_snapshot_cache["last_refresh_finished"] = time.monotonic()
            _local_snapshot_cache["last_refresh_error"] = error


def _queue_local_snapshot_refresh(server_timestamp: str | None = None, *, force: bool = False) -> bool:
    now = time.monotonic()
    with _lock:
        if _remote_state["snapshot"] is not None:
            return False
        if _local_snapshot_cache["refreshing"]:
            return False
        if not force and (now - float(_local_snapshot_cache.get("last_refresh_started") or 0.0)) < 3.0:
            return False
        _local_snapshot_cache["refreshing"] = True
        _local_snapshot_cache["last_refresh_started"] = now
    thread = threading.Thread(
        target=_refresh_local_snapshot_worker,
        kwargs={"server_timestamp": server_timestamp},
        daemon=True,
        name="dashboard-snapshot-refresh",
    )
    thread.start()
    return True


def _hydrate_snapshot_payload(data: dict, *, server_timestamp: str | None = None) -> dict:
    snapshot = data.get("snapshot")
    if isinstance(snapshot, dict) and "state" in snapshot:
        if _snapshot_is_prebuilt(snapshot) and "daily_radar" in snapshot:
            # A pushed dashboard snapshot is already the canonical public
            # schema. Preserve every current and future field verbatim rather
            # than rebuilding it through an older server schema.
            hydrated = dict(snapshot)
            hydrated["state"] = augment_state(snapshot.get("state") or {})
            hydrated["control"] = normalize_control(snapshot.get("control"))
            hydrated.setdefault("schemaVersion", 1)
            hydrated.setdefault("server_time", server_timestamp or snapshot.get("updatedAt"))
            return hydrated
        return build_dashboard_snapshot(
            snapshot.get("state"),
            snapshot.get("trades", []),
            snapshot.get("control"),
            snapshot.get("market_map"),
            snapshot.get("trade_reviews"),
            decision_review_report=snapshot.get("decision_review_report"),
            challenger_report=snapshot.get("challenger_report"),
            missed_move_report=snapshot.get("missed_move_report"),
            asset_dossiers=snapshot.get("asset_dossiers"),
            llm_referee_report=snapshot.get("llm_referee_report"),
            playbook_distiller_report=snapshot.get("playbook_distiller_report"),
            policy_health_report=snapshot.get("policy_health_report"),
            proactive_trader_report=snapshot.get("proactive_trader_report"),
            server_timestamp=server_timestamp or snapshot.get("server_time"),
        )
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
        policy_health_report=data.get("policy_health_report"),
        proactive_trader_report=data.get("proactive_trader_report"),
        server_timestamp=server_timestamp,
    )


def _accept_dashboard_payload(data: dict) -> tuple[dict, int]:
    if "snapshot" not in data and "state" not in data:
        abort(400, "Missing snapshot/state in payload")

    snapshot = _hydrate_snapshot_payload(data)
    with _lock:
        current_remote = _remote_state.get("snapshot")
    current = current_remote if isinstance(current_remote, dict) else _load_snapshot_local()
    incoming_revision = _snapshot_revision(snapshot)
    current_revision = _snapshot_revision(current)
    if isinstance(current, dict) and incoming_revision < current_revision:
        return {
            "ok": True,
            "accepted": False,
            "reason": "stale_snapshot",
            "incomingRevision": incoming_revision,
            "currentRevision": current_revision,
        }, 202

    _save_snapshot_local(snapshot)
    with _lock:
        _remote_state["snapshot"] = snapshot

    state = snapshot.get("state") or {}
    return {
        "ok": True,
        "accepted": True,
        "cycle": state.get("cycle_number", 0),
        "version": snapshot.get("version"),
        "updatedAt": snapshot.get("updatedAt"),
    }, 200


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


@app.route("/ask-confidence.js")
def ask_confidence_script():
    response = send_file(HOSTED_ASK_CONFIDENCE, mimetype="application/javascript")
    response.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
    return response


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
    needs_refresh = snapshot is None or _snapshot_needs_refresh(snapshot)
    if snapshot is not None:
        if needs_refresh:
            _queue_local_snapshot_refresh()
        if SNAPSHOT.exists():
            response = send_file(
                SNAPSHOT,
                mimetype="application/json",
                conditional=True,
                max_age=0,
            )
            response.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
            return response
        return jsonify(snapshot)
    snapshot = _build_local_snapshot()
    _save_snapshot_local(snapshot)
    response = send_file(
        SNAPSHOT,
        mimetype="application/json",
        conditional=True,
        max_age=0,
    )
    response.headers["Cache-Control"] = "no-cache, max-age=0, must-revalidate"
    return response


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


@app.route("/api/ask-forecast", methods=["GET", "POST"])
def api_ask_forecast():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        incoming = data.get("records") if isinstance(data, dict) else []
        if not isinstance(incoming, list):
            return jsonify({"ok": False, "error": "records must be an array"}), 400
        rows = ask_call_learning.upsert_forecasts(incoming[:500])
    else:
        rows = ask_call_learning.upsert_forecasts([])
    return jsonify({
        "ok": True,
        "records": rows[-500:],
        "summary": ask_call_learning.forecast_summary(rows),
    })


@app.route("/api/push", methods=["POST"])
def api_push():
    """Accept the agent's canonical snapshot and persist it durably."""
    _require_push_token()
    data = _decode_push_payload(request.get_json(silent=True))
    response, status = _accept_dashboard_payload(data)
    return jsonify(response), status


@app.route("/api/push-chunk", methods=["POST"])
def api_push_chunk():
    """Assemble oversized snapshots without relying on process memory."""
    _require_push_token()
    data = request.get_json(silent=True) or {}
    session_id = str(data.get("session_id") or "")
    if not _SAFE_SESSION_ID.fullmatch(session_id):
        abort(400, "Invalid session_id")
    try:
        chunk_index = int(data.get("chunk_index"))
        chunk_count = int(data.get("chunk_count"))
    except (TypeError, ValueError):
        abort(400, "Invalid chunk metadata")
    chunk = data.get("chunk")
    if not isinstance(chunk, str):
        abort(400, "Missing chunk")
    if not (1 <= chunk_count <= 100) or not (0 <= chunk_index < chunk_count):
        abort(400, "Chunk index/count out of range")
    if len(chunk) > 1_000_000:
        abort(413, "Chunk is too large")

    PUSH_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for candidate in PUSH_CHUNKS_DIR.iterdir():
        try:
            if candidate.is_dir() and now - candidate.stat().st_mtime > 3600:
                shutil.rmtree(candidate, ignore_errors=True)
        except OSError:
            continue

    session_dir = PUSH_CHUNKS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    part_path = session_dir / f"{chunk_index:04d}.part"
    temp_path = session_dir / f".{chunk_index:04d}.tmp"
    temp_path.write_text(chunk, encoding="utf-8")
    temp_path.replace(part_path)

    part_paths = [session_dir / f"{index:04d}.part" for index in range(chunk_count)]
    received = sum(path.exists() for path in part_paths)
    if received < chunk_count:
        return jsonify({
            "ok": True,
            "assembled": False,
            "session_id": session_id,
            "received": received,
            "chunk_count": chunk_count,
        })

    try:
        payload_text = "".join(path.read_text(encoding="utf-8") for path in part_paths)
        if len(payload_text.encode("utf-8")) > MAX_DECOMPRESSED_PUSH_BYTES:
            abort(413, "Assembled dashboard payload is too large")
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            abort(400, "Assembled payload must be a JSON object")
        response, status = _accept_dashboard_payload(payload)
        response.update({"assembled": True, "session_id": session_id})
        return jsonify(response), status
    finally:
        shutil.rmtree(session_dir, ignore_errors=True)


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
        state = augment_state(state)
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
            state = augment_state((_remote_state["snapshot"] or {}).get("state") or {})
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
@app.route("/healthz")
def health():
    snapshot = _load_snapshot_local()
    return jsonify({
        "status": "ok",
        "version": (snapshot or {}).get("version"),
        "updatedAt": (snapshot or {}).get("updatedAt"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"\n  Trading Agent Dashboard")
    print(f"  Open in your browser: http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
