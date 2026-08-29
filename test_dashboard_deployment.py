import base64
import gzip
import json

import dashboard.app as dashboard_app


def _snapshot(version: int, cycle: int, marker: str) -> dict:
    return {
        "schemaVersion": 4,
        "version": version,
        "updatedAt": f"2026-08-29T00:00:{cycle:02d}Z",
        "state": {
            "cycle_number": cycle,
            "last_cycle": f"2026-08-29 00:00:{cycle:02d}",
            "positions": [],
            "signals": {},
        },
        "stats": {},
        "runtime": {},
        "daily_radar": {"top_assets": []},
        "future_schema_field": {"marker": marker},
    }


def _configure_test_storage(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(dashboard_app, "SNAPSHOT", tmp_path / "dashboard_snapshot.json")
    monkeypatch.setattr(dashboard_app, "PUSH_CHUNKS_DIR", tmp_path / "push_chunks")
    monkeypatch.setattr(dashboard_app, "PUSH_TOKEN", "test-token")
    monkeypatch.setattr(dashboard_app, "PUBLIC_READ_ONLY", True)
    with dashboard_app._lock:
        dashboard_app._remote_state["snapshot"] = None
        dashboard_app._local_snapshot_cache.update({
            "snapshot": None,
            "mtime_ns": None,
            "refreshing": False,
            "last_refresh_started": 0.0,
            "last_refresh_finished": 0.0,
            "last_refresh_error": "",
        })


def test_compressed_push_persists_full_schema_and_rejects_stale(monkeypatch, tmp_path) -> None:
    _configure_test_storage(monkeypatch, tmp_path)
    client = dashboard_app.app.test_client()

    newest = {"snapshot": _snapshot(10, 10, "newest")}
    encoded = base64.b64encode(gzip.compress(json.dumps(newest).encode())).decode()
    response = client.post(
        "/api/push",
        json={"encoding": "gzip-base64", "payload": encoded},
        headers={"X-Token": "test-token"},
    )
    assert response.status_code == 200
    assert response.get_json()["accepted"] is True
    saved = json.loads(dashboard_app.SNAPSHOT.read_text())
    assert saved["future_schema_field"] == {"marker": "newest"}

    stale = client.post(
        "/api/push",
        json={"snapshot": _snapshot(9, 9, "stale")},
        headers={"X-Token": "test-token"},
    )
    assert stale.status_code == 202
    assert stale.get_json()["accepted"] is False
    assert json.loads(dashboard_app.SNAPSHOT.read_text())["version"] == 10


def test_chunked_push_assembles_on_disk_and_public_dashboard_is_read_only(monkeypatch, tmp_path) -> None:
    _configure_test_storage(monkeypatch, tmp_path)
    client = dashboard_app.app.test_client()
    payload_text = json.dumps({"snapshot": _snapshot(11, 11, "chunked")})
    midpoint = len(payload_text) // 2
    chunks = [payload_text[:midpoint], payload_text[midpoint:]]

    first = client.post(
        "/api/push-chunk",
        json={
            "session_id": "deployment-test",
            "chunk_index": 0,
            "chunk_count": 2,
            "chunk": chunks[0],
        },
        headers={"X-Token": "test-token"},
    )
    assert first.status_code == 200
    assert first.get_json()["assembled"] is False

    second = client.post(
        "/api/push-chunk",
        json={
            "session_id": "deployment-test",
            "chunk_index": 1,
            "chunk_count": 2,
            "chunk": chunks[1],
        },
        headers={"X-Token": "test-token"},
    )
    assert second.status_code == 200
    assert second.get_json()["assembled"] is True
    assert json.loads(dashboard_app.SNAPSHOT.read_text())["future_schema_field"]["marker"] == "chunked"

    blocked = client.post("/api/kill", json={"active": True})
    assert blocked.status_code == 403
