"""
hosted_state_sync.py — Git-backed fallback publisher for the public dashboard.

Why this exists:
- The local dashboard is the source of truth.
- The public Vercel dashboard used Vercel Blob for persistence.
- If hosted storage is unavailable, we still want the public dashboard to
  mirror the exact canonical snapshot the local agent writes every cycle.

This module publishes those JSON snapshot files into a dedicated Git branch
that the hosted dashboard can read directly from GitHub as a fallback source.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from logger import get_logger
from paths import DASHBOARD_STATE_SYNC_REPO

log = get_logger("hosted_state_sync")

DEFAULT_REMOTE_URL = "https://github.com/punkypunk936-coder/crypto-trading-agent.git"
DEFAULT_BRANCH = "codex/dashboard-state"
DEFAULT_PUBLIC_TAG = "dashboard-state-live"
DEFAULT_AUTHOR_NAME = "Punky Dashboard Sync"
DEFAULT_AUTHOR_EMAIL = "punkypunk936@gmail.com"


def is_enabled() -> bool:
    raw = str(os.environ.get("DASHBOARD_STATE_GIT_SYNC_ENABLED", "1") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def state_branch_ref() -> str:
    return str(os.environ.get("DASHBOARD_STATE_GIT_BRANCH", DEFAULT_BRANCH) or DEFAULT_BRANCH).strip()


def state_remote_url() -> str:
    return str(os.environ.get("DASHBOARD_STATE_GIT_URL", DEFAULT_REMOTE_URL) or DEFAULT_REMOTE_URL).strip()


def state_public_tag() -> str:
    return str(os.environ.get("DASHBOARD_STATE_GIT_TAG", DEFAULT_PUBLIC_TAG) or DEFAULT_PUBLIC_TAG).strip()


def github_raw_fallback_url(pathname: str) -> str:
    tag = state_public_tag()
    return (
        "https://raw.githubusercontent.com/punkypunk936-coder/crypto-trading-agent/"
        f"{tag}/{pathname}"
    )


def _run_git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
    )
    return (proc.stdout or "").strip()


def _git_ok(args: list[str], cwd: Path) -> bool:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


def _has_staged_dashboard_changes(repo_dir: Path) -> bool:
    proc = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", "dashboard", "README.md"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode != 0


def _wipe_worktree(repo_dir: Path) -> None:
    for entry in repo_dir.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink(missing_ok=True)


def _ensure_checkout(repo_dir: Path, remote_url: str, branch: str) -> None:
    if not (repo_dir / ".git").exists():
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", "--depth", "1", remote_url, str(repo_dir)], repo_dir.parent)

    _run_git(["config", "user.name", os.environ.get("DASHBOARD_STATE_GIT_AUTHOR_NAME", DEFAULT_AUTHOR_NAME)], repo_dir)
    _run_git(["config", "user.email", os.environ.get("DASHBOARD_STATE_GIT_AUTHOR_EMAIL", DEFAULT_AUTHOR_EMAIL)], repo_dir)

    remote_branch_exists = _git_ok(["ls-remote", "--exit-code", "--heads", "origin", branch], repo_dir)
    if remote_branch_exists:
        try:
            _run_git(["checkout", branch], repo_dir)
        except Exception:
            _run_git(["fetch", "origin", branch, "--depth", "1"], repo_dir)
            _run_git(["checkout", "-b", branch, "FETCH_HEAD"], repo_dir)
        try:
            _run_git(["pull", "--ff-only", "origin", branch], repo_dir)
        except Exception:
            pass
        return

    local_branch_exists = _git_ok(["rev-parse", "--verify", branch], repo_dir)
    if local_branch_exists:
        _run_git(["checkout", branch], repo_dir)
        return

    _run_git(["checkout", "--orphan", branch], repo_dir)
    _wipe_worktree(repo_dir)
    subprocess.run(["git", "rm", "-rf", "--cached", "."], cwd=str(repo_dir), capture_output=True, text=True, check=False)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def _write_payload_files(
    repo_dir: Path,
    snapshot: dict,
    state: Any,
    trades: Any,
    control: Any,
    market_map: Any,
    trade_reviews: Any,
    decision_review_report: Any,
    challenger_report: Any,
    missed_move_report: Any,
    asset_dossiers: Any,
    llm_referee_report: Any,
    playbook_distiller_report: Any,
) -> None:
    dashboard_dir = repo_dir / "dashboard"
    _write_json(dashboard_dir / "dashboard_snapshot.json", snapshot)
    _write_json(dashboard_dir / "current-state.json", state or {})
    _write_json(dashboard_dir / "trades.json", trades or [])
    _write_json(dashboard_dir / "control.json", control or {})
    _write_json(dashboard_dir / "daily_market_map.json", market_map or {})
    _write_json(dashboard_dir / "trade_reviews.json", trade_reviews or {})
    _write_json(dashboard_dir / "decision_review_report.json", decision_review_report or {})
    _write_json(dashboard_dir / "challenger_model_report.json", challenger_report or {})
    _write_json(dashboard_dir / "missed_move_report.json", missed_move_report or {})
    _write_json(dashboard_dir / "asset_dossiers.json", asset_dossiers or {})
    _write_json(dashboard_dir / "llm_referee_report.json", llm_referee_report or {})
    _write_json(dashboard_dir / "playbook_distiller_report.json", playbook_distiller_report or {})
    (repo_dir / "README.md").write_text(
        "# Dashboard State Mirror\n\n"
        "This branch is auto-generated by the trading agent.\n"
        "It mirrors the canonical local dashboard snapshot for the public hosted UI.\n"
    )


def publish_snapshot(
    snapshot: dict,
    *,
    state: Any = None,
    trades: Any = None,
    control: Any = None,
    market_map: Any = None,
    trade_reviews: Any = None,
    decision_review_report: Any = None,
    challenger_report: Any = None,
    missed_move_report: Any = None,
    asset_dossiers: Any = None,
    llm_referee_report: Any = None,
    playbook_distiller_report: Any = None,
) -> bool:
    if not is_enabled():
        return False

    repo_dir = DASHBOARD_STATE_SYNC_REPO
    remote_url = state_remote_url()
    branch = state_branch_ref()
    public_tag = state_public_tag()

    try:
        _ensure_checkout(repo_dir, remote_url, branch)
        _write_payload_files(
            repo_dir,
            snapshot,
            state,
            trades,
            control,
            market_map,
            trade_reviews,
            decision_review_report,
            challenger_report,
            missed_move_report,
            asset_dossiers,
            llm_referee_report,
            playbook_distiller_report,
        )
        _run_git(["add", "dashboard", "README.md"], repo_dir)

        if _has_staged_dashboard_changes(repo_dir):
            cycle = ((snapshot or {}).get("state") or {}).get("cycle_number", 0)
            stamp = (snapshot or {}).get("server_time") or "unknown-time"
            _run_git(["commit", "-m", f"sync dashboard snapshot cycle {cycle} @ {stamp}"], repo_dir)
        # This branch is a generated dashboard mirror, not a collaborative branch.
        # Force-pushing keeps the public fallback source aligned to the latest
        # canonical local snapshot even if another stale mirror push landed first.
        _run_git(["push", "--force", "origin", f"HEAD:{branch}"], repo_dir)
        head = _run_git(["rev-parse", "HEAD"], repo_dir)
        _run_git(["tag", "-f", public_tag, head], repo_dir)
        _run_git(["push", "--force", "origin", f"refs/tags/{public_tag}"], repo_dir)
        log.debug(f"Hosted dashboard fallback sync pushed to {branch} and tag {public_tag}")
        return True
    except Exception as exc:
        log.warning(f"Hosted dashboard fallback sync failed: {exc}")
        return False
