#!/usr/bin/env python3
"""Run the frozen Unified MLB V2.1 challenger against the daily pool.

This is production *plumbing*, not a policy promotion.  The scheduled MLB
workflow remains legacy-authoritative.  V2.1 is fetched at an immutable commit,
run from an isolated detached worktree against the just-generated daily web
payloads, and its result is embedded back into ``daily_predictions.json`` as a
shadow-only research surface.

A V2.1 failure is fail-open for the legacy board: the daily payload receives an
explicit ``UNAVAILABLE`` shadow status and the legacy publication continues.
The invariant ``production_authorized == False`` is revalidated before any
V2.1 result reaches the frontend.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DAILY_JSON = REPO_ROOT / "sports/mlb/web/data/daily_predictions.json"
EVIDENCE_LEDGER = REPO_ROOT / "sports/mlb/data/predictions/unified/v2_1_evidence.jsonl"
STATIC_BUILDER = REPO_ROOT / "sports/site/pipeline/build_static_site.py"
V21_BRANCH = "unified-mlb-v2-resolution"
V21_COMMIT = "55b7d07c1e4b58b362cf5f7afccdbf81ee76d9f0"
V21_POLICY_HASH = "80f335d2501d54502909d7f8587ebfef56d725a67d44ae125f9df4337d489b1c"
V2_BASELINE_POLICY_HASH = "52deb038a076b39a1bc840b77ae26648d9e4ffa20194135e7d48b9761edbc611"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def _run(command: list[str], *, cwd: Path = REPO_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=check,
        text=True,
        stdout=None,
        stderr=None,
    )


def _latest_generation(run_date: str) -> dict[str, Any] | None:
    if not EVIDENCE_LEDGER.exists():
        return None
    latest: dict[str, Any] | None = None
    for raw_line in EVIDENCE_LEDGER.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if row.get("schema_version") != "mlb_v2_1_evidence_v1":
            continue
        if row.get("run_date") != run_date:
            continue
        if row.get("challenger_policy_hash") != V21_POLICY_HASH:
            continue
        if int(row.get("revision", 0) or 0) != 1:
            continue
        latest = row
    return latest


def _public_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "candidate_id",
        "player_id",
        "event_id",
        "capability",
        "market_type",
        "side",
        "line",
        "sportsbook",
        "quoted_odds",
        "usable_probability",
        "market_implied_probability",
        "uncertainty",
        "support_score",
        "edge",
        "conservative_ev",
        "admissible",
        "rejection_reasons",
        "ranking_position",
        "final_selection_decision",
    )
    return {key: candidate.get(key) for key in allowed}


def _build_success_payload(worktree: Path, run_date: str) -> dict[str, Any]:
    status = _load_json(worktree / "artifacts/mlb_v2_challenger_status.json")
    daily = _load_json(worktree / "artifacts/mlb_v2_daily_evidence.json")
    manifest = _load_json(worktree / "artifacts/mlb_v2_1_policy_manifest.json")

    if status.get("policy") != "UNIFIED_MLB_V2_1":
        raise ValueError("V2.1 status policy mismatch")
    if status.get("challenger_policy_hash") != V21_POLICY_HASH:
        raise ValueError("V2.1 challenger hash mismatch")
    if manifest.get("policy_hash") != V21_POLICY_HASH:
        raise ValueError("V2.1 manifest hash mismatch")
    if status.get("baseline_policy_hash") != V2_BASELINE_POLICY_HASH:
        raise ValueError("V2 baseline hash mismatch")
    if status.get("production_authorized") is not False:
        raise ValueError("V2.1 unexpectedly claims production authority")
    if manifest.get("configuration", {}).get("production_authorized") is not False:
        raise ValueError("V2.1 manifest unexpectedly claims production authority")
    if daily.get("run_date") != run_date:
        raise ValueError("V2.1 daily evidence date mismatch")

    generation = _latest_generation(run_date) or {}
    challenger_candidates = generation.get("challenger_candidates")
    if not isinstance(challenger_candidates, list):
        challenger_candidates = []
    public_candidates = [
        _public_candidate(candidate)
        for candidate in challenger_candidates
        if isinstance(candidate, dict)
    ]
    selected = [candidate for candidate in public_candidates if candidate.get("final_selection_decision") is True]
    selected.sort(key=lambda row: (row.get("ranking_position") is None, row.get("ranking_position") or 10**9))

    return {
        "schema_version": "mlb_v2_1_shadow_web_v1",
        "policy": "UNIFIED_MLB_V2_1",
        "policy_hash": V21_POLICY_HASH,
        "baseline_policy_hash": V2_BASELINE_POLICY_HASH,
        "implementation_commit": V21_COMMIT,
        "run_date": run_date,
        "generated_at_utc": daily.get("generated_at_utc") or _utc_now(),
        "state": status.get("state") or "PROSPECTIVE_SHADOW",
        "decision": status.get("decision") or "NO_RELIABLE_EDGE_FOUND",
        "production_authorized": False,
        "certification_started": bool(status.get("certification_started", False)),
        "normalized_candidates": int(daily.get("normalized_candidates", 0) or 0),
        "fully_valid_candidates": int(daily.get("fully_valid_candidates", 0) or 0),
        "admissible_candidates": int(daily.get("admissible_candidates", 0) or 0),
        "selected_bets": int(daily.get("selected_bets", 0) or 0),
        "rejection_count_by_reason": daily.get("rejection_count_by_reason") or {},
        "source_status": daily.get("source_status") or {},
        "selected_candidates": selected,
    }


def _unavailable_payload(run_date: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": "mlb_v2_1_shadow_web_v1",
        "policy": "UNIFIED_MLB_V2_1",
        "policy_hash": V21_POLICY_HASH,
        "baseline_policy_hash": V2_BASELINE_POLICY_HASH,
        "implementation_commit": V21_COMMIT,
        "run_date": run_date,
        "generated_at_utc": _utc_now(),
        "state": "UNAVAILABLE",
        "decision": "ABSTAIN",
        "production_authorized": False,
        "certification_started": False,
        "normalized_candidates": 0,
        "fully_valid_candidates": 0,
        "admissible_candidates": 0,
        "selected_bets": 0,
        "rejection_count_by_reason": {"SHADOW_RUNTIME_UNAVAILABLE": 1},
        "source_status": {},
        "selected_candidates": [],
        "availability_note": reason[:240],
    }


def _embed(payload: dict[str, Any], run_date: str) -> None:
    daily = _load_json(DAILY_JSON)
    if daily.get("run_date") != run_date:
        raise ValueError(
            f"refusing to attach V2.1 {run_date} result to daily pool {daily.get('run_date')}"
        )
    if payload.get("production_authorized") is not False:
        raise ValueError("refusing to embed V2.1 payload with production authority")
    daily["v2_1_shadow"] = payload
    _atomic_write_json(DAILY_JSON, daily)


def run(run_date: str) -> dict[str, Any]:
    if not DAILY_JSON.exists():
        raise FileNotFoundError(DAILY_JSON)

    runner_temp = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
    worktree = runner_temp / f"mlb-v2-1-{os.getpid()}"
    if worktree.exists():
        shutil.rmtree(worktree, ignore_errors=True)

    try:
        _run(["git", "fetch", "--no-tags", "origin", V21_BRANCH])
        _run(["git", "cat-file", "-e", f"{V21_COMMIT}^{{commit}}"])
        _run(["git", "worktree", "add", "--detach", str(worktree), V21_COMMIT])
        EVIDENCE_LEDGER.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                sys.executable,
                str(worktree / "sports/mlb/scripts/run_v2_1_challenger.py"),
                "--data-dir",
                str(REPO_ROOT / "sports/mlb/web/data"),
                "--run-date",
                run_date,
                "--evidence-ledger",
                str(EVIDENCE_LEDGER),
            ],
            cwd=worktree,
        )
        return _build_success_payload(worktree, run_date)
    finally:
        if worktree.exists():
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=REPO_ROOT,
                check=False,
                text=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-date", required=True)
    args = parser.parse_args()

    try:
        payload = run(args.run_date)
        print(
            f"V2.1 shadow: state={payload['state']} candidates={payload['normalized_candidates']} "
            f"admissible={payload['admissible_candidates']} selected={payload['selected_bets']}"
        )
    except Exception as exc:  # Shadow failure must never suppress the legacy daily board.
        print(f"::warning::V2.1 shadow unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
        payload = _unavailable_payload(args.run_date, f"{type(exc).__name__}: {exc}")

    _embed(payload, args.run_date)
    _run([sys.executable, str(STATIC_BUILDER)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
