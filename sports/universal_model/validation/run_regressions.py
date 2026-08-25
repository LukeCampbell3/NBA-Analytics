"""Existing-sports regression check (spec section 55.Y, build step 26/86):
runs every pre-existing per-sport pytest suite and records pass/fail, to
confirm this entire universal_model build had zero impact on them (it
only ever ADDED a new, separate package; it never edited any file under
sports/{mlb,nba,nfl,f1,golf}/ outside of sports/universal_model/ itself).

Run: python -m sports.universal_model.validation.run_regressions
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TEST_DIRS = [
    "sports/mlb/tests",
    "sports/nba/tests",
    "sports/nba/analytics/tests",
    "sports/nfl/tests",
    "sports/golf/tests",
    "sports/f1/tests",
]


def run_all() -> dict:
    results = {}
    for rel in TEST_DIRS:
        path = REPO_ROOT / rel
        if not path.exists():
            results[rel] = {"status": "SKIPPED", "reason": "directory not found"}
            continue
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(path), "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        elapsed = time.time() - t0
        tail = "\n".join(proc.stdout.strip().splitlines()[-15:])
        results[rel] = {
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "summary_tail": tail,
        }
        print(f"{rel}: {results[rel]['status']} ({elapsed:.1f}s)\n{tail}\n")
    (REPORTS_DIR / "regression_run.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_all()
