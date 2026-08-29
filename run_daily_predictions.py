#!/usr/bin/env python3
"""
Convenience entrypoint for the shared NBA + MLB daily refresh.

This wrapper forces an immediate run by default so it can be scheduled externally
or launched by hand each day without waiting for the built-in schedule gate.

After a successful MLB refresh, apply the additive tight-quality publication
overlay. The underlying v16 selection artifacts and frozen PARLAY_POLICY_V2
remain untouched; only the public singles payload is tightened.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PIPELINE = REPO_ROOT / "sports" / "site" / "pipeline" / "run_daily_predictions.py"
MLB_TIGHT_QUALITY_OVERLAY = REPO_ROOT / "sports" / "mlb" / "scripts" / "apply_tight_quality_overlay.py"


def main() -> int:
    forwarded_args = list(sys.argv[1:])
    if "--force-run" not in forwarded_args:
        forwarded_args.insert(0, "--force-run")
    command = [sys.executable, str(PIPELINE), *forwarded_args]
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode != 0:
        return int(completed.returncode)

    if "--skip-mlb" not in forwarded_args:
        overlay = subprocess.run([sys.executable, str(MLB_TIGHT_QUALITY_OVERLAY)], cwd=REPO_ROOT)
        if overlay.returncode != 0:
            return int(overlay.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
