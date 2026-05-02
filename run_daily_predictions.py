#!/usr/bin/env python3
"""
Convenience entrypoint for the shared NBA + MLB daily refresh.

This wrapper forces an immediate run by default so it can be scheduled externally
or launched by hand each day without waiting for the built-in schedule gate.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
PIPELINE = REPO_ROOT / "sports" / "site" / "pipeline" / "run_daily_predictions.py"


def main() -> int:
    forwarded_args = list(sys.argv[1:])
    if "--force-run" not in forwarded_args:
        forwarded_args.insert(0, "--force-run")
    command = [sys.executable, str(PIPELINE), *forwarded_args]
    completed = subprocess.run(command, cwd=REPO_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
