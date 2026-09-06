#!/usr/bin/env python3
"""Fail-closed audit for game-conditioned hitter model authority claims."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sports.mlb.advanced.game_conditioned_authority import audit_authority_report


DEFAULT_REPORT = ROOT / "artifacts" / "mlb_game_conditioned_hitter_moe_validation.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute negative/positive authority from validation evidence instead "
            "of trusting model-artifact booleans."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_REPORT,
        help="Validation report or runtime artifact to audit.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    artifact = args.artifact
    if not artifact.is_absolute():
        artifact = (ROOT / artifact).resolve()
    if not artifact.exists():
        print(json.dumps({"valid": False, "error": f"artifact not found: {artifact}"}, indent=2))
        return 2

    report = json.loads(artifact.read_text(encoding="utf-8"))
    audit = audit_authority_report(report)
    audit["source"] = str(artifact)

    rendered = json.dumps(audit, indent=2, sort_keys=True)
    print(rendered)

    if args.output is not None:
        output = args.output
        if not output.is_absolute():
            output = (ROOT / output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")

    return 0 if audit["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
