#!/usr/bin/env python3
"""Apply only source-code wiring for sequential PA production.

Workflow files are intentionally excluded because GITHUB_TOKEN cannot update
workflow definitions without the separate workflows permission.  The canonical
MLB workflow already installs requirements-same-game.txt; that file now pins
pybaseball, so the model can execute without a workflow-definition mutation.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE / "apply_sequential_pa_source_patch_v2.py"
spec = importlib.util.spec_from_file_location("sequential_pa_patch_v2", V2)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {V2}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main() -> int:
    module.patch_selector()
    module.patch_orchestrator()
    module.patch_exporter()
    module.patch_unified_adapter()
    module.patch_pinned_runner()
    print("sequential PA source wiring v3 complete (workflow files intentionally unchanged)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
