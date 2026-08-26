#!/usr/bin/env bash
# Single source of truth for validating the MLB pitcher-strikeouts-only
# parlay pipeline -- CI (.github/workflows/mlb-pitcher-parlay-
# predictions.yml) invokes this same script, mirroring
# validate_same_game_pipeline.sh's exact reasoning (see that file's
# header) so CI is a confirmation, never a debugger.
#
# Usage:
#   sports/mlb/scripts/validate_pitcher_parlay_pipeline.sh <stage> [args...]
#
# Stages:
#   install   Install the pipeline's real, scoped dependency set (same
#             set same-game already uses -- no new dependency needed).
#   test      Run the pitcher-parlay test suite (does NOT install --
#             run `install` first, or use `fast`/`all`).
#   pipeline  Run the real daily orchestrator. Extra args are passed
#             through (e.g. `--calibration-ledger /tmp/x.jsonl`).
#   build     Rebuild the static site from the current publication.
#   fast      install + test only -- no real network hit.
#   all       install + test + pipeline (real live fetch) + build.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STAGE="${1:-fast}"
shift || true

install_deps() {
  echo "[install] sports/mlb/requirements-same-game.txt"
  python -m pip install -q -r sports/mlb/requirements-same-game.txt
}

run_tests() {
  echo "[test] MLB pitcher-parlay suite"
  python -m pytest \
    sports/mlb/tests/test_pitcher_strikeout_model.py \
    sports/mlb/tests/test_select_mlb_pitcher_parlay.py \
    sports/mlb/tests/test_run_mlb_pitcher_parlay_daily.py \
    -q
}

run_pipeline() {
  echo "[pipeline] run_mlb_pitcher_parlay_daily.py (real live schedule + odds fetch)"
  python sports/mlb/scripts/run_mlb_pitcher_parlay_daily.py "$@"
}

run_build() {
  echo "[build] static site"
  python sports/site/pipeline/build_static_site.py
}

case "$STAGE" in
  install) install_deps ;;
  test) run_tests ;;
  pipeline) run_pipeline "$@" ;;
  build) run_build ;;
  fast)
    install_deps
    run_tests
    echo "[fast] OK -- install + tests passed, no real network hit. Run 'all' for a full E2E check."
    ;;
  all)
    install_deps
    run_tests
    run_pipeline "$@"
    run_build
    echo "[all] OK -- full real end-to-end pipeline + static site build passed."
    ;;
  *)
    echo "usage: $0 [install|test|pipeline|build|fast|all] [pipeline-args...]" >&2
    exit 2
    ;;
esac
