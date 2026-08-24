#!/usr/bin/env bash
# Single source of truth for validating the MLB same-game combo pipeline
# -- CI (.github/workflows/mlb-same-game-predictions.yml) invokes this
# same script for its install/test/pipeline/build steps, so there is
# never a hand-maintained second copy of these commands that can drift
# from what's actually verified locally. Built after this pipeline's
# first real CI iterations exposed two real, fixable gaps (a missing
# pytest install, then an unnecessarily heavy dependency set) one at a
# time via slow, one-at-a-time CI round-trips -- this script exists so
# that never needs to happen again: run `fast` locally (install + test,
# a few seconds on a warm pip cache) before ever pushing, and CI becomes
# a confirmation, not a debugger.
#
# Usage:
#   sports/mlb/scripts/validate_same_game_pipeline.sh <stage> [args...]
#
# Stages:
#   install   Install the pipeline's real, scoped dependency set.
#   test      Run the same-game combo test suite (does NOT install --
#             run `install` first, or use `fast`/`all`).
#   pipeline  Run the real daily orchestrator. Extra args are passed
#             through (e.g. `--calibration-ledger /tmp/x.jsonl`).
#   build     Rebuild the static site from the current publication.
#   fast      install + test only -- the sub-minute local loop meant to
#             replace a real CI round-trip during iteration. Does NOT
#             hit any real network (no live schedule/odds fetch).
#   all       install + test + pipeline (real live fetch) + build -- the
#             full real end-to-end check, still entirely local.
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
  echo "[test] MLB same-game combo suite"
  python -m pytest \
    sports/mlb/tests/test_fetch_mlb_pitcher_game_data.py \
    sports/mlb/tests/test_pitcher_bullpen_model.py \
    sports/mlb/tests/test_pitching_enriched_win_model.py \
    sports/mlb/tests/test_game_simulation_model.py \
    sports/mlb/tests/test_select_mlb_same_game_bets.py \
    sports/mlb/tests/test_run_mlb_same_game_daily.py \
    sports/mlb/tests/test_the_odds_api_mlb_team_market_provider.py \
    sports/mlb/tests/test_backtest_pitching_enriched_win_model.py \
    sports/mlb/tests/test_backtest_game_simulation_model.py \
    -q
}

run_pipeline() {
  echo "[pipeline] run_mlb_same_game_daily.py (real live schedule + odds fetch)"
  python sports/mlb/scripts/run_mlb_same_game_daily.py "$@"
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
