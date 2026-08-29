#!/usr/bin/env bash
# Single source of truth for validating the MLB pitcher-strikeouts-only
# parlay pipeline. CI and local iteration call this same script.
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
    sports/mlb/tests/test_parlay_quality_selectors.py \
    -q
}

run_pipeline() {
  echo "[pipeline] run_mlb_pitcher_parlay_quality_daily.py (alt-line frontier; probability floors -> EV)"
  python sports/mlb/scripts/run_mlb_pitcher_parlay_quality_daily.py "$@"
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
