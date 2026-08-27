#!/usr/bin/env python3
"""
Run the shared daily prediction refresh for the published multi-sport site.

This orchestrates:
1. NBA live pipeline refresh or NBA payload export from an existing manifest
2. MLB high-precision selection from the latest raw pool
3. MLB web/dist payload export
4. Unified static-site rebuild into the repo-root dist bundle
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


SCRIPT_PATH = Path(__file__).resolve()
SITE_ROOT = SCRIPT_PATH.parents[1]
SPORTS_ROOT = SITE_ROOT.parent
REPO_ROOT = SPORTS_ROOT.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist"
DEFAULT_PRIVATE_OUTPUT_DIR = REPO_ROOT / "paywall" / "private-content" / "app"

NBA_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
NBA_RUNNER = NBA_PREDICTOR_ROOT / "scripts" / "run_daily_market_pipeline.py"
NBA_EXPORTER = NBA_PREDICTOR_ROOT / "scripts" / "export_daily_predictions_web.py"
# Real, deduplicated local headshot cache -- see
# update_nba_player_headshot_cache.py's module docstring. Additive only:
# downloads any real bettable player not already cached and rewrites the
# board's player_headshot_url to point at the local copy (real remote URL
# kept as fallback), never touches any other key.
NBA_HEADSHOT_CACHE = NBA_PREDICTOR_ROOT / "scripts" / "update_nba_player_headshot_cache.py"
NBA_WEB_JSON = REPO_ROOT / "sports" / "nba" / "web" / "data" / "daily_predictions.json"
NBA_CARDS_JSON = REPO_ROOT / "sports" / "nba" / "web" / "data" / "cards.json"

MLB_DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
MLB_DATA_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
MLB_MANIFEST = MLB_DATA_DIR / "update_manifest_2026.json"
MLB_MARKET_FETCHER = REPO_ROOT / "Player-Predictor" / "scripts" / "fetch_mlb_market_props.py"
MLB_DATA_UPDATER = REPO_ROOT / "Player-Predictor" / "scripts" / "update_mlb_processed_data.py"
MLB_GENERATOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "generate_daily_prediction_pool.py"
MLB_GOVERNANCE_CAPTURE = REPO_ROOT / "sports" / "mlb" / "governance" / "capture_complete_slate.py"
MLB_PROVIDER_OBSERVATIONS = REPO_ROOT / "sports" / "mlb" / "data" / "raw" / "market_odds" / "mlb" / "odds_api_io" / "latest_provider_observations.csv"
MLB_SELECTOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "select_high_precision_predictions.py"
MLB_PARLAY_SELECTOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "select_daily_parlay.py"
# PARLAY_POLICY_V2 -- a SEPARATE product path (mission: MLB dual-path
# integration). Runs additively alongside, never in place of,
# MLB_PARLAY_SELECTOR above; its output is a distinct JSON consumed only
# by the new "parlays" payload key, never by the legacy_parlay_control
# fields the exporter already writes from MLB_PARLAY_SELECTOR's output.
#
# Currently frozen prospective policy: PARLAY_POLICY_V2_PROSPECTIVE_003
# (world_gate_mode=OBSERVE_ONLY -- see manifest.WORLD_GATE_MODE and
# world_gate_research.py for the DEVELOPMENT research this freeze is
# based on; PARLAY_POLICY_V2_PROSPECTIVE_002, world_gate_mode=REQUIRED,
# remains frozen/immutable but is no longer the active policy). This
# script deliberately does NOT pass --world-gate-mode/--world-risk-
# threshold below -- run_parlay_v2.py's CLI defaults BOTH to
# manifest.WORLD_GATE_MODE/WORLD_RISK_THRESHOLD (the CURRENTLY frozen
# policy's config) precisely so a real production/CI run always follows
# whichever policy is currently frozen without this orchestrator needing
# to import MLB research modules itself (it stays a pure subprocess
# orchestrator by design). To replay/audit PROSPECTIVE_002's exact
# behavior instead, invoke MLB_PARLAY_V2_RUNNER_MODULE manually with
# --world-gate-mode REQUIRED.
#
# Invoked as `python -m <dotted path>`, NEVER as a bare script path
# (`python /abs/path/to.py`) -- this module (and the three below it) use
# absolute `sports.*` imports AND relative `from .x import y` imports.
# Running it as a bare script leaves `sys.path`/`__package__` unset for
# either kind of import to resolve, which is a real bug this pipeline hit
# in production: every real CI run silently failed at the first import
# line (ModuleNotFoundError: No module named 'sports'), the exception was
# swallowed by this orchestrator's own deliberate best-effort try/except,
# and the Parlays tab reported PARLAY_V2_ARTIFACT_UNAVAILABLE on every
# single run -- meaning NONE of PARLAY_V2's pipeline (calibration
# ingestion, pair ingestion, evidence settlement, or the policy runner
# itself) had ever actually executed. `-m` is the fix: it puts CWD (here,
# REPO_ROOT, via run_step's own cwd=REPO_ROOT default) on sys.path AND
# correctly sets __package__ so both import styles resolve.
MLB_PARLAY_V2_RUNNER_MODULE = "sports.mlb.parlay_v2.run_parlay_v2"
# The forward-only calibration ledger (STREAM A) -- one persistent file,
# appended to only after settlement is final for a given slate (see
# sports/mlb/parlay_v2/calibration/store.py). This script only READS it
# (never writes); MLB_PARLAY_V2_INGEST_MODULE below is the only writer.
MLB_PARLAY_V2_CALIBRATION_LEDGER = REPO_ROOT / "sports" / "mlb" / "parlay_v2" / "calibration" / "reports" / "calibration_ledger.jsonl"
# Real settlement -> calibration ledger admission (STREAM A's only
# writer). Runs on already-settled PAST days, never on today's own slate
# -- see ingest.py's module docstring for why that ordering is what makes
# this forward-only. Invoked via `-m`, not a bare script path -- see
# MLB_PARLAY_V2_RUNNER_MODULE's comment above for why.
MLB_PARLAY_V2_INGEST_MODULE = "sports.mlb.parlay_v2.calibration.ingest"
# How many past days to (re-)attempt ingestion for on each run. Ingestion
# is idempotent and gracefully admits zero rows for a day whose outcomes
# aren't in Player-Predictor/Data-Proc-MLB yet, so re-attempting recent
# days each run is a cheap, safe way to catch up a day whose data lagged
# behind a prior run without needing separate failure tracking.
MLB_PARLAY_V2_INGEST_LOOKBACK_DAYS = 4
# Pair-level calibration ledger (SEPARATE research stream -- see
# calibration/pair_schema.py's module docstring). Every settled,
# support-passing candidate pair, NOT just the one pair the policy
# selected -- this never feeds policy action, only future joint-
# calibration research (joint_support stays OBSERVE_ONLY regardless).
MLB_PARLAY_V2_PAIR_LEDGER = REPO_ROOT / "sports" / "mlb" / "parlay_v2" / "calibration" / "reports" / "pair_observation_ledger.jsonl"
# Invoked via `-m`, not a bare script path -- see
# MLB_PARLAY_V2_RUNNER_MODULE's comment above for why.
MLB_PARLAY_V2_PAIR_INGEST_MODULE = "sports.mlb.parlay_v2.calibration.pair_ingest"
# Durable per-day DecisionRecord ledger (decision_record_store.py) --
# written by MLB_PARLAY_V2_RUNNER_MODULE itself at decision time, every run.
MLB_PARLAY_V2_DECISION_RECORD_LEDGER = REPO_ROOT / "sports" / "mlb" / "research" / "parlay_certification_v2" / "reports" / "decision_record_ledger.jsonl"
# Settlement -> policy evidence ingestion (settle_evidence.py) -- the only
# writer of FinalEvidenceRecord rows, one file per policy_version under
# this root (evidence_store.py). MLB_PARLAY_V2_POLICY_VERSION MUST match
# sports.mlb.research.parlay_certification_v2.manifest.POLICY_VERSION
# exactly (the STRUCTURAL policy shape identifier stamped onto every
# DecisionRecord -- see settle_evidence.py's module docstring for why
# this is a different identifier than the prospective policy id). Invoked
# via `-m`, not a bare script path -- see MLB_PARLAY_V2_RUNNER_MODULE's
# comment above for why.
MLB_PARLAY_V2_SETTLE_EVIDENCE_MODULE = "sports.mlb.research.parlay_certification_v2.settle_evidence"
MLB_PARLAY_V2_EVIDENCE_STORE_ROOT = REPO_ROOT / "sports" / "mlb" / "research" / "parlay_certification_v2" / "reports" / "evidence"
MLB_PARLAY_V2_POLICY_VERSION = "PARLAY_POLICY_V2_TWO_LEG_SINGLE_ACTION"
MLB_CONFIDENCE_CALIBRATOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "live_board_confidence.py"
MLB_PICK_SURVIVAL_MODEL = REPO_ROOT / "sports" / "mlb" / "scripts" / "pick_survival_model.py"
MLB_LATENT_POOL_REPLAY = REPO_ROOT / "sports" / "mlb" / "scripts" / "backtest_latent_daily_pools.py"
MLB_LATENT_POOL_REPLAY_REPORT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "backtests" / "latent_daily_pool_replay_2026.json"
MLB_MAX_WINRATE_SELECTOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "select_max_winrate_board.py"
MLB_EXPORTER = REPO_ROOT / "sports" / "mlb" / "scripts" / "export_web_prediction_payload.py"
MLB_WEB_JSON = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"
# Newest MLB predictor (real joint Monte Carlo game simulation +
# starting-pitcher/bullpen enrichment, sports/mlb/predictions/
# game_simulation_model.py + pitching_enriched_win_model.py) wired into
# the main single-leg board. Additive only -- merges an
# "mlb_team_market_plays" key into MLB_WEB_JSON, never touches "plays" or
# any other key the exporter above just wrote. Reuses the exact same
# real-data preparation and calibration/support.py REQUIRED gate as the
# separate same-game combo pipeline (mlb-same-game-predictions.yml), so a
# leg is authorized here iff it would be authorized there too -- see
# generate_mlb_team_market_predictions.py's module docstring.
MLB_TEAM_MARKET_PREDICTOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "generate_mlb_team_market_predictions.py"
# Real headshot enrichment for PARLAY_POLICY_V2 legs -- see
# enrich_parlay_leg_headshots.py's module docstring. Additive only:
# attaches player_headshot_url/player_headshot_fallback_url to the
# already-exported parlays.selected_parlay / parlays.shadow_candidate leg
# dicts, never touches any other key.
MLB_PARLAY_HEADSHOT_ENRICHER = REPO_ROOT / "sports" / "mlb" / "scripts" / "enrich_parlay_leg_headshots.py"
# Real FanDuel "Add to Betslip" deep links for PARLAY_POLICY_V2 pairs --
# see enrich_parlay_leg_betslip.py's module docstring. Additive only:
# attaches betslip/betslip_url to the already-exported
# parlays.selected_parlay / parlays.shadow_candidate pairs (only when
# every leg resolves to a real live FanDuel selection), never touches
# any other key.
MLB_PARLAY_BETSLIP_ENRICHER = REPO_ROOT / "sports" / "mlb" / "scripts" / "enrich_parlay_leg_betslip.py"
# Real, deduplicated local headshot cache -- see
# update_mlb_player_headshot_cache.py's module docstring. Additive only:
# downloads any real bettable player not already cached and rewrites
# every board's player_headshot_url to point at the local copy (real
# remote URL kept as fallback), never touches any other key. Runs last
# among the headshot-related steps so it sees every leg's real URL,
# including the parlay-leg enrichment step just above.
MLB_HEADSHOT_CACHE = REPO_ROOT / "sports" / "mlb" / "scripts" / "update_mlb_player_headshot_cache.py"
MLB_PRIMARY_POLICY_PROFILE = "premium_evidence_gated_v10"
MLB_PICK_SURVIVAL_TOP_K = 3
# v9 -> v10: real, backtested return optimization, not a precision change.
# v9's min-hit-probability/min-graded-hit-rate/min-abs-edge/max-push-
# probability are UNCHANGED here (0.55/0.55/0.10/0.15) -- min-abs-edge and
# max-push-probability were directly confirmed non-binding at v9's
# hit-probability floor (a full sweep of edge 0.05-0.25 and push
# 0.05-0.30 produced byte-identical selected pools). What changed is the
# one gate v9's own comment called "deliberately UNCHANGED": --max-per-
# market-bucket and --max-per-team (2 -> 6), plus --min-hit-probability/
# --min-graded-hit-rate loosened once more (0.55 -> 0.45) and --top-n/
# --daily-pick-soft-cap/--max-over-picks raised from 10 to 25 so none of
# the three re-binds as a silent tighter cap now that more real
# candidates clear the bar per day.
#
# Real evidence (optimize_walk_forward_policy.py's select_config/
# score_rows against historical_pool_universe_2026.csv, 156 real dates,
# market_source=='real' only, real per-player/per-game dedup preserved --
# the same machinery already used to certify v9 over v8):
#   v9  (bucket/team=2, hp=0.55): 89 plays,  72-17  (80.9%), +64.71u,  +72.7% ROI/play
#   v10 (bucket/team=6, hp=0.45): 230 plays, 173-57 (75.2%), +147.38u, +64.1% ROI/play
# v10 more than doubles real net units on the identical 11 real
# market-captured days, at a hit rate still comfortably above breakeven
# and an ROI/play within 9pp of v9's. This is a real, stable plateau, not
# a single lucky cell: cap=4..6 x hp=0.40..0.50 all land within a few
# units of each other (checked directly, not asserted) -- picking any one
# of those exact cells over its neighbors would be overfitting the sample,
# picking the plateau itself is not.
#
# Same honest caveat that applied to v9's own backtest: this rests on
# only 11 of 156 real market-captured days (thin evidence -- most of this
# repo's historical universe is synthetic-priced and correctly excluded).
# Treat this as evidence-based and provisional, not proven at scale;
# revisit as more real days accumulate via the settlement pipeline
# (settle_published_predictions.py).
MLB_PRIMARY_POLICY_ARGS = [
    "--top-n", "25",
    "--require-real-market-source",
    "--min-market-books", "5",
    "--min-common-market-books", "2",
    "--min-history-rows", "35",
    "--min-prediction", "0.10",
    "--min-hit-probability", "0.45",
    "--min-graded-hit-rate", "0.45",
    "--max-push-probability", "0.15",
    "--min-abs-edge", "0.10",
    "--min-expected-value", "0.0",
    "--pitcher-k-min-starter-history", "15",
    "--pitcher-k-min-projected-ip", "5.25",
    "--pitcher-k-min-projected-pitches", "75",
    "--pitcher-k-max-days-since-history", "14",
    "--pitcher-k-min-abs-edge", "0.15",
    "--pitcher-k-max-abs-edge", "1.0",
    "--pitcher-k-min-model-hit-probability", "0.50",
    "--pitcher-k-max-model-hit-probability", "0.65",
    "--pitcher-k-min-expected-value", "0.0",
    "--pitcher-k-min-american-price", "-130",
    "--pitcher-k-max-american-price", "130",
    "--max-pitcher-k-picks", "1",
    "--core-min-american-price", "-180",
    "--core-max-american-price", "125",
    "--min-over-picks", "0",
    "--max-over-picks", "25",
    "--max-under-picks", "0",
    "--daily-pick-soft-cap", "25",
    "--post-cap-min-selection-score", "0.50",
    "--max-per-market-bucket", "6",
    "--max-per-team", "6",
    "--min-historical-bet-profile-support", "0",
    "--min-historical-bet-profile-win-rate", "0",
    "--min-historical-market-availability-support", "0",
    "--min-historical-market-availability-rate", "0",
]

BUILD_STATIC_SITE = REPO_ROOT / "sports" / "site" / "pipeline" / "build_static_site.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared NBA + MLB daily prediction refresh.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child steps.")
    parser.add_argument("--run-date", type=str, default=None, help="Optional YYYY-MM-DD run date.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Published static output directory.")
    parser.add_argument("--private-output-dir", type=Path, default=DEFAULT_PRIVATE_OUTPUT_DIR, help="Protected release-source output directory.")
    parser.add_argument("--scheduled-hour", type=int, default=2, help="Local hour when the shared refresh is allowed to run.")
    parser.add_argument("--scheduled-minute", type=int, default=0, help="Local minute when the shared refresh is allowed to run.")
    parser.add_argument("--force-run", action="store_true", help="Bypass the local schedule gate and run immediately.")
    parser.add_argument("--skip-nba", action="store_true", help="Skip NBA prediction refresh/export.")
    parser.add_argument("--skip-mlb", action="store_true", help="Skip MLB prediction refresh/export.")
    parser.add_argument("--skip-build-site", action="store_true", help="Skip rebuilding the shared static site.")

    parser.add_argument("--nba-manifest", type=Path, default=None, help="Reuse an existing NBA manifest instead of running the live NBA pipeline.")
    parser.add_argument("--nba-season", type=int, default=None, help="Optional NBA season override.")
    parser.add_argument("--nba-latest", action="store_true", help="Use latest NBA manifest policy behavior.")
    parser.add_argument(
        "--nba-policy-profile",
        type=str,
        default="production_board_objective_b12",
        help="NBA live policy profile forwarded to the NBA daily runner.",
    )
    parser.add_argument(
        "--nba-shadow-policy-profiles",
        nargs="*",
        default=None,
        help="Optional NBA shadow policy profiles forwarded to the NBA daily runner.",
    )
    parser.add_argument(
        "--nba-market-provider",
        type=str,
        default="rotowire",
        choices=["rotowire", "covers", "sportsgameodds"],
        help="NBA live market provider forwarded to the NBA daily runner.",
    )
    parser.add_argument(
        "--nba-market-bookmakers",
        type=str,
        default="draftkings,fanduel",
        help="Comma-separated NBA bookmakers forwarded to the NBA daily runner.",
    )
    parser.add_argument(
        "--nba-snapshot-policy",
        type=str,
        default="auto",
        choices=["auto", "live_only"],
        help="NBA market snapshot freshness policy forwarded to the NBA daily runner.",
    )
    parser.add_argument("--nba-allow-heuristic-fallback", action="store_true", help="Allow NBA heuristic fallback if model loading fails.")
    parser.add_argument("--nba-skip-update-data", action="store_true", help="Skip the NBA official-data refresh step.")
    parser.add_argument("--nba-skip-collect-market", action="store_true", help="Skip the NBA market collection step.")
    parser.add_argument("--nba-skip-align", action="store_true", help="Skip the NBA market alignment step.")
    parser.add_argument("--nba-skip-backtest", action="store_true", help="Skip the NBA backtest refresh step.")
    parser.add_argument("--nba-skip-cutoff-meta-monitor", action="store_true", help="Skip the NBA cutoff-meta monitor step.")

    parser.add_argument("--mlb-pool-csv", type=Path, default=None, help="Explicit raw MLB daily prediction pool CSV.")
    parser.add_argument("--mlb-skip-fetch-market", action="store_true", help="Skip fetching same-day MLB market props.")
    parser.add_argument("--mlb-skip-update-data", action="store_true", help="Skip rebuilding MLB processed player files from source data.")
    parser.add_argument(
        "--mlb-incremental-update",
        action="store_true",
        help="Update MLB processed data from only newly completed games.",
    )
    parser.add_argument(
        "--mlb-refresh-history-caches",
        action="store_true",
        help="Refresh slower MLB selector history caches during this run.",
    )
    parser.add_argument("--mlb-skip-generate", action="store_true", help="Skip generating a fresh MLB raw prediction pool from processed MLB data.")
    parser.add_argument("--mlb-data-dir", type=Path, default=MLB_DATA_DIR, help="MLB processed-data root used by the raw pool generator.")
    parser.add_argument("--mlb-manifest", type=Path, default=MLB_MANIFEST, help="Optional MLB processed-data manifest used by the raw pool generator.")
    parser.add_argument(
        "--mlb-market-provider",
        type=str,
        default="provider_chain",
        choices=[
            "provider_chain", "scrape", "the_odds_api", "sportsgameodds",
            "existing_provider", "rotowire", "odds_api", "snapshot",
        ],
        help="Provider used by the MLB market fetcher. Use provider_chain for configured priority and fallback.",
    )
    parser.add_argument("--mlb-market-input-path", type=Path, default=None, help="Optional snapshot input for the MLB market fetcher.")
    parser.add_argument(
        "--mlb-fallback-policy",
        type=str,
        default="exact_or_latest",
        choices=["exact_only", "exact_or_latest", "latest_available"],
        help="Fallback policy forwarded to the MLB raw pool generator.",
    )
    parser.add_argument(
        "--mlb-min-publish-plays",
        type=int,
        default=1,
        help="Minimum selected MLB plays required before publishing a generated premium pool.",
    )
    parser.add_argument(
        "--mlb-min-rescue-plays",
        type=int,
        default=1,
        help="Minimum same-day MLB plays required for publication; zero-play boards remain withheld.",
    )
    parser.add_argument("--mlb-top-n", type=int, default=10, help="Maximum number of MLB plays to keep.")
    return parser.parse_args()


def run_step(label: str, command: list[str], cwd: Path = REPO_ROOT) -> None:
    print("\n" + "=" * 88)
    print(label)
    print("=" * 88)
    print("Command:", " ".join(command))
    started = time.perf_counter()
    try:
        subprocess.run(command, cwd=cwd, check=True)
    finally:
        print(f"[timing] {label}: {time.perf_counter() - started:.2f}s")


def format_step_failure(exc: Exception) -> str:
    if isinstance(exc, subprocess.CalledProcessError):
        command = " ".join(str(part) for part in exc.cmd) if exc.cmd else "<unknown command>"
        return f"{type(exc).__name__}: exit code {exc.returncode} from {command}"
    return f"{type(exc).__name__}: {exc}"


def validate_schedule_args(hour: int, minute: int) -> tuple[int, int]:
    if not 0 <= int(hour) <= 23:
        raise SystemExit(f"--scheduled-hour must be between 0 and 23, received {hour!r}")
    if not 0 <= int(minute) <= 59:
        raise SystemExit(f"--scheduled-minute must be between 0 and 59, received {minute!r}")
    return int(hour), int(minute)


EASTERN_TZ = ZoneInfo("America/New_York")


def resolve_effective_run_date(run_date: str | None, now: datetime | None = None) -> date:
    if run_date:
        return datetime.fromisoformat(str(run_date)).date()
    resolved_now = now if now is not None else datetime.now(EASTERN_TZ)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=EASTERN_TZ)
    return resolved_now.astimezone(EASTERN_TZ).date()


def load_payload_run_date(payload_path: Path) -> str | None:
    if not payload_path.exists():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = payload.get("run_date")
    token = str(value).strip() if value is not None else ""
    return token or None


def archive_previous_prediction_payload(payload_path: Path, target_run_date: str) -> Path | None:
    """Preserve the prior public board before a new run replaces it."""
    if not payload_path.exists():
        return None
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    archived_date = str(payload.get("run_date") or "").strip()
    try:
        date.fromisoformat(archived_date)
        date.fromisoformat(str(target_run_date))
    except ValueError:
        return None
    if archived_date == str(target_run_date):
        return None

    history_dir = payload_path.parent / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    archive_path = history_dir / f"{archived_date}.json"
    if archive_path.exists():
        if archive_path.read_bytes() != payload_path.read_bytes():
            print(f"[warning] Preserving immutable history snapshot despite changed source: {archive_path}")
    else:
        shutil.copyfile(payload_path, archive_path)

    archived_dates = sorted(
        (
            path.stem
            for path in history_dir.glob("????-??-??.json")
            if _is_iso_date(path.stem)
        ),
        reverse=True,
    )
    index_payload = {
        "dates": archived_dates,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (history_dir / "index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return archive_path


def _is_iso_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def stale_payload_sports(args: argparse.Namespace, target_run_date: str) -> list[str]:
    stale: list[str] = []
    if not args.skip_nba and load_payload_run_date(NBA_WEB_JSON) != target_run_date:
        stale.append("NBA")
    if not args.skip_mlb and load_payload_run_date(MLB_WEB_JSON) != target_run_date:
        stale.append("MLB")
    return stale


def derive_nba_manifest_path(run_date: date) -> Path:
    run_stamp = run_date.strftime("%Y%m%d")
    return NBA_PREDICTOR_ROOT / "model" / "analysis" / "daily_runs" / run_stamp / f"daily_market_pipeline_manifest_{run_stamp}.json"


def check_schedule_gate(args: argparse.Namespace) -> tuple[bool, str]:
    scheduled_hour, scheduled_minute = validate_schedule_args(args.scheduled_hour, args.scheduled_minute)
    now_local = datetime.now().astimezone()
    target_run_date = str(resolve_effective_run_date(args.run_date))
    stale_sports = stale_payload_sports(args, target_run_date)
    timezone_label = str(now_local.tzname() or "local")
    scheduled_label = f"{scheduled_hour:02d}:{scheduled_minute:02d} {timezone_label}"
    scheduled_time_reached = (now_local.hour, now_local.minute) >= (scheduled_hour, scheduled_minute)

    if args.force_run:
        return True, (
            f"Bypassing schedule gate at {now_local.isoformat()} because --force-run was provided. "
            f"Configured run time remains {scheduled_label}. Target run date: {target_run_date}."
        )

    if scheduled_time_reached and stale_sports:
        return True, (
            f"Schedule gate passed at {now_local.isoformat()} because the configured run time "
            f"({scheduled_label}) has passed and these payloads are still stale or missing for {target_run_date}: "
            f"{', '.join(stale_sports)}."
        )

    if scheduled_time_reached:
        return False, (
            f"Skipping shared daily prediction refresh at {now_local.isoformat()} because the published payloads "
            f"are already current for {target_run_date}. Pass --force-run to rebuild anyway."
        )

    return False, (
        f"Skipping shared daily prediction refresh at {now_local.isoformat()} because the configured run time is "
        f"{scheduled_label}. The current stale/missing payloads for {target_run_date} are: "
        f"{', '.join(stale_sports) if stale_sports else 'none'}. Re-run after the scheduled time or pass "
        f"--force-run for a manual execution."
    )


def run_stamp_from_date(run_date: str | None) -> str | None:
    if not run_date:
        return None
    token = str(run_date).strip().replace("-", "")
    return token if len(token) == 8 and token.isdigit() else None


def candidate_mlb_pool_csvs(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(run_dir.glob("daily_prediction_pool_*.csv")):
        name = path.name
        if "_high_precision_predictions" in name or "_best_predictions" in name:
            continue
        candidates.append(path)
    return candidates


def find_latest_mlb_pool_csv(
    daily_runs_root: Path,
    preferred_run_stamp: str | None,
    exclude_paths: set[Path] | None = None,
) -> Path:
    excluded = {path.resolve() for path in (exclude_paths or set())}
    run_dirs = [path for path in sorted(daily_runs_root.iterdir(), reverse=True) if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No MLB run directories found under {daily_runs_root}")

    if preferred_run_stamp:
        preferred_dir = daily_runs_root / preferred_run_stamp
        if preferred_dir.is_dir():
            preferred_candidates = candidate_mlb_pool_csvs(preferred_dir)
            for candidate in preferred_candidates:
                if candidate.resolve() not in excluded:
                    return candidate

    for run_dir in run_dirs:
        candidates = candidate_mlb_pool_csvs(run_dir)
        for candidate in candidates:
            if candidate.resolve() not in excluded:
                return candidate

    raise FileNotFoundError(f"No raw MLB daily prediction pool CSV was found under {daily_runs_root}")


def derive_mlb_selector_outputs(pool_csv: Path, suffix: str | None = None) -> tuple[Path, Path]:
    stem = pool_csv.stem if not suffix else f"{pool_csv.stem}_{suffix}"
    return (
        pool_csv.with_name(f"{stem}_high_precision_predictions.csv"),
        pool_csv.with_name(f"{stem}_high_precision_predictions_summary.json"),
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_mlb_market_profile(pool_csv: Path) -> dict[str, int]:
    summary = {
        "rows": 0,
        "real_market_rows": 0,
        "synthetic_rows": 0,
        "price_confirmed_rows": 0,
    }
    with open(pool_csv, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            summary["rows"] += 1
            market_source = str(row.get("Market_Source", "")).strip().lower()
            if market_source == "real":
                summary["real_market_rows"] += 1
                direction = "OVER"
                try:
                    edge = float(row.get("Edge", 0.0) or 0.0)
                    direction = "OVER" if edge >= 0 else "UNDER"
                except (TypeError, ValueError):
                    direction = "OVER"
                price_key = "Market_Over_Price" if direction == "OVER" else "Market_Under_Price"
                price_text = str(row.get(price_key, "")).strip()
                if price_text:
                    summary["price_confirmed_rows"] += 1
            else:
                summary["synthetic_rows"] += 1
    return summary


def selected_row_count(summary_json: Path) -> int:
    if not summary_json.exists():
        return 0
    try:
        payload = load_json(summary_json)
    except Exception:
        return 0
    return int(payload.get("rows_selected", 0) or 0)


def promote_mlb_selector_outputs(source_csv: Path, source_summary_json: Path, target_csv: Path, target_summary_json: Path) -> None:
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    target_summary_json.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_csv, target_csv)
    shutil.copyfile(source_summary_json, target_summary_json)


def annotate_mlb_summary(
    summary_json: Path,
    *,
    publication_strategy: str,
    market_profile: dict[str, int],
    publication_state: str = "published_current_pool",
) -> None:
    if not summary_json.exists():
        return
    try:
        payload = load_json(summary_json)
    except Exception:
        return
    payload["publication_strategy"] = str(publication_strategy)
    payload["publication_state"] = str(publication_state)
    payload["pool_market_profile"] = {key: int(value) for key, value in market_profile.items()}
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def derive_generated_mlb_pool_outputs(run_date: str | None) -> tuple[Path, Path]:
    local_date = datetime.now().astimezone().date() if not run_date else datetime.fromisoformat(str(run_date)).date()
    run_stamp = local_date.strftime("%Y%m%d")
    run_dir = MLB_DAILY_RUNS_ROOT / run_stamp
    return (
        run_dir / f"daily_prediction_pool_{run_stamp}.csv",
        run_dir / f"daily_prediction_pool_{run_stamp}.json",
    )


def run_nba(args: argparse.Namespace, output_dir: Path) -> None:
    nba_dist_json = Path(getattr(args, "private_output_dir", DEFAULT_PRIVATE_OUTPUT_DIR)).resolve() / "nba" / "data" / "daily_predictions.json"
    if args.nba_manifest:
        command = [
            args.python,
            str(NBA_EXPORTER),
            "--manifest",
            str(args.nba_manifest.resolve()),
            "--out-json",
            str(NBA_WEB_JSON),
            "--out-dist",
            str(nba_dist_json),
            "--cards-json",
            str(NBA_CARDS_JSON),
        ]
        run_step("Export NBA Predictions From Existing Manifest", command)
        run_nba_headshot_cache(args, nba_dist_json)
        return

    command = [
        args.python,
        str(NBA_RUNNER),
        "--policy-profile",
        str(args.nba_policy_profile),
        "--web-cards-json",
        str(NBA_CARDS_JSON),
        "--skip-build-site",
        "--market-provider",
        str(args.nba_market_provider),
        "--market-bookmakers",
        str(args.nba_market_bookmakers),
        "--snapshot-policy",
        str(args.nba_snapshot_policy),
    ]
    if args.run_date:
        command.extend(["--run-date", str(args.run_date)])
    if args.nba_season is not None:
        command.extend(["--season", str(int(args.nba_season))])
    if args.nba_latest:
        command.append("--latest")
    if args.nba_shadow_policy_profiles:
        command.extend(["--shadow-policy-profiles", *[str(value) for value in args.nba_shadow_policy_profiles]])
    if args.nba_allow_heuristic_fallback:
        command.append("--allow-heuristic-fallback")
    if args.nba_skip_update_data:
        command.append("--skip-update-data")
    if args.nba_skip_collect_market:
        command.append("--skip-collect-market")
    if args.nba_skip_align:
        command.append("--skip-align")
    if args.nba_skip_backtest:
        command.append("--skip-backtest")
    if args.nba_skip_cutoff_meta_monitor:
        command.append("--skip-cutoff-meta-monitor")

    run_step("Run NBA Daily Prediction Pipeline", command)
    run_date = resolve_effective_run_date(args.run_date)
    expected_manifest = derive_nba_manifest_path(run_date)
    if not expected_manifest.exists():
        raise FileNotFoundError(
            "NBA daily pipeline completed but the expected same-day manifest was not found: "
            f"{expected_manifest}"
        )
    run_step(
        "Export NBA Prediction Payload",
        [
            args.python,
            str(NBA_EXPORTER),
            "--manifest",
            str(expected_manifest),
            "--out-json",
            str(NBA_WEB_JSON),
            "--out-dist",
            str(nba_dist_json),
            "--cards-json",
            str(NBA_CARDS_JSON),
        ],
    )
    run_nba_headshot_cache(args, nba_dist_json)


def run_nba_headshot_cache(args: argparse.Namespace, nba_dist_json: Path) -> None:
    if not NBA_HEADSHOT_CACHE.exists():
        return
    try:
        run_step(
            "Cache NBA Player Headshots",
            [
                args.python,
                str(NBA_HEADSHOT_CACHE),
                "--daily-predictions-path", str(NBA_WEB_JSON),
                "--daily-predictions-path", str(nba_dist_json),
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate: additive, never blocks singles publication
        print(f"[warning] NBA headshot cache step failed, images fall back to their real remote CDN URL: {format_step_failure(exc)}")


def run_mlb(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path, Path]:
    preferred_run_stamp = run_stamp_from_date(args.run_date)
    generated_pool_csv: Path | None = None
    generated_summary_json: Path | None = None
    used_generated_pool = False
    market_fetch_failed = False

    if args.mlb_pool_csv:
        pool_csv = args.mlb_pool_csv.resolve()
    else:
        if not args.mlb_skip_fetch_market:
            fetch_command = [
                args.python,
                str(MLB_MARKET_FETCHER),
                "--provider",
                str(args.mlb_market_provider),
            ]
            if args.run_date:
                fetch_command.extend(["--event-date", str(args.run_date)])
            if args.mlb_market_input_path:
                fetch_command.extend(["--input-path", str(args.mlb_market_input_path.resolve())])
            try:
                run_step("Fetch MLB Market Props", fetch_command)
            except Exception as exc:
                market_fetch_failed = True
                print(
                    f"[warning] MLB market fetch failed; continuing with the latest available pool and publication artifacts. {format_step_failure(exc)}"
                )

        if not args.mlb_skip_update_data:
            update_command = [
                args.python,
                str(MLB_DATA_UPDATER),
            ]
            if args.run_date:
                update_command.extend(["--through-date", str(args.run_date)])
            if args.mlb_incremental_update:
                update_command.append("--incremental")
            if MLB_DATA_UPDATER.exists():
                try:
                    run_step("Update MLB Processed Data", update_command)
                except Exception as exc:
                    print(
                        f"[warning] MLB processed-data update failed; continuing with the latest checked-in processed data. {format_step_failure(exc)}"
                    )
            else:
                print(
                    f"[warning] MLB processed-data updater was not found at {MLB_DATA_UPDATER}; "
                    "skipping that step and continuing with the latest checked-in processed data."
                )

        if not args.mlb_skip_generate:
            generated_pool_csv, generated_summary_json = derive_generated_mlb_pool_outputs(args.run_date)
            command = [
                args.python,
                str(MLB_GENERATOR),
                "--daily-runs-root",
                str(MLB_DAILY_RUNS_ROOT),
                "--data-dir",
                str(args.mlb_data_dir.resolve()),
                "--manifest",
                str(args.mlb_manifest.resolve()),
                "--fallback-policy",
                str(args.mlb_fallback_policy),
            ]
            if args.run_date:
                command.extend(["--run-date", str(args.run_date)])
            try:
                run_step("Generate MLB Raw Prediction Pool", command)
            except Exception as exc:
                print(
                    f"[warning] MLB raw pool generation failed; falling back to the latest available run directory. {format_step_failure(exc)}"
                )
            if generated_pool_csv and generated_pool_csv.exists() and generated_summary_json.exists():
                try:
                    summary = json.loads(generated_summary_json.read_text(encoding="utf-8"))
                    if not bool(summary.get("exact_run_date_match", True)):
                        print(
                            "[warning] MLB raw pool used the latest available processed row template for this run date; "
                            "publishing the generated current-day pool anyway."
                        )
                except Exception:
                    pass

        if generated_pool_csv and generated_pool_csv.exists():
            pool_csv = generated_pool_csv
            used_generated_pool = True
        else:
            pool_csv = find_latest_mlb_pool_csv(MLB_DAILY_RUNS_ROOT, preferred_run_stamp)

    mlb_dist_json = Path(getattr(args, "private_output_dir", DEFAULT_PRIVATE_OUTPUT_DIR)).resolve() / "mlb" / "data" / "daily_predictions.json"

    pool_digits = "".join(char for char in pool_csv.stem if char.isdigit())
    governance_status_json = pool_csv.parent / "governance" / "governance_status.json"
    if len(pool_digits) >= 8:
        pool_date = datetime.strptime(pool_digits[:8], "%Y%m%d").date()
        run_step(
            "Capture Immutable MLB Complete Slate",
            [
                args.python,
                str(MLB_GOVERNANCE_CAPTURE),
                "--provider-csv",
                str(MLB_PROVIDER_OBSERVATIONS),
                "--pool-csv",
                str(pool_csv),
                "--run-dir",
                str(pool_csv.parent),
                "--run-date",
                pool_date.isoformat(),
            ],
        )
    if len(pool_digits) >= 8 and MLB_CONFIDENCE_CALIBRATOR.exists():
        run_step(
            "Calibrate MLB Published Confidence",
            [
                args.python,
                str(MLB_CONFIDENCE_CALIBRATOR),
                "--season",
                str(pool_date.year),
                "--before-date",
                pool_date.isoformat(),
                "--official-api-fallback",
                "--policy-version",
                MLB_PRIMARY_POLICY_PROFILE,
            ],
        )
        if MLB_PICK_SURVIVAL_MODEL.exists():
            run_step(
                "Train MLB Pick-Survival Shadow Model",
                [
                    args.python,
                    str(MLB_PICK_SURVIVAL_MODEL),
                    "--season",
                    str(pool_date.year),
                    "--before-date",
                    pool_date.isoformat(),
                    "--top-k",
                    str(MLB_PICK_SURVIVAL_TOP_K),
                ],
            )
        if MLB_LATENT_POOL_REPLAY.exists():
            run_step(
                "Replay MLB Latent Complete-Slate Holdout",
                [
                    args.python,
                    str(MLB_LATENT_POOL_REPLAY),
                    "--daily-runs-root",
                    str(MLB_DAILY_RUNS_ROOT),
                    "--processed-root",
                    str(args.mlb_data_dir.resolve()),
                    "--output-json",
                    str(MLB_LATENT_POOL_REPLAY_REPORT),
                ],
            )

    def run_selector_for(
        active_pool_csv: Path,
        *,
        label: str = "Select MLB High-Precision Prediction Board",
        suffix: str | None = None,
        extra_args: list[str] | None = None,
        refresh_history: bool = False,
    ) -> tuple[Path, Path]:
        active_selected_csv, active_summary_json = derive_mlb_selector_outputs(active_pool_csv, suffix=suffix)
        command = [
            args.python,
            str(MLB_SELECTOR),
            "--pool-csv",
            str(active_pool_csv),
            "--out-csv",
            str(active_selected_csv),
            "--summary-json",
            str(active_summary_json),
            "--top-n",
            str(int(args.mlb_top_n)),
            "--policy-version",
            MLB_PRIMARY_POLICY_PROFILE,
        ]
        if extra_args:
            command.extend(extra_args)
        if refresh_history:
            command.extend(["--refresh-history-cache", "--refresh-bet-profile-cache"])
        run_step(
            label,
            command,
        )
        return active_selected_csv, active_summary_json

    selected_csv, summary_json = run_selector_for(
        pool_csv,
        extra_args=MLB_PRIMARY_POLICY_ARGS,
        refresh_history=bool(args.mlb_refresh_history_caches),
    )
    standard_selected_csv, standard_summary_json = derive_mlb_selector_outputs(pool_csv)
    market_profile = load_mlb_market_profile(pool_csv) if pool_csv.exists() else {
        "rows": 0,
        "real_market_rows": 0,
        "synthetic_rows": 0,
        "price_confirmed_rows": 0,
    }
    publication_strategy = MLB_PRIMARY_POLICY_PROFILE

    if used_generated_pool and summary_json.exists():
        selected_rows = selected_row_count(summary_json)
        min_publish_plays = max(0, int(args.mlb_min_publish_plays))
        min_rescue_plays = max(0, int(args.mlb_min_rescue_plays))
        if selected_rows < min_publish_plays:
            best_selected_csv = selected_csv
            best_summary_json = summary_json
            best_rows = selected_rows
            best_strategy = MLB_PRIMARY_POLICY_PROFILE

            if best_rows >= min_rescue_plays:
                if best_selected_csv != standard_selected_csv or best_summary_json != standard_summary_json:
                    promote_mlb_selector_outputs(
                        best_selected_csv,
                        best_summary_json,
                        standard_selected_csv,
                        standard_summary_json,
                    )
                selected_csv = standard_selected_csv
                summary_json = standard_summary_json
                publication_strategy = best_strategy
                annotate_mlb_summary(summary_json, publication_strategy=publication_strategy, market_profile=market_profile)
            else:
                annotate_mlb_summary(
                    summary_json,
                    publication_strategy=publication_strategy,
                    market_profile=market_profile,
                    publication_state="withheld_current_pool",
                )
                print(
                    "[warning] Generated MLB board was too small for publication "
                    f"({best_rows} plays < {min_publish_plays}); publishing the current-date "
                    "withheld state instead of regressing to an older slate."
                )
        else:
            annotate_mlb_summary(summary_json, publication_strategy=publication_strategy, market_profile=market_profile)
    elif summary_json.exists():
        annotate_mlb_summary(summary_json, publication_strategy=publication_strategy, market_profile=market_profile)

    parlay_json = pool_csv.with_name(f"{pool_csv.stem}_daily_parlay.json")
    run_step(
        "Select MLB Daily Consistency Parlay",
        [
            args.python,
            str(MLB_PARLAY_SELECTOR),
            "--pool-csv",
            str(pool_csv),
            "--out-json",
            str(parlay_json),
        ],
    )

    # PARLAY_POLICY_V2 calibration ledger ingestion (STREAM A) -- admits
    # PAST, already-settled days' real outcomes. Never touches today's own
    # slate_id, which is what keeps this forward-only: whatever V2 decides
    # for today, below, can only ever be informed by strictly earlier
    # calibration_admitted_at rows. Best-effort per day (a day whose
    # outcomes aren't in Data-Proc-MLB yet just admits zero rows this run
    # and gets retried on a later run, per MLB_PARLAY_V2_INGEST_LOOKBACK_DAYS).
    ingest_anchor = resolve_effective_run_date(args.run_date)
    for lookback in range(1, MLB_PARLAY_V2_INGEST_LOOKBACK_DAYS + 1):
        ingest_stamp = (ingest_anchor - timedelta(days=lookback)).strftime("%Y%m%d")
        ingest_pool_csv = MLB_DAILY_RUNS_ROOT / ingest_stamp / f"daily_prediction_pool_{ingest_stamp}.csv"
        if not ingest_pool_csv.exists():
            continue
        try:
            run_step(
                f"Ingest MLB Calibration Observations ({ingest_stamp})",
                [
                    args.python,
                    "-m",
                    MLB_PARLAY_V2_INGEST_MODULE,
                    "--stamp",
                    ingest_stamp,
                    "--ledger",
                    str(MLB_PARLAY_V2_CALIBRATION_LEDGER),
                ],
            )
        except Exception as exc:  # noqa: BLE001 -- deliberate: ingestion is additive, never blocks the rest of the pipeline
            print(f"[warning] Calibration ingestion for {ingest_stamp} failed, will retry on a later run: {format_step_failure(exc)}")

        # Pair-level ledger (SEPARATE research stream, see
        # MLB_PARLAY_V2_PAIR_LEDGER above) -- runs for the same settled
        # day, after the leg-level ledger admission above so it can gate
        # pairing on that day's just-admitted leg observations too.
        # Best-effort, same as the leg-level step: never blocks the rest
        # of the pipeline.
        try:
            run_step(
                f"Ingest MLB Pair Observations ({ingest_stamp})",
                [
                    args.python,
                    "-m",
                    MLB_PARLAY_V2_PAIR_INGEST_MODULE,
                    "--stamp",
                    ingest_stamp,
                    "--pair-ledger",
                    str(MLB_PARLAY_V2_PAIR_LEDGER),
                    "--calibration-ledger",
                    str(MLB_PARLAY_V2_CALIBRATION_LEDGER),
                ],
            )
        except Exception as exc:  # noqa: BLE001 -- deliberate: pair ingestion is additive, never blocks the rest of the pipeline
            print(f"[warning] Pair ingestion for {ingest_stamp} failed, will retry on a later run: {format_step_failure(exc)}")

        # Settlement -> policy evidence (grades that day's ALREADY-FROZEN
        # decision, never re-selects). No-op if no decision was frozen for
        # this day, or if it isn't fully gradeable yet -- see
        # settle_evidence.py's own status codes. Best-effort, same as the
        # two ingestion steps above.
        try:
            run_step(
                f"Settle MLB Policy Evidence ({ingest_stamp})",
                [
                    args.python,
                    "-m",
                    MLB_PARLAY_V2_SETTLE_EVIDENCE_MODULE,
                    "--date",
                    ingest_stamp,
                    "--decision-record-ledger",
                    str(MLB_PARLAY_V2_DECISION_RECORD_LEDGER),
                    "--evidence-store-root",
                    str(MLB_PARLAY_V2_EVIDENCE_STORE_ROOT),
                    "--policy-version",
                    MLB_PARLAY_V2_POLICY_VERSION,
                ],
            )
        except Exception as exc:  # noqa: BLE001 -- deliberate: settlement is additive, never blocks the rest of the pipeline
            print(f"[warning] Evidence settlement for {ingest_stamp} failed, will retry on a later run: {format_step_failure(exc)}")

    # PARLAY_POLICY_V2 -- separate product path, additive step. Failure
    # here must never block the singles/legacy-parlay export above: the
    # exporter treats a missing/unreadable V2 artifact as a clear
    # "unavailable" state (see parlay_v2/frontend_payload.py), never as an
    # export failure.
    parlay_v2_slate_id = pool_csv.stem.removeprefix("daily_prediction_pool_")
    parlay_v2_json = pool_csv.with_name(f"{pool_csv.stem}_parlay_v2.json")
    try:
        run_step(
            "Run PARLAY_POLICY_V2 (Parlays Tab)",
            [
                args.python,
                "-m",
                MLB_PARLAY_V2_RUNNER_MODULE,
                "--pool-csv",
                str(pool_csv),
                "--slate-id",
                parlay_v2_slate_id,
                "--out-json",
                str(parlay_v2_json),
                "--calibration-ledger",
                str(MLB_PARLAY_V2_CALIBRATION_LEDGER),
                "--decision-record-ledger",
                str(MLB_PARLAY_V2_DECISION_RECORD_LEDGER),
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate: V2 is additive, never blocks singles publication
        print(f"[warning] PARLAY_POLICY_V2 step failed, Parlays tab will report unavailable: {format_step_failure(exc)}")

    run_step(
        "Export MLB Prediction Payload",
        [
            args.python,
            str(MLB_EXPORTER),
            "--input-csv",
            str(selected_csv),
            "--summary-json",
            str(summary_json),
            "--parlay-json",
            str(parlay_json),
            "--parlay-v2-json",
            str(parlay_v2_json),
            "--governance-json",
            str(governance_status_json),
            "--output",
            str(MLB_WEB_JSON),
            "--output-dist",
            str(mlb_dist_json),
        ],
    )

    # Newest MLB predictor (real joint game simulation + pitcher/bullpen
    # enrichment) wired into the main board -- see MLB_TEAM_MARKET_PREDICTOR
    # above. Additive only: merges "mlb_team_market_plays" into both real
    # published copies of daily_predictions.json the exporter just wrote,
    # never touches "plays" or anything else in either file. Wrapped like
    # PARLAY_POLICY_V2 above -- a real live-fetch failure here (schedule,
    # odds, or a StatsAPI hiccup) must never block the player-prop board
    # that already published successfully.
    try:
        # Same real, filename-derived effective run date the governance
        # capture step above resolves from pool_csv -- args.run_date alone
        # can be None (the pipeline's own --force-run default), so fall
        # back to it only when the pool filename doesn't carry a real date.
        team_market_digits = "".join(char for char in pool_csv.stem if char.isdigit())
        team_market_run_date = (
            datetime.strptime(team_market_digits[:8], "%Y%m%d").date().isoformat()
            if len(team_market_digits) >= 8
            else args.run_date
        )
        team_market_command = [args.python, str(MLB_TEAM_MARKET_PREDICTOR)]
        if team_market_run_date:
            team_market_command += ["--run-date", str(team_market_run_date)]
        team_market_command += [
            "--daily-predictions-path", str(MLB_WEB_JSON),
            "--daily-predictions-path", str(mlb_dist_json),
        ]
        run_step("Generate MLB Team-Market Predictions (newest predictor)", team_market_command)
    except Exception as exc:  # noqa: BLE001 -- deliberate: additive, never blocks singles publication
        print(f"[warning] MLB team-market predictor step failed, main board keeps its existing player-prop plays only: {format_step_failure(exc)}")

    # Real headshot enrichment for PARLAY_POLICY_V2 legs -- see
    # MLB_PARLAY_HEADSHOT_ENRICHER above. Additive only, and a live
    # MLB Stats API lookup failure here must never block the boards that
    # already published successfully above.
    try:
        run_step(
            "Enrich MLB Parlay Leg Headshots",
            [
                args.python,
                str(MLB_PARLAY_HEADSHOT_ENRICHER),
                "--daily-predictions-path", str(MLB_WEB_JSON),
                "--daily-predictions-path", str(mlb_dist_json),
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate: additive, never blocks singles publication
        print(f"[warning] MLB parlay leg headshot enrichment failed, legs fall back to their monogram: {format_step_failure(exc)}")

    # Real FanDuel betslip deep links for PARLAY_POLICY_V2 pairs -- see
    # MLB_PARLAY_BETSLIP_ENRICHER above. Additive only, and a live
    # FanDuel public-feed failure here must never block the boards that
    # already published successfully above.
    try:
        run_step(
            "Enrich MLB Parlay Leg Betslip Links",
            [
                args.python,
                str(MLB_PARLAY_BETSLIP_ENRICHER),
                "--daily-predictions-path", str(MLB_WEB_JSON),
                "--daily-predictions-path", str(mlb_dist_json),
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate: additive, never blocks singles publication
        print(f"[warning] MLB parlay leg betslip enrichment failed, no betslip link will be shown: {format_step_failure(exc)}")

    # Real, deduplicated local headshot cache -- see MLB_HEADSHOT_CACHE
    # above. Additive only, and a real fetch failure for any one player
    # here must never block the boards that already published above --
    # that player's card simply keeps its real remote URL.
    try:
        run_step(
            "Cache MLB Player Headshots",
            [
                args.python,
                str(MLB_HEADSHOT_CACHE),
                "--daily-predictions-path", str(MLB_WEB_JSON),
                "--daily-predictions-path", str(mlb_dist_json),
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate: additive, never blocks singles publication
        print(f"[warning] MLB headshot cache step failed, images fall back to their real remote CDN URL: {format_step_failure(exc)}")

    # Run max-winrate selector on the raw pool for the tightest possible board
    max_wr_csv = pool_csv.with_name(pool_csv.stem + "_max_winrate.csv")
    max_wr_summary = pool_csv.with_name(pool_csv.stem + "_max_winrate_summary.json")
    max_wr_json = pool_csv.with_name(pool_csv.stem + "_max_winrate_board.json")
    run_step(
        "Select MLB Max Win-Rate Board",
        [
            args.python,
            str(MLB_MAX_WINRATE_SELECTOR),
            "--pool-csv",
            str(pool_csv),
            "--out-csv",
            str(max_wr_csv),
            "--summary-json",
            str(max_wr_summary),
            "--out-json",
            str(max_wr_json),
            "--min-model-prob", "0.92",
            "--min-bucket-win-rate", "0.86",
            "--min-bucket-samples", "500",
            "--min-history-rows", "40",
            "--min-market-books", "5",
            "--min-common-market-books", "2",
            "--max-days-since-history", "3",
            "--min-edge", "0.35",
            "--max-board-size", "7",
        ],
    )

    # The max-win-rate board remains a shadow diagnostic. Publication stays on
    # the fully calibrated selector, which also enforces EV and book identity.
    return pool_csv, selected_csv, summary_json


def build_site(args: argparse.Namespace, output_dir: Path) -> None:
    run_step(
        "Build Unified Static Site",
        [
            args.python,
            str(BUILD_STATIC_SITE),
            "--output",
            str(output_dir),
            "--private-output",
            str(Path(getattr(args, "private_output_dir", DEFAULT_PRIVATE_OUTPUT_DIR)).resolve()),
        ],
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()

    if args.skip_nba and args.skip_mlb:
        raise SystemExit("Nothing to do: both --skip-nba and --skip-mlb were set.")

    should_run, schedule_message = check_schedule_gate(args)
    print("\n" + "=" * 88)
    print("SCHEDULE CHECK")
    print("=" * 88)
    print(schedule_message)
    if not should_run:
        return

    target_run_date = str(resolve_effective_run_date(args.run_date))
    if not args.skip_nba:
        archive_previous_prediction_payload(NBA_WEB_JSON, target_run_date)
    if not args.skip_mlb:
        archive_previous_prediction_payload(MLB_WEB_JSON, target_run_date)

    mlb_pool_csv: Path | None = None
    mlb_selected_csv: Path | None = None
    mlb_summary_json: Path | None = None
    nba_failure: Exception | None = None

    if not args.skip_nba:
        try:
            run_nba(args, output_dir)
        except Exception as exc:
            if args.skip_mlb:
                raise
            nba_failure = exc
            print("\n" + "=" * 88)
            print("NBA Prediction Refresh Failed Safely")
            print("=" * 88)
            print(format_step_failure(exc))
            print(
                "Continuing with MLB refresh and static-site rebuild so other sports can still publish. "
                "The previous NBA payload will remain in place until the NBA pipeline succeeds again."
            )

    if not args.skip_mlb:
        mlb_pool_csv, mlb_selected_csv, mlb_summary_json = run_mlb(args, output_dir)

    if not args.skip_build_site:
        build_site(args, output_dir)

    print("\n" + "=" * 88)
    print("SHARED DAILY PREDICTION REFRESH COMPLETE")
    print("=" * 88)
    print(f"Output directory: {output_dir}")
    if not args.skip_nba:
        if nba_failure is not None:
            print(f"NBA status:       failed safely ({format_step_failure(nba_failure)})")
        print(f"NBA web payload:  {NBA_WEB_JSON}")
        print(f"NBA protected payload: {Path(getattr(args, 'private_output_dir', DEFAULT_PRIVATE_OUTPUT_DIR)).resolve() / 'nba' / 'data' / 'daily_predictions.json'}")
    if not args.skip_mlb:
        print(f"MLB pool CSV:     {mlb_pool_csv}")
        print(f"MLB selected CSV: {mlb_selected_csv}")
        print(f"MLB summary JSON: {mlb_summary_json}")
        print(f"MLB web payload:  {MLB_WEB_JSON}")
        print(f"MLB protected payload: {Path(getattr(args, 'private_output_dir', DEFAULT_PRIVATE_OUTPUT_DIR)).resolve() / 'mlb' / 'data' / 'daily_predictions.json'}")


if __name__ == "__main__":
    main()
