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
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SCRIPT_PATH = Path(__file__).resolve()
SITE_ROOT = SCRIPT_PATH.parents[1]
SPORTS_ROOT = SITE_ROOT.parent
REPO_ROOT = SPORTS_ROOT.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "dist"

NBA_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
NBA_RUNNER = NBA_PREDICTOR_ROOT / "scripts" / "run_daily_market_pipeline.py"
NBA_EXPORTER = NBA_PREDICTOR_ROOT / "scripts" / "export_daily_predictions_web.py"
NBA_WEB_JSON = REPO_ROOT / "sports" / "nba" / "web" / "data" / "daily_predictions.json"
NBA_CARDS_JSON = REPO_ROOT / "sports" / "nba" / "web" / "data" / "cards.json"

MLB_DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"
MLB_DATA_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
MLB_MANIFEST = MLB_DATA_DIR / "update_manifest_2026.json"
MLB_MARKET_FETCHER = REPO_ROOT / "Player-Predictor" / "scripts" / "fetch_mlb_market_props.py"
MLB_DATA_UPDATER = REPO_ROOT / "Player-Predictor" / "scripts" / "update_mlb_processed_data.py"
MLB_GENERATOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "generate_daily_prediction_pool.py"
MLB_SELECTOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "select_high_precision_predictions.py"
MLB_EXPORTER = REPO_ROOT / "sports" / "mlb" / "scripts" / "export_web_prediction_payload.py"
MLB_WEB_JSON = REPO_ROOT / "sports" / "mlb" / "web" / "data" / "daily_predictions.json"

BUILD_STATIC_SITE = REPO_ROOT / "sports" / "site" / "pipeline" / "build_static_site.py"
SPORTSBOOK_TIMEZONE = ZoneInfo("America/New_York")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the shared NBA + MLB daily prediction refresh.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child steps.")
    parser.add_argument("--run-date", type=str, default=None, help="Optional YYYY-MM-DD run date.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Published static output directory.")
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
    parser.add_argument("--nba-allow-heuristic-fallback", action="store_true", help="Allow NBA heuristic fallback if model loading fails.")
    parser.add_argument("--nba-skip-update-data", action="store_true", help="Skip the NBA official-data refresh step.")
    parser.add_argument("--nba-skip-collect-market", action="store_true", help="Skip the NBA market collection step.")
    parser.add_argument("--nba-skip-align", action="store_true", help="Skip the NBA market alignment step.")
    parser.add_argument("--nba-skip-backtest", action="store_true", help="Skip the NBA backtest refresh step.")
    parser.add_argument("--nba-skip-cutoff-meta-monitor", action="store_true", help="Skip the NBA cutoff-meta monitor step.")
    parser.add_argument(
        "--nba-fallback-to-latest-manifest-on-failure",
        action="store_true",
        help="If the live NBA pipeline fails, export from the latest available checked-in NBA manifest instead of aborting.",
    )

    parser.add_argument("--mlb-pool-csv", type=Path, default=None, help="Explicit raw MLB daily prediction pool CSV.")
    parser.add_argument("--mlb-skip-fetch-market", action="store_true", help="Skip fetching same-day MLB market props.")
    parser.add_argument("--mlb-skip-update-data", action="store_true", help="Skip rebuilding MLB processed player files from source data.")
    parser.add_argument("--mlb-skip-generate", action="store_true", help="Skip generating a fresh MLB raw prediction pool from processed MLB data.")
    parser.add_argument("--mlb-data-dir", type=Path, default=MLB_DATA_DIR, help="MLB processed-data root used by the raw pool generator.")
    parser.add_argument("--mlb-manifest", type=Path, default=MLB_MANIFEST, help="Optional MLB processed-data manifest used by the raw pool generator.")
    parser.add_argument(
        "--mlb-market-provider",
        type=str,
        default="rotowire",
        choices=["rotowire", "odds_api", "snapshot"],
        help="Provider used by the MLB market fetcher. 'odds_api' is preserved as a backward-compatible alias.",
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
        default=6,
        help="Minimum selected MLB plays required before publishing a generated pool; otherwise fall back to the latest richer existing pool when available.",
    )
    parser.add_argument(
        "--mlb-min-rescue-plays",
        type=int,
        default=3,
        help="Minimum same-day MLB plays required for an empty-board rescue strategy to publish instead of falling back to an older pool.",
    )
    parser.add_argument("--mlb-top-n", type=int, default=10, help="Maximum number of MLB plays to keep.")
    return parser.parse_args()


def run_step(label: str, command: list[str], cwd: Path = REPO_ROOT) -> None:
    print("\n" + "=" * 88)
    print(label)
    print("=" * 88)
    print("Command:", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def validate_schedule_args(hour: int, minute: int) -> tuple[int, int]:
    if not 0 <= int(hour) <= 23:
        raise SystemExit(f"--scheduled-hour must be between 0 and 23, received {hour!r}")
    if not 0 <= int(minute) <= 59:
        raise SystemExit(f"--scheduled-minute must be between 0 and 59, received {minute!r}")
    return int(hour), int(minute)


def resolve_effective_run_date(run_date: str | None, now: datetime | None = None) -> date:
    if run_date:
        return datetime.fromisoformat(str(run_date)).date()
    current = now or datetime.now(SPORTSBOOK_TIMEZONE)
    if current.tzinfo is None:
        current = current.replace(tzinfo=SPORTSBOOK_TIMEZONE)
    else:
        current = current.astimezone(SPORTSBOOK_TIMEZONE)
    return current.date()


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


def find_latest_nba_manifest(preferred_run_stamp: str | None = None) -> Path:
    manifest_root = NBA_PREDICTOR_ROOT / "model" / "analysis" / "daily_runs"
    manifests = sorted(
        manifest_root.glob("**/daily_market_pipeline_manifest_*.json"),
        key=lambda path: (path.stem, path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not manifests:
        raise FileNotFoundError(f"No NBA daily pipeline manifest was found under {manifest_root}")

    if preferred_run_stamp:
        suffix = f"daily_market_pipeline_manifest_{preferred_run_stamp}"
        for path in manifests:
            if path.stem == suffix:
                return path

    return manifests[0]


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


def annotate_mlb_summary(summary_json: Path, *, publication_strategy: str, market_profile: dict[str, int]) -> None:
    if not summary_json.exists():
        return
    try:
        payload = load_json(summary_json)
    except Exception:
        return
    payload["publication_strategy"] = str(publication_strategy)
    payload["pool_market_profile"] = {key: int(value) for key, value in market_profile.items()}
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def derive_generated_mlb_pool_outputs(run_date: str | None) -> tuple[Path, Path]:
    local_date = resolve_effective_run_date(run_date)
    run_stamp = local_date.strftime("%Y%m%d")
    run_dir = MLB_DAILY_RUNS_ROOT / run_stamp
    return (
        run_dir / f"daily_prediction_pool_{run_stamp}.csv",
        run_dir / f"daily_prediction_pool_{run_stamp}.json",
    )


def run_nba(args: argparse.Namespace, output_dir: Path) -> None:
    nba_dist_json = output_dir / "nba" / "data" / "daily_predictions.json"
    def export_nba_manifest(manifest_path: Path, label: str) -> None:
        command = [
            args.python,
            str(NBA_EXPORTER),
            "--manifest",
            str(manifest_path.resolve()),
            "--out-json",
            str(NBA_WEB_JSON),
            "--out-dist",
            str(nba_dist_json),
            "--cards-json",
            str(NBA_CARDS_JSON),
        ]
        run_step(label, command)

    if args.nba_manifest:
        export_nba_manifest(args.nba_manifest, "Export NBA Predictions From Existing Manifest")
        return

    command = [
        args.python,
        str(NBA_RUNNER),
        "--policy-profile",
        str(args.nba_policy_profile),
        "--web-cards-json",
        str(NBA_CARDS_JSON),
        "--skip-build-site",
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

    run_date = resolve_effective_run_date(args.run_date)
    expected_manifest = derive_nba_manifest_path(run_date)
    preferred_run_stamp = run_date.strftime("%Y%m%d")

    try:
        run_step("Run NBA Daily Prediction Pipeline", command)
        if not expected_manifest.exists():
            raise FileNotFoundError(
                "NBA daily pipeline completed but the expected same-day manifest was not found: "
                f"{expected_manifest}"
            )
        export_nba_manifest(expected_manifest, "Export NBA Prediction Payload")
    except Exception as exc:
        if not args.nba_fallback_to_latest_manifest_on_failure:
            raise
        fallback_manifest = expected_manifest if expected_manifest.exists() else find_latest_nba_manifest(preferred_run_stamp)
        print(
            "[warning] Live NBA pipeline failed; falling back to checked-in manifest export. "
            f"Using manifest: {fallback_manifest}. Root error: {exc}"
        )
        export_nba_manifest(fallback_manifest, "Export NBA Prediction Payload From Fallback Manifest")


def run_mlb(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path, Path]:
    preferred_run_stamp = run_stamp_from_date(args.run_date)
    generated_pool_csv: Path | None = None
    generated_summary_json: Path | None = None
    used_generated_pool = False

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
            run_step("Fetch MLB Market Props", fetch_command)

        if not args.mlb_skip_update_data:
            update_command = [
                args.python,
                str(MLB_DATA_UPDATER),
            ]
            if args.run_date:
                update_command.extend(["--through-date", str(args.run_date)])
            if MLB_DATA_UPDATER.exists():
                run_step("Update MLB Processed Data", update_command)
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
            run_step("Generate MLB Raw Prediction Pool", command)
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

    mlb_dist_json = output_dir / "mlb" / "data" / "daily_predictions.json"

    def run_selector_for(
        active_pool_csv: Path,
        *,
        label: str = "Select MLB High-Precision Prediction Board",
        suffix: str | None = None,
        extra_args: list[str] | None = None,
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
        ]
        if extra_args:
            command.extend(extra_args)
        run_step(
            label,
            command,
        )
        return active_selected_csv, active_summary_json

    selected_csv, summary_json = run_selector_for(pool_csv)
    standard_selected_csv, standard_summary_json = derive_mlb_selector_outputs(pool_csv)
    market_profile = load_mlb_market_profile(pool_csv) if used_generated_pool and pool_csv.exists() else {
        "rows": 0,
        "real_market_rows": 0,
        "synthetic_rows": 0,
        "price_confirmed_rows": 0,
    }
    publication_strategy = "primary"

    if used_generated_pool and summary_json.exists():
        selected_rows = selected_row_count(summary_json)
        min_publish_plays = max(0, int(args.mlb_min_publish_plays))
        min_rescue_plays = max(0, int(args.mlb_min_rescue_plays))
        if selected_rows < min_publish_plays:
            best_selected_csv = selected_csv
            best_summary_json = summary_json
            best_rows = selected_rows
            best_strategy = "primary"

            if market_profile.get("real_market_rows", 0) > 0:
                rescue_selected_csv, rescue_summary_json = run_selector_for(
                    pool_csv,
                    label="Rescue MLB Board With Relaxed Real-Market Filters",
                    suffix="real_market_rescue",
                    extra_args=[
                        "--min-market-books",
                        "1",
                        "--min-hit-probability",
                        "0.60",
                        "--min-graded-hit-rate",
                        "0.72",
                        "--min-abs-edge",
                        "0.40",
                    ],
                )
                rescue_rows = selected_row_count(rescue_summary_json)
                if rescue_rows > best_rows:
                    best_selected_csv = rescue_selected_csv
                    best_summary_json = rescue_summary_json
                    best_rows = rescue_rows
                    best_strategy = "real_market_rescue"

            if best_rows < min_rescue_plays:
                synthetic_top_n = max(min_rescue_plays, min(int(args.mlb_top_n), 4))
                rescue_selected_csv, rescue_summary_json = run_selector_for(
                    pool_csv,
                    label="Rescue MLB Board With Historically Validated Synthetic Fallback",
                    suffix="synthetic_rescue",
                    extra_args=[
                        "--top-n",
                        str(int(synthetic_top_n)),
                        "--min-market-books",
                        "0",
                        "--min-hit-probability",
                        "0.64",
                        "--min-graded-hit-rate",
                        "0.78",
                        "--min-abs-edge",
                        "0.55",
                        "--max-per-market-bucket",
                        "2",
                        "--prefer-confident-side",
                        "--min-historical-bet-profile-support",
                        "12",
                        "--min-historical-bet-profile-win-rate",
                        "0.55",
                        "--min-historical-market-availability-support",
                        "20",
                        "--min-historical-market-availability-rate",
                        "0.45",
                    ],
                )
                rescue_rows = selected_row_count(rescue_summary_json)
                if rescue_rows > best_rows:
                    best_selected_csv = rescue_selected_csv
                    best_summary_json = rescue_summary_json
                    best_rows = rescue_rows
                    best_strategy = "synthetic_rescue"

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
                annotate_mlb_summary(summary_json, publication_strategy=best_strategy, market_profile=market_profile)
                try:
                    fallback_pool_csv = find_latest_mlb_pool_csv(
                        MLB_DAILY_RUNS_ROOT,
                        preferred_run_stamp,
                        exclude_paths={pool_csv},
                    )
                except FileNotFoundError:
                    fallback_pool_csv = None
                if fallback_pool_csv is not None:
                    print(
                        "[warning] Generated MLB board was too small for publication "
                        f"({best_rows} plays < {min_publish_plays}); "
                        "falling back to the latest richer existing MLB pool."
                    )
                    pool_csv = fallback_pool_csv
                    selected_csv, summary_json = run_selector_for(pool_csv)
                    publication_strategy = "latest_existing_pool"
        else:
            annotate_mlb_summary(summary_json, publication_strategy=publication_strategy, market_profile=market_profile)

    run_step(
        "Export MLB Prediction Payload",
        [
            args.python,
            str(MLB_EXPORTER),
            "--input-csv",
            str(selected_csv),
            "--summary-json",
            str(summary_json),
            "--output",
            str(MLB_WEB_JSON),
            "--output-dist",
            str(mlb_dist_json),
        ],
    )

    return pool_csv, selected_csv, summary_json


def build_site(args: argparse.Namespace, output_dir: Path) -> None:
    run_step(
        "Build Unified Static Site",
        [
            args.python,
            str(BUILD_STATIC_SITE),
            "--output",
            str(output_dir),
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

    mlb_pool_csv: Path | None = None
    mlb_selected_csv: Path | None = None
    mlb_summary_json: Path | None = None

    if not args.skip_nba:
        run_nba(args, output_dir)

    if not args.skip_mlb:
        mlb_pool_csv, mlb_selected_csv, mlb_summary_json = run_mlb(args, output_dir)

    if not args.skip_build_site:
        build_site(args, output_dir)

    print("\n" + "=" * 88)
    print("SHARED DAILY PREDICTION REFRESH COMPLETE")
    print("=" * 88)
    print(f"Output directory: {output_dir}")
    if not args.skip_nba:
        print(f"NBA web payload:  {NBA_WEB_JSON}")
        print(f"NBA dist payload: {output_dir / 'nba' / 'data' / 'daily_predictions.json'}")
    if not args.skip_mlb:
        print(f"MLB pool CSV:     {mlb_pool_csv}")
        print(f"MLB selected CSV: {mlb_selected_csv}")
        print(f"MLB summary JSON: {mlb_summary_json}")
        print(f"MLB web payload:  {MLB_WEB_JSON}")
        print(f"MLB dist payload: {output_dir / 'mlb' / 'data' / 'daily_predictions.json'}")


if __name__ == "__main__":
    main()
