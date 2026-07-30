#!/usr/bin/env python3
"""
Daily unattended market pipeline runner.

This script is designed to run once per day, typically around 2am local time.
It performs:
1. refresh current-season official game logs through yesterday
2. collect recent Covers prop lines for both recent historical games and the
   current/upcoming slate
3. align historical market lines onto the current season processed files
4. refresh the historical inference backtest calibration CSV
5. build a filtered current-slate market snapshot
6. run the market decision pipeline and write dated outputs
7. rebuild the static site bundle so dist/predictions/index.html is current
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
SPORT_ROOT = REPO_ROOT.parents[1]
WORKSPACE_ROOT = SPORT_ROOT.parents[1]
UNIFIED_SITE_PIPELINE_ROOT = WORKSPACE_ROOT / "sports" / "site" / "pipeline"
MARKET_ROOT = REPO_ROOT / "data copy" / "raw" / "market_odds" / "nba"
ANALYSIS_ROOT = REPO_ROOT / "model" / "analysis" / "daily_runs"
DATA_PROC_ROOT = REPO_ROOT / "Data-Proc"
EASTERN_TZ = ZoneInfo("America/New_York")


def resolve_run_timestamp(run_date: str | None, now: datetime | None = None) -> pd.Timestamp:
    if run_date:
        return pd.Timestamp(run_date).normalize()
    resolved_now = now if now is not None else datetime.now(EASTERN_TZ)
    if resolved_now.tzinfo is None:
        resolved_now = resolved_now.replace(tzinfo=EASTERN_TZ)
    return pd.Timestamp(resolved_now.astimezone(EASTERN_TZ)).normalize().tz_localize(None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily market data + prediction pipeline.")
    parser.add_argument("--season", type=int, default=None, help="Season end year. Defaults from current date.")
    parser.add_argument("--latest", action="store_true", help="Use latest manifest instead of production for the final board.")
    parser.add_argument(
        "--policy-profile",
        type=str,
        default="production_board_objective_b12",
        help="Primary market policy profile for the live board.",
    )
    parser.add_argument(
        "--shadow-policy-profiles",
        nargs="*",
        default=["shadow_edge_append_agree1_p90_x1"],
        help="Optional additional policy profiles to run for research/monitoring only.",
    )
    parser.add_argument(
        "--cutoff-meta-monitor-lookback-days",
        type=int,
        default=21,
        help="Lookback span for daily unified cutoff-meta monitor.",
    )
    parser.add_argument(
        "--cutoff-meta-monitor-start-date",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD override for monitor start date.",
    )
    parser.add_argument(
        "--cutoff-meta-monitor-end-date",
        type=str,
        default=None,
        help="Optional YYYY-MM-DD override for monitor end date.",
    )
    parser.add_argument(
        "--cutoff-meta-monitor-board-size",
        type=int,
        default=12,
        help="Board size for unified cutoff-meta monitor.",
    )
    parser.add_argument(
        "--cutoff-meta-monitor-live-veto-corr-score",
        type=float,
        default=1.25,
        help="Unified veto corr score for operational challenger monitor.",
    )
    parser.add_argument(
        "--cutoff-meta-monitor-research-veto-corr-score",
        type=float,
        default=999.0,
        help="Unified veto corr score for research comparator monitor.",
    )
    parser.add_argument(
        "--cutoff-meta-monitor-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for monitor artifacts. Defaults under the run folder.",
    )
    parser.add_argument(
        "--skip-cutoff-meta-monitor",
        action="store_true",
        help="Skip the daily unified cutoff-meta side-by-side monitor step.",
    )
    parser.add_argument("--history-csv", type=Path, default=REPO_ROOT / "model" / "analysis" / "latest_market_comparison_strict_rows.csv", help="Historical row-level backtest CSV for edge calibration.")
    parser.add_argument(
        "--history-fallback-rows-csv",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "date_hit_rate_validation_rows.csv",
        help="Fallback long-format validation rows used to rebuild history-csv when missing.",
    )
    parser.add_argument(
        "--disable-history-fallback",
        action="store_true",
        help="Disable automatic history-csv reconstruction from validation rows when history-csv is missing.",
    )
    parser.add_argument("--lookback-days", type=int, default=10, help="How many recent days of historical market lines to collect.")
    parser.add_argument("--future-days", type=int, default=2, help="How many days ahead of today to keep in the current slate snapshot.")
    parser.add_argument(
        "--snapshot-policy",
        type=str,
        default="auto",
        choices=["auto", "live_only"],
        help=(
            "Snapshot fallback policy. "
            "'auto' allows historical backfill and stale fallback when the requested window is empty. "
            "'live_only' disables both fallbacks and fails if no requested-window rows exist."
        ),
    )
    parser.add_argument("--collect-scan-count", type=int, default=800, help="Maximum Covers matchup ids to scan for the nightly collection window.")
    parser.add_argument("--run-date", type=str, default=None, help="Optional YYYY-MM-DD override for local run date.")
    parser.add_argument("--skip-update-data", action="store_true", help="Skip official game-log refresh.")
    parser.add_argument("--skip-collect-market", action="store_true", help="Skip market odds collection.")
    parser.add_argument(
        "--market-provider",
        type=str,
        default="rotowire",
        choices=["rotowire", "covers", "sportsgameodds"],
        help="Market snapshot source used when --skip-collect-market is not set.",
    )
    parser.add_argument(
        "--market-bookmakers",
        type=str,
        default="draftkings,fanduel",
        help="Comma-separated bookmakers for SportsGameOdds live snapshot filtering.",
    )
    parser.add_argument("--skip-align", action="store_true", help="Skip market alignment onto processed files.")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip refreshing the historical inference calibration CSV.")
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Allow market pipeline to run with heuristic-only predictions when model load fails.",
    )
    parser.add_argument("--skip-export-web", action="store_true", help="Skip exporting web daily predictions payload.")
    parser.add_argument(
        "--web-cards-json",
        type=Path,
        default=SPORT_ROOT / "web" / "data" / "cards.json",
        help="cards.json used during web export to enrich player display names/headshots.",
    )
    parser.add_argument("--skip-build-site", action="store_true", help="Skip static site rebuild step.")
    parser.add_argument(
        "--selected-board-calibrator-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "selected_board_calibrator.json",
        help="Selected-board calibrator payload JSON forwarded to run_market_pipeline.py.",
    )
    parser.add_argument(
        "--disable-selected-board-calibration",
        dest="disable_selected_board_calibration",
        action="store_true",
        default=False,
        help="Disable selected-board calibration in child market pipeline runs.",
    )
    parser.add_argument(
        "--enable-selected-board-calibration",
        dest="disable_selected_board_calibration",
        action="store_false",
        help="Enable selected-board calibration in child market pipeline runs.",
    )
    parser.add_argument(
        "--selected-board-calibration-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month override for selected-board calibration (defaults to run month).",
    )
    parser.add_argument(
        "--learned-gate-json",
        type=Path,
        default=REPO_ROOT / "model" / "analysis" / "calibration" / "learned_pool_gate.json",
        help="Learned pool-gate payload JSON forwarded to run_market_pipeline.py.",
    )
    parser.add_argument(
        "--disable-learned-gate",
        action="store_true",
        help="Disable learned pool-gate filtering in child market pipeline runs.",
    )
    parser.add_argument(
        "--learned-gate-month",
        type=str,
        default=None,
        help="Optional YYYY-MM month override for learned pool-gate thresholds (defaults to run month).",
    )
    parser.add_argument(
        "--disable-initial-pool-gate",
        action="store_true",
        help="Disable pre-board initial pool pruning in child market pipeline runs.",
    )
    parser.add_argument(
        "--initial-pool-gate-drop-fraction",
        type=float,
        default=None,
        help="Optional override for initial pool drop fraction in child market pipeline runs.",
    )
    parser.add_argument(
        "--initial-pool-gate-score-col",
        type=str,
        default=None,
        help="Optional override for ranking column used by initial pool pruning in child runs.",
    )
    parser.add_argument(
        "--initial-pool-gate-min-keep-rows",
        type=int,
        default=None,
        help="Optional override for minimum rows kept after initial pool pruning in child runs.",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use for child steps.")
    return parser.parse_args()


def infer_season(local_date: pd.Timestamp) -> int:
    return local_date.year + 1 if local_date.month >= 9 else local_date.year


def run_step(label: str, args: list[str]) -> None:
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)
    print("Command:", " ".join(args))
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def print_notice(label: str, message: str) -> None:
    print("\n" + "=" * 90)
    print(label)
    print("=" * 90)
    print(message)


def rebuild_history_csv_from_validation_rows(source_csv: Path, history_csv: Path) -> tuple[bool, str]:
    required = ["player", "market_date", "target", "prediction", "market_line", "actual"]
    if not source_csv.exists():
        return False, f"fallback source not found: {source_csv}"

    try:
        source_df = pd.read_csv(source_csv, usecols=lambda c: c in set(required))
    except Exception as exc:
        return False, f"failed reading fallback source {source_csv}: {exc}"

    missing_cols = [c for c in required if c not in source_df.columns]
    if missing_cols:
        return False, f"fallback source missing required columns: {missing_cols}"

    working = source_df.copy()
    working["target"] = working["target"].astype(str).str.upper().str.strip()
    working = working.loc[working["target"].isin(["PTS", "TRB", "AST"])].copy()
    if working.empty:
        return False, "fallback source has no PTS/TRB/AST rows"

    working["market_date"] = pd.to_datetime(working["market_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    working["prediction"] = pd.to_numeric(working["prediction"], errors="coerce")
    working["market_line"] = pd.to_numeric(working["market_line"], errors="coerce")
    working["actual"] = pd.to_numeric(working["actual"], errors="coerce")
    working["player"] = working["player"].astype(str).str.strip()
    working = working.loc[
        working["market_date"].notna()
        & (working["player"] != "")
        & working["prediction"].notna()
        & working["market_line"].notna()
    ].copy()
    if working.empty:
        return False, "fallback source has no usable rows after cleaning"

    working = (
        working.sort_values(["market_date", "player", "target"])
        .drop_duplicates(subset=["player", "market_date", "target"], keep="last")
        .reset_index(drop=True)
    )

    base = working[["player", "market_date"]].drop_duplicates().reset_index(drop=True)
    out = base.copy()
    for target in ["PTS", "TRB", "AST"]:
        part = (
            working.loc[working["target"] == target, ["player", "market_date", "prediction", "market_line", "actual"]]
            .rename(
                columns={
                    "prediction": f"pred_{target}",
                    "market_line": f"market_{target}",
                    "actual": f"actual_{target}",
                }
            )
            .copy()
        )
        out = out.merge(part, on=["player", "market_date"], how="left")

    out["did_not_play"] = 0
    out["minutes"] = 1.0

    market_cols = [f"market_{t}" for t in ["PTS", "TRB", "AST"]]
    out = out.loc[out[market_cols].notna().any(axis=1)].copy()
    if out.empty:
        return False, "reconstructed history had zero rows after market filters"

    history_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(history_csv, index=False)
    return True, f"reconstructed {len(out):,} rows from {source_csv}"


def _normalize_player_name(value: str) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        (".", ""),
        ("'", ""),
        (",", ""),
        ("/", "-"),
        ("\\", "-"),
        (":", ""),
    ]:
        out = out.replace(old, new)
    return out


def derive_market_snapshot_id(*, provider: str, fetched_at_utc: object, fallback_label: object = "") -> str:
    provider_text = str(provider or "").strip() or "unknown_provider"
    fetched_text = str(fetched_at_utc or "").strip()
    if fetched_text:
        return f"{provider_text}:{fetched_text}"
    fallback_text = str(fallback_label or "").strip()
    if fallback_text:
        return f"{provider_text}:{fallback_text}"
    return provider_text


def load_market_snapshot_manifest(snapshot_path: Path) -> dict:
    candidates = [
        snapshot_path.parent / "latest_manifest.json",
        snapshot_path.parent.parent / "latest_manifest.json",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def annotate_market_snapshot_provenance(frame: pd.DataFrame, source_manifest: dict | None = None) -> pd.DataFrame:
    out = frame.copy()
    manifest = dict(source_manifest or {})
    provider_default = str(manifest.get("provider", "snapshot")).strip() or "snapshot"
    source_default = str(manifest.get("input_path") or manifest.get("provider") or "current_market_snapshot").strip()
    if "Market_Provider" not in out.columns:
        out["Market_Provider"] = provider_default
    else:
        out["Market_Provider"] = out["Market_Provider"].fillna("").astype(str).replace("", provider_default)
    if "Market_Book" not in out.columns:
        out["Market_Book"] = "aggregate_market_snapshot"
    else:
        out["Market_Book"] = out["Market_Book"].fillna("").astype(str).replace("", "aggregate_market_snapshot")
    if "Market_Price_Source" not in out.columns:
        out["Market_Price_Source"] = source_default
    else:
        out["Market_Price_Source"] = out["Market_Price_Source"].fillna("").astype(str).replace("", source_default)
    fetched_series = out.get("Market_Fetched_At_UTC", pd.Series("", index=out.index)).fillna("").astype(str)
    source_type_default = fetched_series.str.strip().ne("").map(lambda has_time: "ARCHIVED_ENTRY" if has_time else "UNKNOWN")
    if "Market_Price_Source_Type" not in out.columns:
        out["Market_Price_Source_Type"] = source_type_default
    else:
        existing_source_type = out["Market_Price_Source_Type"].fillna("").astype(str)
        out["Market_Price_Source_Type"] = existing_source_type.where(existing_source_type.str.strip().ne(""), source_type_default)
    if "Market_Snapshot_ID" not in out.columns:
        out["Market_Snapshot_ID"] = ""
    provider_series = out["Market_Provider"].fillna(provider_default).astype(str)
    event_series = out.get("Market_Event_ID", pd.Series("", index=out.index)).fillna("").astype(str)
    snapshot_ids = [
        derive_market_snapshot_id(provider=provider, fetched_at_utc=fetched_at, fallback_label=event_id)
        for provider, fetched_at, event_id in zip(provider_series, fetched_series, event_series)
    ]
    existing_snapshot_ids = out["Market_Snapshot_ID"].fillna("").astype(str)
    out["Market_Snapshot_ID"] = existing_snapshot_ids.where(existing_snapshot_ids.str.strip().ne(""), snapshot_ids)
    return out


def persist_market_snapshot_manifest(
    *,
    source_snapshot_path: Path,
    output_snapshot_path: Path,
    run_stamp: str,
    snapshot_meta: dict,
) -> dict:
    source_manifest = load_market_snapshot_manifest(source_snapshot_path)
    payload = dict(source_manifest)
    payload.setdefault("provider", "snapshot")
    payload["source_snapshot_path"] = str(source_snapshot_path.resolve())
    payload["output_snapshot_path"] = str(output_snapshot_path.resolve())
    payload["snapshot_selection_meta"] = dict(snapshot_meta)
    latest_manifest_path = output_snapshot_path.parent / "latest_manifest.json"
    stamped_manifest_path = output_snapshot_path.parent / f"current_market_snapshot_manifest_{run_stamp}.json"
    required_manifest_path = output_snapshot_path.parent / f"current_market_snapshot_{run_stamp}_manifest.json"
    latest_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    stamped_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    required_manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "latest_manifest_path": str(latest_manifest_path),
        "stamped_manifest_path": str(stamped_manifest_path),
        "required_manifest_path": str(required_manifest_path),
        "provider": str(payload.get("provider", "snapshot")),
    }


def build_historical_market_snapshot(
    out_path: Path,
    run_date: pd.Timestamp,
    future_days: int,
    season: int,
) -> tuple[int, dict]:
    start_date = run_date.normalize()
    end_date = (run_date + pd.Timedelta(days=int(future_days))).normalize()
    use_cols = {
        "Date",
        "Player",
        "Market_Provider",
        "Market_Book",
        "Market_Price_Source",
        "Market_Price_Source_Type",
        "Market_Snapshot_ID",
        "Market_PTS",
        "Market_TRB",
        "Market_AST",
        "Market_PTS_books",
        "Market_TRB_books",
        "Market_AST_books",
        "Market_PTS_over_price",
        "Market_TRB_over_price",
        "Market_AST_over_price",
        "Market_PTS_under_price",
        "Market_TRB_under_price",
        "Market_AST_under_price",
        "Market_PTS_line_std",
        "Market_TRB_line_std",
        "Market_AST_line_std",
        "Market_Fetched_At_UTC",
    }
    rows: list[dict] = []
    for csv_path in sorted(DATA_PROC_ROOT.glob(f"*/*{season}_processed_processed.csv")):
        try:
            df = pd.read_csv(csv_path, usecols=lambda col: col in use_cols)
        except Exception:
            continue
        if df.empty or "Date" not in df.columns:
            continue

        date_series = pd.to_datetime(df["Date"], errors="coerce")
        mask = (date_series >= start_date) & (date_series <= end_date)
        if not mask.any():
            continue

        subset = df.loc[mask].copy()
        if subset.empty:
            continue

        for column in [
            "Market_PTS",
            "Market_TRB",
            "Market_AST",
            "Market_PTS_books",
            "Market_TRB_books",
            "Market_AST_books",
            "Market_PTS_over_price",
            "Market_TRB_over_price",
            "Market_AST_over_price",
            "Market_PTS_under_price",
            "Market_TRB_under_price",
            "Market_AST_under_price",
            "Market_PTS_line_std",
            "Market_TRB_line_std",
            "Market_AST_line_std",
        ]:
            if column in subset.columns:
                subset[column] = pd.to_numeric(subset[column], errors="coerce")

        line_cols = [column for column in ["Market_PTS", "Market_TRB", "Market_AST"] if column in subset.columns]
        if line_cols:
            subset = subset.loc[subset[line_cols].notna().any(axis=1)].copy()
        if subset.empty:
            continue

        player_series = subset["Player"] if "Player" in subset.columns else pd.Series(csv_path.parent.name, index=subset.index)
        player_series = player_series.fillna(csv_path.parent.name).astype(str)
        market_dates = pd.to_datetime(subset["Date"], errors="coerce")

        for idx in subset.index:
            market_date = market_dates.loc[idx]
            if pd.isna(market_date):
                continue
            player_raw = str(player_series.loc[idx]).strip()
            if not player_raw:
                player_raw = csv_path.parent.name.replace("_", " ")
            row = {
                "Player": _normalize_player_name(player_raw),
                "Market_Player_Raw": player_raw,
                "Market_Date": market_date,
            }
            for column in [
                "Market_Provider",
                "Market_Book",
                "Market_Price_Source",
                "Market_Price_Source_Type",
                "Market_Snapshot_ID",
                "Market_PTS",
                "Market_TRB",
                "Market_AST",
                "Market_PTS_books",
                "Market_TRB_books",
                "Market_AST_books",
                "Market_PTS_over_price",
                "Market_TRB_over_price",
                "Market_AST_over_price",
                "Market_PTS_under_price",
                "Market_TRB_under_price",
                "Market_AST_under_price",
                "Market_PTS_line_std",
                "Market_TRB_line_std",
                "Market_AST_line_std",
                "Market_Fetched_At_UTC",
            ]:
                row[column] = subset.loc[idx, column] if column in subset.columns else None
            rows.append(row)

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return 0, {
            "mode": "historical_backfill_empty",
            "requested_start_date": str(start_date.date()),
            "requested_end_date": str(end_date.date()),
            "selected_row_count": 0,
        }

    frame["Market_Date"] = pd.to_datetime(frame["Market_Date"], errors="coerce")
    frame = frame.loc[frame["Market_Date"].notna()].copy()
    frame = frame.sort_values(["Market_Date", "Player"]).drop_duplicates(subset=["Market_Date", "Player"], keep="last").reset_index(drop=True)
    if frame.empty:
        return 0, {
            "mode": "historical_backfill_empty",
            "requested_start_date": str(start_date.date()),
            "requested_end_date": str(end_date.date()),
            "selected_row_count": 0,
        }

    frame = annotate_market_snapshot_provenance(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        frame.to_parquet(out_path, index=False)
    else:
        frame.to_csv(out_path, index=False)

    return int(len(frame)), {
        "mode": "historical_backfill_window",
        "requested_start_date": str(start_date.date()),
        "requested_end_date": str(end_date.date()),
        "selected_market_date_min": str(pd.Timestamp(frame["Market_Date"].min()).date()),
        "selected_market_date_max": str(pd.Timestamp(frame["Market_Date"].max()).date()),
        "selected_row_count": int(len(frame)),
    }


def filter_current_market_snapshot(
    source_path: Path,
    out_path: Path,
    run_date: pd.Timestamp,
    future_days: int,
    season: int,
    allow_historical_backfill: bool = True,
    allow_stale_fallback: bool = True,
) -> tuple[int, dict]:
    if not source_path.exists():
        raise FileNotFoundError(f"Market snapshot not found: {source_path}")
    source_manifest = load_market_snapshot_manifest(source_path)
    if source_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        df = pd.read_csv(source_path)
    if df.empty:
        raise RuntimeError(f"Market snapshot is empty: {source_path}")
    if "Market_Date" not in df.columns:
        raise RuntimeError(f"Market snapshot is missing Market_Date: {source_path}")

    market_dates = pd.to_datetime(df["Market_Date"], errors="coerce")
    start_date = run_date.normalize()
    end_date = (run_date + pd.Timedelta(days=int(future_days))).normalize()
    filtered = df.loc[(market_dates >= start_date) & (market_dates <= end_date)].copy()
    if filtered.empty:
        if allow_historical_backfill:
            historical_rows, historical_meta = build_historical_market_snapshot(
                out_path=out_path,
                run_date=run_date,
                future_days=future_days,
                season=season,
            )
            if historical_rows > 0:
                return historical_rows, historical_meta

        if allow_stale_fallback:
            fallback_date = market_dates.max()
            if pd.isna(fallback_date):
                raise RuntimeError(f"No valid Market_Date values found in {source_path}")
            filtered = df.loc[market_dates == fallback_date].copy()
            if filtered.empty:
                raise RuntimeError(
                    f"No current/upcoming market rows found between {start_date.date()} and {end_date.date()} in {source_path}"
                )
            snapshot_meta = {
                "mode": "stale_fallback",
                "requested_start_date": str(start_date.date()),
                "requested_end_date": str(end_date.date()),
                "selected_market_date": str(pd.Timestamp(fallback_date).date()),
                "selected_row_count": int(len(filtered)),
            }
        else:
            raise RuntimeError(
                "Requested market window is empty and snapshot-policy disallows historical/stale fallback. "
                f"Window={start_date.date()}..{end_date.date()} source={source_path}"
            )
    else:
        snapshot_meta = {
            "mode": "requested_window",
            "requested_start_date": str(start_date.date()),
            "requested_end_date": str(end_date.date()),
            "selected_market_date_min": str(pd.Timestamp(filtered["Market_Date"].min()).date()),
            "selected_market_date_max": str(pd.Timestamp(filtered["Market_Date"].max()).date()),
            "selected_row_count": int(len(filtered)),
        }

    filtered = annotate_market_snapshot_provenance(filtered, source_manifest=source_manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".parquet":
        filtered.to_parquet(out_path, index=False)
    else:
        filtered.to_csv(out_path, index=False)
    return int(len(filtered)), snapshot_meta


def main() -> None:
    args = parse_args()
    local_date = resolve_run_timestamp(args.run_date)
    season = args.season or infer_season(local_date)
    selected_board_calibration_month = str(args.selected_board_calibration_month or local_date.strftime("%Y-%m"))
    selected_board_calibrator_path = args.selected_board_calibrator_json.resolve()
    selected_board_calibrator_meta = {
        "path": str(selected_board_calibrator_path),
        "exists": bool(selected_board_calibrator_path.exists()),
        "enabled": not bool(args.disable_selected_board_calibration),
        "calibration_month": selected_board_calibration_month,
    }
    learned_gate_month = str(args.learned_gate_month or local_date.strftime("%Y-%m"))
    learned_gate_path = args.learned_gate_json.resolve()
    learned_gate_meta = {
        "path": str(learned_gate_path),
        "exists": bool(learned_gate_path.exists()),
        "enabled": not bool(args.disable_learned_gate),
        "month": learned_gate_month,
    }
    initial_pool_gate_meta = {
        "enabled": not bool(args.disable_initial_pool_gate),
        "drop_fraction": float(args.initial_pool_gate_drop_fraction)
        if args.initial_pool_gate_drop_fraction is not None
        else None,
        "score_col": str(args.initial_pool_gate_score_col)
        if args.initial_pool_gate_score_col is not None
        else None,
        "min_keep_rows": int(args.initial_pool_gate_min_keep_rows)
        if args.initial_pool_gate_min_keep_rows is not None
        else None,
    }
    initial_pool_gate_args = [
        *(["--disable-initial-pool-gate"] if args.disable_initial_pool_gate else []),
        *(
            ["--initial-pool-gate-drop-fraction", str(float(args.initial_pool_gate_drop_fraction))]
            if args.initial_pool_gate_drop_fraction is not None
            else []
        ),
        *(
            ["--initial-pool-gate-score-col", str(args.initial_pool_gate_score_col)]
            if args.initial_pool_gate_score_col is not None
            else []
        ),
        *(
            ["--initial-pool-gate-min-keep-rows", str(int(args.initial_pool_gate_min_keep_rows))]
            if args.initial_pool_gate_min_keep_rows is not None
            else []
        ),
    ]
    primary_policy = str(args.policy_profile)
    shadow_policies = [profile for profile in args.shadow_policy_profiles if str(profile) and str(profile) != primary_policy]
    yesterday = (local_date - pd.Timedelta(days=1)).date()
    lookback_start = (local_date - pd.Timedelta(days=int(args.lookback_days))).date()
    future_end = (local_date + pd.Timedelta(days=int(args.future_days))).date()
    history_csv_path = args.history_csv.resolve()
    backtest_script_path = REPO_ROOT / "scripts" / "backtest_inference_accuracy.py"
    history_fallback_source_path = args.history_fallback_rows_csv.resolve()

    run_stamp = local_date.strftime("%Y%m%d")
    run_dir = ANALYSIS_ROOT / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_update_data:
        run_step(
            f"Update Official NBA Data Through {yesterday}",
            [
                args.python,
                "scripts/update_nba_processed_data.py",
                "--season",
                str(season),
                "--through-date",
                str(yesterday),
            ],
        )

    if not args.skip_collect_market and args.market_provider == "rotowire":
        run_step(
            "Scrape RotoWire Current NBA Props",
            [
                args.python,
                "scripts/fetch_nba_market_props.py",
                "--provider",
                "rotowire",
                "--event-date",
                str(local_date.date()),
                "--outdir",
                str(MARKET_ROOT),
            ],
        )
    elif not args.skip_collect_market and args.market_provider == "covers":
        run_step(
            f"Collect Covers Props {lookback_start} -> {future_end}",
            [
                args.python,
                "scripts/collect_covers_historical_props.py",
                "--date-from",
                str(lookback_start),
                "--date-to",
                str(future_end),
                "--scan-count",
                str(int(args.collect_scan_count)),
            ],
        )
    elif not args.skip_collect_market and args.market_provider == "sportsgameodds":
        run_step(
            "Fetch SportsGameOdds Current NBA Props",
            [
                args.python,
                "scripts/fetch_nba_market_props.py",
                "--provider",
                "sportsgameodds",
                "--bookmakers",
                str(args.market_bookmakers),
                "--outdir",
                str(MARKET_ROOT),
            ],
        )

    if not args.skip_align:
        run_step(
            f"Align Historical Market Lines For Season {season}",
            [
                args.python,
                "scripts/align_historical_market_lines.py",
                "--season",
                str(season),
                "--skip-market-anchor",
            ],
        )

    if not args.skip_backtest:
        if backtest_script_path.exists():
            run_step(
                f"Refresh Historical Inference Backtest For Season {season}",
                [
                    args.python,
                    "scripts/backtest_inference_accuracy.py",
                    "--season",
                    str(season),
                    "--strict-csv-out",
                    str(args.history_csv),
                    *(["--latest"] if args.latest else []),
                ],
            )
        else:
            if history_csv_path.exists():
                print_notice(
                    "Backtest Step Skipped (Script Missing)",
                    "scripts/backtest_inference_accuracy.py was not found. "
                    f"Continuing with existing history CSV: {history_csv_path}",
                )
            elif not args.disable_history_fallback:
                rebuilt, detail = rebuild_history_csv_from_validation_rows(
                    history_fallback_source_path,
                    history_csv_path,
                )
                if rebuilt:
                    print_notice(
                        "Backtest Step Skipped (Script Missing, Fallback Rebuilt)",
                        detail,
                    )
                else:
                    raise FileNotFoundError(
                        "Backtest step cannot run because scripts/backtest_inference_accuracy.py is missing, "
                        f"history CSV is missing ({history_csv_path}), and fallback reconstruction failed ({detail})."
                    )
            else:
                raise FileNotFoundError(
                    "Backtest step cannot run because scripts/backtest_inference_accuracy.py is missing and "
                    f"history CSV is missing: {history_csv_path}"
                )

    if not history_csv_path.exists():
        if not args.disable_history_fallback:
            rebuilt, detail = rebuild_history_csv_from_validation_rows(
                history_fallback_source_path,
                history_csv_path,
            )
            if rebuilt:
                print_notice("History CSV Rebuilt", detail)
            else:
                raise FileNotFoundError(
                    f"History CSV not found at {history_csv_path}, and fallback reconstruction failed ({detail})."
                )
        else:
            raise FileNotFoundError(
                f"History CSV not found at {history_csv_path}. "
                "Either provide --history-csv, enable backtest refresh, or remove --disable-history-fallback."
            )

    latest_market_path = MARKET_ROOT / "latest_player_props_wide.parquet"
    current_snapshot_path = run_dir / f"current_market_snapshot_{run_stamp}.parquet"
    allow_fallbacks = str(args.snapshot_policy).lower() != "live_only"
    current_rows, snapshot_meta = filter_current_market_snapshot(
        latest_market_path,
        current_snapshot_path,
        local_date,
        args.future_days,
        season,
        allow_historical_backfill=allow_fallbacks,
        allow_stale_fallback=allow_fallbacks,
    )
    snapshot_manifest_meta = persist_market_snapshot_manifest(
        source_snapshot_path=latest_market_path,
        output_snapshot_path=current_snapshot_path,
        run_stamp=run_stamp,
        snapshot_meta=snapshot_meta,
    )

    final_csv = run_dir / f"final_market_plays_{run_stamp}.csv"
    final_json = run_dir / f"final_market_plays_{run_stamp}.json"
    slate_csv = run_dir / f"upcoming_market_slate_{run_stamp}.csv"
    selector_csv = run_dir / f"upcoming_market_play_selector_{run_stamp}.csv"

    run_step(
        "Run Market Decision Pipeline",
        [
            args.python,
            "scripts/run_market_pipeline.py",
            "--season",
            str(season),
            "--policy-profile",
            primary_policy,
            "--history-csv",
            str(args.history_csv),
            "--market-wide-path",
            str(current_snapshot_path),
            "--slate-csv-out",
            str(slate_csv),
            "--selector-csv-out",
            str(selector_csv),
            "--final-csv-out",
            str(final_csv),
            "--final-json-out",
            str(final_json),
            "--selected-board-calibrator-json",
            str(selected_board_calibrator_path),
            "--selected-board-calibration-month",
            selected_board_calibration_month,
            "--learned-gate-json",
            str(learned_gate_path),
            "--learned-gate-month",
            learned_gate_month,
            *initial_pool_gate_args,
            *(["--disable-selected-board-calibration"] if args.disable_selected_board_calibration else []),
            *(["--disable-learned-gate"] if args.disable_learned_gate else []),
            *(["--allow-heuristic-fallback"] if args.allow_heuristic_fallback else []),
            *(["--latest"] if args.latest else []),
        ],
    )

    shadow_outputs: list[dict] = []
    for profile in shadow_policies:
        shadow_dir = run_dir / "shadow" / profile
        shadow_dir.mkdir(parents=True, exist_ok=True)
        shadow_final_csv = shadow_dir / f"final_market_plays_{run_stamp}_{profile}.csv"
        shadow_final_json = shadow_dir / f"final_market_plays_{run_stamp}_{profile}.json"
        shadow_slate_csv = shadow_dir / f"upcoming_market_slate_{run_stamp}_{profile}.csv"
        shadow_selector_csv = shadow_dir / f"upcoming_market_play_selector_{run_stamp}_{profile}.csv"
        run_step(
            f"Run Shadow Market Decision Pipeline [{profile}]",
            [
                args.python,
                "scripts/run_market_pipeline.py",
                "--season",
                str(season),
                "--policy-profile",
                str(profile),
                "--history-csv",
                str(args.history_csv),
                "--market-wide-path",
                str(current_snapshot_path),
                "--slate-csv-out",
                str(shadow_slate_csv),
                "--selector-csv-out",
                str(shadow_selector_csv),
                "--final-csv-out",
                str(shadow_final_csv),
                "--final-json-out",
                str(shadow_final_json),
                "--selected-board-calibrator-json",
                str(selected_board_calibrator_path),
                "--selected-board-calibration-month",
                selected_board_calibration_month,
                "--learned-gate-json",
                str(learned_gate_path),
                "--learned-gate-month",
                learned_gate_month,
                *initial_pool_gate_args,
                *(["--disable-selected-board-calibration"] if args.disable_selected_board_calibration else []),
                *(["--disable-learned-gate"] if args.disable_learned_gate else []),
                *(["--allow-heuristic-fallback"] if args.allow_heuristic_fallback else []),
                *(["--latest"] if args.latest else []),
            ],
        )
        shadow_outputs.append(
            {
                "policy_profile": str(profile),
                "slate_csv": str(shadow_slate_csv),
                "selector_csv": str(shadow_selector_csv),
                "final_csv": str(shadow_final_csv),
                "final_json": str(shadow_final_json),
            }
        )

    cutoff_meta_monitor_outputs: dict = {}
    if not args.skip_cutoff_meta_monitor:
        monitor_dir = (args.cutoff_meta_monitor_output_dir or (run_dir / "shadow" / "unified_cutoff_meta_monitor")).resolve()
        monitor_compare_json = monitor_dir / f"cutoff_meta_monitor_compare_{run_stamp}.json"
        monitor_compare_csv = monitor_dir / f"cutoff_meta_monitor_compare_{run_stamp}.csv"
        monitor_cmd = [
            args.python,
            "scripts/run_daily_cutoff_meta_monitor.py",
            "--run-date",
            str(local_date.date()),
            "--daily-runs-dir",
            str(ANALYSIS_ROOT),
            "--lookback-days",
            str(int(args.cutoff_meta_monitor_lookback_days)),
            "--board-size",
            str(int(args.cutoff_meta_monitor_board_size)),
            "--live-unified-veto-corr-score",
            str(float(args.cutoff_meta_monitor_live_veto_corr_score)),
            "--research-unified-veto-corr-score",
            str(float(args.cutoff_meta_monitor_research_veto_corr_score)),
            "--output-dir",
            str(monitor_dir),
            "--comparison-json-out",
            str(monitor_compare_json),
            "--comparison-csv-out",
            str(monitor_compare_csv),
            "--python",
            str(args.python),
        ]
        if args.cutoff_meta_monitor_start_date:
            monitor_cmd.extend(["--start-date", str(args.cutoff_meta_monitor_start_date)])
        if args.cutoff_meta_monitor_end_date:
            monitor_cmd.extend(["--end-date", str(args.cutoff_meta_monitor_end_date)])

        run_step(
            "Run Unified Cutoff Meta Daily Monitor (Live vs Research Corr Veto)",
            monitor_cmd,
        )

        cutoff_meta_monitor_outputs = {
            "enabled": True,
            "run_date": str(local_date.date()),
            "lookback_days": int(args.cutoff_meta_monitor_lookback_days),
            "start_date_override": str(args.cutoff_meta_monitor_start_date) if args.cutoff_meta_monitor_start_date else None,
            "end_date_override": str(args.cutoff_meta_monitor_end_date) if args.cutoff_meta_monitor_end_date else None,
            "board_size": int(args.cutoff_meta_monitor_board_size),
            "live_veto_corr_score": float(args.cutoff_meta_monitor_live_veto_corr_score),
            "research_veto_corr_score": float(args.cutoff_meta_monitor_research_veto_corr_score),
            "output_dir": str(monitor_dir),
            "comparison_json": str(monitor_compare_json),
            "comparison_csv": str(monitor_compare_csv),
        }
    else:
        cutoff_meta_monitor_outputs = {
            "enabled": False,
        }

    manifest = {
        "run_date": str(local_date.date()),
        "season": int(season),
        "through_date": str(yesterday),
        "lookback_start": str(lookback_start),
        "future_end": str(future_end),
        "snapshot_policy": str(args.snapshot_policy),
        "history_csv": str(args.history_csv),
        "current_market_snapshot": str(current_snapshot_path),
        "current_market_rows": int(current_rows),
        "current_market_snapshot_meta": snapshot_meta,
        "current_market_snapshot_manifest": snapshot_manifest_meta,
        "final_csv": str(final_csv),
        "final_json": str(final_json),
        "slate_csv": str(slate_csv),
        "selector_csv": str(selector_csv),
        "policy_profile": primary_policy,
        "shadow_policy_profiles": shadow_policies,
        "shadow_runs": shadow_outputs,
        "cutoff_meta_monitor": cutoff_meta_monitor_outputs,
        "used_latest_manifest": bool(args.latest),
        "skip_update_data": bool(args.skip_update_data),
        "skip_collect_market": bool(args.skip_collect_market),
        "skip_align": bool(args.skip_align),
        "skip_backtest": bool(args.skip_backtest),
        "skip_cutoff_meta_monitor": bool(args.skip_cutoff_meta_monitor),
        "skip_export_web": bool(args.skip_export_web),
        "skip_build_site": bool(args.skip_build_site),
        "allow_heuristic_fallback": bool(args.allow_heuristic_fallback),
        "selected_board_calibrator": selected_board_calibrator_meta,
        "learned_pool_gate": learned_gate_meta,
        "initial_pool_gate": initial_pool_gate_meta,
        "updated_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    manifest_path = run_dir / f"daily_market_pipeline_manifest_{run_stamp}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_export_web:
        run_step(
            "Export Static Daily Predictions Page Data",
            [
                args.python,
                "scripts/export_daily_predictions_web.py",
                "--manifest",
                str(manifest_path),
                "--out-json",
                str(SPORT_ROOT / "web" / "data" / "daily_predictions.json"),
                "--out-dist",
                str(WORKSPACE_ROOT / "dist" / "nba" / "data" / "daily_predictions.json"),
                "--cards-json",
                str(args.web_cards_json),
            ],
        )
    else:
        print(
            "[warning] Web export skipped; web/dist daily_predictions.json may be stale "
            "relative to this run's final board."
        )
    if not args.skip_build_site:
        run_step(
            "Rebuild Static Site Bundle",
            [
                args.python,
                str(UNIFIED_SITE_PIPELINE_ROOT / "build_static_site.py"),
                "--output",
                str(WORKSPACE_ROOT / "dist"),
            ],
        )
    else:
        print(
            "[warning] Static site rebuild skipped; dist assets may not reflect the latest web payload."
        )

    print("\n" + "=" * 90)
    print("DAILY MARKET PIPELINE COMPLETE")
    print("=" * 90)
    print(f"Run date:             {local_date.date()}")
    print(f"Season:               {season}")
    print(f"Current market rows:  {current_rows}")
    print(f"Snapshot mode:        {snapshot_meta['mode']}")
    print(f"Primary policy:       {primary_policy}")
    if shadow_policies:
        print(f"Shadow policies:      {', '.join(shadow_policies)}")
    if cutoff_meta_monitor_outputs.get("enabled"):
        print(f"Cutoff monitor:       {cutoff_meta_monitor_outputs.get('comparison_json')}")
    print(f"Selected calibrator:  {selected_board_calibrator_meta}")
    print(f"Learned pool gate:    {learned_gate_meta}")
    print(f"Initial pool gate:    {initial_pool_gate_meta}")
    if snapshot_meta["mode"] == "stale_fallback":
        print(f"Selected market date: {snapshot_meta['selected_market_date']}")
    print(f"Run directory:        {run_dir}")
    print(f"Final board:          {final_csv}")
    print(f"Manifest:             {manifest_path}")


if __name__ == "__main__":
    main()
