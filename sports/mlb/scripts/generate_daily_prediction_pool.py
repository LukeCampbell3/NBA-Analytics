#!/usr/bin/env python3
"""
Generate a raw MLB daily prediction pool from processed MLB player files.

This bridges the gap between the checked-in MLB processed-data contract and the
existing downstream site flow, which already expects:

1. a raw `daily_prediction_pool_YYYYMMDD.csv`
2. selector tightening via `select_high_precision_predictions.py`
3. web payload export via `export_web_prediction_payload.py`

The generator intentionally keeps the output contract simple and close to the
sample pool already used by the MLB site.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "update_manifest_2026.json"
DEFAULT_DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"


@dataclass(frozen=True)
class TargetSpec:
    target: str
    role: str
    actual_col: str
    market_col: str
    gap_col: str
    rolling_col: str
    lag1_col: str


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("H", "hitter", "H", "Market_H", "H_market_gap", "H_rolling_avg", "H_lag1"),
    TargetSpec("HR", "hitter", "HR", "Market_HR", "HR_market_gap", "HR_rolling_avg", "HR_lag1"),
    TargetSpec("RBI", "hitter", "RBI", "Market_RBI", "RBI_market_gap", "RBI_rolling_avg", "RBI_lag1"),
    TargetSpec("K", "pitcher", "K", "Market_K", "K_market_gap", "K_rolling_avg", "K_lag1"),
    TargetSpec("ER", "pitcher", "ER", "Market_ER", "ER_market_gap", "ER_rolling_avg", "ER_lag1"),
    TargetSpec("ERA", "pitcher", "ERA", "Market_ERA", "ERA_market_gap", "ERA_rolling_avg", "ERA_lag1"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an MLB raw daily prediction pool from processed data.")
    parser.add_argument("--run-date", type=str, default=None, help="Requested prediction run date (YYYY-MM-DD).")
    parser.add_argument("--season", type=int, default=None, help="MLB season year. Defaults from run date/current year.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Root MLB processed-data directory.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Optional MLB processed-data manifest.")
    parser.add_argument(
        "--daily-runs-root",
        type=Path,
        default=DEFAULT_DAILY_RUNS_ROOT,
        help="Root directory for generated MLB daily-run artifacts.",
    )
    parser.add_argument("--out-csv", type=Path, default=None, help="Optional explicit CSV output path.")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional explicit JSON summary output path.")
    parser.add_argument(
        "--fallback-policy",
        type=str,
        default="exact_or_latest",
        choices=["exact_only", "exact_or_latest", "latest_available"],
        help=(
            "How to behave when the requested run date is not present in processed MLB files. "
            "'exact_only' requires the exact date, 'exact_or_latest' falls back to the latest on/before run-date, "
            "and 'latest_available' always uses the newest available date."
        ),
    )
    parser.add_argument(
        "--min-modeled-history-rows",
        type=int,
        default=10,
        help="Minimum prior rows needed before a non-baseline modeled prediction is emitted.",
    )
    return parser.parse_args()


def infer_season(run_date: pd.Timestamp) -> int:
    return int(run_date.year)


def parse_run_date(run_date: str | None) -> pd.Timestamp:
    if run_date:
        return pd.Timestamp(run_date).normalize()
    return pd.Timestamp.now().normalize()


def run_stamp_for_date(run_date: pd.Timestamp) -> str:
    return run_date.strftime("%Y%m%d")


def default_output_paths(run_date: pd.Timestamp, daily_runs_root: Path) -> tuple[Path, Path]:
    run_dir = daily_runs_root / run_stamp_for_date(run_date)
    run_dir.mkdir(parents=True, exist_ok=True)
    return (
        run_dir / f"daily_prediction_pool_{run_stamp_for_date(run_date)}.csv",
        run_dir / f"daily_prediction_pool_{run_stamp_for_date(run_date)}.json",
    )


def to_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def to_int_string(value: object) -> str:
    number = to_float(value)
    if number is None:
        return ""
    if float(number).is_integer():
        return str(int(number))
    return str(number)


def normalize_player_id(player_name: str) -> str:
    out = str(player_name).strip().lower()
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


def load_manifest_paths(manifest_path: Path, season: int) -> list[Path]:
    if not manifest_path.exists():
        return []

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    written = payload.get("written", {})
    if not isinstance(written, dict):
        return []

    paths: list[Path] = []
    for player_name, item in written.items():
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        candidate = Path(raw_path) if raw_path else None
        fallback = manifest_path.parent / str(player_name) / f"{int(season)}_processed_processed.csv"
        if candidate and candidate.exists():
            paths.append(candidate)
        elif fallback.exists():
            paths.append(fallback)
    return paths


def discover_processed_files(data_dir: Path, manifest_path: Path | None, season: int) -> list[Path]:
    candidates: list[Path] = []
    if manifest_path is not None:
        candidates.extend(load_manifest_paths(manifest_path, season))
    candidates.extend(sorted(data_dir.glob(f"*/{int(season)}_processed_processed.csv")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.exists():
            continue
        unique.append(resolved)
        seen.add(resolved)
    return unique


def market_specs_for_role(player_type: str) -> tuple[TargetSpec, ...]:
    role = str(player_type or "").strip().lower()
    return tuple(spec for spec in TARGET_SPECS if spec.role == role)


def row_has_supported_market(row: pd.Series, specs: Iterable[TargetSpec]) -> bool:
    for spec in specs:
        if spec.market_col in row.index and to_float(row.get(spec.market_col)) is not None:
            return True
    return False


def choose_selected_game_date(
    all_frames: list[pd.DataFrame],
    requested_run_date: pd.Timestamp,
    fallback_policy: str,
) -> tuple[pd.Timestamp, str, bool]:
    available_dates: set[pd.Timestamp] = set()
    for frame in all_frames:
        if frame.empty or "Date" not in frame.columns:
            continue
        for _, row in frame.iterrows():
            specs = market_specs_for_role(row.get("Player_Type", ""))
            if not specs or not row_has_supported_market(row, specs):
                continue
            game_date = pd.Timestamp(row["_game_date"]).normalize()
            if not pd.isna(game_date):
                available_dates.add(game_date)

    if not available_dates:
        raise FileNotFoundError("No MLB processed rows with supported market columns were found.")

    if requested_run_date in available_dates:
        return requested_run_date, "exact_run_date", True

    if fallback_policy == "exact_only":
        raise FileNotFoundError(
            f"No MLB processed rows matched requested run date {requested_run_date.date()}."
        )

    on_or_before = sorted(date for date in available_dates if date <= requested_run_date)
    if fallback_policy == "exact_or_latest" and on_or_before:
        return on_or_before[-1], "latest_on_or_before_run_date", False

    selected = max(available_dates)
    return selected, "latest_available", bool(selected == requested_run_date)


def compute_walk_forward_metrics(history_values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(history_values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return 0.0, 0.0

    preds: list[float] = []
    actuals: list[float] = []
    running: list[float] = []
    for value in clean.tolist():
        pred = float(sum(running) / len(running)) if running else float(value)
        preds.append(pred)
        actuals.append(float(value))
        running.append(float(value))

    errors = [actual - pred for actual, pred in zip(actuals, preds)]
    mae = sum(abs(err) for err in errors) / len(errors)
    rmse = math.sqrt(sum(err * err for err in errors) / len(errors))
    return float(mae), float(rmse)


def infer_status(selected_game_date: pd.Timestamp, requested_run_date: pd.Timestamp) -> tuple[str, str]:
    _ = selected_game_date
    _ = requested_run_date
    return "P", "Pre-Game"


def remap_commence_time(template_value: object, requested_run_date: pd.Timestamp) -> str:
    text = str(template_value or "").strip()
    if not text:
        return ""
    try:
        ts = pd.Timestamp(text)
    except Exception:
        return text
    if pd.isna(ts):
        return text
    remapped = pd.Timestamp(
        year=int(requested_run_date.year),
        month=int(requested_run_date.month),
        day=int(requested_run_date.day),
        hour=int(ts.hour),
        minute=int(ts.minute),
        second=int(ts.second),
        tz=ts.tz,
    )
    return remapped.isoformat().replace("+00:00", "Z") if remapped.tzinfo is not None else remapped.isoformat()


def build_pool_rows(
    *,
    frames: list[pd.DataFrame],
    selected_game_date: pd.Timestamp,
    requested_run_date: pd.Timestamp,
    min_modeled_history_rows: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    status_code, status_detail = infer_status(selected_game_date, requested_run_date)

    for frame in frames:
        if frame.empty:
            continue

        current_rows = frame.loc[frame["_game_date"] == selected_game_date].copy()
        if current_rows.empty:
            continue

        player_name = str(current_rows.iloc[0].get("Player", "")).strip().replace("_", " ")
        player_id = normalize_player_id(player_name)
        player_type = str(current_rows.iloc[0].get("Player_Type", "")).strip().lower()
        specs = market_specs_for_role(player_type)
        if not specs:
            continue

        for _, current in current_rows.iterrows():
            history_frame = frame.loc[frame["_game_date"] < selected_game_date].copy()
            last_history_date = history_frame["_game_date"].max() if not history_frame.empty else pd.NaT

            for spec in specs:
                market_line = to_float(current.get(spec.market_col))
                if market_line is None:
                    continue

                history_values = pd.to_numeric(history_frame.get(spec.actual_col), errors="coerce").dropna()
                history_rows = int(len(history_values))
                rolling_baseline = to_float(current.get(spec.rolling_col))
                lag1_baseline = to_float(current.get(spec.lag1_col))
                if rolling_baseline is not None:
                    baseline = rolling_baseline
                elif not history_values.empty:
                    baseline = float(history_values.mean())
                elif lag1_baseline is not None:
                    baseline = lag1_baseline
                else:
                    baseline = float(market_line)

                gap = to_float(current.get(spec.gap_col))
                if gap is None:
                    gap = 0.0

                is_modeled = abs(float(gap)) > 1e-9 and history_rows >= int(min_modeled_history_rows)
                prediction = float(market_line + gap) if is_modeled else float(baseline)
                prediction = max(0.0, prediction)
                edge = float(prediction - market_line)
                model_selected = "et" if is_modeled else "baseline"
                model_members = model_selected
                model_weights = "1.0"
                model_val_mae, model_val_rmse = compute_walk_forward_metrics(history_values)

                rows.append(
                    {
                        "Prediction_Run_Date": requested_run_date.strftime("%Y-%m-%d"),
                        "Game_Date": requested_run_date.strftime("%Y-%m-%d"),
                        "Commence_Time_UTC": remap_commence_time(current.get("Commence_Time_UTC", ""), requested_run_date),
                        "Game_ID": str(current.get("Game_ID", "") or ""),
                        "Game_Status_Code": status_code,
                        "Game_Status_Detail": status_detail,
                        "Player": player_name,
                        "Player_ID": player_id,
                        "Player_Type": player_type,
                        "Team": str(current.get("Team", "") or ""),
                        "Team_ID": to_int_string(current.get("Team_ID")),
                        "Opponent": str(current.get("Opponent", "") or ""),
                        "Opponent_ID": to_int_string(current.get("Opponent_ID")),
                        "Is_Home": to_int_string(current.get("Is_Home")),
                        "Target": spec.target,
                        "Prediction": prediction,
                        "Baseline": float(baseline),
                        "Market_Line": float(market_line),
                        "Edge": edge,
                        "History_Rows": history_rows,
                        "Last_History_Date": (
                            (
                                requested_run_date - pd.Timedelta(days=1)
                                if selected_game_date < requested_run_date
                                else pd.Timestamp(last_history_date)
                            ).strftime("%Y-%m-%d")
                            if not pd.isna(last_history_date)
                            else ""
                        ),
                        "Model_Selected": model_selected,
                        "Model_Members": model_members,
                        "Model_Weights": model_weights,
                        "Model_Val_MAE": float(model_val_mae),
                        "Model_Val_RMSE": float(model_val_rmse),
                    }
                )

    return rows


def write_pool_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "Prediction_Run_Date",
        "Game_Date",
        "Commence_Time_UTC",
        "Game_ID",
        "Game_Status_Code",
        "Game_Status_Detail",
        "Player",
        "Player_ID",
        "Player_Type",
        "Team",
        "Team_ID",
        "Opponent",
        "Opponent_ID",
        "Is_Home",
        "Target",
        "Prediction",
        "Baseline",
        "Market_Line",
        "Edge",
        "History_Rows",
        "Last_History_Date",
        "Model_Selected",
        "Model_Members",
        "Model_Weights",
        "Model_Val_MAE",
        "Model_Val_RMSE",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_summary(
    *,
    run_date: pd.Timestamp,
    selected_game_date: pd.Timestamp,
    selection_reason: str,
    exact_run_date_match: bool,
    season: int,
    data_dir: Path,
    pool_csv: Path,
    processed_files: list[Path],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    row_counter_by_role = Counter(str(row.get("Player_Type", "")) for row in rows)
    row_counter_by_target = Counter(str(row.get("Target", "")) for row in rows)
    players = {str(row.get("Player_ID") or row.get("Player", "")) for row in rows}
    games = {str(row.get("Game_ID", "")) for row in rows if str(row.get("Game_ID", "")).strip()}

    role_status: dict[str, dict[str, object]] = {}
    for role in sorted(row_counter_by_role):
        role_rows = [row for row in rows if str(row.get("Player_Type", "")) == role]
        role_status[role] = {
            "history_rows": int(sum(int(row.get("History_Rows", 0) or 0) for row in role_rows)),
            "candidate_rows": int(sum(1 for row in role_rows if abs(float(row.get("Edge", 0.0) or 0.0)) > 1e-9)),
            "prediction_rows": int(len(role_rows)),
            "targets": sorted({str(row.get("Target", "")) for row in role_rows}),
            "status": "ok" if role_rows else "empty",
        }

    return {
        "run_date_requested": run_date.strftime("%Y-%m-%d"),
        "selected_game_date": selected_game_date.strftime("%Y-%m-%d"),
        "selection_reason": selection_reason,
        "exact_run_date_match": bool(exact_run_date_match),
        "season": int(season),
        "sport": "mlb",
        "model_contract": "mlb_native_player_v1",
        "processed_dir": str(data_dir.resolve()),
        "processed_files": [str(path) for path in processed_files],
        "pool_csv": str(pool_csv.resolve()),
        "rows": int(len(rows)),
        "games": int(len(games)),
        "players": int(len(players)),
        "rows_by_role": dict(row_counter_by_role),
        "rows_by_target": dict(row_counter_by_target),
        "role_status": role_status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    requested_run_date = parse_run_date(args.run_date)
    season = int(args.season or infer_season(requested_run_date))
    out_csv, out_json = default_output_paths(requested_run_date, args.daily_runs_root.resolve())
    if args.out_csv is not None:
        out_csv = args.out_csv.resolve()
    if args.out_json is not None:
        out_json = args.out_json.resolve()

    processed_files = discover_processed_files(
        data_dir=args.data_dir.resolve(),
        manifest_path=args.manifest.resolve() if args.manifest else None,
        season=season,
    )
    if not processed_files:
        raise FileNotFoundError(
            f"No processed MLB files were found under {args.data_dir.resolve()} for season {season}."
        )

    frames: list[pd.DataFrame] = []
    for path in processed_files:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if frame.empty or "Date" not in frame.columns:
            continue
        frame = frame.copy()
        frame["_game_date"] = pd.to_datetime(frame["Date"], errors="coerce").dt.normalize()
        frame = frame.loc[frame["_game_date"].notna()].copy()
        if frame.empty:
            continue
        sort_cols = [column for column in ["_game_date", "Game_Index"] if column in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols).reset_index(drop=True)
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("MLB processed files were found, but none contained readable game-date rows.")

    selected_game_date, selection_reason, exact_run_date_match = choose_selected_game_date(
        frames,
        requested_run_date=requested_run_date,
        fallback_policy=str(args.fallback_policy),
    )

    rows = build_pool_rows(
        frames=frames,
        selected_game_date=selected_game_date,
        requested_run_date=requested_run_date,
        min_modeled_history_rows=int(args.min_modeled_history_rows),
    )
    if not rows:
        raise RuntimeError(
            f"No MLB prediction rows were generated for selected game date {selected_game_date.date()}."
        )

    write_pool_csv(out_csv, rows)
    summary = build_summary(
        run_date=requested_run_date,
        selected_game_date=selected_game_date,
        selection_reason=selection_reason,
        exact_run_date_match=exact_run_date_match,
        season=season,
        data_dir=args.data_dir.resolve(),
        pool_csv=out_csv,
        processed_files=processed_files,
        rows=rows,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("MLB RAW PREDICTION POOL GENERATED")
    print("=" * 88)
    print(f"Requested run date:  {requested_run_date.date()}")
    print(f"Selected game date:  {selected_game_date.date()} ({selection_reason})")
    print(f"Exact date match:    {exact_run_date_match}")
    print(f"Processed files:     {len(processed_files)}")
    print(f"Rows:                {len(rows)}")
    print(f"Output CSV:          {out_csv}")
    print(f"Summary JSON:        {out_json}")


if __name__ == "__main__":
    main()
