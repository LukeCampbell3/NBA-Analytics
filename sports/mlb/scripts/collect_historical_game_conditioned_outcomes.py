#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

if str(Path(__file__).resolve().parents[3]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = REPO_ROOT / "Player-Predictor" / "Data-Proc-MLB"
DEFAULT_LEDGER = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "backtests" / "game_conditioned_historical_outcomes_2026.jsonl.gz"
DEFAULT_SUMMARY = REPO_ROOT / "artifacts" / "mlb_game_conditioned_historical_outcomes.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "mlb_game_conditioned_historical_outcomes.md"

SCHEMA_VERSION = "mlb_game_conditioned_historical_outcomes_v1"
EVIDENCE_CLASS = "HISTORICAL_REALIZED_OUTCOME_LABEL_NOT_PREGAME_FEATURE_EVIDENCE"
TARGET_LINES = {"H": 0.5, "TB": 1.5, "HR": 0.5}


class OutcomeConflictError(RuntimeError):
    pass


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _integer_count(value: Any) -> int | None:
    parsed = _finite(value)
    if parsed is None or parsed < 0:
        return None
    rounded = int(round(parsed))
    if abs(parsed - rounded) > 1e-6:
        return None
    return rounded


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def outcome_key(row: dict[str, Any]) -> str:
    return f"{row['season']}|{row['game_id']}|{row['player_id']}"


def _validate_counts(*, hits: int, total_bases: int, home_runs: int, at_bats: int, plate_appearances: int) -> list[str]:
    issues: list[str] = []
    if hits > at_bats:
        issues.append("H_GT_AB")
    if at_bats > plate_appearances:
        issues.append("AB_GT_PA")
    if home_runs > hits:
        issues.append("HR_GT_H")
    if total_bases < hits:
        issues.append("TB_LT_H")
    if total_bases < 4 * home_runs:
        issues.append("TB_LT_4X_HR")
    return issues


def build_outcome_record(raw: dict[str, Any], *, source_file: str, source_row: int, season: int) -> tuple[dict[str, Any] | None, str | None]:
    if str(raw.get("Player_Type") or "").strip().lower() != "hitter":
        return None, "NON_HITTER"
    if _truthy(raw.get("Did_Not_Play")):
        return None, "DID_NOT_PLAY"

    game_id = str(raw.get("Game_ID") or "").strip()
    player_id_value = _integer_count(raw.get("Player_MLBAM_ID"))
    date = str(raw.get("Date") or "").strip()[:10]
    if not game_id or not player_id_value or len(date) != 10:
        return None, "IDENTITY_MISSING"

    hits = _integer_count(raw.get("H"))
    total_bases = _integer_count(raw.get("TB"))
    home_runs = _integer_count(raw.get("HR"))
    plate_appearances = _integer_count(raw.get("PA"))
    at_bats = _integer_count(raw.get("AB"))
    if None in {hits, total_bases, home_runs, plate_appearances, at_bats}:
        return None, "REALIZED_STAT_MISSING_OR_INVALID"
    assert hits is not None and total_bases is not None and home_runs is not None
    assert plate_appearances is not None and at_bats is not None
    if plate_appearances <= 0:
        return None, "ZERO_PA"

    issues = _validate_counts(
        hits=hits,
        total_bases=total_bases,
        home_runs=home_runs,
        at_bats=at_bats,
        plate_appearances=plate_appearances,
    )
    if issues:
        return None, "+".join(sorted(issues))

    team = str(raw.get("Team") or "").strip()
    opponent = str(raw.get("Opponent") or "").strip()
    player_name = str(raw.get("Player") or "").replace("_", " ").strip()
    realized = {
        "H": hits,
        "TB": total_bases,
        "HR": home_runs,
        "PA": plate_appearances,
        "AB": at_bats,
    }
    outcomes = {
        "H_OVER_0_5": int(hits >= 1),
        "TB_OVER_1_5": int(total_bases >= 2),
        "HR_OVER_0_5": int(home_runs >= 1),
    }
    identity_and_result = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "season": int(season),
        "date": date,
        "game_id": game_id,
        "player_id": int(player_id_value),
        "player": player_name,
        "team": team,
        "opponent": opponent,
        "realized": realized,
        "outcomes": outcomes,
    }
    record = {
        **identity_and_result,
        "source": {
            "kind": "processed_historical_game_log",
            "file": source_file,
            "row": int(source_row),
        },
        "pregame_features_included": False,
        "market_data_included": False,
        "settlement_timestamp_available": False,
        "certification_use": "OUTCOME_LABEL_ONLY",
        "outcome_sha256": _canonical_hash(identity_and_result),
    }
    return record, None


def collect_outcomes(data_root: Path, *, season: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    indexed: dict[str, dict[str, Any]] = {}
    skipped: dict[str, int] = {}
    files = sorted(data_root.glob(f"*/{season}_processed_processed.csv"))
    for path in files:
        relative = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for source_row, raw in enumerate(reader, start=2):
                    record, reason = build_outcome_record(raw, source_file=relative, source_row=source_row, season=season)
                    if record is None:
                        skipped[reason or "UNKNOWN"] = skipped.get(reason or "UNKNOWN", 0) + 1
                        continue
                    key = outcome_key(record)
                    previous = indexed.get(key)
                    if previous is None:
                        indexed[key] = record
                        continue
                    if previous["outcome_sha256"] != record["outcome_sha256"]:
                        raise OutcomeConflictError(f"conflicting realized outcome for {key}: {previous['source']} vs {record['source']}")
                    skipped["EXACT_DUPLICATE"] = skipped.get("EXACT_DUPLICATE", 0) + 1
        except UnicodeDecodeError:
            skipped["FILE_DECODE_ERROR"] = skipped.get("FILE_DECODE_ERROR", 0) + 1
    rows = sorted(indexed.values(), key=lambda row: (row["date"], row["game_id"], row["player_id"]))
    return rows, skipped


def write_ledger(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    kwargs = {"encoding": "utf-8", "newline": ""}
    with opener(path, "wt", **kwargs) as handle:  # type: ignore[arg-type]
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def summarize(rows: list[dict[str, Any]], skipped: dict[str, int], *, season: int) -> dict[str, Any]:
    player_ids = {int(row["player_id"]) for row in rows}
    game_ids = {str(row["game_id"]) for row in rows}
    dates = {str(row["date"]) for row in rows}
    target_summary: dict[str, Any] = {}
    for target, key in (("H", "H_OVER_0_5"), ("TB", "TB_OVER_1_5"), ("HR", "HR_OVER_0_5")):
        wins = sum(int(row["outcomes"][key]) for row in rows)
        target_summary[target] = {
            "line": TARGET_LINES[target],
            "rows": len(rows),
            "clears": wins,
            "observed_clear_rate": wins / len(rows) if rows else None,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": EVIDENCE_CLASS,
        "season": int(season),
        "hitter_games": len(rows),
        "unique_players": len(player_ids),
        "unique_games": len(game_ids),
        "unique_dates": len(dates),
        "first_date": min(dates) if dates else None,
        "last_date": max(dates) if dates else None,
        "targets": target_summary,
        "skipped": dict(sorted(skipped.items())),
        "contract": {
            "pregame_features_included": False,
            "market_data_included": False,
            "outcome_fields": ["H", "TB", "HR", "PA", "AB"],
            "join_key": ["season", "game_id", "player_id"],
            "allowed_use": "historical outcome labels for retrospective validation and later settlement joins",
            "not_sufficient_for": "point-in-time feature certification or exact decision-time price certification",
        },
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# MLB Game-Conditioned Historical Outcome Collection",
        "",
        f"Evidence: `{summary['evidence_class']}`",
        "",
        f"Season: **{summary['season']}**",
        "",
        f"Collected hitter-games: **{summary['hitter_games']:,}** across **{summary['unique_players']:,}** hitters, **{summary['unique_games']:,}** games, and **{summary['unique_dates']:,}** dates.",
        "",
        "| Target | Threshold | Rows | Clears | Observed clear rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for target in ("H", "TB", "HR"):
        item = summary["targets"][target]
        rate = "n/a" if item["observed_clear_rate"] is None else f"{100.0 * float(item['observed_clear_rate']):.2f}%"
        lines.append(f"| {target} | O {item['line']} | {item['rows']:,} | {item['clears']:,} | {rate} |")
    lines += [
        "",
        "The ledger contains only realized H/TB/HR/PA/AB outcomes plus identity/provenance. Projection, market, matchup, rolling-form, and model fields are intentionally excluded so historical settlement can be joined after pregame feature construction.",
        "",
        "Historical labels are valid retrospective outcomes, but they do not by themselves prove that reconstructed historical features were available exactly as represented before first pitch.",
        "",
        "## Skipped rows",
        "",
    ]
    if summary["skipped"]:
        for reason, count in summary["skipped"].items():
            lines.append(f"- `{reason}`: {count:,}")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--output-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, skipped = collect_outcomes(args.data_root, season=args.season)
    if not rows:
        raise SystemExit("no historical hitter outcomes collected")
    write_ledger(rows, args.output_ledger)
    summary = summarize(rows, skipped, season=args.season)
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
