from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .allocation_path import (
    PathBuildResult,
    SettlementResult,
    attach_realized_allocations,
    build_allocation_paths,
)
from .protocol import ALLOCATION_PATH_PROTOCOL, AllocationPathProtocol


MARKET_ALIASES = {
    "pts": "player_points",
    "points": "player_points",
    "player_points": "player_points",
    "player_points_over_under": "player_points",
}


@dataclass(frozen=True)
class FrozenDatasetResult:
    normalized_quotes: pd.DataFrame
    path_result: PathBuildResult
    settlement_result: SettlementResult | None
    manifest: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return str(path)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported table format: {path}")


def _first_column(frame: pd.DataFrame, names: Iterable[str], *, required: bool = True) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    if required:
        raise ValueError(f"none of the required aliases are present: {list(names)}")
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def normalize_quote_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize known repository/provider schemas without inventing team identity."""

    normalized = pd.DataFrame(index=frame.index)
    normalized["event_id"] = _first_column(frame, ["event_id", "game_id"])
    normalized["event_start_time_utc"] = _first_column(
        frame, ["event_start_time_utc", "game_start_time", "commence_time_utc"]
    )
    normalized["snapshot_time_utc"] = _first_column(
        frame, ["snapshot_time_utc", "snapshot_time", "fetched_at_utc"]
    )
    normalized["player"] = _first_column(
        frame, ["player", "player_name_norm", "player_name", "player_name_raw"]
    )
    normalized["team"] = _first_column(
        frame, ["team", "player_team", "team_abbreviation"], required=False
    )
    raw_market = _first_column(frame, ["market", "market_key", "provider_market_key"])
    normalized["market"] = raw_market.astype(str).str.lower().map(MARKET_ALIASES).fillna(
        raw_market.astype(str).str.lower()
    )
    normalized["line"] = _first_column(frame, ["line", "current_line"])
    normalized["book"] = _first_column(
        frame, ["book", "book_key", "bookmaker_key", "bookmaker_title"]
    )
    normalized["engine"] = _first_column(
        frame, ["engine", "pricing_engine"], required=False
    )
    normalized["source"] = _first_column(frame, ["source", "provider_name"], required=False)
    return normalized


def build_frozen_dataset(
    quote_paths: Iterable[Path],
    *,
    outcome_path: Path | None = None,
    protocol: AllocationPathProtocol = ALLOCATION_PATH_PROTOCOL,
) -> FrozenDatasetResult:
    source_entries: list[dict[str, object]] = []
    quote_frames: list[pd.DataFrame] = []
    for path in quote_paths:
        resolved = Path(path).resolve()
        source = read_table(resolved)
        quote_frames.append(normalize_quote_table(source))
        source_entries.append(
            {
                "path": _manifest_path(resolved),
                "sha256": sha256_file(resolved),
                "rows": int(len(source)),
                "role": "quotes",
            }
        )
    if not quote_frames:
        raise ValueError("at least one quote path is required")

    normalized_quotes = pd.concat(quote_frames, ignore_index=True)
    path_result = build_allocation_paths(normalized_quotes, protocol=protocol)
    settlement_result = None
    if outcome_path is not None:
        resolved_outcomes = Path(outcome_path).resolve()
        outcomes = read_table(resolved_outcomes)
        settlement_result = attach_realized_allocations(path_result.player_features, outcomes)
        source_entries.append(
            {
                "path": _manifest_path(resolved_outcomes),
                "sha256": sha256_file(resolved_outcomes),
                "rows": int(len(outcomes)),
                "role": "outcomes",
            }
        )

    status_counts = (
        path_result.quality_ledger["status"].value_counts(dropna=False).to_dict()
        if not path_result.quality_ledger.empty
        else {}
    )
    manifest = {
        "dataset_version": f"{protocol.version}_DATASET",
        "representation_version": protocol.version,
        "source_files": source_entries,
        "normalized_quote_rows": int(len(normalized_quotes)),
        "valid_path_units": int(len(path_result.event_features)),
        "valid_player_coordinates": int(len(path_result.player_features)),
        "path_quality_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "settled_path_units": (
            int(settlement_result.settled_player_features["unit_id"].nunique())
            if settlement_result is not None and not settlement_result.settled_player_features.empty
            else 0
        ),
        "evidence_label": (
            "FULL_REPEATED_PATH_HISTORY"
            if len(path_result.event_features) > 0
            else "INSUFFICIENT_REAL_PATH_HISTORY"
        ),
    }
    return FrozenDatasetResult(normalized_quotes, path_result, settlement_result, manifest)


def write_frozen_dataset(result: FrozenDatasetResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.normalized_quotes.to_csv(output_dir / "market_snapshots.csv", index=False)
    result.path_result.player_features.to_csv(output_dir / "allocation_path_player_features.csv", index=False)
    result.path_result.event_features.to_csv(output_dir / "allocation_path_event_features.csv", index=False)
    result.path_result.quality_ledger.to_csv(output_dir / "allocation_path_quality_ledger.csv", index=False)
    if result.settlement_result is not None:
        result.settlement_result.settled_player_features.to_csv(
            output_dir / "allocation_path_settled_player_features.csv", index=False
        )
        result.settlement_result.quality_ledger.to_csv(
            output_dir / "allocation_path_settlement_quality.csv", index=False
        )
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(result.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
