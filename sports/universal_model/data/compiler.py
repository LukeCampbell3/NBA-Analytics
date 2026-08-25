"""Dataset compiler (spec section 35): builds the real deterministic
Parquet-shard dataset from every sport adapter reporting
``sufficient_for_training=True``, plus the split and dataset manifests.

Run as:
    python -m sports.universal_model.data.compiler
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from sports.universal_model.adapters.base import SourceCoverage
from sports.universal_model.adapters.registry import ALL_ADAPTERS
from sports.universal_model.data.schema import UniversalEvent, UniversalFeature, schema_hash
from sports.universal_model.data.splits import write_split_manifest

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
DATASET_DIR = MANIFESTS_DIR / "dataset"
FEATURE_REGISTRY_PATH = MANIFESTS_DIR / "feature_registry.json"


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def collect_sufficient_sports() -> tuple[dict[str, list[UniversalEvent]], dict[str, list[UniversalFeature]], dict[str, SourceCoverage], dict[str, list[str]]]:
    events_by_sport: dict[str, list[UniversalEvent]] = {}
    features_by_sport: dict[str, list[UniversalFeature]] = {}
    coverage_by_sport: dict[str, SourceCoverage] = {}
    source_paths: dict[str, list[str]] = {}
    for sport, cls in ALL_ADAPTERS.items():
        adapter = cls()
        events, coverage = adapter.build_observations()
        coverage_by_sport[sport] = coverage
        source_paths[sport] = adapter.discover_sources()
        if not coverage.sufficient_for_training:
            continue
        violations = adapter.validate_timestamps(events) + adapter.validate_provenance(events)
        if violations:
            raise ValueError(f"{sport}: {len(violations)} leakage/provenance violations, e.g. {violations[:3]}")
        events_by_sport[sport] = events
        features_by_sport[sport] = adapter.map_universal_features(events) + adapter.map_namespaced_features(events)
    return events_by_sport, features_by_sport, coverage_by_sport, source_paths


def _to_wide_frame(events: list[UniversalEvent], features: list[UniversalFeature]) -> pd.DataFrame:
    event_df = pd.DataFrame([e.to_dict() for e in events])
    if features:
        feat_df = pd.DataFrame([f.to_dict() for f in features])
        wide = feat_df.pivot_table(index="observation_id", columns="feature_name", values="value", aggfunc="first")
        missing = feat_df.pivot_table(index="observation_id", columns="feature_name", values="missing", aggfunc="first")
        missing.columns = [f"{c}__missing" for c in missing.columns]
        wide = wide.join(missing)
        event_df = event_df.merge(wide, on="observation_id", how="left")
    event_df["_event_date"] = event_df["event_time"].str.slice(0, 10)
    return event_df


def compile_dataset() -> dict:
    events_by_sport, features_by_sport, coverage_by_sport, source_paths = collect_sufficient_sports()
    if not events_by_sport:
        raise RuntimeError("no sport reported sufficient_for_training=True; nothing to compile")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    total_events = 0
    for sport, events in events_by_sport.items():
        frame = _to_wide_frame(events, features_by_sport.get(sport, []))
        sport_dir = DATASET_DIR / f"sport={sport}"
        sport_dir.mkdir(parents=True, exist_ok=True)
        for season, season_frame in frame.groupby("season"):
            path = sport_dir / f"season={season}.parquet"
            season_frame.to_parquet(path, index=False)
        total_rows += len(frame)
        total_events += frame["event_id"].nunique()

    schema_hash_value = schema_hash()
    split_manifest = write_split_manifest(
        events_by_sport, source_paths, schema_hash_value, MANIFESTS_DIR / "split_manifest.json"
    )

    all_source_files: list[Path] = []
    for paths in source_paths.values():
        for rel in paths:
            p = REPO_ROOT / rel
            if p.is_file():
                all_source_files.append(p)
            elif p.is_dir():
                all_source_files.extend(sorted(f for f in p.rglob("*") if f.is_file())[:5])  # sample, dirs are coverage-only sources

    dataset_manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "schema_hash": schema_hash_value,
        "feature_registry_hash": _file_hash(FEATURE_REGISTRY_PATH),
        "split_manifest_hash": _file_hash(MANIFESTS_DIR / "split_manifest.json"),
        "sports_included": sorted(events_by_sport.keys()),
        "sports_excluded": {
            sport: {
                "reason": cov.reason,
                "row_count_available": cov.row_count,
                "event_count_available": cov.event_count,
            }
            for sport, cov in coverage_by_sport.items()
            if not cov.sufficient_for_training
        },
        "total_rows": int(total_rows),
        "total_events": int(total_events),
        "per_sport_rows": {sport: len(evs) for sport, evs in events_by_sport.items()},
        "per_sport_events": {sport: len({e.event_id for e in evs}) for sport, evs in events_by_sport.items()},
        "source_hashes": {str(p.relative_to(REPO_ROOT)): _file_hash(p) for p in all_source_files},
        "dataset_dir": str(DATASET_DIR.relative_to(REPO_ROOT)),
        "leakage_audit_pass": split_manifest["leakage_audit"]["pass"],
    }
    (MANIFESTS_DIR / "universal_dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2))
    return dataset_manifest


if __name__ == "__main__":
    manifest = compile_dataset()
    print(json.dumps({k: v for k, v in manifest.items() if k != "source_hashes"}, indent=2))
