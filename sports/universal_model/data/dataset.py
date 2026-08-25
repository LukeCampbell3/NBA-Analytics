"""Torch Dataset over the compiled Parquet shards, applying the frozen
split/normalization manifests. One example = one ``UniversalEvent`` row,
already reduced to fixed-shape tensors so the model never sees raw text.

Cache invalidation (spec section 36): ``UniversalDataset`` records the
schema/feature-registry/normalization/split hashes it was built from in
``last_build_signature``; callers should compare this against the current
manifests before reusing a pickled/cached dataset.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from sports.universal_model.data.normalization import (
    DATASET_DIR,
    MANIFESTS_DIR,
    NUMERIC_UNIVERSAL_COLUMNS,
    apply_robust_scale,
)

ENTITY_HASH_BUCKETS = 8192
N_TOKENS = 8  # SPORT, ENTITY, ROLE, OPPORTUNITY, TEMPORAL, MARKET, UNCERTAINTY, TARGET


def _hash_bucket(value: str, n_buckets: int) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest(), 16) % n_buckets


@dataclass(frozen=True)
class DatasetSignature:
    schema_hash: str
    feature_registry_hash: str
    normalization_hash: str
    split_hash: str

    def key(self) -> str:
        return f"{self.schema_hash}:{self.feature_registry_hash}:{self.normalization_hash}:{self.split_hash}"


def current_signature() -> DatasetSignature:
    def _h(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else "missing"

    dataset_manifest = json.loads((MANIFESTS_DIR / "universal_dataset_manifest.json").read_text())
    return DatasetSignature(
        schema_hash=dataset_manifest["schema_hash"],
        feature_registry_hash=dataset_manifest["feature_registry_hash"],
        normalization_hash=_h(MANIFESTS_DIR / "normalization_manifest.json"),
        split_hash=dataset_manifest["split_manifest_hash"],
    )


class UniversalDataset(Dataset):
    """split: 'DERIVE' | 'SELECT' | 'TEST'. sports: subset to include
    (leave-one-sport-out transfer tests construct this with a reduced list).
    split_kind: 'per_sport' (default) or 'global' -- selects which cutover
    dates from split_manifest.json to use."""

    def __init__(
        self,
        split: str,
        sports: Optional[list[str]] = None,
        split_kind: str = "per_sport",
        split_manifest_path: Path = MANIFESTS_DIR / "split_manifest.json",
        normalization_path: Path = MANIFESTS_DIR / "normalization_manifest.json",
    ) -> None:
        assert split in ("DERIVE", "SELECT", "TEST")
        self.split = split
        self.norm = json.loads(normalization_path.read_text())
        split_manifest = json.loads(split_manifest_path.read_text())
        all_sports = sports or split_manifest["sports_included"]

        rows: list[pd.DataFrame] = []
        for sport in all_sports:
            frame = self._load_sport_frame(sport)
            if split_kind == "per_sport":
                key = f"{split.lower()}_dates_full"
                dates = set(split_manifest["per_sport"][sport][key])
            else:
                dates = set(split_manifest["global"][f"{split.lower()}_dates_full"])
            subset = frame[frame["_event_date"].isin(dates)]
            if len(subset):
                rows.append(subset)
        self.frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        self.signature = current_signature()

    @staticmethod
    def _load_sport_frame(sport: str) -> pd.DataFrame:
        sport_dir = DATASET_DIR / f"sport={sport}"
        frames = [pd.read_parquet(p) for p in sorted(sport_dir.glob("*.parquet"))]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[idx]
        sport = row["sport"]
        vocabs = self.norm["vocabs"]

        sport_id = vocabs["sport"].get(sport, 0)
        entity_bucket = _hash_bucket(f"{sport}:{row['entity_id']}", ENTITY_HASH_BUCKETS)
        role_key = f"{sport}.{row['role']}" if pd.notna(row.get("role")) else None
        role_id = vocabs["role"].get(role_key, -1) if role_key else -1
        target_key = f"{sport}.{row['target']}"
        target_id = vocabs["target"].get(target_key, -1)
        home_id = vocabs["home_away"].get(row.get("home_away"), -1) if pd.notna(row.get("home_away")) else -1

        numeric_stats = self.norm["numeric_stats"].get(sport, {})
        numeric_vec = []
        missing_vec = []
        for col in NUMERIC_UNIVERSAL_COLUMNS:
            raw = row.get(col)
            stats = numeric_stats.get(col)
            is_missing = raw is None or (isinstance(raw, float) and np.isnan(raw)) or stats is None
            numeric_vec.append(0.0 if is_missing else apply_robust_scale(raw, stats))
            missing_vec.append(1.0 if is_missing else 0.0)

        line = row.get("line")
        actual = row.get("actual_value")
        has_line = line is not None and not (isinstance(line, float) and np.isnan(line))
        y_over = -1.0  # -1 = masked/unavailable
        if has_line and actual is not None and not (isinstance(actual, float) and np.isnan(actual)):
            if actual > line:
                y_over = 1.0
            elif actual < line:
                y_over = 0.0
            # actual == line (push): left masked (-1)

        target_stats = self.norm["target_stats"].get(sport, {}).get(row["target"])
        z_actual = apply_robust_scale(actual, target_stats) if target_stats else 0.0
        z_valid = 1.0 if (target_stats and actual is not None and not (isinstance(actual, float) and np.isnan(actual))) else 0.0

        return {
            "sport_id": torch.tensor(sport_id, dtype=torch.long),
            "entity_bucket": torch.tensor(entity_bucket, dtype=torch.long),
            "role_id": torch.tensor(role_id + 1, dtype=torch.long),  # shift so -1(unknown)->0
            "target_id": torch.tensor(max(target_id, 0), dtype=torch.long),
            "home_id": torch.tensor(home_id + 1, dtype=torch.long),
            "numeric": torch.tensor(numeric_vec, dtype=torch.float32),
            "missing": torch.tensor(missing_vec, dtype=torch.float32),
            "y_over": torch.tensor(y_over, dtype=torch.float32),
            "y_over_mask": torch.tensor(1.0 if y_over >= 0 else 0.0, dtype=torch.float32),
            "z_actual": torch.tensor(z_actual, dtype=torch.float32),
            "z_valid": torch.tensor(z_valid, dtype=torch.float32),
        }
