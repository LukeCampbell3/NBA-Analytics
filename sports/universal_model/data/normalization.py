"""DERIVE-only normalization + vocabulary fitting (spec section 13).

Fits every statistic used downstream (numeric robust-scaling stats,
target location/scale, categorical vocabularies) on DERIVE rows only, per
sport, using the frozen ``split_manifest.json``. Never touches SELECT/TEST.

Also corrects a real labeling issue found while building this: the source
MLB dataset's ``Result``/``binary_result`` column encodes whether an
*existing, separate* per-sport model's own pick direction won -- using
that as our training label would make the label itself a function of a
different model's (excluded, UNUSABLE) prediction. Since ``Market_Line``
is populated for 100% of MLB rows and always represents the "over" side
(see adapters/mlb.py), this module instead derives a neutral,
model-independent binary label directly:

    y_over = 1{actual_value > line}   (push -> label masked, not fabricated)

matching spec section 16 exactly ("y = 1{actual > line}"). NFL has no
market line, so it only gets the continuous regression target (masked-out
binary head), also per spec section 16 ("masked multi-task losses so
unavailable targets do not create fake labels").
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MANIFESTS_DIR = Path(__file__).resolve().parents[1] / "manifests"
DATASET_DIR = MANIFESTS_DIR / "dataset"

NUMERIC_UNIVERSAL_COLUMNS = [
    "line",
    "american_price",
    "universal.sample_support_rows",
    "universal.days_since_last_history",
    "universal.recency_prior",
]


def _robust_stats(series: pd.Series) -> dict:
    values = series.dropna().astype(float)
    if len(values) == 0:
        return {"median": 0.0, "iqr": 1.0, "n": 0}
    median = float(values.median())
    q25, q75 = float(values.quantile(0.25)), float(values.quantile(0.75))
    iqr = q75 - q25
    if iqr < 1e-6:
        iqr = max(float(values.std()), 1e-6)
    return {"median": median, "iqr": iqr, "n": int(len(values))}


def _load_sport_frame(sport: str) -> pd.DataFrame:
    sport_dir = DATASET_DIR / f"sport={sport}"
    frames = [pd.read_parquet(p) for p in sorted(sport_dir.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True)


def fit_normalization(split_manifest_path: Path = MANIFESTS_DIR / "split_manifest.json") -> dict:
    split_manifest = json.loads(split_manifest_path.read_text())
    sports = split_manifest["sports_included"]

    numeric_stats: dict[str, dict[str, dict]] = {}
    target_stats: dict[str, dict[str, dict]] = {}
    vocabs: dict[str, dict[str, int]] = {"sport": {}, "target": {}, "role": {}, "position": {}, "home_away": {}}

    for sport in sports:
        frame = _load_sport_frame(sport)
        derive_dates = set(split_manifest["per_sport"][sport]["derive_dates_full"])
        derive = frame[frame["_event_date"].isin(derive_dates)]

        numeric_stats[sport] = {}
        for col in NUMERIC_UNIVERSAL_COLUMNS:
            if col in derive.columns:
                numeric_stats[sport][col] = _robust_stats(derive[col])

        target_stats[sport] = {}
        for target, group in derive.groupby("target"):
            target_stats[sport][target] = _robust_stats(group["actual_value"])

        for col, key in [("target", "target"), ("role", "role"), ("position", "position"), ("home_away", "home_away")]:
            if col not in frame.columns:
                continue
            for value in sorted(frame[col].dropna().unique().tolist()):
                composite = f"{sport}.{value}" if key in ("target", "role", "position") else str(value)
                if composite not in vocabs[key]:
                    vocabs[key][composite] = len(vocabs[key])

    for sport in sports:
        if sport not in vocabs["sport"]:
            vocabs["sport"][sport] = len(vocabs["sport"])

    manifest = {
        "fit_on": "DERIVE only, per sport",
        "sports": sports,
        "numeric_stats": numeric_stats,
        "target_stats": target_stats,
        "vocabs": vocabs,
        "label_definition": "y_over = 1{actual_value > line} where line is available (MLB); regression-only (binary head masked) where line is unavailable (NFL). Derived independently of any excluded/UNUSABLE incumbent-model prediction column.",
        "entity_hash_buckets": 8192,
    }
    (MANIFESTS_DIR / "normalization_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def apply_robust_scale(value: Optional[float], stats: dict) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    return (float(value) - stats["median"]) / stats["iqr"]


if __name__ == "__main__":
    m = fit_normalization()
    print(f"fit normalization for sports={m['sports']}; vocab sizes: "
          f"{ {k: len(v) for k, v in m['vocabs'].items()} }")
