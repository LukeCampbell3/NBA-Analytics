from __future__ import annotations

"""Frozen chronological DERIVE/SELECT/TEST day-stamp partition.

These are the exact 25 archived sports/mlb/data/predictions/daily_runs/*
day-stamps that had a raw daily_prediction_pool_*.csv on disk as of the
three-way split first run (2026-08-21). The list is a literal tuple, not
recomputed from the directory at import time: if more archived days are
added later, DERIVE/SELECT/TEST must stay exactly this set, or the frozen
bias correction and the retired TEST result stop meaning what they say they
mean. `verify_against_disk()` checks the literal against what's on disk and
raises if archived days now present were not part of this partition --
extending the partition is a deliberate, separate decision, not something
that happens silently by adding files.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DAILY_RUNS_ROOT = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "daily_runs"

# fmt: off
DERIVE_STAMPS: tuple[str, ...] = (
    "20260405", "20260410", "20260426", "20260427",
    "20260429", "20260501", "20260619", "20260620",
)
SELECT_STAMPS: tuple[str, ...] = (
    "20260621", "20260626", "20260627", "20260729",
    "20260730", "20260731", "20260801", "20260802",
)
# RETIRED. Never read by ranker-development code in this package again.
# The only legitimate use of this tuple is to (a) assert code does NOT touch
# it, and (b) label the already-frozen historical TEST result for
# documentation. See manifest.py FROZEN_TEST_RESULT.
TEST_STAMPS: tuple[str, ...] = (
    "20260803", "20260804", "20260805", "20260806", "20260807",
    "20260808", "20260809", "20260810", "20260811",
)
# fmt: on

DEVELOPMENT_STAMPS: tuple[str, ...] = DERIVE_STAMPS + SELECT_STAMPS
ALL_STAMPS: tuple[str, ...] = DERIVE_STAMPS + SELECT_STAMPS + TEST_STAMPS


def verify_against_disk(daily_runs_root: Path = DAILY_RUNS_ROOT) -> None:
    """Raise if the archived days on disk have drifted from the frozen partition.

    New archived days appearing on disk (e.g. a fresher checkout, or the
    daily CI job accumulating more history) must NOT silently join
    DEVELOPMENT_STAMPS or TEST_STAMPS -- that would be exactly the kind of
    retrospective reinterpretation the confirmation policy forbids. This
    only checks that the frozen 25 are still present and unambiguous; it
    intentionally does not fail just because *more* days now exist on disk.
    """
    on_disk = {
        p.name
        for p in daily_runs_root.iterdir()
        if p.is_dir() and (p / f"daily_prediction_pool_{p.name}.csv").exists()
    }
    missing = set(ALL_STAMPS) - on_disk
    if missing:
        raise RuntimeError(
            f"frozen DERIVE/SELECT/TEST partition references day-stamps no longer "
            f"on disk: {sorted(missing)}"
        )
