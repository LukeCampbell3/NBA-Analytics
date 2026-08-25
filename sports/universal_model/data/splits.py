"""Deterministic chronological DERIVE/SELECT/TEST splits (spec sections
10-11). Splitting is done at EVENT-DATE granularity, never at the row
level, and always at the whole-sporting-event level: since every
observation's ``event_id`` maps to exactly one calendar date, grouping by
date automatically groups by event too -- no row from the same game can
land on both sides of a boundary.

Two split concepts are built, matching two different validation questions:

- ``per_sport``: each sport's own timeline is independently cut at
  ~70/15/15 by row count (spec section 11.A, within-sport chronological).
- ``global``: the pooled cross-sport timeline is cut once (spec section
  11.B, global chronological). Because NFL's only real season (2025) fully
  predates MLB's 2026 season in this repository, a single global cutover
  necessarily puts all of NFL's data before it -- this is reported
  explicitly in the manifest rather than silently producing an empty NFL
  SELECT/TEST slice under the global cut.

Splits are computed only over sports whose adapter reported
``sufficient_for_training=True`` (spec section 7: "do not force a sport
into training if its historical data is insufficient").
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from sports.universal_model.data.schema import UniversalEvent

DERIVE_FRAC = 0.70
SELECT_FRAC = 0.15
# remainder is TEST


@dataclass(frozen=True)
class SportSplit:
    sport: str
    derive_dates: list[str]
    select_dates: list[str]
    test_dates: list[str]
    derive_rows: int
    select_rows: int
    test_rows: int
    derive_events: int
    select_events: int
    test_events: int


def _event_date(event: UniversalEvent) -> str:
    return event.event_time[:10]


def _cut_dates_by_row_fraction(date_row_counts: list[tuple[str, int]]) -> tuple[list[str], list[str], list[str]]:
    """date_row_counts must be sorted ascending by date."""
    total = sum(c for _, c in date_row_counts)
    derive_target = total * DERIVE_FRAC
    select_target = total * (DERIVE_FRAC + SELECT_FRAC)
    derive, select, test = [], [], []
    running = 0
    for date, count in date_row_counts:
        running += count
        if running <= derive_target or not derive and not select and not test:
            derive.append(date)
        elif running <= select_target:
            select.append(date)
        else:
            test.append(date)
    # Guarantee non-empty SELECT/TEST when there are enough distinct dates
    # (a real dataset with >=3 distinct dates should never produce an empty
    # tail bucket purely from rounding).
    if not select and len(derive) > 1:
        select.append(derive.pop())
    if not test and len(select) > 1:
        test.append(select.pop())
    elif not test and len(derive) > 1:
        test.append(derive.pop())
    return derive, select, test


def build_per_sport_split(events: list[UniversalEvent]) -> SportSplit:
    by_date: dict[str, list[UniversalEvent]] = defaultdict(list)
    for e in events:
        by_date[_event_date(e)].append(e)
    dates_sorted = sorted(by_date.keys())
    date_row_counts = [(d, len(by_date[d])) for d in dates_sorted]
    derive_d, select_d, test_d = _cut_dates_by_row_fraction(date_row_counts)

    def _rows(dates: list[str]) -> int:
        return sum(len(by_date[d]) for d in dates)

    def _n_events(dates: list[str]) -> int:
        ids = set()
        for d in dates:
            for e in by_date[d]:
                ids.add(e.event_id)
        return len(ids)

    sport = events[0].sport
    return SportSplit(
        sport=sport,
        derive_dates=derive_d,
        select_dates=select_d,
        test_dates=test_d,
        derive_rows=_rows(derive_d),
        select_rows=_rows(select_d),
        test_rows=_rows(test_d),
        derive_events=_n_events(derive_d),
        select_events=_n_events(select_d),
        test_events=_n_events(test_d),
    )


def build_global_split(events_by_sport: dict[str, list[UniversalEvent]]) -> SportSplit:
    pooled: list[UniversalEvent] = [e for evs in events_by_sport.values() for e in evs]
    by_date: dict[str, list[UniversalEvent]] = defaultdict(list)
    for e in pooled:
        by_date[_event_date(e)].append(e)
    dates_sorted = sorted(by_date.keys())
    date_row_counts = [(d, len(by_date[d])) for d in dates_sorted]
    derive_d, select_d, test_d = _cut_dates_by_row_fraction(date_row_counts)

    def _rows(dates: list[str]) -> int:
        return sum(len(by_date[d]) for d in dates)

    def _n_events(dates: list[str]) -> int:
        ids = set()
        for d in dates:
            for e in by_date[d]:
                ids.add(f"{e.sport}:{e.event_id}")
        return len(ids)

    return SportSplit(
        sport="__global__",
        derive_dates=derive_d,
        select_dates=select_d,
        test_dates=test_d,
        derive_rows=_rows(derive_d),
        select_rows=_rows(select_d),
        test_rows=_rows(test_d),
        derive_events=_n_events(derive_d),
        select_events=_n_events(select_d),
        test_events=_n_events(test_d),
    )


def assign_split(event: UniversalEvent, split: SportSplit) -> str:
    date = _event_date(event)
    if date in split.derive_dates:
        return "DERIVE"
    if date in split.select_dates:
        return "SELECT"
    if date in split.test_dates:
        return "TEST"
    raise ValueError(f"date {date} not covered by split for sport={split.sport}")


def audit_no_cross_split_event_leakage(events: list[UniversalEvent], split: SportSplit) -> list[str]:
    """Spec section 9: never allow rows from the same event on both sides
    of a boundary. Verifies this directly rather than assuming the
    date-grouping logic is correct."""
    event_to_splits: dict[str, set[str]] = defaultdict(set)
    for e in events:
        event_to_splits[f"{e.sport}:{e.event_id}"].add(assign_split(e, split))
    violations = [k for k, splits in event_to_splits.items() if len(splits) > 1]
    return violations


def _source_hash(paths: Iterable[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p).encode("utf-8"))
        if p.exists():
            h.update(str(p.stat().st_size).encode("utf-8"))
            h.update(str(p.stat().st_mtime_ns).encode("utf-8"))
    return h.hexdigest()[:16]


def write_split_manifest(
    events_by_sport: dict[str, list[UniversalEvent]],
    source_paths: dict[str, list[str]],
    schema_hash_value: str,
    out_path: Path,
) -> dict:
    per_sport_splits = {sport: build_per_sport_split(evs) for sport, evs in events_by_sport.items()}
    global_split = build_global_split(events_by_sport)

    leakage_report: dict[str, list[str]] = {}
    for sport, evs in events_by_sport.items():
        leakage_report[sport] = audit_no_cross_split_event_leakage(evs, per_sport_splits[sport])
    leakage_report["__global__"] = audit_no_cross_split_event_leakage(
        [e for evs in events_by_sport.values() for e in evs], global_split
    )

    def _serialize(s: SportSplit) -> dict:
        return {
            "sport": s.sport,
            "derive_dates": [s.derive_dates[0], s.derive_dates[-1]] if s.derive_dates else [],
            "select_dates": [s.select_dates[0], s.select_dates[-1]] if s.select_dates else [],
            "test_dates": [s.test_dates[0], s.test_dates[-1]] if s.test_dates else [],
            "derive_dates_full": s.derive_dates,
            "select_dates_full": s.select_dates,
            "test_dates_full": s.test_dates,
            "row_counts": {"derive": s.derive_rows, "select": s.select_rows, "test": s.test_rows},
            "event_counts": {"derive": s.derive_events, "select": s.select_events, "test": s.test_events},
        }

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "derive_fraction_target": DERIVE_FRAC,
        "select_fraction_target": SELECT_FRAC,
        "schema_hash": schema_hash_value,
        "sports_included": sorted(events_by_sport.keys()),
        "source_paths": source_paths,
        "per_sport": {sport: _serialize(s) for sport, s in per_sport_splits.items()},
        "global": _serialize(global_split),
        "leakage_audit": {
            "same_event_crosses_split_boundary": leakage_report,
            "pass": all(len(v) == 0 for v in leakage_report.values()),
        },
        "note_nfl_global_split": (
            "NFL's only real season (2025) fully predates MLB's 2026 season in this "
            "repository's data, so the GLOBAL chronological cutover places 100% of NFL rows "
            "in DERIVE. This is a real property of the available data (documented, not a bug); "
            "NFL's own within-sport split under 'per_sport' is unaffected and still holds out "
            "SELECT/TEST from NFL's own 2025 timeline."
            if "nfl" in events_by_sport
            else ""
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    return manifest
