from __future__ import annotations

"""Build checkpoint path-feature CSVs from recovered MLB market snapshots.

Reads the normalized, timestamped MLB prop-odds snapshot history
(``sports/mlb/data/raw/market_odds/mlb/**/normalized/*.csv``, the same
schema ``sports/mlb/scripts/recover_historical_market_snapshots.py``
produces) and buckets it, per ``(event, player, market)``, into the five
checkpoints declared in ``protocol.ALLOCATION_PATH_PROTOCOL``. Each
checkpoint is filled by the snapshot nearest that offset before
``commence_time_utc``, within ``max_checkpoint_age_minutes``; a coordinate
missing a usable snapshot at any checkpoint is dropped rather than
interpolated.

``share_m*`` is the no-vig implied probability of the play's own side
(computed from ``over_price``/``under_price``); ``line_m*`` is the prop
line at that snapshot. Both require a play's ``side`` (OVER/UNDER) to be
known, so this script is driven by a reservoir CSV (see
``build_reservoir_from_history.py``) rather than the raw snapshot file
alone.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import ALLOCATION_PATH_PROTOCOL

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SNAPSHOT_GLOBS = (
    "sports/mlb/data/raw/market_odds/mlb/odds_api_io/normalized/*.csv",
    "sports/mlb/data/raw/market_odds/mlb/historical_recovered/*.csv",
)

CHECKPOINT_MINUTES = tuple(abs(value) for value in ALLOCATION_PATH_PROTOCOL.checkpoints_minutes)
SHARE_COLUMNS = tuple(f"share_m{value}" for value in CHECKPOINT_MINUTES)
LINE_COLUMNS = tuple(f"line_m{value}" for value in CHECKPOINT_MINUTES)

# Reservoir "market" values are the short target codes generate_daily_prediction_pool.py
# and the published board use (H/TB/R/HR/RBI/K/ER); the odds-provider snapshot
# CSV's "market_key" uses the odds API's own long names. A substring match
# between the two is unreliable (e.g. "K" matches inside "stri-k-eouts" by
# accident), so this is an explicit crosswalk.
TARGET_TO_MARKET_KEY = {
    "H": "batter_hits",
    "TB": "batter_total_bases",
    "R": "batter_runs_scored",
    "HR": "batter_home_runs",
    "RBI": "batter_rbis",
    "K": "pitcher_strikeouts",
    "ER": "pitcher_earned_runs",
}


def _american_to_probability(price: float) -> float:
    if price > 0:
        return 100.0 / (price + 100.0)
    return -price / (-price + 100.0)


def _no_vig_probability(side: str, over_price: float, under_price: float) -> float | None:
    if not np.isfinite(over_price) or not np.isfinite(under_price):
        return None
    over_p = _american_to_probability(over_price)
    under_p = _american_to_probability(under_price)
    total = over_p + under_p
    if total <= 0:
        return None
    over_novig = over_p / total
    return over_novig if side.upper() == "OVER" else 1.0 - over_novig


def load_snapshots(repo_root: Path, globs: tuple[str, ...] = DEFAULT_SNAPSHOT_GLOBS) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in globs:
        for path in sorted(repo_root.glob(pattern)):
            try:
                frames.append(pd.read_csv(path, low_memory=False))
            except (OSError, pd.errors.ParserError):
                continue
    if not frames:
        return pd.DataFrame(
            columns=[
                "fetched_at_utc",
                "event_id",
                "commence_time_utc",
                "player_name_norm",
                "market_key",
                "line",
                "over_price",
                "under_price",
            ]
        )
    return pd.concat(frames, ignore_index=True, sort=False)


def build_path_features(
    reservoir: pd.DataFrame,
    snapshots: pd.DataFrame,
    *,
    max_checkpoint_age_minutes: int = ALLOCATION_PATH_PROTOCOL.max_checkpoint_age_minutes,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required_reservoir = {"event_date", "player", "market", "side"}
    missing = sorted(required_reservoir - set(reservoir.columns))
    if missing:
        raise ValueError(f"reservoir is missing columns: {missing}")
    required_snapshot = {
        "fetched_at_utc",
        "commence_time_utc",
        "player_name_norm",
        "market_key",
        "line",
        "over_price",
        "under_price",
    }
    missing_snapshot = sorted(required_snapshot - set(snapshots.columns))
    if missing_snapshot:
        raise ValueError(f"snapshots are missing columns: {missing_snapshot}")

    snap = snapshots.copy()
    snap["fetched_at_utc"] = pd.to_datetime(snap["fetched_at_utc"], errors="coerce", utc=True)
    snap["commence_time_utc"] = pd.to_datetime(snap["commence_time_utc"], errors="coerce", utc=True)
    snap = snap.dropna(subset=["fetched_at_utc", "commence_time_utc"])
    # The reservoir's event_date (from the published board's official game
    # date) and the snapshot's commence_time_utc come from different
    # providers with different event-ID namespaces (MLB StatsAPI gamePk vs.
    # the odds API's own hex event_id), so they cannot be joined on ID.
    # Both sides do agree on the calendar date of the game (US Eastern, MLB's
    # scheduling convention), which is what we match on instead.
    snap["commence_date_et"] = (
        snap["commence_time_utc"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    )
    snap["player_name_norm"] = snap["player_name_norm"].astype(str).str.replace("_", " ").str.casefold()
    snap["market_key"] = snap["market_key"].astype(str)
    snap["over_price"] = pd.to_numeric(snap["over_price"], errors="coerce")
    snap["under_price"] = pd.to_numeric(snap["under_price"], errors="coerce")
    snap["line"] = pd.to_numeric(snap["line"], errors="coerce")

    rows: list[dict[str, object]] = []
    dropped_incomplete = 0
    dropped_unmapped_market = 0
    tolerance = pd.Timedelta(minutes=max_checkpoint_age_minutes)

    deduped = reservoir.drop_duplicates(subset=["event_date", "player", "market"]).copy()
    deduped["event_date"] = pd.to_datetime(deduped["event_date"], errors="coerce").dt.normalize()

    for _, candidate in deduped.iterrows():
        market_key = TARGET_TO_MARKET_KEY.get(str(candidate["market"]))
        if market_key is None:
            dropped_unmapped_market += 1
            continue
        group = snap.loc[
            snap["commence_date_et"].eq(candidate["event_date"])
            & snap["player_name_norm"].eq(str(candidate["player"]).casefold())
            & snap["market_key"].eq(market_key)
        ]
        if group.empty:
            dropped_incomplete += 1
            continue
        commence = group["commence_time_utc"].iloc[0]

        # No event_id column: the reservoir's event_id (MLB StatsAPI gamePk)
        # and the snapshot's event_id (odds-provider hex ID) are different
        # namespaces (see above), and merge_candidates_with_paths prefers
        # event_id over event_date whenever both sides have it -- emitting a
        # non-crosswalked event_id here would silently break that join.
        row: dict[str, object] = {
            "event_date": candidate["event_date"],
            "player": candidate["player"],
            "market": candidate["market"],
        }
        complete = True
        for offset in CHECKPOINT_MINUTES:
            target_time = commence - pd.Timedelta(minutes=offset)
            deltas = (group["fetched_at_utc"] - target_time).abs()
            within = deltas <= tolerance
            if not bool(within.any()):
                complete = False
                break
            nearest = group.loc[within].loc[deltas.loc[within].idxmin()]
            probability = _no_vig_probability(
                str(candidate["side"]), float(nearest["over_price"]), float(nearest["under_price"])
            )
            if probability is None or not np.isfinite(nearest["line"]):
                complete = False
                break
            row[f"share_m{offset}"] = probability
            row[f"line_m{offset}"] = float(nearest["line"])
        if not complete:
            dropped_incomplete += 1
            continue
        rows.append(row)

    # Always emit the path-features schema, even with zero rows: real MLB
    # snapshot collection is sparse enough today that "no coordinate has a
    # complete checkpoint path yet" is an expected outcome, not an error, and
    # it must not crash a columnless-DataFrame check downstream.
    feature_columns = ["event_date", "player", "market", *SHARE_COLUMNS, *LINE_COLUMNS]
    features = pd.DataFrame(rows, columns=feature_columns)
    summary = {
        "coordinates_considered": int(len(deduped)),
        "coordinates_written": int(len(features)),
        "coordinates_dropped_incomplete_path": int(dropped_incomplete),
        "coordinates_dropped_unmapped_market": int(dropped_unmapped_market),
    }
    return features, summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build MLB checkpoint path-feature CSV from recovered market snapshots."
    )
    parser.add_argument("--reservoir-csv", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-csv", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    reservoir = pd.read_csv(args.reservoir_csv, low_memory=False)
    snapshots = load_snapshots(args.repo_root)
    features, summary = build_path_features(reservoir, snapshots)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_csv, index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
