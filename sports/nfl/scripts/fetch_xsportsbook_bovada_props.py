#!/usr/bin/env python3
"""Download and normalize the free 2021-2022 XSportsbook Bovada archive.

The publisher explicitly offers these CSV files for model training.  The
normalized output keeps only posted lines and two-sided prices; result columns
from the source file are intentionally discarded so they cannot leak into line
selection.  The source has no capture timestamp, so it is research evidence and
cannot by itself pass the repository's strict deployment provenance gate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd
import requests


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sports.nfl.predictions.market_sources import (  # noqa: E402
    flatten_xsportsbook_bovada_archive,
)


NFL_ROOT = REPO_ROOT / "sports" / "nfl"
SCHEDULE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "schedules/games.parquet"
)
ARCHIVES = {
    2021: {
        "url": "https://xsportsbook.com/wp-content/uploads/NFL-2021-Player-Prop-Results-1.csv",
        "sha256": "b377fadca650d3f699f781cc95e78cb7ae33bcb733416a0a87dfc9899d05ab2c",
    },
    2022: {
        "url": "https://xsportsbook.com/wp-content/uploads/NFL-2022-Player-Prop-Results-1.csv",
        "sha256": "9d056342cf8a1c967522e49950dd9a230ba27b2c493b87dd911e1f0acd456548",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2021,2022", help="Comma-separated subset of 2021,2022.")
    parser.add_argument(
        "--output",
        type=Path,
        default=NFL_ROOT / "data" / "raw" / "xsportsbook_bovada_player_props.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=NFL_ROOT / "data" / "raw" / "xsportsbook_bovada_player_props_manifest.json",
    )
    parser.add_argument(
        "--allow-source-change",
        action="store_true",
        help="Accept a source file whose checksum changed after manual review.",
    )
    return parser.parse_args()


def _download(session: requests.Session, season: int, allow_source_change: bool) -> tuple[pd.DataFrame, dict]:
    metadata = ARCHIVES[season]
    response = session.get(metadata["url"], timeout=60)
    response.raise_for_status()
    payload = response.content
    digest = hashlib.sha256(payload).hexdigest()
    if digest != metadata["sha256"] and not allow_source_change:
        raise RuntimeError(
            f"The {season} archive checksum changed ({digest}); inspect it and rerun "
            "with --allow-source-change only if the schema and provenance remain valid."
        )
    frame = pd.read_csv(io.BytesIO(payload), low_memory=False)
    return frame, {
        "season": season,
        "url": metadata["url"],
        "sha256": digest,
        "expected_sha256": metadata["sha256"],
        "bytes": len(payload),
        "raw_rows": int(len(frame)),
    }


def main() -> int:
    args = parse_args()
    seasons = sorted({int(value.strip()) for value in args.seasons.split(",") if value.strip()})
    unsupported = set(seasons).difference(ARCHIVES)
    if not seasons or unsupported:
        raise ValueError(f"Available seasons are {sorted(ARCHIVES)}; requested {seasons}.")

    schedule = pd.read_parquet(SCHEDULE_URL)
    schedule = schedule.loc[
        schedule["season"].isin(seasons) & schedule["game_type"].eq("REG")
    ].copy()
    session = requests.Session()
    session.headers.update(
        {"Accept": "text/csv", "User-Agent": "NFL-Predictor-Research/1.0"}
    )
    outputs: list[pd.DataFrame] = []
    source_files: list[dict] = []
    season_audits: dict[str, dict] = {}
    for season in seasons:
        raw, file_metadata = _download(session, season, args.allow_source_change)
        normalized, audit = flatten_xsportsbook_bovada_archive(
            raw,
            season=season,
            schedule=schedule.loc[schedule["season"].eq(season)],
        )
        outputs.append(normalized)
        source_files.append(file_metadata)
        season_audits[str(season)] = audit

    output = pd.concat(outputs, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    manifest = {
        "provider": "xsportsbook_bovada_archive",
        "publisher_page": "https://xsportsbook.com/nfl-player-prop-betting-detail-2021/",
        "bookmaker": "bovada",
        "seasons": seasons,
        "markets": ["player_pass_yds", "player_rush_yds", "player_reception_yds"],
        "source_files": source_files,
        "season_audits": season_audits,
        "normalized_rows": int(len(output)),
        "two_sided_price_rows": int(
            (output["over_price"].notna() & output["under_price"].notna()).sum()
        ),
        "strict_pregame_verified_rows": 0,
        "provenance_limitation": (
            "Publisher identifies historical Bovada prop bets and explicitly offers CSV downloads, "
            "but supplies neither capture timestamps nor an explicit closing-line guarantee."
        ),
        "result_columns_discarded": True,
        "output": str(args.output),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
