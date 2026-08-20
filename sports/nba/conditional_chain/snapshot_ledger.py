from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Mapping


SNAPSHOT_COLUMNS = (
    "slate_id",
    "event_id",
    "game_id",
    "sport",
    "event_start_time_utc",
    "snapshot_time_utc",
    "player_id",
    "player_name",
    "market",
    "line",
    "side",
    "book",
    "engine",
    "decimal_odds",
    "source",
    "raw_source_hash",
    "parser_version",
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _missing(value: object) -> bool:
    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or str(value).strip() == ""
    )


def _canonical_snapshot_id(row: Mapping[str, object]) -> str:
    payload = {column: row.get(column) for column in SNAPSHOT_COLUMNS}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class MarketSnapshotLedger:
    """Append-only SQLite ledger for immutable, timestamped market quotes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    slate_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    sport TEXT NOT NULL,
                    event_start_time_utc TEXT NOT NULL,
                    snapshot_time_utc TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    market TEXT NOT NULL,
                    line REAL NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('OVER', 'UNDER')),
                    book TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    decimal_odds REAL NOT NULL CHECK(decimal_odds > 1.0),
                    source TEXT NOT NULL,
                    raw_source_hash TEXT NOT NULL,
                    parser_version TEXT NOT NULL,
                    inserted_at_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS market_snapshots_coordinate_time
                ON market_snapshots(event_id, player_id, market, line, side, snapshot_time_utc);
                CREATE TRIGGER IF NOT EXISTS market_snapshots_no_update
                BEFORE UPDATE ON market_snapshots
                BEGIN
                    SELECT RAISE(ABORT, 'market_snapshots is append-only');
                END;
                CREATE TRIGGER IF NOT EXISTS market_snapshots_no_delete
                BEFORE DELETE ON market_snapshots
                BEGIN
                    SELECT RAISE(ABORT, 'market_snapshots is append-only');
                END;
                """
            )

    def append(self, rows: Iterable[Mapping[str, object]]) -> int:
        prepared: list[tuple[object, ...]] = []
        for row in rows:
            missing = [
                column for column in SNAPSHOT_COLUMNS if _missing(row.get(column))
            ]
            if missing:
                raise ValueError(
                    f"market snapshot is missing required values: {missing}"
                )
            side = str(row["side"]).upper()
            if side not in {"OVER", "UNDER"}:
                raise ValueError(f"unsupported market side: {side}")
            line = float(row["line"])
            decimal_odds = float(row["decimal_odds"])
            if not math.isfinite(line) or line <= 0.0:
                raise ValueError("market snapshot line must be finite and positive")
            if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
                raise ValueError("market snapshot decimal_odds must be greater than 1")
            raw_hash = str(row["raw_source_hash"]).lower()
            if not SHA256_PATTERN.fullmatch(raw_hash):
                raise ValueError("market snapshot raw_source_hash must be SHA-256")
            values = dict(row)
            values["side"] = side
            values["line"] = line
            values["decimal_odds"] = decimal_odds
            values["raw_source_hash"] = raw_hash
            snapshot_id = str(row.get("snapshot_id") or _canonical_snapshot_id(values))
            prepared.append(
                (snapshot_id, *(values[column] for column in SNAPSHOT_COLUMNS))
            )
        if not prepared:
            return 0

        placeholders = ",".join("?" for _ in range(len(SNAPSHOT_COLUMNS) + 1))
        with self._connect() as connection:
            before = connection.total_changes
            connection.executemany(
                f"INSERT OR IGNORE INTO market_snapshots "
                f"(snapshot_id,{','.join(SNAPSHOT_COLUMNS)}) VALUES ({placeholders})",
                prepared,
            )
            return int(connection.total_changes - before)

    def rows(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            records = connection.execute(
                "SELECT * FROM market_snapshots ORDER BY snapshot_time_utc, snapshot_id"
            ).fetchall()
        return [dict(record) for record in records]
