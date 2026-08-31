from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


APPROVED_SETTLEMENT_SOURCES = {"MLB_STATSAPI_FINAL_FEED"}


def settlement_key(*, source_commit: str, game_id: Any, player_id: Any, market: Any, side: Any, line: Any) -> tuple[str, ...]:
    return (
        str(source_commit), str(game_id), str(player_id).strip().lower(),
        str(market).strip().upper(), str(side).strip().upper(), str(float(line)),
    )


def _parse_utc(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("settlement timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_historical_settlement(record: dict[str, Any], snapshot_generated_at: str) -> None:
    required = {
        "source_commit", "game_id", "player_id", "market", "side", "line",
        "actual_value", "settlement", "source_type", "source_url", "source_sha256",
        "game_finalized_at_utc", "retrieved_at_utc",
    }
    missing = sorted(required - record.keys())
    if missing:
        raise ValueError(f"historical settlement missing fields: {','.join(missing)}")
    if record["source_type"] not in APPROVED_SETTLEMENT_SOURCES:
        raise ValueError("historical settlement source is not approved")
    if not str(record["source_url"]).startswith("https://statsapi.mlb.com/"):
        raise ValueError("historical settlement URL is not an approved MLB endpoint")
    if len(str(record["source_sha256"])) != 64:
        raise ValueError("historical settlement source hash is invalid")
    if _parse_utc(record["game_finalized_at_utc"]) <= _parse_utc(snapshot_generated_at):
        raise ValueError("settlement cannot predate the frozen prediction")
    if _parse_utc(record["retrieved_at_utc"]) < _parse_utc(record["game_finalized_at_utc"]):
        raise ValueError("settlement retrieval cannot predate game finalization")
    actual, line = float(record["actual_value"]), float(record["line"])
    side = str(record["side"]).upper()
    expected = "push" if actual == line else ("won" if (actual > line) == (side == "OVER") else "lost")
    if str(record["settlement"]).lower() != expected:
        raise ValueError("historical settlement result disagrees with actual value")


def load_historical_settlements(repo_root: Path) -> dict[tuple[str, ...], dict[str, Any]]:
    path = repo_root / "sports/mlb/data/predictions/unified/historical_settlements.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in payload.get("settlements", []):
        key = settlement_key(
            source_commit=record.get("source_commit"), game_id=record.get("game_id"),
            player_id=record.get("player_id"), market=record.get("market"),
            side=record.get("side"), line=record.get("line"),
        )
        if key in rows:
            raise ValueError("duplicate historical settlement identity")
        rows[key] = record
    return rows
