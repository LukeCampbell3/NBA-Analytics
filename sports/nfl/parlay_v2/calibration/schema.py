from __future__ import annotations

"""CalibrationObservation -- one SETTLED predictive event, the atomic unit
of the calibration ledger (mission section 3). Exact-event identity
(player, game, target, side, line, book) is part of the schema by
construction -- there is no way to construct an observation that pools
two different lines under one identity (see row_hash / observation_id,
both derived from the full exact-event key plus the quote actually taken).
"""

import hashlib
import json
from dataclasses import asdict, dataclass

SCHEMA_VERSION = "CALIBRATION_OBSERVATION_V1"


def exact_event_identity(player_id: str, game_id: str, target: str, side: str, line: float, book: str) -> tuple:
    """Canonical event identity (mission section 6). Two rows differing
    only in `line` are DIFFERENT events -- this key never collapses them."""
    return (str(player_id), str(game_id), str(target), str(side), float(line), str(book))


def _row_hash(fields: dict) -> str:
    canonical = json.dumps(fields, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CalibrationObservation:
    observation_id: str
    slate_id: str
    game_id: str
    event_date: str

    player_id: str
    player_name: str

    target: str
    side: str
    line: float
    book: str

    quote_decimal: float
    quote_timestamp: str

    prediction_value: float
    predictive_probability_if_available: float | None
    state_version: str
    predictive_version: str

    market_bucket: str
    line_bucket: str
    state_bucket: str

    settlement_status: str
    actual_outcome: float | None
    actual_unit_return: float | None

    decision_frozen_at: str
    settled_at: str
    calibration_admitted_at: str

    source_id: str
    source_hash: str
    row_hash: str

    calibration_version: str = SCHEMA_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


def build_observation(
    *,
    slate_id: str,
    game_id: str,
    event_date: str,
    player_id: str,
    player_name: str,
    target: str,
    side: str,
    line: float,
    book: str,
    quote_decimal: float,
    quote_timestamp: str,
    prediction_value: float,
    predictive_probability_if_available: float | None,
    state_version: str,
    predictive_version: str,
    market_bucket: str,
    line_bucket: str,
    state_bucket: str,
    settlement_status: str,
    actual_outcome: float | None,
    actual_unit_return: float | None,
    decision_frozen_at: str,
    settled_at: str,
    calibration_admitted_at: str,
    source_id: str,
    source_hash: str,
) -> CalibrationObservation:
    """Builds a CalibrationObservation with a content-derived
    observation_id and row_hash. Two calls with identical exact-event
    identity + quote + settlement content always produce the same
    observation_id/row_hash -- this is what makes ingestion idempotent
    (see store.py) and snapshots reproducible (see snapshot.py)."""
    identity = exact_event_identity(player_id, game_id, target, side, line, book)
    identity_fields = {
        "identity": identity,
        "quote_decimal": quote_decimal,
        "quote_timestamp": quote_timestamp,
        "settlement_status": settlement_status,
        "actual_outcome": actual_outcome,
        "source_id": source_id,
    }
    observation_id = _row_hash(identity_fields)
    row_hash = _row_hash({**identity_fields, "prediction_value": prediction_value, "source_hash": source_hash})
    return CalibrationObservation(
        observation_id=observation_id,
        slate_id=slate_id,
        game_id=game_id,
        event_date=event_date,
        player_id=player_id,
        player_name=player_name,
        target=target,
        side=side,
        line=line,
        book=book,
        quote_decimal=quote_decimal,
        quote_timestamp=quote_timestamp,
        prediction_value=prediction_value,
        predictive_probability_if_available=predictive_probability_if_available,
        state_version=state_version,
        predictive_version=predictive_version,
        market_bucket=market_bucket,
        line_bucket=line_bucket,
        state_bucket=state_bucket,
        settlement_status=settlement_status,
        actual_outcome=actual_outcome,
        actual_unit_return=actual_unit_return,
        decision_frozen_at=decision_frozen_at,
        settled_at=settled_at,
        calibration_admitted_at=calibration_admitted_at,
        source_id=source_id,
        source_hash=source_hash,
        row_hash=row_hash,
    )
