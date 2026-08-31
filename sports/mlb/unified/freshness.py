from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any


def validate_freshness(payload: dict[str, Any], *, expected_slate_date: date,
                       now: datetime, maximum_age_hours: float) -> None:
    if str(payload.get("run_date") or "") != expected_slate_date.isoformat():
        raise ValueError("ARTIFACT_SLATE_DATE_MISMATCH")
    raw = payload.get("generated_at_utc")
    if not raw:
        raise ValueError("ARTIFACT_GENERATED_TIMESTAMP_MISSING")
    try:
        generated = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError as error:
        raise ValueError("ARTIFACT_GENERATED_TIMESTAMP_INVALID") from error
    current = now.astimezone(timezone.utc)
    age_hours = (current - generated).total_seconds() / 3600
    if age_hours < -0.25:
        raise ValueError("ARTIFACT_TIMESTAMP_IN_FUTURE")
    if age_hours > maximum_age_hours:
        raise ValueError("ARTIFACT_STALE")
