from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schemas import BetCandidate


DATA_CONTRACT_FIELDS = (
    "game_id", "player_id", "team", "opponent", "market_type", "side", "line",
    "sportsbook", "quoted_odds", "market_id", "selection_id", "odds_snapshot_time",
    "prediction_time", "raw_structural_probability", "calibrated_probability",
    "market_conditioned_probability", "usable_probability", "lineup_status",
    "player_status", "support_status", "identity_status",
)


def _utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class CanonicalCandidate:
    candidate_id: str
    slate_id: str
    game_id: str
    player_id: str
    team: str
    opponent: str
    lineup_status: str
    batting_order: int | None
    player_status: str
    role_status: str
    market_type: str
    side: str
    line: float | None
    sportsbook: str
    quoted_odds: float | None
    market_id: str | None
    selection_id: str | None
    odds_snapshot_time: str | None
    prediction_time: str
    game_start_time: str | None
    raw_structural_probability: float | None
    calibrated_probability: float | None
    market_conditioned_probability: float | None
    usable_probability: float | None
    calibration_bucket: str | None
    uncertainty_components: dict[str, float] | None
    support_score: int | None
    support_status: str
    ood_status: str
    identity_status: str
    settlement_identity: str | None
    market_implied_probability: float | None
    no_vig_market_probability: float | None
    expected_plate_appearances: float | None
    pa_probability_3_plus: float | None
    source_products: tuple[str, ...] = field(default_factory=tuple)

    @property
    def quote_age_seconds(self) -> float | None:
        quote, prediction = _utc(self.odds_snapshot_time), _utc(self.prediction_time)
        return (prediction - quote).total_seconds() if quote and prediction else None

    @property
    def missing_contract_fields(self) -> list[str]:
        row = self.to_dict()
        required = DATA_CONTRACT_FIELDS
        return [name for name in required if row.get(name) in (None, "", "UNKNOWN")]

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["source_products"] = list(self.source_products)
        row["quote_age_seconds"] = self.quote_age_seconds
        return row


def from_bet_candidate(candidate: BetCandidate, *, slate_id: str, prediction_time: str) -> CanonicalCandidate:
    """Lossless adapter. Facts resolved upstream are reused, never re-inferred here."""
    source = dict(candidate.source_payload or {})
    market_probability = candidate.no_vig_probability or candidate.market_break_even_probability
    return CanonicalCandidate(
        candidate_id=candidate.candidate_id, slate_id=slate_id, game_id=candidate.game_id,
        player_id=candidate.subject_id, team=candidate.team, opponent=candidate.opponent,
        lineup_status=str(candidate.lineup_status).upper(), batting_order=source.get("batting_order"),
        player_status=str(source.get("player_status") or "UNKNOWN").upper(),
        role_status=str(candidate.role_status).upper(), market_type=candidate.market_type,
        side=candidate.side, line=candidate.line, sportsbook=candidate.sportsbook,
        quoted_odds=candidate.american_price, market_id=candidate.sportsbook_market_id,
        selection_id=candidate.sportsbook_selection_id,
        odds_snapshot_time=source.get("odds_snapshot_time") or source.get("quote_timestamp") or source.get("selected_side_price_time"),
        prediction_time=prediction_time, game_start_time=source.get("game_start_time"),
        raw_structural_probability=candidate.structural_probability if candidate.structural_probability is not None else candidate.raw_probability,
        calibrated_probability=candidate.calibrated_probability,
        market_conditioned_probability=candidate.market_conditioned_probability,
        usable_probability=candidate.usable_probability,
        calibration_bucket=source.get("calibration_bucket"),
        uncertainty_components=source.get("uncertainty_components"),
        support_score=source.get("historical_bucket_support") or source.get("support_size"),
        support_status=candidate.support_status, ood_status=source.get("ood_status") or "UNMEASURED",
        identity_status=candidate.identity_status, settlement_identity=source.get("settlement_identity") or candidate.event_identity,
        market_implied_probability=market_probability, no_vig_market_probability=candidate.no_vig_probability,
        expected_plate_appearances=source.get("expected_plate_appearances"),
        pa_probability_3_plus=source.get("pa_probability_3_plus"),
        source_products=tuple(source.get("source_products") or [source.get("adapter") or "unknown"]),
    )


def terminal_decision(evaluated: list[dict[str, Any]]) -> str:
    if not evaluated:
        return "NO_FULLY_EVALUABLE_CANDIDATES"
    integrity = {
        "IDENTITY_INVALID", "LINEUP_INVALID", "PLAYER_STATUS_INVALID", "SUPPORT_INVALID",
        "OUT_OF_SUPPORT", "QUOTE_FRESHNESS_UNPROVABLE", "QUOTE_STALE",
        "EXACT_SELECTION_UNAVAILABLE", "PROBABILITY_UNAVAILABLE", "UNCERTAINTY_INVALID",
        "UNCERTAINTY_COMPONENTS_UNAVAILABLE", "UNCERTAINTY_COMPONENTS_INVALID", "PRICE_UNAVAILABLE",
    }
    fully_evaluable = [row for row in evaluated if not integrity.intersection(row.get("rejection_reasons") or [])]
    if not fully_evaluable:
        return "DATA_CONTRACT_INCOMPLETE"
    if any(row.get("final_selection_decision") for row in evaluated):
        return "CHALLENGER_SELECTIONS_AVAILABLE"
    return "NO_RELIABLE_EDGE_FOUND"
