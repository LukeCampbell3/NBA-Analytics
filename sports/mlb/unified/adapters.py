from __future__ import annotations

import hashlib
from urllib.parse import parse_qs, urlparse

from .decision import american_to_decimal
from .market_registry import CAPABILITIES
from .schemas import BetCandidate, EvidenceState

TARGET_MARKETS = {"H": "batter_hits", "TB": "batter_total_bases", "R": "batter_runs_scored", "RBI": "batter_rbis", "HR": "batter_home_runs", "K": "pitcher_strikeouts", "OUTS": "pitcher_outs"}


def _id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:24]


def _selection_ids(url: str | None) -> tuple[str | None, str | None]:
    if not url:
        return None, None
    query = parse_qs(urlparse(url).query)
    market = (query.get("marketId") or query.get("marketId[0]") or [None])[0]
    selection = (query.get("selectionId") or query.get("selectionId[0]") or [None])[0]
    return market, selection


def adapt_legacy_play(play: dict) -> BetCandidate:
    market = TARGET_MARKETS.get(str(play.get("target", "")).upper(), str(play.get("target", "")).lower())
    final_probability = play.get("final_hit_probability")
    price = play.get("selected_side_price", play.get("american_price"))
    deeplink = play.get("sportsbook_deeplink")
    market_id, selection_id = _selection_ids(deeplink)
    lineup = str(play.get("lineup_status") or "UNKNOWN").upper()
    role = "CONFIRMED" if (market != "pitcher_strikeouts" or bool(play.get("starter_confirmed"))) else "UNKNOWN"
    support = "SUPPORTED" if float(play.get("historical_bucket_support") or 0) >= 50 else "WEAK"
    identity = "CONFIRMED" if play.get("game_id") and (play.get("player_id") or play.get("player")) else "UNKNOWN"
    # final_hit_probability is already the legacy pipeline's conservative
    # post-calibration value. Preserve that semantic instead of subtracting a
    # second invented uncertainty haircut.
    uncertainty = 0.0 if final_probability is not None else None
    return BetCandidate(
        candidate_id=str(play.get("play_key") or _id(play.get("game_id"), play.get("player"), market, play.get("direction"), play.get("market_line"))),
        game_id=str(play.get("game_id") or ""), subject_type="pitcher" if market.startswith("pitcher_") else "player",
        subject_id=str(play.get("player_id") or play.get("player_mlbam_id") or play.get("player") or ""),
        team=str(play.get("team") or play.get("confirmed_team") or ""), opponent=str(play.get("opponent") or ""),
        market_type=market, period="game", event_identity=f"{play.get('game_id')}:{play.get('player_id') or play.get('player')}:game",
        side=str(play.get("direction") or "").lower(), line=float(play["market_line"]) if play.get("market_line") is not None else None,
        sportsbook=str(play.get("selected_sportsbook_key") or play.get("sportsbook") or ""),
        sportsbook_market_id=market_id, sportsbook_selection_id=selection_id,
        american_price=float(price) if price is not None else None, decimal_price=american_to_decimal(price),
        structural_probability=play.get("model_hit_probability"), market_conditioned_probability=None,
        raw_probability=play.get("estimated_hit_probability"), calibrated_probability=final_probability,
        uncertainty=uncertainty, usable_probability=None, support_status=support,
        lineup_status="CONFIRMED" if lineup == "CONFIRMED" else ("NOT_APPLICABLE" if market.startswith("pitcher_") else lineup),
        role_status=role if market.startswith("pitcher_") else "CONFIRMED", identity_status=identity,
        evidence_state=EvidenceState.PROSPECTIVE_SHADOW, publication_authority=False,
        source_payload={"adapter": "legacy_play", "probability_semantics": "already_conservative", "deeplink": deeplink},
    )


def adapt_team_leg(leg: dict, game: dict) -> BetCandidate:
    market = str(leg.get("market") or "")
    price = leg.get("price_american")
    market_id, selection_id = _selection_ids(leg.get("sportsbook_deeplink"))
    blockers = list(leg.get("support_blocking_dimensions") or [])
    capability = CAPABILITIES.get(market)
    support = "SUPPORTED" if leg.get("leg_authorized") and not blockers else "WEAK"
    probability = leg.get("model_probability")
    # Aggregate game legs have no universal uncertainty estimate yet. They
    # enter the pool but fail the primary gate until adapted calibration can
    # supply it.
    return BetCandidate(
        candidate_id=_id(game.get("game_id"), market, leg.get("side"), leg.get("line")),
        game_id=str(game.get("game_id") or ""), subject_type="team" if market == "moneyline" else "game",
        subject_id=str(game.get("home_team") if leg.get("side") == "home" else game.get("away_team") if leg.get("side") == "away" else game.get("game_id")),
        team=str(game.get("home_team") if leg.get("side") == "home" else game.get("away_team") if leg.get("side") == "away" else ""),
        opponent="", market_type=market, period="first_5" if market == "first_5_innings_total" else "game",
        event_identity=f"{game.get('game_id')}:{market}", side=str(leg.get("side") or ""),
        line=float(leg["line"]) if leg.get("line") is not None else None, sportsbook=str(leg.get("sportsbook") or ""),
        sportsbook_market_id=market_id, sportsbook_selection_id=selection_id,
        american_price=float(price) if price is not None else None, decimal_price=american_to_decimal(price),
        structural_probability=probability, market_conditioned_probability=None, raw_probability=probability,
        calibrated_probability=probability, uncertainty=None, usable_probability=None,
        support_status=support, lineup_status="NOT_APPLICABLE", role_status="NOT_APPLICABLE",
        identity_status="CONFIRMED" if game.get("game_id") and capability else "UNKNOWN",
        evidence_state=EvidenceState.PROSPECTIVE_SHADOW, publication_authority=False,
        source_payload={"adapter": "same_game_team_leg", "support_blockers": blockers},
    )


def adapt_pitcher_leg(leg: dict) -> BetCandidate:
    price = leg.get("price_american")
    market_id, selection_id = _selection_ids(leg.get("sportsbook_deeplink"))
    blockers = list(leg.get("support_blocking_dimensions") or [])
    probability = leg.get("model_probability")
    return BetCandidate(
        candidate_id=_id(leg.get("game_id"), leg.get("pitcher_id") or leg.get("pitcher_name"), "pitcher_strikeouts", leg.get("side"), leg.get("line")),
        game_id=str(leg.get("game_id") or ""), subject_type="pitcher",
        subject_id=str(leg.get("pitcher_id") or leg.get("pitcher_name") or ""), team=str(leg.get("team") or ""),
        opponent=str(leg.get("opponent") or ""), market_type="pitcher_strikeouts", period="game",
        event_identity=f"{leg.get('game_id')}:{leg.get('pitcher_id') or leg.get('pitcher_name')}:game",
        side=str(leg.get("side") or "").lower(), line=float(leg["line"]) if leg.get("line") is not None else None,
        sportsbook=str(leg.get("sportsbook") or ""), sportsbook_market_id=market_id, sportsbook_selection_id=selection_id,
        american_price=float(price) if price is not None else None, decimal_price=american_to_decimal(price),
        structural_probability=probability, market_conditioned_probability=None, raw_probability=probability,
        calibrated_probability=probability, uncertainty=None, usable_probability=None,
        support_status="SUPPORTED" if leg.get("leg_authorized") and not blockers else "WEAK",
        lineup_status="NOT_APPLICABLE", role_status="CONFIRMED" if leg.get("price_confirmed") else "UNKNOWN",
        identity_status="CONFIRMED" if leg.get("game_id") and (leg.get("pitcher_id") or leg.get("pitcher_name")) else "UNKNOWN",
        evidence_state=EvidenceState.PROSPECTIVE_SHADOW, publication_authority=False,
        source_payload={"adapter": "pitcher_parlay_leg", "support_blockers": blockers},
    )
