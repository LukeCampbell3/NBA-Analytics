from __future__ import annotations

from .schemas import CapabilityState, MarketCapability


CAPABILITIES = {
    "batter_hits": MarketCapability("batter_hits", CapabilityState.SUPPORTED, "legacy_player_projection", "H"),
    "batter_total_bases": MarketCapability("batter_total_bases", CapabilityState.SUPPORTED, "legacy_player_projection", "TB"),
    "batter_runs_scored": MarketCapability("batter_runs_scored", CapabilityState.SUPPORTED, "legacy_player_projection", "R"),
    "batter_rbis": MarketCapability("batter_rbis", CapabilityState.SUPPORTED, "legacy_player_projection", "RBI"),
    "batter_home_runs": MarketCapability("batter_home_runs", CapabilityState.SUPPORTED, "legacy_player_projection", "HR"),
    "pitcher_strikeouts": MarketCapability("pitcher_strikeouts", CapabilityState.SUPPORTED, "pitcher_strikeout_model", "K"),
    "pitcher_outs": MarketCapability("pitcher_outs", CapabilityState.SHADOW_ONLY, "pitcher_workload", "pitcher_outs", "Calibration/quote support is sparse"),
    "moneyline": MarketCapability("moneyline", CapabilityState.SHADOW_ONLY, "game_simulation_model", "moneyline"),
    "game_total": MarketCapability("game_total", CapabilityState.SHADOW_ONLY, "game_simulation_model", "game_total"),
    "first_5_innings_total": MarketCapability("first_5_innings_total", CapabilityState.SHADOW_ONLY, "game_simulation_model", "first_5_total"),
    "team_total": MarketCapability("team_total", CapabilityState.DISCOVERY, "team_run_distribution", "team_runs", "Exact two-sided quote support is incomplete"),
    "team_hits": MarketCapability("team_hits", CapabilityState.MODEL_REQUIRED, blocker="TEAM_HITS_GENERATIVE_MODEL_REQUIRED"),
    "runs_inning": MarketCapability("runs_inning", CapabilityState.EVENT_MODEL_REQUIRED, blocker="Pitch/play trajectory model unavailable", requires_event_identity=True),
    "team_runs_inning": MarketCapability("team_runs_inning", CapabilityState.EVENT_MODEL_REQUIRED, blocker="Pitch/play trajectory model unavailable", requires_event_identity=True),
    "pitcher_strikeouts_inning": MarketCapability("pitcher_strikeouts_inning", CapabilityState.EVENT_MODEL_REQUIRED, blocker="Inning K trajectory model unavailable", requires_event_identity=True),
    "pitcher_pitches_inning": MarketCapability("pitcher_pitches_inning", CapabilityState.EVENT_MODEL_REQUIRED, blocker="Inning pitch trajectory model unavailable", requires_event_identity=True),
    "plate_appearance_pitch_count": MarketCapability("plate_appearance_pitch_count", CapabilityState.EVENT_IDENTITY_UNAVAILABLE, blocker="Exact PA ordinal cannot be resolved", requires_event_identity=True),
}


def capability_payload() -> dict[str, dict]:
    return {
        name: {
            "status": item.status.value,
            "model": item.model,
            "settlement": item.settlement,
            "blocker": item.blocker,
            "requires_event_identity": item.requires_event_identity,
        }
        for name, item in sorted(CAPABILITIES.items())
    }
