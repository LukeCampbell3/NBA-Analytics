from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .game_conditioned_authority import validate_target_authority
from .schema import AdvancedCandidateContext
from .sequential_pa_model import SequentialPAResult

MODEL_VERSION = "game_conditioned_hitter_moe_v2"
SCHEMA_VERSION = "mlb_game_conditioned_hitter_moe_v2"
TARGETS = ("H", "TB", "HR")
EXPERT_NAMES = (
    "strikeout_contact",
    "contact_quality",
    "power_tb",
    "defense_conversion",
    "pa_opportunity",
    "bullpen_transition",
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = REPO_ROOT / "sports" / "mlb" / "data" / "predictions" / "calibration" / "game_conditioned_hitter_moe_v2.json"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _finite(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _optional_finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def logit(probability: float) -> float:
    p = _clamp(float(probability), 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def logistic(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _direct_matchup_state(context: AdvancedCandidateContext) -> dict[str, float]:
    """Build heavily-shrunk direct BvP process signals.

    Direct batter/pitcher history is intentionally never allowed to dominate a
    game state.  The stored shrinkage weight is multiplied by a small-sample
    support term and capped at 0.30 before it can affect any expert.
    """

    direct = context.direct_matchup
    if direct is None or int(direct.pa or 0) <= 0:
        return {
            "weight": 0.0,
            "strikeout_contact": 0.0,
            "contact_quality": 0.0,
            "power_tb": 0.0,
        }

    pa = max(1, int(direct.pa or 0))
    stored_weight = _clamp(_finite(direct.shrinkage_weight, 0.0), 0.0, 1.0)
    sample_support = _clamp(pa / 24.0, 0.0, 1.0)
    weight = min(0.30, stored_weight * sample_support)

    strikeout_terms = [(0.225 - (float(direct.strikeouts or 0) / pa)) / 0.12]
    direct_whiff = _optional_finite(direct.whiff_rate)
    if direct_whiff is not None:
        strikeout_terms.append((0.235 - direct_whiff) / 0.12)

    contact_terms: list[float] = []
    direct_xwoba = _optional_finite(direct.xwoba_contact)
    direct_xba = _optional_finite(direct.xba_contact)
    direct_ev = _optional_finite(direct.avg_ev)
    if direct_xwoba is not None:
        contact_terms.append((direct_xwoba - 0.320) / 0.075)
    if direct_xba is not None:
        contact_terms.append((direct_xba - 0.250) / 0.055)
    if direct_ev is not None:
        contact_terms.append((direct_ev - 88.5) / 5.5)

    power_terms: list[float] = []
    direct_xslg = _optional_finite(direct.xslg_contact)
    direct_barrel = _optional_finite(direct.barrel_rate)
    if direct_xslg is not None:
        power_terms.append((direct_xslg - 0.420) / 0.17)
    if direct_barrel is not None:
        power_terms.append((direct_barrel - 0.075) / 0.075)
    power_terms.append(((float(direct.home_runs or 0) / pa) - 0.030) / 0.035)

    return {
        "weight": weight,
        "strikeout_contact": _clamp(_mean(strikeout_terms), -2.0, 2.0),
        "contact_quality": _clamp(_mean(contact_terms), -2.0, 2.0),
        "power_tb": _clamp(_mean(power_terms), -2.0, 2.0),
    }


_ROLLING_ALIASES = {
    "xwoba": "xwoba_contact",
    "xba": "xba_contact",
    "xslg": "xslg_contact",
    "hard_hit_rate": "hard_hit_rate",
    "barrel_rate": "barrel_rate",
    "k_rate": "k_rate",
    "bb_rate": "bb_rate",
    "hr_rate": "hr_rate",
    "whiff_rate": "whiff_rate",
}


def _profile_recent(profile: Any, key: str, default: float) -> float:
    """Build a support-weighted recent-state estimate from overlapping windows.

    The rolling windows are deliberately anchored to season skill. They can
    change today's latent state when multiple windows agree, but no short sample
    can replace the global prior by itself.
    """

    rolling = getattr(profile, "rolling", None) or {}
    if not isinstance(rolling, Mapping):
        return float(default)
    metric_key = _ROLLING_ALIASES.get(key, key)
    windows = (("last_15", 0.45, 15.0), ("last_30", 0.32, 30.0), ("last_60", 0.23, 60.0))
    weighted = 0.0
    weight_sum = 0.0
    for window, base_weight, expected_pa in windows:
        data = rolling.get(window)
        if not isinstance(data, Mapping):
            continue
        try:
            parsed = float(data.get(metric_key))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(parsed):
            continue
        sample_pa = _finite(data.get("pa"), 0.0)
        support = _clamp(sample_pa / expected_pa, 0.0, 1.0)
        weight = base_weight * support
        weighted += weight * parsed
        weight_sum += weight
    if weight_sum <= 1e-9:
        return float(default)
    recent = weighted / weight_sum
    recent_authority = _clamp(0.35 + 0.30 * weight_sum, 0.35, 0.65)
    return (1.0 - recent_authority) * float(default) + recent_authority * recent


@dataclass(frozen=True)
class ExpertState:
    signals: dict[str, float]
    activations: dict[str, float]
    effective_features: dict[str, float]
    evidence_strength: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class GameConditionedProbability:
    model_version: str
    target: str
    prior_probability: float
    candidate_probability: float
    production_probability: float
    lower_bound_probability: float
    residual_logit: float
    residual_before_shrinkage: float
    evidence_strength: float
    positive_authority: bool
    authority_status: str
    expert_weights: dict[str, float]
    expert_contributions: dict[str, float]
    expert_signals: dict[str, float]
    expert_activations: dict[str, float]
    model_source: str
    validation: dict[str, Any]


def _empty_target() -> dict[str, Any]:
    return {
        "intercept": 0.0,
        "coefficients": {name: 0.0 for name in EXPERT_NAMES},
        "feature_means": {name: 0.0 for name in EXPERT_NAMES},
        "feature_scales": {name: 1.0 for name in EXPERT_NAMES},
        "positive_authority": False,
        "validation": {"status": "UNVALIDATED", "statistical_gate_passed": False},
    }


DEFAULT_ARTIFACT: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "model_version": MODEL_VERSION,
    "training_status": "UNFITTED_SAFE_PRIOR_ONLY",
    "evidence_class": "NONE",
    "max_abs_residual_logit": 0.35,
    "targets": {target: _empty_target() for target in TARGETS},
}


def load_model_artifact(path: Path | None = None) -> dict[str, Any]:
    model_path = path or DEFAULT_MODEL_PATH
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(json.dumps(DEFAULT_ARTIFACT))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        return json.loads(json.dumps(DEFAULT_ARTIFACT))
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        return json.loads(json.dumps(DEFAULT_ARTIFACT))

    # Backward-compatible migration: an H/TB artifact remains usable while the
    # first HR-aware validation fit is being generated. Missing targets are
    # neutral and therefore cannot acquire production authority.
    migrated = json.loads(json.dumps(payload))
    migrated_targets = migrated.setdefault("targets", {})
    for target in TARGETS:
        if not isinstance(migrated_targets.get(target), dict):
            migrated_targets[target] = _empty_target()
    return migrated


def build_expert_state(
    context: AdvancedCandidateContext,
    sequential: SequentialPAResult,
    *,
    target: str,
    pitch_compatibility_score: float = 0.0,
) -> ExpertState:
    """Build game-specific expert signals and relevance gates."""

    target = str(target).upper()
    if target not in TARGETS:
        raise ValueError(f"unsupported game-conditioned target: {target}")

    batter = context.batter
    pitcher = context.pitcher
    direct_state = _direct_matchup_state(context)
    direct_weight = direct_state["weight"]

    batter_k = _finite(batter.k_rate, 0.225)
    pitcher_k = _finite(pitcher.k_rate, 0.225)
    batter_whiff = _finite(batter.whiff_rate, 0.235)
    pitcher_whiff = _finite(pitcher.whiff_rate, 0.235)
    batter_chase = _finite(batter.chase_rate, 0.285)
    pitcher_kbb = _finite(pitcher.k_minus_bb_rate, pitcher_k - _finite(pitcher.bb_rate, 0.085))

    batter_recent_k = _profile_recent(batter, "k_rate", batter_k)
    pitcher_recent_k = _profile_recent(pitcher, "k_rate", pitcher_k)
    batter_recent_whiff = _profile_recent(batter, "whiff_rate", batter_whiff)
    pitcher_recent_whiff = _profile_recent(pitcher, "whiff_rate", pitcher_whiff)

    strikeout_contact = _mean([
        (0.225 - batter_k) / 0.10,
        (0.225 - pitcher_k) / 0.10,
        (0.235 - batter_whiff) / 0.12,
        (0.235 - pitcher_whiff) / 0.12,
        (0.285 - batter_chase) / 0.11,
        (0.140 - pitcher_kbb) / 0.12,
        (batter_k - batter_recent_k) / 0.08,
        (pitcher_k - pitcher_recent_k) / 0.08,
        (batter_whiff - batter_recent_whiff) / 0.09,
        (pitcher_whiff - pitcher_recent_whiff) / 0.09,
    ])
    strikeout_contact += direct_weight * direct_state["strikeout_contact"]

    batter_xwoba = _finite(batter.xwoba, _finite(batter.woba, 0.320))
    pitcher_xwoba = _finite(pitcher.xwoba_allowed, 0.320)
    batter_xba = _finite(batter.xba, 0.250)
    pitcher_xba = _finite(pitcher.xba_allowed, 0.250)
    batter_hard = _finite(batter.hard_hit_rate, 0.38)
    pitcher_hard = _finite(pitcher.hard_hit_rate_allowed, 0.38)
    batter_avg_ev = _finite(batter.avg_ev, 88.5)
    pitcher_avg_ev = _finite(pitcher.avg_ev_allowed, 88.5)
    batter_ev90 = _finite(batter.ev90, 104.0)
    batter_sweet = _finite(batter.sweet_spot_rate, 0.33)
    pitcher_sweet = _finite(pitcher.sweet_spot_rate_allowed, 0.33)

    batter_recent_xwoba = _profile_recent(batter, "xwoba", batter_xwoba)
    pitcher_recent_xwoba = _profile_recent(pitcher, "xwoba", pitcher_xwoba)
    batter_recent_hard = _profile_recent(batter, "hard_hit_rate", batter_hard)
    pitcher_recent_hard = _profile_recent(pitcher, "hard_hit_rate", pitcher_hard)

    batter_quality_trend = _mean([
        (batter_recent_xwoba - batter_xwoba) / 0.060,
        (batter_recent_hard - batter_hard) / 0.10,
    ])
    pitcher_quality_trend_for_hitter = _mean([
        (pitcher_recent_xwoba - pitcher_xwoba) / 0.060,
        (pitcher_recent_hard - pitcher_hard) / 0.10,
    ])

    contact_quality = _mean([
        (batter_xwoba - 0.320) / 0.075,
        (pitcher_xwoba - 0.320) / 0.075,
        (batter_xba - 0.250) / 0.055,
        (pitcher_xba - 0.250) / 0.055,
        (batter_hard - 0.38) / 0.13,
        (pitcher_hard - 0.38) / 0.13,
        (batter_avg_ev - 88.5) / 5.5,
        (pitcher_avg_ev - 88.5) / 5.5,
        (batter_ev90 - 104.0) / 7.0,
        (batter_sweet - 0.33) / 0.12,
        (pitcher_sweet - 0.33) / 0.12,
        _clamp(pitch_compatibility_score, -1.5, 1.5),
        _clamp(batter_quality_trend, -1.5, 1.5),
        _clamp(pitcher_quality_trend_for_hitter, -1.5, 1.5),
    ])
    contact_quality += direct_weight * direct_state["contact_quality"]

    batter_xslg = _finite(batter.xslg, 0.420)
    pitcher_xslg = _finite(pitcher.xslg_allowed, 0.420)
    batter_barrel = _finite(batter.barrel_rate, 0.075)
    pitcher_barrel = _finite(pitcher.barrel_rate_allowed, 0.075)
    pitcher_gb = _finite(pitcher.gb_rate, 0.43)
    batter_hr = _finite(batter.hr_rate, 0.030)
    pitcher_hr = _finite(pitcher.hr_rate, 0.030)

    batter_recent_xslg = _profile_recent(batter, "xslg", batter_xslg)
    pitcher_recent_xslg = _profile_recent(pitcher, "xslg", pitcher_xslg)
    batter_recent_barrel = _profile_recent(batter, "barrel_rate", batter_barrel)
    pitcher_recent_barrel = _profile_recent(pitcher, "barrel_rate", pitcher_barrel)
    batter_recent_hr = _profile_recent(batter, "hr_rate", batter_hr)
    pitcher_recent_hr = _profile_recent(pitcher, "hr_rate", pitcher_hr)

    power_trend = _mean([
        (batter_recent_xslg - batter_xslg) / 0.14,
        (pitcher_recent_xslg - pitcher_xslg) / 0.14,
        (batter_recent_barrel - batter_barrel) / 0.06,
        (pitcher_recent_barrel - pitcher_barrel) / 0.06,
        (batter_recent_hr - batter_hr) / 0.025,
        (pitcher_recent_hr - pitcher_hr) / 0.025,
    ])

    temperature_f = context.temperature_f
    weather_power_signal = 0.0 if temperature_f is None else _clamp((float(temperature_f) - 72.0) / 30.0, -1.0, 1.0)
    power_tb = _mean([
        (batter_xslg - 0.420) / 0.17,
        (pitcher_xslg - 0.420) / 0.17,
        (batter_barrel - 0.075) / 0.075,
        (pitcher_barrel - 0.075) / 0.075,
        (batter_hr - 0.030) / 0.035,
        (pitcher_hr - 0.030) / 0.035,
        (batter_avg_ev - 88.5) / 5.5,
        (pitcher_avg_ev - 88.5) / 5.5,
        (batter_ev90 - 104.0) / 7.0,
        (batter_sweet - 0.33) / 0.12,
        (0.43 - pitcher_gb) / 0.18,
        (float(context.park_factor or 1.0) - 1.0) / 0.10,
        _clamp(power_trend, -1.5, 1.5),
        0.35 * weather_power_signal,
    ])
    power_tb += direct_weight * direct_state["power_tb"]

    specific_defense = str(context.defense_status or "").upper().startswith("SPECIFIC")
    defense_conversion = (
        _clamp(float(context.defense_residual or 0.0) / 0.035, -2.0, 2.0)
        if specific_defense and target != "HR"
        else 0.0
    )

    expected_pa = _finite(sequential.expected_pa, 4.2)
    team_runs = _finite(context.team_expected_runs, 4.5)
    batting_order = int(context.batting_order or 6)
    order_signal = (5.0 - min(9, max(1, batting_order))) / 4.0
    pa_opportunity = _mean([
        (expected_pa - 4.2) / 0.65,
        (team_runs - 4.5) / 1.5,
        order_signal,
        -0.18 if context.is_home else 0.08,
    ])

    projected_ip = _finite(pitcher.projected_ip, 5.5)
    starter_fraction = _clamp(projected_ip / 7.0, 0.35, 0.90)
    starter_quality = _mean([
        (3.95 - _finite(pitcher.xfip, _finite(pitcher.era, 4.10))) / 1.35,
        (3.90 - _finite(pitcher.siera, _finite(pitcher.era, 4.10))) / 1.35,
        (0.320 - pitcher_xwoba) / 0.075,
    ])
    bullpen_transition = starter_quality * (1.0 - starter_fraction) * max(0.6, expected_pa / 4.2)

    signals = {
        "strikeout_contact": _clamp(strikeout_contact, -2.5, 2.5),
        "contact_quality": _clamp(contact_quality, -2.5, 2.5),
        "power_tb": _clamp(power_tb, -2.5, 2.5),
        "defense_conversion": _clamp(defense_conversion, -2.5, 2.5),
        "pa_opportunity": _clamp(pa_opportunity, -2.5, 2.5),
        "bullpen_transition": _clamp(bullpen_transition, -2.5, 2.5),
    }

    high_k_relevance = _clamp(
        abs(pitcher_recent_k - 0.225) / 0.12 + abs(batter_recent_k - 0.225) / 0.12,
        0.0,
        1.6,
    )
    low_k_contact_relevance = _clamp((0.255 - pitcher_recent_k) / 0.10, 0.0, 1.4)
    coherent_batter_form = _clamp(max(0.0, batter_quality_trend) + max(0.0, power_trend), 0.0, 1.5)
    coherent_pitcher_decline = _clamp(
        max(0.0, pitcher_quality_trend_for_hitter) + max(0.0, pitcher_k - pitcher_recent_k) / 0.08,
        0.0,
        1.5,
    )

    power_base = {"H": 0.72, "TB": 1.35, "HR": 1.62}[target]
    contact_base = {"H": 1.00, "TB": 0.88, "HR": 0.72}[target]
    defense_base = 0.10 if target == "HR" else (1.20 if specific_defense else 0.25)

    activations = {
        "strikeout_contact": _clamp(
            0.85 + 0.55 * high_k_relevance + 0.08 * abs(pitcher_k - pitcher_recent_k) / 0.08 + 0.10 * direct_weight,
            0.55,
            1.75,
        ),
        "contact_quality": _clamp(
            contact_base
            + 0.35 * low_k_contact_relevance
            + 0.15 * abs(pitch_compatibility_score)
            + 0.10 * coherent_batter_form
            + 0.10 * coherent_pitcher_decline
            + 0.10 * direct_weight,
            0.50,
            1.70,
        ),
        "power_tb": _clamp(power_base + 0.18 * abs(power_tb) + 0.08 * coherent_batter_form + 0.10 * direct_weight, 0.55, 1.90),
        "defense_conversion": defense_base,
        "pa_opportunity": _clamp(0.90 + 0.18 * abs(pa_opportunity), 0.70, 1.45),
        "bullpen_transition": _clamp(0.65 + 0.65 * (1.0 - starter_fraction), 0.55, 1.35),
    }
    effective = {name: signals[name] * activations[name] for name in EXPERT_NAMES}

    profile_support = _mean([
        _clamp01(_finite(batter.support, 0.0)),
        _clamp01(_finite(pitcher.support, 0.0)),
        _clamp01(_finite(sequential.support, 0.0)),
    ])
    missing_penalty = min(0.45, 0.055 * len(context.missing_components or ()))
    freshness = 1.0 if context.data_freshness_status == "FRESH" else 0.72 if context.data_freshness_status == "DEGRADED" else 0.35
    evidence_strength = _clamp01(
        profile_support
        * freshness
        * (1.0 - missing_penalty)
        * (1.0 - 0.55 * _clamp01(sequential.uncertainty))
    )

    batter_hand = str(batter.handedness or "").strip().upper()
    pitcher_hand = str(pitcher.handedness or "").strip().upper()
    handedness_available = bool(batter_hand and pitcher_hand)
    handedness_matchup = f"{batter_hand}_VS_{pitcher_hand}" if handedness_available else "UNKNOWN"

    diagnostics = {
        "target": target,
        "batter_k_rate": batter_k,
        "batter_recent_k_rate": batter_recent_k,
        "pitcher_k_rate": pitcher_k,
        "pitcher_recent_k_rate": pitcher_recent_k,
        "batter_whiff_rate": batter_whiff,
        "batter_recent_whiff_rate": batter_recent_whiff,
        "pitcher_whiff_rate": pitcher_whiff,
        "pitcher_recent_whiff_rate": pitcher_recent_whiff,
        "batter_chase_rate": batter_chase,
        "batter_xwoba": batter_xwoba,
        "batter_recent_xwoba": batter_recent_xwoba,
        "pitcher_xwoba_allowed": pitcher_xwoba,
        "pitcher_recent_xwoba_allowed": pitcher_recent_xwoba,
        "batter_avg_ev": batter_avg_ev,
        "pitcher_avg_ev_allowed": pitcher_avg_ev,
        "batter_ev90": batter_ev90,
        "batter_sweet_spot_rate": batter_sweet,
        "pitcher_sweet_spot_rate_allowed": pitcher_sweet,
        "batter_hr_rate": batter_hr,
        "batter_recent_hr_rate": batter_recent_hr,
        "pitcher_hr_rate_allowed_proxy": pitcher_hr,
        "pitcher_recent_hr_rate_allowed_proxy": pitcher_recent_hr,
        "batter_quality_trend": batter_quality_trend,
        "pitcher_quality_trend_for_hitter": pitcher_quality_trend_for_hitter,
        "power_trend": power_trend,
        "pitcher_xfip": pitcher.xfip,
        "pitcher_siera": pitcher.siera,
        "pitch_compatibility_score": pitch_compatibility_score,
        "pitch_context_available": bool(batter.pitch_type_xwoba and pitcher.arsenal),
        "direct_matchup_pa": int(context.direct_matchup.pa) if context.direct_matchup is not None else 0,
        "direct_matchup_weight": direct_weight,
        "direct_matchup_strikeout_contact_signal": direct_state["strikeout_contact"],
        "direct_matchup_contact_quality_signal": direct_state["contact_quality"],
        "direct_matchup_power_signal": direct_state["power_tb"],
        "batter_handedness": batter_hand,
        "pitcher_handedness": pitcher_hand,
        "handedness_context_available": handedness_available,
        "handedness_matchup": handedness_matchup,
        "temperature_f": temperature_f,
        "weather_power_signal": weather_power_signal,
        "expected_pa": expected_pa,
        "team_expected_runs": context.team_expected_runs,
        "specific_defense": specific_defense,
        "projected_starter_ip": projected_ip,
        "starter_fraction": starter_fraction,
        "profile_support": profile_support,
        "missing_component_count": len(context.missing_components or ()),
    }
    return ExpertState(
        signals=signals,
        activations=activations,
        effective_features=effective,
        evidence_strength=evidence_strength,
        diagnostics=diagnostics,
    )


def _target_payload(artifact: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    targets = artifact.get("targets") if isinstance(artifact, Mapping) else None
    payload = (targets or {}).get(target) if isinstance(targets, Mapping) else None
    return payload if isinstance(payload, Mapping) else DEFAULT_ARTIFACT["targets"][target]


def condition_probability(
    prior_probability: float,
    *,
    target: str,
    state: ExpertState,
    artifact: Mapping[str, Any] | None = None,
    sequential_uncertainty: float = 0.0,
) -> GameConditionedProbability:
    target = str(target).upper()
    if target not in TARGETS:
        raise ValueError(f"unsupported game-conditioned target: {target}")

    model = artifact if isinstance(artifact, Mapping) else DEFAULT_ARTIFACT
    payload = _target_payload(model, target)
    coefficients = payload.get("coefficients") or {}
    means = payload.get("feature_means") or {}
    scales = payload.get("feature_scales") or {}
    intercept = _finite(payload.get("intercept"), 0.0)

    contributions: dict[str, float] = {}
    raw_residual = intercept
    for name in EXPERT_NAMES:
        feature = _finite(state.effective_features.get(name), 0.0)
        mean = _finite(means.get(name), 0.0)
        scale = max(1e-6, abs(_finite(scales.get(name), 1.0)))
        standardized = (feature - mean) / scale
        contribution = _finite(coefficients.get(name), 0.0) * standardized
        contributions[name] = contribution
        raw_residual += contribution

    max_abs = _clamp(abs(_finite(model.get("max_abs_residual_logit"), 0.35)), 0.05, 0.75)
    evidence_strength = _clamp01(state.evidence_strength)
    residual = _clamp(raw_residual * evidence_strength, -max_abs, max_abs)
    prior = _clamp(float(prior_probability), 1e-5, 1.0 - 1e-5)
    candidate = logistic(logit(prior) + residual)

    validation = dict(payload.get("validation") or {})
    authority = validate_target_authority(model, target)
    validation["independent_authority_audit"] = authority.to_dict()
    positive_authority = authority.positive_authority_allowed
    negative_authority = authority.negative_authority_allowed

    uncertainty = _clamp01(float(sequential_uncertainty))
    validation_brier = _finite(validation.get("candidate_brier"), _finite(validation.get("prior_brier"), 0.25))
    prior_brier = _finite(validation.get("prior_brier"), validation_brier)
    calibration_risk = _clamp(max(0.0, validation_brier - prior_brier), 0.0, 0.08)
    probability_haircut = min(0.10, 0.035 * uncertainty + calibration_risk)
    lower_bound = _clamp01(candidate - probability_haircut)

    if positive_authority:
        production = lower_bound
        authority_status = "PROMOTED_RESIDUAL_POSITIVE_AND_NEGATIVE_AUTHORITY"
    elif negative_authority:
        production = min(prior, lower_bound)
        authority_status = "INDEPENDENTLY_VALIDATED_NEGATIVE_AUTHORITY_ONLY"
    else:
        production = prior
        authority_status = "SHADOW_ONLY_NO_PRODUCTION_AUTHORITY"

    absolute = {name: abs(contributions[name]) for name in EXPERT_NAMES}
    total = sum(absolute.values())
    if total <= 1e-12:
        activation_total = sum(abs(state.activations[name]) for name in EXPERT_NAMES) or 1.0
        weights = {name: abs(state.activations[name]) / activation_total for name in EXPERT_NAMES}
    else:
        weights = {name: absolute[name] / total for name in EXPERT_NAMES}

    return GameConditionedProbability(
        model_version=MODEL_VERSION,
        target=target,
        prior_probability=prior,
        candidate_probability=candidate,
        production_probability=production,
        lower_bound_probability=lower_bound,
        residual_logit=residual,
        residual_before_shrinkage=raw_residual,
        evidence_strength=evidence_strength,
        positive_authority=positive_authority,
        authority_status=authority_status,
        expert_weights=weights,
        expert_contributions=contributions,
        expert_signals=dict(state.signals),
        expert_activations=dict(state.activations),
        model_source=str(model.get("training_status") or "UNFITTED_SAFE_PRIOR_ONLY"),
        validation=validation,
    )


def american_implied_probability(price: Any) -> float | None:
    try:
        odds = float(price)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(odds) or abs(odds) < 100.0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)


def no_vig_over_probability(over_price: Any, under_price: Any) -> float | None:
    over = american_implied_probability(over_price)
    under = american_implied_probability(under_price)
    if over is None or under is None or over + under <= 0:
        return None
    return over / (over + under)


def poisson_over_probability(mean: float, line: float) -> float:
    threshold = math.floor(float(line)) + 1
    lam = max(0.0, float(mean))
    term = math.exp(-lam)
    cdf = term
    for k in range(1, threshold):
        term *= lam / k
        cdf += term
    return _clamp01(1.0 - cdf)


def choose_prior_probability(
    *,
    legacy_projection: float,
    market_line: float,
    over_price: Any = None,
    under_price: Any = None,
    legacy_weight: float = 0.72,
) -> tuple[float, dict[str, Any]]:
    """Build the pre-residual prior from legacy structure plus no-vig market."""

    legacy = poisson_over_probability(legacy_projection, market_line)
    market = no_vig_over_probability(over_price, under_price)
    if market is None:
        return legacy, {
            "legacy_probability": legacy,
            "market_no_vig_probability": None,
            "legacy_weight": 1.0,
        }
    weight = _clamp(float(legacy_weight), 0.20, 0.95)
    prior = logistic(weight * logit(legacy) + (1.0 - weight) * logit(market))
    return prior, {
        "legacy_probability": legacy,
        "market_no_vig_probability": market,
        "legacy_weight": weight,
    }
