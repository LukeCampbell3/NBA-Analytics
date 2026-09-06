from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .schema import AdvancedCandidateContext
from .sequential_pa_model import SequentialPAResult

MODEL_VERSION = "game_conditioned_hitter_moe_v2"
SCHEMA_VERSION = "mlb_game_conditioned_hitter_moe_v2"
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
    """Build a support-weighted recent-state estimate from nested Statcast windows.

    The short windows intentionally never receive full authority on their own.
    last-15/30/60 are overlapping views, so their weights are capped and then
    blended back toward the season/default value. This supplies state movement
    without turning one hot week into a new player skill estimate.
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
        value = data.get(metric_key)
        try:
            parsed = float(value)
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
    # Season skill remains a material anchor even when every rolling window is full.
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


DEFAULT_ARTIFACT: dict[str, Any] = {
    "schema_version": SCHEMA_VERSION,
    "model_version": MODEL_VERSION,
    "training_status": "UNFITTED_SAFE_PRIOR_ONLY",
    "evidence_class": "NONE",
    "max_abs_residual_logit": 0.35,
    "targets": {
        "H": {
            "intercept": 0.0,
            "coefficients": {name: 0.0 for name in EXPERT_NAMES},
            "feature_means": {name: 0.0 for name in EXPERT_NAMES},
            "feature_scales": {name: 1.0 for name in EXPERT_NAMES},
            "positive_authority": False,
            "validation": {"status": "UNVALIDATED"},
        },
        "TB": {
            "intercept": 0.0,
            "coefficients": {name: 0.0 for name in EXPERT_NAMES},
            "feature_means": {name: 0.0 for name in EXPERT_NAMES},
            "feature_scales": {name: 1.0 for name in EXPERT_NAMES},
            "positive_authority": False,
            "validation": {"status": "UNVALIDATED"},
        },
    },
}


def load_model_artifact(path: Path | None = None) -> dict[str, Any]:
    model_path = path or DEFAULT_MODEL_PATH
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except Exception:
        return json.loads(json.dumps(DEFAULT_ARTIFACT))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        return json.loads(json.dumps(DEFAULT_ARTIFACT))
    targets = payload.get("targets") or {}
    if not all(target in targets for target in ("H", "TB")):
        return json.loads(json.dumps(DEFAULT_ARTIFACT))
    return payload


def build_expert_state(
    context: AdvancedCandidateContext,
    sequential: SequentialPAResult,
    *,
    target: str,
    pitch_compatibility_score: float = 0.0,
) -> ExpertState:
    """Build game-specific expert signals and relevance gates.

    Signals are centered roughly around league-neutral state. Activations encode
    how relevant an expert is for this matchup. The fitted model learns global
    coefficients; effective per-game weights are coefficient x activation.
    """

    target = str(target).upper()
    batter = context.batter
    pitcher = context.pitcher

    batter_k = _finite(batter.k_rate, 0.225)
    pitcher_k = _finite(pitcher.k_rate, 0.225)
    batter_whiff = _finite(batter.whiff_rate, 0.235)
    pitcher_whiff = _finite(pitcher.whiff_rate, 0.235)
    pitcher_kbb = _finite(pitcher.k_minus_bb_rate, pitcher_k - _finite(pitcher.bb_rate, 0.085))

    batter_recent_k = _profile_recent(batter, "k_rate", batter_k)
    pitcher_recent_k = _profile_recent(pitcher, "k_rate", pitcher_k)
    batter_recent_whiff = _profile_recent(batter, "whiff_rate", batter_whiff)
    pitcher_recent_whiff = _profile_recent(pitcher, "whiff_rate", pitcher_whiff)

    # Positive means contact survival conditions favor the hitter. The trend
    # terms answer whether today's state is moving away from the season prior.
    strikeout_contact = _mean([
        (0.225 - batter_k) / 0.10,
        (0.225 - pitcher_k) / 0.10,
        (0.235 - batter_whiff) / 0.12,
        (0.235 - pitcher_whiff) / 0.12,
        (0.140 - pitcher_kbb) / 0.12,
        (batter_k - batter_recent_k) / 0.08,
        (pitcher_k - pitcher_recent_k) / 0.08,
        (batter_whiff - batter_recent_whiff) / 0.09,
        (pitcher_whiff - pitcher_recent_whiff) / 0.09,
    ])

    batter_xwoba = _finite(batter.xwoba, _finite(batter.woba, 0.320))
    pitcher_xwoba = _finite(pitcher.xwoba_allowed, 0.320)
    batter_xba = _finite(batter.xba, 0.250)
    pitcher_xba = _finite(pitcher.xba_allowed, 0.250)
    batter_hard = _finite(batter.hard_hit_rate, 0.38)
    pitcher_hard = _finite(pitcher.hard_hit_rate_allowed, 0.38)
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
        _clamp(pitch_compatibility_score, -1.5, 1.5),
        _clamp(batter_quality_trend, -1.5, 1.5),
        _clamp(pitcher_quality_trend_for_hitter, -1.5, 1.5),
    ])

    batter_xslg = _finite(batter.xslg, 0.420)
    pitcher_xslg = _finite(pitcher.xslg_allowed, 0.420)
    batter_barrel = _finite(batter.barrel_rate, 0.075)
    pitcher_barrel = _finite(pitcher.barrel_rate_allowed, 0.075)
    pitcher_gb = _finite(pitcher.gb_rate, 0.43)
    batter_recent_xslg = _profile_recent(batter, "xslg", batter_xslg)
    pitcher_recent_xslg = _profile_recent(pitcher, "xslg", pitcher_xslg)
    batter_recent_barrel = _profile_recent(batter, "barrel_rate", batter_barrel)
    pitcher_recent_barrel = _profile_recent(pitcher, "barrel_rate", pitcher_barrel)
    power_trend = _mean([
        (batter_recent_xslg - batter_xslg) / 0.14,
        (pitcher_recent_xslg - pitcher_xslg) / 0.14,
        (batter_recent_barrel - batter_barrel) / 0.06,
        (pitcher_recent_barrel - pitcher_barrel) / 0.06,
    ])
    temperature_f = context.temperature_f
    weather_power_signal = 0.0 if temperature_f is None else _clamp((float(temperature_f) - 72.0) / 30.0, -1.0, 1.0)
    power_tb = _mean([
        (batter_xslg - 0.420) / 0.17,
        (pitcher_xslg - 0.420) / 0.17,
        (batter_barrel - 0.075) / 0.075,
        (pitcher_barrel - 0.075) / 0.075,
        (0.43 - pitcher_gb) / 0.18,
        (float(context.park_factor or 1.0) - 1.0) / 0.10,
        _clamp(power_trend, -1.5, 1.5),
        0.35 * weather_power_signal,
    ])

    specific_defense = str(context.defense_status or "").upper().startswith("SPECIFIC")
    defense_conversion = _clamp(float(context.defense_residual or 0.0) / 0.035, -2.0, 2.0) if specific_defense else 0.0

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

    high_k_relevance = _clamp(abs(pitcher_recent_k - 0.225) / 0.12 + abs(batter_recent_k - 0.225) / 0.12, 0.0, 1.6)
    low_k_contact_relevance = _clamp((0.255 - pitcher_recent_k) / 0.10, 0.0, 1.4)
    coherent_batter_form = _clamp(max(0.0, batter_quality_trend) + max(0.0, power_trend), 0.0, 1.5)
    coherent_pitcher_decline = _clamp(max(0.0, pitcher_quality_trend_for_hitter) + max(0.0, pitcher_k - pitcher_recent_k) / 0.08, 0.0, 1.5)
    pitch_context_available = bool(batter.pitch_type_xwoba and pitcher.arsenal)
    handedness_known = bool(str(batter.handedness or "").strip() and str(pitcher.handedness or "").strip())

    activations = {
        "strikeout_contact": _clamp(0.85 + 0.55 * high_k_relevance + 0.08 * abs(pitcher_k - pitcher_recent_k) / 0.08, 0.55, 1.75),
        "contact_quality": _clamp(0.85 + 0.35 * low_k_contact_relevance + 0.15 * abs(pitch_compatibility_score) + 0.10 * coherent_batter_form + 0.10 * coherent_pitcher_decline, 0.60, 1.70),
        "power_tb": _clamp((1.35 if target == "TB" else 0.72) + 0.18 * abs(power_tb) + 0.08 * coherent_batter_form, 0.55, 1.80),
        "defense_conversion": 1.20 if specific_defense else 0.25,
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
    evidence_strength = _clamp01(profile_support * freshness * (1.0 - missing_penalty) * (1.0 - 0.55 * _clamp01(sequential.uncertainty)))

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
        "batter_xwoba": batter_xwoba,
        "batter_recent_xwoba": batter_recent_xwoba,
        "pitcher_xwoba_allowed": pitcher_xwoba,
        "pitcher_recent_xwoba_allowed": pitcher_recent_xwoba,
        "batter_quality_trend": batter_quality_trend,
        "pitcher_quality_trend_for_hitter": pitcher_quality_trend_for_hitter,
        "power_trend": power_trend,
        "pitcher_xfip": pitcher.xfip,
        "pitcher_siera": pitcher.siera,
        "pitch_compatibility_score": pitch_compatibility_score,
        "pitch_context_available": pitch_context_available,
        "batter_handedness": batter.handedness,
        "pitcher_handedness": pitcher.handedness,
        "handedness_context_available": handedness_known,
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
    return ExpertState(signals=signals, activations=activations, effective_features=effective, evidence_strength=evidence_strength, diagnostics=diagnostics)


def _target_payload(artifact: Mapping[str, Any], target: str) -> Mapping[str, Any]:
    targets = artifact.get("targets") if isinstance(artifact, Mapping) else None
    payload = (targets or {}).get(target) if isinstance(targets, Mapping) else None
    if not isinstance(payload, Mapping):
        return DEFAULT_ARTIFACT["targets"][target]
    return payload


def condition_probability(
    prior_probability: float,
    *,
    target: str,
    state: ExpertState,
    artifact: Mapping[str, Any] | None = None,
    sequential_uncertainty: float = 0.0,
) -> GameConditionedProbability:
    target = str(target).upper()
    if target not in {"H", "TB"}:
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
    positive_authority = bool(payload.get("positive_authority", False))
    evidence_class = str(model.get("evidence_class") or "NONE")
    positive_authority = positive_authority and evidence_class in {
        "EXACT_POINT_IN_TIME_PROSPECTIVE",
        "EXACT_POINT_IN_TIME_LOCKED_VALIDATION",
    }

    uncertainty = _clamp01(float(sequential_uncertainty))
    validation_brier = _finite(validation.get("candidate_brier"), _finite(validation.get("prior_brier"), 0.25))
    prior_brier = _finite(validation.get("prior_brier"), validation_brier)
    calibration_risk = _clamp(max(0.0, validation_brier - prior_brier), 0.0, 0.08)
    probability_haircut = min(0.10, 0.035 * uncertainty + calibration_risk)
    lower_bound = _clamp01(candidate - probability_haircut)

    if positive_authority:
        production = lower_bound
        authority_status = "PROMOTED_RESIDUAL_POSITIVE_AND_NEGATIVE_AUTHORITY"
    else:
        production = min(prior, candidate, lower_bound)
        authority_status = "SHADOW_RESIDUAL_NEGATIVE_AUTHORITY_ONLY"

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
    """Build a stable pre-residual prior from legacy structure + no-vig market."""

    legacy = poisson_over_probability(legacy_projection, market_line)
    market = no_vig_over_probability(over_price, under_price)
    if market is None:
        return legacy, {"legacy_probability": legacy, "market_no_vig_probability": None, "legacy_weight": 1.0}
    weight = _clamp(float(legacy_weight), 0.20, 0.95)
    prior = logistic(weight * logit(legacy) + (1.0 - weight) * logit(market))
    return prior, {"legacy_probability": legacy, "market_no_vig_probability": market, "legacy_weight": weight}
