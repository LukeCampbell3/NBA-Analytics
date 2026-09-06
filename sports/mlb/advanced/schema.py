from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


ADVANCED_SCHEMA_VERSION = "mlb_advanced_profiles_v1"
SEQUENTIAL_OUTPUT_SCHEMA_VERSION = "mlb_sequential_pa_output_v1"


@dataclass(frozen=True)
class BatterProcessProfile:
    player_id: int
    player_name: str
    as_of_date: str
    source: str = "baseball_savant_statcast"
    sample_pa: int = 0
    sample_bbe: int = 0
    handedness: str = ""
    k_rate: float = 0.225
    bb_rate: float = 0.085
    hbp_rate: float = 0.012
    hr_rate: float = 0.030
    contact_rate: float = 0.775
    whiff_rate: float = 0.225
    chase_rate: float | None = None
    woba: float | None = None
    xwoba: float | None = None
    xba: float | None = None
    xslg: float | None = None
    avg_ev: float | None = None
    ev90: float | None = None
    hard_hit_rate: float | None = None
    barrel_rate: float | None = None
    sweet_spot_rate: float | None = None
    gb_rate: float | None = None
    ld_rate: float | None = None
    fb_rate: float | None = None
    single_share_non_hr_hits: float = 0.70
    double_share_non_hr_hits: float = 0.27
    triple_share_non_hr_hits: float = 0.03
    rolling: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    pitch_type_xwoba: dict[str, float] = field(default_factory=dict)
    pitch_type_whiff_rate: dict[str, float] = field(default_factory=dict)
    support: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PitcherProcessProfile:
    player_id: int
    player_name: str
    as_of_date: str
    source: str = "baseball_savant_statcast"
    sample_pa: int = 0
    sample_bbe: int = 0
    handedness: str = ""
    k_rate: float = 0.225
    bb_rate: float = 0.085
    hbp_rate: float = 0.012
    hr_rate: float = 0.030
    k_minus_bb_rate: float = 0.14
    whiff_rate: float = 0.225
    csw_rate: float | None = None
    xwoba_allowed: float | None = None
    xba_allowed: float | None = None
    xslg_allowed: float | None = None
    avg_ev_allowed: float | None = None
    hard_hit_rate_allowed: float | None = None
    barrel_rate_allowed: float | None = None
    sweet_spot_rate_allowed: float | None = None
    gb_rate: float | None = None
    fb_rate: float | None = None
    era: float | None = None
    fip: float | None = None
    xfip: float | None = None
    siera: float | None = None
    xera: float | None = None
    projected_ip: float | None = None
    projected_pitches: float | None = None
    arsenal: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    rolling: dict[str, dict[str, float | int | None]] = field(default_factory=dict)
    support: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectMatchupProcess:
    batter_id: int
    pitcher_id: int
    pa: int
    strikeouts: int
    walks: int
    hbp: int
    home_runs: int
    non_hr_contacts: int
    hard_contacts: int
    weak_contacts: int
    xwoba_contact: float | None
    xba_contact: float | None
    xslg_contact: float | None
    avg_ev: float | None
    barrel_rate: float | None
    whiff_rate: float | None
    shrinkage_weight: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdvancedCandidateContext:
    game_id: str
    run_date: str
    batter: BatterProcessProfile
    pitcher: PitcherProcessProfile
    direct_matchup: DirectMatchupProcess | None
    batting_order: int | None
    is_home: bool
    team_expected_runs: float | None
    park_factor: float
    defense_residual: float
    defense_status: str
    data_freshness_status: str
    missing_components: tuple[str, ...] = field(default_factory=tuple)
    temperature_f: float | None = None


@dataclass(frozen=True)
class SequentialPAResult:
    model_version: str
    run_date: str
    game_id: str
    player_id: int
    pitcher_id: int
    trials: int
    expected_pa: float
    expected_ab: float
    expected_hits: float
    expected_tb: float
    pa_distribution: dict[str, float]
    p_h_0: float
    p_h_1: float
    p_h_ge_2: float
    p_tb_0: float
    p_tb_1: float
    p_tb_ge_2: float
    p_hr_ge_1: float
    hit_over_0_5_probability: float
    tb_over_1_5_probability: float
    market_clear_probabilities: dict[str, float]
    probability_standard_error: float
    raw_structural_probability: float
    calibrated_probability: float
    usable_probability: float
    probability_lcb: float
    uncertainty: float
    uncertainty_components: dict[str, float]
    support: float
    support_status: str
    calibration_status: str
    data_freshness_status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
