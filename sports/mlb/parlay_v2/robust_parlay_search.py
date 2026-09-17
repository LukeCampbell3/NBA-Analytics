from __future__ import annotations

"""Risk-aware MLB two-leg parlay valuation and incremental search.

This module is proposal/research logic only. It never turns a search result
into a settlement prediction or production authorization. Probability and
joint-world evidence must be supplied by upstream statistical pipelines.

Core separation:
    PCA/support evidence -> probability/joint evidence -> robust valuation
        -> LPA* combination search.

LPA* only organizes already-valued combinations. It never estimates an
outcome probability.
"""

from dataclasses import dataclass, field, asdict
import heapq
import math
from typing import Any, Iterable, Sequence

import numpy as np

CERTIFIED_PAIR = "CERTIFIED_PAIR"
PROBABLE_PAIR = "PROBABLE_PAIR"
LOTTERY_TAIL = "LOTTERY_TAIL"
UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class LegProbabilityEvidence:
    leg_id: str
    game_id: str
    mean_probability: float
    decimal_price: float | None
    effective_support: int
    probability_samples: Sequence[float] = field(default_factory=tuple)
    pca_edge_certified: bool = False
    market: str | None = None
    team: str | None = None
    book: str | None = None
    sampling_source: str = "UNKNOWN"


@dataclass(frozen=True)
class JointDependencyEvidence:
    """Evidence for the relationship between the two legs.

    Modes:
      INDEPENDENT  - cross-game only; joint samples are products of marginal
                     probability samples.
      JOINT_SAMPLES - caller supplies bootstrap/posterior samples for P(A & B).
      COMMON_WORLD  - caller supplies Mx2 settled/simulated binary outcomes from
                      the same worlds.
      UNSUPPORTED   - no defensible dependence evidence.
    """

    mode: str
    joint_probability_samples: Sequence[float] = field(default_factory=tuple)
    common_world_outcomes: Sequence[Sequence[int]] = field(default_factory=tuple)
    support_count: int = 0
    note: str = ""


@dataclass(frozen=True)
class RobustParlayPolicy:
    bootstrap_draws: int = 10_000
    lower_quantile: float = 0.10
    p_ev_positive_threshold: float = 0.80
    min_leg_support: int = 50
    min_joint_support: int = 1_000
    min_joint_mean_probability: float = 0.50
    max_loss_probability: float = 0.50
    lottery_joint_probability: float = 0.25
    max_dependency_burden: float = 0.20
    kelly_cap: float = 0.05
    kelly_multiplier: float = 0.25
    cvar_alpha: float = 0.10
    uncertainty_penalty: float = 0.50
    dependency_penalty: float = 0.25
    concentration_penalty: float = 0.25
    require_pipeline_samples_for_certified: bool = True
    require_pca_certificate_for_certified: bool = True
    random_seed: int = 20260917
    utility_scale: float = 0.01


@dataclass(frozen=True)
class RobustParlayValuation:
    pair_id: str
    leg_ids: tuple[str, str]
    classification: str
    classification_reasons: tuple[str, ...]
    p_joint: float
    p_joint_lcb: float
    d_fair: float
    d_book: float | None
    ev_mean: float | None
    ev_lcb: float | None
    p_ev_positive: float | None
    p_loss: float
    cvar: float | None
    kelly_fraction: float | None
    conservative_kelly_fraction: float | None
    dependency_burden: float
    evidence_support: int
    joint_support: int
    joint_method: str
    parlay_log_growth: float | None
    separate_singles_log_growth: float | None
    growth_advantage: float | None
    uncertainty_width: float
    robust_utility: float
    same_game: bool
    explicit_pipeline_uncertainty: bool
    pca_certified_legs: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    selected_pair_id: str | None
    selected: RobustParlayValuation | None
    path: tuple[str, ...]
    total_cost: float | None
    expansions: int
    mode: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_pair_id": self.selected_pair_id,
            "selected": self.selected.as_dict() if self.selected else None,
            "path": list(self.path),
            "total_cost": self.total_cost,
            "expansions": self.expansions,
            "mode": self.mode,
        }


def _clip_probability(x: float) -> float:
    return float(np.clip(float(x), 1e-6, 1.0 - 1e-6))


def _resample_probability(
    evidence: LegProbabilityEvidence,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    supplied = np.asarray(tuple(evidence.probability_samples), dtype=float)
    supplied = supplied[np.isfinite(supplied)]
    supplied = supplied[(supplied > 0.0) & (supplied < 1.0)]
    if supplied.size >= 20:
        idx = rng.integers(0, supplied.size, size=draws)
        return supplied[idx], True

    # Research fallback only. This is deliberately not considered explicit
    # pipeline uncertainty for CERTIFIED_PAIR unless policy opts in.
    p = _clip_probability(evidence.mean_probability)
    n = max(int(evidence.effective_support), 2)
    alpha = 0.5 + p * n
    beta = 0.5 + (1.0 - p) * n
    return rng.beta(alpha, beta, size=draws), False


def _kelly_fraction(p: float, d: float) -> float:
    if not (0.0 < p < 1.0) or d <= 1.0:
        return 0.0
    b = d - 1.0
    return float(max(0.0, min(1.0, (p * d - 1.0) / b)))


def _binary_log_growth(p_samples: np.ndarray, decimal_price: float, stake_fraction: float) -> float:
    f = float(np.clip(stake_fraction, 0.0, 0.95))
    if f <= 0.0 or decimal_price <= 1.0:
        return 0.0
    win_log = math.log1p(f * (decimal_price - 1.0))
    loss_log = math.log1p(-f)
    vals = p_samples * win_log + (1.0 - p_samples) * loss_log
    return float(np.mean(vals))


def _separate_singles_growth(
    p1_samples: np.ndarray,
    p2_samples: np.ndarray,
    d1: float | None,
    d2: float | None,
    total_exposure: float,
) -> float | None:
    if d1 is None or d2 is None or d1 <= 1.0 or d2 <= 1.0:
        return None
    each = float(total_exposure) / 2.0
    return _binary_log_growth(p1_samples, d1, each) + _binary_log_growth(p2_samples, d2, each)


def _return_cvar(
    p_samples: np.ndarray,
    decimal_price: float,
    *,
    alpha: float,
    rng: np.random.Generator,
) -> float:
    outcomes = rng.random(p_samples.size) < p_samples
    returns = np.where(outcomes, decimal_price - 1.0, -1.0)
    cutoff = max(1, int(math.ceil(alpha * returns.size)))
    return float(np.mean(np.partition(returns, cutoff - 1)[:cutoff]))


def _joint_samples(
    leg_1_samples: np.ndarray,
    leg_2_samples: np.ndarray,
    dependency: JointDependencyEvidence,
    *,
    same_game: bool,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, str, int, bool, str | None]:
    mode = str(dependency.mode or "UNSUPPORTED").upper()
    if mode == "INDEPENDENT":
        if same_game:
            return None, mode, 0, False, "SAME_GAME_REQUIRES_COMMON_WORLD_OR_JOINT_SAMPLES"
        p1 = leg_1_samples[rng.permutation(draws)]
        p2 = leg_2_samples[rng.permutation(draws)]
        return p1 * p2, mode, min(draws, int(dependency.support_count or draws)), True, None

    if mode == "JOINT_SAMPLES":
        vals = np.asarray(tuple(dependency.joint_probability_samples), dtype=float)
        vals = vals[np.isfinite(vals)]
        vals = vals[(vals > 0.0) & (vals < 1.0)]
        if vals.size < 20:
            return None, mode, int(vals.size), False, "INSUFFICIENT_JOINT_PROBABILITY_SAMPLES"
        idx = rng.integers(0, vals.size, size=draws)
        return vals[idx], mode, int(dependency.support_count or vals.size), True, None

    if mode == "COMMON_WORLD":
        worlds = np.asarray(tuple(tuple(r) for r in dependency.common_world_outcomes), dtype=float)
        if worlds.ndim != 2 or worlds.shape[1] != 2 or worlds.shape[0] < 20:
            return None, mode, int(worlds.shape[0] if worlds.ndim == 2 else 0), False, "INSUFFICIENT_COMMON_WORLDS"
        valid = np.all(np.isin(worlds, [0.0, 1.0]), axis=1)
        worlds = worlds[valid]
        if worlds.shape[0] < 20:
            return None, mode, int(worlds.shape[0]), False, "INVALID_COMMON_WORLD_OUTCOMES"
        both = np.logical_and(worlds[:, 0] == 1.0, worlds[:, 1] == 1.0)
        wins = int(np.sum(both))
        losses = int(both.size - wins)
        vals = rng.beta(wins + 0.5, losses + 0.5, size=draws)
        return vals, mode, int(dependency.support_count or worlds.shape[0]), True, None

    return None, mode, 0, False, "DEPENDENCY_UNSUPPORTED"


def value_two_leg_parlay(
    pair_id: str,
    leg_1: LegProbabilityEvidence,
    leg_2: LegProbabilityEvidence,
    dependency: JointDependencyEvidence,
    *,
    combined_decimal_price: float | None = None,
    policy: RobustParlayPolicy | None = None,
) -> RobustParlayValuation:
    policy = policy or RobustParlayPolicy()
    pair_seed = (sum(ord(c) for c in str(pair_id)) + policy.random_seed) % (2**32 - 1)
    rng = np.random.default_rng(pair_seed)

    p1, p1_explicit = _resample_probability(leg_1, draws=policy.bootstrap_draws, rng=rng)
    p2, p2_explicit = _resample_probability(leg_2, draws=policy.bootstrap_draws, rng=rng)
    same_game = str(leg_1.game_id) == str(leg_2.game_id)

    p_joint_samples, joint_method, joint_support, dependency_supported, dependency_error = _joint_samples(
        p1,
        p2,
        dependency,
        same_game=same_game,
        draws=policy.bootstrap_draws,
        rng=rng,
    )

    d_book = combined_decimal_price
    if d_book is None and not same_game and leg_1.decimal_price is not None and leg_2.decimal_price is not None:
        d_book = float(leg_1.decimal_price) * float(leg_2.decimal_price)
    if d_book is not None and (not np.isfinite(d_book) or d_book <= 1.0):
        d_book = None

    evidence_support = min(int(leg_1.effective_support), int(leg_2.effective_support))
    pca_certified = bool(leg_1.pca_edge_certified and leg_2.pca_edge_certified)
    explicit_pipeline_uncertainty = bool(p1_explicit and p2_explicit and dependency_supported)

    reasons: list[str] = []
    if p_joint_samples is None:
        reasons.append(dependency_error or "NO_JOINT_DISTRIBUTION")
        independent_samples = np.clip(p1 * p2, 1e-9, 1.0 - 1e-9)
        independent_mean = float(np.mean(independent_samples))
        return RobustParlayValuation(
            pair_id=pair_id,
            leg_ids=(leg_1.leg_id, leg_2.leg_id),
            classification=UNSUPPORTED,
            classification_reasons=tuple(reasons),
            p_joint=independent_mean,
            p_joint_lcb=float(np.quantile(independent_samples, policy.lower_quantile)),
            d_fair=float(1.0 / max(independent_mean, 1e-9)),
            d_book=d_book,
            ev_mean=None,
            ev_lcb=None,
            p_ev_positive=None,
            p_loss=float(1.0 - independent_mean),
            cvar=None,
            kelly_fraction=None,
            conservative_kelly_fraction=None,
            dependency_burden=0.0,
            evidence_support=evidence_support,
            joint_support=joint_support,
            joint_method=joint_method,
            parlay_log_growth=None,
            separate_singles_log_growth=None,
            growth_advantage=None,
            uncertainty_width=float(independent_mean - np.quantile(independent_samples, policy.lower_quantile)),
            robust_utility=-math.inf,
            same_game=same_game,
            explicit_pipeline_uncertainty=explicit_pipeline_uncertainty,
            pca_certified_legs=pca_certified,
        )

    p_joint_samples = np.clip(p_joint_samples, 1e-9, 1.0 - 1e-9)
    p_joint = float(np.mean(p_joint_samples))
    p_lcb = float(np.quantile(p_joint_samples, policy.lower_quantile))
    independent_samples = np.clip(p1 * p2, 1e-9, 1.0 - 1e-9)
    independent_mean = float(np.mean(independent_samples))
    dependency_burden = float(abs(p_joint - independent_mean) / max(independent_mean, 1e-9))
    p_loss = float(1.0 - p_joint)
    uncertainty_width = float(max(0.0, p_joint - p_lcb))
    d_fair = float(1.0 / max(p_joint, 1e-9))

    ev_mean = ev_lcb = p_ev_positive = cvar = None
    kelly = conservative_kelly = None
    parlay_growth = singles_growth = growth_advantage = None
    if d_book is None:
        reasons.append("MISSING_COMBINED_PRICE")
    else:
        ev_samples = p_joint_samples * d_book - 1.0
        ev_mean = float(p_joint * d_book - 1.0)
        ev_lcb = float(p_lcb * d_book - 1.0)
        p_ev_positive = float(np.mean(ev_samples > 0.0))
        cvar = _return_cvar(p_joint_samples, d_book, alpha=policy.cvar_alpha, rng=rng)
        kelly = _kelly_fraction(p_joint, d_book)
        conservative_kelly = _kelly_fraction(p_lcb, d_book)
        stake = min(policy.kelly_cap, policy.kelly_multiplier * conservative_kelly)
        parlay_growth = _binary_log_growth(p_joint_samples, d_book, stake)
        singles_growth = _separate_singles_growth(p1, p2, leg_1.decimal_price, leg_2.decimal_price, stake)
        if singles_growth is not None:
            growth_advantage = float(parlay_growth - singles_growth)

    if evidence_support < policy.min_leg_support:
        reasons.append("INSUFFICIENT_LEG_SUPPORT")
    if same_game and joint_support < policy.min_joint_support:
        reasons.append("INSUFFICIENT_JOINT_SUPPORT")
    if dependency_burden > policy.max_dependency_burden:
        reasons.append("DEPENDENCY_BURDEN_TOO_HIGH")
    if policy.require_pca_certificate_for_certified and not pca_certified:
        reasons.append("PCA_EDGE_NOT_CERTIFIED")
    if policy.require_pipeline_samples_for_certified and not explicit_pipeline_uncertainty:
        reasons.append("PIPELINE_UNCERTAINTY_SAMPLES_REQUIRED")

    base_growth = growth_advantage if growth_advantage is not None else -1.0
    robust_utility = float(
        base_growth
        - policy.uncertainty_penalty * uncertainty_width
        - policy.dependency_penalty * dependency_burden
        - policy.concentration_penalty * max(0.0, p_loss - policy.max_loss_probability)
    )

    hard_unsupported = (
        d_book is None
        or not dependency_supported
        or evidence_support < policy.min_leg_support
        or (same_game and joint_support < policy.min_joint_support)
    )
    lottery = p_joint < policy.lottery_joint_probability
    certified_tests = [
        ev_lcb is not None and ev_lcb > 0.0,
        p_ev_positive is not None and p_ev_positive >= policy.p_ev_positive_threshold,
        p_joint >= policy.min_joint_mean_probability,
        p_loss <= policy.max_loss_probability,
        dependency_burden <= policy.max_dependency_burden,
        growth_advantage is not None and growth_advantage > 0.0,
        (pca_certified or not policy.require_pca_certificate_for_certified),
        (explicit_pipeline_uncertainty or not policy.require_pipeline_samples_for_certified),
    ]

    if hard_unsupported:
        classification = UNSUPPORTED
    elif lottery:
        classification = LOTTERY_TAIL
        reasons.append("LOW_JOINT_PROBABILITY")
    elif all(certified_tests):
        classification = CERTIFIED_PAIR
        reasons.append("ROBUST_CERTIFICATE_PASSED")
    elif ev_mean is not None and ev_mean > 0.0:
        classification = PROBABLE_PAIR
        if ev_lcb is not None and ev_lcb <= 0.0:
            reasons.append("LOWER_BOUND_EV_NOT_POSITIVE")
        if p_ev_positive is not None and p_ev_positive < policy.p_ev_positive_threshold:
            reasons.append("P_EV_POSITIVE_BELOW_THRESHOLD")
        if p_loss > policy.max_loss_probability:
            reasons.append("LOSS_PROBABILITY_TOO_HIGH")
        if growth_advantage is None or growth_advantage <= 0.0:
            reasons.append("PARLAY_DOES_NOT_BEAT_SEPARATE_SINGLES")
    else:
        classification = UNSUPPORTED
        reasons.append("NONPOSITIVE_MEAN_EV")

    return RobustParlayValuation(
        pair_id=pair_id,
        leg_ids=(leg_1.leg_id, leg_2.leg_id),
        classification=classification,
        classification_reasons=tuple(dict.fromkeys(reasons)),
        p_joint=p_joint,
        p_joint_lcb=p_lcb,
        d_fair=d_fair,
        d_book=d_book,
        ev_mean=ev_mean,
        ev_lcb=ev_lcb,
        p_ev_positive=p_ev_positive,
        p_loss=p_loss,
        cvar=cvar,
        kelly_fraction=kelly,
        conservative_kelly_fraction=conservative_kelly,
        dependency_burden=dependency_burden,
        evidence_support=evidence_support,
        joint_support=joint_support,
        joint_method=joint_method,
        parlay_log_growth=parlay_growth,
        separate_singles_log_growth=singles_growth,
        growth_advantage=growth_advantage,
        uncertainty_width=uncertainty_width,
        robust_utility=robust_utility,
        same_game=same_game,
        explicit_pipeline_uncertainty=explicit_pipeline_uncertainty,
        pca_certified_legs=pca_certified,
    )


class LPAParlaySearch:
    """Incremental shortest-path selector over pre-valued two-leg parlays.

    Graph topology:
        START -> canonical first-leg node -> pair node -> GOAL.

    Pair->GOAL cost is a positive monotone transform of robust_utility, so
    shortest path is the highest-utility admissible pair. LPA* uses h=0,
    which is admissible and keeps search correctness independent of learning.
    """

    START = "START"
    GOAL = "GOAL"

    def __init__(self, valuations: Iterable[RobustParlayValuation], *, policy: RobustParlayPolicy | None = None, mode: str = "certified"):
        self.policy = policy or RobustParlayPolicy()
        self.mode = mode
        self.valuations: dict[str, RobustParlayValuation] = {v.pair_id: v for v in valuations}
        self.succ: dict[str, dict[str, float]] = {}
        self.pred: dict[str, dict[str, float]] = {}
        self.g: dict[str, float] = {}
        self.rhs: dict[str, float] = {}
        self._heap: list[tuple[float, float, int, str]] = []
        self._version: dict[str, int] = {}
        self._counter = 0
        self.expansions = 0
        self._build_graph()
        self._initialize()

    def _eligible(self, v: RobustParlayValuation) -> bool:
        if self.mode == "certified":
            return v.classification == CERTIFIED_PAIR
        if self.mode == "shadow":
            return v.classification in {CERTIFIED_PAIR, PROBABLE_PAIR}
        if self.mode == "all_supported":
            return v.classification != UNSUPPORTED
        raise ValueError(f"unknown mode: {self.mode}")

    def _pair_cost(self, v: RobustParlayValuation) -> float:
        scale = max(float(self.policy.utility_scale), 1e-9)
        u = float(np.clip(v.robust_utility / scale, -50.0, 50.0))
        return float(1.0 - math.tanh(u))

    def _add_edge(self, u: str, v: str, cost: float) -> None:
        self.succ.setdefault(u, {})[v] = float(cost)
        self.pred.setdefault(v, {})[u] = float(cost)
        self.succ.setdefault(v, {})
        self.pred.setdefault(u, {})

    def _remove_edge(self, u: str, v: str) -> None:
        self.succ.get(u, {}).pop(v, None)
        self.pred.get(v, {}).pop(u, None)

    def _build_graph(self) -> None:
        self.succ = {self.START: {}, self.GOAL: {}}
        self.pred = {self.START: {}, self.GOAL: {}}
        for v in self.valuations.values():
            leg = min(v.leg_ids)
            leg_node = f"LEG::{leg}"
            pair_node = f"PAIR::{v.pair_id}"
            self._add_edge(self.START, leg_node, 0.0)
            self._add_edge(leg_node, pair_node, 0.0)
            if self._eligible(v):
                self._add_edge(pair_node, self.GOAL, self._pair_cost(v))

    def _nodes(self) -> set[str]:
        return set(self.succ) | set(self.pred)

    def _key(self, u: str) -> tuple[float, float]:
        m = min(self.g.get(u, math.inf), self.rhs.get(u, math.inf))
        return (m, m)

    def _push(self, u: str) -> None:
        self._counter += 1
        version = self._version.get(u, 0) + 1
        self._version[u] = version
        k1, k2 = self._key(u)
        heapq.heappush(self._heap, (k1, k2, version, u))

    def _peek_valid(self) -> tuple[float, float, str] | None:
        while self._heap:
            k1, k2, ver, u = self._heap[0]
            if self._version.get(u) != ver or self.g.get(u, math.inf) == self.rhs.get(u, math.inf):
                heapq.heappop(self._heap)
                continue
            return k1, k2, u
        return None

    def _pop_valid(self) -> tuple[str, tuple[float, float]] | None:
        top = self._peek_valid()
        if top is None:
            return None
        k1, k2, u = top
        heapq.heappop(self._heap)
        return u, (k1, k2)

    def _initialize(self) -> None:
        self.g = {u: math.inf for u in self._nodes()}
        self.rhs = {u: math.inf for u in self._nodes()}
        self.rhs[self.START] = 0.0
        self._heap = []
        self._version = {}
        self.expansions = 0
        self._push(self.START)

    def _update_vertex(self, u: str) -> None:
        if u != self.START:
            preds = self.pred.get(u, {})
            self.rhs[u] = min((self.g.get(p, math.inf) + c for p, c in preds.items()), default=math.inf)
        self._version[u] = self._version.get(u, 0) + 1
        if self.g.get(u, math.inf) != self.rhs.get(u, math.inf):
            self._push(u)

    def compute_shortest_path(self) -> int:
        before = self.expansions
        while True:
            top = self._peek_valid()
            goal_key = self._key(self.GOAL)
            goal_consistent = self.g.get(self.GOAL, math.inf) == self.rhs.get(self.GOAL, math.inf)
            if top is None or ((top[0], top[1]) >= goal_key and goal_consistent):
                break
            popped = self._pop_valid()
            if popped is None:
                break
            u, _ = popped
            self.expansions += 1
            gu = self.g.get(u, math.inf)
            ru = self.rhs.get(u, math.inf)
            if gu > ru:
                self.g[u] = ru
                for s in self.succ.get(u, {}):
                    self._update_vertex(s)
            else:
                self.g[u] = math.inf
                self._update_vertex(u)
                for s in self.succ.get(u, {}):
                    self._update_vertex(s)
        return self.expansions - before

    def update_valuation(self, valuation: RobustParlayValuation) -> None:
        pair_id = valuation.pair_id
        old = self.valuations.get(pair_id)
        if old is None:
            self.valuations[pair_id] = valuation
            self._build_graph()
            self._initialize()
            return
        self.valuations[pair_id] = valuation
        pair_node = f"PAIR::{pair_id}"
        was = self._eligible(old)
        now = self._eligible(valuation)
        if was and not now:
            self._remove_edge(pair_node, self.GOAL)
        elif now:
            self._add_edge(pair_node, self.GOAL, self._pair_cost(valuation))
        self._update_vertex(self.GOAL)

    def result(self) -> SearchResult:
        self.compute_shortest_path()
        goal_cost = self.g.get(self.GOAL, math.inf)
        if not np.isfinite(goal_cost):
            return SearchResult(None, None, tuple(), None, self.expansions, self.mode)
        best_pair = None
        best_cost = math.inf
        for pred, edge_cost in self.pred.get(self.GOAL, {}).items():
            total = self.g.get(pred, math.inf) + edge_cost
            if total < best_cost:
                best_cost = total
                best_pair = pred
        if best_pair is None or not best_pair.startswith("PAIR::"):
            return SearchResult(None, None, tuple(), None, self.expansions, self.mode)
        pair_id = best_pair.split("PAIR::", 1)[1]
        selected = self.valuations[pair_id]
        leg_node = f"LEG::{min(selected.leg_ids)}"
        return SearchResult(
            selected_pair_id=pair_id,
            selected=selected,
            path=(self.START, leg_node, best_pair, self.GOAL),
            total_cost=float(best_cost),
            expansions=self.expansions,
            mode=self.mode,
        )


def value_and_search(
    pair_inputs: Iterable[tuple[str, LegProbabilityEvidence, LegProbabilityEvidence, JointDependencyEvidence, float | None]],
    *,
    policy: RobustParlayPolicy | None = None,
    mode: str = "certified",
) -> tuple[list[RobustParlayValuation], SearchResult]:
    policy = policy or RobustParlayPolicy()
    valuations = [
        value_two_leg_parlay(pair_id, leg_1, leg_2, dependency, combined_decimal_price=price, policy=policy)
        for pair_id, leg_1, leg_2, dependency, price in pair_inputs
    ]
    search = LPAParlaySearch(valuations, policy=policy, mode=mode)
    return valuations, search.result()
