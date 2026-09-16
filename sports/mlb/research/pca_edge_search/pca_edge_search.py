from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PCAConfig:
    n_components: int = 4
    neighbors: int = 24
    graph_neighbors: int = 8
    prior_strength: float = 12.0
    confidence_z: float = 1.645
    minimum_effective_support: float = 20.0
    edge_threshold: float = 0.03
    max_ood_distance: float = 3.0
    distance_epsilon: float = 1e-9


@dataclass(frozen=True)
class LocalEdgeEstimate:
    index: int
    support: float
    empirical_probability: float
    local_market_probability: float
    raw_residual: float
    standard_error: float
    conservative_residual: float
    ood_distance: float
    in_support: bool


@dataclass(frozen=True)
class EdgeSearchResult:
    status: str
    start_index: int
    goal_index: int | None
    path: tuple[int, ...]
    path_cost: float | None
    expansions: int
    estimate: LocalEdgeEstimate | None


@dataclass(frozen=True)
class HoldoutReport:
    selected: int
    wins: int
    losses: int
    pushes: int
    observed_hit_rate: float | None
    mean_market_probability: float | None
    realized_residual: float | None


class PCAStrategyEdgeSearch:
    """PCA representation + deterministic A* search for sportsbook residual regions.

    This class does not produce sporting outcomes. Historical binary outcomes are only
    accepted during fit so that local strategy-vs-market residuals can be estimated.
    Current candidates are transformed/scored using frozen training statistics.

    A* success means EDGE_FOUND: a supported representation region whose conservative
    historical residual exceeds the configured threshold. It never means WIN.
    """

    def __init__(self, config: PCAConfig | None = None):
        self.config = config or PCAConfig()
        self.feature_names: tuple[str, ...] = ()
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.training_z_: np.ndarray | None = None
        self.training_market_: np.ndarray | None = None
        self.training_outcome_: np.ndarray | None = None
        self.training_market_type_: tuple[str, ...] = ()
        self.training_side_: tuple[str, ...] = ()
        self.estimates_: tuple[LocalEdgeEstimate, ...] = ()
        self.graph_: tuple[tuple[int, ...], ...] = ()
        self.lipschitz_: float = 1.0

    @staticmethod
    def _finite_matrix(rows: Sequence[dict[str, Any]], features: Sequence[str]) -> np.ndarray:
        matrix = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=float)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("PCA feature matrix must be finite")
        return matrix

    @staticmethod
    def _market_probability(row: dict[str, Any]) -> float:
        value = row.get("no_vig_market_probability")
        if value is None:
            value = row.get("market_probability")
        p = float(value)
        if not (0.0 < p < 1.0):
            raise ValueError("market probability must be strictly between zero and one")
        return p

    @staticmethod
    def _outcome(row: dict[str, Any]) -> float:
        value = row.get("settled_hit")
        if value is None:
            settlement = str(row.get("settlement", "")).strip().lower()
            if settlement == "won":
                value = 1.0
            elif settlement == "lost":
                value = 0.0
            else:
                raise ValueError("training rows require settled_hit or won/lost settlement")
        y = float(value)
        if y not in (0.0, 1.0):
            raise ValueError("training outcome must be binary; pushes should be excluded")
        return y

    def fit(self, rows: Sequence[dict[str, Any]], *, feature_names: Sequence[str]) -> "PCAStrategyEdgeSearch":
        if len(rows) < max(8, self.config.neighbors + 1):
            raise ValueError("insufficient rows for configured neighborhood")
        self.feature_names = tuple(feature_names)
        X = self._finite_matrix(rows, self.feature_names)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0, ddof=1)
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        standardized = (X - self.mean_) / self.scale_
        _, _, vt = np.linalg.svd(standardized, full_matrices=False)
        k = min(self.config.n_components, vt.shape[0])
        self.components_ = vt[:k]
        self.training_z_ = standardized @ self.components_.T
        self.training_market_ = np.asarray([self._market_probability(row) for row in rows], dtype=float)
        self.training_outcome_ = np.asarray([self._outcome(row) for row in rows], dtype=float)
        self.training_market_type_ = tuple(str(row.get("market_type", "")).upper() for row in rows)
        self.training_side_ = tuple(str(row.get("side", "")).upper() for row in rows)

        estimates = [self._estimate_training_index(i) for i in range(len(rows))]
        self.estimates_ = tuple(estimates)
        self.graph_ = self._build_graph()
        self.lipschitz_ = self._certified_graph_lipschitz()
        return self

    def transform(self, rows: Sequence[dict[str, Any]]) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None or self.components_ is None:
            raise RuntimeError("fit must be called first")
        X = self._finite_matrix(rows, self.feature_names)
        return ((X - self.mean_) / self.scale_) @ self.components_.T

    def _compatible(self, market_type: str, side: str, j: int) -> bool:
        return market_type == self.training_market_type_[j] and side == self.training_side_[j]

    def _neighbor_indices(self, z: np.ndarray, market_type: str, side: str, *, exclude: int | None = None) -> np.ndarray:
        assert self.training_z_ is not None
        distances = np.linalg.norm(self.training_z_ - z, axis=1)
        valid = np.asarray([
            self._compatible(market_type, side, j) and (exclude is None or j != exclude)
            for j in range(len(distances))
        ])
        candidates = np.flatnonzero(valid)
        if len(candidates) == 0:
            return np.asarray([], dtype=int)
        order = candidates[np.argsort(distances[candidates])]
        return order[: self.config.neighbors]

    def _estimate(self, z: np.ndarray, market_type: str, side: str, *, index: int, exclude: int | None = None) -> LocalEdgeEstimate:
        assert self.training_z_ is not None and self.training_market_ is not None and self.training_outcome_ is not None
        idx = self._neighbor_indices(z, market_type, side, exclude=exclude)
        if len(idx) == 0:
            return LocalEdgeEstimate(index, 0.0, 0.5, 0.5, 0.0, 1.0, -1.0, math.inf, False)

        distances = np.linalg.norm(self.training_z_[idx] - z, axis=1)
        positive = distances[distances > self.config.distance_epsilon]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
        bandwidth = max(bandwidth, self.config.distance_epsilon)
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weight_sum = float(weights.sum())
        weight_sq = float(np.square(weights).sum())
        n_eff = (weight_sum * weight_sum / weight_sq) if weight_sq > 0 else 0.0
        market_local = float(np.dot(weights, self.training_market_[idx]) / weight_sum)
        wins = float(np.dot(weights, self.training_outcome_[idx]))
        prior = self.config.prior_strength
        empirical = (wins + prior * market_local) / (weight_sum + prior)
        raw_residual = empirical - market_local
        total_eff = n_eff + prior
        standard_error = math.sqrt(max(empirical * (1.0 - empirical), 1e-9) / max(total_eff, 1.0))
        ood = float(distances[0])
        conservative = raw_residual - self.config.confidence_z * standard_error
        in_support = n_eff >= self.config.minimum_effective_support and ood <= self.config.max_ood_distance
        return LocalEdgeEstimate(
            index=index,
            support=n_eff,
            empirical_probability=empirical,
            local_market_probability=market_local,
            raw_residual=raw_residual,
            standard_error=standard_error,
            conservative_residual=conservative,
            ood_distance=ood,
            in_support=in_support,
        )

    def _estimate_training_index(self, i: int) -> LocalEdgeEstimate:
        assert self.training_z_ is not None
        return self._estimate(
            self.training_z_[i], self.training_market_type_[i], self.training_side_[i], index=i, exclude=i
        )

    def score_current(self, rows: Sequence[dict[str, Any]]) -> tuple[LocalEdgeEstimate, ...]:
        z = self.transform(rows)
        out = []
        for i, (row, zi) in enumerate(zip(rows, z)):
            out.append(self._estimate(zi, str(row.get("market_type", "")).upper(), str(row.get("side", "")).upper(), index=i))
        return tuple(out)

    def _build_graph(self) -> tuple[tuple[int, ...], ...]:
        assert self.training_z_ is not None
        n = len(self.training_z_)
        graph: list[tuple[int, ...]] = []
        for i in range(n):
            distances = np.linalg.norm(self.training_z_ - self.training_z_[i], axis=1)
            candidates = [
                j for j in np.argsort(distances)
                if j != i and self._compatible(self.training_market_type_[i], self.training_side_[i], int(j))
            ]
            graph.append(tuple(int(j) for j in candidates[: self.config.graph_neighbors]))
        return tuple(graph)

    def _certified_graph_lipschitz(self) -> float:
        assert self.training_z_ is not None
        slopes = [0.0]
        for i, neighbors in enumerate(self.graph_):
            ri = self.estimates_[i].conservative_residual
            for j in neighbors:
                d = float(np.linalg.norm(self.training_z_[i] - self.training_z_[j]))
                if d > self.config.distance_epsilon:
                    slopes.append(abs(ri - self.estimates_[j].conservative_residual) / d)
        return max(max(slopes), self.config.distance_epsilon)

    def _heuristic(self, i: int) -> float:
        gap = max(0.0, self.config.edge_threshold - self.estimates_[i].conservative_residual)
        return gap / self.lipschitz_

    def _is_goal(self, i: int) -> bool:
        e = self.estimates_[i]
        return e.in_support and e.conservative_residual >= self.config.edge_threshold

    def search_training_graph(self, start_index: int) -> EdgeSearchResult:
        if not (0 <= start_index < len(self.estimates_)):
            raise IndexError("start_index out of range")
        assert self.training_z_ is not None
        open_heap: list[tuple[float, float, int]] = []
        g = {start_index: 0.0}
        parent: dict[int, int | None] = {start_index: None}
        heapq.heappush(open_heap, (self._heuristic(start_index), 0.0, start_index))
        closed: set[int] = set()
        expansions = 0

        while open_heap:
            _, current_g, u = heapq.heappop(open_heap)
            if current_g != g.get(u) or u in closed:
                continue
            closed.add(u)
            expansions += 1
            if self._is_goal(u):
                path = []
                cur: int | None = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return EdgeSearchResult(
                    status="EDGE_FOUND",
                    start_index=start_index,
                    goal_index=u,
                    path=tuple(path),
                    path_cost=current_g,
                    expansions=expansions,
                    estimate=self.estimates_[u],
                )
            for v in self.graph_[u]:
                step = float(np.linalg.norm(self.training_z_[u] - self.training_z_[v]))
                tentative = current_g + step
                if tentative < g.get(v, math.inf):
                    g[v] = tentative
                    parent[v] = u
                    heapq.heappush(open_heap, (tentative + self._heuristic(v), tentative, v))

        return EdgeSearchResult(
            status="NO_EDGE_PATH",
            start_index=start_index,
            goal_index=None,
            path=(),
            path_cost=None,
            expansions=expansions,
            estimate=None,
        )

    def select_current_edges(self, rows: Sequence[dict[str, Any]]) -> tuple[int, ...]:
        estimates = self.score_current(rows)
        return tuple(
            i for i, e in enumerate(estimates)
            if e.in_support and e.conservative_residual >= self.config.edge_threshold
        )

    @staticmethod
    def evaluate_holdout(rows: Sequence[dict[str, Any]], selected_indices: Iterable[int]) -> HoldoutReport:
        selected = list(selected_indices)
        wins = losses = pushes = 0
        market = []
        for i in selected:
            row = rows[i]
            settlement = str(row.get("settlement", "")).lower()
            if settlement == "won":
                wins += 1
            elif settlement == "lost":
                losses += 1
            elif settlement == "push":
                pushes += 1
            else:
                raise ValueError("holdout rows must be settled before evaluation")
            value = row.get("no_vig_market_probability", row.get("market_probability"))
            if value is not None:
                market.append(float(value))
        graded = wins + losses
        hit_rate = wins / graded if graded else None
        mean_market = float(np.mean(market)) if market else None
        residual = hit_rate - mean_market if hit_rate is not None and mean_market is not None else None
        return HoldoutReport(len(selected), wins, losses, pushes, hit_rate, mean_market, residual)
