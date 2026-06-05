#!/usr/bin/env python3
"""
NBA v9 Hidden Markov Model Regime Detection

Replaces heuristic regime flags with probabilistic state inference.

Research basis:
  - HMM + player network models show player performance has hidden game states
  - A player tonight is not one projection; it's a probability mixture over
    possible game states (normal, usage-spike, blowout, foul-trouble, hot/cold)
  - P(over) should be computed as a mixture:
      P(over) = sum_k P(state=k) * P(over | state=k)

This module:
  1. Fits a Gaussian HMM per player archetype on historical stat sequences
  2. Infers the most likely current regime and regime probabilities
  3. Provides regime-conditional stat distributions
  4. Computes mixture P(over) across regimes
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    SKLEARN_CLUSTER_AVAILABLE = True
except ImportError:
    SKLEARN_CLUSTER_AVAILABLE = False

from scipy.stats import norm as scipy_norm


# Default regime count - research suggests 4-6 states capture most behavior
N_REGIMES = 5
REGIME_NAMES = [
    "normal",           # Typical role, stable minutes
    "usage_spike",      # Elevated usage (teammate out, hot hand)
    "suppressed",       # Blowout, low minutes, role compression
    "high_volatility",  # Unstable game (OT risk, foul trouble, competitive)
    "cold_outlier",     # Cold shooting, low efficiency
]

# Observation features for HMM
HMM_OBS_COLS = ["MP", "USG%", "TS%", "FGA", "PLUS_MINUS"]


@dataclass
class RegimeState:
    """Current regime inference result for a player."""
    regime_probs: np.ndarray          # (N_REGIMES,) probability of each state
    most_likely_regime: int           # argmax regime index
    regime_name: str                  # human-readable name
    regime_means: np.ndarray          # (N_REGIMES, n_obs) mean per regime
    regime_covars: np.ndarray         # (N_REGIMES, n_obs) variance per regime
    confidence: float                 # max probability (how certain)
    sequence_log_likelihood: float    # log-likelihood of observed sequence


@dataclass
class RegimeConditionalDistribution:
    """Stat distribution conditioned on each regime."""
    stat: str
    line: float
    regime_means: dict[str, float]     # regime_name -> mean stat
    regime_stds: dict[str, float]      # regime_name -> std stat
    regime_p_over: dict[str, float]    # regime_name -> P(over | regime)
    mixture_p_over: float              # weighted P(over) across regimes
    dominant_regime: str               # regime with highest probability
    uncertainty: float                 # entropy of regime distribution


class PlayerRegimeHMM:
    """
    Gaussian HMM for detecting player performance regimes.

    Fits on sequences of (MP, USG%, TS%, FGA, PLUS_MINUS) to learn
    hidden states that correspond to different game contexts.
    """

    def __init__(
        self,
        n_regimes: int = N_REGIMES,
        covariance_type: str = "diag",
        n_iter: int = 100,
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: Optional[GaussianHMM] = None
        self.is_fitted = False
        self._obs_means: Optional[np.ndarray] = None
        self._obs_stds: Optional[np.ndarray] = None
        self._regime_stat_profiles: dict = {}
        self._obs_cols: list[str] = []
        self._fallback_centers: Optional[np.ndarray] = None

    def fit(
        self,
        game_sequences: pd.DataFrame,
        stat_cols: list[str] = None,
    ) -> "PlayerRegimeHMM":
        """
        Fit HMM on player game sequences.

        Args:
            game_sequences: DataFrame with HMM_OBS_COLS + stat columns, sorted by date
            stat_cols: target stat columns to profile per regime (e.g., PTS, TRB, AST)
        """
        obs_cols = [c for c in HMM_OBS_COLS if c in game_sequences.columns]
        self._obs_cols = obs_cols
        if len(obs_cols) < 3:
            self.is_fitted = True
            return self

        stat_cols = stat_cols or ["PTS", "TRB", "AST"]

        # Prepare observations
        obs_data = game_sequences[obs_cols].fillna(0).values.astype(np.float64)

        # Standardize
        self._obs_means = obs_data.mean(axis=0)
        self._obs_stds = obs_data.std(axis=0)
        self._obs_stds[self._obs_stds < 1e-6] = 1.0
        obs_scaled = (obs_data - self._obs_means) / self._obs_stds

        if not HMM_AVAILABLE:
            return self._fit_cluster_fallback(game_sequences, obs_scaled, stat_cols)

        # Fit HMM
        n_samples = len(obs_scaled)
        if n_samples < self.n_regimes * 3:
            # Not enough data for reliable HMM
            return self._fit_cluster_fallback(game_sequences, obs_scaled, stat_cols)

        try:
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.model.fit(obs_scaled)
            self.is_fitted = True

            # Profile each regime's stat distributions
            states = self.model.predict(obs_scaled)
            self._regime_stat_profiles = {}
            for stat in stat_cols:
                if stat not in game_sequences.columns:
                    continue
                stat_vals = game_sequences[stat].fillna(0).values
                profiles = {}
                for regime_idx in range(self.n_regimes):
                    mask = states == regime_idx
                    if mask.sum() > 2:
                        profiles[regime_idx] = {
                            "mean": float(np.mean(stat_vals[mask])),
                            "std": float(np.std(stat_vals[mask])),
                            "median": float(np.median(stat_vals[mask])),
                            "count": int(mask.sum()),
                        }
                    else:
                        profiles[regime_idx] = {
                            "mean": float(np.mean(stat_vals)),
                            "std": float(np.std(stat_vals)),
                            "median": float(np.median(stat_vals)),
                            "count": 0,
                        }
                self._regime_stat_profiles[stat] = profiles

        except Exception:
            # HMM fitting can fail with degenerate data
            return self._fit_cluster_fallback(game_sequences, obs_scaled, stat_cols)

        return self

    def _fit_cluster_fallback(
        self,
        game_sequences: pd.DataFrame,
        obs_scaled: np.ndarray,
        stat_cols: list[str],
    ) -> "PlayerRegimeHMM":
        """
        Fit a production fallback when hmmlearn is unavailable.

        This is not a Markov model, but it still learns hidden performance
        regimes from the same state features and preserves the mixture layer.
        """
        n_samples = len(obs_scaled)
        if n_samples < max(3, self.n_regimes):
            self.is_fitted = True
            return self

        n_clusters = min(self.n_regimes, max(2, n_samples // 10))
        if SKLEARN_CLUSTER_AVAILABLE:
            clusterer = KMeans(
                n_clusters=n_clusters,
                n_init=10,
                random_state=self.random_state,
            )
            states = clusterer.fit_predict(obs_scaled)
            centers = clusterer.cluster_centers_
        else:
            # NumPy-only fallback: rank by opportunity and split into buckets.
            score = obs_scaled[:, 0] + obs_scaled[:, 1] + 0.5 * obs_scaled[:, 3]
            quantiles = np.quantile(score, np.linspace(0, 1, n_clusters + 1))
            states = np.digitize(score, quantiles[1:-1], right=False)
            centers = np.vstack([
                obs_scaled[states == k].mean(axis=0) if np.any(states == k) else np.zeros(obs_scaled.shape[1])
                for k in range(n_clusters)
            ])

        if n_clusters < self.n_regimes:
            pad = np.repeat(centers[-1:], self.n_regimes - n_clusters, axis=0)
            centers = np.vstack([centers, pad])
        self._fallback_centers = centers[:self.n_regimes]
        self.is_fitted = True
        self.model = None

        self._regime_stat_profiles = {}
        for stat in stat_cols:
            if stat not in game_sequences.columns:
                continue
            stat_vals = game_sequences[stat].fillna(0).values
            profiles = {}
            for regime_idx in range(self.n_regimes):
                mask = states == min(regime_idx, n_clusters - 1)
                if mask.sum() > 2:
                    profiles[regime_idx] = {
                        "mean": float(np.mean(stat_vals[mask])),
                        "std": float(np.std(stat_vals[mask])),
                        "median": float(np.median(stat_vals[mask])),
                        "count": int(mask.sum()),
                    }
                else:
                    profiles[regime_idx] = {
                        "mean": float(np.mean(stat_vals)),
                        "std": float(np.std(stat_vals)),
                        "median": float(np.median(stat_vals)),
                        "count": 0,
                    }
            self._regime_stat_profiles[stat] = profiles

        return self

    def infer_regime(
        self,
        recent_games: pd.DataFrame,
        n_recent: int = 10,
    ) -> RegimeState:
        """
        Infer current regime from recent game sequence.

        Args:
            recent_games: last N games with HMM_OBS_COLS, sorted by date
            n_recent: how many recent games to use

        Returns:
            RegimeState with regime probabilities and metadata
        """
        # Default fallback
        default_probs = np.ones(self.n_regimes) / self.n_regimes
        default_state = RegimeState(
            regime_probs=default_probs,
            most_likely_regime=0,
            regime_name=REGIME_NAMES[0],
            regime_means=np.zeros((self.n_regimes, len(HMM_OBS_COLS))),
            regime_covars=np.ones((self.n_regimes, len(HMM_OBS_COLS))),
            confidence=1.0 / self.n_regimes,
            sequence_log_likelihood=0.0,
        )

        obs_cols = [c for c in HMM_OBS_COLS if c in recent_games.columns]
        if len(obs_cols) < 3:
            return default_state

        # Take last n_recent games
        games = recent_games.tail(n_recent)
        obs_data = games[obs_cols].fillna(0).values.astype(np.float64)

        if len(obs_data) < 2:
            return default_state

        # Scale using training statistics
        obs_scaled = (obs_data - self._obs_means[:len(obs_cols)]) / self._obs_stds[:len(obs_cols)]

        if self.model is None and self._fallback_centers is not None:
            try:
                current = obs_scaled[-1]
                centers = self._fallback_centers[:, :len(obs_cols)]
                distances = np.linalg.norm(centers - current, axis=1)
                inv = np.exp(-distances)
                probs = inv / max(inv.sum(), 1e-12)
                most_likely = int(np.argmax(probs))
                regime_name = REGIME_NAMES[most_likely] if most_likely < len(REGIME_NAMES) else "unknown"
                regime_means = centers * self._obs_stds[:len(obs_cols)] + self._obs_means[:len(obs_cols)]
                return RegimeState(
                    regime_probs=probs,
                    most_likely_regime=most_likely,
                    regime_name=regime_name,
                    regime_means=regime_means,
                    regime_covars=np.ones((self.n_regimes, len(obs_cols))),
                    confidence=float(np.max(probs)),
                    sequence_log_likelihood=float(-distances[most_likely]),
                )
            except Exception:
                return default_state

        try:
            # Get state probabilities for the last observation
            log_likelihood = self.model.score(obs_scaled)
            posteriors = self.model.predict_proba(obs_scaled)
            last_probs = posteriors[-1]  # regime probs for most recent game

            most_likely = int(np.argmax(last_probs))
            regime_name = REGIME_NAMES[most_likely] if most_likely < len(REGIME_NAMES) else "unknown"

            # Extract regime means/covars in original scale
            regime_means = self.model.means_ * self._obs_stds[:len(obs_cols)] + self._obs_means[:len(obs_cols)]
            if self.covariance_type == "diag":
                regime_covars = self.model.covars_ * (self._obs_stds[:len(obs_cols)] ** 2)
            else:
                regime_covars = np.ones((self.n_regimes, len(obs_cols)))

            return RegimeState(
                regime_probs=last_probs,
                most_likely_regime=most_likely,
                regime_name=regime_name,
                regime_means=regime_means,
                regime_covars=regime_covars,
                confidence=float(np.max(last_probs)),
                sequence_log_likelihood=float(log_likelihood),
            )

        except Exception:
            return default_state

    def compute_regime_conditional_p_over(
        self,
        regime_state: RegimeState,
        stat: str,
        line: float,
    ) -> RegimeConditionalDistribution:
        """
        Compute P(over) as a mixture across regimes.

        P(over) = sum_k P(regime=k) * P(stat > line | regime=k)

        This is the key insight: a player's over probability is not one number,
        it's a weighted sum over possible game states.
        """
        regime_means = {}
        regime_stds = {}
        regime_p_over = {}

        profiles = self._regime_stat_profiles.get(stat, {})

        for k in range(self.n_regimes):
            name = REGIME_NAMES[k] if k < len(REGIME_NAMES) else f"regime_{k}"
            if k in profiles and profiles[k]["count"] > 0:
                mu = profiles[k]["mean"]
                sigma = max(profiles[k]["std"], 0.5)
            else:
                # Fallback to overall distribution
                mu = np.mean([p["mean"] for p in profiles.values()]) if profiles else line
                sigma = np.mean([p["std"] for p in profiles.values()]) if profiles else 5.0

            regime_means[name] = mu
            regime_stds[name] = sigma

            # P(stat > line | regime=k) using Gaussian CDF
            z = (line - mu) / sigma
            p_over_k = float(1.0 - scipy_norm.cdf(z))
            regime_p_over[name] = np.clip(p_over_k, 0.01, 0.99)

        # Mixture P(over)
        mixture_p_over = 0.0
        for k in range(self.n_regimes):
            name = REGIME_NAMES[k] if k < len(REGIME_NAMES) else f"regime_{k}"
            mixture_p_over += regime_state.regime_probs[k] * regime_p_over[name]

        # Entropy of regime distribution (uncertainty about which state we're in)
        probs = regime_state.regime_probs
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = float(np.log(self.n_regimes))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        dominant_name = REGIME_NAMES[regime_state.most_likely_regime] \
            if regime_state.most_likely_regime < len(REGIME_NAMES) else "unknown"

        return RegimeConditionalDistribution(
            stat=stat,
            line=line,
            regime_means=regime_means,
            regime_stds=regime_stds,
            regime_p_over=regime_p_over,
            mixture_p_over=float(np.clip(mixture_p_over, 0.01, 0.99)),
            dominant_regime=dominant_name,
            uncertainty=normalized_entropy,
        )

    def save(self, path: str | Path) -> None:
        """Save fitted HMM to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "n_regimes": self.n_regimes,
            "covariance_type": self.covariance_type,
            "model": self.model,
            "is_fitted": self.is_fitted,
            "obs_means": self._obs_means,
            "obs_stds": self._obs_stds,
            "obs_cols": self._obs_cols,
            "fallback_centers": self._fallback_centers,
            "regime_stat_profiles": self._regime_stat_profiles,
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "PlayerRegimeHMM":
        """Load fitted HMM from disk."""
        data = joblib.load(str(path))
        hmm = cls(
            n_regimes=data["n_regimes"],
            covariance_type=data["covariance_type"],
        )
        hmm.model = data["model"]
        hmm.is_fitted = data["is_fitted"]
        hmm._obs_means = data["obs_means"]
        hmm._obs_stds = data["obs_stds"]
        hmm._obs_cols = data.get("obs_cols", [])
        hmm._fallback_centers = data.get("fallback_centers")
        hmm._regime_stat_profiles = data["regime_stat_profiles"]
        return hmm


class ArchetypeRegimeLibrary:
    """
    Manages HMMs for player archetypes rather than individual players.

    Research insight: Bayesian hierarchical models show players should share
    structure through archetypes. We don't model every player from scratch.

    Archetypes:
      - high_usage_guard: Primary scorers/playmakers (30%+ USG)
      - secondary_scorer: 22-30% USG wings/guards
      - low_usage_big: Centers/PFs with <22% USG, high rebounds
      - playmaker: High assist rate, moderate scoring
      - role_player: Low usage, high variance, minutes-dependent
    """

    ARCHETYPES = [
        "high_usage_guard",
        "secondary_scorer",
        "low_usage_big",
        "playmaker",
        "role_player",
    ]

    def __init__(self):
        self.hmms: dict[str, PlayerRegimeHMM] = {}
        self._archetype_rules: dict[str, callable] = {
            "high_usage_guard": lambda row: row.get("USG%", 0) >= 0.30 and row.get("AST", 0) >= 4,
            "secondary_scorer": lambda row: 0.22 <= row.get("USG%", 0) < 0.30,
            "low_usage_big": lambda row: row.get("USG%", 0) < 0.22 and row.get("TRB", 0) >= 6,
            "playmaker": lambda row: row.get("AST", 0) >= 7,
            "role_player": lambda row: True,  # fallback
        }

    def classify_archetype(self, player_stats: dict) -> str:
        """Classify a player into an archetype based on season averages."""
        for archetype, rule in self._archetype_rules.items():
            if rule(player_stats):
                return archetype
        return "role_player"

    def fit_archetype(
        self,
        archetype: str,
        all_games: pd.DataFrame,
    ) -> None:
        """Fit HMM for a specific archetype using pooled player data."""
        hmm = PlayerRegimeHMM()
        hmm.fit(all_games)
        self.hmms[archetype] = hmm

    def infer_regime(
        self,
        archetype: str,
        recent_games: pd.DataFrame,
    ) -> RegimeState:
        """Infer regime for a player of given archetype."""
        if archetype not in self.hmms:
            archetype = "role_player"
        if archetype not in self.hmms:
            # Return uniform default
            probs = np.ones(N_REGIMES) / N_REGIMES
            return RegimeState(
                regime_probs=probs,
                most_likely_regime=0,
                regime_name=REGIME_NAMES[0],
                regime_means=np.zeros((N_REGIMES, len(HMM_OBS_COLS))),
                regime_covars=np.ones((N_REGIMES, len(HMM_OBS_COLS))),
                confidence=1.0 / N_REGIMES,
                sequence_log_likelihood=0.0,
            )
        return self.hmms[archetype].infer_regime(recent_games)

    def save(self, dir_path: str | Path) -> None:
        """Save all archetype HMMs."""
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        for archetype, hmm in self.hmms.items():
            hmm.save(dir_path / f"hmm_{archetype}.pkl")
        # Save metadata
        meta = {"archetypes": list(self.hmms.keys())}
        (dir_path / "archetype_meta.json").write_text(
            __import__("json").dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, dir_path: str | Path) -> "ArchetypeRegimeLibrary":
        """Load all archetype HMMs."""
        dir_path = Path(dir_path)
        lib = cls()
        meta_path = dir_path / "archetype_meta.json"
        if meta_path.exists():
            import json
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for archetype in meta.get("archetypes", []):
                hmm_path = dir_path / f"hmm_{archetype}.pkl"
                if hmm_path.exists():
                    lib.hmms[archetype] = PlayerRegimeHMM.load(hmm_path)
        return lib


if __name__ == "__main__":
    np.random.seed(42)
    print("Testing HMM Regime Detection...")

    # Simulate player game sequence with regime switches
    n_games = 60
    # Normal regime (games 0-19)
    normal = np.column_stack([
        np.random.normal(34, 3, 20),   # MP
        np.random.normal(0.28, 0.03, 20),  # USG%
        np.random.normal(0.58, 0.05, 20),  # TS%
        np.random.normal(18, 3, 20),   # FGA
        np.random.normal(3, 8, 20),    # PLUS_MINUS
    ])
    # Usage spike (games 20-34)
    spike = np.column_stack([
        np.random.normal(37, 2, 15),
        np.random.normal(0.35, 0.03, 15),
        np.random.normal(0.55, 0.06, 15),
        np.random.normal(22, 3, 15),
        np.random.normal(5, 10, 15),
    ])
    # Suppressed (games 35-44)
    suppressed = np.column_stack([
        np.random.normal(24, 4, 10),
        np.random.normal(0.22, 0.04, 10),
        np.random.normal(0.52, 0.08, 10),
        np.random.normal(12, 3, 10),
        np.random.normal(-10, 8, 10),
    ])
    # Back to normal (games 45-59)
    normal2 = np.column_stack([
        np.random.normal(33, 3, 15),
        np.random.normal(0.27, 0.03, 15),
        np.random.normal(0.57, 0.05, 15),
        np.random.normal(17, 3, 15),
        np.random.normal(2, 7, 15),
    ])

    obs = np.vstack([normal, spike, suppressed, normal2])
    pts = np.concatenate([
        np.random.normal(24, 5, 20),
        np.random.normal(32, 6, 15),
        np.random.normal(14, 4, 10),
        np.random.normal(23, 5, 15),
    ])

    df = pd.DataFrame(obs, columns=HMM_OBS_COLS)
    df["PTS"] = pts
    df["TRB"] = np.random.normal(6, 2, n_games)
    df["AST"] = np.random.normal(5, 2, n_games)

    # Fit HMM
    hmm = PlayerRegimeHMM(n_regimes=4)
    hmm.fit(df)

    # Infer regime from last 10 games
    regime = hmm.infer_regime(df)
    print(f"  Current regime: {regime.regime_name} (confidence: {regime.confidence:.2f})")
    print(f"  Regime probs: {regime.regime_probs}")

    # Compute regime-conditional P(over)
    dist = hmm.compute_regime_conditional_p_over(regime, stat="PTS", line=24.5)
    print(f"  Mixture P(over 24.5): {dist.mixture_p_over:.3f}")
    print(f"  Regime P(over): {dist.regime_p_over}")
    print(f"  Uncertainty: {dist.uncertainty:.3f}")

    print("\nHMM Regime Detection smoke test PASSED")
