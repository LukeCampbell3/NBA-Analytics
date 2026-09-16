import unittest

import numpy as np

from sports.mlb.research.pca_edge_search.pca_edge_search import PCAConfig, PCAStrategyEdgeSearch


class PCAEdgeSearchTests(unittest.TestCase):
    def _rows(self, n=240, seed=7):
        rng = np.random.default_rng(seed)
        rows = []
        for _ in range(n):
            x1 = float(rng.normal())
            x2 = float(rng.normal())
            market = 0.50
            # Synthetic representation region with persistent edge vs flat market baseline.
            true_p = 0.78 if (x1 > 0.55 and x2 > 0.20) else 0.50
            y = float(rng.random() < true_p)
            rows.append({
                "f1": x1,
                "f2": x2,
                "market_type": "BATTER_HITS",
                "side": "OVER",
                "no_vig_market_probability": market,
                "settled_hit": y,
                "settlement": "won" if y else "lost",
            })
        return rows

    def test_discovers_supported_residual_region_without_calling_it_win(self):
        rows = self._rows()
        model = PCAStrategyEdgeSearch(PCAConfig(
            n_components=2,
            neighbors=45,
            graph_neighbors=12,
            prior_strength=4.0,
            confidence_z=1.0,
            minimum_effective_support=12.0,
            edge_threshold=0.025,
        )).fit(rows, feature_names=["f1", "f2"])
        goals = [
            i for i, estimate in enumerate(model.estimates_)
            if estimate.in_support and estimate.conservative_residual >= model.config.edge_threshold
        ]
        self.assertTrue(goals)
        start = min(range(len(rows)), key=lambda i: model.estimates_[i].conservative_residual)
        result = model.search_training_graph(start)
        self.assertEqual(result.status, "EDGE_FOUND")
        self.assertNotIn("WIN", result.status)
        self.assertGreaterEqual(result.estimate.conservative_residual, model.config.edge_threshold)

    def test_current_scoring_uses_frozen_historical_outcomes_only(self):
        rows = self._rows()
        model = PCAStrategyEdgeSearch(PCAConfig(
            n_components=2,
            neighbors=45,
            graph_neighbors=10,
            prior_strength=4.0,
            confidence_z=1.0,
            minimum_effective_support=10.0,
            edge_threshold=0.02,
        )).fit(rows, feature_names=["f1", "f2"])
        current = [{
            "f1": 1.1,
            "f2": 0.8,
            "market_type": "BATTER_HITS",
            "side": "OVER",
            "no_vig_market_probability": 0.50,
        }]
        estimate = model.score_current(current)[0]
        self.assertTrue(np.isfinite(estimate.conservative_residual))
        self.assertGreater(estimate.raw_residual, 0.0)

    def test_no_path_when_threshold_is_unreachable(self):
        rows = self._rows()
        model = PCAStrategyEdgeSearch(PCAConfig(
            n_components=2,
            neighbors=45,
            graph_neighbors=10,
            prior_strength=8.0,
            confidence_z=1.645,
            minimum_effective_support=10.0,
            edge_threshold=0.45,
        )).fit(rows, feature_names=["f1", "f2"])
        result = model.search_training_graph(0)
        self.assertEqual(result.status, "NO_EDGE_PATH")
        self.assertIsNone(result.goal_index)

    def test_holdout_settlement_is_separate_from_discovery(self):
        rows = self._rows(80, seed=31)
        report = PCAStrategyEdgeSearch.evaluate_holdout(rows, [0, 1, 2, 3, 4])
        self.assertEqual(report.selected, 5)
        self.assertEqual(report.wins + report.losses + report.pushes, 5)


if __name__ == "__main__":
    unittest.main()
