from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]


def test_all_mutating_mlb_workflows_share_one_non_canceling_lock():
    names = ["mlb-predictions.yml", "mlb-settle-predictions.yml", "mlb-same-game-predictions.yml", "mlb-pitcher-parlay-predictions.yml"]
    for name in names:
        payload = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        assert payload["concurrency"]["group"] == "mlb-production-static-deployment"
        assert payload["concurrency"]["cancel-in-progress"] is False
