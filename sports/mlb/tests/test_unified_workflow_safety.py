from pathlib import Path

import json
import yaml

from sports.mlb.unified.policy_manifest import build_policy_manifest


ROOT = Path(__file__).resolve().parents[3]


def test_committed_engine_manifest_matches_runtime_policy_hash():
    manifest = json.loads((ROOT / "artifacts/mlb_engine_manifest.json").read_text())
    assert manifest["policy_hash"] == build_policy_manifest(ROOT)["policy_hash"]


def test_all_mutating_mlb_workflows_share_one_non_canceling_lock():
    names = ["mlb-predictions.yml", "mlb-settle-predictions.yml", "mlb-same-game-predictions.yml", "mlb-pitcher-parlay-predictions.yml"]
    for name in names:
        payload = yaml.safe_load((ROOT / ".github/workflows" / name).read_text())
        assert payload["concurrency"]["group"] == "mlb-production-static-deployment"
        assert payload["concurrency"]["cancel-in-progress"] is False
