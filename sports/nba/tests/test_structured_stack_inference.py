from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PLAYER_PREDICTOR_ROOT = REPO_ROOT / "sports" / "nba" / "predictions" / "Player-Predictor"
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT))
sys.path.insert(0, str(PLAYER_PREDICTOR_ROOT / "inference"))

import structured_stack_inference as ssi


def test_structured_stack_inference_falls_back_to_surrogate_on_contract_failure(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "lstm_v7_metadata.json").write_text(json.dumps({"target_columns": ["PTS", "TRB", "AST"]}), encoding="utf-8")

    monkeypatch.setattr(ssi.joblib, "load", lambda _path: {})

    def fail_contract(self) -> None:
        raise ValueError("invalid metadata contract")

    monkeypatch.setattr(ssi.StructuredStackInference, "_validate_or_repair_metadata_contract", fail_contract)

    def init_surrogate(self, reason: str) -> bool:
        self.surrogate_mode = True
        self.surrogate_predictor = object()
        self.artifact_free = False
        self.artifact_free_reason = None
        self.metadata = {
            "model_type": "surrogate_market_predictor",
            "run_id": "surrogate_tabular_v1",
            "target_columns": ["PTS", "TRB", "AST"],
            "recovered_from_legacy_artifact_error": str(reason),
        }
        self.target_columns = ["PTS", "TRB", "AST"]
        self.feature_columns = []
        self.baseline_features = ["PTS_rolling_avg", "TRB_rolling_avg", "AST_rolling_avg"]
        self.feature_spec = {}
        self.seq_len = 5
        self.n_features = 0
        self.n_targets = 3
        self.player_mapping = {}
        self.team_mapping = {}
        self.opponent_mapping = {}
        self.counts = {"players": 1, "teams": 1, "opponents": 1}
        self.member_configs = []
        self.val_losses = []
        self.catboost_model_info = {}
        self.required_feature_versions = {"surrogate_tabular_v1"}
        self.feature_trainer = None
        self.models = []
        self.pts_branch = None
        self.pts_ablate_feature_key = None
        self.pts_ablate_blocks = []
        self.enable_pts_residual_split = False
        return True

    monkeypatch.setattr(ssi.StructuredStackInference, "_init_surrogate_mode", init_surrogate)

    predictor = ssi.StructuredStackInference(model_dir=str(model_dir))

    assert predictor.surrogate_mode is True
    assert predictor.artifact_free is False
    assert predictor.metadata["run_id"] == "surrogate_tabular_v1"
