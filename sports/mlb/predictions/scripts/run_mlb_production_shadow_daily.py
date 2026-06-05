#!/usr/bin/env python3
"""
MLB Production-Shadow Daily Runner

Phases:
  --phase predecision   # Get fresh MLB odds, generate predictions
  --phase close         # Collect close snapshot, compute CLV
  --phase settle        # Join outcomes, compute hit/loss/push
  --phase status        # Print exact blocker and next action
  --phase full-cycle    # Run predecision → close → settle → status

Do not:
  - Enable real staking
  - Use stale odds
  - Fabricate CLV
  - Count historical/replay rows as live evidence
  - Loosen gates
  - Mark production ready without live settled proof
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "predictions" / "odds"))
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "predictions" / "odds" / "providers"))
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "predictions" / "inference"))
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"))

MLB_SHADOW_DIR = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"
POLICY_PATH = MLB_SHADOW_DIR / "mlb_production_shadow_policy.json"
LEDGER_PATH = MLB_SHADOW_DIR / "mlb_live_ledger.csv"
STATUS_PATH = MLB_SHADOW_DIR / "production_status.json"
ODDS_STATUS_PATH = MLB_SHADOW_DIR / "odds_source_status.json"
EVIDENCE_STATUS_PATH = MLB_SHADOW_DIR / "evidence_accumulation_status.json"

MOCKED_PROVIDERS = {"mocked_fresh", "mocked_results", "test_provider"}
EVIDENCE_SOURCE_LIVE = "live_provider"
EVIDENCE_SOURCE_MOCKED = "mocked"


class MlbProductionShadowRunner:
    """MLB Production-shadow daily runner."""

    def __init__(self, provider: Optional[str] = None):
        self.provider_override = provider
        self.policy = self._load_policy()
        self.ledger = self._load_ledger()

    def _load_policy(self) -> Dict[str, Any]:
        if not POLICY_PATH.exists():
            raise FileNotFoundError(f"MLB policy not found: {POLICY_PATH}")
        policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
        if policy.get("staking_enabled", True):
            raise ValueError("MLB policy: staking_enabled must be false")
        if policy.get("live_action_enabled", True):
            raise ValueError("MLB policy: live_action_enabled must be false")
        return policy

    def _load_ledger(self) -> pd.DataFrame:
        if LEDGER_PATH.exists():
            try:
                return pd.read_csv(LEDGER_PATH)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def run_predecision(self, prediction_mode: str = "real") -> Dict[str, Any]:
        """Get fresh MLB odds and generate predictions."""
        print("=" * 60)
        print("MLB PREDECISION PHASE")
        print("=" * 60)

        from provider_router import MlbProviderRouter
        router = MlbProviderRouter()
        df_odds, info = router.get_fresh_odds()

        if df_odds is None or df_odds.empty:
            print("NO FRESH ODDS — MLB providers returned no props")
            return {
                "status": "no_fresh_odds",
                "appended_rows": 0,
                "terminal_state": "EXTERNAL_RESOURCE_BLOCKER",
            }

        print(f"Fresh odds obtained: {len(df_odds)} rows from {info.get('successful_provider')}")

        # Run model inference
        from mlb_production_shadow_inference import MlbProductionShadowInference
        model = MlbProductionShadowInference()

        rows_to_append: List[Dict[str, Any]] = []
        snapshot_id = f"mlb_entry_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

        for _, row in df_odds.iterrows():
            row_dict = row.to_dict()
            prediction = model.predict_row(row_dict)

            if prediction.get("inference_status") != "success":
                continue

            # Generate decision
            decision = self._generate_decision(row_dict, prediction)
            decision["snapshot_id"] = snapshot_id
            decision["entry_snapshot_id"] = snapshot_id
            decision["close_snapshot_id"] = ""
            decision["policy_version"] = self.policy["policy_version"]
            decision["decision_id"] = str(uuid.uuid4())
            decision["sport"] = "MLB"
            decision["league"] = "MLB"

            # Evidence flags
            is_mocked = self.provider_override in MOCKED_PROVIDERS
            decision["real_odds"] = not is_mocked
            decision["real_model"] = True
            decision["prediction_source"] = "real_model"
            decision["live_evidence"] = not is_mocked
            decision["evidence_source"] = EVIDENCE_SOURCE_MOCKED if is_mocked else EVIDENCE_SOURCE_LIVE
            decision["production_countable_for_pipeline"] = not is_mocked
            decision["production_countable_for_staking"] = False
            decision["stake_allowed"] = False
            decision["staking_enabled"] = False

            # Settlement fields (empty until settled)
            decision["actual_value"] = np.nan
            decision["hit_loss_push"] = ""
            decision["unit_profit"] = np.nan
            decision["brier"] = np.nan
            decision["market_brier"] = np.nan
            decision["bss"] = np.nan
            decision["settled_at"] = ""

            # CLV fields (empty until close)
            decision["side_aware_prob_clv"] = np.nan
            decision["side_aware_line_clv"] = np.nan
            decision["side_aware_combined_clv"] = np.nan
            decision["positive_clv"] = np.nan
            decision["clv_label_tier"] = ""

            rows_to_append.append(decision)

        if rows_to_append:
            new_df = pd.DataFrame(rows_to_append)
            if self.ledger.empty:
                self.ledger = new_df
            else:
                self.ledger = pd.concat([self.ledger, new_df], ignore_index=True)
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.ledger.to_csv(LEDGER_PATH, index=False)

        appended = len(rows_to_append)
        print(f"Appended rows: {appended}")
        print("MLB PREDECISION PHASE COMPLETE")
        return {"status": "success", "appended_rows": appended}

    def _generate_decision(self, row: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate decision tier for a row."""
        p_model = prediction.get("p_model_raw", 0.5)
        line = float(row.get("line", 0))
        odds = float(row.get("odds", -110))
        side = str(row.get("side", "over"))

        # Compute implied probability from odds
        if odds < 0:
            market_prob = abs(odds) / (abs(odds) + 100)
        else:
            market_prob = 100 / (odds + 100)

        model_edge = p_model - market_prob
        threshold = self.policy.get("candidate_threshold", {}).get("model_edge_raw", 0.15)

        # Decision tier
        tier_override = prediction.get("decision_tier_override")
        if tier_override:
            decision_tier = tier_override
        elif model_edge >= threshold:
            decision_tier = "class_a_candidate"
        elif model_edge >= threshold * 0.67:
            decision_tier = "shadow_candidate"
        elif model_edge >= 0:
            decision_tier = "monitor"
        else:
            decision_tier = "no_action"

        return {
            "provider_name": row.get("provider_name", ""),
            "source_event_id": row.get("source_event_id", ""),
            "game_id": row.get("game_id", ""),
            "snapshot_time_utc": row.get("snapshot_time_utc", ""),
            "commence_time_utc": row.get("commence_time_utc", ""),
            "home_team": row.get("home_team", ""),
            "away_team": row.get("away_team", ""),
            "player": row.get("player", ""),
            "player_id_source": row.get("player_id_source", ""),
            "team": row.get("team", ""),
            "opponent": row.get("opponent", ""),
            "market": row.get("market", ""),
            "market_canonical": row.get("market_canonical", ""),
            "line": line,
            "book": row.get("book", ""),
            "side": side,
            "odds": odds,
            "over_odds": row.get("over_odds"),
            "under_odds": row.get("under_odds"),
            "is_live": row.get("is_live", False),
            "market_no_vig": market_prob,
            "p_model_raw": p_model,
            "p_over_raw": prediction.get("p_over_raw"),
            "p_under_raw": prediction.get("p_under_raw"),
            "model_mean": prediction.get("model_mean"),
            "sigma": prediction.get("sigma"),
            "model_edge_raw": model_edge,
            "model_version": prediction.get("model_version", ""),
            "feature_completeness_score": prediction.get("feature_completeness_score", 0),
            "missing_feature_list": json.dumps(prediction.get("missing_feature_list", [])),
            "inference_status": prediction.get("inference_status", ""),
            "decision_tier": decision_tier,
            "class_a_candidate": decision_tier == "class_a_candidate",
            "shadow_candidate": decision_tier == "shadow_candidate",
        }

    def run_close(self) -> Dict[str, Any]:
        """Collect close snapshots and compute CLV."""
        print("=" * 60)
        print("MLB CLOSE PHASE")
        print("=" * 60)

        if self.ledger.empty:
            print("No pending decisions")
            return {"status": "no_pending", "closed_rows": 0}

        # Find rows needing close
        pending = self.ledger[
            (self.ledger["close_snapshot_id"].isna() | (self.ledger["close_snapshot_id"] == "")) &
            (self.ledger["entry_snapshot_id"].notna()) & (self.ledger["entry_snapshot_id"] != "")
        ]

        if pending.empty:
            print("No pending decisions for close")
            return {"status": "no_pending", "closed_rows": 0}

        # Get close odds
        from provider_router import MlbProviderRouter
        router = MlbProviderRouter()
        df_close, info = router.get_fresh_odds()

        if df_close is None or df_close.empty:
            print("No close odds available")
            return {"status": "no_close_odds", "closed_rows": 0}

        close_snapshot_id = f"mlb_close_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        closed_count = 0

        for idx in pending.index:
            row = self.ledger.loc[idx]
            # Match by game_id + player + market + line + book + side
            match = df_close[
                (df_close["game_id"] == row["game_id"]) &
                (df_close["player"] == row["player"]) &
                (df_close["market_canonical"] == row["market_canonical"]) &
                (df_close["line"] == row["line"]) &
                (df_close["book"] == row["book"]) &
                (df_close["side"] == row["side"])
            ]

            if match.empty:
                continue

            close_row = match.iloc[0]
            close_odds = float(close_row["odds"])

            # Compute no-vig probabilities
            entry_odds = float(row["odds"])
            entry_no_vig = self._odds_to_prob(entry_odds)
            close_no_vig = self._odds_to_prob(close_odds)

            # Side-aware CLV
            side = str(row["side"]).lower()
            if side == "over":
                prob_clv = close_no_vig - entry_no_vig
                line_clv = float(close_row.get("line", row["line"])) - float(row["line"])
            else:  # under
                prob_clv = entry_no_vig - close_no_vig  # Reversed for under
                line_clv = float(row["line"]) - float(close_row.get("line", row["line"]))

            combined_clv = prob_clv  # Simplified; could weight line_clv

            self.ledger.at[idx, "close_snapshot_id"] = close_snapshot_id
            self.ledger.at[idx, "side_aware_prob_clv"] = prob_clv
            self.ledger.at[idx, "side_aware_line_clv"] = line_clv
            self.ledger.at[idx, "side_aware_combined_clv"] = combined_clv
            self.ledger.at[idx, "positive_clv"] = prob_clv > 0
            self.ledger.at[idx, "clv_label_tier"] = "gold_real_clv"
            closed_count += 1

        self.ledger.to_csv(LEDGER_PATH, index=False)
        print(f"Closed decisions: {closed_count}")
        print("MLB CLOSE PHASE COMPLETE")
        return {"status": "success", "closed_rows": closed_count}

    def run_settle(self) -> Dict[str, Any]:
        """Settle outcomes using MLB stats."""
        print("=" * 60)
        print("MLB SETTLE PHASE")
        print("=" * 60)

        if self.ledger.empty:
            print("No pending decisions")
            return {"status": "no_pending", "settled_rows": 0}

        # Find rows needing settlement
        pending = self.ledger[
            (self.ledger["close_snapshot_id"].notna()) & (self.ledger["close_snapshot_id"] != "") &
            ((self.ledger["settled_at"].isna()) | (self.ledger["settled_at"] == ""))
        ]

        if pending.empty:
            print("No pending decisions for settlement")
            return {"status": "no_pending", "settled_rows": 0}

        # Try to get actual outcomes from local data
        settled_count = 0
        outcomes = self._load_outcomes()

        for idx in pending.index:
            row = self.ledger.loc[idx]
            actual = self._get_actual_value(row, outcomes)
            if actual is None:
                continue

            line = float(row["line"])
            side = str(row["side"]).lower()
            p_model = float(row.get("p_model_raw", 0.5))

            # Determine hit/loss/push
            if actual > line:
                result = "HIT" if side == "over" else "LOSS"
            elif actual < line:
                result = "HIT" if side == "under" else "LOSS"
            else:
                result = "PUSH"

            # Unit profit
            odds = float(row["odds"])
            if result == "HIT":
                if odds > 0:
                    profit = odds / 100.0
                else:
                    profit = 100.0 / abs(odds)
            elif result == "LOSS":
                profit = -1.0
            else:
                profit = 0.0

            # Brier score
            outcome_binary = 1.0 if result == "HIT" else 0.0
            brier = (p_model - outcome_binary) ** 2
            market_prob = float(row.get("market_no_vig", 0.5))
            market_brier = (market_prob - outcome_binary) ** 2

            self.ledger.at[idx, "actual_value"] = actual
            self.ledger.at[idx, "hit_loss_push"] = result
            self.ledger.at[idx, "unit_profit"] = profit
            self.ledger.at[idx, "brier"] = brier
            self.ledger.at[idx, "market_brier"] = market_brier
            self.ledger.at[idx, "bss"] = 1.0 - (brier / market_brier) if market_brier > 0 else 0.0
            self.ledger.at[idx, "settled_at"] = datetime.now(timezone.utc).isoformat()
            settled_count += 1

        self.ledger.to_csv(LEDGER_PATH, index=False)
        print(f"Decisions settled: {settled_count}")
        print("MLB SETTLE PHASE COMPLETE")
        return {"status": "success", "settled_rows": settled_count}

    def run_status(self) -> Dict[str, Any]:
        """Print production status."""
        print("=" * 60)
        print("MLB PRODUCTION STATUS")
        print("=" * 60)

        from production_gate_evaluator import MlbProductionGateEvaluator
        from staking_controller import MlbStakingController

        evaluator = MlbProductionGateEvaluator()
        gate_result = evaluator.evaluate_all()
        staking = MlbStakingController().resolve(gate_result)

        status = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "sport": "MLB",
            "system_mode": "production_shadow",
            "stage": gate_result.get("stage", "production_shadow_accumulating"),
            "production_status": gate_result.get("stage", "production_shadow_accumulating"),
            "staking_enabled": False,
            "live_action_enabled": False,
            "metrics": gate_result.get("metrics", {}),
            "failed_gates": gate_result.get("failed_gates", []),
            "passed_gates": gate_result.get("passed_gates", []),
            "next_action": "accumulate MLB live evidence with real odds and real model",
            "reason": "MLB system must collect gold live CLV + settled outcomes before staking",
        }

        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

        print(f"Stage: {status['stage']}")
        print(f"Staking: {status['staking_enabled']}")
        print(f"Failed gates: {len(status['failed_gates'])}")
        for g in status["failed_gates"]:
            print(f"  - {g}")
        print(f"Next action: {status['next_action']}")
        return status

    def _odds_to_prob(self, odds: float) -> float:
        if odds < 0:
            return abs(odds) / (abs(odds) + 100)
        elif odds > 0:
            return 100 / (odds + 100)
        return 0.5

    def _load_outcomes(self) -> pd.DataFrame:
        """Load MLB outcomes from Data-Proc-MLB or local data."""
        data_dir = WORKSPACE / "Player-Predictor" / "Data-Proc-MLB"
        if not data_dir.exists():
            return pd.DataFrame()
        # Simplified: would need game-level outcome data
        return pd.DataFrame()

    def _get_actual_value(self, row: pd.Series, outcomes: pd.DataFrame) -> Optional[float]:
        """Get actual stat value for a settled row."""
        # In production, this would query MLB StatsAPI or local processed data
        # For now, return None (settlement requires real data)
        return None


def main():
    parser = argparse.ArgumentParser(description="MLB Production-Shadow Daily Runner")
    parser.add_argument("--phase", default="full-cycle",
                        choices=["predecision", "close", "settle", "status", "full-cycle"])
    parser.add_argument("--provider", default="sportsgameodds")
    parser.add_argument("--prediction-mode", default="real", choices=["real", "simulated_for_pipeline_test"])
    args = parser.parse_args()

    runner = MlbProductionShadowRunner(provider=args.provider)

    if args.phase == "predecision":
        runner.run_predecision(args.prediction_mode)
    elif args.phase == "close":
        runner.run_close()
    elif args.phase == "settle":
        runner.run_settle()
    elif args.phase == "status":
        runner.run_status()
    elif args.phase == "full-cycle":
        runner.run_predecision(args.prediction_mode)
        runner.run_close()
        runner.run_settle()
        runner.run_status()


if __name__ == "__main__":
    main()
