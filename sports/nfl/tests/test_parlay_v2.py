from __future__ import annotations

"""Regression tests for the NFL PARLAY_CERTIFICATION_V2 / PARLAY_V2
replication (ported from MLB's sports/mlb/research/parlay_certification_v2
and sports/mlb/parlay_v2). Mirrors the structure of
sports/mlb/tests/test_parlay_v2_integration.py and
test_parlay_v2_calibration.py, adapted to NFL's play-dict schema and
weekly cadence.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from sports.nfl.parlay_v2 import comparison, frontend_payload, legacy_control, program_alpha, run_parlay_v2
from sports.nfl.parlay_v2.calibration import historical_backfill, ingest, pair_ingest
from sports.nfl.parlay_v2.calibration.settlement_source import grade_play
from sports.nfl.parlay_v2.calibration.snapshot import build_snapshot
from sports.nfl.parlay_v2.calibration.store import CalibrationStore
from sports.nfl.parlay_v2.candidate_adapter import Leg, build_candidates_for_week, build_week_action_plays, exact_event_key
from sports.nfl.research.parlay_certification_v2 import manifest
from sports.nfl.research.parlay_certification_v2.eligibility import EligibilityInputs, evaluate_eligibility
from sports.nfl.research.parlay_certification_v2.state_machine import PolicyStatus, next_status

REPO_ROOT = Path(__file__).resolve().parents[3]


def _play(player_id, event_id, target="passing", direction="OVER", line=249.5, price=-120.0, prob=0.62, in_support=True, team=None):
    return {
        "player": f"Player {player_id}", "player_id": player_id, "position": "QB", "team": team or player_id,
        "opponent": "OPP", "event_id": event_id, "game_start_utc": "2026-09-13T17:00:00Z",
        "market": target, "target": target, "direction": direction, "line": line,
        "projection": line + 10, "raw_model_probability": prob, "calibrated_hit_probability": prob,
        "model_hit_probability": prob, "no_vig_probability": prob - 0.05, "probability_advantage": 0.05,
        "meta_policy_score": 0.9, "confidence_in_support": in_support,
        "selected_side_price": price, "selected_sportsbook_key": "draftkings",
        "market_books": 3, "market_common_books": 2, "available_sportsbooks": ["draftkings", "fanduel", "betmgm"],
        "offers": {"draftkings": {"price": price, "snapshot_time_utc": "2026-09-13T12:00:00Z"}},
        "market_source": "live", "price_confirmed": True, "snapshot_time_utc": "2026-09-13T12:00:00Z",
        "price_age_seconds": 100, "policy_version": "nfl_passing_loss_aware_meta_policy_v2",
        "candidate_authorized": False, "action_status": "review", "risk_flags": [],
    }


# ---------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------

def test_eligibility_no_games_scheduled_reason():
    decision = evaluate_eligibility(EligibilityInputs(
        date="2026-W02", required_feed_available=True, week_has_games=False,
        required_system_component_available=True, decision_cutoff_met=True,
    ))
    assert decision.eligible is False
    assert decision.reason == "no_games_scheduled"


def test_eligibility_operationally_eligible():
    decision = evaluate_eligibility(EligibilityInputs(
        date="2026-W02", required_feed_available=True, week_has_games=True,
        required_system_component_available=True, decision_cutoff_met=True,
    ))
    assert decision.eligible is True
    assert decision.reason == "operationally_eligible"


# ---------------------------------------------------------------------
# candidate_adapter: distinctness rule + exact-event identity
# ---------------------------------------------------------------------

def test_build_candidates_excludes_same_event_pairs():
    plays = [_play("p1", "evt1", target="passing"), _play("p2", "evt1", target="receiving")]
    candidates = build_candidates_for_week(plays, week_id="2026-W02", aps_threshold=1.0, calibration_slates=0, predictive_version="V1", state_version="S1")
    assert candidates == []


def test_build_candidates_excludes_same_player_pairs():
    plays = [_play("p1", "evt1", target="passing"), _play("p1", "evt2", target="rushing")]
    candidates = build_candidates_for_week(plays, week_id="2026-W02", aps_threshold=1.0, calibration_slates=0, predictive_version="V1", state_version="S1")
    assert candidates == []


def test_build_candidates_pairs_distinct_event_and_player():
    plays = [_play("p1", "evt1", target="passing"), _play("p2", "evt2", target="receiving")]
    candidates = build_candidates_for_week(plays, week_id="2026-W02", aps_threshold=1.0, calibration_slates=0, predictive_version="V1", state_version="S1")
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.leg_1.decimal_price is not None and candidate.leg_2.decimal_price is not None
    # descriptive-only: no authoritative field on the candidate
    assert not hasattr(candidate, "certified")
    assert not hasattr(candidate, "production_authorized")


def test_alternate_lines_never_share_probability_or_price():
    """Two plays differing ONLY in line must never be collapsed into one
    event identity -- exact_event_key must distinguish them."""
    key_low = exact_event_key("p1", "evt1", "passing", "OVER", 220.5)
    key_high = exact_event_key("p1", "evt1", "passing", "OVER", 260.5)
    assert key_low != key_high


def test_leg_as_dict_has_no_authoritative_fields():
    leg = Leg(player="A", player_id="p1", event_id="e1", target="passing", side="OVER", line=1.0, book="b", decimal_price=1.5, quote_timestamp="t", model_probability_estimate=0.6, in_support=True)
    d = leg.as_dict()
    for forbidden in ("certified", "safe", "supported", "production_authorized", "risk_passed"):
        assert forbidden not in d


# ---------------------------------------------------------------------
# run_parlay_v2: eligibility / freeze-gating / shadow candidate
# ---------------------------------------------------------------------

def test_build_week_payload_ineligible_when_no_plays():
    eligibility_inputs = EligibilityInputs(date="2026-W02", required_feed_available=True, week_has_games=False, required_system_component_available=True, decision_cutoff_met=True)
    payload = run_parlay_v2.build_week_payload(
        plays=[], week_id="2026-W02", eligibility_inputs=eligibility_inputs,
        predictive_version="V1", state_version="S1",
    )
    assert payload["eligible"] is False
    assert payload["abstain_reason"] == "OPERATIONALLY_INELIGIBLE"
    assert payload["action"] == "ABSTAIN"
    assert payload["staking_authorized"] is False


def test_build_week_payload_abstains_no_real_quote_when_eligible_but_empty():
    eligibility_inputs = EligibilityInputs(date="2026-W02", required_feed_available=True, week_has_games=True, required_system_component_available=True, decision_cutoff_met=True)
    payload = run_parlay_v2.build_week_payload(
        plays=[], week_id="2026-W02", eligibility_inputs=eligibility_inputs,
        predictive_version="V1", state_version="S1",
    )
    assert payload["abstain_reason"] == "NO_REAL_QUOTE"


def test_build_week_payload_shows_shadow_candidate_but_abstains_policy_not_frozen(monkeypatch):
    """Exercises the POLICY_NOT_FROZEN guard directly (via monkeypatch)
    rather than assuming manifest.STATUS's real current value -- this
    policy is frozen as of 2026-08-23 (see manifest.py's
    CONCLUSION_REASONING), so the real STATUS no longer takes this path,
    but the guard itself must still work correctly whenever a future
    policy version starts out DEVELOPMENT again."""
    monkeypatch.setattr(manifest, "STATUS", "DEVELOPMENT")
    plays = [_play("p1", "evt1", target="passing"), _play("p2", "evt2", target="receiving")]
    eligibility_inputs = EligibilityInputs(date="2026-W02", required_feed_available=True, week_has_games=True, required_system_component_available=True, decision_cutoff_met=True)
    payload = run_parlay_v2.build_week_payload(
        plays=plays, week_id="2026-W02", eligibility_inputs=eligibility_inputs,
        predictive_version="V1", state_version="S1",
    )
    assert payload["abstain_reason"] == "POLICY_NOT_FROZEN"
    assert payload["shadow_candidate"] is not None
    assert payload["action"] == "ABSTAIN"
    assert payload["staking_authorized"] is False


def test_build_week_payload_abstains_no_state_support_once_frozen_with_empty_ledger(tmp_path):
    """Now that the real policy is frozen, an eligible week with real
    priced candidates but an empty (or missing) calibration ledger must
    abstain on a REAL support reason -- never ACT, and never a bare
    generic reason."""
    assert manifest.STATUS == "FROZEN_PROSPECTIVE_INCONCLUSIVE"
    plays = [_play("p1", "evt1", target="passing"), _play("p2", "evt2", target="receiving")]
    eligibility_inputs = EligibilityInputs(date="2026-W02", required_feed_available=True, week_has_games=True, required_system_component_available=True, decision_cutoff_met=True)
    calibration_store = CalibrationStore(tmp_path / "calibration.jsonl")
    payload = run_parlay_v2.build_week_payload(
        plays=plays, week_id="2026-W02", eligibility_inputs=eligibility_inputs,
        predictive_version="V1", state_version="S1", calibration_store=calibration_store,
    )
    assert payload["action"] == "ABSTAIN"
    assert payload["staking_authorized"] is False
    assert payload["abstain_reason"] in ("NO_STATE_SUPPORT", "NO_LEG_MARKET_SUPPORT", "NO_LEG_LINE_SUPPORT")
    assert payload["shadow_candidate"] is not None


def test_build_week_payload_never_sets_staking_authorized_true():
    """No code path in build_week_payload may ever set staking_authorized
    True -- selection alone never authorizes production/real-money
    staking."""
    source = (REPO_ROOT / "sports" / "nfl" / "parlay_v2" / "run_parlay_v2.py").read_text()
    assert '"staking_authorized"] = True' not in source
    assert "staking_authorized=True" not in source


# ---------------------------------------------------------------------
# settlement_source.grade_play
# ---------------------------------------------------------------------

def _actuals_frame():
    return pd.DataFrame([
        {"player_id": "p1", "season": 2026, "week": 2, "passing_yards": 275.0, "rushing_yards": 10.0, "receiving_yards": 0.0},
        {"player_id": "p2", "season": 2026, "week": 2, "passing_yards": 0.0, "rushing_yards": 0.0, "receiving_yards": 40.0},
    ])


def test_grade_play_over_win():
    play = _play("p1", "evt1", target="passing", direction="OVER", line=249.5)
    assert grade_play(play, _actuals_frame(), season=2026, week=2) is True


def test_grade_play_over_loss():
    play = _play("p2", "evt2", target="receiving", direction="OVER", line=59.5)
    assert grade_play(play, _actuals_frame(), season=2026, week=2) is False


def test_grade_play_returns_none_when_ungraded():
    play = _play("p9", "evt9", target="passing", direction="OVER", line=200.0)
    assert grade_play(play, _actuals_frame(), season=2026, week=2) is None


def test_grade_play_returns_none_on_push():
    play = _play("p1", "evt1", target="passing", direction="OVER", line=275.0)
    assert grade_play(play, _actuals_frame(), season=2026, week=2) is None


# ---------------------------------------------------------------------
# calibration ingest: never raises on a zero-row / missing-snapshot week
# ---------------------------------------------------------------------

def test_ingest_settled_week_missing_snapshot_returns_zero_rows(tmp_path):
    result = ingest.ingest_settled_week(
        tmp_path / "does_not_exist.json", season=2026, week=2, ledger_path=tmp_path / "ledger.jsonl",
    )
    assert result["admitted"] == 0
    assert result["reason"] == "snapshot_not_found"


def test_ingest_settled_week_admits_real_graded_plays(tmp_path, monkeypatch):
    plays = [_play("p1", "evt1", target="passing", direction="OVER", line=249.5)]
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"plays": plays}))
    monkeypatch.setattr(ingest, "load_season_actuals", lambda season, cache_path=None: _actuals_frame())
    result = ingest.ingest_settled_week(snapshot_path, season=2026, week=2, ledger_path=tmp_path / "ledger.jsonl")
    assert result["admitted"] == 1
    assert result["action_eligible_rows"] == 1

    # idempotent re-run
    result2 = ingest.ingest_settled_week(snapshot_path, season=2026, week=2, ledger_path=tmp_path / "ledger.jsonl")
    assert result2["admitted"] == 0
    assert result2["already_present"] == 1


def test_ingest_settled_pairs_admits_real_pairs(tmp_path, monkeypatch):
    plays = [_play("p1", "evt1", target="passing", direction="OVER", line=249.5), _play("p2", "evt2", target="receiving", direction="OVER", line=59.5)]
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps({"plays": plays}))
    monkeypatch.setattr(pair_ingest, "load_season_actuals", lambda season, cache_path=None: _actuals_frame())
    calibration_store = CalibrationStore(tmp_path / "calibration.jsonl")
    # both legs need REQUIRED support to be pairable -- with an empty
    # calibration ledger and independent_slate_count=0, market/line/state
    # support all correctly FAIL, so zero pairs should be admitted (an
    # honest result, not a crash).
    result = pair_ingest.ingest_settled_pairs(
        snapshot_path, season=2026, week=2, pair_ledger_path=tmp_path / "pairs.jsonl", calibration_store=calibration_store,
    )
    assert result["pairs_admitted"] == 0
    assert result["action_eligible_legs"] == 2


# ---------------------------------------------------------------------
# legacy_control: reads NFL's old build_shadow_parlay output correctly
# ---------------------------------------------------------------------

def test_legacy_control_reads_daily_parlay_key():
    raw = {
        "daily_parlay": {
            "policy_version": "nfl_distinct_game_parlay_v1",
            "status": "withheld",
            "available": True,
            "selected_ticket": {
                "legs": [
                    {"player": "A", "market": "passing", "line": 249.5, "direction": "OVER"},
                    {"player": "B", "market": "receiving", "line": 59.5, "direction": "OVER"},
                ],
                "combined_decimal_price": 3.5,
                "projected_probability": 0.36,
                "sportsbook_key": "draftkings",
            },
            "validation_status": "failed_locked_holdout",
            "candidate_authorized": False,
        }
    }
    control = legacy_control.load_legacy_parlay_control_from_payload(raw)
    assert control.available is True
    assert control.old_control_pair is not None
    assert len(control.old_control_pair) == 2
    assert control.old_control_probability == 0.36


def test_legacy_control_handles_missing_daily_parlay():
    control = legacy_control.load_legacy_parlay_control_from_payload({})
    assert control.available is False
    assert control.reason == "old_parlay_diagnostic_no_daily_parlay_key"


# ---------------------------------------------------------------------
# state_machine sanity (ported verbatim, but confirm NFL's manifest
# actually starts DEVELOPMENT)
# ---------------------------------------------------------------------

def test_manifest_status_is_a_valid_policy_status():
    assert manifest.STATUS in {status.value for status in PolicyStatus}


def test_manifest_production_authorized_is_always_false():
    assert manifest.PRODUCTION_AUTHORIZED is False


def test_manifest_world_gate_mode_is_observe_only():
    assert manifest.WORLD_GATE_MODE == "OBSERVE_ONLY"
    assert manifest.WORLD_RISK_THRESHOLD is None


# ---------------------------------------------------------------------
# frontend_payload: additive-only embedding
# ---------------------------------------------------------------------

def test_frontend_payload_unavailable_when_no_artifact(tmp_path):
    result = frontend_payload.embed_parlays_v2({"plays": ["x"]}, tmp_path / "missing.json")
    assert result["plays"] == ["x"]  # untouched
    assert result["parlays"]["abstain_reason"] == "PARLAY_V2_ARTIFACT_UNAVAILABLE"


def test_frontend_payload_never_overwrites_existing_keys(tmp_path):
    artifact = tmp_path / "parlay_v2.json"
    artifact.write_text(json.dumps({"system": "PARLAY_POLICY_V2", "action": "ACT"}))
    original = {"plays": [1, 2, 3], "policy_governance": {"a": 1}}
    result = frontend_payload.embed_parlays_v2(original, artifact)
    assert result["plays"] == [1, 2, 3]
    assert result["policy_governance"] == {"a": 1}
    assert result["parlays"]["action"] == "ACT"


# ---------------------------------------------------------------------
# program_alpha: fresh spend, no prior conflict
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# historical_backfill: REAL historical data only, exact authorized scope
# ---------------------------------------------------------------------

def test_historical_backfill_sources_are_exactly_the_authorized_scope():
    """Guards against silent scope creep -- see historical_backfill.py's
    module docstring for the exact chat-authorized scope (2025 full
    season, 2022 weeks 1-2 only, 2021 explicitly NOT authorized). A
    future change to SOURCES needs a fresh explicit check-in, exactly
    like the original authorization did -- this test is the tripwire."""
    sources_by_name = {Path(s["path"]).name: s["weeks"] for s in historical_backfill.SOURCES}
    assert sources_by_name == {
        "recent_selector_pool_2025.csv": None,
        "market_selector_pool_2022.csv": (1, 2),
    }
    assert "market_selector_pool_2021.csv" not in sources_by_name
    assert "market_selector_validated_pool_2022.csv" not in sources_by_name  # the actual locked holdout file -- must never be read here


def test_historical_backfill_admits_real_rows_and_is_idempotent(tmp_path):
    fixture = tmp_path / "fixture_pool.csv"
    fixture.write_text(
        "season,week,player_id,player_display_name,target,side,line,selected_price,"
        "estimated_side_probability,current_prediction,result,recent_team,opponent_team,"
        "bookmaker,snapshot_time_utc,commence_time_utc\n"
        "2025,1,p1,Player One,passing,over,249.5,-120,0.6,265.0,win,AAA,BBB,draftkings,"
        "2025-09-07T12:00:00Z,2025-09-07T17:00:00Z\n"
        "2025,1,p2,Player Two,receiving,under,59.5,-110,0.55,45.0,loss,CCC,DDD,fanduel,"
        "2025-09-07T12:00:00Z,2025-09-07T17:00:00Z\n"
    )
    ledger_path = tmp_path / "ledger.jsonl"
    sources = ({"path": fixture, "weeks": None},)

    result = historical_backfill.backfill_historical_pool(ledger_path, sources=sources)
    assert result["admitted"] == 2
    assert result["independent_weeks_admitted"] == 1
    assert result["skipped_incomplete"] == 0

    store = CalibrationStore(ledger_path)
    snapshot = build_snapshot(store, as_of="2099-01-01T00:00:00Z")
    assert snapshot.market_support_summary.get("passing") == 1
    assert snapshot.line_support_summary.get("receiving|UNDER|59.5") == 1

    # idempotent re-run
    result2 = historical_backfill.backfill_historical_pool(ledger_path, sources=sources)
    assert result2["admitted"] == 0
    assert result2["already_present"] == 2


def test_historical_backfill_skips_incomplete_rows_without_fabricating(tmp_path):
    fixture = tmp_path / "fixture_incomplete.csv"
    fixture.write_text(
        "season,week,player_id,player_display_name,target,side,line,selected_price,"
        "estimated_side_probability,current_prediction,result,recent_team,opponent_team,"
        "bookmaker,snapshot_time_utc,commence_time_utc\n"
        "2025,1,p1,Player One,passing,over,,-120,0.6,265.0,win,AAA,BBB,draftkings,"
        "2025-09-07T12:00:00Z,2025-09-07T17:00:00Z\n"
    )
    ledger_path = tmp_path / "ledger.jsonl"
    result = historical_backfill.backfill_historical_pool(ledger_path, sources=({"path": fixture, "weeks": None},))
    assert result["admitted"] == 0
    assert result["skipped_incomplete"] == 1


def test_comparison_record_flags_pair_disagreement():
    legacy = legacy_control.LegacyParlayControl(
        available=True,
        old_control_pair=[{"player": "A", "target": "passing", "line": 249.5}],
        old_control_probability=0.5, old_control_quote={"combined_decimal_price": 3.0}, reason="old_parlay_diagnostic_loaded",
    )
    record = comparison.build_comparison_record(
        date="2026-W02", policy_version=manifest.POLICY_VERSION, legacy=legacy,
        new_v2_pair=[{"player": "B", "target": "receiving", "line": 59.5}],
        new_joint_score=0.4, new_quote={"combined_decimal_price": 3.2}, new_action="ACT", new_policy_status="FROZEN_PROSPECTIVE_INCONCLUSIVE",
    )
    assert record.same_pair is False


def test_program_alpha_fresh_spend_succeeds(tmp_path):
    ledger_path = tmp_path / "program_alpha_ledger.json"
    ledger = program_alpha.ProgramAlphaLedger(ledger_path, manifest.ALPHA_PROGRAM)
    ledger.spend(program_alpha.AlphaSpend(
        policy_version=manifest.PROSPECTIVE_POLICY_ID, alpha_policy=manifest.ALPHA_TOTAL,
        reason="initial_freeze", recorded_at_utc="2026-08-23T00:00:00Z",
    ))
    assert ledger.net_spent_for(manifest.PROSPECTIVE_POLICY_ID) == pytest.approx(manifest.ALPHA_TOTAL)


# ---------------------------------------------------------------------
# CLI invocation regression: the exact bug class already found once for
# MLB (bare script path instead of `python -m`) must never recur for NFL.
# ---------------------------------------------------------------------

NFL_PARLAY_V2_CLI_MODULES = (
    "sports.nfl.parlay_v2.run_parlay_v2",
    "sports.nfl.parlay_v2.calibration.ingest",
    "sports.nfl.parlay_v2.calibration.pair_ingest",
    "sports.nfl.research.parlay_certification_v2.settle_evidence",
    "sports.nfl.parlay_v2.freeze_prospective",
    "sports.nfl.scripts.stage_parlay_v2",
)


@pytest.mark.parametrize("module", NFL_PARLAY_V2_CLI_MODULES)
def test_nfl_parlay_v2_cli_module_is_invocable_via_python_dash_m(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"{module} --help failed:\n{result.stderr}"
    assert "ModuleNotFoundError" not in result.stderr
    assert "ImportError" not in result.stderr


def test_nfl_predictions_workflow_never_invokes_parlay_v2_scripts_as_a_bare_path():
    """Text-level guard on .github/workflows/nfl-predictions.yml: unlike
    MLB (which shells out to PARLAY_V2 scripts from inside
    run_daily_predictions.py), NFL's workflow invokes every step directly
    via bash `python ...`. Whenever a PARLAY_V2 CLI module is wired into
    that workflow, each invocation line must use `python -m
    <dotted.module>`, never a bare `.py` path -- this is the exact bug
    that silently broke every MLB PARLAY_V2 CI run since inception (see
    sports/mlb/tests/test_parlay_v2_integration.py's identical guard)."""
    workflow_path = REPO_ROOT / ".github" / "workflows" / "nfl-predictions.yml"
    text = workflow_path.read_text()
    module_basenames = {m.split(".")[-1] for m in NFL_PARLAY_V2_CLI_MODULES}
    for line in text.splitlines():
        if "parlay_v2" not in line and "parlay_certification_v2" not in line:
            continue
        if not any(name in line for name in module_basenames):
            continue
        if ".py" in line and "-m " not in line and "--" not in line.split(".py")[0].split()[-1]:
            pytest.fail(f"possible bare-.py-path invocation of a PARLAY_V2 script found: {line!r}")
        if "python" in line and "-m" not in line:
            pytest.fail(f"PARLAY_V2 invocation without -m module flag: {line!r}")
