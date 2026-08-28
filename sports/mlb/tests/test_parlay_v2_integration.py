from __future__ import annotations

import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from sports.mlb.parlay_v2.candidate_adapter import (
    Leg,
    PairCandidate,
    build_candidates_for_day,
    build_pregame_action_rows,
    exact_event_key,
)
from sports.mlb.parlay_v2.calibration.schema import build_observation
from sports.mlb.parlay_v2.calibration.store import CalibrationStore
from sports.mlb.parlay_v2.frontend_payload import embed_parlays_v2
from sports.mlb.parlay_v2.program_alpha import AlphaSpend, ProgramAlphaLedger
from sports.mlb.parlay_v2.run_parlay_v2 import _to_candidate_wager, build_slate_payload
from sports.mlb.research.parlay_certification_v2 import manifest
from sports.mlb.research.parlay_certification_v2.eligibility import EligibilityInputs
from sports.mlb.research.parlay_certification_v2.policy import select_action_for_day

REPO_ROOT = Path(__file__).resolve().parents[3]
PARLAY_V2_PKG = REPO_ROOT / "sports" / "mlb" / "parlay_v2"
PARLAY_CERT_V2_PKG = REPO_ROOT / "sports" / "mlb" / "research" / "parlay_certification_v2"
OLD_SINGLES_PREDICTOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "select_high_precision_predictions.py"
OLD_PARLAY_SELECTOR = REPO_ROOT / "sports" / "mlb" / "scripts" / "select_daily_parlay.py"


def _imported_module_roots(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text())
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0] + ("." + alias.name.split(".")[1] if "." in alias.name else ""))
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module)
    return roots


def _make_row(*, player, player_key, game_id, target, direction, line, over_price=-150, under_price=130, prediction=None, rmse=0.4, history_rows=50, market_source="real"):
    pred = prediction if prediction is not None else (line + 0.3)
    return {
        "Player": player, "Player_ID": player_key, "Game_ID": game_id, "Target": target,
        "Prediction": pred, "Market_Line": line,
        "Market_Over_Price": over_price, "Market_Under_Price": under_price,
        "Model_Val_RMSE": rmse, "History_Rows": history_rows, "Market_Source": market_source,
        "Player_Type": "hitter", "Opposing_Pitcher": "Test Starter",
        "Game_Date": "2026-08-21",
    }


# ======================================================================
# A. Singles isolation
# ======================================================================


def test_old_singles_predictor_never_imports_v2():
    imports = _imported_module_roots(OLD_SINGLES_PREDICTOR)
    assert not any("parlay_v2" in m or "parlay_certification_v2" in m for m in imports), (
        "the old single-bet predictor must be structurally incapable of being affected by V2 config"
    )


# ======================================================================
# B. Parlay isolation (V2 does not depend on old ranking/selection code)
# ======================================================================


def test_v2_package_never_imports_old_selection_scripts():
    for py_file in list(PARLAY_V2_PKG.glob("*.py")) + list(PARLAY_CERT_V2_PKG.glob("*.py")):
        imports = _imported_module_roots(py_file)
        assert not any("select_high_precision_predictions" in m or "select_daily_parlay" in m for m in imports), (
            f"{py_file} must not depend on the old ranking/selection scripts (found in imports: {imports})"
        )


def test_legacy_control_is_read_only_diagnostic_never_an_input_to_policy():
    # legacy_control.py may be imported by comparison.py (diagnostics) but
    # never by candidate_adapter.py, run_parlay_v2.py's decision path, or
    # anything in parlay_certification_v2/.
    for name in ("candidate_adapter.py",):
        imports = _imported_module_roots(PARLAY_V2_PKG / name)
        assert not any("legacy_control" in m for m in imports)
    for py_file in PARLAY_CERT_V2_PKG.glob("*.py"):
        imports = _imported_module_roots(py_file)
        assert not any("legacy_control" in m for m in imports)


# ======================================================================
# C. Authority separation
# ======================================================================


def test_no_old_system_field_can_reach_state_machine_or_manifest():
    for py_file in PARLAY_CERT_V2_PKG.glob("*.py"):
        imports = _imported_module_roots(py_file)
        assert not any(
            "select_high_precision_predictions" in m or "select_daily_parlay" in m or "parlay_analysis" in m
            for m in imports
        ), f"{py_file} must not import any old-system module"
    assert manifest.PRODUCTION_AUTHORIZED is False


# ======================================================================
# D. Prediction/certification separation
# ======================================================================


FORBIDDEN_AUTHORITATIVE_KEYS = {"certified", "safe", "supported", "production_authorized", "risk_passed"}


def _walk_keys(obj, found: set[str]):
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k).lower())
            _walk_keys(v, found)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _walk_keys(item, found)


def test_pair_candidate_cannot_contain_authoritative_fields():
    leg = Leg("P", "p1", "g1", "H", "OVER", 0.5, "real", 1.5, "2026-08-21", 0.6, True)
    leg2 = Leg("Q", "q1", "g2", "H", "OVER", 0.5, "real", 1.6, "2026-08-21", 0.55, True)
    candidate = PairCandidate(
        slate_id="s", candidate_id="c1", leg_1=leg, leg_2=leg2,
        joint_probability_estimate=0.3, joint_probability_method="independence_binary_world_model",
        joint_score=0.9, support={"leg_1_support": True, "leg_2_support": True, "state_support": True, "in_support": True},
        world_diagnostics={"retained_world_count": 4, "retained_probability_mass": 1.0, "counterexample_count": 0, "counterexample_mass": 0.0, "nonvacuous_world_certificate": True},
        predictive_version="v1", state_version="s1", adapter_version="a1",
    )
    found: set[str] = set()
    _walk_keys(candidate.as_dict(), found)
    assert found.isdisjoint(FORBIDDEN_AUTHORITATIVE_KEYS)


# ======================================================================
# E. Alternate-line identity -- MANDATORY regression test (mission 5/14)
# ======================================================================


def test_alternate_lines_never_share_probability_or_price():
    """Player hits OVER 0.5 at -200 and OVER 1.5 at +300 on the same day:
    the system MUST NOT copy the -200-implied probability/price to the 1.5
    event, or vice versa."""
    pool = pd.DataFrame([
        _make_row(player="Ohtani", player_key="ohtani", game_id="g1", target="H", direction="OVER", line=0.5, over_price=-200, prediction=0.8),
        _make_row(player="Ohtani", player_key="ohtani", game_id="g1", target="H", direction="OVER", line=1.5, over_price=300, prediction=1.8),
    ])
    rows = build_pregame_action_rows(pool, stamp="20260821", mode="broad", targets=("H",))
    ohtani_rows = rows[rows["player"] == "Ohtani"].sort_values("market_line")
    assert len(ohtani_rows) == 2, "both alternate-line events must survive as distinct rows, not be deduped"

    half_line = ohtani_rows.iloc[0]
    full_line = ohtani_rows.iloc[1]
    assert half_line["market_line"] == pytest.approx(0.5)
    assert full_line["market_line"] == pytest.approx(1.5)
    # Prices must be scoped to their own line, never swapped/shared.
    assert half_line["decimal_price"] == pytest.approx(1.0 + 100.0 / 200.0)  # -200 -> 1.50
    assert full_line["decimal_price"] == pytest.approx(1.0 + 300.0 / 100.0)  # +300 -> 4.00
    assert half_line["decimal_price"] != pytest.approx(full_line["decimal_price"])
    # Probabilities must differ (different predictions/lines feed the frozen marginal model).
    assert half_line["marginal_probability"] != pytest.approx(full_line["marginal_probability"])


def test_exact_event_key_distinguishes_lines():
    key_half = exact_event_key("ohtani", "g1", "H", "OVER", 0.5)
    key_full = exact_event_key("ohtani", "g1", "H", "OVER", 1.5)
    assert key_half != key_full


# ======================================================================
# F. Missing exact price
# ======================================================================


def test_missing_price_excludes_candidate_from_economic_evaluation():
    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="H", direction="OVER", line=0.5, over_price=None, under_price=None),
    ])
    rows = build_pregame_action_rows(pool, stamp="20260821", mode="broad", targets=("H",))
    assert rows.empty, "a row with no real price must never enter the action universe"


def test_hitter_without_opposing_probable_starter_is_excluded():
    row = _make_row(player="A", player_key="a", game_id="g1", target="H", direction="OVER", line=0.5)
    row["Opposing_Pitcher"] = ""
    rows = build_pregame_action_rows(pd.DataFrame([row]), stamp="20260821", mode="broad", targets=("H",))
    assert rows.empty


# ======================================================================
# G. Cross-game pair / H. Same-game pair
# ======================================================================


def test_cross_game_pair_gets_product_price_same_game_pair_gets_none():
    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="H", direction="OVER", line=0.5, over_price=-150, prediction=0.9),
        _make_row(player="B", player_key="b", game_id="g2", target="H", direction="OVER", line=0.5, over_price=-130, prediction=0.9),
        _make_row(player="C", player_key="c", game_id="g1", target="H", direction="OVER", line=0.5, over_price=-140, prediction=0.9),
    ])
    rows = build_pregame_action_rows(pool, stamp="20260821", mode="broad", targets=("H",))
    candidates = build_candidates_for_day(rows, slate_id="20260821", aps_threshold=1.0, calibration_slates=25, predictive_version="v1", state_version="s1")
    cross = next(c for c in candidates if c.leg_1.game_id != c.leg_2.game_id)
    same = next(c for c in candidates if c.leg_1.game_id == c.leg_2.game_id)

    cross_wager = _to_candidate_wager(cross)
    same_wager = _to_candidate_wager(same)
    assert cross_wager.decimal_price == pytest.approx(cross.leg_1.decimal_price * cross.leg_2.decimal_price)
    assert same_wager.decimal_price is None


def test_same_game_pair_without_real_sgp_quote_never_produces_executable_action():
    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="H", direction="OVER", line=0.5, over_price=-150, prediction=0.95),
        _make_row(player="C", player_key="c", game_id="g1", target="H", direction="OVER", line=0.5, over_price=-140, prediction=0.95),
    ])
    rows = build_pregame_action_rows(pool, stamp="20260821", mode="broad", targets=("H",))
    candidates = build_candidates_for_day(rows, slate_id="20260821", aps_threshold=1.0, calibration_slates=25, predictive_version="v1", state_version="s1")
    assert len(candidates) == 1 and candidates[0].leg_1.game_id == candidates[0].leg_2.game_id
    wager = _to_candidate_wager(candidates[0])
    from sports.mlb.research.parlay_certification_v2.eligibility import EligibilityInputs, evaluate_eligibility
    elig = evaluate_eligibility(EligibilityInputs("20260821", True, True, True, True))
    selection = select_action_for_day(elig, [wager], r_max=manifest.R_MAX_ACCEPTED)
    assert selection.action == 0


# ======================================================================
# I. One action per eligible slate
# ======================================================================


def test_at_most_one_action_per_slate():
    assert manifest.MAX_ACTIONS_PER_ELIGIBLE_SLATE == 1
    pool = pd.DataFrame([
        _make_row(player=f"P{i}", player_key=f"p{i}", game_id=f"g{i}", target="H", direction="OVER", line=0.5, over_price=-150, prediction=0.9)
        for i in range(6)
    ])
    payload = build_slate_payload(
        pool_csv=pool, slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", True, True, True, True),
        predictive_version="v1", state_version="s1",
    )
    # With FROZEN_CALIBRATION_SLATES=0 this abstains (CERTIFICATION_STREAM_NOT_READY);
    # the structural guarantee under test is that `selected_parlay` is
    # never a list/multiple wagers -- at most one candidate object or None.
    assert payload["selected_parlay"] is None or isinstance(payload["selected_parlay"], dict)


# ======================================================================
# J. Abstention: E=1, A=0 (never E=0)
# ======================================================================


def test_eligible_slate_with_no_qualifying_pair_stays_eligible():
    pool = pd.DataFrame([_make_row(player="A", player_key="a", game_id="g1", target="H", direction="OVER", line=0.5, over_price=None, under_price=None)])
    payload = build_slate_payload(
        pool_csv=pool, slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", True, True, True, True),
        predictive_version="v1", state_version="s1",
    )
    assert payload["eligible"] is True
    assert payload["action"] == "ABSTAIN"
    assert payload["abstain_reason"] != "OPERATIONALLY_INELIGIBLE"


def test_operationally_ineligible_slate_is_e0_not_an_abstention():
    payload = build_slate_payload(
        pool_csv=pd.DataFrame(), slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", False, False, True, True),
        predictive_version="v1", state_version="s1",
    )
    assert payload["eligible"] is False
    assert payload["abstain_reason"] == "OPERATIONALLY_INELIGIBLE"


# ======================================================================
# K. Frontend contract
# ======================================================================


def test_frontend_payload_embedding_never_touches_existing_keys():
    original = {"plays": ["singles"], "daily_parlay": {"legacy": True}, "summary": {"n": 1}}
    embedded = embed_parlays_v2(original, None)
    assert embedded["plays"] == ["singles"]
    assert embedded["daily_parlay"] == {"legacy": True}
    assert embedded["summary"] == {"n": 1}
    assert embedded["parlays"]["system"] == "PARLAY_POLICY_V2"
    assert "parlays" not in original  # original dict is not mutated


def test_frontend_payload_embedding_reads_real_v2_json(tmp_path):
    v2_path = tmp_path / "parlay_v2.json"
    v2_path.write_text(json.dumps({"system": "PARLAY_POLICY_V2", "action": "ABSTAIN", "policy_status": "DEVELOPMENT"}))
    embedded = embed_parlays_v2({"plays": []}, v2_path)
    assert embedded["parlays"]["policy_status"] == "DEVELOPMENT"


def test_predictions_html_has_the_parlay_v2_content_and_no_legacy_parlay_section():
    html = (REPO_ROOT / "sports" / "mlb" / "web" / "predictions.html").read_text()
    # The board was later simplified into one plain container (no
    # separate per-product sections/headings -- see predictions.html's
    # own comment) -- parlayV2Content lives directly in #board rather
    # than inside its own #parlayV2Section wrapper.
    assert 'id="board"' in html
    assert 'id="parlayV2Content"' in html
    assert 'id="parlayV2Section"' not in html
    # The legacy ticket system is no longer shown as its own "parlay"
    # section -- its legs are folded into the main board instead (see
    # mergeLegacySoloBets), since that system was never V2-certified.
    assert 'id="dailyParlaySection"' not in html


def test_predictions_js_renders_parlay_v2_and_folds_legacy_legs_into_solo_bets():
    js = (REPO_ROOT / "sports" / "mlb" / "web" / "predictions.js").read_text()
    assert "renderParlayV2" in js
    assert "this.data?.parlays" in js
    assert "renderDailyParlay" not in js  # legacy ticket renderer removed, not just unused
    assert "mergeLegacySoloBets" in js
    assert "this.data?.daily_parlay" in js  # legacy data still READ, just no longer given its own "parlay" section


# ======================================================================
# L. Replay does not invoke the old singles predictor
# ======================================================================


def test_evidence_store_never_imports_old_predictor():
    imports = _imported_module_roots(PARLAY_CERT_V2_PKG / "evidence_store.py")
    assert not any("select_high_precision_predictions" in m for m in imports)


# ======================================================================
# M. Program alpha never exceeds alpha_program
# ======================================================================


def test_program_alpha_ledger_enforces_budget(tmp_path):
    ledger = ProgramAlphaLedger(tmp_path / "ledger.json", alpha_program=0.05)
    ledger.spend(AlphaSpend("POLICY_A", 0.03, "frozen_for_prospective_confirmation", "2026-08-21T00:00:00Z"))
    ledger.spend(AlphaSpend("POLICY_B", 0.015, "frozen_for_prospective_confirmation", "2026-08-22T00:00:00Z"))
    assert ledger.total_spent() == pytest.approx(0.045)
    with pytest.raises(ValueError):
        ledger.spend(AlphaSpend("POLICY_C", 0.01, "frozen_for_prospective_confirmation", "2026-08-23T00:00:00Z"))
    # idempotent re-spend for an already-recorded version is a no-op, not an error
    ledger.spend(AlphaSpend("POLICY_A", 0.03, "frozen_for_prospective_confirmation", "2026-08-21T00:00:00Z"))
    assert ledger.total_spent() == pytest.approx(0.045)


def test_manifest_d_max_derives_r_max():
    assert manifest.R_MAX_ACCEPTED == pytest.approx(manifest.D_MAX - 1.0)


# ======================================================================
# Comparison artifact (mission section 16) -- diagnostic only, immutable once frozen
# ======================================================================


def test_comparison_record_is_write_once_and_settlement_is_additive(tmp_path):
    from sports.mlb.parlay_v2.comparison import build_comparison_record, settle_comparison_record, write_comparison_record
    from sports.mlb.parlay_v2.legacy_control import LegacyParlayControl

    legacy = LegacyParlayControl(True, [{"player": "Kwan", "target": "H", "line": 0.5}], 0.42, {"combined_american_price": 260}, "old_parlay_diagnostic_loaded")
    record = build_comparison_record(
        date="20260821", policy_version="TEST_POLICY", legacy=legacy,
        new_v2_pair=[{"player": "Judge", "target": "H", "line": 0.5}], new_joint_score=0.9,
        new_quote={"decimal_price": 3.5}, new_action="ACT", new_policy_status="DEVELOPMENT",
    )
    assert record.same_pair is False  # different players -- correctly detected as NOT the same pair
    root = tmp_path / "comparison"
    assert write_comparison_record(root, record) is True
    assert write_comparison_record(root, record) is False  # frozen once written
    assert settle_comparison_record(root, "20260821", "TEST_POLICY", old_settlement={"result": "loss"}, new_settlement={"result": "win"}) is True

    import json
    saved = json.loads((root / "20260821_TEST_POLICY.json").read_text())
    assert saved["eventual_new_settlement"] == {"result": "win"}
    assert saved["new_v2_candidate"] == [{"player": "Judge", "target": "H", "line": 0.5}]  # decision-time field untouched by settlement


def test_comparison_and_legacy_control_never_touch_v2_authority():
    for name in ("comparison.py", "legacy_control.py"):
        imports = _imported_module_roots(PARLAY_V2_PKG / name)
        assert not any("parlay_certification_v2.policy" in m or "parlay_certification_v2.state_machine" in m for m in imports)


# ======================================================================
# Shadow candidate: always shown on an abstain day, never a substitute action
# ======================================================================


def test_shadow_candidate_present_when_no_calibration_ledger_but_action_stays_abstain():
    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="R", direction="OVER", line=0.5, over_price=-150, prediction=0.9),
        _make_row(player="B", player_key="b", game_id="g2", target="R", direction="OVER", line=0.5, over_price=-130, prediction=0.9),
    ])
    payload = build_slate_payload(
        pool_csv=pool, slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", True, True, True, True),
        predictive_version="v1", state_version="s1",
        # no calibration_store -- must not block the shadow candidate
    )
    assert payload["action"] == "ABSTAIN"
    assert payload["selected_parlay"] is None
    assert payload["shadow_candidate"] is not None
    legs = {payload["shadow_candidate"]["leg_1"]["player"], payload["shadow_candidate"]["leg_2"]["player"]}
    assert legs == {"A", "B"}
    # manifest.STATUS is now FROZEN_PROSPECTIVE_INCONCLUSIVE (the mission's
    # own deliberate freeze), so this reaches REAL support evaluation with
    # zero accumulated ledger data -- an honest, specific reason
    # (NO_STATE_SUPPORT), never the old generic circular one.
    assert payload["abstain_reason"] == "NO_STATE_SUPPORT"
    assert payload["selection_status"] == "ABSTAIN"
    assert payload["shadow_execution_status"] == "NOT_EXECUTED"
    assert payload["staking_authorized"] is False


def test_policy_not_frozen_guard_blocks_selection_before_status_is_advanced(monkeypatch):
    """The POLICY_NOT_FROZEN guard itself, isolated from the current
    default STATUS: with STATUS forced back to DEVELOPMENT, selection
    must abstain for that reason specifically, before any support
    evaluation runs -- the shadow candidate display is unaffected."""
    monkeypatch.setattr(manifest, "STATUS", "DEVELOPMENT")
    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="R", direction="OVER", line=0.5, over_price=-150, prediction=0.9),
        _make_row(player="B", player_key="b", game_id="g2", target="R", direction="OVER", line=0.5, over_price=-130, prediction=0.9),
    ])
    payload = build_slate_payload(
        pool_csv=pool, slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", True, True, True, True),
        predictive_version="v1", state_version="s1",
    )
    assert payload["abstain_reason"] == "POLICY_NOT_FROZEN"
    assert payload["shadow_candidate"] is not None


def test_no_circular_block_once_policy_is_frozen_and_ledger_is_empty(monkeypatch):
    """THE FIX this mission makes, exercised end to end: once
    POLICY_NOT_FROZEN is lifted (a frozen policy_status), an EMPTY or
    missing calibration ledger must still abstain -- but for an honest,
    specific REQUIRED-support reason (state_support has zero accumulated
    slates), never for the old generic circular reason. joint_support/
    shift_status being permanently UNESTABLISHED must never appear as the
    reason at all."""
    monkeypatch.setattr(manifest, "STATUS", "FROZEN_PROSPECTIVE_INCONCLUSIVE")
    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="R", direction="OVER", line=0.5, over_price=-150, prediction=0.9),
        _make_row(player="B", player_key="b", game_id="g2", target="R", direction="OVER", line=0.5, over_price=-130, prediction=0.9),
    ])
    payload = build_slate_payload(
        pool_csv=pool, slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", True, True, True, True),
        predictive_version="v1", state_version="s1",
        # still no calibration_store
    )
    assert payload["action"] == "ABSTAIN"
    assert payload["abstain_reason"] in ("NO_STATE_SUPPORT", "NO_LEG_MARKET_SUPPORT", "NO_LEG_LINE_SUPPORT")
    assert payload["abstain_reason"] not in ("CERTIFICATION_STREAM_NOT_READY", "NO_PAIR_IN_SUPPORT", "POLICY_NOT_FROZEN")
    assert payload["shadow_candidate"] is not None  # diagnostic display is unaffected


def test_support_no_longer_blocks_once_required_dimensions_accumulate(tmp_path, monkeypatch):
    """THE core deliverable of this mission, end to end: with the policy
    frozen and a calibration ledger carrying real accumulated evidence for
    BOTH legs' market/line/state buckets, the candidate reaches the real
    certified-policy machinery (select_action_for_day) at all -- something
    the old all-five-dimensions-required rule made permanently impossible
    (joint_support/shift_status could never pass, so no candidate ever got
    this far). It is correctly NOT expected to reach ACT here: the
    separate, deliberately conservative, UNTOUCHED G_C/G_L/G_V world
    certificate (FROZEN_APS_THRESHOLD=1.0, retain-all) still requires zero
    loss-counterexample mass, which no non-deterministic prediction can
    satisfy -- that is a real, intentional bottleneck this mission must
    NOT weaken, wholly separate from the circular support bug it does
    fix. The proof of the fix is exactly that the abstain reason changes
    from a support-blocking one (or the old circular one) to the
    certificate's own honest reason."""
    monkeypatch.setattr(manifest, "STATUS", "FROZEN_PROSPECTIVE_INCONCLUSIVE")
    predictive_version, state_version = "v1", "s1"
    state_bucket = f"{predictive_version}|{state_version}"

    # calibration_admitted_at/settled_at must be REAL ISO-with-dashes
    # timestamps strictly in the past (matching production's
    # calibration/ingest.py, which always sets both to
    # datetime.now(timezone.utc).isoformat()) -- a compact "YYYYMMDD..."
    # string sorts lexically AFTER a dashed ISO "now" cutoff and would be
    # wrongly filtered out by the forward-only guard.
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    for i in range(25):
        slate_dt = base + timedelta(days=i)
        slate_id = slate_dt.strftime("%Y%m%d")
        settled_at = (slate_dt + timedelta(hours=23)).isoformat()
        admitted_at = (slate_dt + timedelta(hours=23, minutes=30)).isoformat()
        for player, line in (("A", 0.5), ("B", 0.5)):
            obs = build_observation(
                slate_id=slate_id, game_id=f"g{i}", event_date=slate_id,
                player_id=f"{player.lower()}{i}", player_name=f"{player}{i}",
                target="R", side="OVER", line=line, book="real",
                quote_decimal=1.9, quote_timestamp=f"{slate_id}T12:00:00Z",
                prediction_value=0.6, predictive_probability_if_available=0.6,
                state_version=state_version, predictive_version=predictive_version,
                market_bucket="R", line_bucket=f"R|OVER|{line}", state_bucket=state_bucket,
                decision_frozen_at=f"{slate_id}T17:00:00Z",
                settled_at=settled_at, settlement_status="win",
                actual_outcome=1.0, actual_unit_return=0.9,
                calibration_admitted_at=admitted_at,
                source_id=f"src{i}_{player}", source_hash="h",
            )
            store.admit(obs)

    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="R", direction="OVER", line=0.5, over_price=-150, prediction=0.9),
        _make_row(player="B", player_key="b", game_id="g2", target="R", direction="OVER", line=0.5, over_price=-130, prediction=0.9),
    ])
    payload = build_slate_payload(
        pool_csv=pool, slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", True, True, True, True),
        predictive_version=predictive_version, state_version=state_version,
        calibration_store=store,
    )
    # THE FIX: reaches the real certificate machinery -- proven by a
    # candidate_universe_size > 0 decision record and an abstain reason
    # that comes from the certificate, not from support or the old
    # circular block.
    assert payload["abstain_reason"] == "NO_PAIR_PASSES_FROZEN_POLICY"
    assert payload["abstain_reason"] not in (
        "CERTIFICATION_STREAM_NOT_READY", "NO_PAIR_IN_SUPPORT", "POLICY_NOT_FROZEN",
        "NO_STATE_SUPPORT", "NO_LEG_MARKET_SUPPORT", "NO_LEG_LINE_SUPPORT", "NO_CANDIDATES",
    )
    assert payload["decision_record"]["candidate_universe_size"] > 0
    assert payload["action"] == "ABSTAIN"
    assert payload["selection_status"] == "ABSTAIN"
    assert payload["shadow_execution_status"] == "NOT_EXECUTED"
    assert payload["staking_authorized"] is False


def test_shadow_candidate_excludes_same_game_and_unpriced_pairs():
    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="R", direction="OVER", line=0.5, over_price=-150, prediction=0.9),
        _make_row(player="C", player_key="c", game_id="g1", target="R", direction="OVER", line=0.5, over_price=-140, prediction=0.9),  # same game as A
        _make_row(player="D", player_key="d", game_id="g2", target="R", direction="OVER", line=0.5, over_price=None, under_price=None, prediction=0.9),  # unpriced
    ])
    payload = build_slate_payload(
        pool_csv=pool, slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", True, True, True, True),
        predictive_version="v1", state_version="s1",
    )
    # Only A+C (same game) and A+D-priceless exist as pairs -- neither
    # qualifies as a shadow candidate (same-game or unpriced), so none is shown.
    assert payload["shadow_candidate"] is None


def test_shadow_candidate_never_appears_as_selected_parlay():
    """Structural guarantee: shadow_candidate and selected_parlay are
    always mutually exclusive in what they IMPLY -- selected_parlay only
    appears when action == ACT, and action only becomes ACT through
    select_action_for_day's real certification, never by promoting the
    shadow candidate."""
    pool = pd.DataFrame([
        _make_row(player="A", player_key="a", game_id="g1", target="R", direction="OVER", line=0.5, over_price=-150, prediction=0.9),
        _make_row(player="B", player_key="b", game_id="g2", target="R", direction="OVER", line=0.5, over_price=-130, prediction=0.9),
    ])
    payload = build_slate_payload(
        pool_csv=pool, slate_id="20260821",
        eligibility_inputs=EligibilityInputs("20260821", True, True, True, True),
        predictive_version="v1", state_version="s1",
    )
    assert payload["action"] == "ABSTAIN"
    assert payload["selected_parlay"] is None
    assert payload["shadow_candidate"] is not None  # present as a diagnostic
    # The two are never the same object/claim -- shadow_candidate carries
    # no certification-implying fields (reuses the same check as PairCandidate).
    found: set[str] = set()
    _walk_keys(payload["shadow_candidate"], found)
    assert found.isdisjoint(FORBIDDEN_AUTHORITATIVE_KEYS)


# ======================================================================
# N. CLI entry points are invocable exactly how run_daily_predictions.py
# invokes them -- regression coverage for a real production bug: every
# PARLAY_V2 CLI script uses absolute `sports.*` AND relative `from .x`
# imports, so invoking it as a bare script path (`python /abs/path.py`)
# leaves sys.path/__package__ unset for either to resolve. That failure
# (ModuleNotFoundError: No module named 'sports') was silently swallowed
# by run_daily_predictions.py's own deliberate best-effort try/except on
# every single real CI run, so the Parlays tab reported
# PARLAY_V2_ARTIFACT_UNAVAILABLE forever and none of PARLAY_V2's pipeline
# (calibration ingestion, pair ingestion, evidence settlement, the policy
# runner itself) had ever actually executed in production.
# ======================================================================

PARLAY_V2_CLI_MODULES = (
    "sports.mlb.parlay_v2.run_parlay_v2",
    "sports.mlb.parlay_v2.calibration.ingest",
    "sports.mlb.parlay_v2.calibration.pair_ingest",
    "sports.mlb.research.parlay_certification_v2.settle_evidence",
)


@pytest.mark.parametrize("module", PARLAY_V2_CLI_MODULES)
def test_parlay_v2_cli_module_is_invocable_via_python_dash_m(module):
    """Exactly how run_daily_predictions.py invokes these -- `--help`
    exits 0 with no data needed, so a clean pass here means every import
    (absolute AND relative) resolved correctly."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert "ImportError" not in result.stderr


def test_run_daily_predictions_never_invokes_parlay_v2_scripts_as_a_bare_path():
    """Structural guard against reintroducing the bug above for this
    script or any future PARLAY_V2 CLI added to the daily pipeline: parse
    run_daily_predictions.py's own source and confirm every subprocess
    argv list that references a sports.mlb.parlay_v2 /
    sports.mlb.research.parlay_certification_v2 module uses `-m`, never a
    bare script path."""
    source = (REPO_ROOT / "sports" / "site" / "pipeline" / "run_daily_predictions.py").read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        elements = [ast.literal_eval(e) if isinstance(e, ast.Constant) else None for e in node.elts]
        joined = " ".join(str(e) for e in elements if e is not None)
        if "parlay_v2" not in joined and "parlay_certification_v2" not in joined:
            continue
        # A bare-script invocation would embed a "*.py" path string as one
        # of the argv elements; a correct `-m` invocation never does.
        assert not any(isinstance(e, str) and e.endswith(".py") and ("parlay_v2" in e or "parlay_certification_v2" in e) for e in elements), (
            f"found a bare .py script path referencing parlay_v2/parlay_certification_v2 in argv list: {elements}"
        )
        assert "-m" in elements, f"expected -m module invocation in argv list: {elements}"
