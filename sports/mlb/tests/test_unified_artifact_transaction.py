import json

import pytest

from sports.mlb.unified.evidence_ledger import append_generation, append_revision, read_ledger
from sports.mlb.unified.pipeline import validate_payload, write_payload


def valid_payload():
    return {"schema_version":"unified_mlb_v1","generated_at_utc":"2026-08-31T12:00:00Z","generation_id":"g1","run_date":"2026-08-31","policy_hash":"a"*64,"engine_state":"PRODUCTION_DEPLOYED_DARK","evidence":{},"singles":[],"parlays":{"two_leg":[],"three_leg":[],"four_leg":[]},"same_game_parlays":[],"exotic":[],"diagnostics":{},"capabilities":{}}


def test_invalid_replacement_leaves_previous_artifact_intact(tmp_path):
    path = tmp_path / "unified.json"
    good = valid_payload()
    write_payload(good, path)
    bad = dict(good)
    bad["schema_version"] = "wrong"
    with pytest.raises(ValueError):
        write_payload(bad, path)
    assert json.loads(path.read_text()) == good


def test_evidence_append_is_idempotent_and_collision_safe(tmp_path):
    path = tmp_path / "ledger.jsonl"
    row = {"generation_id":"g1","generated_at_utc":"2026-08-31T12:00:00Z"}
    assert append_generation(path, row) is True
    assert append_generation(path, row) is False
    with pytest.raises(ValueError, match="collision"):
        append_generation(path, {**row, "x":1})


def test_settlement_revision_is_append_only_and_hash_linked(tmp_path):
    path = tmp_path / "ledger.jsonl"
    append_generation(path, {"generation_id":"g1","generated_at_utc":"2026-08-31T12:00:00Z","revision":1})
    revision = {"generation_id":"g1","revision":2,"supersedes_revision":1,"settlement":{"c1":"won"}}
    assert append_revision(path, revision) is True
    assert len(read_ledger(path)) == 2
    with pytest.raises(ValueError, match="revision must be 3"):
        append_revision(path, revision)


def test_selected_negative_ev_or_missing_probability_is_invalid():
    payload = valid_payload()
    payload["singles"] = [{"usable_probability":None,"decimal_price":2,"market_break_even_probability":.5,"probability_edge":.1,"conservative_expected_value":.2,"rejection_reasons":[]}]
    with pytest.raises(ValueError, match="usable_probability"):
        validate_payload(payload)
