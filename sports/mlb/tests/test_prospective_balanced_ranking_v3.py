from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "sports" / "mlb" / "scripts"))

import prospective_balanced_ranking_v3 as prospective  # noqa: E402


def _candidate(player: str = "A Hitter", probability: float = 0.62) -> SimpleNamespace:
    return SimpleNamespace(
        run_date=date(2026, 8, 30), game_id="game-1", player=player,
        target="H", direction="OVER", market_line=0.5, market_source="real",
        price_confirmed=True, selected_side_price=-120.0,
        final_hit_probability=probability, market_implied_probability=0.545,
        selected_sportsbook_key="book", commence_time_utc="2026-08-30T18:00:00Z",
    )


def _patch_candidates(monkeypatch: pytest.MonkeyPatch, candidates: list[SimpleNamespace]) -> None:
    monkeypatch.setattr(prospective.shp, "prepare_candidates", lambda _: (candidates, None, None))
    monkeypatch.setattr(prospective, "parse_v11_args", lambda _: object())


def test_snapshot_is_pregame_only_and_contains_full_ranking_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_candidates(monkeypatch, [_candidate()])
    payload = prospective.build_snapshot(tmp_path / "pool.csv", observed_at_utc="2026-08-30T12:00:00Z")
    encoded = json.dumps(payload).lower()
    assert "result" not in encoded
    assert '"win"' not in encoded
    assert payload["candidate_count"] == 1
    assert set(prospective.ranking.SCORE_FIELDS) <= set(payload["candidates"][0])


def test_capture_is_idempotent_but_conflicting_scientific_content_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_candidates(monkeypatch, [_candidate()])
    path, first = prospective.capture(tmp_path / "pool.csv", tmp_path / "evidence", observed_at_utc="2026-08-30T12:00:00Z")
    _, second = prospective.capture(tmp_path / "pool.csv", tmp_path / "evidence", observed_at_utc="2026-08-30T12:01:00Z")
    assert first == "created" and second == "unchanged"
    original = path.read_bytes()
    _patch_candidates(monkeypatch, [_candidate(probability=0.70)])
    with pytest.raises(RuntimeError, match="immutable artifact conflict"):
        prospective.capture(tmp_path / "pool.csv", tmp_path / "evidence", observed_at_utc="2026-08-30T12:02:00Z")
    assert path.read_bytes() == original


def test_settlement_requires_later_date_and_links_snapshot_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_candidates(monkeypatch, [_candidate()])
    snapshot_path, _ = prospective.capture(tmp_path / "pool.csv", tmp_path / "evidence", observed_at_utc="2026-08-30T12:00:00Z")
    with pytest.raises(ValueError, match="strictly after"):
        prospective.settle_snapshot(snapshot_path, tmp_path / "evidence", settled_at_utc="x", as_of_date=date(2026, 8, 30))
    monkeypatch.setattr(prospective, "build_actual_lookup", lambda _: {("2026-08-30", "a_hitter", "H", "game-1"): 1.0})
    monkeypatch.setattr(prospective, "grade_result", lambda *_: "win")
    settlement_path, status = prospective.settle_snapshot(
        snapshot_path, tmp_path / "evidence", settled_at_utc="2026-08-31T12:00:00Z", as_of_date=date(2026, 8, 31)
    )
    snapshot = json.loads(snapshot_path.read_text())
    settlement = json.loads(settlement_path.read_text())
    assert status == "created"
    assert settlement["snapshot_identity_sha256"] == snapshot["identity_sha256"]
    assert "balanced_probability" not in settlement["results"][0]


def test_loader_rejects_wrong_hash_and_counts_slates_not_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_candidates(monkeypatch, [_candidate("A"), _candidate("B")])
    snapshot_path, _ = prospective.capture(tmp_path / "pool.csv", tmp_path / "evidence", observed_at_utc="2026-08-30T12:00:00Z")
    snapshot = json.loads(snapshot_path.read_text())
    settlement_path = snapshot_path.with_name(prospective.SETTLEMENT_NAME)
    settlement_path.write_text(json.dumps({
        "snapshot_identity_sha256": snapshot["identity_sha256"],
        "results": [{"candidate_id": row["candidate_id"], "win": index % 2} for index, row in enumerate(snapshot["candidates"])],
    }))
    rows = prospective.load_settled_rows(tmp_path / "evidence")
    assert len(rows) == 2
    assert len({row["date"] for row in rows}) == 1
    settlement = json.loads(settlement_path.read_text())
    settlement["snapshot_identity_sha256"] = "wrong"
    settlement_path.write_text(json.dumps(settlement))
    with pytest.raises(RuntimeError, match="does not reference"):
        prospective.load_settled_rows(tmp_path / "evidence")


def test_snapshot_keeps_rejected_candidates_and_filters_other_families(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rejected = _candidate(probability=0.55)
    other = _candidate("Other")
    other.target = "HR"
    _patch_candidates(monkeypatch, [rejected, other])
    payload = prospective.build_snapshot(tmp_path / "pool.csv", observed_at_utc="x")
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["v19_eligible"] is False
