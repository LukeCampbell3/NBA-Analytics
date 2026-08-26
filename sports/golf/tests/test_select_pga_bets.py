from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLF_PARLAY_V2_ROOT = REPO_ROOT / "sports" / "golf" / "parlay_v2"
GOLF_PREDICTIONS_ROOT = REPO_ROOT / "sports" / "golf" / "predictions"
sys.path.insert(0, str(GOLF_PARLAY_V2_ROOT))
sys.path.insert(0, str(GOLF_PREDICTIONS_ROOT))

from odds_provider import OddsRow  # noqa: E402
from score_model import FieldOutcomeProbabilities  # noqa: E402
from calibration.store import CalibrationStore  # noqa: E402
from calibration.schema import build_observation  # noqa: E402
import select_pga_bets as select  # noqa: E402


def _odds_row(player_name: str, market: str, price: float, book: str = "draftkings") -> OddsRow:
    return OddsRow(
        player_name=player_name, market=market, side="YES", price_american=price,
        sportsbook_key=book, sportsbook_title=book.title(), event_id="evt1", event_name="Test Event",
        commence_time_utc="2026-08-27T13:00:00Z", observed_at_utc="2026-08-24T12:00:00Z",
    )


def test_american_to_decimal_and_implied_probability() -> None:
    assert select.american_to_decimal(150.0) == 2.5
    assert select.american_to_decimal(-200.0) == 1.5
    assert select.implied_probability(-200.0) == 1.0 / 1.5
    assert select.american_to_decimal(0.0) is None  # invalid price, never a real American odds value


def test_no_vig_field_probabilities_normalizes_across_the_whole_field() -> None:
    rows = [
        _odds_row("A", "WINNER", 100.0),  # implied 0.5
        _odds_row("B", "WINNER", 100.0),  # implied 0.5
    ]
    # Raw sum = 1.0 here (no vig baked into this fixture), so normalized == raw.
    probs = select.no_vig_field_probabilities(rows, market="WINNER", sportsbook_key="draftkings")
    assert abs(probs["A"] - 0.5) < 1e-9
    assert abs(probs["B"] - 0.5) < 1e-9


def test_no_vig_field_probabilities_removes_real_vig() -> None:
    # Both priced at -110-equivalent-ish favorites-heavy book -- raw implied sums > 1.0 (real vig).
    rows = [_odds_row("A", "WINNER", -150.0), _odds_row("B", "WINNER", 300.0)]
    raw_a = select.implied_probability(-150.0)
    raw_b = select.implied_probability(300.0)
    probs = select.no_vig_field_probabilities(rows, market="WINNER", sportsbook_key="draftkings")
    assert abs(probs["A"] - raw_a / (raw_a + raw_b)) < 1e-9
    assert abs(sum(probs.values()) - 1.0) < 1e-9


def test_build_candidates_stays_unauthorized_with_no_calibration_ledger() -> None:
    """Matches this repo's established shadow-until-earned discipline:
    with no calibration_store at all, every real, priced, positive-EV
    candidate must still come back candidate_authorized=False."""
    outcomes = [FieldOutcomeProbabilities("p1", "Scottie Scheffler", win_probability=0.30, top5_probability=0.6, top10_probability=0.75, top20_probability=0.9, make_cut_probability=None)]
    odds = [_odds_row("Scottie Scheffler", "WINNER", 250.0), _odds_row("Someone Else", "WINNER", 900.0)]
    candidates = select.build_candidates(outcomes, odds, event_id="evt1")
    winner_candidate = next(c for c in candidates if c.market == "WINNER")
    assert winner_candidate.price_confirmed is True
    assert winner_candidate.expected_value_per_unit is not None
    assert winner_candidate.candidate_authorized is False  # no calibration_store passed -> no support -> never authorized


def test_build_candidates_attaches_real_headshot_url_when_provided() -> None:
    outcomes = [FieldOutcomeProbabilities("p1", "Scottie Scheffler", win_probability=0.30, top5_probability=0.6, top10_probability=0.75, top20_probability=0.9, make_cut_probability=None)]
    odds = [_odds_row("Scottie Scheffler", "WINNER", 250.0), _odds_row("Someone Else", "WINNER", 900.0)]
    candidates = select.build_candidates(
        outcomes, odds, event_id="evt1",
        player_headshots={"p1": "https://a.espncdn.com/i/headshots/golf/players/full/9478.png"},
    )
    winner_candidate = next(c for c in candidates if c.market == "WINNER")
    assert winner_candidate.player_headshot_url == "https://a.espncdn.com/i/headshots/golf/players/full/9478.png"
    assert winner_candidate.as_dict()["player_headshot_url"] == "https://a.espncdn.com/i/headshots/golf/players/full/9478.png"


def test_build_candidates_headshot_url_empty_when_not_provided() -> None:
    outcomes = [FieldOutcomeProbabilities("p1", "Scottie Scheffler", win_probability=0.30, top5_probability=0.6, top10_probability=0.75, top20_probability=0.9, make_cut_probability=None)]
    odds = [_odds_row("Scottie Scheffler", "WINNER", 250.0), _odds_row("Someone Else", "WINNER", 900.0)]
    candidates = select.build_candidates(outcomes, odds, event_id="evt1")
    winner_candidate = next(c for c in candidates if c.market == "WINNER")
    assert winner_candidate.player_headshot_url == ""


def test_build_candidates_produces_no_candidate_for_unpriced_market() -> None:
    """MAKE_CUT probability exists (real, non-cut event reports None so
    this wouldn't apply, but simulate a cut event with no real market
    price posted anywhere) -- must never fabricate a price."""
    outcomes = [FieldOutcomeProbabilities("p1", "Player One", win_probability=0.05, top5_probability=0.2, top10_probability=0.35, top20_probability=0.6, make_cut_probability=0.8)]
    candidates = select.build_candidates(outcomes, odds_rows=[], event_id="evt1")
    make_cut_candidate = next(c for c in candidates if c.market == "MAKE_CUT")
    assert make_cut_candidate.selected_side_price is None
    assert make_cut_candidate.price_confirmed is False
    assert make_cut_candidate.candidate_authorized is False


def test_build_candidates_skips_markets_the_model_has_no_probability_for() -> None:
    """A real no-cut event (make_cut_probability=None) must never produce
    a MAKE_CUT candidate at all -- not an unpriced one, none."""
    outcomes = [FieldOutcomeProbabilities("p1", "Player One", win_probability=0.05, top5_probability=0.2, top10_probability=0.35, top20_probability=0.6, make_cut_probability=None)]
    candidates = select.build_candidates(outcomes, odds_rows=[], event_id="evt1")
    assert all(c.market != "MAKE_CUT" for c in candidates)


def test_build_candidates_becomes_authorized_once_real_calibration_support_exists(tmp_path) -> None:
    """Proves the gate is real and reachable, not permanently closed:
    admitting >=20 real settled observations into market_bucket/line_bucket
    (and 20 independent slates) for this exact candidate must flip
    candidate_authorized to True, with everything else held fixed."""
    store = CalibrationStore(tmp_path / "ledger.jsonl")
    for i in range(25):
        obs = build_observation(
            slate_id=f"slate_{i}", game_id=f"evt_{i}", event_date=f"2026-01-{(i % 28) + 1:02d}",
            player_id="p1", player_name="Scottie Scheffler", target="WINNER", side="YES",
            line=0.0, book="draftkings", quote_decimal=3.5, quote_timestamp="2026-01-01T00:00:00Z",
            prediction_value=0.3, predictive_probability_if_available=0.3,
            state_version="v1", predictive_version="v1",
            market_bucket="WINNER", line_bucket="WINNER|p1", state_bucket="pga_field_relative_form_v1",
            settlement_status="settled", actual_outcome=0.0, actual_unit_return=-1.0,
            decision_frozen_at="2026-01-01T00:00:00Z", settled_at="2026-01-02T00:00:00Z",
            calibration_admitted_at="2026-01-02T00:00:01Z", source_id=f"src_{i}", source_hash=f"hash_{i}",
        )
        store.admit(obs)

    outcomes = [FieldOutcomeProbabilities("p1", "Scottie Scheffler", win_probability=0.30, top5_probability=0.6, top10_probability=0.75, top20_probability=0.9, make_cut_probability=None)]
    # A real field has many priced outcomes, not just two -- with only two
    # priced players, normalizing to remove vig concentrates almost all
    # probability mass on them, swamping any real edge. Model a realistic
    # field: Scheffler at +400 (implied 0.20) plus 9 real-shaped longshots
    # at +900 (implied 0.10 each) so the no-vig field sums close to 1.0
    # and Scheffler's normalized share (~0.18) sits well below the 0.30
    # model probability -- a real, comfortably-above-threshold edge.
    odds = [_odds_row("Scottie Scheffler", "WINNER", 400.0)] + [
        _odds_row(f"Field Player {i}", "WINNER", 900.0) for i in range(9)
    ]
    candidates = select.build_candidates(outcomes, odds, event_id="evt_new", calibration_store=store, calibration_as_of="2026-06-01T00:00:00Z")
    winner_candidate = next(c for c in candidates if c.market == "WINNER")
    assert winner_candidate.probability_edge is not None and winner_candidate.probability_edge >= select.MIN_ABS_EDGE
    assert winner_candidate.candidate_authorized is True


def test_top_candidates_per_market_ranks_by_expected_value() -> None:
    a = select.PgaCandidate("1", "A", "WINNER", 0.3, 0.2, 0.1, 250.0, "dk", 0.5, 3, True, False, [])
    b = select.PgaCandidate("2", "B", "WINNER", 0.1, 0.2, -0.1, 250.0, "dk", -0.2, 3, True, False, [])
    ranked = select.top_candidates_per_market([a, b], max_per_market=5)
    assert ranked[0].player_name == "A"
