import numpy as np

from sports.mlb.unified.parlay import TicketPolicy, construct_ticket_class
from sports.mlb.unified.schemas import BetCandidate
from sports.mlb.unified.trajectory import TrajectoryBatch, simulate_team_runs


def candidate(identifier, *, game="g", p=.75, price=1.5, ev=.125, mask=None):
    return BetCandidate(identifier, game, "player", identifier, "A", "B", "batter_hits", "game", f"{game}:{identifier}", "over", .5, "fanduel", "m", identifier, None, price, p, None, p, p, 0, p, conservative_expected_value=ev, support_status="SUPPORTED", lineup_status="CONFIRMED", role_status="CONFIRMED", identity_status="CONFIRMED", trajectory_mask_reference=mask)


def test_price_trap_and_negative_ev_never_enter_ticket():
    good = [candidate("a", game="a"), candidate("b", game="b")]
    trap = candidate("trap", game="c", price=1.02, ev=.01)
    loss = candidate("loss", game="d", price=2, ev=-.1)
    tickets, counts = construct_ticket_class(good + [trap, loss], TicketPolicy(2, minimum_joint_probability=.1))
    assert len(tickets) == 1
    assert {leg.candidate_id for leg in tickets[0].legs} == {"a", "b"}
    assert counts["price_trap"] == 1


def test_leg_counts_are_independent_and_search_is_bounded():
    pool = [candidate(str(i), game=str(i)) for i in range(10)]
    two, c2 = construct_ticket_class(pool, TicketPolicy(2, top_k=4, minimum_joint_probability=.1))
    three, c3 = construct_ticket_class(pool, TicketPolicy(3, top_k=4, minimum_joint_probability=.1))
    assert c2["enumerated"] == 6 and c3["enumerated"] == 4
    assert all(ticket.leg_count == 2 for ticket in two)
    assert all(ticket.leg_count == 3 for ticket in three)


def test_same_game_requires_and_uses_common_world_masks():
    batch = TrajectoryBatch(np.zeros((4, 9), dtype=int), np.zeros((4, 9), dtype=int), masks={"a": np.array([1,1,0,0], bool), "b": np.array([1,0,1,0], bool)})
    batch.validate()
    legs = [candidate("a", mask="a", price=3.0), candidate("b", mask="b", price=3.0)]
    rejected, _ = construct_ticket_class(legs, TicketPolicy(2, minimum_joint_probability=.1))
    accepted, _ = construct_ticket_class(legs, TicketPolicy(2, minimum_joint_probability=.1), trajectories={"g": batch})
    assert rejected == []
    assert accepted[0].joint_probability == .25
    assert accepted[0].joint_probability <= min(leg.usable_probability for leg in legs)


def test_seeded_team_run_worlds_are_deterministic_and_consistent():
    a = simulate_team_runs(4.5, 4.1, trials=200, seed=7)
    b = simulate_team_runs(4.5, 4.1, trials=200, seed=7)
    assert np.array_equal(a.home_runs_by_inning, b.home_runs_by_inning)
    assert np.all(a.home_runs_by_inning[:, :5].sum(axis=1) <= a.home_runs)
