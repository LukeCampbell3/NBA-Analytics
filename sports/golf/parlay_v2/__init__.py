"""PGA golf single-leg selection system -- the source for the PGA board.

Unlike MLB/NFL's parlay_v2 (which pairs two legs into a parlay for a
Parlays tab), golf's real bet-type scope for this build is single-leg
only: outright tournament winner, top-5/10/20 finish, and make/miss cut.
So this package holds a single-leg selection + calibration ledger,
structured like MLB's own board (select_high_precision_predictions.py +
optimize_walk_forward_policy.py), reusing calibration/'s ledger mechanics
(schema.py/store.py/support.py/snapshot.py/versioning.py) verbatim from
MLB since they are fully sport-agnostic.

Modules:
    select_pga_bets.py: builds real candidates from the score-projection
        model's outcome probabilities + real market prices, applies the
        frozen selection policy's gates, and reports which candidates are
        certified vs. shadow-only.
    settle_pga_calibration.py: settles completed real tournament outcomes
        into the calibration ledger (the daily CI step that keeps this
        ledger growing, mirroring settle_parlay_v2_calibration.py's role
        for NFL).

Authority boundary: no candidate here is ever authorized for real
staking until the calibration ledger has accumulated enough real settled
observations to pass this policy's gates -- exactly the same
shadow-until-earned discipline as every other frozen policy in this repo.
"""
