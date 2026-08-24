"""Forward-only predictive calibration ledger for single-leg PGA golf
bets (outright winner, top-5/10/20 finish, make/miss cut). Ported
verbatim from sports/mlb/parlay_v2/calibration (schema.py, store.py,
snapshot.py, support.py, versioning.py are byte-identical to MLB's --
this ledger's mechanics are fully sport-agnostic).

Golf does not use MLB/NFL's two-leg pairing/world-certificate machinery
(parlay_v2/candidate_adapter.py, freeze_prospective.py, program_alpha.py,
etc.) -- head-to-head 2-ball matchups are explicitly out of scope for this
build, so there is nothing here to pair. This ledger backs a single-leg
selection policy instead, structured like MLB's own board
(select_high_precision_predictions.py + optimize_walk_forward_policy.py),
not the Parlays-tab pairing system.

Modules:
    schema.py     CalibrationObservation -- one settled predictive event.
    store.py      Forward-only, append-only, idempotent ledger.
    snapshot.py   Immutable, reproducible, as-of snapshots of the ledger.
    support.py    Multidimensional CandidateSupport, built from a snapshot.
    versioning.py CALIBRATION_VERSION + compatibility checks.
"""
