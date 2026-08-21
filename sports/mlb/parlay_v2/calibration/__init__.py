"""Forward-only predictive calibration ledger -- STREAM A (mission:
"CRITICAL TWO-STREAM SEPARATION"). Provides ONLY prior-settled information
needed to judge whether today's candidates have sufficient support. It may
influence candidate support/ranking; it may NEVER certify the policy, move
PolicyStatus, or authorize production, and it may NEVER see today's own
outcome before today's decision is frozen.

STREAM B (the policy evidence stream: one record per eligible slate,
driving G_C/G_L/G_V and PolicyStatus) remains
sports/mlb/research/parlay_certification_v2/evidence_store.py, reused
unchanged -- see that package for STREAM B.

Modules:
    schema.py     CalibrationObservation -- one settled predictive event.
    store.py      Forward-only, append-only, idempotent ledger.
    snapshot.py   Immutable, reproducible, as-of snapshots of the ledger.
    support.py    Multidimensional CandidateSupport, built from a snapshot.
    replay.py     Deterministic replay for the calibration side (see
                  parlay_certification_v2/evidence_store.py + this
                  package's replay.py for the policy side).
    versioning.py CALIBRATION_VERSION + compatibility checks.
"""
