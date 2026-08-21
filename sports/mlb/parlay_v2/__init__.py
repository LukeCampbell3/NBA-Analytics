"""PARLAY_POLICY_V2 product path -- the sole source for the Parlays tab.

candidate_adapter.py: builds descriptive/proposal candidate pairs (never
authoritative -- see its module docstring) from today's pregame predictive
universe.
program_alpha.py: research-level prospective alpha budget across policy
versions (mission section 13).
legacy_control.py: read-only diagnostic access to the OLD parlay
subsystem's output, for comparison only -- never an input to V2.
comparison.py: builds the immutable daily old-vs-new comparison artifact
(research/debug only, not authorization logic).
run_parlay_v2.py: CLI entry point wiring adapter -> V2 policy -> JSON,
mirroring select_daily_parlay.py's CLI shape for pipeline symmetry.

Authority boundary (do not blur): this package PROPOSES. Only
sports/mlb/research/parlay_certification_v2/ CERTIFIES. Nothing here may
set `action`, `certified`, `supported`, or `production_authorized`.
"""
