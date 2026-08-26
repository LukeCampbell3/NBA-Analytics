"""Leakage-safe two-leg parlay eligibility gate (additive, shadow-only, MLB).

Ported from sports/nba/predictions/Player-Predictor/research/parlay_policy_v2/
(same generic, sport-agnostic `policy.py`) after that module's gate was
backtested for real against this sport's own settled data -- see
`real_data_backtest.py` and REPORT.md.

This package is additive only: nothing here imports from, modifies, or is
imported by `sports/parlay_analysis.py` (CONTROL) or
`sports/mlb/research/parlay_certification_v2` / `joint_position_builder_v2`
(MLB's own, much more heavily-instrumented V2 program, which already
concluded INSUFFICIENT_EVIDENCE / production_authorized=False). See
REPORT.md for exactly how this relates to that program and why this module
is not itself production-authorized either.
"""
