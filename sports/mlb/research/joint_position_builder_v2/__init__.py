"""JOINT_POSITION_BUILDER_V2 (experimental, shadow-only).

Core correction under test: individual leg EV_i>0 is NOT a candidate-
admission requirement for a 2-leg pair. A pair can be +EV even when one leg
is individually -EV (EV(pair) = P(A∩B)*D_S - 1). This package tests that
theory without simply removing the filter and ranking on noisy joint EV --
see manifest.py for the architecture and status, and REPORT.md for results.

This is additive: CONTROL is sports/mlb/scripts/select_daily_parlay.py /
sports/parlay_analysis.py, unmodified. Nothing here changes what the
production pipeline selects or publishes. production_authorized is always
False (see manifest.py).
"""
