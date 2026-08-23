"""NFL joint-outcome-world machinery for PARLAY_CERTIFICATION_V2 (sport-agnostic port).

This package intentionally ports only ``outcome_worlds.py`` (+ its
``protocol.py`` dependency) from ``sports/mlb/conditional_chain`` --
the sport-agnostic joint-outcome-world / all-plausible-states (APS)
machinery that ``sports/nfl/parlay_v2/candidate_adapter.py`` uses to build
each pair's world certificate. That module carries zero MLB-specific
logic (confirmed by inspection: no baseball terms, no MLB-only fields),
and this is its second cross-sport port -- it was itself originally a
scoped port of sports/nba/conditional_chain's path-evidence layer's
joint-outcome-world component into MLB.

Unlike sports/mlb/conditional_chain, this package does NOT include MLB's
pregame path-evidence shadow research pipeline (build_reservoir_from_history,
build_path_features_from_market_snapshots, path_conditioned_backtest,
path_conditioned_cli, proof_trajectory, path_world_evidence) -- that is a
separate, additional research product built on top of outcome_worlds.py,
not a dependency of PARLAY_V2 itself, and was out of scope for this
replication. If NFL ever wants its own path-evidence shadow layer, those
MLB modules are the reference to port next.
"""
