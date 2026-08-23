"""PARLAY_CERTIFICATION_V2 -- the sole authoritative outer decision/
certification layer for the NFL parlay research system.

Separates five objects (see manifest.py / MIGRATION.md for the full
theory): OPERATIONAL ELIGIBILITY (eligibility.py), PREDICTIVE/WORLD MODEL
(reused unchanged from joint_position_builder_v2 -- e.g. pairs.py's
CandidatePair, outcome_worlds.py's world distribution), DECISION POLICY
(policy.py), SETTLEMENT (settlement.py), PROSPECTIVE EVIDENCE
(evidence_store.py + anytime_monitor.py + state_machine.py).

Supersedes joint_position_builder_v2/legacy/risk_gate_v1_ARCHIVED.py.
production_authorized stays False here always -- see manifest.py.
"""
