# Safe-State Evidence Gap Report

## Executive Summary
- Total candidates: 47
- Production rows: 1
- Price-defense candidates: 18
- Price-defense-only board rows: 1
- SAFE_STATE_CORE rows: 0
- SAFE_STATE_NEAR_CORE rows: 0
- Production behavior changed: false
- Promotion claim: false

## Primary Blockers
- PRICE_GAP: 29
- FORECASTABILITY_GAP_MINUTES_STATE: 10
- FORECASTABILITY_GAP_ROLE_STATE: 6
- FORECASTABILITY_GAP_USAGE_STATE: 2

## Near-Core Candidates
- None

## Feature Gap Ranking
- usage_proxy: blocks 2 candidates (2 EDGE_DEFENDABLE); priority HIGH
- teammate_availability: blocks 2 candidates (2 EDGE_DEFENDABLE); priority HIGH
- opponent_context: blocks 2 candidates (2 EDGE_DEFENDABLE); priority HIGH
- distribution_quantiles: blocks 2 candidates (2 EDGE_DEFENDABLE); priority HIGH
- similar_state_sample: blocks 2 candidates (2 EDGE_DEFENDABLE); priority HIGH
- structural_pathway: blocks 2 candidates (2 EDGE_DEFENDABLE); priority HIGH
- usage_proxy: blocks 29 candidates (0 EDGE_DEFENDABLE); priority HIGH
- teammate_availability: blocks 29 candidates (0 EDGE_DEFENDABLE); priority HIGH
- opponent_context: blocks 29 candidates (0 EDGE_DEFENDABLE); priority HIGH
- distribution_quantiles: blocks 29 candidates (0 EDGE_DEFENDABLE); priority HIGH

## Actionable Evidence Gaps
- usage_proxy: USAGE_SAMPLE_INSUFFICIENT (NEEDS_MORE_SAMPLE) -> BUILD_SAMPLE_AND_RECHECK
- teammate_availability: USAGE_SAMPLE_INSUFFICIENT (NEEDS_MORE_SAMPLE) -> BUILD_SAMPLE_AND_RECHECK
- opponent_context: USAGE_SAMPLE_INSUFFICIENT (NEEDS_MORE_SAMPLE) -> BUILD_SAMPLE_AND_RECHECK
- distribution_quantiles: USAGE_SAMPLE_INSUFFICIENT (NEEDS_MORE_SAMPLE) -> BUILD_SAMPLE_AND_RECHECK
- similar_state_sample: USAGE_SAMPLE_INSUFFICIENT (NEEDS_MORE_SAMPLE) -> BUILD_SAMPLE_AND_RECHECK
- structural_pathway: USAGE_SAMPLE_INSUFFICIENT (NEEDS_MORE_SAMPLE) -> BUILD_SAMPLE_AND_RECHECK
- usage_proxy: USAGE_PROXY_MISSING (price_pipeline_blocked) -> WATCH
- teammate_availability: TEAMMATE_AVAILABILITY_MISSING (price_pipeline_blocked) -> WATCH
- opponent_context: OPPONENT_CONTEXT_MISSING (price_pipeline_blocked) -> WATCH
- distribution_quantiles: DISTRIBUTION_QUANTILES_MISSING (price_pipeline_blocked) -> WATCH

## Non-Actionable True Instability
- minutes_state: MINUTES_LOW_FLOOR -> KEEP_UNSAFE_TRUE_VOLATILITY
- usage_proxy: MINUTES_LOW_FLOOR -> KEEP_UNSAFE_TRUE_VOLATILITY
- usage_proxy: USAGE_PROXY_MISSING -> KEEP_UNSAFE_TRUE_VOLATILITY
- teammate_availability: MINUTES_LOW_FLOOR -> KEEP_UNSAFE_TRUE_VOLATILITY
- teammate_availability: TEAMMATE_AVAILABILITY_MISSING -> KEEP_UNSAFE_TRUE_VOLATILITY
- opponent_context: MINUTES_LOW_FLOOR -> KEEP_UNSAFE_TRUE_VOLATILITY
- opponent_context: OPPONENT_CONTEXT_MISSING -> KEEP_UNSAFE_TRUE_VOLATILITY
- distribution_quantiles: DISTRIBUTION_QUANTILES_MISSING -> KEEP_UNSAFE_TRUE_VOLATILITY
- distribution_quantiles: MINUTES_LOW_FLOOR -> KEEP_UNSAFE_TRUE_VOLATILITY
- similar_state_sample: MINUTES_LOW_FLOOR -> KEEP_UNSAFE_TRUE_VOLATILITY

## Guardrails
- Diagnostic only.
- No production gate or threshold was changed.
- No sidecar was materialized.
- No promotion claim is made.
