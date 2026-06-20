# MLB Walk-Forward Policy Optimization

Generated: 2026-06-19T20:50:53.051622+00:00

## Split

- Training: 2026-03-15 through 2026-05-21
- Validation: 2026-05-22 through 2026-06-04
- Untouched holdout: 2026-06-05 through 2026-06-18

## Selected Policy

- Config: `{"max_per_game": 2, "max_per_market_bucket": 2, "max_per_player": 1, "max_per_team": 3, "max_push_probability": 0.12, "min_abs_edge": 0.4, "min_graded_hit_rate": 0.72, "min_historical_bet_profile_support": 12, "min_historical_bet_profile_win_rate": 0.55, "min_historical_market_availability_rate": 0.45, "min_historical_market_availability_support": 20, "min_hit_probability": 0.6, "name": "candidate_290", "top_n": 10}`
- Training: 174-2-1 (98.9%), +156.18u proxy
- Validation: 115-2-0 (98.3%), +102.55u proxy
- Holdout: 124-4-0 (96.9%), +108.73u proxy
- Recent seven days: 63-3-0 (95.5%), +54.27u proxy

## Holdout Comparison

| Policy | W-L-P | Hit rate | 95% low | Proxy units | Drawdown |
|---|---:|---:|---:|---:|---:|
| production_current | 132-8-0 | 94.3% | 89.1% | +112.00 | -2.00 |
| guardrailed_six | 79-5-0 | 94.0% | 86.8% | +66.82 | -2.00 |
| optimized_candidate | 124-4-0 | 96.9% | 92.2% | +108.73 | -2.00 |

**Promotion verdict: shadow_only**

- fewer than 30 valid price-confirmed holdout plays

## Guardrails

- Configuration selection never reads holdout outcomes.
- Invalid American prices are excluded from price-confirmed ROI.
- Proxy profit assumes flat -110 stakes and is not executable ROI.
- Production promotion requires prospective price-confirmed holdout volume.
