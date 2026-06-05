# v10.8 CLV Hybrid Surrogate Synthetic Dataset

Rows: 1000
Purpose: Development fixture for hybrid CLV surrogate implementation.
Warning: This is a synthetic fixture. It is not production evidence, not staking evidence, and not real live CLV.

## Label tiers
{
  "bronze_synthetic_clv": 595,
  "silver_proxy_clv": 261,
  "gold_real_clv": 144
}

## Synthetic scenarios
{
  "stale_book_correction": 213,
  "book_vs_consensus_outlier": 253,
  "no_lapse": 188,
  "market_velocity_continuation": 164,
  "alt_line_mismatch_trap": 53,
  "adverse_correction": 129
}

## Aggregate synthetic target stats
- positive_clv_label_rate: 0.622
- mean_target_side_aware_clv: 0.00661

## Core fields
- clv_label_tier: gold_real_clv, silver_proxy_clv, bronze_synthetic_clv
- label_weight: gold=1.0, silver=0.25-0.50, bronze decayed around 0.063
- true_side_aware_clv: present only for gold-template rows
- proxy_side_aware_clv: present only for silver rows
- synthetic_side_aware_clv: present only for bronze rows
- target_side_aware_clv: the tier-specific training target
- production_countable_for_staking: always false
- live_evidence: always false
- dataset_row_origin: synthetic_fixture_not_production_evidence

## Intended usage
Train or test the v10.8 CLV surrogate pipeline using weighted labels.
Validate/promote only on real gold live CLV rows collected by the production-shadow pipeline.
