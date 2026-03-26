# NBA-VAR v1.1 Final Cards - Validation Report

## Summary

- **Total Cards Validated**: 569
- **Cards Without Issues**: 554 (97.4%)
- **Cards With Issues**: 15 (2.6%)
- **Average Validation Score**: 99.5%
- **Minimum Score**: 80.0%
- **Maximum Score**: 100.0%

## Issues by Category

| Category | Issue Count | % of Cards |
|----------|-------------|------------|
| Schema Integrity | 0 | 0.0% |
| Data Quality | 15 | 2.6% |
| Archetype Logic | 0 | 0.0% |
| Business Logic | 0 | 0.0% |
| Visualization Ready | 0 | 0.0% |

## Cards with Issues (Top 20)

| Player | Position | Archetype | Score | Issues |
|--------|----------|-----------|-------|--------|
| Brook Lopez | Big | defensive_anchor | 80.0% | 1 |
| Bub Carrington | Guard | versatile_wing | 80.0% | 1 |
| Christian Braun | Big | stretch_big | 80.0% | 1 |
| DeMar DeRozan | Guard | versatile_wing | 80.0% | 1 |
| Deni Avdija | Guard | versatile_wing | 80.0% | 1 |
| Dyson Daniels | Guard | point_of_attack | 80.0% | 1 |
| Jaden McDaniels | Big | rim_runner | 80.0% | 1 |
| Jalen Williams | Guard | point_of_attack | 80.0% | 1 |
| Jrue Holiday | Guard | versatile_wing | 80.0% | 1 |
| Julius Randle | Guard | versatile_wing | 80.0% | 1 |
| Kelly Oubre Jr. | Big | rim_runner | 80.0% | 1 |
| Pascal Siakam | Guard | versatile_wing | 80.0% | 1 |
| Scottie Barnes | Guard | versatile_wing | 80.0% | 1 |
| Tobias Harris | Big | rim_runner | 80.0% | 1 |
| Toumani Camara | Big | stretch_big | 80.0% | 1 |

## Validation Criteria

### ✅ Schema Integrity
- All required fields present
- Shot frequencies sum to ~1.0 (±0.01)
- Possession decomposition formulas consistent
- Trust score matches subscores (±5 points)

### ✅ Data Quality
- Estimated data sources clearly marked
- Trust score reflects data quality
- Uncertainty estimates reasonable for sample size
- Defense visibility honest about limitations

### ✅ Archetype Logic
- Archetype appropriate for position
- Bigs have big-specific archetypes
- Scenario constraints match archetype
- Comparables have compatible archetypes

### ✅ Business Logic
- Scenario constraints player-specific
- Usage/minutes caps realistic
- Confidence intervals properly ordered (low ≤ mean ≤ high)
- Percentile ranks consistent with metrics

### ✅ Visualization Ready
- All numeric fields have reasonable ranges
- Text fields free of template placeholders
- Metadata includes version and fixes
- Quality flags for known issues

## Recommendations

⚠ **15 cards have issues** that should be reviewed:
1. Review cards with schema integrity issues
2. Check data quality flags for high-estimated-data cards
3. Verify archetype-position alignment for remaining mismatches
4. Ensure scenario constraints are player-specific

## Next Steps

1. **Visualization**: Load cards from `all_final_enhanced_cards/`
2. **Quality Filtering**: Use quality flags to filter low-confidence cards
3. **User Guidance**: Leverage semantic explanations for interpretation
4. **Monitoring**: Track validation scores for future card generations
