# PAR-F v0.7 Empirical Atom Persistence Validation

Milestones frozen:

- `PAR_SEVEN_SEASON_BOX_DOMINANCE_CONFIRMED`
- `PARF_V0_6_INCREMENTAL_FORECAST_VALUE_REJECTED`
- `PARF_V0_7_EMPIRICAL_ATOM_PERSISTENCE_VALIDATION`

Result: `PARF_V0_7` is not accepted. It improves MAE slightly but fails the
primary Spearman gate against the permanent `CURRENT_PAR_BASELINE` champion.

| model | Pearson | Spearman | MAE | RMSE | tier accuracy |
|---|---:|---:|---:|---:|---:|
| Current PAR | 0.797262 | 0.743996 | 184.762571 | 261.418213 | 0.619941 |
| PAR-F v0.6 | 0.787348 | 0.738017 | 207.483904 | 297.017207 | 0.614076 |
| PAR-F v0.7 | 0.795924 | 0.739441 | 182.713078 | 256.816366 | 0.622287 |

Primary deltas vs champion:

- PAR-F v0.6 Spearman: `-0.005979`; MAE: `-22.721333`
- PAR-F v0.7 Spearman: `-0.004555`; MAE: `+2.049493`

Bootstrap confidence intervals:

- PAR-F v0.6 MAE improvement CI: `[-30.821671, -22.654991, -14.271625]`
- PAR-F v0.6 Spearman delta CI: `[-0.011385, -0.006034, -0.000355]`
- PAR-F v0.7 MAE improvement CI: `[-3.572881, 2.043004, 7.463235]`
- PAR-F v0.7 Spearman delta CI: `[-0.011799, -0.004573, 0.002489]`

Validation-contract finding:

The historical `MAE < 35 PAR` gate is scale-incompatible with season-total PAR
on the seven-season 500-minute population. The versioned corrected gate is
`PARF_TOTAL_SEASON_500MP_GATE_V1`: a challenger must beat current PAR on
Spearman and MAE out of sample.

Opportunity-denominator limitation:

The frozen seven-season Basketball-Reference validation rows do not contain
atom-specific opportunity denominators. v0.7 therefore models atom rate per
1,000 minutes and projected minutes separately, while marking atom-specific
opportunity persistence as unavailable instead of fabricating denominators.
