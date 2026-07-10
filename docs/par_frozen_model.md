# Frozen PAR/PAR-F Analytical Identities

This document freezes the repository-local source of truth for:

- PAR-PVG v0.5 Realistic Tuning Baseline
- PAR-F v0.6 Role-Continuity Baseline

Canonical build identifiers:

- `par_model_version: par_pvg_v0_5`
- `parf_model_version: parf_v0_6`
- `points_per_win: 30.4`

PAR means Points Above Replacement and is measured in points. Wins are a
downstream presentation value.

## Current PAR Identity

```text
PAR =
    BoxVisiblePAR
    + ConfirmedHiddenRolePAR
    + ShrunkProxyPAR
    - OverlapLeakage
```

Preserved presentation transforms:

```text
WAR_equivalent = PAR / 30.4
PAR_1000 = (PAR / Minutes) * 1000
PVGScore = 50 + 45 * tanh(PAR_1000 / 210)
```

## Isolation Rule

Context labels may overlap. Value labels may not double count.

Every point of PAR must have one accounting destination:

```text
player_total_par == sum(player_value_atom_par) == sum(player_category_par)
```

Failure of this identity is a model error. The build may not silently normalize
totals after calculation.

## Value Atom Categories

The player-facing rollups are:

- SCORING
- CREATION
- BALL_SECURITY
- PLAYTYPE_PNR
- SPACING
- REBOUNDING
- PERIMETER_DISRUPTION
- RIM_DEFENSE
- CONTEST_DEFENSE
- HUSTLE
- RESIDUAL

Only registered value atoms can contribute. Unsupported, unready CV, and
unready proxy sources must contribute zero production PAR.

## Current Direct Source Adapter

The repository currently has game-log level direct box-visible source data. That
source can support only these production atoms:

- `scoring_volume_above_replacement`
- `passing_creation`
- `negative_turnover_value`
- `steals`

It does not create spacing, tracking contest, rim deterrence, hustle, residual,
or hidden-role value. Those categories must be displayed as limited evidence
when no ready atom source exists.

## PAR-F Identity

```text
ProjectedPAR_t+1 =
    [StablePAR_t + lambda * VolatilePAR_t + TrendPAR_t]
    * RoleContinuity
    * MinutesFactor
    * HealthFactor
    * AgeCurve
    + FitLift
```

Atom-level interpretation:

```text
ProjectedPAR_t+1 =
    sum(PAR_atom_t * Persistence_atom * Reliability_atom)
    * MinutesFactor
    * RoleFactor
    * HealthFactor
    + FitLift
```

The persistence values are frozen in `sports/nba/analytics/par/config.py`.
