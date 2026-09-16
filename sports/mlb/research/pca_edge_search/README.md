# PCA + A* Sportsbook Residual Search (Research Only)

This experiment extends the MLB PCA representation into a deterministic search layer for **strategy-conditioned sportsbook residuals**.

It does **not** generate sporting outcomes and it does **not** redefine A* success as a bet win.

- `EDGE_FOUND`: A* reached a historically supported PCA region whose conservative empirical residual exceeds the configured threshold.
- `NO_EDGE_PATH`: No such region is reachable in the fitted graph.
- `won/lost/push`: Only assigned later by the existing settlement pipeline.

## Representation

Historical rows are standardized and projected with PCA:

```text
raw strategy features -> standardization -> PCA z
```

The module uses NumPy SVD rather than a learned neural encoder. Search edges connect nearby PCA states **only within the same market type and side**, so search cannot jump from one sportsbook contract family to another just because two states are numerically close.

## Residual estimate

For a PCA state `z`, its historical neighborhood estimates:

```text
local empirical hit probability
minus
local no-vig sportsbook probability
```

The empirical estimate is shrunk toward the local market probability. A conservative residual subtracts a sampling-error penalty:

```text
conservative_residual = empirical_probability
                      - local_market_probability
                      - z_confidence * standard_error
```

A state is eligible as an edge region only when effective neighborhood support and OOD distance pass configured floors.

## A* heuristic

The research graph uses PCA distance as transition cost. After fitting local conservative residuals, the module computes the maximum observed residual slope over every graph edge:

```text
L = max |R_i - R_j| / distance(i, j)
```

For an edge threshold `R*`, the heuristic is:

```text
h(i) = max(0, R* - R_i) / L
```

On the finite fitted graph this is a lower bound on PCA path distance needed to close the residual gap, assuming non-negative distance costs. This is safer than using raw hit confidence directly as the A* heuristic.

## Current candidate scoring

Current sportsbook candidates are transformed using the frozen historical PCA basis and scored only from historical neighbors. Their unknown future settlement is never used.

The V2 candidate contract already carries `no_vig_market_probability`; this experiment consumes that field rather than reconstructing sportsbook probabilities inside the search layer.

## Required research discipline

1. Fit PCA, neighborhood residuals, and graph topology on development data only.
2. Freeze the representation and all thresholds before holdout evaluation.
3. Search current/holdout candidates using only information available at prediction time.
4. Settle later through the existing settlement pipeline.
5. Report whether discovered residual regions remain positive on chronologically held-out future slates.

A discovered path is evidence of a **market-representation discrepancy**, not evidence that an individual bet is guaranteed to win.

## Suggested feature families

The first real-data pass should favor stable, pregame, point-in-time features already present or derivable in the MLB stack, such as:

- batting-order / expected-PA opportunity;
- batter contact, power, and on-base profile;
- opposing starter contact/K/power vulnerability;
- handedness/platoon interaction;
- park/weather context if timestamp-correct;
- bullpen context if timestamp-correct;
- line-relative descriptors available at prediction time;
- support and uncertainty diagnostics.

Do not include post-settlement information, closing-line information unavailable at prediction time, or future lineup/status changes.

## Validation target

The primary research question is not whether A* can find historical winners. It is:

> Do PCA regions discovered from development data retain a positive conservative residual versus the sportsbook on chronologically held-out slates?

Key metrics should include discovery count, effective support, raw and conservative residual, holdout hit rate, holdout mean no-vig market probability, realized residual, Brier/log-loss deltas, ROI, and concentration by player/market/slate.
