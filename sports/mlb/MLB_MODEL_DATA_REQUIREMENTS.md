# MLB Isolated Model Requirements

MLB uses the same high-level modeling archetype as NBA (sequence + refinement) but has a fully MLB-native contract and targets.

## Isolation Rules

- No shared features, labels, aliases, or transforms with NBA/NFL data.
- MLB trainer reads only `sports/mlb/data/processed`.
- MLB targets are only:
  - Hitters: `H`, `HR`, `RBI`
  - Pitchers: `K`, `ER`, `ERA`

## Data Grain

- One row per `player x game`.
- Sort order per player must be chronological by `Date`, then `Game_ID`.
- `Game_Index` must be strictly increasing per player.
- Minimum history:
  - Hard minimum: `11` rows per player (`seq_len=10` + next game label).
  - Recommended: `80+` rows per player.

## Required Raw Inputs (Collection Layer)

Collector output files:

- `sports/mlb/data/raw/games.csv`
- `sports/mlb/data/raw/hitter_game_logs.csv`
- `sports/mlb/data/raw/pitcher_game_logs.csv`
- `sports/mlb/data/raw/collection_manifest.json`

Required raw fields for hitters:

- IDs/context: `Date`, `Season`, `Game_ID`, `Player`, `Player_ID`, `Team`, `Team_ID`, `Opponent`, `Opponent_ID`, `Is_Home`
- Targets/box score: `H`, `HR`, `RBI`
- Supporting stats: `PA`, `AB`, `BB`, `SO`, `SB`, `R`, `TB`, `2B`, `3B`, `HBP`, `IBB`, `SF`, `SH`, `Batting_Order`
- Environment: `Temp_F`, `Wind_Out_MPH`

Required raw fields for pitchers:

- IDs/context: `Date`, `Season`, `Game_ID`, `Player`, `Player_ID`, `Team`, `Team_ID`, `Opponent`, `Opponent_ID`, `Is_Home`
- Targets/box score: `K`, `ER`, `ERA`
- Supporting stats: `IP`, `BF`, `Pitches`, `BB_allowed`, `H_allowed`, `HR_allowed`, `HBP_allowed`, `Is_Starter`
- Environment: `Temp_F`, `Wind_Out_MPH`

## Required Processed Fields (Training Contract)

Common:

- `Date`, `Player`, `Player_Type`, `Team`, `Opponent`
- `Season`, `Game_ID`, `Game_Index`
- `Team_ID`, `Opponent_ID`, `Is_Home`
- `Did_Not_Play`, `Rest_Days`
- `Month_sin`, `Month_cos`, `DayOfWeek_sin`, `DayOfWeek_cos`
- `Market_Fetched_At_UTC`

Hitter pregame features:

- Form/skill priors: `xwOBA`, `Barrel%`, `HardHit%`
- Opponent context: `Opp_Pitcher_ERA_3`, `Opp_Pitcher_K9_3`, `Opp_Bullpen_ERA_7`
- Environment: `Park_Factor`, `Wind_Out_MPH`, `Temp_F`
- Target history: `H_rolling_avg`, `HR_rolling_avg`, `RBI_rolling_avg`, `H_lag1`, `HR_lag1`, `RBI_lag1`
- Market: `Market_H`, `Market_HR`, `Market_RBI`, `Synthetic_Market_H`, `Synthetic_Market_HR`, `Synthetic_Market_RBI`, all corresponding books/price/std columns, plus `H_market_gap`, `HR_market_gap`, `RBI_market_gap`

Pitcher pregame features:

- Form priors: `FIP`, `xFIP`
- Opponent context: `Opp_Lineup_wOBA_3`, `Opp_Lineup_K_rate_3`
- Environment: `Park_Factor`, `Wind_Out_MPH`, `Temp_F`
- Role/availability: `Is_Starter`, `Rest_Days`, `Game_Index`
- Target history: `K_rolling_avg`, `ER_rolling_avg`, `ERA_rolling_avg`, `K_lag1`, `ER_lag1`, `ERA_lag1`
- Market: `Market_K`, `Market_ER`, `Market_ERA`, `Synthetic_Market_K`, `Synthetic_Market_ER`, `Synthetic_Market_ERA`, all corresponding books/price/std columns, plus `K_market_gap`, `ER_market_gap`, `ERA_market_gap`

Labels (must exist in processed rows, but not used as same-game inputs):

- Hitters: `H`, `HR`, `RBI`
- Pitchers: `K`, `ER`, `ERA`

## Accuracy And Leakage Controls

- Synthetic market fallback must use pregame-safe priors only (rolling/lag), never same-game targets.
- Trainer must exclude same-game outcome stats from feature set (for example `PA`, `AB`, `IP`, `Pitches`, etc.).
- Validation split must be time-ordered (no random split).
- Model selection must include rolling-baseline comparison and keep baseline if ML does not improve MAE.

## Training Reliability Gates

- Minimum rows per role before training: `>= 200`.
- Recommended unique game dates in processed set: `>= 20` for stable validation.
- Stop training and investigate if:
  - target variance is near-zero for a role/target,
  - selected model MAE is worse than rolling baseline MAE,
  - contract validator reports any file failures.

## Validation And Execution

Run end-to-end:

```bash
python sports/mlb/scripts/run_mlb_pipeline.py --start-date 2026-03-20 --end-date 2026-04-08 --min-train-rows 200
```

For short smoke windows, lower validator gate:

```bash
python sports/mlb/scripts/run_mlb_pipeline.py --start-date 2026-04-01 --end-date 2026-04-03 --min-processed-rows 1 --min-train-rows 200
```

Daily inference and real-world scoring:

```bash
python sports/mlb/scripts/build_mlb_daily_prediction_pool.py --run-date 2026-04-10 --season 2026
python sports/mlb/scripts/score_mlb_prediction_pool.py --pool-csv sports/mlb/data/predictions/daily_runs/20260410/daily_prediction_pool_20260410.csv
python sports/mlb/scripts/run_mlb_daily_prediction_pipeline.py --run-date 2026-04-10 --score-all-unscored
```

One-command final shortlist:

```bash
python sports/mlb/scripts/select_mlb_best_predictions.py --run-date 2026-04-10 --season 2026 --top-n 50 --min-abs-edge 0.35 --min-history-rows 10 --max-per-player 1
```

Run checks individually:

```bash
python sports/mlb/scripts/build_mlb_features.py --season 2026
python sports/mlb/scripts/validate_mlb_processed_contract.py --data-dir sports/mlb/data/processed
python sports/mlb/scripts/train_mlb_models.py --season 2026 --min-rows 200
```
