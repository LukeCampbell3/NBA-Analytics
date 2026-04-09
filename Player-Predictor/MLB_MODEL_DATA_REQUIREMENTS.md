# MLB Isolated Model Requirements (Archetype Reuse Only)

## Architecture Decision

Use the NBA model as an archetype only:

- Keep the modeling pattern: sequence encoder + residual/delta refinement stack.
- Do not reuse NBA targets, names, or feature semantics.
- Keep MLB models fully isolated from NBA/NFL data contracts.

## Isolated MLB Model Families

Train separate models by player role.

### Hitter model

- Targets: `H`, `HR`, `RBI`
- Typical context: batting order, plate appearances, contact/power quality, opposing pitcher profile, park/weather, market lines.

### Pitcher model

- Targets: `K`, `ER`, `ERA`
- Typical context: workload, pitch quality/mix, control metrics, opponent lineup quality, park/weather, market lines.

## Data Grain And History

- One row per `player x game`.
- Chronological order by `Date` and/or `Game_Index`.
- Minimum rows per player file: `11` (`seq_len=10` + next-game target row).
- Recommended rows per player file: `80+`.

## Required Columns (Common)

- `Date`, `Player`, `Player_Type`, `Team`, `Opponent`
- `Season`, `Game_ID`, `Game_Index`
- `Team_ID`, `Opponent_ID`, `Is_Home`
- `Did_Not_Play`, `Rest_Days`
- `Month_sin`, `Month_cos`, `DayOfWeek_sin`, `DayOfWeek_cos`
- `Market_Fetched_At_UTC`

## Required Columns (Hitter)

- Targets and sequence anchors:
  - `H`, `HR`, `RBI`
  - `H_rolling_avg`, `HR_rolling_avg`, `RBI_rolling_avg`
  - `H_lag1`, `HR_lag1`, `RBI_lag1`
- Opportunity/skill:
  - `PA`, `AB`, `BB`, `SO`, `Batting_Order`, `Team_PA_share`
  - `wOBA`, `xwOBA`, `ISO`, `Barrel%`, `HardHit%`
- Opponent/context:
  - `Opp_Pitcher_ERA_3`, `Opp_Pitcher_K9_3`, `Opp_Bullpen_ERA_7`
  - `Park_Factor`, `Wind_Out_MPH`, `Temp_F`
- Market:
  - `Market_H`, `Market_HR`, `Market_RBI`
  - `Synthetic_Market_H`, `Synthetic_Market_HR`, `Synthetic_Market_RBI`
  - `Market_Source_H`, `Market_Source_HR`, `Market_Source_RBI`
  - `Market_H_books`, `Market_HR_books`, `Market_RBI_books`
  - `Market_H_over_price`, `Market_HR_over_price`, `Market_RBI_over_price`
  - `Market_H_under_price`, `Market_HR_under_price`, `Market_RBI_under_price`
  - `Market_H_line_std`, `Market_HR_line_std`, `Market_RBI_line_std`
  - `H_market_gap`, `HR_market_gap`, `RBI_market_gap`

## Required Columns (Pitcher)

- Targets and sequence anchors:
  - `K`, `ER`, `ERA`
  - `K_rolling_avg`, `ER_rolling_avg`, `ERA_rolling_avg`
  - `K_lag1`, `ER_lag1`, `ERA_lag1`
- Workload/skill:
  - `IP`, `BF`, `Pitches`, `BB_allowed`, `H_allowed`, `HR_allowed`
  - `FIP`, `xFIP`, `CSW%`, `Whiff%`
- Opponent/context:
  - `Opp_Lineup_wOBA_3`, `Opp_Lineup_K_rate_3`
  - `Park_Factor`, `Wind_Out_MPH`, `Temp_F`
- Market:
  - `Market_K`, `Market_ER`, `Market_ERA`
  - `Synthetic_Market_K`, `Synthetic_Market_ER`, `Synthetic_Market_ERA`
  - `Market_Source_K`, `Market_Source_ER`, `Market_Source_ERA`
  - `Market_K_books`, `Market_ER_books`, `Market_ERA_books`
  - `Market_K_over_price`, `Market_ER_over_price`, `Market_ERA_over_price`
  - `Market_K_under_price`, `Market_ER_under_price`, `Market_ERA_under_price`
  - `Market_K_line_std`, `Market_ER_line_std`, `Market_ERA_line_std`
  - `K_market_gap`, `ER_market_gap`, `ERA_market_gap`

## Quality Rules

- No duplicate `Player + Date` rows.
- `Game_Index` strictly increasing by player.
- `Did_Not_Play` must be `0/1`.
- Required numeric columns must be finite.
- Market line std values must be non-negative.

## Included Assets

- Native schema contract:
  - `Data-Proc-MLB/schema/mlb_native_player_schema_v1.json`
- Hitter sample data:
  - `Data-Proc-MLB/Mookie_Betts/2026_processed_processed.csv`
- Sample manifest:
  - `Data-Proc-MLB/update_manifest_2026.json`
- Contract validator:
  - `scripts/validate_mlb_processed_contract.py`

Run validation:

```bash
python scripts/validate_mlb_processed_contract.py --data-dir Data-Proc-MLB
```
