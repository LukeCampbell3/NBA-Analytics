# Decision Engine

This package replays historical market decisions through the full policy stack:

`prediction -> gating -> selection -> sizing -> outcome -> validation`

## Main entrypoint

Run the simulator with:

```powershell
python scripts\validate_decision_policy.py
```

Outputs land in `model/analysis/decision_policy/` by default:

- `strategy_results.csv`: shadow-strategy leaderboard
- `all_strategy_decisions.csv`: one row per historical opportunity per strategy
- `all_strategy_daily.csv`: daily bankroll path per strategy
- `best_strategy_decisions.csv`: selected strategy decision log
- `decision_policy_summary.json`: validation summary, calibration checks, and alerts

## Modules

- `gating.py`: walk-forward calibration and candidate scoring
- `selection.py`: policy gates, caps, and final-board selection
- `sizing.py`: flat or capped fractional-Kelly sizing
- `simulation.py`: historical replay loop
- `validation.py`: policy diagnostics and drift alerts
- `policy_tuning.py`: shadow strategies and counterfactual sweeps
