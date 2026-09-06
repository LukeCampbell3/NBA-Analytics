# MLB Game-Conditioned MoE — Outcome-Joined Historical Validation

Model: `game_conditioned_hitter_moe_v2`

Sampled hitter-games: **6,000**; joined: **6,000**; target examples: **18,000**.

The current game outcome is resolved only through the separate hash-verified outcome ledger. Current-game H/TB/HR/PA/AB and target-derived rolling/gap fields are masked before feature construction. Historical prior outcomes remain available only through strictly earlier games. Doubleheaders are ordered by Date + Game_Index + Game_ID and require exact Game_ID equality before feature construction.

| Target | Fit rows | OOF rows | Folds pass | Prior Brier | Candidate | Brier gain | Prior LL | Candidate | LL gain | Diagnostic NR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| H | 6,000 | 5,687 | 6/6 | 0.24854 | 0.24384 | 0.00470 | 0.69154 | 0.68108 | 0.01046 | False |
| TB | 6,000 | 5,687 | 6/6 | 0.22487 | 0.22020 | 0.00467 | 0.64178 | 0.63186 | 0.00992 | False |
| HR | 6,000 | 5,687 | 4/6 | 0.09652 | 0.09616 | 0.00036 | 0.34119 | 0.33993 | 0.00126 | False |

This remains diagnostic evidence, not production authority: exact live pitch compatibility, BvP process state, handedness splits, weather, and defense still require snapshot-backed train/serve parity.
