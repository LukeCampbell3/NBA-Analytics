# MLB Advanced Data Source Contract

The sequential plate-appearance hitter model uses reproducible public pregame data and fails closed or degrades explicitly when those inputs are unavailable.

## Baseball Savant / Statcast

Primary process source: Baseball Savant Statcast, retrieved through pinned `pybaseball==2.2.7` calls.

Daily profiles are bounded to observations strictly before the run date. The active implementation derives batter contact/process profiles, pitcher contact-quality allowed, pitch arsenals, pitch-type compatibility inputs, and direct batter-vs-pitcher process evidence from Statcast pitch/PA records. Raw season-scale pitch data are not committed to Git; processed dated partitions live under `sports/mlb/data/advanced/` in the workflow/runtime cache.

## FanGraphs-compatible advanced pitching data

`pybaseball.pitching_stats` is used as the reproducible FanGraphs-compatible source for season-to-date ERA/FIP/xFIP/SIERA/xERA fields when the upstream source exposes them. Missing FanGraphs fields do not get fabricated. They are carried as missing advanced dimensions and increase uncertainty/support penalties.

## Freshness and leakage

For a run on date `D`, process observations must have an effective/as-of date no later than `D-1`. Historical replay follows the same cutoff. Current lineup/identity facts come from MLB Stats API and are not inferred from future game results.

Each processed partition records source, fetch timestamp, effective/as-of date, MLBAM IDs, schema version, failures, and freshness status. The model must not silently treat a failed source refresh as fresh data.

## Authority

`sequential_pa_contact_model_v1` currently has negative-only publication authority: it may lower or veto an overconfident legacy H/TB probability, but it cannot increase the legacy probability until leakage-safe independent calibration earns that authority.
