# Unified MLB V1 failure diagnosis

## Qualification audit

All eight recovered rows are authentic pregame candidate artifacts, but **zero can prove the complete frozen qualification contract**. The missing per-quote timestamp makes quote freshness unprovable; a separately preserved player-status assertion is also absent. They are retained as `EXACT_CANDIDATE_ONLY` and consumed diagnostics, not deleted.

| Capability | Records | Expected wins | Actual wins | P(W ≤ observed) | Brier | ROI |
|---|---:|---:|---:|---:|---:|---:|
| batter_hits | 6 | 3.9888 | 2 | 0.1018 | 0.3182 | -46.03% |
| batter_total_bases | 2 | 1.2262 | 1 | 0.6241 | 0.2621 | -11.54% |
| combined | 8 | 5.2150 | 3 | 0.1035 | 0.3041 | -37.41% |

## Exact observations

| Player | Market | P usable | Market P | Edge | EV | Result | Audit blockers |
|---|---|---:|---:|---:|---:|---|---|
| cal_raleigh | H 0.5 | 66.76% | 60.00% | 6.76% | 11.27% | won | quote_freshness, player_status |
| austin_riley | H 0.5 | 70.49% | 63.64% | 6.85% | 10.77% | won | quote_freshness, player_status |
| caleb_durbin | H 0.5 | 66.35% | 64.29% | 2.07% | 3.21% | lost | quote_freshness, player_status |
| spencer_horwitz | H 0.5 | 64.12% | 60.78% | 3.34% | 5.49% | lost | quote_freshness, player_status |
| munetaka_murakami | H 0.5 | 65.32% | 63.64% | 1.69% | 2.65% | lost | quote_freshness, player_status |
| brenton_doyle | H 0.5 | 65.83% | 64.29% | 1.55% | 2.40% | lost | quote_freshness, player_status |
| pete_alonso | TB 1.5 | 61.24% | 60.00% | 1.24% | 2.06% | lost | quote_freshness, player_status |
| pete_crow-armstrong | TB 1.5 | 61.38% | 56.52% | 4.86% | 8.60% | won | quote_freshness, player_status |

## Evidence-based diagnosis

1. V1 expected **5.2150 wins** and observed **3**; the combined lower-tail probability is **0.1035**. This is adverse but not independently informative because every row belongs to one slate.
2. The model and FanDuel baseline were nearly tied on the consumed sample. Eight dependent observations cannot establish Hits overconfidence or Total Bases calibration.
3. The uncertainty field was exactly zero for all eight rows by adapter convention, not measurement. It therefore had no discrimination and must not be treated as empirical uncertainty.
4. Edge and conservative EV did not cleanly separate winners from losses. Several losses sat just above the 1 pp edge boundary, identifying a boundary hypothesis—not a validated replacement threshold.
5. No identity mismatch or settlement corruption was found. The structural failures are incomplete qualification evidence, non-measured uncertainty, and absence of demonstrated incremental information beyond market probability.
6. The 242,425-row universe is useful for outcome-model error studies but lacks frozen probability/lineup/calibration state; it cannot legitimately validate a V2 wagering selector.
