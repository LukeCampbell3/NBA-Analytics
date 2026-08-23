# Advantage Routing: Drive-Pass and Post/Interior-Pass Analysis

## What this is

Two related analysis models over the same underlying representation:

1. **Drive-Pass Advantage Routing** -- when a player creates advantage off
   the dribble (a drive), where does the ball go, what kind of action does
   it create, and who benefits?
2. **Post/Interior-Pass Advantage Routing** -- the same questions for a
   player operating from the post or as an interior hub (catching the ball
   in the paint/short roll and creating from there).

The explicit objective is **not** a pretty box-score dashboard. It is a
detailed, auditable engine that answers: when a player creates or receives
an advantage, where does he route the ball, what type of shot/action does
the pass create, which teammates benefit, what type of scoring gravity
caused the defensive reaction, and how might those outcomes change under a
different offensive role. Every one of those questions is only worth
answering honestly -- see the next section.

## The one non-negotiable rule

Never mix OBSERVED, DERIVED, RECONSTRUCTED, and SIMULATED data, and never
fabricate precision because an event-level field is unavailable. Every
metric in this system is wrapped in a `Metric` (or `ShrunkRate`) object
carrying an explicit `status` field from this vocabulary
(`models/schemas.py::EvidenceStatus`):

| Status | Meaning |
|---|---|
| `OBSERVED` | Read directly from a real source (box score, scraped play-by-play text, a shooting-zone table). |
| `DERIVED` | A deterministic function of `OBSERVED` data (a rate, a ratio, a zone classified from a reported distance). |
| `RECONSTRUCTED` | Inferred from partial/indirect real evidence via a documented method, always carrying a `confidence`. |
| `SIMULATED` | The output of the scenario simulator -- a conditional projection, never a claim about anything that happened. |
| `UNAVAILABLE` | No real source for this field is reachable right now. The field is present with `value=None` and a `reason` -- **never silently dropped, never filled with an invented number.** |

A bare float with no status attached anywhere in this codebase is a bug,
not a shortcut.

## Data sources: what is reachable, and what genuinely is not

This is the single most consequential fact governing the whole system, so
it is stated plainly and up front.

**`stats.nba.com`** (the NBA's own tracking/Synergy/SportVU API, which is
what a tool like `nba_api` wraps) **is completely unreachable from this
build environment.** This was verified directly, not assumed: every live
endpoint call times out with zero bytes returned, while general internet
egress (e.g. `google.com`, `basketball-reference.com`) works fine. This
means touches, post-ups, drives, tracking passing-network data, and exact
shot x/y coordinates are **structurally unavailable** for this whole
project, by design of the environment, not by oversight in the code.

**Basketball-Reference (`basketball-reference.com`) is reachable** and is
this pipeline's real data source, scraped and cached to disk under
`sports/nba/analytics/advantage_routing/data/raw/bball_ref/`
(`sources/cache_manifest.py` writes a `.html` + `.manifest.json` sha256
pair per request; `sources/bball_ref.py` enforces a 2.2s delay between live
requests). What it actually provides:

- Player-search slug resolution (`resolve_player_slug`)
- Real season shooting-zone tables, with real `%Ast'd` per zone and per
  shot-type (`fetch_season_shooting_table`)
- Real per-game play-by-play text, from which real assist attribution and
  real turnovers are parsed (`fetch_season_game_ids`, `fetch_game_events`)
- A real, single-request league-wide shooting baseline -- FG% and shot
  frequency by zone (`fetch_league_shooting_baseline`)

Real per-game box scores also come from this repo's existing
`Player-Predictor/Data-Proc/{Player_Name}/{season}_processed_processed.csv`
files (`sources/boxscore.py`) -- the same real box-score data the rest of
this repo's prediction pipelines already use.

**Sampling strategy:** recipient-network reconstruction samples the 25
most recent real games per player (`sources/collect.py::GAMES_SAMPLED_PER_PLAYER`),
not a full season, to bound network/time risk. This is disclosed in every
player artifact's `sample_description` and `games_sampled` /
`games_available_total` fields -- never presented as a full-season number.

## What this means for the spec's original field list

The original spec assumes touch-level tracking (who touched the ball, in
what state, and where every pass -- not just assisted passes -- went).
Basketball-Reference's play-by-play only records a "pass" when it becomes
a made-shot assist: it has no signal for a missed kick-out, a reset pass,
or any non-scoring pass, and no signal at all for the *origin action*
(drive vs. post touch vs. short roll) that produced a pass. Rather than
approximate these with invented numbers, the pipeline draws an explicit
line:

**Honestly UNAVAILABLE, always:**
- Routing-*state* classification (`SPRAY_3`, `RIM_FEED`, `CUT_RIM`, ... --
  `routing/states.py::classify_routing_state`). Requires possession/touch
  tracking; no reachable source provides it.
- Total pass count, pass share, AST/pass, and the spec's original
  `recipient_leverage` formula (`routing/recipients.py`). Requires knowing
  every pass, not just assisted ones.
- `SHORT_ROLL_GRAVITY` (`gravity/gravity_model.py`). Requires roll-man/
  ball-screen touch tracking.
- `ADVANTAGE_PASS` metrics and the drive/post `routing_vector`
  (`routing/common.py`, `routing/drive.py`, `routing/post.py`). All
  downstream of the same origin-touch requirement.

**What is real and reachable, honestly labeled:**
- Real assist counts, assist share, and zone breakdown per recipient
  (`routing/recipients.py::build_recipient_network`)
- A genuine, disclosed analog of "shot leverage" restricted to the
  assisted-shot sample: `high_value_share_index` (this recipient's
  high-value-assist rate relative to the player's own overall rate) --
  explicitly **not** the spec's original pass-share-based formula.
- Real, `DERIVED` shot-zone classification from reported shot distance +
  2pt/3pt flag (`routing/states.py::classify_shot_zone_from_text`).
  **Cannot distinguish `CORNER_3` from `ABOVE_BREAK_3`** -- bball-ref's
  play-by-play text does not report corner-vs-arc; every 3PA is classified
  `ABOVE_BREAK_3` with an explicit caveat, never silently guessed.
- Five of the six gravity mechanisms (`gravity/gravity_model.py`), built
  from real zone/shot-type shooting splits -- see below.
- An expected-pass-value model built from a real league-wide shooting
  baseline (`stats/pass_value.py`).

## Gravity model (not raw FG%)

`gravity/gravity_model.py::build_gravity_profile` separates scoring gravity
into six mechanisms (`GravityMechanism`), never a single FG% number:

| Mechanism | Status | Built from |
|---|---|---|
| `PAINT_FACEUP_GRAVITY` | `DERIVED` | Real rim + short-paint FGA/FG%, real FTA/FGA rate as a foul-drawing proxy |
| `VERTICAL_GRAVITY` | `OBSERVED`/`DERIVED` | Real dunk FGA/FG%, real 0-3ft rim FGA/FG% -- dunks are an almost-unambiguous vertical-gravity tell |
| `POP_GRAVITY` | `OBSERVED`/`DERIVED` | Real season 3PA, real 3P%. Cannot isolate catch-and-shoot from off-the-dribble (tracking-only split) |
| `PERIMETER_GRAVITY` | `DERIVED` | Real share of FGA from 10ft+ (broader than `POP_GRAVITY` on purpose -- includes mid-range) |
| `POST_SCORING_GRAVITY` | `RECONSTRUCTED`, confidence 0.45 | Real hook-shot volume/efficiency + real complement of the short-paint assisted rate, combined via a documented, transparent formula (`0.5*(hook_FGA/season_FGA)*10 + 0.5*unassisted_short_paint_rate`). This is the one mechanism where a real proxy genuinely substitutes for the ideal (touch-tracked) signal, and it is labeled as such rather than presented as equal in kind to `VERTICAL_GRAVITY`. |
| `SHORT_ROLL_GRAVITY` | `UNAVAILABLE` | No real proxy exists for roll-man frequency without ball-screen/touch tracking; none is invented. |

## Expected pass value

`stats/pass_value.py` implements the spec's "empirical state expectations"
practical-first version:

```
E[points | zone]     = league_fg_pct[zone] * points_value[zone]
E[points | baseline] = sum_z( league_freq[z] * E[points | z] )   (real, zone-selection-weighted league average)
AddedPassValue(zone)  = E[points | zone] - E[points | baseline]
```

Both `league_fg_pct` and `league_freq` come from one real, cached
Basketball-Reference league-wide shooting-baseline request per season
(`sources/bball_ref.py::fetch_league_shooting_baseline`). This is **not**
a possession-level EPV claim -- true possession-level EPV needs play-type/
possession tracking this environment cannot reach.

## Bayesian shrinkage (never hide the raw value)

Low-usage players -- a primary target of this whole system -- have small
real sample sizes, sometimes single-digit assists to a given recipient.
Trusting a raw rate naively is exactly the false precision the
non-negotiable rule above forbids. `stats/shrinkage.py` provides two
textbook, fully transparent tools, both of which **always return the raw
(unshrunk) value alongside the shrunk one**:

- `beta_binomial_shrink(successes, trials, *, prior_mean, prior_strength=8.0, credible_level=0.80)`
  -- shrinks a single rate toward `prior_mean` via a `Beta(alpha, beta)`
  posterior, returning the posterior mean and an equal-tailed credible
  interval.
- `dirichlet_shrink(counts, *, prior=None, prior_strength=6.0)` -- shrinks
  a whole probability vector (e.g. a recipient-share vector) toward a
  prior vector via a Dirichlet concentration; defaults to a uniform prior
  over the observed categories.

This pipeline has no separately-built positional/role prior population
(that itself requires the same unreachable touch-tracking data), so the
default prior is the simplest defensible, fully disclosed choice: the
empirical mean/uniform vector over the observed population passed in by
the caller. Callers may pass a better prior when one is available.

## Simulation: usage is not passing

`simulation/usage.py` implements the role/usage simulator. The guiding
principle, stated explicitly in the module docstring and enforced by
construction: **higher USG% never automatically means more assists.**
Target usage, decision-touch growth, and pass tendency are kept as
separate, explicit inputs.

**Data-honesty note on the baseline:** the spec's model scales a
`baseline_passes` quantity. This pipeline has no real total-pass count
(see above). What it actually scales is `baseline_decision_touches` =
real, `DERIVED` `(FGA + AST + TOV)` per game -- a standard "how often did
this player make a scoring-relevant decision" proxy, **not** a claim about
true touches or true pass volume. Every simulated output is `SIMULATED`
regardless of how good this proxy is, and the simulation is conditional on
this specific proxy choice.

The formulas (verbatim from the spec):

```
H = 1 + e * ((target_usage / current_usage) - 1)
simulated_decision_touches = baseline_decision_touches * H
simulated_passes           = simulated_decision_touches * (1 + pass_tendency_change)
simulated_assists          = simulated_passes * baseline_AST_per_touch * efficiency_retention
simulated_receiver_makes   = simulated_passes * baseline_makes_per_touch * efficiency_retention
simulated_turnovers        = simulated_passes * baseline_TOV_per_touch * (1 + turnover_growth)
```

Role saturation (`simulation/saturation.py`):

```
retention = exp(-k * max(0, H - 1))
```

`retention == 1.0` whenever `H <= 1` (a flat or shrinking role never
"loses" efficiency by this curve's own construction) and decays smoothly
as `H` grows. `k` is an explicit, adjustable slider
(`ScenarioParameters.saturation_k`). The default (non-overridden)
turnover-growth assumption moves the *opposite* way: `turnover_growth =
1/retention - 1` -- the same saturation loss that erodes efficiency also
inflates turnover risk, by construction of this first model.

**Three standard scenarios** (`standard_scenarios`) differ by *fixed*
retention/turnover-growth values rather than re-deriving from the same
saturation curve, so `OPTIMISTIC` really is a distinct assumption from
`CONSERVATIVE`:

| Scenario | Efficiency retention | Turnover growth | Saturation k |
|---|---|---|---|
| `OPTIMISTIC` | 0.97 (fixed) | 0.05 (fixed) | 0.35 |
| `NEUTRAL` | dynamic (saturation curve) | dynamic (saturation curve) | 0.55 |
| `CONSERVATIVE` | 0.80 (fixed) | 0.25 (fixed) | 0.85 |

**Known non-monotonicity, documented rather than hidden:** because
`NEUTRAL` uses its own dynamic saturation curve while `OPTIMISTIC` and
`CONSERVATIVE` use fixed values, `NEUTRAL`'s simulated output is **not
guaranteed to fall between** `OPTIMISTIC` and `CONSERVATIVE` at every
target usage -- at a large role jump, `NEUTRAL`'s saturation-derived
retention can actually be lower than `CONSERVATIVE`'s fixed 0.80. This is
a real, tested property of the model
(`sports/nba/tests/test_advantage_routing.py::test_standard_scenarios_optimistic_beats_conservative_on_fixed_retention`),
not a bug -- the only guaranteed ordering is `OPTIMISTIC` > `CONSERVATIVE`
on both simulated assists (higher) and simulated turnovers (lower), since
those two scenarios share the same touch multiplier `H` and only differ in
their fixed retention/turnover-growth values.

## Uncertainty: Monte Carlo, not a single number

`simulation/monte_carlo.py::run_monte_carlo` draws the underlying
per-touch rates (AST/touch, makes/touch, TOV/touch) from their Beta
posteriors (same prior convention as `stats/shrinkage.py`), propagates
each of 4,000 draws through the scenario formula, and reports
median/P10/P25/P75/P90 rather than a single deterministic number. Every
call takes an explicit integer seed
(`DEFAULT_SEED = 20260823`) and uses a dedicated `numpy.random.Generator`
(never the global numpy random state), so identical inputs always produce
byte-identical output distributions -- this reproducibility is directly
tested.

## Archetypes are descriptive, not grades

`build/archetype.py::build_research_summary` labels a player with zero or
more `Archetype` tags (`POST_MANIPULATOR`, `ADVANTAGE_PROCESSOR`,
`STRUCTURAL_HUB`, `CONNECTOR`, `VERTICAL_GRAVITY_PROCESSOR`,
`SCORE_FIRST_POST`, `FINISHER`) purely as a transparent, rule-based
function of already-computed real metrics -- never a fitted model, never a
ranking, and a player may carry more than one tag.

## Known limitations

- **Touch-level fields are structurally unavailable in this environment**
  (see "Data sources" above) -- this is the dominant limitation and shapes
  every other section of this document.
- **Corner-3 vs. above-break-3 cannot be distinguished** from
  Basketball-Reference's play-by-play text; all 3PA are classified
  `ABOVE_BREAK_3` with an explicit caveat.
- **Pass volume is undercounted by construction** -- only passes that
  became made-shot assists are visible. `passes`, `pass_share`,
  `ast_per_pass`, and the original `recipient_leverage` formula are
  `UNAVAILABLE`, not approximated.
- **`POST_SCORING_GRAVITY` is a moderate-low-confidence reconstruction**
  (confidence 0.45), not a tracking-grade measurement.
- **Recipient-network sampling is bounded to the 25 most recent games**
  per player, not a full season -- always disclosed in
  `sample_description`.
- **A pre-existing data-quality issue, out of scope for this project:**
  `Jamal Murray`'s real `USG%` values in
  `Player-Predictor/Data-Proc/Jamal_Murray/2026_processed_processed.csv`
  (11-18% range) are implausibly low for his real, correctly-populated
  PTS/FGA/MP/AST stat line (25.5 PPG, 18.2 FGA, 35 MP -- all real and
  matching public Murray stats). This is almost certainly a pre-existing
  formula bug in `Player-Predictor`'s own USG% calculation (most likely a
  missing team-pace normalization step), predating and unrelated to this
  project. It was deliberately **not** fixed here because
  `Player-Predictor`'s pipeline is used elsewhere in this repo for real
  predictions and is out of this project's scope -- but it is disclosed
  here because Jamal Murray's `current_usage_pct` baseline (and therefore
  every usage-simulation output for him) inherits this understated number.

## File layout

```
sports/nba/analytics/advantage_routing/
  models/schemas.py          Metric/EvidenceStatus, state + gravity + archetype vocab, AdvantageEvent
  sources/
    boxscore.py               real per-game box scores (Player-Predictor CSVs)
    bball_ref.py               real Basketball-Reference scraper (cached)
    cache_manifest.py          disk cache with sha256 manifests
    collect.py                  per-player real-data bundle collector (25-game sample)
  routing/
    states.py                   shot-zone classification (DERIVED) + routing-state honesty boundary (UNAVAILABLE)
    recipients.py                real recipient-network analysis
    common.py, drive.py, post.py  routing vectors / advantage-pass metrics (UNAVAILABLE, see above)
  gravity/gravity_model.py       six-mechanism gravity model
  stats/
    shrinkage.py                 Beta-Binomial + Dirichlet shrinkage
    pass_value.py                 expected pass value model
  simulation/
    saturation.py                 role-saturation curve
    usage.py                       role/usage scenario simulator
    monte_carlo.py                  Beta-posterior Monte Carlo uncertainty
  build/
    build_player.py                per-player orchestrator -> JSON artifact
    build_all.py                     seed-population batch build + players.json index
    validate.py                       reconciliation report generator
    archetype.py                       rule-based archetype labeling

sports/nba/web/
  advantage-routing.html / .css / .js   the frontend page (flat top-level files, matching predictions.html's convention)
  data/advantage-routing/*.json         generated player artifacts + players.json index

sports/nba/tests/test_advantage_routing.py   unit + reconciliation tests (no live network required)
docs/advantage-routing.md                     this document
```

## CLI

```bash
# Build one player's artifact
python -m sports.nba.analytics.advantage_routing.build.build_player \
  --player "Derik Queen" --season 2025-26 [--mode drive|post|both] \
  [--games-sampled 25] [--output-root PATH]

# Build the full seed population + players.json index
python -m sports.nba.analytics.advantage_routing.build.build_all \
  --season 2025-26 [--players "Name A" "Name B" ...] [--output-root PATH]

# Reconciliation report over already-built artifacts
python -m sports.nba.analytics.advantage_routing.build.validate \
  [--output-root PATH]   # exits 1 on any FAIL

# Tests
python -m pytest sports/nba/tests/test_advantage_routing.py -q
```

## Adding another player

The pipeline generalizes to any player with a real box-score CSV under
`Player-Predictor/Data-Proc/{Player_Name}/` and a resolvable
Basketball-Reference slug -- it is **not** hardcoded around the seed
population (`Derik Queen`, `Collin Murray-Boyles`, `Donovan Clingan`,
`Yves Missi`, `Jamal Murray`). To add one:

1. Confirm real box-score data exists:
   `sources/boxscore.py::list_available_players(season)`.
2. Run `build_player.py --player "New Player Name"` directly, or add the
   name to `build/build_all.py::SEED_PLAYERS` (or pass `--players` to
   `build_all.py`) to include them in the batch build.
3. Re-run `validate.py` and confirm the new artifact reconciles cleanly.
4. The frontend's player selector is driven entirely by the generated
   `players.json` index -- no frontend code change is needed to add a
   player.

Each per-player build fails independently (`build_all.py` wraps each
player in its own try/except) so one player's data problem never corrupts
the rest of the batch.

## Frontend data contract

The frontend (`sports/nba/web/advantage-routing.js`) fetches
`data/advantage-routing/players.json` (the player index) and
`data/advantage-routing/{slug}.json` (one player's full artifact) relative
to the page -- matching the same `data/...`-relative convention already
used by `predictions.js`. Every numeric value the frontend renders is
read from a `Metric`-shaped object (`{value, status, source|method,
confidence?, reason?}`); the page never renders a bare number without
also rendering its provenance badge, and an `UNAVAILABLE` metric is always
shown as an explicit, worded notice -- never as a blank, a zero, or a
silently-omitted section. The interactive usage/pass-tendency simulator
re-implements `simulation/usage.py`'s exact formula client-side
(`AdvantageRoutingPage.simulateLive`) against the real baseline numbers
already embedded in the player JSON, so slider interaction requires no
server round-trip.

## Build/integration note: static-site pipeline

`sports/site/pipeline/build_static_site.py` prunes every sport's `web/`
output down to an allowlist of prediction-related files
(`PREDICTION_TOP_LEVEL_FILES`, `PREDICTION_PAGE_STEMS`,
`PREDICTION_DATA_FILES`) before publishing to `dist/`. The
advantage-routing page is registered in all three: `advantage-routing` is
in `PREDICTION_PAGE_STEMS` (nav discovery + clean-URL routing via
`create_clean_routes`), `advantage-routing.html/.js/.css` are in
`PREDICTION_TOP_LEVEL_FILES` (survive pruning), and
`data/advantage-routing/*.json` survives via a dedicated
`is_advantage_routing_payload` check in `prune_non_prediction_assets`
(mirroring the existing `history/` subdirectory allowance). The page's
asset references use the same flat, non-nested convention as
`predictions.html` (`vault/...`, rewritten to `/vault/...` by
`use_public_shared_assets` at build time) -- a nested subdirectory page
would have its relative vault references silently broken by that same
rewrite step, which only processes top-level HTML files.
