# MLB unified production preflight audit

Audit baseline: `static-deployment@c42bf2c1579d140b72efc5597fb9d074834ddfb4`  
Frozen unified hypothesis: `7c56729a9914eb9f903edffe9ca58b1a0a749ad4`

## Repository and deployment

The repository is `LukeCampbell3/NBA-Analytics`. Promotion work is isolated on
`unified-mlb-production-promotion`, derived exactly from the frozen unified
commit. The production branch is `static-deployment`.

MLB generation is orchestrated by `.github/workflows/mlb-predictions.yml` and
`sports/site/pipeline/run_daily_predictions.py`. The workflow checks out
`static-deployment`, generates same-game, pitcher, exotic and daily artifacts,
validates them, rebuilds the static site, stages the generated files, and pushes
a normal fast-forward commit back to `static-deployment`. Workflow concurrency
uses `mlb-predictions-static-deployment` with `cancel-in-progress: false`.

The site builder copies `sports/*/web` into `dist/<sport>` and the compatibility
tree under `paywall/private-content/app/<sport>`. Repository documentation states
that `dist/` is the public static component. The canonical deployed MLB route is
`https://inthecardsanalytics.com/mlb/predictions/`.

The MLB frontend currently consumes:

* `daily_predictions.json`;
* `same_game_predictions.json`;
* `pitcher_parlay_predictions.json`;
* `exotic_market_predictions.json`;
* `unified_predictions.json` on the unified branch;
* dated history and history index artifacts.

The legacy artifacts and renderer remain the rollback path. No production
authority is inferred from the existence of `unified_predictions.json`.

## Environment and dependencies

Actions use Ubuntu and Python 3.11. Root `requirements.txt` supplies NumPy,
SciPy, pandas, PyArrow, requests, BeautifulSoup, joblib, scikit-learn, CatBoost,
XGBoost, h5py, PyYAML and TensorFlow; the same-game workflow adds
`sports/mlb/requirements-same-game.txt`. Node is used only for frontend syntax
validation in this migration; repository `package.json` requires Node >=18 and
has a committed lock file.

The daily workflow uses America/New_York for slate dates. Timestamps in evidence
and artifacts are UTC. Relevant optional/provider configuration includes
`MLB_ODDS_PROVIDER_PRIORITY`, `MLB_ODDS_MAX_AGE_SECONDS`, scrape authorization,
`THE_ODDS_API_KEY`, `MLB_FANDUEL_REGION`, and FanDuel rate limiting. Missing
optional provider state must block only the affected capability. No secret is
written to artifacts.

## Branch and release controls

GitHub's branch summary reported `static-deployment` as not protected; the
integration received HTTP 403 when querying the detailed protection endpoint,
so no assumption is made about unobservable rules. No force push or protection
change is authorized. Production rollback is the existing legacy engine plus
the exact pre-migration production SHA. A durable tag is deliberately not
created unless the historical promotion gate passes and a production merge is
actually authorized.

## Historical evidence preflight

The repository contains 242,425 settled rows across 156 slates in
`historical_pool_universe_2026.csv`. It contains structural predictions and
sparse real quote/timestamp fields, but it does not contain the frozen unified
policy's final/usable probability, uncertainty, lineup/role state, or exact
calibration state.

Git history preserves the authoritative daily artifact only for August 29–30.
It yields one deduplicated exact pregame unified candidate, but no preserved
settlement for that candidate. Consequently, the exact settled corpus for the
frozen unified policy is empty. Recomputing a different probability from the
large settled universe would validate another policy and is prohibited.

## Preflight conclusion

Software hardening and validation tooling may proceed. Statistical promotion,
production dark merge, live canary authority, and activation are blocked unless
the locked corpus meets the predeclared policy. The legacy production system
therefore remains authoritative.
