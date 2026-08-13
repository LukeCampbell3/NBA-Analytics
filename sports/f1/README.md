# Formula 1 race prediction workspace

This workspace mirrors the NBA, MLB, and NFL publication pattern with an
independent model, odds collector, validation command, web route, and scheduled
GitHub Actions workflow.

## What it predicts

The v1 model publishes win, podium, and top-six probabilities for the next
scheduled Grand Prix. The visible market comparison is deliberately limited to
race-winner outrights because it is the cleanest common market across providers.
All outputs remain `live_shadow`; staking is disabled while prospective results
accumulate.

Pre-race features are calculated chronologically and include starting grid when
available, driver and constructor rolling form, DNF rate, circuit history,
season points share, and championship position. Every historical row is built
from state available before that race. The last 20% of races are held out in
time order for the published Brier, winner log-loss, and top-pick metrics.

Historical results and the schedule come from the
[Jolpica F1 API](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md).
[OpenF1](https://openf1.org/docs/) is an optional best-effort post-qualifying
grid source. Winner prices come from the documented public
[Polymarket](https://docs.polymarket.com/api-reference/introduction) and
[Kalshi](https://docs.kalshi.com/getting_started/quick_start_market_data) read
APIs. Both are credential-free and return executable YES asks; absence of a
matching market never turns into a fabricated price.

## Run locally

```bash
python -m pip install -r sports/f1/requirements.txt
python sports/f1/scripts/run_f1_daily_predictions.py
python sports/f1/scripts/validate_f1_publication.py
python sports/site/pipeline/build_static_site.py
```

No odds credential or paid subscription is required. Optional configuration:

- `F1_ODDS_PROVIDER_PRIORITY=polymarket,kalshi`

Use `--skip-odds` to inspect model probabilities without making market-data
requests. The scheduled workflow lives at `.github/workflows/f1-predictions.yml`.
