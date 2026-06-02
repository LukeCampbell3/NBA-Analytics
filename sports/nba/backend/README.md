# NBA Analytics Access Control Backend

Lightweight API for auth, Stripe billing, entitlements, API keys, and artifact-gated data access.

**Doctrine:** cloud handles access control only; all model compute stays local.

## Setup

```bash
pip install -r sports/nba/backend/requirements-backend.txt
cp sports/nba/backend/.env.example sports/nba/backend/.env
python sports/nba/backend/db/migrate.py
uvicorn sports.nba.backend.api.app:app --reload --port 8787
```

Set `meta name="nba-api-base"` in static pages to `http://localhost:8787` during local dev.

## Local artifact publish

```bash
python sports/nba/predictions/Player-Predictor/research/site_export/publish_local_artifacts.py ^
  --source-dir sports/nba/web/data ^
  --target-dir sports/nba/web/data
```

## Stripe

1. Create products/prices in Stripe.
2. Update `plans.stripe_price_id` in the database.
3. Point webhook to `POST /api/stripe/webhook`.

## Tests

```bash
pytest sports/nba/tests/test_monetization_layer.py -q
```
