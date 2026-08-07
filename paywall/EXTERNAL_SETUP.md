# Credential-gated production validation

No secret value belongs in this repository. Configure these values in a local
secret manager or encrypted DigitalOcean environment variables and never print
them in CI logs.

## Private R2 integration

Use disposable buckets whose names contain `test` or `staging`. The integration
suite refuses to write anywhere else.

State test credential (read, write, delete exact test objects):

- `R2_TEST_STATE_ENDPOINT`
- `R2_TEST_STATE_ACCESS_KEY_ID`
- `R2_TEST_STATE_SECRET_ACCESS_KEY`
- `R2_TEST_STATE_BUCKET`

Content deployment test credential (read, write, delete exact test objects):

- `R2_TEST_CONTENT_DEPLOY_ENDPOINT`
- `R2_TEST_CONTENT_DEPLOY_ACCESS_KEY_ID`
- `R2_TEST_CONTENT_DEPLOY_SECRET_ACCESS_KEY`
- `R2_TEST_CONTENT_DEPLOY_BUCKET`

Content runtime test credential (read and presign the same content test bucket):

- `R2_TEST_CONTENT_RUNTIME_ENDPOINT`
- `R2_TEST_CONTENT_RUNTIME_ACCESS_KEY_ID`
- `R2_TEST_CONTENT_RUNTIME_SECRET_ACCESS_KEY`
- `R2_TEST_CONTENT_RUNTIME_BUCKET`

Run from `paywall/tools`:

```text
go test -tags=integration -run R2 -v ./integration
```

## DigitalOcean and custom domain

Required operator input:

- A scoped `DIGITALOCEAN_ACCESS_TOKEN` for App Platform create/update/read.
- The production custom HTTPS origin and hostname.
- The private GitHub repository connection if DigitalOcean cannot already read it.
- The Cloudflare DNS zone authority needed to create the DigitalOcean verification/CNAME records.

Use `.do/app.yaml.example` as the source template. The production Discord
callback is:

`https://<domain>/functions/paywall/gateway/auth/discord/callback`

## Discord test application

- `DISCORD_CLIENT_ID`
- `DISCORD_CLIENT_SECRET`
- The callback URL above registered exactly.
- A test Discord account allowed to authorize the application.

Only the `identify` scope is requested.

## Stripe test mode

- `PAYMENT_SECRET_KEY` (`sk_test_...`)
- `PAYMENT_WEBHOOK_SECRET` (`whsec_...`) for the deployed webhook endpoint
- `PAYMENT_PRICE_ID` (`price_...`) for one recurring individual plan
- A Stripe test customer/payment method created through Checkout

Configure the webhook endpoint as:

`https://<domain>/functions/paywall/payment-webhook/api/webhooks/stripe`

Subscribe only to `customer.subscription.created`,
`customer.subscription.updated`, and `customer.subscription.deleted`. Verify
in Stripe test mode that dispute handling cancels the subscription immediately,
as required by `POLICIES.md`.

## Remote safe smoke tests

Set `PAYWALL_E2E_BASE_URL=https://<domain>` and run:

```text
python -m pytest tests/integration/test_paywall_external.py -q
```

These checks are non-mutating: health, public-byte boundary, unauthenticated
content denial, Discord authorization redirect shape, and invalid webhook
signature rejection. Complete Checkout, valid webhook replay/order, billing
portal, refund/cancellation, session refresh, and protected download tests using
Stripe test mode and the test Discord account.
