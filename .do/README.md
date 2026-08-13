# DigitalOcean deployment boundary

`app.yaml.example` is intentionally non-deployable and contains no credentials.
DigitalOcean Functions components expose actions as
`component-route/package/function`; this repository therefore mounts the component
at `/functions`, the package at `/paywall`, and the gateway action at `/gateway`.
The application router receives the remaining suffix, such as `/api/account`.

Before deployment:

1. Copy the example spec to a temporary path outside the repository.
2. Replace every `REPLACE_*` value and add the custom production domain.
3. Keep all credentials and cryptographic keys as `SECRET` environment values.
4. Run `doctl apps spec validate <temporary-spec>`.
5. Create or update the app only after the R2 integration suite passes.
6. Delete the temporary plaintext spec after DigitalOcean has encrypted its values.

The `public-site` component tracks `static-deployment` with
`deploy_on_push: true`, so each push to that branch starts a frontend build. The
`paywall-functions` component remains manual (`deploy_on_push: false`) so a
frontend publication does not unnecessarily redeploy the functions.

The webhook URL is:

`https://<domain>/functions/paywall/payment-webhook/api/webhooks/stripe`

The Discord callback URL is:

`https://<domain>/functions/paywall/gateway/auth/discord/callback`
