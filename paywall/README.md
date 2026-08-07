# Security-first paywall service

This directory contains the tested security core for the DigitalOcean Functions
and private Cloudflare R2 architecture. The repository builder now emits an
explicitly public `dist/` shell and a separate protected release source here.

## Implemented

- Explicit, time-bounded account authorization policy.
- HMAC Discord identity indexes and random immutable account IDs.
- Index-first, idempotent, repairable account creation.
- Strict account decoding, strong schema validation, and fail-closed unknown statuses.
- ETag compare-and-swap account updates and immutable revision history.
- R2 state adapter with `GetObject` and conditional `PutObject` only. There is no listing API.
- Separate R2 content adapter for bounded reads and 30-120 second presigned GETs.
- One-time, ETag-consumed OAuth state with redirect allowlisting.
- Discord authorization-code client requesting only `identify`.
- HMAC sessions with current/previous key rotation and short authorization leases.
- Session-bound double-submit CSRF, exact-origin checks, secure `__Host-` cookies, and browser headers.
- AES-256-GCM PII fields using account ID as associated authenticated data.
- Per-account checkout locks with provider idempotency-key reuse.
- Raw-body Stripe signature verification with timestamp tolerance.
- Payment-event idempotency, processing leases, authoritative entitlement replacement, and replay recovery.
- Immutable content manifests with exact logical-path and plan resolution.
- Deployable DigitalOcean `gateway` and raw `payment-webhook` function entrypoints.
- Discord login, session bootstrap, account status, logout, logout-all, Checkout, and billing portal routes.
- Stripe v85 Checkout, customer portal, and authoritative subscription reconciliation.
- Protected HTML/data proxying, fresh-authorized short-lived downloads, immutable release pointers, and manifest caching without bucket listing.
- Offline immutable content deploy and conditional rollback commands with post-upload SHA-256 verification.
- Offline administrative suspension and explicit-revision account recovery with automatic session revocation.
- Fail-closed 24-hour lazy Stripe reconciliation on login and authorization refresh for subscribed accounts.
- Immutable date-partitioned audit events for login, checkout, billing portal, logout-all, payment reconciliation, suspension, and recovery.

## Important session correction

An authorization lease cannot safely be represented by `authz_exp` alone when a
new login may belong to a pending account. Tokens therefore also carry a plan and
an entitlement expiry. The signer refuses to create a future authorization lease
without both. Pending accounts receive identity-only tokens whose authorization
lease expires immediately.

## Production boundary status

The route/asset classification, public/private build split, cancellation/refund/
deletion/grace policy, App Platform routing template, and credential-gated test
harnesses are implemented. The remaining work is intentionally external:

1. Run the integration-tag suite against private non-production R2 buckets.
2. Fill and validate `.do/app.yaml.example`, attach the custom domain, and run a remote build.
3. Register the exact Discord callback and Stripe webhook URLs.
4. Run Discord OAuth, Stripe test-mode, and protected-content end-to-end tests.

`dist/` is now safe-by-construction as the public static artifact. Boundary tests
reject sport roots, prediction payloads/scripts, protected route roots, and byte-
identical copies of private release objects.

## DigitalOcean layout correction

DigitalOcean Functions only includes `project.yml`, `packages/`, and `lib/` in a
build. Shared code therefore lives under `lib/`, and each entrypoint lives under
`packages/paywall/`. Both functions use `web: raw`; the webhook consequently
receives the unmodified request body required for signature verification.

App Platform Functions URLs include `component-route/package/function`. The
component is mounted at `/functions`, so the public gateway prefix is
`/functions/paywall/gateway`; the raw event path after that prefix remains the
documented API surface. The webhook prefix is
`/functions/paywall/payment-webhook`.

## Content control-plane correction

The activation pointer lives at `system/current-content-release.json` in the
private content bucket, not the account-state bucket. This is required to keep
the offline content-deploy credential out of the bucket containing accounts,
OAuth state, payment events, and audit history. The gateway's runtime content
credential reads the pointer; only the offline deploy credential can replace it.

The deployment tools are under `tools/cmd/content-deploy` and
`tools/cmd/content-rollback`. Release objects and manifests are immutable, every
uploaded object is read back and SHA-256 verified, and pointer replacement uses
`If-None-Match` or `If-Match`.

The offline operator tool is under `tools/cmd/admin-cli`. It supports manual
account suspension and recovery of a deleted canonical account from an
explicitly selected immutable history revision. Recovery refuses to overwrite
an existing canonical object or cross a newer history revision.

`tools/cmd/audit-export` lists only one validated UTC date prefix and writes
strictly decoded JSON Lines. Bucket listing remains absent from every runtime
interface and normal account workflow.

## Test

Use Go 1.25:

```text
cd lib
go test -race ./...
```

The suite includes the required 100-concurrent-request identity test along with
CAS, replay, lease recovery, session, CSRF, signature, traversal, and payment-order tests.

Run the real R2 tests only against bucket names containing `test` or `staging`:

```text
cd tools
go test -tags=integration -run R2 -v ./integration
```

The tests refuse other bucket names and delete only the exact random keys they
create. Required variables are documented in `EXTERNAL_SETUP.md`.
