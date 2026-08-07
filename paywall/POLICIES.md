# Production membership policies

These rules are part of authorization behavior, not frontend suggestions.

## Cancellation and access

- Cancellation scheduled for the end of a paid period keeps the Stripe subscription `active`; access therefore continues until Stripe ends that period.
- Immediate cancellation produces a non-authorized account status and ends access on the next webhook or authorization refresh.
- `past_due`, `unpaid`, `paused`, `incomplete`, and unknown provider states do not receive a grace lease. This is the deliberate fail-closed grace policy for the first release.
- Support may grant an exceptional grace period only through an explicit, audited administrative procedure that sets `status=grace` and an absolute `valid_until`. There is no automatic rolling grace extension.

## Refunds and disputes

- Duplicate charges and verified billing errors should be refunded.
- Other requests are reviewed under applicable law and Stripe rules.
- A full or prorated refund must be paired with immediate subscription cancellation in Stripe. The subscription update/deletion webhook is the authoritative access revocation event.
- A dispute or chargeback must be configured in Stripe to cancel the related subscription immediately. Until that Stripe setting is verified in the credential-gated configuration test, production launch is blocked.
- Issuing a refund without canceling the subscription is an operator error because a refund event alone is not treated as entitlement authority.

## Account deletion

- Active, grace, past-due, suspended, or banned accounts cannot self-delete. Billing/support must first reach the terminal `canceled` state.
- Pending and canceled accounts may self-delete with a fresh signed session, exact-origin CSRF validation, and a conditional account mutation.
- Deletion pseudonymizes the display name, removes encrypted payment references and entitlement data, clears checkout state, increments `session_epoch`, and records immutable history/audit evidence.
- The HMAC identity index and minimal deleted-account tombstone remain during the security, dispute, and statutory retention period. They prevent duplicate-account recreation and payment-event resurrection.
- Permanent tombstone/history destruction is an offline administrative retention operation and must not be exposed as a browser endpoint.
