"""Stripe billing helpers — secrets stay server-side."""
from __future__ import annotations

import os
from typing import Any


def stripe_enabled() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY", "").strip())


def get_stripe_module():
    import stripe

    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "").strip()
    return stripe


def create_checkout_session(
    *,
    customer_id: str | None,
    customer_email: str,
    stripe_price_id: str,
    success_url: str,
    cancel_url: str,
    metadata: dict[str, str],
) -> dict[str, Any]:
    stripe = get_stripe_module()
    payload: dict[str, Any] = {
        "mode": "subscription",
        "line_items": [{"price": stripe_price_id, "quantity": 1}],
        "success_url": success_url,
        "cancel_url": cancel_url,
        "metadata": metadata,
        "allow_promotion_codes": True,
    }
    if customer_id:
        payload["customer"] = customer_id
    else:
        payload["customer_email"] = customer_email
    session = stripe.checkout.Session.create(**payload)
    return {"id": session.id, "url": session.url}


def create_portal_session(*, customer_id: str, return_url: str) -> dict[str, Any]:
    stripe = get_stripe_module()
    session = stripe.billing_portal.Session.create(customer=customer_id, return_url=return_url)
    return {"id": session.id, "url": session.url}


def construct_webhook_event(payload: bytes, signature: str):
    stripe = get_stripe_module()
    secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip()
    return stripe.Webhook.construct_event(payload, signature, secret)
