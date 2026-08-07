"""Credential-gated checks against a deployed custom-domain environment."""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request

import pytest


BASE_URL = os.getenv("PAYWALL_E2E_BASE_URL", "").rstrip("/")
GATEWAY = os.getenv("PAYWALL_E2E_GATEWAY_PATH", "/functions/paywall/gateway")
WEBHOOK = os.getenv("PAYWALL_E2E_WEBHOOK_PATH", "/functions/paywall/payment-webhook")

pytestmark = pytest.mark.skipif(not BASE_URL, reason="PAYWALL_E2E_BASE_URL is not configured")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(path: str, *, method: str = "GET", data: bytes | None = None, headers=None):
    opener = urllib.request.build_opener(NoRedirect)
    req = urllib.request.Request(BASE_URL + path, method=method, data=data, headers=headers or {})
    try:
        return opener.open(req, timeout=15)
    except urllib.error.HTTPError as error:
        return error


def test_live_endpoint_and_public_boundary():
    live = request(GATEWAY + "/health/live")
    assert live.status == 200
    assert request("/nba/data/daily_predictions.json").status == 404
    assert request("/mlb/predictions/").status == 404


def test_protected_content_fails_closed_without_session():
    response = request(GATEWAY + "/app/")
    assert response.status in {401, 403}
    assert b"releases/" not in response.read()


def test_discord_start_uses_state_and_identify_only():
    response = request(GATEWAY + "/auth/discord/start?redirect=/app/")
    assert response.status == 302
    location = response.headers["Location"]
    parsed = urllib.parse.urlparse(location)
    query = urllib.parse.parse_qs(parsed.query)
    assert parsed.netloc == "discord.com"
    assert query.get("scope") == ["identify"]
    assert query.get("state") and len(query["state"][0]) >= 40
    cookie = response.headers.get("Set-Cookie", "")
    assert "__Host-oauth_state=" in cookie
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=Lax" in cookie


def test_invalid_webhook_signature_is_rejected():
    response = request(
        WEBHOOK + "/api/webhooks/stripe",
        method="POST",
        data=b'{}',
        headers={"Content-Type": "application/json", "Stripe-Signature": "invalid"},
    )
    assert response.status == 400
