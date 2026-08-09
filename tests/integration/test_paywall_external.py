"""Checks against a deployed public prediction environment."""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest


BASE_URL = os.getenv("SITE_E2E_BASE_URL", os.getenv("PAYWALL_E2E_BASE_URL", "")).rstrip("/")

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


def test_prediction_routes_and_payloads_are_public():
    assert request("/").status == 200
    assert request("/nba/predictions/").status == 200
    assert request("/mlb/predictions/").status == 200
    assert request("/mlb/data/daily_predictions.json").status == 200
