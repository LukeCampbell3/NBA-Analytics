from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from sports.nba.backend.api.app import app
from sports.nba.backend.db.connection import get_connection, run_migrations
from sports.nba.backend.db.repository import (
    generate_api_key,
    get_entitlements_for_user,
    get_or_create_user,
    hash_api_key,
    upsert_subscription,
)
from sports.nba.backend.entitlements.entitlement_rules import (
    PLAN_ENTITLEMENTS,
    entitlements_for_plan,
)
from sports.nba.backend.services import artifacts


@pytest.fixture()
def test_env(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("JWT_SECRET", "test-secret")
    monkeypatch.setenv("ALLOW_DEV_AUTH", "1")
    monkeypatch.setenv("SKIP_DB_MIGRATIONS", "0")
    monkeypatch.setenv("ARTIFACT_DATA_DIR", str(data_dir))
    monkeypatch.delenv("STRIPE_SECRET_KEY", raising=False)
    run_migrations()
    _write_sample_artifacts(data_dir)
    with TestClient(app) as client:
        yield client, data_dir


def _write_sample_artifacts(data_dir: Path) -> None:
    cards = [
        {
            "player": "Alpha",
            "market_type": "PTS_OVER",
            "side": "OVER",
            "line": 20.5,
            "safe_state_tier": "SAFE",
            "edge_defendability_tier": "DEFENSIBLE",
            "recommended_action": "MONITOR",
            "settlement_status": "PENDING",
            "explanation": "shadow",
            "warning_badges": [],
            "shadow_only": True,
        },
        {
            "player": "Beta",
            "market_type": "REB_OVER",
            "side": "OVER",
            "line": 8.5,
            "safe_state_tier": "WATCH",
            "edge_defendability_tier": "UNKNOWN",
            "recommended_action": "CANDIDATE_POOL_ONLY",
            "settlement_status": "PENDING",
            "explanation": "shadow",
            "warning_badges": [],
            "shadow_only": True,
        },
        {
            "player": "Gamma",
            "market_type": "AST_OVER",
            "side": "OVER",
            "line": 6.5,
            "safe_state_tier": "SAFE",
            "edge_defendability_tier": "DEFENSIBLE",
            "recommended_action": "MONITOR",
            "settlement_status": "SETTLED",
            "explanation": "shadow",
            "warning_badges": [],
            "shadow_only": True,
        },
        {
            "player": "Delta",
            "market_type": "PRA_OVER",
            "side": "OVER",
            "line": 30.5,
            "safe_state_tier": "SAFE",
            "edge_defendability_tier": "DEFENSIBLE",
            "recommended_action": "MONITOR",
            "settlement_status": "PENDING",
            "explanation": "shadow",
            "warning_badges": [],
            "shadow_only": True,
        },
    ]
    latest = {
        "run_id": "run_test_1",
        "run_date": "2026-05-29",
        "data_cutoff_date": "2026-05-28",
        "shadow_only": True,
        "promotion_ready": False,
        "production_behavior_changed": False,
        "cards": cards,
    }
    (data_dir / "safe_state_latest.json").write_text(json.dumps(latest), encoding="utf-8")
    (data_dir / "safe_state_cards.json").write_text(json.dumps(cards), encoding="utf-8")
    (data_dir / "site_manifest.json").write_text(json.dumps({"run_id": "run_test_1"}), encoding="utf-8")
    (data_dir / "player_simulation_cards.json").write_text(
        json.dumps([{"player": "Alpha", "player_id": "alpha", "confidence_tier": "MEDIUM", "volatility_score": 0.2, "missing_data_warnings": [], "data_cutoff_date": "2026-05-28", "pts": {"p10": 1, "p50": 2, "p90": 3}}]),
        encoding="utf-8",
    )


def _login(client: TestClient, email: str = "paid@example.com") -> str:
    response = client.post("/api/auth/dev-session", json={"email": email})
    assert response.status_code == 200
    token = response.json()["access_token"]
    return token


def test_migrations_create_core_tables(test_env):
    client, _ = test_env
    with get_connection() as conn:
        from sports.nba.backend.db.connection import _fetchall

        tables = _fetchall(
            conn,
            "SELECT name FROM sqlite_master WHERE type='table'",
        )
        names = {row["name"] for row in tables}
    for table in ["users", "plans", "subscriptions", "entitlements", "api_keys", "api_usage", "artifact_runs", "audit_events"]:
        assert table in names
    assert client.get("/api/health").json()["inference"] == "disabled"


def test_entitlement_rules_map_plans():
    free = entitlements_for_plan("free", "inactive")
    plus = entitlements_for_plan("plus", "active")
    pro = entitlements_for_plan("pro", "active")
    api = entitlements_for_plan("api", "active")
    assert free.max_cards_per_day == 3
    assert plus.can_view_full_safe_state is True
    assert pro.can_export_csv is True
    assert api.can_use_api is True


def test_api_key_stores_hash_only(test_env):
    _, _ = test_env
    user = get_or_create_user("api@example.com")
    raw, row = generate_api_key(user["id"], "test")
    assert raw.startswith("nba_live_")
    assert row["key_prefix"]
    with get_connection() as conn:
        from sports.nba.backend.db.connection import _fetchone

        stored = _fetchone(conn, "SELECT key_hash FROM api_keys WHERE id = ?", (row["id"],))
    assert stored["key_hash"] == hash_api_key(raw)
    assert raw not in stored["key_hash"]


def test_checkout_rejects_invalid_plan(test_env):
    client, _ = test_env
    token = _login(client)
    response = client.post(
        "/api/billing/create-checkout-session",
        headers={"Authorization": f"Bearer {token}"},
        json={"plan_id": "free", "success_url": "http://x/s", "cancel_url": "http://x/c"},
    )
    assert response.status_code in {400, 503}


def test_webhook_requires_signature(test_env, monkeypatch):
    client, _ = test_env
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_x")
    response = client.post("/api/stripe/webhook", content=b"{}", headers={"stripe-signature": "bad"})
    assert response.status_code == 400


def test_subscription_updates_entitlements(test_env):
    client, _ = test_env
    user = get_or_create_user("plus@example.com")
    upsert_subscription(
        user_id=user["id"],
        stripe_customer_id="cus_test",
        stripe_subscription_id="sub_test",
        stripe_price_id="price_plus",
        plan_id="plus",
        status="active",
    )
    entitlements = get_entitlements_for_user(user["id"])
    assert entitlements["plan_id"] == "plus"
    assert entitlements["can_view_full_safe_state"] in {True, 1}


def test_canceled_subscription_removes_paid_entitlements(test_env):
    user = get_or_create_user("cancel@example.com")
    upsert_subscription(
        user_id=user["id"],
        stripe_customer_id="cus_cancel",
        stripe_subscription_id="sub_cancel",
        stripe_price_id="price_pro",
        plan_id="pro",
        status="active",
    )
    upsert_subscription(
        user_id=user["id"],
        stripe_customer_id="cus_cancel",
        stripe_subscription_id="sub_cancel",
        stripe_price_id="price_pro",
        plan_id="pro",
        status="canceled",
    )
    entitlements = get_entitlements_for_user(user["id"])
    assert entitlements["plan_id"] == "free"


def test_free_user_limited_cards(test_env):
    client, _ = test_env
    response = client.get("/api/safe-state/cards")
    assert response.status_code == 200
    body = response.json()
    assert len(body["cards"]) <= 3
    assert body["shadow_only"] is True


def test_plus_user_full_safe_state(test_env):
    client, _ = test_env
    user = get_or_create_user("plus2@example.com")
    upsert_subscription(
        user_id=user["id"],
        stripe_customer_id="cus_plus2",
        stripe_subscription_id="sub_plus2",
        stripe_price_id="price_plus",
        plan_id="plus",
        status="active",
    )
    token = _login(client, "plus2@example.com")
    response = client.get("/api/safe-state/cards", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert len(response.json()["cards"]) >= 3


def test_pro_user_csv_export(test_env):
    client, _ = test_env
    user = get_or_create_user("pro@example.com")
    upsert_subscription(
        user_id=user["id"],
        stripe_customer_id="cus_pro",
        stripe_subscription_id="sub_pro",
        stripe_price_id="price_pro",
        plan_id="pro",
        status="active",
    )
    token = _login(client, "pro@example.com")
    response = client.get("/api/safe-state/export.csv", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert "player" in response.text


def test_api_user_access_and_non_api_blocked(test_env):
    client, _ = test_env
    user = get_or_create_user("apiuser@example.com")
    upsert_subscription(
        user_id=user["id"],
        stripe_customer_id="cus_api",
        stripe_subscription_id="sub_api",
        stripe_price_id="price_api",
        plan_id="api",
        status="active",
    )
    raw_key, _ = generate_api_key(user["id"], "primary")
    response = client.get("/api/safe-state/latest", headers={"X-API-Key": raw_key})
    assert response.status_code == 200
    free_token = _login(client, "freeapi@example.com")
    blocked = client.get("/api/safe-state/latest", headers={"Authorization": f"Bearer {free_token}", "X-API-Key": "nba_live_bad_bad"})
    assert blocked.status_code in {401, 403}


def test_rate_limit_headers(test_env):
    client, _ = test_env
    user = get_or_create_user("ratelimit@example.com")
    upsert_subscription(
        user_id=user["id"],
        stripe_customer_id="cus_rate",
        stripe_subscription_id="sub_rate",
        stripe_price_id="price_api",
        plan_id="api",
        status="active",
    )
    with get_connection() as conn:
        from sports.nba.backend.db.connection import _execute

        _execute(conn, "UPDATE plans SET id='api' WHERE id='api'")
        _execute(conn, "UPDATE entitlements SET api_daily_limit = 1 WHERE user_id = ?", (user["id"],))
    raw_key, _ = generate_api_key(user["id"], "limited")
    first = client.get("/api/model/status", headers={"X-API-Key": raw_key})
    assert first.status_code == 200
    assert "X-RateLimit-Limit" in first.headers
    second = client.get("/api/model/status", headers={"X-API-Key": raw_key})
    assert second.status_code == 429


def test_shadow_only_fields_present(test_env):
    client, _ = test_env
    response = client.get("/api/model/status")
    body = response.json()
    assert body["shadow_only"] is True
    assert body["production_behavior_changed"] is False
    assert body["staking_enabled"] is False
    assert body["auto_bet_enabled"] is False


def test_frontend_pricing_page_strings():
    pricing = (REPO_ROOT / "sports" / "nba" / "web" / "pricing.html").read_text(encoding="utf-8")
    pricing_js = (REPO_ROOT / "sports" / "nba" / "web" / "pricing.js").read_text(encoding="utf-8")
    assert "create-checkout-session" not in pricing
    assert "startCheckout" in pricing_js
    assert "no guaranteed outcomes" in pricing.lower()


def test_safe_state_frontend_lock_copy():
    safe_js = (REPO_ROOT / "sports" / "nba" / "web" / "safe-state.js").read_text(encoding="utf-8")
    assert "can_view_full_safe_state" in safe_js
    assert "SHADOW" in safe_js


def test_account_page_references_plan():
    account_js = (REPO_ROOT / "sports" / "nba" / "web" / "account.js").read_text(encoding="utf-8")
    assert "fetchMe" in account_js
    assert "createApiKey" in account_js


def test_api_docs_page_exists():
    api_html = (REPO_ROOT / "sports" / "nba" / "web" / "api.html").read_text(encoding="utf-8")
    assert "/api/safe-state/cards" in api_html
    assert "shadow_only" in api_html


def test_publish_local_artifacts_copies_and_hashes(tmp_path, monkeypatch):
    import importlib.util

    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    cards = [
        {
            "player": "A",
            "market_type": "PTS_OVER",
            "side": "OVER",
            "line": 1.5,
            "safe_state_tier": "SAFE",
            "edge_defendability_tier": "DEFENSIBLE",
            "recommended_action": "MONITOR",
            "settlement_status": "PENDING",
            "explanation": "shadow",
            "warning_badges": [],
            "shadow_only": True,
        }
    ]
    latest = {
        "run_id": "run_pub",
        "run_date": "2026-05-29",
        "shadow_only": True,
        "promotion_ready": False,
        "production_behavior_changed": False,
        "cards": cards,
    }
    (source / "safe_state_latest.json").write_text(json.dumps(latest), encoding="utf-8")
    (source / "safe_state_latest.csv").write_text("player\nA\n", encoding="utf-8")
    (source / "safe_state_cards.json").write_text(json.dumps(cards), encoding="utf-8")
    (source / "site_manifest.json").write_text(json.dumps({"run_id": "run_pub"}), encoding="utf-8")
    db_path = tmp_path / "pub.db"
    db_url = f"sqlite:///{db_path.as_posix()}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    script_path = (
        REPO_ROOT
        / "sports"
        / "nba"
        / "predictions"
        / "Player-Predictor"
        / "research"
        / "site_export"
        / "publish_local_artifacts.py"
    )
    spec = importlib.util.spec_from_file_location("publish_local_artifacts", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    summary = module.publish(source, target, db_url)
    assert (target / "safe_state_latest.json").exists()
    assert summary["shadow_only"] is True
    with get_connection(db_url) as conn:
        from sports.nba.backend.db.connection import _fetchone

        row = _fetchone(conn, "SELECT COUNT(*) AS count FROM artifact_runs")
        assert row["count"] >= 1


def test_stripe_secret_not_in_frontend():
    for name in ["pricing.html", "account.html", "auth-client.js", "safe-state.html"]:
        text = (REPO_ROOT / "sports" / "nba" / "web" / name).read_text(encoding="utf-8")
        assert "STRIPE_SECRET" not in text
        assert "sk_live" not in text


def test_health_endpoint_no_inference(test_env):
    client, _ = test_env
    body = client.get("/api/health").json()
    assert body["compute"] == "local-only"
