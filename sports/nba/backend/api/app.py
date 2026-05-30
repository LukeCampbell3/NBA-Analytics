"""FastAPI application — auth, billing, entitlements, data API. No model inference."""
from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field

from sports.nba.backend.auth.jwt_auth import create_access_token, decode_access_token
from sports.nba.backend.billing import stripe_client
from sports.nba.backend.db.connection import get_database_url, run_migrations
from sports.nba.backend.db.repository import (
    capabilities_dict,
    generate_api_key,
    get_entitlements_for_user,
    get_or_create_user,
    get_plan,
    get_subscription_for_user,
    get_usage_today,
    get_user_by_id,
    increment_api_usage,
    insert_artifact_run,
    list_api_keys,
    list_plans,
    lookup_api_key,
    record_audit_event,
    revoke_api_key,
    upsert_subscription,
)
from sports.nba.backend.entitlements.entitlement_rules import ACTIVE_SUBSCRIPTION_STATUSES
from sports.nba.backend.services import artifacts

app = FastAPI(title="NBA Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CheckoutSessionRequest(BaseModel):
    plan_id: str
    success_url: str
    cancel_url: str


class PortalSessionRequest(BaseModel):
    return_url: str


class DevSessionRequest(BaseModel):
    email: EmailStr


class ApiKeyCreateRequest(BaseModel):
    name: str | None = None


class AuthContext(BaseModel):
    user_id: str
    email: str
    auth_type: str
    api_key_id: str | None = None


def _site_url() -> str:
    return os.environ.get("SITE_URL", "http://localhost:8000").rstrip("/")


def _require_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> AuthContext | None:
    if x_api_key:
        row = lookup_api_key(x_api_key.strip())
        if not row:
            raise HTTPException(status_code=401, detail="Invalid API key")
        user = get_user_by_id(row["user_id"])
        if not user:
            raise HTTPException(status_code=401, detail="Invalid API key user")
        return AuthContext(
            user_id=user["id"],
            email=user["email"],
            auth_type="api_key",
            api_key_id=row["id"],
        )
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
        try:
            payload = decode_access_token(token)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
        return AuthContext(
            user_id=str(payload["sub"]),
            email=str(payload.get("email") or ""),
            auth_type="session",
        )
    return None


def _require_authenticated_user(ctx: AuthContext | None = Depends(_require_user)) -> AuthContext:
    if not ctx:
        raise HTTPException(status_code=401, detail="Authentication required")
    return ctx


def _entitlements_for_auth(ctx: AuthContext | None) -> dict[str, Any]:
    if not ctx:
        return capabilities_dict(get_entitlements_for_user(get_or_create_user("anonymous@preview.local")["id"]))
    return capabilities_dict(get_entitlements_for_user(ctx.user_id))


def _apply_rate_limit(response: Response, ctx: AuthContext, endpoint: str) -> None:
    entitlements = get_entitlements_for_user(ctx.user_id)
    caps = capabilities_dict(entitlements)
    if not caps["can_use_api"]:
        raise HTTPException(status_code=403, detail="API access not included in current plan")
    limit = int(caps["api_daily_limit"] or 0)
    if limit <= 0:
        raise HTTPException(status_code=403, detail="API daily limit not configured")
    used_before = get_usage_today(ctx.user_id, ctx.api_key_id)
    if used_before >= limit:
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = "0"
        response.headers["X-RateLimit-Reset"] = "midnight UTC"
        raise HTTPException(status_code=429, detail="API daily limit exceeded")
    used_after = increment_api_usage(ctx.user_id, ctx.api_key_id, endpoint)
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, limit - used_after))
    response.headers["X-RateLimit-Reset"] = "midnight UTC"


def _plan_id_from_stripe_price(stripe_price_id: str | None) -> str:
    if not stripe_price_id:
        return "free"
    for plan in list_plans():
        if plan.get("stripe_price_id") == stripe_price_id:
            return str(plan["id"])
    return "free"


@app.on_event("startup")
def startup() -> None:
    if os.environ.get("SKIP_DB_MIGRATIONS") != "1":
        run_migrations()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "compute": "local-only", "inference": "disabled"}


@app.post("/api/auth/dev-session")
def create_dev_session(body: DevSessionRequest) -> dict[str, Any]:
    if os.environ.get("ALLOW_DEV_AUTH") != "1":
        raise HTTPException(status_code=403, detail="Dev auth disabled")
    user = get_or_create_user(body.email)
    token = create_access_token(user_id=user["id"], email=user["email"])
    record_audit_event(user["id"], "dev_session_created", {"email": user["email"]})
    return {"access_token": token, "token_type": "bearer", "user": {"id": user["id"], "email": user["email"]}}


@app.get("/api/me")
def get_me(ctx: AuthContext = Depends(_require_authenticated_user)) -> dict[str, Any]:
    user = get_user_by_id(ctx.user_id)
    subscription = get_subscription_for_user(ctx.user_id)
    entitlements = get_entitlements_for_user(ctx.user_id)
    usage_today = get_usage_today(ctx.user_id)
    return {
        "user": user,
        "subscription": subscription,
        "entitlements": capabilities_dict(entitlements),
        "usage_today": usage_today,
    }


@app.get("/api/entitlements")
def get_entitlements(ctx: AuthContext | None = Depends(_require_user)) -> dict[str, Any]:
    if ctx:
        entitlements = get_entitlements_for_user(ctx.user_id)
        caps = capabilities_dict(entitlements)
        return {
            "plan": caps["plan_id"],
            "capabilities": caps,
            "usage_today": get_usage_today(ctx.user_id, ctx.api_key_id),
            "api_limit": caps["api_daily_limit"],
        }
    free_user = get_or_create_user("anonymous@preview.local")
    caps = capabilities_dict(get_entitlements_for_user(free_user["id"]))
    return {"plan": "free", "capabilities": caps, "usage_today": 0, "api_limit": 0}


@app.get("/api/plans")
def get_plans() -> dict[str, Any]:
    return {"plans": list_plans()}


@app.post("/api/billing/create-checkout-session")
def create_checkout_session_route(
    body: CheckoutSessionRequest,
    ctx: AuthContext = Depends(_require_authenticated_user),
) -> dict[str, Any]:
    if not stripe_client.stripe_enabled():
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    plan = get_plan(body.plan_id)
    if not plan or plan["id"] == "free":
        raise HTTPException(status_code=400, detail="Invalid plan_id")
    stripe_price_id = plan.get("stripe_price_id")
    if not stripe_price_id:
        raise HTTPException(status_code=400, detail="Plan is not billable yet")
    subscription = get_subscription_for_user(ctx.user_id)
    session = stripe_client.create_checkout_session(
        customer_id=subscription.get("stripe_customer_id") if subscription else None,
        customer_email=ctx.email,
        stripe_price_id=stripe_price_id,
        success_url=body.success_url,
        cancel_url=body.cancel_url,
        metadata={"user_id": ctx.user_id, "plan_id": plan["id"]},
    )
    record_audit_event(ctx.user_id, "checkout_session_created", {"plan_id": plan["id"], "session_id": session["id"]})
    return {"checkout_url": session["url"]}


@app.post("/api/billing/create-portal-session")
def create_portal_session_route(
    body: PortalSessionRequest,
    ctx: AuthContext = Depends(_require_authenticated_user),
) -> dict[str, Any]:
    if not stripe_client.stripe_enabled():
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    subscription = get_subscription_for_user(ctx.user_id)
    customer_id = subscription.get("stripe_customer_id") if subscription else None
    if not customer_id:
        raise HTTPException(status_code=400, detail="No Stripe customer on file")
    session = stripe_client.create_portal_session(customer_id=customer_id, return_url=body.return_url)
    record_audit_event(ctx.user_id, "portal_session_created", {"customer_id": customer_id})
    return {"portal_url": session["url"]}


@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request) -> dict[str, str]:
    if not stripe_client.stripe_enabled():
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    payload = await request.body()
    signature = request.headers.get("stripe-signature", "")
    try:
        event = stripe_client.construct_webhook_event(payload, signature)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid Stripe signature") from exc

    event_type = event["type"]
    data_object = event["data"]["object"]

    if event_type == "checkout.session.completed":
        metadata = data_object.get("metadata") or {}
        user_id = metadata.get("user_id")
        plan_id = metadata.get("plan_id") or "free"
        if user_id:
            upsert_subscription(
                user_id=user_id,
                stripe_customer_id=data_object.get("customer"),
                stripe_subscription_id=data_object.get("subscription"),
                stripe_price_id=None,
                plan_id=plan_id,
                status="active",
            )
            record_audit_event(user_id, event_type, {"session_id": data_object.get("id")})

    if event_type in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
        "invoice.payment_succeeded",
        "invoice.payment_failed",
    }:
        subscription_obj = data_object
        if event_type.startswith("invoice."):
            subscription_obj = {"id": data_object.get("subscription"), "customer": data_object.get("customer")}
        stripe_subscription_id = subscription_obj.get("id")
        stripe_customer_id = subscription_obj.get("customer")
        stripe_price_id = None
        items = subscription_obj.get("items", {}).get("data", []) if isinstance(subscription_obj.get("items"), dict) else []
        if items:
            stripe_price_id = items[0].get("price", {}).get("id")
        status = subscription_obj.get("status") or ("canceled" if event_type.endswith(".deleted") else "inactive")
        plan_id = _plan_id_from_stripe_price(stripe_price_id)
        user_id = (subscription_obj.get("metadata") or {}).get("user_id")
        if not user_id and stripe_customer_id:
            # Best-effort lookup via existing subscription row
            from sports.nba.backend.db.connection import get_connection, _fetchone

            with get_connection() as conn:
                row = _fetchone(conn, "SELECT user_id FROM subscriptions WHERE stripe_customer_id = ?", (stripe_customer_id,))
                user_id = row["user_id"] if row else None
        if user_id:
            effective_status = status if status in ACTIVE_SUBSCRIPTION_STATUSES else status
            upsert_subscription(
                user_id=user_id,
                stripe_customer_id=stripe_customer_id,
                stripe_subscription_id=stripe_subscription_id,
                stripe_price_id=stripe_price_id,
                plan_id=plan_id if status in ACTIVE_SUBSCRIPTION_STATUSES else "free",
                status=effective_status,
                current_period_start=_stripe_ts(subscription_obj.get("current_period_start")),
                current_period_end=_stripe_ts(subscription_obj.get("current_period_end")),
                cancel_at_period_end=bool(subscription_obj.get("cancel_at_period_end")),
            )
            record_audit_event(user_id, event_type, {"subscription_id": stripe_subscription_id, "status": status})

    return {"received": "true"}


def _stripe_ts(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


@app.post("/api/api-keys")
def create_api_key_route(
    body: ApiKeyCreateRequest,
    ctx: AuthContext = Depends(_require_authenticated_user),
) -> dict[str, Any]:
    entitlements = capabilities_dict(get_entitlements_for_user(ctx.user_id))
    if not entitlements["can_use_api"]:
        raise HTTPException(status_code=403, detail="API plan required to create keys")
    raw_key, row = generate_api_key(ctx.user_id, body.name)
    record_audit_event(ctx.user_id, "api_key_created", {"key_prefix": row.get("key_prefix")})
    return {
        "api_key": raw_key,
        "key_prefix": row.get("key_prefix"),
        "name": row.get("name"),
        "id": row.get("id"),
        "message": "Store this key now. It will not be shown again.",
    }


@app.get("/api/api-keys")
def list_api_keys_route(ctx: AuthContext = Depends(_require_authenticated_user)) -> dict[str, Any]:
    return {"keys": list_api_keys(ctx.user_id)}


@app.delete("/api/api-keys/{key_id}")
def delete_api_key_route(key_id: str, ctx: AuthContext = Depends(_require_authenticated_user)) -> dict[str, str]:
    if not revoke_api_key(ctx.user_id, key_id):
        raise HTTPException(status_code=404, detail="API key not found")
    record_audit_event(ctx.user_id, "api_key_revoked", {"key_id": key_id})
    return {"status": "revoked"}


def _data_access_guard(ctx: AuthContext | None, response: Response, endpoint: str, require_api: bool = False) -> dict[str, Any]:
    if require_api:
        if not ctx or ctx.auth_type != "api_key":
            raise HTTPException(status_code=403, detail="API key required")
        _apply_rate_limit(response, ctx, endpoint)
        return capabilities_dict(get_entitlements_for_user(ctx.user_id))
    caps = _entitlements_for_auth(ctx)
    if ctx and ctx.auth_type == "api_key":
        _apply_rate_limit(response, ctx, endpoint)
    return caps


@app.get("/api/safe-state/latest")
def safe_state_latest(
    response: Response,
    ctx: AuthContext | None = Depends(_require_user),
) -> dict[str, Any]:
    caps = _data_access_guard(ctx, response, "/api/safe-state/latest")
    latest = artifacts.load_safe_state_latest()
    cards = artifacts.filter_cards_for_entitlements(
        artifacts.load_safe_state_cards(),
        max_cards_per_day=caps["max_cards_per_day"],
        can_view_candidate_pool=caps["can_view_candidate_pool"],
    )
    meta = artifacts.response_meta()
    return {**meta, "cards_preview_count": len(cards), "latest": latest, "locked": not caps["can_view_full_safe_state"]}


@app.get("/api/safe-state/cards")
def safe_state_cards(
    response: Response,
    ctx: AuthContext | None = Depends(_require_user),
) -> dict[str, Any]:
    caps = _data_access_guard(ctx, response, "/api/safe-state/cards")
    all_cards = artifacts.load_safe_state_cards()
    cards = all_cards if caps["can_view_full_safe_state"] else all_cards[: caps["max_cards_per_day"] or 3]
    cards = artifacts.filter_cards_for_entitlements(
        cards,
        max_cards_per_day=caps["max_cards_per_day"] if not caps["can_view_full_safe_state"] else None,
        can_view_candidate_pool=caps["can_view_candidate_pool"],
    )
    meta = artifacts.response_meta()
    payload: dict[str, Any] = {**meta, "cards": cards, "total_available": len(all_cards), "plan": caps["plan_id"]}
    if caps["can_export_csv"]:
        payload["csv_available"] = True
    return payload


@app.get("/api/players/simulations")
def player_simulations(
    response: Response,
    ctx: AuthContext | None = Depends(_require_user),
) -> dict[str, Any]:
    caps = _data_access_guard(ctx, response, "/api/players/simulations")
    all_cards = artifacts.load_simulation_cards()
    cards = artifacts.filter_simulations_for_entitlements(
        all_cards,
        max_cards_per_day=caps["max_cards_per_day"],
        can_view_simulation_filters=caps["can_view_simulation_filters"],
    )
    meta = artifacts.response_meta()
    return {
        **meta,
        "cards": cards,
        "total_available": len(all_cards),
        "research_label": "research projection / uncalibrated",
        "plan": caps["plan_id"],
    }


@app.get("/api/players/{player_id}/simulation")
def player_simulation_detail(
    player_id: str,
    response: Response,
    ctx: AuthContext | None = Depends(_require_user),
) -> dict[str, Any]:
    caps = _data_access_guard(ctx, response, f"/api/players/{player_id}/simulation")
    cards = artifacts.filter_simulations_for_entitlements(
        artifacts.load_simulation_cards(),
        max_cards_per_day=caps["max_cards_per_day"],
        can_view_simulation_filters=caps["can_view_simulation_filters"],
        preview_limit=1,
    )
    match = next(
        (
            card
            for card in cards
            if str(card.get("player_id") or card.get("id") or card.get("player") or "").lower() == player_id.lower()
        ),
        None,
    )
    if not match:
        raise HTTPException(status_code=404, detail="Simulation card not found or not entitled")
    return {**artifacts.response_meta(), "card": match}


@app.get("/api/settlement/history")
def settlement_history(
    response: Response,
    ctx: AuthContext | None = Depends(_require_user),
) -> dict[str, Any]:
    caps = _data_access_guard(ctx, response, "/api/settlement/history")
    rows = artifacts.settlement_history_rows()
    if caps["settlement_delay_hours"] > 0:
        rows = rows[: caps["max_cards_per_day"] or 3]
    return {**artifacts.response_meta(), "rows": rows, "settlement_delay_hours": caps["settlement_delay_hours"]}


@app.get("/api/model/status")
def model_status(response: Response, ctx: AuthContext | None = Depends(_require_user)) -> dict[str, Any]:
    _data_access_guard(ctx, response, "/api/model/status")
    return artifacts.load_model_status()


@app.get("/api/safe-state/export.csv")
def safe_state_export_csv(
    response: Response,
    ctx: AuthContext = Depends(_require_authenticated_user),
) -> Response:
    caps = capabilities_dict(get_entitlements_for_user(ctx.user_id))
    if not caps["can_export_csv"]:
        raise HTTPException(status_code=403, detail="CSV export requires Pro or API plan")
    cards = artifacts.filter_cards_for_entitlements(
        artifacts.load_safe_state_cards(),
        max_cards_per_day=None,
        can_view_candidate_pool=caps["can_view_candidate_pool"],
    )
    csv_text = artifacts.safe_state_csv(cards)
    return Response(content=csv_text, media_type="text/csv")


def create_app() -> FastAPI:
    return app
