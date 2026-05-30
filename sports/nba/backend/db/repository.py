"""User, subscription, entitlement, API key, and audit persistence."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import date
from typing import Any

from sports.nba.backend.db.connection import (
    _execute,
    _fetchall,
    _fetchone,
    get_connection,
    json_dumps,
    new_uuid,
    row_to_bool,
    utcnow_iso,
)
from sports.nba.backend.entitlements.entitlement_rules import (
    EntitlementCapabilities,
    entitlements_for_plan,
    entitlements_to_row,
)


def get_or_create_user(email: str) -> dict[str, Any]:
    email_norm = email.strip().lower()
    with get_connection() as conn:
        user = _fetchone(conn, "SELECT * FROM users WHERE email = ?", (email_norm,))
        if user:
            return user
        user_id = new_uuid()
        _execute(
            conn,
            "INSERT INTO users (id, email, created_at, role, status) VALUES (?, ?, ?, 'user', 'active')",
            (user_id, email_norm, utcnow_iso()),
        )
        _sync_entitlements(conn, user_id, "free", "inactive")
        return _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        return _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))


def get_plan(plan_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        return _fetchone(conn, "SELECT * FROM plans WHERE id = ? AND is_active = 1", (plan_id.lower(),))


def list_plans() -> list[dict[str, Any]]:
    with get_connection() as conn:
        return _fetchall(conn, "SELECT * FROM plans WHERE is_active = 1 ORDER BY monthly_price_cents ASC")


def get_subscription_for_user(user_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        return _fetchone(
            conn,
            """
            SELECT * FROM subscriptions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (user_id,),
        )


def upsert_subscription(
    *,
    user_id: str,
    stripe_customer_id: str | None,
    stripe_subscription_id: str | None,
    stripe_price_id: str | None,
    plan_id: str,
    status: str,
    current_period_start: str | None = None,
    current_period_end: str | None = None,
    cancel_at_period_end: bool = False,
) -> dict[str, Any]:
    with get_connection() as conn:
        existing = None
        if stripe_subscription_id:
            existing = _fetchone(
                conn,
                "SELECT * FROM subscriptions WHERE stripe_subscription_id = ?",
                (stripe_subscription_id,),
            )
        if existing:
            _execute(
                conn,
                """
                UPDATE subscriptions SET
                    user_id = ?,
                    stripe_customer_id = ?,
                    stripe_price_id = ?,
                    plan_id = ?,
                    status = ?,
                    current_period_start = ?,
                    current_period_end = ?,
                    cancel_at_period_end = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    user_id,
                    stripe_customer_id,
                    stripe_price_id,
                    plan_id,
                    status,
                    current_period_start,
                    current_period_end,
                    int(cancel_at_period_end),
                    utcnow_iso(),
                    existing["id"],
                ),
            )
            sub_id = existing["id"]
        else:
            sub_id = new_uuid()
            _execute(
                conn,
                """
                INSERT INTO subscriptions (
                    id, user_id, stripe_customer_id, stripe_subscription_id, stripe_price_id,
                    plan_id, status, current_period_start, current_period_end,
                    cancel_at_period_end, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sub_id,
                    user_id,
                    stripe_customer_id,
                    stripe_subscription_id,
                    stripe_price_id,
                    plan_id,
                    status,
                    current_period_start,
                    current_period_end,
                    int(cancel_at_period_end),
                    utcnow_iso(),
                    utcnow_iso(),
                ),
            )
        _sync_entitlements(conn, user_id, plan_id, status)
        return _fetchone(conn, "SELECT * FROM subscriptions WHERE id = ?", (sub_id,))


def _sync_entitlements(conn: Any, user_id: str, plan_id: str, subscription_status: str) -> None:
    caps = entitlements_for_plan(plan_id, subscription_status)
    row = entitlements_to_row(user_id, caps)
    existing = _fetchone(conn, "SELECT id FROM entitlements WHERE user_id = ?", (user_id,))
    if existing:
        _execute(
            conn,
            """
            UPDATE entitlements SET
                plan_id = ?,
                can_view_full_safe_state = ?,
                can_view_candidate_pool = ?,
                can_view_simulation_filters = ?,
                can_export_csv = ?,
                can_use_api = ?,
                max_cards_per_day = ?,
                settlement_delay_hours = ?,
                api_daily_limit = ?,
                updated_at = ?
            WHERE user_id = ?
            """,
            (
                row["plan_id"],
                int(row["can_view_full_safe_state"]),
                int(row["can_view_candidate_pool"]),
                int(row["can_view_simulation_filters"]),
                int(row["can_export_csv"]),
                int(row["can_use_api"]),
                row["max_cards_per_day"],
                row["settlement_delay_hours"],
                row["api_daily_limit"],
                utcnow_iso(),
                user_id,
            ),
        )
        return

    _execute(
        conn,
        """
        INSERT INTO entitlements (
            id, user_id, plan_id, can_view_full_safe_state, can_view_candidate_pool,
            can_view_simulation_filters, can_export_csv, can_use_api, max_cards_per_day,
            settlement_delay_hours, api_daily_limit, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            new_uuid(),
            user_id,
            row["plan_id"],
            int(row["can_view_full_safe_state"]),
            int(row["can_view_candidate_pool"]),
            int(row["can_view_simulation_filters"]),
            int(row["can_export_csv"]),
            int(row["can_use_api"]),
            row["max_cards_per_day"],
            row["settlement_delay_hours"],
            row["api_daily_limit"],
            utcnow_iso(),
        ),
    )


def get_entitlements_for_user(user_id: str) -> dict[str, Any]:
    with get_connection() as conn:
        row = _fetchone(conn, "SELECT * FROM entitlements WHERE user_id = ?", (user_id,))
        if row:
            return _normalize_entitlement_row(row)
        _sync_entitlements(conn, user_id, "free", "inactive")
        row = _fetchone(conn, "SELECT * FROM entitlements WHERE user_id = ?", (user_id,))
        return _normalize_entitlement_row(row or {})


def _normalize_entitlement_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "can_view_full_safe_state": row_to_bool(row.get("can_view_full_safe_state")),
        "can_view_candidate_pool": row_to_bool(row.get("can_view_candidate_pool")),
        "can_view_simulation_filters": row_to_bool(row.get("can_view_simulation_filters")),
        "can_export_csv": row_to_bool(row.get("can_export_csv")),
        "can_use_api": row_to_bool(row.get("can_use_api")),
        "cancel_at_period_end": row_to_bool(row.get("cancel_at_period_end")),
    }


def capabilities_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "plan_id": row.get("plan_id", "free"),
        "can_view_full_safe_state": row_to_bool(row.get("can_view_full_safe_state")),
        "can_view_candidate_pool": row_to_bool(row.get("can_view_candidate_pool")),
        "can_view_simulation_filters": row_to_bool(row.get("can_view_simulation_filters")),
        "can_export_csv": row_to_bool(row.get("can_export_csv")),
        "can_use_api": row_to_bool(row.get("can_use_api")),
        "max_cards_per_day": row.get("max_cards_per_day"),
        "settlement_delay_hours": int(row.get("settlement_delay_hours") or 24),
        "api_daily_limit": int(row.get("api_daily_limit") or 0),
    }


def record_audit_event(user_id: str | None, event_type: str, metadata: dict[str, Any] | None = None) -> None:
    with get_connection() as conn:
        _execute(
            conn,
            "INSERT INTO audit_events (id, user_id, event_type, metadata, created_at) VALUES (?, ?, ?, ?, ?)",
            (new_uuid(), user_id, event_type, json_dumps(metadata or {}), utcnow_iso()),
        )


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key(user_id: str, name: str | None = None) -> tuple[str, dict[str, Any]]:
    prefix = secrets.token_hex(4)
    secret = secrets.token_urlsafe(24)
    raw_key = f"nba_live_{prefix}_{secret}"
    key_hash = hash_api_key(raw_key)
    key_id = new_uuid()
    with get_connection() as conn:
        _execute(
            conn,
            """
            INSERT INTO api_keys (id, user_id, key_hash, key_prefix, name, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'active', ?)
            """,
            (key_id, user_id, key_hash, prefix, name or "Default", utcnow_iso()),
        )
        row = _fetchone(conn, "SELECT * FROM api_keys WHERE id = ?", (key_id,))
    return raw_key, row or {}


def list_api_keys(user_id: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        return _fetchall(
            conn,
            """
            SELECT id, user_id, key_prefix, name, status, created_at, last_used_at, revoked_at
            FROM api_keys
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        )


def revoke_api_key(user_id: str, key_id: str) -> bool:
    with get_connection() as conn:
        row = _fetchone(conn, "SELECT * FROM api_keys WHERE id = ? AND user_id = ?", (key_id, user_id))
        if not row:
            return False
        _execute(
            conn,
            "UPDATE api_keys SET status = 'revoked', revoked_at = ? WHERE id = ?",
            (utcnow_iso(), key_id),
        )
        return True


def lookup_api_key(raw_key: str) -> dict[str, Any] | None:
    key_hash = hash_api_key(raw_key)
    with get_connection() as conn:
        row = _fetchone(
            conn,
            "SELECT * FROM api_keys WHERE key_hash = ? AND status = 'active'",
            (key_hash,),
        )
        if not row:
            return None
        _execute(conn, "UPDATE api_keys SET last_used_at = ? WHERE id = ?", (utcnow_iso(), row["id"]))
        return row


def get_usage_today(user_id: str, api_key_id: str | None = None) -> int:
    usage_date = date.today().isoformat()
    with get_connection() as conn:
        if api_key_id:
            row = _fetchone(
                conn,
                """
                SELECT COALESCE(SUM(request_count), 0) AS total
                FROM api_usage
                WHERE user_id = ? AND api_key_id = ? AND usage_date = ?
                """,
                (user_id, api_key_id, usage_date),
            )
        else:
            row = _fetchone(
                conn,
                """
                SELECT COALESCE(SUM(request_count), 0) AS total
                FROM api_usage
                WHERE user_id = ? AND usage_date = ?
                """,
                (user_id, usage_date),
            )
        return int(row["total"]) if row else 0


def increment_api_usage(user_id: str, api_key_id: str | None, endpoint: str) -> int:
    usage_date = date.today().isoformat()
    with get_connection() as conn:
        existing = _fetchone(
            conn,
            """
            SELECT id, request_count FROM api_usage
            WHERE user_id = ? AND api_key_id IS ? AND endpoint = ? AND usage_date = ?
            """,
            (user_id, api_key_id, endpoint, usage_date),
        )
        if existing:
            new_count = int(existing["request_count"]) + 1
            _execute(conn, "UPDATE api_usage SET request_count = ? WHERE id = ?", (new_count, existing["id"]))
            return new_count
        usage_id = new_uuid()
        _execute(
            conn,
            """
            INSERT INTO api_usage (id, user_id, api_key_id, endpoint, request_count, usage_date, created_at)
            VALUES (?, ?, ?, ?, 1, ?, ?)
            """,
            (usage_id, user_id, api_key_id, endpoint, usage_date, utcnow_iso()),
        )
        return 1


def insert_artifact_run(payload: dict[str, Any]) -> dict[str, Any]:
    run_row_id = new_uuid()
    with get_connection() as conn:
        _execute(
            conn,
            """
            INSERT INTO artifact_runs (
                id, run_id, run_date, sport, artifact_type, artifact_path, artifact_hash,
                card_count, simulation_count, shadow_only, promotion_ready,
                production_behavior_changed, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_row_id,
                payload["run_id"],
                payload.get("run_date"),
                payload.get("sport", "nba"),
                payload["artifact_type"],
                payload["artifact_path"],
                payload["artifact_hash"],
                int(payload.get("card_count") or 0),
                int(payload.get("simulation_count") or 0),
                int(payload.get("shadow_only", True)),
                int(payload.get("promotion_ready", False)),
                int(payload.get("production_behavior_changed", False)),
                utcnow_iso(),
            ),
        )
        return _fetchone(conn, "SELECT * FROM artifact_runs WHERE id = ?", (run_row_id,)) or {}
