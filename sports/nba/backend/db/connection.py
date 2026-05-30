"""Database connection and lightweight query helpers."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Generator, Iterable
from urllib.parse import urlparse

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS_DIR = BACKEND_ROOT / "db" / "migrations"


def _is_sqlite_url(database_url: str) -> bool:
    return database_url.startswith("sqlite:")


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///sports/nba/backend/data/nba_analytics.db")


@contextmanager
def get_connection(database_url: str | None = None) -> Generator[Any, None, None]:
    url = database_url or get_database_url()
    if _is_sqlite_url(url):
        path = url.replace("sqlite:///", "", 1)
        db_path = Path(path)
        if not db_path.is_absolute():
            repo_root = BACKEND_ROOT.parents[2]
            db_path = repo_root / path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return

    import psycopg2
    import psycopg2.extras

    parsed = urlparse(url)
    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        dbname=parsed.path.lstrip("/"),
    )
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _adapt_sql(conn: Any, sql: str) -> str:
    if isinstance(conn, sqlite3.Connection):
        return sql
    return sql.replace("?", "%s")


def _fetchall(conn: Any, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    sql = _adapt_sql(conn, sql)
    if isinstance(conn, sqlite3.Connection):
        cur = conn.execute(sql, tuple(params))
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    import psycopg2.extras

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, tuple(params))
        return [dict(row) for row in cur.fetchall()]


def _fetchone(conn: Any, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    rows = _fetchall(conn, sql, params)
    return rows[0] if rows else None


def _execute(conn: Any, sql: str, params: Iterable[Any] = ()) -> None:
    sql = _adapt_sql(conn, sql)
    if isinstance(conn, sqlite3.Connection):
        conn.execute(sql, tuple(params))
        return
    with conn.cursor() as cur:
        cur.execute(sql, tuple(params))


def new_uuid() -> str:
    return str(uuid.uuid4())


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            role TEXT NOT NULL DEFAULT 'user',
            status TEXT NOT NULL DEFAULT 'active'
        );
        CREATE TABLE IF NOT EXISTS plans (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            stripe_price_id TEXT,
            monthly_price_cents INTEGER NOT NULL DEFAULT 0,
            tier TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS subscriptions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT UNIQUE,
            stripe_price_id TEXT,
            plan_id TEXT REFERENCES plans(id),
            status TEXT NOT NULL DEFAULT 'inactive',
            current_period_start TEXT,
            current_period_end TEXT,
            cancel_at_period_end INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS entitlements (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
            plan_id TEXT NOT NULL REFERENCES plans(id),
            can_view_full_safe_state INTEGER NOT NULL DEFAULT 0,
            can_view_candidate_pool INTEGER NOT NULL DEFAULT 0,
            can_view_simulation_filters INTEGER NOT NULL DEFAULT 0,
            can_export_csv INTEGER NOT NULL DEFAULT 0,
            can_use_api INTEGER NOT NULL DEFAULT 0,
            max_cards_per_day INTEGER,
            settlement_delay_hours INTEGER NOT NULL DEFAULT 24,
            api_daily_limit INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            key_hash TEXT UNIQUE NOT NULL,
            key_prefix TEXT NOT NULL,
            name TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_used_at TEXT,
            revoked_at TEXT
        );
        CREATE TABLE IF NOT EXISTS api_usage (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            api_key_id TEXT REFERENCES api_keys(id) ON DELETE SET NULL,
            endpoint TEXT NOT NULL,
            request_count INTEGER NOT NULL DEFAULT 1,
            usage_date TEXT NOT NULL DEFAULT (date('now')),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (user_id, api_key_id, endpoint, usage_date)
        );
        CREATE TABLE IF NOT EXISTS artifact_runs (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            run_date TEXT,
            sport TEXT NOT NULL DEFAULT 'nba',
            artifact_type TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            artifact_hash TEXT NOT NULL,
            card_count INTEGER NOT NULL DEFAULT 0,
            simulation_count INTEGER NOT NULL DEFAULT 0,
            shadow_only INTEGER NOT NULL DEFAULT 1,
            promotion_ready INTEGER NOT NULL DEFAULT 0,
            production_behavior_changed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE (run_id, artifact_type)
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            id TEXT PRIMARY KEY,
            user_id TEXT REFERENCES users(id) ON DELETE SET NULL,
            event_type TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    seed_sqlite_plans(conn)


def seed_sqlite_plans(conn: sqlite3.Connection) -> None:
    plans = [
        ("free", "Free Research", None, 0, "free", 1),
        ("plus", "Plus Analytics", None, 1900, "plus", 1),
        ("pro", "Pro Research", None, 4900, "pro", 1),
        ("api", "API Access", None, 9900, "api", 1),
    ]
    for row in plans:
        conn.execute(
            """
            INSERT INTO plans (id, name, stripe_price_id, monthly_price_cents, tier, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                monthly_price_cents=excluded.monthly_price_cents,
                tier=excluded.tier,
                is_active=excluded.is_active
            """,
            row,
        )


def run_migrations(database_url: str | None = None) -> None:
    url = database_url or get_database_url()
    if _is_sqlite_url(url):
        with get_connection(url) as conn:
            ensure_sqlite_schema(conn)
        return

    migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    with get_connection(url) as conn:
        for path in migration_files:
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)


def row_to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return bool(int(value or 0))
