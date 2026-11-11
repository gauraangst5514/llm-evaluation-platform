"""Test fixtures.

Tests run against a real PostgreSQL (the same engine as production), in a separate
database so they can drop and recreate the schema freely. The schema is built by the
Alembic migrations, not by create_all, so migrations are exercised on every test run.

Set TEST_DATABASE_URL to point elsewhere; the default matches docker-compose.
"""

import asyncio
import os

import asyncpg
import pytest
from alembic.config import Config
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command
from app.config import get_settings
from app.db.models import Base

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+asyncpg://llmeval:llmeval@localhost:5432/llmeval_test"
)


def _asyncpg_dsn(url: str, dbname: str | None = None) -> str:
    dsn = url.replace("postgresql+asyncpg://", "postgresql://")
    if dbname is not None:
        dsn = dsn.rsplit("/", 1)[0] + "/" + dbname
    return dsn


async def _ensure_test_database() -> None:
    """Create the test database if it does not exist yet."""
    name = TEST_DATABASE_URL.rsplit("/", 1)[1]
    conn = await asyncpg.connect(_asyncpg_dsn(TEST_DATABASE_URL, "postgres"))
    try:
        exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


@pytest.fixture(scope="session", autouse=True)
def test_settings():
    """Point the app at the test database for the whole session."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["SYNC_DATASETS_ON_STARTUP"] = "false"
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def alembic_config(test_settings) -> Config:
    cfg = Config("alembic.ini")
    cfg.set_main_option("script_location", "alembic")
    return cfg


@pytest.fixture(scope="session", autouse=True)
def migrated_database(alembic_config):
    """Fresh schema from migrations once per session (sync fixture: pytest-asyncio
    loops are function-scoped, so the async work runs in its own short-lived loop)."""

    async def _reset():
        await _ensure_test_database()
        engine = create_async_engine(TEST_DATABASE_URL)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            await conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
        await engine.dispose()

    asyncio.run(_reset())
    command.upgrade(alembic_config, "head")
    yield


@pytest.fixture(autouse=True)
async def clean_tables():
    """Empty every table between tests so they stay independent."""
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        names = ", ".join(t.name for t in Base.metadata.sorted_tables)
        await conn.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
    await engine.dispose()
    yield


@pytest.fixture
async def client():
    """HTTP client bound to the ASGI app, with the lifespan (engine setup) run."""
    from app.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            c.app = app  # type: ignore[attr-defined]
            yield c
