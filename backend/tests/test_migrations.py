"""The migration chain must produce exactly the schema the ORM models describe."""

import asyncio

from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.models import Base
from tests.conftest import TEST_DATABASE_URL


def test_models_match_migrations():
    async def _diff():
        engine = create_async_engine(TEST_DATABASE_URL)
        try:
            async with engine.connect() as conn:
                return await conn.run_sync(
                    lambda c: compare_metadata(MigrationContext.configure(c), Base.metadata)
                )
        finally:
            await engine.dispose()

    diff = asyncio.run(_diff())
    assert diff == [], f"ORM models and migrations have drifted: {diff}"
