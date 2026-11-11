from sqlalchemy.ext.asyncio import create_async_engine

from app.api.health import check_database


async def test_health_ok_with_database(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "status": "ok",
        "version": body["version"],
        "database": "ok",
        "database_error": None,
    }


async def test_check_database_reports_unreachable_host():
    # Port 9 (discard) is never a Postgres server, so this must fail fast and cleanly.
    engine = create_async_engine("postgresql+asyncpg://u:p@127.0.0.1:9/db")
    try:
        status, error = await check_database(engine)
    finally:
        await engine.dispose()
    assert status == "error"
    assert error
