from sqlalchemy.ext.asyncio import create_async_engine

from app.api.health import check_database


async def test_health_endpoint_reports_shape(client):
    """The endpoint always answers 200 and reports database state honestly.

    Without a reachable Postgres the status is 'degraded' with an error string;
    with one it is 'ok'. Either way the shape is stable for the dashboard.
    """
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"status", "version", "database", "database_error"}
    if body["database"] == "ok":
        assert body["status"] == "ok" and body["database_error"] is None
    else:
        assert body["status"] == "degraded" and body["database_error"]


async def test_check_database_reports_unreachable_host():
    # Port 9 (discard) is never a Postgres server, so this must fail fast and cleanly.
    engine = create_async_engine("postgresql+asyncpg://u:p@127.0.0.1:9/db")
    try:
        status, error = await check_database(engine)
    finally:
        await engine.dispose()
    assert status == "error"
    assert error
