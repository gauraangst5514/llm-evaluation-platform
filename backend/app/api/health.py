from fastapi import APIRouter, Request
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    database_error: str | None = None


async def check_database(engine: AsyncEngine) -> tuple[str, str | None]:
    """Return ("ok", None) if the database answers SELECT 1, else ("error", reason)."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return "ok", None
    except Exception as exc:  # noqa: BLE001 - we want to report any failure, not crash
        return "error", f"{type(exc).__name__}: {exc}"


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    settings = get_settings()
    engine: AsyncEngine = request.app.state.engine
    db_status, db_error = await check_database(engine)
    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        version=settings.version,
        database=db_status,
        database_error=db_error,
    )
