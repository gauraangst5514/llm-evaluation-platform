import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import datasets, health, models
from app.config import get_settings
from app.datasets.registry import sync_registry
from app.db.session import make_engine, make_session_factory

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.engine = make_engine(settings.database_url)
    app.state.session_factory = make_session_factory(app.state.engine)
    if settings.sync_datasets_on_startup:
        try:
            async with app.state.session_factory() as session:
                inserted = await sync_registry(session)
            log.info("Dataset registry synced (%d new).", inserted)
        except Exception as exc:  # noqa: BLE001 - startup must not die if the DB is late
            log.warning("Dataset registry sync skipped: %s", exc)
    try:
        yield
    finally:
        await app.state.engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(datasets.router)
    return app


app = create_app()
