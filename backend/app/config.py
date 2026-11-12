from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "LLM Eval Platform"
    version: str = "0.1.0"
    database_url: str = "postgresql+asyncpg://llmeval:llmeval@localhost:5432/llmeval"
    # Comma-separated origins allowed to call the API from a browser.
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Upsert the static dataset registry into the DB when the API starts.
    sync_datasets_on_startup: bool = True
    # Where huggingface_hub caches downloaded dataset files (never committed).
    hf_cache_dir: str | None = None
    # Model-server request limits used by the run executor.
    request_timeout_s: float = 120.0
    default_concurrency: int = 2
    default_max_tokens: int = 16

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
