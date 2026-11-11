"""Pydantic v2 request/response schemas for the API."""

from datetime import datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

Backend = Literal["llama_cpp", "vllm", "openai_compat"]


class ModelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200, examples=["Qwen2.5-7B-Instruct"])
    hf_repo: str | None = Field(default=None, max_length=200, examples=["Qwen/Qwen2.5-7B-Instruct"])
    backend: Backend
    # Stored as a plain string (HttpUrl would re-append a trailing slash); validated below.
    endpoint_url: str = Field(max_length=500, examples=["http://host.docker.internal:8080"])
    quantization: str | None = Field(default=None, max_length=50, examples=["Q4_K_M"])
    params_b: float | None = Field(default=None, gt=0)
    notes: str | None = None

    @field_validator("endpoint_url")
    @classmethod
    def validate_and_normalise_url(cls, v: str) -> str:
        parsed = urlparse(v.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("endpoint_url must be an absolute http(s) URL")
        return v.strip().rstrip("/")


class ModelRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    hf_repo: str | None
    backend: str
    endpoint_url: str
    quantization: str | None
    params_b: float | None
    notes: str | None
    created_at: datetime


class DatasetRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source_url: str
    license: str
    language: str
    n_items: int
    split: str
    seed: int
    config_json: dict
    # True once the fixed subset has been materialised and its item ids stored.
    materialised: bool = False
