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
    resource_agent_url: str | None = Field(
        default=None, max_length=500, examples=["http://host.docker.internal:9108"]
    )

    @field_validator("endpoint_url")
    @classmethod
    def validate_and_normalise_url(cls, v: str) -> str:
        parsed = urlparse(v.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("endpoint_url must be an absolute http(s) URL")
        return v.strip().rstrip("/")


class ModelUpdate(BaseModel):
    """Partial update; only provided fields change."""

    hf_repo: str | None = Field(default=None, max_length=200)
    endpoint_url: str | None = Field(default=None, max_length=500)
    params_b: float | None = Field(default=None, gt=0)
    notes: str | None = None
    resource_agent_url: str | None = Field(default=None, max_length=500)

    _validate_url = field_validator("endpoint_url")(
        lambda cls, v: None if v is None else ModelCreate.validate_and_normalise_url(v)
    )


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
    resource_agent_url: str | None
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


Dimension = Literal["accuracy", "latency", "resources", "tokenizer", "reliability"]


class RunCreate(BaseModel):
    model_id: int
    dataset_id: int
    dimensions: list[Dimension] = Field(
        default=["accuracy", "latency", "resources", "tokenizer", "reliability"]
    )
    n_items: int = Field(default=50, ge=1, le=5000)
    seed: int = 42
    concurrency: int = Field(default=2, ge=1, le=64)
    max_tokens: int = Field(default=16, ge=1, le=512)
    timeout_s: float | None = Field(default=None, gt=0)


class RunProgress(BaseModel):
    n_total: int
    n_done: int
    n_correct: int
    n_scored: int  # items with a response (errors excluded)
    n_errors: int
    n_unparseable: int


class MetricRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dimension: str
    name: str
    value: float | None
    unit: str | None
    language: str | None
    extra_json: dict | None


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    model_id: int
    dataset_id: int
    status: str
    config_json: dict
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: str | None


class RunSummary(RunRead):
    model_name: str
    quantization: str | None
    dataset_name: str
    language: str
    progress: RunProgress
    # Share of scored items answered correctly; None until something is scored.
    accuracy: float | None


class RunDetail(RunSummary):
    metrics: list[MetricRead]


class ResultItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    item_id: str
    language: str | None
    subject: str | None
    prompt: str
    response: str | None
    parsed_answer: str | None
    correct: bool | None
    ttft_ms: float | None
    latency_ms: float | None
    output_tokens: int | None
    error: str | None
    extra_json: dict | None


class CompareCell(BaseModel):
    run_id: int
    value: float | None
    extra_json: dict | None


class CompareRow(BaseModel):
    dimension: str
    name: str
    unit: str | None
    language: str | None
    cells: list[CompareCell]


class CompareResponse(BaseModel):
    runs: list[RunSummary]
    rows: list[CompareRow]


class Page(BaseModel):
    items: list[ResultItemRead]
    total: int
    page: int
    page_size: int
