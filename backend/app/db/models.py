"""SQLAlchemy 2 ORM models. Schema changes go through Alembic migrations."""

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Model(Base):
    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("name", "quantization", name="uq_models_name_quant"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    hf_repo: Mapped[str | None] = mapped_column(String(200))
    backend: Mapped[str] = mapped_column(String(50))  # llama_cpp | vllm | openai_compat
    endpoint_url: Mapped[str] = mapped_column(String(500))
    quantization: Mapped[str | None] = mapped_column(String(50))  # e.g. Q4_K_M, fp16
    params_b: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list["Run"]] = relationship(back_populates="model")


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    source_url: Mapped[str] = mapped_column(String(500))
    license: Mapped[str] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(20))  # ISO 639-1 code, e.g. en, hi, bn
    n_items: Mapped[int] = mapped_column(Integer)  # size of the fixed subset
    split: Mapped[str] = mapped_column(String(50))
    seed: Mapped[int] = mapped_column(Integer)
    # Loader settings: HF repo, config, field mapping, parallel group, etc.
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    # Ordered item ids of the materialised subset; null until first loaded.
    item_ids_json: Mapped[list | None] = mapped_column(JSON)

    runs: Mapped[list["Run"]] = relationship(back_populates="dataset")


class Run(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_model_id", "model_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("models.id"))
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"))
    # pending | running | completed | failed | cancelled
    status: Mapped[str] = mapped_column(String(20), default="pending")
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    model: Mapped["Model"] = relationship(back_populates="runs")
    dataset: Mapped["Dataset"] = relationship(back_populates="runs")
    items: Mapped[list["ResultItem"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    metrics: Mapped[list["Metric"]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class ResultItem(Base):
    __tablename__ = "results_item"
    __table_args__ = (Index("ix_results_item_run_id", "run_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    item_id: Mapped[str] = mapped_column(String(200))
    language: Mapped[str | None] = mapped_column(String(20))
    subject: Mapped[str | None] = mapped_column(String(100))
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str | None] = mapped_column(Text)
    parsed_answer: Mapped[str | None] = mapped_column(String(50))
    correct: Mapped[bool | None] = mapped_column(Boolean)
    ttft_ms: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    extra_json: Mapped[dict | None] = mapped_column(JSON)

    run: Mapped["Run"] = relationship(back_populates="items")


class Metric(Base):
    __tablename__ = "metrics"
    __table_args__ = (Index("ix_metrics_run_id_dimension", "run_id", "dimension"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"))
    dimension: Mapped[str] = mapped_column(String(50))  # accuracy | latency | resources | ...
    name: Mapped[str] = mapped_column(String(100))
    # Null means "could not be measured"; never an estimate.
    value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(30))
    language: Mapped[str | None] = mapped_column(String(20))
    extra_json: Mapped[dict | None] = mapped_column(JSON)

    run: Mapped["Run"] = relationship(back_populates="metrics")
