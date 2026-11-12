"""Evaluation dimension interface.

Each dimension is a class with `name` and `async run(ctx) -> list[Metric]`. The run
executor calls every requested dimension after the per-item pass and stores the
returned metrics. A dimension that cannot measure something returns a Metric with
value=None and the reason in `extra`; it never estimates.
"""

from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.backends.openai_compat import OpenAICompatClient
from app.db.models import Dataset, Model, ResultItem, Run
from app.evals.mcq import MCQItem


@dataclass
class Metric:
    dimension: str
    name: str
    value: float | None
    unit: str | None = None
    language: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class EvalContext:
    run: Run
    model: Model
    dataset: Dataset
    items: list[MCQItem]  # the subset that was evaluated, in order
    results: list[ResultItem]  # one row per item from the accuracy pass
    client: OpenAICompatClient
    session_factory: async_sessionmaker[AsyncSession]
    seed: int
    concurrency: int

    @property
    def language(self) -> str:
        return self.dataset.language


class Eval(Protocol):
    name: str

    async def run(self, ctx: EvalContext) -> list[Metric]: ...


def percentile(values: list[float], p: float) -> float | None:
    """Linear-interpolated percentile (like numpy's default); None for empty input."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for a proportion; None when n == 0."""
    if n <= 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
