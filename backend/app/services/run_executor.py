"""Executes evaluation runs in the background.

One asyncio task per run, bounded concurrency per run, per-request timeout, one retry
on transient failures, cooperative cancellation, and an honest terminal status:
completed | failed | cancelled, with the error text stored on the run.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.backends.base import ChatResult
from app.backends.openai_compat import (
    OpenAICompatClient,
    PermanentBackendError,
    TransientBackendError,
)
from app.config import get_settings
from app.datasets.loader import load_items
from app.db.models import Dataset, Metric, Model, ResultItem, Run
from app.evals.base import Eval, EvalContext
from app.evals.mcq import (
    PROMPT_TEMPLATE_VERSION,
    MCQItem,
    build_messages,
    parse_answer,
    render_prompt,
)
from app.evals.registry import default_evals

log = logging.getLogger(__name__)

ALL_DIMENSIONS = ("accuracy", "latency", "resources", "tokenizer", "reliability")
# If more than this share of items fail at the transport level after a retry, the run
# is marked failed rather than reported as a low-accuracy result.
MAX_ERROR_FRACTION = 0.10


@dataclass
class RunConfig:
    dimensions: list[str]
    n_items: int
    seed: int
    concurrency: int
    max_tokens: int
    temperature: float = 0.0
    timeout_s: float | None = None

    @classmethod
    def from_json(cls, cfg: dict) -> "RunConfig":
        s = get_settings()
        return cls(
            dimensions=list(cfg.get("dimensions") or ALL_DIMENSIONS),
            n_items=int(cfg["n_items"]),
            seed=int(cfg.get("seed", 42)),
            concurrency=int(cfg.get("concurrency") or s.default_concurrency),
            max_tokens=int(cfg.get("max_tokens") or s.default_max_tokens),
            temperature=float(cfg.get("temperature", 0.0)),
            timeout_s=cfg.get("timeout_s") or s.request_timeout_s,
        )


def _now() -> datetime:
    return datetime.now(UTC)


class RunExecutor:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        evals: dict[str, Eval] | None = None,
    ):
        self._sf = session_factory
        self._tasks: dict[int, asyncio.Task] = {}
        self._evals = evals if evals is not None else default_evals()

    # ---- lifecycle -----------------------------------------------------------------

    async def fail_orphans(self) -> int:
        """On startup: anything still 'running' or 'pending' cannot be resumed."""
        async with self._sf() as session:
            result = await session.execute(
                update(Run)
                .where(Run.status.in_(["running", "pending"]))
                .values(
                    status="failed",
                    finished_at=_now(),
                    error="API process restarted before the run finished.",
                )
            )
            await session.commit()
            return result.rowcount or 0

    def start(self, run_id: int) -> None:
        if run_id in self._tasks:
            return
        task = asyncio.create_task(self._execute(run_id), name=f"run-{run_id}")
        self._tasks[run_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(run_id, None))

    def is_active(self, run_id: int) -> bool:
        return run_id in self._tasks

    async def cancel(self, run_id: int) -> bool:
        task = self._tasks.get(run_id)
        if task is None:
            return False
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - task already recorded its state
            pass
        return True

    async def wait(self, run_id: int) -> None:
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.shield(task)

    async def shutdown(self) -> None:
        for run_id in list(self._tasks):
            await self.cancel(run_id)

    # ---- execution -------------------------------------------------------------------

    async def _set_status(self, run_id: int, **values) -> None:
        async with self._sf() as session:
            await session.execute(update(Run).where(Run.id == run_id).values(**values))
            await session.commit()

    async def _execute(self, run_id: int) -> None:
        try:
            async with self._sf() as session:
                run = await session.get(Run, run_id)
                if run is None:
                    return
                model = await session.get(Model, run.model_id)
                dataset = await session.get(Dataset, run.dataset_id)
                cfg = RunConfig.from_json(run.config_json)
                run.status = "running"
                run.started_at = _now()
                run.config_json = {
                    **run.config_json,
                    "prompt_template": PROMPT_TEMPLATE_VERSION,
                    "resolved_concurrency": cfg.concurrency,
                }
                await session.commit()
                items = await load_items(dataset, session, cfg.n_items)

            client = OpenAICompatClient(model.endpoint_url, default_timeout_s=cfg.timeout_s)
            try:
                await client.ping()  # fail fast and clearly if the server is not there
                await self._run_items(run_id, client, items, cfg)

                n_errors = await self._count_errors(run_id)
                if items and n_errors / len(items) > MAX_ERROR_FRACTION:
                    await self._set_status(
                        run_id,
                        status="failed",
                        finished_at=_now(),
                        error=f"{n_errors}/{len(items)} requests failed after retry.",
                    )
                    return
                await self._run_evals(run, model, dataset, items, client, cfg)
            finally:
                await client.aclose()
            await self._set_status(run_id, status="completed", finished_at=_now())
        except asyncio.CancelledError:
            await self._set_status(
                run_id, status="cancelled", finished_at=_now(), error="Cancelled by user."
            )
            raise
        except Exception as exc:  # noqa: BLE001 - whatever happened, record it on the run
            log.exception("Run %s failed", run_id)
            await self._set_status(
                run_id, status="failed", finished_at=_now(), error=f"{type(exc).__name__}: {exc}"
            )

    async def _run_evals(
        self,
        run: Run,
        model: Model,
        dataset: Dataset,
        items: list[MCQItem],
        client: OpenAICompatClient,
        cfg: RunConfig,
    ) -> None:
        """Run each requested dimension and store its metrics. One dimension failing is
        recorded as a null metric with the error; it does not fail the run."""
        async with self._sf() as session:
            results = (
                await session.scalars(
                    select(ResultItem).where(ResultItem.run_id == run.id).order_by(ResultItem.id)
                )
            ).all()
        ctx = EvalContext(
            run=run,
            model=model,
            dataset=dataset,
            items=items,
            results=list(results),
            client=client,
            session_factory=self._sf,
            seed=cfg.seed,
            concurrency=cfg.concurrency,
        )
        for dim in cfg.dimensions:
            evaluator = self._evals.get(dim)
            if evaluator is None:
                metrics = [
                    Metric(
                        run_id=run.id,
                        dimension=dim,
                        name=f"{dim}_error",
                        value=None,
                        extra_json={"error": "unknown dimension"},
                    )
                ]
            else:
                try:
                    computed = await evaluator.run(ctx)
                    metrics = [
                        Metric(
                            run_id=run.id,
                            dimension=m.dimension,
                            name=m.name,
                            value=m.value,
                            unit=m.unit,
                            language=m.language,
                            extra_json=m.extra,
                        )
                        for m in computed
                    ]
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - record and move on to the next dimension
                    log.exception("Dimension %s failed for run %s", dim, run.id)
                    metrics = [
                        Metric(
                            run_id=run.id,
                            dimension=dim,
                            name=f"{dim}_error",
                            value=None,
                            extra_json={"error": f"{type(exc).__name__}: {exc}"[:500]},
                        )
                    ]
            async with self._sf() as session:
                session.add_all(metrics)
                await session.commit()

    async def _count_errors(self, run_id: int) -> int:
        async with self._sf() as session:
            rows = await session.scalars(
                select(ResultItem.id).where(
                    ResultItem.run_id == run_id, ResultItem.error.is_not(None)
                )
            )
            return len(rows.all())

    async def _run_items(
        self, run_id: int, client: OpenAICompatClient, items: list[MCQItem], cfg: RunConfig
    ) -> None:
        sem = asyncio.Semaphore(cfg.concurrency)

        async def one(item: MCQItem) -> None:
            async with sem:
                await self._run_item(run_id, client, item, cfg)

        tasks = [asyncio.create_task(one(it)) for it in items]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            raise

    async def _run_item(
        self, run_id: int, client: OpenAICompatClient, item: MCQItem, cfg: RunConfig
    ) -> None:
        messages = build_messages(item)
        prompt_text = render_prompt(messages)
        result: ChatResult | None = None
        error: str | None = None
        attempts = 0
        for attempt in range(2):  # at most one retry, and only on transient failures
            attempts = attempt + 1
            try:
                result = await client.chat(
                    messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    seed=cfg.seed,
                    timeout_s=cfg.timeout_s,
                )
                error = None
                break
            except TransientBackendError as exc:
                error = f"transient: {exc}"
                continue
            except PermanentBackendError as exc:
                error = f"permanent: {exc}"
                break

        parsed = parse_answer(result.text) if result else None
        row = ResultItem(
            run_id=run_id,
            item_id=item.item_id,
            language=item.language,
            subject=item.subject,
            prompt=prompt_text,
            response=result.text if result else None,
            parsed_answer=parsed,
            # Unparseable output counts as wrong; a failed request is not scored (None).
            correct=(parsed == item.answer) if result else None,
            ttft_ms=result.ttft_ms if result else None,
            latency_ms=result.latency_ms if result else None,
            output_tokens=result.output_tokens if result else None,
            error=error,
            extra_json={
                "expected": item.answer,
                "attempts": attempts,
                "finish_reason": result.finish_reason if result else None,
                "prompt_tokens": result.prompt_tokens if result else None,
                "n_chunks": result.n_chunks if result else None,
                "subject_category": item.subject_category,
            },
        )
        async with self._sf() as session:
            session.add(row)
            await session.commit()
