from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Dataset, Model, ResultItem, Run
from app.db.session import SessionDep
from app.schemas import Page, RunCreate, RunDetail, RunProgress, RunRead, RunSummary
from app.services.run_executor import RunExecutor

router = APIRouter(prefix="/runs", tags=["runs"])


async def _progress_map(session: AsyncSession, run_ids: list[int]) -> dict[int, dict]:
    if not run_ids:
        return {}
    q = (
        select(
            ResultItem.run_id,
            func.count().label("n_done"),
            func.sum(case((ResultItem.correct.is_(True), 1), else_=0)).label("n_correct"),
            func.sum(case((ResultItem.error.is_(None), 1), else_=0)).label("n_scored"),
            func.sum(case((ResultItem.error.is_not(None), 1), else_=0)).label("n_errors"),
            func.sum(
                case(
                    (
                        (ResultItem.error.is_(None)) & (ResultItem.parsed_answer.is_(None)),
                        1,
                    ),
                    else_=0,
                )
            ).label("n_unparseable"),
        )
        .where(ResultItem.run_id.in_(run_ids))
        .group_by(ResultItem.run_id)
    )
    return {r.run_id: r._asdict() for r in (await session.execute(q)).all()}


def _summary(run: Run, model: Model, dataset: Dataset, prog: dict | None) -> RunSummary:
    p = prog or {}
    n_scored = int(p.get("n_scored") or 0)
    n_correct = int(p.get("n_correct") or 0)
    return RunSummary(
        **RunRead.model_validate(run).model_dump(),
        model_name=model.name,
        quantization=model.quantization,
        dataset_name=dataset.name,
        language=dataset.language,
        progress=RunProgress(
            n_total=int(run.config_json.get("n_items", 0)),
            n_done=int(p.get("n_done") or 0),
            n_correct=n_correct,
            n_scored=n_scored,
            n_errors=int(p.get("n_errors") or 0),
            n_unparseable=int(p.get("n_unparseable") or 0),
        ),
        accuracy=(n_correct / n_scored) if n_scored else None,
    )


def _executor(request: Request) -> RunExecutor:
    return request.app.state.executor


@router.post("", response_model=RunRead, status_code=status.HTTP_202_ACCEPTED)
async def create_run(body: RunCreate, session: SessionDep, request: Request):
    model = await session.get(Model, body.model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    dataset = await session.get(Dataset, body.dataset_id)
    if dataset is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    if body.n_items > dataset.n_items:
        raise HTTPException(
            422,
            f"n_items={body.n_items} exceeds the dataset's fixed subset ({dataset.n_items}).",
        )
    run = Run(
        model_id=model.id,
        dataset_id=dataset.id,
        status="pending",
        config_json=body.model_dump(exclude={"model_id", "dataset_id"}, exclude_none=True),
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    _executor(request).start(run.id)
    return run


@router.get("", response_model=list[RunSummary])
async def list_runs(session: SessionDep):
    rows = (
        await session.execute(
            select(Run, Model, Dataset)
            .join(Model, Run.model_id == Model.id)
            .join(Dataset, Run.dataset_id == Dataset.id)
            .order_by(Run.id.desc())
        )
    ).all()
    prog = await _progress_map(session, [r.Run.id for r in rows])
    return [_summary(r.Run, r.Model, r.Dataset, prog.get(r.Run.id)) for r in rows]


async def _load(session: AsyncSession, run_id: int):
    row = (
        await session.execute(
            select(Run, Model, Dataset)
            .join(Model, Run.model_id == Model.id)
            .join(Dataset, Run.dataset_id == Dataset.id)
            .where(Run.id == run_id)
        )
    ).first()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    return row


@router.get("/{run_id}", response_model=RunDetail)
async def get_run(run_id: int, session: SessionDep):
    row = await _load(session, run_id)
    prog = await _progress_map(session, [run_id])
    summary = _summary(row.Run, row.Model, row.Dataset, prog.get(run_id))
    await session.refresh(row.Run, ["metrics"])
    return RunDetail(**summary.model_dump(), metrics=row.Run.metrics)


@router.get("/{run_id}/items", response_model=Page)
async def list_items(
    run_id: int,
    session: SessionDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    only_failed: bool = Query(False, description="Only wrong, unparseable or errored items"),
):
    await _load(session, run_id)
    q = select(ResultItem).where(ResultItem.run_id == run_id)
    if only_failed:
        q = q.where(ResultItem.correct.is_not(True))
    total = await session.scalar(select(func.count()).select_from(q.subquery()))
    rows = (
        await session.scalars(
            q.order_by(ResultItem.id).offset((page - 1) * page_size).limit(page_size)
        )
    ).all()
    return Page(items=rows, total=total or 0, page=page, page_size=page_size)


@router.post("/{run_id}/cancel", response_model=RunRead)
async def cancel_run(run_id: int, session: SessionDep, request: Request):
    row = await _load(session, run_id)
    if row.Run.status not in ("pending", "running"):
        raise HTTPException(status.HTTP_409_CONFLICT, f"Run is already {row.Run.status}.")
    cancelled = await _executor(request).cancel(run_id)
    if not cancelled:
        # Not tracked by this process (e.g. created before a restart): close it out.
        row.Run.status = "cancelled"
        row.Run.error = "Cancelled; run was not active in this process."
        await session.commit()
    await session.refresh(row.Run)
    return row.Run


@router.delete("/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_run(run_id: int, session: SessionDep, request: Request):
    row = await _load(session, run_id)
    if _executor(request).is_active(run_id):
        raise HTTPException(status.HTTP_409_CONFLICT, "Cancel the run before deleting it.")
    await session.delete(row.Run)
    await session.commit()
