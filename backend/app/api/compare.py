import csv
import io
import json

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.runs import _load, _progress_map, _summary
from app.db.models import Dataset, Metric, Model, ResultItem, Run
from app.db.session import SessionDep
from app.schemas import CompareCell, CompareResponse, CompareRow

router = APIRouter(tags=["compare"])


def _parse_ids(raw: str) -> list[int]:
    try:
        ids = [int(x) for x in raw.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(422, "run_ids must be comma-separated integers") from None
    if not 1 <= len(ids) <= 12:
        raise HTTPException(422, "Provide between 1 and 12 run ids")
    return ids


@router.get("/compare", response_model=CompareResponse)
async def compare(session: SessionDep, run_ids: str = Query(..., examples=["1,2,3"])):
    ids = _parse_ids(run_ids)
    rows = (
        await session.execute(
            select(Run, Model, Dataset)
            .join(Model, Run.model_id == Model.id)
            .join(Dataset, Run.dataset_id == Dataset.id)
            .where(Run.id.in_(ids))
        )
    ).all()
    found = {r.Run.id: r for r in rows}
    missing = [i for i in ids if i not in found]
    if missing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run(s) not found: {missing}")
    prog = await _progress_map(session, ids)
    summaries = [_summary(found[i].Run, found[i].Model, found[i].Dataset, prog.get(i)) for i in ids]

    metrics = (
        await session.scalars(select(Metric).where(Metric.run_id.in_(ids)).order_by(Metric.id))
    ).all()
    # Group by (dimension, name, language); one cell per run, in the requested order.
    grouped: dict[tuple, dict] = {}
    for m in metrics:
        key = (m.dimension, m.name, m.language)
        g = grouped.setdefault(key, {"unit": m.unit, "cells": {}})
        g["cells"][m.run_id] = CompareCell(run_id=m.run_id, value=m.value, extra_json=m.extra_json)
    out_rows = [
        CompareRow(
            dimension=dim,
            name=name,
            unit=g["unit"],
            language=lang,
            cells=[
                g["cells"].get(i, CompareCell(run_id=i, value=None, extra_json=None)) for i in ids
            ],
        )
        for (dim, name, lang), g in grouped.items()
    ]
    return CompareResponse(runs=summaries, rows=out_rows)


@router.get("/export/{run_id}.csv")
async def export_csv(
    run_id: int, session: SessionDep, kind: str = Query("items", pattern="^(items|metrics)$")
):
    row = await _load(session, run_id)
    buf = io.StringIO()
    w = csv.writer(buf)
    if kind == "metrics":
        w.writerow(
            [
                "run_id",
                "model",
                "quantization",
                "dataset",
                "language",
                "dimension",
                "name",
                "value",
                "unit",
                "extra_json",
            ]
        )
        for m in (
            await session.scalars(select(Metric).where(Metric.run_id == run_id).order_by(Metric.id))
        ).all():
            w.writerow(
                [
                    run_id,
                    row.Model.name,
                    row.Model.quantization,
                    row.Dataset.name,
                    m.language,
                    m.dimension,
                    m.name,
                    m.value,
                    m.unit,
                    json.dumps(m.extra_json, ensure_ascii=False),
                ]
            )
    else:
        w.writerow(
            [
                "run_id",
                "model",
                "quantization",
                "dataset",
                "item_id",
                "language",
                "subject",
                "expected",
                "parsed_answer",
                "correct",
                "ttft_ms",
                "latency_ms",
                "output_tokens",
                "error",
                "response",
            ]
        )
        for it in (
            await session.scalars(
                select(ResultItem).where(ResultItem.run_id == run_id).order_by(ResultItem.id)
            )
        ).all():
            w.writerow(
                [
                    run_id,
                    row.Model.name,
                    row.Model.quantization,
                    row.Dataset.name,
                    it.item_id,
                    it.language,
                    it.subject,
                    (it.extra_json or {}).get("expected"),
                    it.parsed_answer,
                    it.correct,
                    it.ttft_ms,
                    it.latency_ms,
                    it.output_tokens,
                    it.error,
                    it.response,
                ]
            )
    buf.seek(0)
    filename = f"run_{run_id}_{kind}.csv"
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
