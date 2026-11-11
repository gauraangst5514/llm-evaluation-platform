from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.db.models import Dataset
from app.db.session import SessionDep
from app.schemas import DatasetRead

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _to_read(d: Dataset) -> DatasetRead:
    out = DatasetRead.model_validate(d)
    out.materialised = bool(d.item_ids_json)
    return out


@router.get("", response_model=list[DatasetRead])
async def list_datasets(session: SessionDep):
    rows = (await session.scalars(select(Dataset).order_by(Dataset.id))).all()
    return [_to_read(d) for d in rows]


@router.get("/{dataset_id}", response_model=DatasetRead)
async def get_dataset(dataset_id: int, session: SessionDep):
    d = await session.get(Dataset, dataset_id)
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found")
    return _to_read(d)
