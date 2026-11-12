from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db.models import Model, Run
from app.db.session import SessionDep
from app.schemas import ModelCreate, ModelRead, ModelUpdate

router = APIRouter(prefix="/models", tags=["models"])


@router.post("", response_model=ModelRead, status_code=status.HTTP_201_CREATED)
async def create_model(body: ModelCreate, session: SessionDep):
    model = Model(**body.model_dump(mode="json"))
    session.add(model)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"A model named {body.name!r} with quantization {body.quantization!r} already exists.",
        ) from None
    await session.refresh(model)
    return model


@router.get("", response_model=list[ModelRead])
async def list_models(session: SessionDep):
    return (await session.scalars(select(Model).order_by(Model.id))).all()


@router.get("/{model_id}", response_model=ModelRead)
async def get_model(model_id: int, session: SessionDep):
    model = await session.get(Model, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return model


@router.patch("/{model_id}", response_model=ModelRead)
async def update_model(model_id: int, body: ModelUpdate, session: SessionDep):
    model = await session.get(Model, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(model, key, value)
    await session.commit()
    await session.refresh(model)
    return model


@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(model_id: int, session: SessionDep):
    model = await session.get(Model, model_id)
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    n_runs = await session.scalar(
        select(func.count()).select_from(Run).where(Run.model_id == model_id)
    )
    if n_runs:
        # Runs are evidence; refuse to orphan or silently drop them.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"Model has {n_runs} run(s). Delete those runs first.",
        )
    await session.delete(model)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
