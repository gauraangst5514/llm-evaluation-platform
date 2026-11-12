"""Materialise fixed, seeded subsets of registered datasets.

Data is fetched from the Hugging Face Hub at runtime into the local HF cache and is
never written into the repo. The chosen item ids are stored on the dataset row so a
later run, or a run in another language of the same parallel group, uses the same
items.
"""

import random
from collections.abc import Callable

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.datasets.registry import get_spec
from app.db.models import Dataset
from app.evals.mcq import MCQItem


def _parquet_path(hf_config: str, split: str) -> str:
    return f"{hf_config}/{split}-00000-of-00001.parquet"


def download_parquet(hf_repo: str, path_in_repo: str) -> pa.Table:
    local = hf_hub_download(
        repo_id=hf_repo,
        filename=path_in_repo,
        repo_type="dataset",
        cache_dir=get_settings().hf_cache_dir,
    )
    return pq.read_table(local)


# Tests swap this for an in-memory source; production uses the Hub.
TableSource = Callable[[str, str], pa.Table]
_table_source: TableSource = download_parquet


def set_table_source(source: TableSource | None) -> None:
    global _table_source
    _table_source = source or download_parquet


def _load_table(cfg: dict, split: str, hf_config: str | None = None) -> pa.Table:
    return _table_source(cfg["hf_repo"], _parquet_path(hf_config or cfg["hf_config"], split))


def choose_item_ids(all_ids: list[str], n: int, seed: int) -> list[str]:
    """Deterministic subset: sort for stability, then sample with a seeded RNG."""
    pool = sorted(set(all_ids))
    if n >= len(pool):
        return pool
    return random.Random(seed).sample(pool, n)


async def materialise(dataset: Dataset, session: AsyncSession) -> list[str]:
    """Store (and return) the dataset's item ids. Idempotent."""
    if dataset.item_ids_json:
        return list(dataset.item_ids_json)
    cfg = dataset.config_json
    group = cfg.get("parallel_group")
    if group:
        # Reuse ids already chosen for a sibling in the same group, if any.
        siblings = (await session.scalars(select(Dataset))).all()
        for sib in siblings:
            if (
                sib.id != dataset.id
                and sib.config_json.get("parallel_group") == group
                and sib.item_ids_json
            ):
                dataset.item_ids_json = list(sib.item_ids_json)
                await session.commit()
                return dataset.item_ids_json
        anchor = cfg.get("parallel_anchor", cfg["hf_config"])
        table = _load_table(cfg, dataset.split, anchor)
    else:
        table = _load_table(cfg, dataset.split)
    id_col = cfg["fields"]["id"]
    ids = choose_item_ids(table.column(id_col).to_pylist(), dataset.n_items, dataset.seed)
    dataset.item_ids_json = ids
    await session.commit()
    return ids


async def load_items(
    dataset: Dataset, session: AsyncSession, n_items: int | None = None
) -> list[MCQItem]:
    """Return the first n_items of the materialised subset, in stored order."""
    ids = await materialise(dataset, session)
    if n_items is not None:
        ids = ids[:n_items]
    cfg = dataset.config_json
    f = cfg["fields"]
    table = _load_table(cfg, dataset.split)
    wanted = set(ids)
    rows = {}
    for row in table.to_pylist():
        if row[f["id"]] in wanted:
            rows[row[f["id"]]] = row
    missing = [i for i in ids if i not in rows]
    if missing:
        raise ValueError(
            f"{dataset.name}: {len(missing)} subset ids missing from source, e.g. {missing[:3]}"
        )
    items = []
    for item_id in ids:
        row = rows[item_id]
        items.append(
            MCQItem(
                item_id=item_id,
                language=dataset.language,
                subject=row.get(f["subject"]),
                question=row[f["question"]],
                options=tuple(row[c] for c in f["options"]),
                answer=str(row[f["answer"]]).strip().upper(),
                subject_category=row.get(f.get("subject_category", ""))
                if f.get("subject_category")
                else None,
            )
        )
    return items


def spec_exists(name: str) -> bool:
    return get_spec(name) is not None
