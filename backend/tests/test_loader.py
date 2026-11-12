"""Loader tests use an in-memory parquet-like table; nothing touches the network."""

import pyarrow as pa
import pytest

from app.datasets import loader
from app.datasets.registry import GLOBAL_MMLU_FIELDS, sync_registry
from app.db.models import Dataset

N_SOURCE = 50


def fake_table(lang: str) -> pa.Table:
    ids = [f"subj_{i % 5}/test/{i}" for i in range(N_SOURCE)]
    return pa.table(
        {
            "sample_id": ids,
            "subject": [f"subj_{i % 5}" for i in range(N_SOURCE)],
            "question": [f"[{lang}] question {i}" for i in range(N_SOURCE)],
            "option_a": ["a"] * N_SOURCE,
            "option_b": ["b"] * N_SOURCE,
            "option_c": ["c"] * N_SOURCE,
            "option_d": ["d"] * N_SOURCE,
            "answer": ["ABCD"[i % 4] for i in range(N_SOURCE)],
        }
    )


@pytest.fixture
def fake_source():
    calls: list[tuple[str, str]] = []

    def source(repo: str, path: str) -> pa.Table:
        calls.append((repo, path))
        lang = path.split("/")[0]
        return fake_table(lang)

    loader.set_table_source(source)
    yield calls
    loader.set_table_source(None)


def test_choose_item_ids_is_deterministic_and_order_independent():
    ids = [f"x/{i}" for i in range(100)]
    a = loader.choose_item_ids(ids, 10, seed=42)
    b = loader.choose_item_ids(list(reversed(ids)), 10, seed=42)
    c = loader.choose_item_ids(ids, 10, seed=43)
    assert a == b and len(a) == 10 and len(set(a)) == 10
    assert a != c
    assert loader.choose_item_ids(ids, 500, seed=1) == sorted(ids)  # n >= pool -> everything


async def test_parallel_group_shares_ids_and_loads_per_language(client, fake_source):
    factory = client.app.state.session_factory
    async with factory() as session:
        await sync_registry(session)
        ds = {
            d.language: d
            for d in (await session.scalars(__import__("sqlalchemy").select(Dataset))).all()
        }
        for d in ds.values():
            d.n_items = 12
        await session.commit()

        hi_items = await loader.load_items(ds["hi"], session)
        assert len(hi_items) == 12
        assert all(it.language == "hi" and it.question.startswith("[hi]") for it in hi_items)
        assert all(it.answer in "ABCD" for it in hi_items)
        # ids were chosen from the English anchor file, then hi was loaded
        assert fake_source[0][1].startswith("en/") and fake_source[1][1].startswith("hi/")

        en_items = await loader.load_items(ds["en"], session, n_items=5)
        bn_items = await loader.load_items(ds["bn"], session)
        assert [i.item_id for i in en_items] == [i.item_id for i in hi_items][:5]
        assert [i.item_id for i in bn_items] == [i.item_id for i in hi_items]

    # Stored on the rows, so visible through the API and reproducible later.
    rows = (await client.get("/datasets")).json()
    assert all(r["materialised"] for r in rows)

    assert GLOBAL_MMLU_FIELDS["id"] == "sample_id"
