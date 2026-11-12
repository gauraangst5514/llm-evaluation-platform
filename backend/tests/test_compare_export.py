"""Compare and CSV export over two real (mock-served) runs with all five dimensions."""

import csv
import io

import pyarrow as pa
import pytest

from app.datasets import loader
from app.evals.tokenizer import TokenizerEval
from app.services.run_executor import ALL_DIMENSIONS
from tests.mock_openai import MockBehaviour, MockServer
from tests.test_evals import FakeTokenizer
from tests.test_loader import fake_table
from tests.test_runs_api import answer_from_prompt, seed, wait_terminal


@pytest.fixture
def fake_source():
    loader.set_table_source(lambda repo, path: fake_table(path.split("/")[0]))
    yield
    loader.set_table_source(None)


@pytest.fixture
def fake_tokenizer(client):
    """Swap the tokenizer loader in the live executor so no Hub download happens."""
    evals = client.app.state.executor._evals
    original = evals["tokenizer"]
    evals["tokenizer"] = TokenizerEval(loader=lambda repo: FakeTokenizer())
    yield
    evals["tokenizer"] = original


def smart_reply(msg: str) -> str:
    if "JSON object" in msg:
        return '{"city": "Mumbai", "country": "India", "population_estimate": 20000000}'
    if "paragraph" in msg:
        return "word " * 30
    return answer_from_prompt(msg)


async def test_full_run_has_all_five_dimensions_then_compare_and_export(
    client, fake_source, fake_tokenizer
):
    beh = MockBehaviour(reply=smart_reply)
    async with MockServer(beh) as srv:
        model_id, dataset_id = await seed(client, srv.base_url, n_items=12)
        # second model row (same server) so compare has two distinct runs
        m2 = await client.post(
            "/models",
            json={
                "name": "mock",
                "backend": "openai_compat",
                "endpoint_url": srv.base_url,
                "quantization": "Q4_K_M",
                "hf_repo": "org/x",
            },
        )
        assert m2.status_code == 201
        ids = []
        for mid in (model_id, m2.json()["id"]):
            r = await client.post(
                "/runs",
                json={
                    "model_id": mid,
                    "dataset_id": dataset_id,
                    "n_items": 6,
                    "concurrency": 2,
                    "dimensions": list(ALL_DIMENSIONS),
                },
            )
            assert r.status_code == 202
            ids.append(r.json()["id"])
        runs = [await wait_terminal(client, i, timeout=60) for i in ids]

    for run in runs:
        assert run["status"] == "completed", run["error"]
        dims = {m["dimension"] for m in run["metrics"]}
        assert dims == set(ALL_DIMENSIONS)
        names = {m["name"]: m for m in run["metrics"]}
        assert names["accuracy"]["value"] == 1.0 and names["accuracy"]["extra_json"]["n"] == 6
        assert names["ttft_ms_p50"]["value"] > 0
        assert names["decode_tokens_per_s_p50"]["value"] > 0
        assert names["peak_rss_bytes"]["value"] is None  # no agent configured: null, not a guess
        assert names["json_valid_rate"]["value"] == 1.0
        assert names["consistency_unanimous_rate"]["extra_json"]["n_items"] == 6
    # first model has no hf_repo -> tokenizer metrics null; second has one -> measured
    t0 = {m["name"]: m for m in runs[0]["metrics"]}["fertility"]
    t1 = {m["name"]: m for m in runs[1]["metrics"]}["fertility"]
    assert t0["value"] is None and t1["value"] > 0

    cmp = await client.get("/compare", params={"run_ids": f"{ids[0]},{ids[1]}"})
    assert cmp.status_code == 200
    body = cmp.json()
    assert [r["id"] for r in body["runs"]] == ids
    acc_row = next(r for r in body["rows"] if r["name"] == "accuracy")
    assert (
        acc_row["dimension"] == "accuracy"
        and acc_row["language"] == "hi"
        and acc_row["unit"] == "fraction"
    )
    assert [c["run_id"] for c in acc_row["cells"]] == ids and all(
        c["value"] == 1.0 for c in acc_row["cells"]
    )
    fert_row = next(r for r in body["rows"] if r["name"] == "fertility")
    assert fert_row["cells"][0]["value"] is None and fert_row["cells"][1]["value"] > 0

    assert (await client.get("/compare", params={"run_ids": "1,999"})).status_code == 404
    assert (await client.get("/compare", params={"run_ids": "a,b"})).status_code == 422

    items_csv = await client.get(f"/export/{ids[0]}.csv")
    assert items_csv.status_code == 200 and items_csv.headers["content-type"].startswith("text/csv")
    rows = list(csv.DictReader(io.StringIO(items_csv.text)))
    assert len(rows) == 6 and rows[0]["correct"] == "True" and rows[0]["language"] == "hi"
    metrics_csv = await client.get(f"/export/{ids[0]}.csv", params={"kind": "metrics"})
    mrows = list(csv.DictReader(io.StringIO(metrics_csv.text)))
    assert {r["dimension"] for r in mrows} == set(ALL_DIMENSIONS)
    assert (await client.get("/export/999.csv")).status_code == 404
    assert pa is not None
