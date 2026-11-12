"""End-to-end: POST /runs against the mock OpenAI server, items land in Postgres."""

import asyncio

import pyarrow as pa
import pytest
from sqlalchemy import select

from app.datasets import loader
from app.datasets.registry import sync_registry
from app.db.models import Dataset
from tests.mock_openai import MockBehaviour, MockServer
from tests.test_loader import fake_table


@pytest.fixture
def fake_source():
    loader.set_table_source(lambda repo, path: fake_table(path.split("/")[0]))
    yield
    loader.set_table_source(None)


def answer_from_prompt(user_msg: str) -> str:
    """The fake data encodes the answer as 'ABCD'[i % 4] where i is the question index."""
    if "question " not in user_msg:  # latency/reliability probes
        return "A"
    i = int(user_msg.split("question ")[1].split("\n")[0])
    return "ABCD"[i % 4]


async def seed(client, endpoint_url: str, n_items: int = 12) -> tuple[int, int]:
    factory = client.app.state.session_factory
    async with factory() as session:
        await sync_registry(session)
        for d in (await session.scalars(select(Dataset))).all():
            d.n_items = n_items
        await session.commit()
    model = await client.post(
        "/models",
        json={
            "name": "mock",
            "backend": "openai_compat",
            "endpoint_url": endpoint_url,
            "quantization": "none",
        },
    )
    assert model.status_code == 201
    datasets = (await client.get("/datasets")).json()
    hi = next(d for d in datasets if d["language"] == "hi")
    return model.json()["id"], hi["id"]


async def wait_terminal(client, run_id: int, timeout: float = 20.0) -> dict:
    await asyncio.wait_for(client.app.state.executor.wait(run_id), timeout)
    return (await client.get(f"/runs/{run_id}")).json()


async def test_run_completes_and_stores_items(client, fake_source):
    beh = MockBehaviour(reply=answer_from_prompt)
    async with MockServer(beh) as srv:
        model_id, dataset_id = await seed(client, srv.base_url)
        resp = await client.post(
            "/runs",
            json={
                "model_id": model_id,
                "dataset_id": dataset_id,
                "n_items": 10,
                "concurrency": 3,
                "seed": 7,
                "dimensions": ["accuracy"],
            },
        )
        assert resp.status_code == 202, resp.text
        run_id = resp.json()["id"]
        run = await wait_terminal(client, run_id)

    assert run["status"] == "completed", run
    assert run["error"] is None and run["started_at"] and run["finished_at"]
    assert run["progress"] == {
        "n_total": 10,
        "n_done": 10,
        "n_correct": 10,
        "n_scored": 10,
        "n_errors": 0,
        "n_unparseable": 0,
    }
    assert run["accuracy"] == 1.0
    assert run["model_name"] == "mock" and run["language"] == "hi"
    assert run["config_json"]["prompt_template"] == "mcq-v1"
    assert run["config_json"]["resolved_concurrency"] == 3
    names = {m["name"] for m in run["metrics"]}
    assert "accuracy" in names and all(m["dimension"] == "accuracy" for m in run["metrics"])

    page = (await client.get(f"/runs/{run_id}/items", params={"page_size": 4})).json()
    assert page["total"] == 10 and len(page["items"]) == 4 and page["page"] == 1
    item = page["items"][0]
    assert item["correct"] is True and item["parsed_answer"] == item["extra_json"]["expected"]
    assert item["ttft_ms"] > 0 and item["latency_ms"] >= item["ttft_ms"]
    assert item["output_tokens"] == 1 and item["prompt"].startswith("[system]")
    assert item["language"] == "hi" and item["subject"].startswith("subj_")
    assert len(beh.received) == 10
    assert all(
        r["temperature"] == 0.0 and r["seed"] == 7 and r["max_tokens"] == 16 for r in beh.received
    )

    listed = (await client.get("/runs")).json()
    assert [r["id"] for r in listed] == [run_id]


async def test_wrong_and_unparseable_answers_are_scored_honestly(client, fake_source):
    replies = iter(["A", "I am not sure, maybe B or C", "Z", "D"] * 10)
    beh = MockBehaviour(reply=lambda _m: next(replies))
    async with MockServer(beh) as srv:
        model_id, dataset_id = await seed(client, srv.base_url)
        resp = await client.post(
            "/runs",
            json={"model_id": model_id, "dataset_id": dataset_id, "n_items": 8, "concurrency": 1},
        )
        run = await wait_terminal(client, resp.json()["id"])
    assert run["status"] == "completed"
    p = run["progress"]
    assert p["n_done"] == 8 and p["n_errors"] == 0 and p["n_scored"] == 8
    assert p["n_unparseable"] == 4  # two of every four replies
    assert 0 <= p["n_correct"] <= 4
    failed = (await client.get(f"/runs/{run['id']}/items", params={"only_failed": "true"})).json()
    assert failed["total"] == 8 - p["n_correct"]


async def test_transient_failure_is_retried_once(client, fake_source):
    beh = MockBehaviour(reply=answer_from_prompt, fail_first_n_requests=1)
    async with MockServer(beh) as srv:
        model_id, dataset_id = await seed(client, srv.base_url)
        resp = await client.post(
            "/runs",
            json={
                "model_id": model_id,
                "dataset_id": dataset_id,
                "n_items": 3,
                "concurrency": 1,
                "dimensions": ["accuracy"],
            },
        )
        run = await wait_terminal(client, resp.json()["id"])
    assert run["status"] == "completed" and run["progress"]["n_errors"] == 0
    assert len(beh.received) == 4  # 3 items + 1 retry
    items = (await client.get(f"/runs/{run['id']}/items")).json()["items"]
    assert sorted(i["extra_json"]["attempts"] for i in items) == [1, 1, 2]


async def test_persistent_failures_mark_run_failed(client, fake_source):
    beh = MockBehaviour(reply=answer_from_prompt, fail_first_n_requests=100)
    async with MockServer(beh) as srv:
        model_id, dataset_id = await seed(client, srv.base_url)
        resp = await client.post(
            "/runs",
            json={"model_id": model_id, "dataset_id": dataset_id, "n_items": 4, "concurrency": 2},
        )
        run = await wait_terminal(client, resp.json()["id"])
    assert run["status"] == "failed"
    assert "4/4 requests failed" in run["error"]
    assert run["progress"]["n_errors"] == 4 and run["accuracy"] is None
    items = (await client.get(f"/runs/{run['id']}/items")).json()["items"]
    assert all(i["correct"] is None and i["error"].startswith("transient") for i in items)


async def test_unreachable_server_fails_fast(client, fake_source):
    model_id, dataset_id = await seed(client, "http://127.0.0.1:9")
    resp = await client.post(
        "/runs", json={"model_id": model_id, "dataset_id": dataset_id, "n_items": 2}
    )
    run = await wait_terminal(client, resp.json()["id"])
    assert run["status"] == "failed" and "Connect" in run["error"]
    assert run["progress"]["n_done"] == 0


async def test_cancel_running_run(client, fake_source):
    beh = MockBehaviour(reply=answer_from_prompt, first_token_delay_s=0.3)
    async with MockServer(beh) as srv:
        model_id, dataset_id = await seed(client, srv.base_url)
        resp = await client.post(
            "/runs",
            json={"model_id": model_id, "dataset_id": dataset_id, "n_items": 12, "concurrency": 1},
        )
        run_id = resp.json()["id"]
        await asyncio.sleep(0.8)  # let a couple of items finish
        cancelled = await client.post(f"/runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        run = await wait_terminal(client, run_id)
    assert run["status"] == "cancelled" and run["error"] == "Cancelled by user."
    assert 0 < run["progress"]["n_done"] < 12
    assert (await client.post(f"/runs/{run_id}/cancel")).status_code == 409
    assert (await client.delete(f"/runs/{run_id}")).status_code == 204
    assert (await client.get(f"/runs/{run_id}")).status_code == 404


async def test_validation_errors(client, fake_source):
    async with MockServer() as srv:
        model_id, dataset_id = await seed(client, srv.base_url, n_items=5)
        assert (
            await client.post("/runs", json={"model_id": 999, "dataset_id": dataset_id})
        ).status_code == 404
        assert (
            await client.post("/runs", json={"model_id": model_id, "dataset_id": 999})
        ).status_code == 404
        too_many = await client.post(
            "/runs", json={"model_id": model_id, "dataset_id": dataset_id, "n_items": 6}
        )
        assert too_many.status_code == 422 and "exceeds" in too_many.json()["detail"]
        bad_dim = await client.post(
            "/runs", json={"model_id": model_id, "dataset_id": dataset_id, "dimensions": ["vibes"]}
        )
        assert bad_dim.status_code == 422
        assert (
            await client.delete(f"/models/{model_id}")
        ).status_code == 204  # no runs were created


async def test_restart_marks_interrupted_runs_failed(client, fake_source):
    from app.db.models import Run

    factory = client.app.state.session_factory
    async with factory() as session:
        await sync_registry(session)
        from app.db.models import Model

        session.add(Model(name="m", backend="vllm", endpoint_url="http://x", quantization=None))
        await session.flush()
        session.add(Run(model_id=1, dataset_id=1, status="running", config_json={"n_items": 3}))
        await session.commit()
    n = await client.app.state.executor.fail_orphans()
    assert n == 1
    run = (await client.get("/runs/1")).json()
    assert run["status"] == "failed" and "restarted" in run["error"]
    assert pa is not None
