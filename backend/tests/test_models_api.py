import pytest

MODEL = {
    "name": "Qwen2.5-7B-Instruct",
    "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
    "backend": "llama_cpp",
    "endpoint_url": "http://host.docker.internal:8080/",
    "quantization": "Q4_K_M",
    "params_b": 7.6,
    "notes": "local GGUF",
}


async def test_create_list_get_delete_model(client):
    created = await client.post("/models", json=MODEL)
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["id"] == 1
    assert body["endpoint_url"] == "http://host.docker.internal:8080"  # trailing slash stripped
    assert body["created_at"]

    listed = await client.get("/models")
    assert listed.status_code == 200
    assert [m["name"] for m in listed.json()] == [MODEL["name"]]

    got = await client.get("/models/1")
    assert got.status_code == 200 and got.json()["quantization"] == "Q4_K_M"

    deleted = await client.delete("/models/1")
    assert deleted.status_code == 204
    assert (await client.get("/models")).json() == []
    assert (await client.get("/models/1")).status_code == 404
    assert (await client.delete("/models/1")).status_code == 404


async def test_duplicate_name_and_quantization_is_409(client):
    assert (await client.post("/models", json=MODEL)).status_code == 201
    dup = await client.post("/models", json=MODEL)
    assert dup.status_code == 409
    other_quant = await client.post("/models", json={**MODEL, "quantization": "Q8_0"})
    assert other_quant.status_code == 201


@pytest.mark.parametrize(
    "patch",
    [
        {"backend": "ollama"},  # not a supported backend
        {"endpoint_url": "not-a-url"},
        {"name": ""},
        {"params_b": -1},
    ],
)
async def test_invalid_model_payloads_are_422(client, patch):
    resp = await client.post("/models", json={**MODEL, **patch})
    assert resp.status_code == 422, resp.text


async def test_delete_model_with_runs_is_refused(client):
    from sqlalchemy import insert

    from app.db.models import Dataset, Run

    assert (await client.post("/models", json=MODEL)).status_code == 201
    factory = client.app.state.session_factory
    async with factory() as session:
        await session.execute(
            insert(Dataset).values(
                id=1,
                name="d",
                source_url="u",
                license="l",
                language="en",
                n_items=1,
                split="test",
                seed=0,
                config_json={},
            )
        )
        await session.execute(
            insert(Run).values(model_id=1, dataset_id=1, status="pending", config_json={})
        )
        await session.commit()
    resp = await client.delete("/models/1")
    assert resp.status_code == 409
    assert "1 run" in resp.json()["detail"]
