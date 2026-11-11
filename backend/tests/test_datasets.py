from app.datasets.registry import REGISTRY, sync_registry


async def test_registry_entries_are_well_formed():
    assert len(REGISTRY) >= 3
    assert {s.language for s in REGISTRY} >= {"en", "hi", "bn"}
    for spec in REGISTRY:
        assert spec.license, f"{spec.name} has no licence recorded"
        assert spec.source_url.startswith("https://")
        assert spec.n_items > 0 and spec.split
        assert spec.config_json["kind"] == "mcq"


async def test_sync_is_idempotent_and_preserves_ids(client):
    factory = client.app.state.session_factory
    async with factory() as session:
        assert await sync_registry(session) == len(REGISTRY)
    async with factory() as session:
        assert await sync_registry(session) == 0  # second sync inserts nothing

    resp = await client.get("/datasets")
    assert resp.status_code == 200
    rows = resp.json()
    assert [r["name"] for r in rows] == [s.name for s in REGISTRY]
    assert all(r["materialised"] is False for r in rows)
    assert rows[0]["license"] == "Apache-2.0"

    one = await client.get(f"/datasets/{rows[0]['id']}")
    assert one.status_code == 200 and one.json()["language"] == "en"
    assert (await client.get("/datasets/999")).status_code == 404


async def test_datasets_empty_before_sync(client):
    assert (await client.get("/datasets")).json() == []
