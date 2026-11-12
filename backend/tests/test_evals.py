"""Unit tests for the five dimensions, using in-memory rows and the mock server."""

import json

import pytest

from app.backends.openai_compat import OpenAICompatClient
from app.db.models import Dataset, Model, ResultItem, Run
from app.evals.accuracy import AccuracyEval
from app.evals.base import EvalContext, percentile, wilson_interval
from app.evals.latency import LatencyEval
from app.evals.mcq import MCQItem
from app.evals.reliability import ReliabilityEval, json_is_valid
from app.evals.resources import ResourcesEval
from app.evals.tokenizer import TokenizerEval, item_text
from tests.mock_openai import MockBehaviour, MockServer


def make_items(n: int, lang: str = "hi") -> list[MCQItem]:
    return [
        MCQItem(
            item_id=f"s{i % 3}/test/{i}",
            language=lang,
            subject=f"s{i % 3}",
            question=f"प्रश्न {i} what is it?",
            options=("one", "two", "three", "four"),
            answer="ABCD"[i % 4],
            subject_category="STEM" if i % 2 == 0 else "Humanities",
        )
        for i in range(n)
    ]


def make_results(items, parsed, errors=(), ttft=50.0, latency=120.0, prompt_tokens=100):
    rows = []
    for i, (it, p) in enumerate(zip(items, parsed, strict=True)):
        err = "transient: boom" if i in errors else None
        rows.append(
            ResultItem(
                run_id=1,
                item_id=it.item_id,
                language=it.language,
                subject=it.subject,
                prompt="p",
                response=None if err else (p or "garbage"),
                parsed_answer=None if err else p,
                correct=None if err else (p == it.answer),
                ttft_ms=None if err else ttft + i,
                latency_ms=None if err else latency + i,
                output_tokens=None if err else 2,
                error=err,
                extra_json={
                    "expected": it.answer,
                    "prompt_tokens": prompt_tokens,
                    "subject_category": it.subject_category,
                },
            )
        )
    return rows


def make_ctx(
    items, results, client=None, hf_repo="org/repo", agent_url=None, seed=1, concurrency=2
):
    model = Model(
        id=1,
        name="m",
        backend="llama_cpp",
        endpoint_url="http://x",
        hf_repo=hf_repo,
        resource_agent_url=agent_url,
    )
    dataset = Dataset(
        id=1,
        name="d",
        source_url="u",
        license="l",
        language=items[0].language if items else "en",
        n_items=len(items),
        split="test",
        seed=seed,
        config_json={},
    )
    from datetime import UTC, datetime

    run = Run(
        id=1,
        model_id=1,
        dataset_id=1,
        status="running",
        config_json={},
        started_at=datetime.now(UTC),
    )
    return EvalContext(
        run=run,
        model=model,
        dataset=dataset,
        items=items,
        results=results,
        client=client,
        session_factory=None,
        seed=seed,
        concurrency=concurrency,
    )


def by_name(metrics):
    return {m.name: m for m in metrics}


# ---- helpers ------------------------------------------------------------------------


def test_percentile_and_wilson():
    assert percentile([], 50) is None
    assert percentile([7.0], 95) == 7.0
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert wilson_interval(0, 0) is None
    lo, hi = wilson_interval(7, 20)
    assert 0.17 < lo < 0.19 and 0.56 < hi < 0.58  # 35% with n=20 is a wide interval
    assert wilson_interval(20, 20)[1] == 1.0


# ---- accuracy -----------------------------------------------------------------------


async def test_accuracy_metrics_scored_honestly():
    items = make_items(10)
    # expected A B C D A B C D A B; give 6 right, 2 wrong, 1 unparseable, 1 error
    parsed = ["A", "B", "C", "D", "A", "B", "A", "A", None, "B"]
    results = make_results(items, parsed, errors={9})
    m = by_name(await AccuracyEval().run(make_ctx(items, results)))
    acc = m["accuracy"]
    assert acc.value == pytest.approx(6 / 9) and acc.language == "hi" and acc.unit == "fraction"
    assert acc.extra["n"] == 9 and acc.extra["n_total"] == 10 and acc.extra["n_errors"] == 1
    assert acc.extra["ci95_low"] < acc.value < acc.extra["ci95_high"]
    assert m["unparseable_rate"].value == pytest.approx(1 / 9)
    assert m["request_error_rate"].value == pytest.approx(1 / 10)
    cats = {k: v for k, v in m.items() if k.startswith("accuracy_by_category:")}
    assert set(cats) == {"accuracy_by_category:STEM", "accuracy_by_category:Humanities"}
    assert sum(v.extra["n"] for v in cats.values()) == 9
    subj = m["accuracy_by_subject"]
    assert subj.value is None and sum(r["n"] for r in subj.extra["rows"]) == 9
    assert m["answer_distribution"].extra["counts"]["unparseable"] == 1


async def test_accuracy_with_nothing_scored_is_null():
    items = make_items(3)
    results = make_results(items, [None, None, None], errors={0, 1, 2})
    m = by_name(await AccuracyEval().run(make_ctx(items, results)))
    assert m["accuracy"].value is None and m["accuracy"].extra["n"] == 0
    assert m["request_error_rate"].value == 1.0


# ---- latency ------------------------------------------------------------------------


async def test_latency_percentiles_and_probes():
    items = make_items(5)
    results = make_results(items, ["A"] * 5, ttft=100.0, latency=200.0, prompt_tokens=250)
    beh = MockBehaviour(
        reply=lambda _m: "x" * 40, first_token_delay_s=0.05, inter_chunk_delay_s=0.005
    )
    async with MockServer(beh) as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            m = by_name(
                await LatencyEval(n_probes=2, probe_max_tokens=40).run(
                    make_ctx(items, results, client)
                )
            )
        finally:
            await client.aclose()
    assert m["ttft_ms_p50"].value == 102.0 and m["ttft_ms_p95"].value == pytest.approx(103.8)
    assert m["latency_ms_p50"].value == 202.0 and m["ttft_ms_p50"].extra == {
        "n": 5,
        "source": "mcq_items",
        "concurrency": 2,
    }
    assert m["prefill_tokens_per_s_p50"].value == pytest.approx(250 / 0.102)
    tps = m["decode_tokens_per_s_p50"]
    assert tps.value is not None and tps.value > 0 and tps.unit == "tokens/s"
    assert (
        tps.extra["n"] == 2
        and tps.extra["concurrency"] == 1
        and tps.extra["output_tokens"] == [40, 40]
    )
    assert len(beh.received) == 2 and beh.received[0]["max_tokens"] == 40
    assert "Hindi" in beh.received[0]["messages"][0]["content"]


async def test_latency_without_usage_reports_null_throughput():
    items = make_items(2)
    results = make_results(items, ["A", "B"])
    async with MockServer(MockBehaviour(reply=lambda _m: "y" * 20, include_usage=False)) as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            m = by_name(await LatencyEval(n_probes=1).run(make_ctx(items, results, client)))
        finally:
            await client.aclose()
    assert m["decode_tokens_per_s_p50"].value is None
    assert "not measurable" in m["decode_tokens_per_s_p50"].extra["note"]


# ---- resources ----------------------------------------------------------------------


async def test_resources_without_agent_is_null_with_reason():
    items = make_items(1)
    async with MockServer() as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            m = by_name(
                await ResourcesEval().run(make_ctx(items, make_results(items, ["A"]), client))
            )
        finally:
            await client.aclose()
    assert set(m) == {"peak_rss_bytes", "peak_gpu_memory_bytes", "model_file_bytes"}
    assert all(x.value is None and x.extra["measured"] is False for x in m.values())
    assert "resource_agent_url" in m["peak_rss_bytes"].extra["reason"]


async def test_resources_with_agent(tmp_path):
    """Fake agent + fake /props: the eval must pass model_path through and report sizes."""
    import httpx

    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"0" * 1234)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_path": str(model_file)})
        if request.url.path == "/sample":
            seen.update(dict(request.url.params))
            return httpx.Response(
                200,
                json={
                    "pid": 42,
                    "process": "llama-server",
                    "n_samples": 10,
                    "host": {"machine": "arm64"},
                    "peak_rss_bytes": 5_000_000,
                    "peak_gpu_memory_bytes": None,
                    "gpu_reason": "nvidia-smi not available on this host",
                    "model_file_bytes": 1234,
                    "model_files": [{"path": str(model_file), "bytes": 1234}],
                },
            )
        return httpx.Response(404)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    items = make_items(1)
    client = OpenAICompatClient("http://model.test", model="x", client=http)
    ctx = make_ctx(items, make_results(items, ["A"]), client, agent_url="http://agent.test")
    m = by_name(await ResourcesEval(http=http).run(ctx))
    await http.aclose()
    assert m["peak_rss_bytes"].value == 5_000_000 and m["peak_rss_bytes"].extra["measured"] is True
    assert m["peak_rss_bytes"].extra["pid"] == 42 and m["peak_rss_bytes"].extra["since"]
    assert (
        m["peak_gpu_memory_bytes"].value is None
        and "nvidia-smi" in m["peak_gpu_memory_bytes"].extra["reason"]
    )
    assert m["model_file_bytes"].value == 1234 and m["model_file_bytes"].extra["model_path"] == str(
        model_file
    )
    assert seen["model_path"] == str(model_file) and "since" in seen


async def test_resources_agent_unreachable_is_null():
    items = make_items(1)
    client = OpenAICompatClient("http://127.0.0.1:9", model="x")
    try:
        m = by_name(
            await ResourcesEval().run(
                make_ctx(items, make_results(items, ["A"]), client, agent_url="http://127.0.0.1:9")
            )
        )
    finally:
        await client.aclose()
    assert all(x.value is None for x in m.values())
    assert "unreachable" in m["peak_rss_bytes"].extra["reason"]


# ---- tokenizer ----------------------------------------------------------------------


class FakeTokenizer:
    """Splits on characters; 'x' is lost on decode to exercise the round-trip check."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if chr(i) != "x")


async def test_tokenizer_fertility_and_roundtrip():
    items = [
        MCQItem(
            "a", "hi", None, "ab cd", ("e", "f", "g", "h"), "A"
        ),  # 12 chars incl. newlines, 5 words
        MCQItem("b", "hi", None, "xx", ("e", "f", "g", "h"), "A"),  # contains x -> round trip fails
    ]
    loaded = []
    ev = TokenizerEval(loader=lambda repo: (loaded.append(repo), FakeTokenizer())[1])
    m = by_name(await ev.run(make_ctx(items, [], hf_repo="org/tok")))
    assert loaded == ["org/tok"]
    t1, t2 = item_text(items[0]), item_text(items[1])
    n_tokens = len(t1) + len(t2)
    n_words = len(t1.split()) + len(t2.split())
    assert m["fertility"].value == pytest.approx(n_tokens / n_words)
    assert m["tokens_per_char"].value == pytest.approx(1.0)
    assert m["roundtrip_pass_rate"].value == 0.5 and m["roundtrip_pass_rate"].extra["n_pass"] == 1
    assert (
        m["fertility"].language == "hi"
        and m["fertility"].extra["tokenizer_class"] == "FakeTokenizer"
    )


async def test_tokenizer_without_repo_or_with_failing_load_is_null():
    items = make_items(2)
    m = by_name(await TokenizerEval().run(make_ctx(items, [], hf_repo=None)))
    assert all(x.value is None and "hf_repo" in x.extra["reason"] for x in m.values())

    def boom(repo):
        raise OSError("no such repo")

    m = by_name(await TokenizerEval(loader=boom).run(make_ctx(items, [], hf_repo="nope/nope")))
    assert all(x.value is None and "OSError" in x.extra["reason"] for x in m.values())


# ---- reliability --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,ok",
    [
        ('{"city": "Mumbai", "country": "India", "population_estimate": 20000000}', True),
        (
            '```json\n{"city": "Dhaka", "country": "BD", "population_estimate": 23000}\n```',
            True,
        ),
        ('{"city": "Mumbai", "country": "India"}', False),  # missing key
        ('{"city": "Mumbai", "country": "India", "population_estimate": "20 million"}', False),
        ('{"city": "Mumbai", "country": "India", "population_estimate": true}', False),
        ('Sure! {"city": "Mumbai", "country": "India", "population_estimate": 1}', False),
        ("", False),
        (None, False),
    ],
)
def test_json_is_valid(text, ok):
    assert json_is_valid(text) is ok


async def test_reliability_consistency_and_json():
    items = make_items(4)
    results = make_results(items, ["A", None, "C", "D"])  # 3 of 4 valid format
    state = {"n": 0}

    def reply(msg: str) -> str:
        if "JSON object" in msg:
            return json.dumps({"city": "Mumbai", "country": "India", "population_estimate": 1})
        # Item 0 -> always A (unanimous). Other items -> cycle A/B/C (not unanimous).
        if "प्रश्न 0 " in msg:
            return "A"
        state["n"] += 1
        return "ABC"[state["n"] % 3]

    async with MockServer(MockBehaviour(reply=reply)) as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            ev = ReliabilityEval(n_items=4, n_repeats=3, n_json=2)
            m = by_name(await ev.run(make_ctx(items, results, client, concurrency=1)))
        finally:
            await client.aclose()
    assert m["mcq_valid_format_rate"].value == 0.75 and m["mcq_valid_format_rate"].extra["n"] == 4
    assert m["json_valid_rate"].value == 1.0 and m["json_valid_rate"].extra["n"] == 2
    cons = m["consistency_unanimous_rate"]
    assert cons.value == 0.25 and cons.extra["n_items"] == 4 and cons.extra["temperature"] == 0.7
    pair = m["consistency_pairwise_agreement"].value
    assert 0.25 <= pair < 1.0
    temps = {
        r["temperature"] for r in beh_received(srv) if "JSON" not in r["messages"][-1]["content"]
    }
    assert temps == {0.7}


def beh_received(srv: MockServer):
    return srv.behaviour.received
