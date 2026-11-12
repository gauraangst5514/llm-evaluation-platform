import pytest

from app.backends.base import ChatMessage
from app.backends.openai_compat import (
    OpenAICompatClient,
    PermanentBackendError,
    TransientBackendError,
)
from tests.mock_openai import MockBehaviour, MockServer

MSGS = [ChatMessage("system", "s"), ChatMessage("user", "What is 2+2? A. 3 B. 4")]


async def test_streams_and_measures_ttft():
    beh = MockBehaviour(
        reply=lambda _m: "B. Four", first_token_delay_s=0.2, inter_chunk_delay_s=0.02
    )
    async with MockServer(beh) as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            res = await client.chat(MSGS, max_tokens=16, seed=1)
        finally:
            await client.aclose()
    assert res.text == "B. Four"
    assert res.ttft_ms is not None and 150 <= res.ttft_ms < 1500
    # 6 inter-chunk gaps of 20ms: decode time must be visibly larger than zero
    assert res.latency_ms - res.ttft_ms >= 100
    assert res.output_tokens == 7 and res.prompt_tokens is not None
    assert res.finish_reason == "stop" and res.n_chunks == 7
    assert res.tokens_per_second and res.tokens_per_second > 0
    sent = beh.received[0]
    assert sent["stream"] is True and sent["temperature"] == 0.0 and sent["seed"] == 1
    assert sent["model"] == "mock-model"  # resolved from /v1/models


async def test_usage_absent_means_tokens_none_not_guessed():
    async with MockServer(MockBehaviour(include_usage=False)) as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            res = await client.chat(MSGS)
        finally:
            await client.aclose()
    assert res.text == "B"
    assert res.output_tokens is None and res.tokens_per_second is None
    assert res.n_chunks == 1


async def test_5xx_is_transient_and_4xx_is_permanent():
    async with MockServer(MockBehaviour(fail_first_n_requests=1)) as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            with pytest.raises(TransientBackendError):
                await client.chat(MSGS)
            res = await client.chat(MSGS)  # second request succeeds
            assert res.text == "B"
        finally:
            await client.aclose()
    async with MockServer(MockBehaviour(fail_first_n_requests=1, fail_status=400)) as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            with pytest.raises(PermanentBackendError):
                await client.chat(MSGS)
        finally:
            await client.aclose()


async def test_timeout_is_transient():
    async with MockServer(MockBehaviour(hang_requests={1})) as srv:
        client = OpenAICompatClient(srv.base_url)
        try:
            with pytest.raises(TransientBackendError):
                await client.chat(MSGS, timeout_s=0.3)
        finally:
            await client.aclose()


async def test_connection_refused_is_transient():
    client = OpenAICompatClient("http://127.0.0.1:9", model="x")
    try:
        with pytest.raises(TransientBackendError):
            await client.chat(MSGS, timeout_s=2)
    finally:
        await client.aclose()
