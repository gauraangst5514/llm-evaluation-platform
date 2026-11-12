"""A tiny OpenAI-compatible streaming server for tests.

Runs on a real socket (uvicorn in-process) so TTFT and streaming behaviour are
measured for real, not through an ASGI buffer. Behaviour is controlled per test
through the `MockBehaviour` object.
"""

import asyncio
import json
import socket
from collections.abc import Callable
from dataclasses import dataclass, field

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse


@dataclass
class MockBehaviour:
    # Given the user message text, return the reply text.
    reply: Callable[[str], str] = lambda _msg: "B"
    chunk_size: int = 1  # characters per SSE chunk
    first_token_delay_s: float = 0.05
    inter_chunk_delay_s: float = 0.0
    include_usage: bool = True
    fail_first_n_requests: int = 0  # respond 503 to the first n requests
    fail_status: int = 503
    hang_requests: set[int] = field(default_factory=set)  # request numbers that never answer
    requests_seen: int = 0
    received: list[dict] = field(default_factory=list)


def make_app(behaviour: MockBehaviour) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": "mock-model", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        behaviour.requests_seen += 1
        n = behaviour.requests_seen
        behaviour.received.append(body)
        if n <= behaviour.fail_first_n_requests:
            return JSONResponse({"error": "overloaded"}, status_code=behaviour.fail_status)
        if n in behaviour.hang_requests:
            await asyncio.sleep(3600)
        user_msg = next(
            (m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), ""
        )
        text = behaviour.reply(user_msg)
        max_tokens = int(body.get("max_tokens", 16))
        chunks = [
            text[i : i + behaviour.chunk_size] for i in range(0, len(text), behaviour.chunk_size)
        ]
        chunks = chunks[:max_tokens]
        prompt_tokens = sum(len(m["content"].split()) for m in body["messages"])

        async def gen():
            def event(delta: dict, finish: str | None = None, usage: dict | None = None) -> str:
                payload = {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion.chunk",
                    "model": "mock-model",
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                if usage is not None:
                    payload["usage"] = usage
                return f"data: {json.dumps(payload)}\n\n"

            yield event({"role": "assistant"})
            await asyncio.sleep(behaviour.first_token_delay_s)
            for i, c in enumerate(chunks):
                if i:
                    await asyncio.sleep(behaviour.inter_chunk_delay_s)
                yield event({"content": c})
            finish = "length" if len(chunks) < len(text) or len(chunks) == max_tokens else "stop"
            usage = None
            if behaviour.include_usage and body.get("stream_options", {}).get("include_usage"):
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(chunks),
                    "total_tokens": prompt_tokens + len(chunks),
                }
            yield event({}, finish=finish, usage=usage)
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockServer:
    def __init__(self, behaviour: MockBehaviour | None = None):
        self.behaviour = behaviour or MockBehaviour()
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._server: uvicorn.Server | None = None
        self._task: asyncio.Task | None = None

    async def __aenter__(self) -> "MockServer":
        config = uvicorn.Config(
            make_app(self.behaviour),
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            lifespan="off",
        )
        self._server = uvicorn.Server(config)
        self._task = asyncio.create_task(self._server.serve())
        while not self._server.started:
            await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *exc) -> None:
        assert self._server and self._task
        self._server.should_exit = True
        # Hanging handlers would block graceful shutdown; force it.
        self._server.force_exit = True
        try:
            await asyncio.wait_for(self._task, timeout=5)
        except (TimeoutError, asyncio.CancelledError):
            self._task.cancel()
