"""Streaming client for any OpenAI-compatible /v1/chat/completions server
(llama.cpp llama-server, vLLM, ...). Measures TTFT from the stream itself."""

import json
import time

import httpx

from app.backends.base import ChatMessage, ChatResult


class TransientBackendError(Exception):
    """Retry-once class of failure: connection problems, timeouts, 5xx, 429."""


class PermanentBackendError(Exception):
    """4xx other than 429, malformed streams, etc. Retrying will not help."""


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        default_timeout_s: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        # llama-server ignores the model name; vLLM needs it. Fall back to the served id.
        self.model = model
        self.api_key = api_key
        self.default_timeout_s = default_timeout_s
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def ping(self) -> dict:
        r = await self._client.get(
            f"{self.base_url}/v1/models", headers=self._headers(), timeout=10.0
        )
        r.raise_for_status()
        return r.json()

    async def resolve_model(self) -> str:
        """Return the configured model id, or the first one the server lists."""
        if self.model:
            return self.model
        data = await self.ping()
        models = data.get("data") or []
        if not models:
            raise PermanentBackendError("Server lists no models at /v1/models")
        self.model = models[0]["id"]
        return self.model

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        seed: int | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        model = await self.resolve_model()
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            payload["seed"] = seed
        timeout = httpx.Timeout(timeout_s or self.default_timeout_s)

        text_parts: list[str] = []
        ttft_ms: float | None = None
        n_chunks = 0
        finish_reason: str | None = None
        usage: dict | None = None
        start = time.perf_counter()
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=self._headers(),
                timeout=timeout,
            ) as resp:
                if resp.status_code >= 500 or resp.status_code == 429:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    raise TransientBackendError(f"HTTP {resp.status_code}: {body}")
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")[:500]
                    raise PermanentBackendError(f"HTTP {resp.status_code}: {body}")
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise PermanentBackendError(f"Bad SSE JSON: {data[:200]}") from exc
                    if event.get("usage"):
                        usage = event["usage"]
                    for choice in event.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = delta.get("content")
                        if content:
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start) * 1000.0
                            text_parts.append(content)
                            n_chunks += 1
                        if choice.get("finish_reason"):
                            finish_reason = choice["finish_reason"]
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise TransientBackendError(f"{type(exc).__name__}: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        return ChatResult(
            text="".join(text_parts),
            ttft_ms=ttft_ms,
            latency_ms=latency_ms,
            output_tokens=(usage or {}).get("completion_tokens"),
            prompt_tokens=(usage or {}).get("prompt_tokens"),
            finish_reason=finish_reason,
            n_chunks=n_chunks,
            raw_usage=usage,
        )
