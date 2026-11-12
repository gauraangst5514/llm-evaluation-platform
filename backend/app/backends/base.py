"""Backend-agnostic chat interface. Every model server speaks this."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatResult:
    text: str
    ttft_ms: float | None  # time to first content token; None if no token arrived
    latency_ms: float  # request start -> stream end
    # From the server's usage block when it reports one; None otherwise. Never estimated.
    output_tokens: int | None
    prompt_tokens: int | None
    finish_reason: str | None
    n_chunks: int  # content chunks received; a rough, honest fallback signal
    raw_usage: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def generation_ms(self) -> float | None:
        """Time spent after the first token, i.e. decode time."""
        if self.ttft_ms is None:
            return None
        return max(self.latency_ms - self.ttft_ms, 0.0)

    @property
    def tokens_per_second(self) -> float | None:
        """Decode throughput from server-reported tokens; None when not measurable."""
        gen = self.generation_ms
        if self.output_tokens is None or gen is None or gen <= 0 or self.output_tokens < 2:
            return None
        # (n-1) intervals happen after the first token.
        return (self.output_tokens - 1) / (gen / 1000.0)


class ChatBackend(Protocol):
    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 16,
        seed: int | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult: ...

    async def ping(self) -> dict: ...
