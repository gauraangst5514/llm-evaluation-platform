"""Dimension 2: latency and throughput.

TTFT and total latency come from the accuracy pass (streamed, at the run's
concurrency). Decode throughput needs longer generations than a one-letter answer,
so this dimension additionally sends a few open-ended probe prompts at concurrency 1
with max_tokens=128 and measures tokens/s from the server-reported token counts.
"""

from app.backends.base import ChatMessage
from app.backends.openai_compat import PermanentBackendError, TransientBackendError
from app.evals.base import EvalContext, Metric, percentile

DIM = "latency"
PROBE_MAX_TOKENS = 128
N_PROBES = 5

# Same probe topics for every language; the instruction asks for the target language.
PROBE_TOPICS = [
    "the water cycle",
    "how bicycles work",
    "a market in the morning",
    "why we sleep",
    "rivers",
]
LANG_NAME = {"en": "English", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu"}


def probe_messages(topic: str, language: str) -> list[ChatMessage]:
    lang_name = LANG_NAME.get(language, language)
    return [
        ChatMessage("user", f"Write one detailed paragraph in {lang_name} about {topic}."),
    ]


def _dist(name: str, values: list[float], unit: str, lang: str, **extra) -> list[Metric]:
    base = {"n": len(values), **extra}
    return [
        Metric(DIM, f"{name}_p50", percentile(values, 50), unit, lang, base),
        Metric(DIM, f"{name}_p95", percentile(values, 95), unit, lang, base),
    ]


class LatencyEval:
    name = DIM

    def __init__(self, n_probes: int = N_PROBES, probe_max_tokens: int = PROBE_MAX_TOKENS):
        self.n_probes = n_probes
        self.probe_max_tokens = probe_max_tokens

    async def run(self, ctx: EvalContext) -> list[Metric]:
        lang = ctx.language
        ok = [r for r in ctx.results if r.error is None]
        ttft = [r.ttft_ms for r in ok if r.ttft_ms is not None]
        lat = [r.latency_ms for r in ok if r.latency_ms is not None]
        prefill = [
            (r.extra_json["prompt_tokens"] / (r.ttft_ms / 1000.0))
            for r in ok
            if r.ttft_ms and (r.extra_json or {}).get("prompt_tokens")
        ]
        ctx_note = {"source": "mcq_items", "concurrency": ctx.concurrency}
        metrics = _dist("ttft_ms", ttft, "ms", lang, **ctx_note)
        metrics += _dist("latency_ms", lat, "ms", lang, **ctx_note)
        metrics += _dist(
            "prefill_tokens_per_s",
            prefill,
            "tokens/s",
            lang,
            **ctx_note,
            note="prompt_tokens / TTFT; includes queueing at the run's concurrency",
        )

        # Decode throughput probes, sequential so the number is per-stream speed.
        tps: list[float] = []
        probe_errors: list[str] = []
        out_tokens: list[int] = []
        for topic in PROBE_TOPICS[: self.n_probes]:
            try:
                res = await ctx.client.chat(
                    probe_messages(topic, lang),
                    temperature=0.0,
                    max_tokens=self.probe_max_tokens,
                    seed=ctx.seed,
                )
            except (TransientBackendError, PermanentBackendError) as exc:
                probe_errors.append(str(exc)[:200])
                continue
            if res.output_tokens is not None:
                out_tokens.append(res.output_tokens)
            if res.tokens_per_second is not None:
                tps.append(res.tokens_per_second)
        probe_note = {
            "source": "generation_probes",
            "concurrency": 1,
            "max_tokens": self.probe_max_tokens,
            "n_requested": min(self.n_probes, len(PROBE_TOPICS)),
            "n_errors": len(probe_errors),
            "errors": probe_errors,
            "output_tokens": out_tokens,
        }
        if not tps and out_tokens:
            probe_note["note"] = "server reported tokens but generations were too short to time"
        elif not tps:
            probe_note["note"] = "no server-reported token counts; throughput not measurable"
        metrics += _dist("decode_tokens_per_s", tps, "tokens/s", lang, **probe_note)
        return metrics
