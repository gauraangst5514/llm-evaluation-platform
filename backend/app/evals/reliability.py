"""Dimension 5: output reliability.

(a) Valid-format rate: share of MCQ replies that parsed as a single option letter
    (from the accuracy pass), plus a JSON probe: a few prompts that ask for a JSON
    object with fixed keys, scored by strict json.loads + key check.
(b) Consistency: the first K items are re-run 3 times at temperature 0.7; agreement
    is the share of items whose 3 parsed answers are identical. Mean pairwise
    agreement is recorded too.
"""

import asyncio
import itertools
import json
import re

from app.backends.base import ChatMessage
from app.backends.openai_compat import PermanentBackendError, TransientBackendError
from app.evals.base import EvalContext, Metric
from app.evals.mcq import build_messages, parse_answer

DIM = "reliability"
N_CONSISTENCY_ITEMS = 10
N_REPEATS = 3
CONSISTENCY_TEMPERATURE = 0.7
JSON_KEYS = ("city", "country", "population_estimate")
JSON_PROBES = ["Mumbai", "Dhaka", "Chennai", "Kolkata", "Delhi"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def json_probe_messages(city: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            "user",
            'Return only a JSON object with exactly these keys: "city" (string), '
            f'"country" (string), "population_estimate" (integer). The city is {city}. '
            "No prose, no markdown.",
        )
    ]


def json_is_valid(text: str | None) -> bool:
    """Strict: parses as JSON (an optional code fence is stripped) with exactly the keys."""
    if not text:
        return False
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict) or set(obj) != set(JSON_KEYS):
        return False
    return (
        isinstance(obj["city"], str)
        and isinstance(obj["country"], str)
        and isinstance(obj["population_estimate"], int)
        and not isinstance(obj["population_estimate"], bool)
    )


class ReliabilityEval:
    name = DIM

    def __init__(
        self,
        n_items: int = N_CONSISTENCY_ITEMS,
        n_repeats: int = N_REPEATS,
        n_json: int = len(JSON_PROBES),
    ):
        self.n_items = n_items
        self.n_repeats = n_repeats
        self.n_json = n_json

    async def run(self, ctx: EvalContext) -> list[Metric]:
        lang = ctx.language
        scored = [r for r in ctx.results if r.error is None]
        n_valid = sum(1 for r in scored if r.parsed_answer is not None)
        metrics = [
            Metric(
                DIM,
                "mcq_valid_format_rate",
                (n_valid / len(scored)) if scored else None,
                "fraction",
                lang,
                {"n": len(scored), "n_valid": n_valid, "source": "accuracy pass, temperature 0"},
            )
        ]

        # JSON probe
        json_ok = 0
        json_errors = 0
        samples = []
        for city in JSON_PROBES[: self.n_json]:
            try:
                res = await ctx.client.chat(
                    json_probe_messages(city), temperature=0.0, max_tokens=96, seed=ctx.seed
                )
            except (TransientBackendError, PermanentBackendError):
                json_errors += 1
                continue
            ok = json_is_valid(res.text)
            json_ok += int(ok)
            samples.append({"city": city, "valid": ok, "response": res.text[:200]})
        n_json = self.n_json - json_errors
        metrics.append(
            Metric(
                DIM,
                "json_valid_rate",
                (json_ok / n_json) if n_json else None,
                "fraction",
                lang,
                {
                    "n": n_json,
                    "n_valid": json_ok,
                    "n_errors": json_errors,
                    "keys": list(JSON_KEYS),
                    "samples": samples,
                },
            )
        )

        # Consistency at temperature 0.7
        items = ctx.items[: self.n_items]
        sem = asyncio.Semaphore(ctx.concurrency)

        async def ask(item, rep: int):
            async with sem:
                try:
                    res = await ctx.client.chat(
                        build_messages(item),
                        temperature=CONSISTENCY_TEMPERATURE,
                        max_tokens=16,
                        seed=ctx.seed
                        + rep,  # different seed per repeat, else sampling may be identical
                    )
                except (TransientBackendError, PermanentBackendError):
                    return None
                return parse_answer(res.text) or "unparseable"

        answers = await asyncio.gather(
            *[ask(it, rep) for it in items for rep in range(self.n_repeats)]
        )
        per_item = [
            answers[i * self.n_repeats : (i + 1) * self.n_repeats] for i in range(len(items))
        ]
        complete = [a for a in per_item if all(x is not None for x in a)]
        unanimous = sum(1 for a in complete if len(set(a)) == 1)
        pair_agree = []
        for a in complete:
            pairs = list(itertools.combinations(a, 2))
            pair_agree.append(sum(1 for x, y in pairs if x == y) / len(pairs))
        n = len(complete)
        extra = {
            "n_items": n,
            "n_items_requested": len(items),
            "n_repeats": self.n_repeats,
            "temperature": CONSISTENCY_TEMPERATURE,
            "n_request_errors": sum(1 for a in per_item for x in a if x is None),
            "mean_pairwise_agreement": (sum(pair_agree) / n) if n else None,
        }
        metrics.append(
            Metric(
                DIM,
                "consistency_unanimous_rate",
                (unanimous / n) if n else None,
                "fraction",
                lang,
                extra,
            )
        )
        metrics.append(
            Metric(
                DIM,
                "consistency_pairwise_agreement",
                extra["mean_pairwise_agreement"],
                "fraction",
                lang,
                extra,
            )
        )
        return metrics
