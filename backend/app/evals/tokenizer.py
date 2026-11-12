"""Dimension 4: tokenizer efficiency.

Uses the model's own tokenizer (AutoTokenizer from `models.hf_repo`) on the exact
question + option text of the evaluated items. Reports fertility (tokens per
whitespace word), tokens per character, and the decode(encode(x)) == x pass rate.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from app.evals.base import EvalContext, Metric
from app.evals.mcq import MCQItem

DIM = "tokenizer"

TokenizerLoader = Callable[[str], Any]


def default_loader(hf_repo: str):
    from transformers import AutoTokenizer  # heavy import, keep it lazy

    return AutoTokenizer.from_pretrained(hf_repo)


def item_text(item: MCQItem) -> str:
    return "\n".join([item.question, *item.options])


class TokenizerEval:
    name = DIM

    def __init__(self, loader: TokenizerLoader = default_loader):
        self._loader = loader

    async def run(self, ctx: EvalContext) -> list[Metric]:
        lang = ctx.language
        repo = ctx.model.hf_repo
        if not repo:
            reason = "model has no hf_repo; cannot load its tokenizer"
            return [
                Metric(DIM, n, None, u, lang, {"measured": False, "reason": reason})
                for n, u in (
                    ("fertility", "tokens/word"),
                    ("tokens_per_char", "tokens/char"),
                    ("roundtrip_pass_rate", "fraction"),
                )
            ]
        try:
            tok = await asyncio.to_thread(self._loader, repo)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the run
            reason = f"tokenizer load failed for {repo}: {type(exc).__name__}: {exc}"[:300]
            return [
                Metric(DIM, n, None, u, lang, {"measured": False, "reason": reason})
                for n, u in (
                    ("fertility", "tokens/word"),
                    ("tokens_per_char", "tokens/char"),
                    ("roundtrip_pass_rate", "fraction"),
                )
            ]

        texts = [item_text(it) for it in ctx.items]

        def measure() -> dict:
            n_tokens = n_words = n_chars = n_rt_ok = 0
            per_text = []
            for t in texts:
                ids = tok.encode(t, add_special_tokens=False)
                words = len(t.split())
                n_tokens += len(ids)
                n_words += words
                n_chars += len(t)
                ok = tok.decode(ids, skip_special_tokens=True) == t
                n_rt_ok += int(ok)
                per_text.append(len(ids) / words if words else None)
            return {
                "n_tokens": n_tokens,
                "n_words": n_words,
                "n_chars": n_chars,
                "n_rt_ok": n_rt_ok,
                "per_text_fertility": [x for x in per_text if x is not None],
            }

        m = await asyncio.to_thread(measure)
        n = len(texts)
        extra = {
            "measured": True,
            "hf_repo": repo,
            "tokenizer_class": type(tok).__name__,
            "n_texts": n,
            "n_tokens": m["n_tokens"],
            "n_words": m["n_words"],
            "n_chars": m["n_chars"],
            "method": "corpus-level: total tokens / total whitespace words over question+options",
        }
        return [
            Metric(
                DIM,
                "fertility",
                (m["n_tokens"] / m["n_words"]) if m["n_words"] else None,
                "tokens/word",
                lang,
                extra,
            ),
            Metric(
                DIM,
                "tokens_per_char",
                (m["n_tokens"] / m["n_chars"]) if m["n_chars"] else None,
                "tokens/char",
                lang,
                extra,
            ),
            Metric(
                DIM,
                "roundtrip_pass_rate",
                (m["n_rt_ok"] / n) if n else None,
                "fraction",
                lang,
                {**extra, "n_pass": m["n_rt_ok"]},
            ),
        ]
