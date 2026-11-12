"""Dimension 1: task accuracy on the fixed multiple-choice subset.

Computed from the per-item rows written by the executor (greedy decoding, strict
parsing). Unparseable replies count as wrong; failed requests are excluded from the
denominator and reported separately.
"""

from collections import defaultdict

from app.evals.base import EvalContext, Metric, wilson_interval

DIM = "accuracy"


def _acc_metric(name: str, correct: int, n: int, language: str, **extra) -> Metric:
    ci = wilson_interval(correct, n)
    return Metric(
        DIM,
        name,
        (correct / n) if n else None,
        unit="fraction",
        language=language,
        extra={
            "n": n,
            "n_correct": correct,
            "ci95_low": ci[0] if ci else None,
            "ci95_high": ci[1] if ci else None,
            **extra,
        },
    )


class AccuracyEval:
    name = DIM

    async def run(self, ctx: EvalContext) -> list[Metric]:
        lang = ctx.language
        scored = [r for r in ctx.results if r.error is None]
        n_total = len(ctx.results)
        n_scored = len(scored)
        n_correct = sum(1 for r in scored if r.correct)
        n_unparseable = sum(1 for r in scored if r.parsed_answer is None)
        n_errors = n_total - n_scored

        metrics = [
            _acc_metric("accuracy", n_correct, n_scored, lang, n_total=n_total, n_errors=n_errors),
            Metric(
                DIM,
                "unparseable_rate",
                (n_unparseable / n_scored) if n_scored else None,
                "fraction",
                lang,
                {"n": n_scored, "n_unparseable": n_unparseable},
            ),
            Metric(
                DIM,
                "request_error_rate",
                (n_errors / n_total) if n_total else None,
                "fraction",
                lang,
                {"n": n_total, "n_errors": n_errors},
            ),
        ]

        # Breakdown by subject category (4 groups in Global-MMLU) as separate metrics,
        # and the full per-subject table in one metric's extra for the dashboard.
        by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        by_subj: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for r in scored:
            cat = (r.extra_json or {}).get("subject_category") or "unknown"
            by_cat[cat][0] += int(bool(r.correct))
            by_cat[cat][1] += 1
            subj = r.subject or "unknown"
            by_subj[subj][0] += int(bool(r.correct))
            by_subj[subj][1] += 1
        for cat, (c, n) in sorted(by_cat.items()):
            metrics.append(_acc_metric(f"accuracy_by_category:{cat}", c, n, lang, category=cat))
        metrics.append(
            Metric(
                DIM,
                "accuracy_by_subject",
                None,
                "table",
                lang,
                {
                    "note": "value is null on purpose; see rows. n per subject is small.",
                    "rows": [
                        {"subject": s, "n": n, "n_correct": c, "accuracy": c / n}
                        for s, (c, n) in sorted(by_subj.items())
                    ],
                },
            )
        )
        # Answer distribution: exposes position bias (e.g. a model that always says A).
        dist = defaultdict(int)
        for r in scored:
            dist[r.parsed_answer or "unparseable"] += 1
        metrics.append(
            Metric(
                DIM,
                "answer_distribution",
                None,
                "table",
                lang,
                {"counts": dict(dist), "n": n_scored},
            )
        )
        return metrics
