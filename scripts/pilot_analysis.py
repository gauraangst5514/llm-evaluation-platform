#!/usr/bin/env python3
"""Analyse pilot exports: accuracy per quantization and language with 95% CIs, the
paired drop versus the highest-precision quantization, and fertility.

    python scripts/pilot_analysis.py results/pilot_qwen2.5-7b

Reads <prefix>_<quant>_<lang>_{metrics,items}.csv written by scripts/pilot_matrix.py.
Because every quantization answered the *same* items, drops are computed on paired
per-item correctness (discordant pairs), which is tighter than comparing two
independent intervals. Also writes <prefix>.csv: every metric row of every cell in
one long-format file (the spec's results/pilot_<model>.csv). Stdlib only.
"""

import csv
import glob
import json
import math
import re
import sys

QUANT_RANK = {
    "Q2_K": 2.6, "Q3_K_M": 3.7, "Q4_0": 4.5, "Q4_K_M": 4.8,
    "Q5_K_M": 5.7, "Q6_K": 6.6, "Q8_0": 8.5, "FP16": 16,
}  # fmt: skip
LANG = {"en": "English", "hi": "Hindi", "bn": "Bengali"}
Metrics = dict[str, tuple[float | None, dict]]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def paired_diff(
    base: dict[str, bool], other: dict[str, bool]
) -> tuple[float, float, float, int, int, int]:
    """accuracy(base) - accuracy(other) on shared items, 95% CI from discordant pairs."""
    ids = [i for i in base if i in other]
    n = len(ids)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan, 0, 0, 0
    b = sum(1 for i in ids if base[i] and not other[i])  # base right, other wrong
    c = sum(1 for i in ids if other[i] and not base[i])  # other right, base wrong
    d = (b - c) / n
    var = (b + c - (b - c) ** 2 / n) / (n * n)
    half = 1.96 * math.sqrt(max(var, 0.0))
    return d, d - half, d + half, b, c, n


def load(
    prefix: str,
) -> tuple[dict[tuple[str, str], Metrics], dict[tuple[str, str], dict[str, bool]]]:
    metrics: dict[tuple[str, str], Metrics] = {}
    items: dict[tuple[str, str], dict[str, bool]] = {}
    pat = re.compile(re.escape(prefix) + r"_([a-z0-9_]+)_([a-z]{2})_(metrics|items)\.csv$")
    for path in sorted(glob.glob(f"{prefix}_*_*.csv")):
        m = pat.search(path)
        if not m:
            continue
        quant, lang, kind = m.group(1).upper(), m.group(2), m.group(3)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if kind == "metrics":
            metrics[(quant, lang)] = {
                r["name"]: (
                    float(r["value"]) if r["value"] not in ("", "None") else None,
                    json.loads(r["extra_json"] or "{}"),
                )
                for r in rows
            }
        else:
            items[(quant, lang)] = {
                r["item_id"]: r["correct"] == "True" for r in rows if r["error"] in ("", "None")
            }
    return metrics, items


def pct(x: float | None) -> str:
    return "n/a" if x is None or math.isnan(x) else f"{100 * x:.1f}%"


def value(m: Metrics, name: str) -> float | None:
    return m.get(name, (None, {}))[0]


def table(header: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print()


def main(prefix: str) -> None:
    metrics, items = load(prefix)
    if not metrics:
        sys.exit(f"no files matching {prefix}_*_*_metrics.csv")
    quants = sorted({q for q, _ in metrics}, key=lambda q: -QUANT_RANK.get(q, 0))
    langs = [code for code in ("en", "hi", "bn") if any(lang == code for _, lang in metrics)]
    names = [LANG.get(code, code) for code in langs]
    base = quants[0]

    print("## Accuracy by quantization and language (95% Wilson CI, n scored items)\n")
    rows = []
    for q in quants:
        cells = []
        for lang in langs:
            m = metrics.get((q, lang))
            acc = value(m, "accuracy") if m else None
            if m is None or acc is None:
                cells.append("—")
                continue
            x = m["accuracy"][1]
            lo, hi, n = x["ci95_low"], x["ci95_high"], x["n"]
            cells.append(f"{100 * acc:.1f}% ({100 * lo:.0f}–{100 * hi:.0f}%), n={n}")
        rows.append([q, *cells])
    table(["Quant", *names], rows)

    print(f"## Paired accuracy drop versus {base} (percentage points, same items)\n")
    print(
        f"Positive = quantized model worse. b/c = items only {base} got right / "
        "items only the quantized model got right.\n"
    )
    drops: dict[tuple[str, str], float] = {}
    rows = []
    for q in quants[1:]:
        cells = []
        for lang in langs:
            if (base, lang) not in items or (q, lang) not in items:
                cells.append("—")
                continue
            d, lo, hi, b, c, n = paired_diff(items[(base, lang)], items[(q, lang)])
            drops[(q, lang)] = d
            cells.append(
                f"{100 * d:+.1f} pp ({100 * lo:+.0f} to {100 * hi:+.0f}), b/c={b}/{c}, n={n}"
            )
        rows.append([q, *cells])
    table(["Quant", *names], rows)

    print("## Tokenizer fertility, latency and resources\n")
    rows = []
    for q in quants:
        for lang in langs:
            m = metrics.get((q, lang))
            if not m or value(m, "fertility") is None:
                continue
            tps = value(m, "decode_tokens_per_s_p50")
            rss, fsz = value(m, "peak_rss_bytes"), value(m, "model_file_bytes")
            rows.append([
                q, lang,
                f"{value(m, 'fertility'):.2f}",
                f"{value(m, 'tokens_per_char'):.3f}",
                f"{value(m, 'ttft_ms_p50'):.0f} ms",
                f"{value(m, 'latency_ms_p50'):.0f} ms",
                "n/a" if tps is None else f"{tps:.1f}",
                "n/a" if rss is None else f"{rss / 1e9:.2f} GB",
                "n/a" if fsz is None else f"{fsz / 1e9:.2f} GB",
                pct(value(m, "unparseable_rate")),
                pct(value(m, "request_error_rate")),
            ])  # fmt: skip
    table(
        ["Quant", "Lang", "Fertility", "Tok/char", "TTFT p50", "Latency p50",
         "Decode tok/s", "Peak RSS", "Model file", "Unparseable", "Req. errors"],
        rows,
    )  # fmt: skip

    if drops:
        print("## Fertility versus paired drop, per language\n")
        rows = []
        for lang in langs:
            ds = [drops[(q, lang)] for q in quants[1:] if (q, lang) in drops]
            fert = value(metrics.get((base, lang), {}), "fertility")
            if ds and fert is not None:
                rows.append([
                    LANG.get(lang, lang), f"{fert:.2f}",
                    f"{100 * sum(ds) / len(ds):+.1f} pp", f"{100 * max(ds):+.1f} pp",
                ])  # fmt: skip
        table(["Language", f"Fertility ({base})", "Mean drop", "Largest drop"], rows)

    print("## Reliability and answer-letter distribution\n")
    rows = []
    for q in quants:
        for lang in langs:
            m = metrics.get((q, lang))
            if not m:
                continue
            dist = m.get("answer_distribution", (None, {}))[1].get("counts", {})
            rows.append([
                q, lang,
                pct(value(m, "json_valid_rate")),
                pct(value(m, "consistency_unanimous_rate")),
                pct(value(m, "consistency_pairwise_agreement")),
                ", ".join(f"{k}:{v}" for k, v in sorted(dist.items())),
            ])  # fmt: skip
    table(
        ["Quant", "Lang", "JSON valid", "Unanimous @0.7", "Pairwise @0.7", "Answer letters"], rows
    )


def write_combined(prefix: str) -> str:
    """Concatenate the per-cell metrics exports, ordered by quant precision and language."""
    order = {"en": 0, "hi": 1, "bn": 2}
    paths = []
    for path in glob.glob(f"{prefix}_*_metrics.csv"):
        m = re.match(re.escape(prefix) + r"_(.+)_([a-z]{2})_metrics\.csv$", path)
        if m:
            quant = m.group(1).upper()
            paths.append((-QUANT_RANK.get(quant, 0), order.get(m.group(2), 9), path))
    out = f"{prefix}.csv"
    header: list[str] | None = None
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        for *_, path in sorted(paths):
            with open(path, newline="", encoding="utf-8") as src:
                reader = csv.reader(src)
                head = next(reader)
                if header is None:
                    header = head
                    writer.writerow(header)
                elif head != header:
                    sys.exit(f"{path}: header differs from the other exports")
                writer.writerows(reader)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
    print(f"\nwrote {write_combined(sys.argv[1])}", file=sys.stderr)
