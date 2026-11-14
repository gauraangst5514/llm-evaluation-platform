#!/usr/bin/env python3
"""Build the README results table from committed metrics exports.

    python scripts/results_table.py results/*_metrics.csv

Every number printed comes from a *_metrics.csv produced by scripts/export_results.py,
so the README never carries a figure the platform did not store.
"""

import csv
import json
import sys
from collections import OrderedDict

COLUMNS = [
    ("accuracy", "Accuracy", "pct_ci"),
    ("ttft_ms_p50", "TTFT p50", "ms"),
    ("decode_tokens_per_s_p50", "Decode tok/s p50", "num1"),
    ("fertility", "Fertility (tok/word)", "num2"),
    ("json_valid_rate", "JSON valid", "pct"),
    ("consistency_unanimous_rate", "Consistency@0.7", "pct"),
    ("peak_rss_bytes", "Peak RSS", "mb"),
]


def fmt(kind: str, value: str, extra: dict) -> str:
    if value in ("", "None"):
        return "n/a"
    v = float(value)
    if kind == "pct_ci":
        lo, hi, n = extra.get("ci95_low"), extra.get("ci95_high"), extra.get("n")
        ci = f" ({lo * 100:.0f}–{hi * 100:.0f}%)" if lo is not None and hi is not None else ""
        return f"{v * 100:.1f}%{ci}, n={n}"
    if kind == "pct":
        return f"{v * 100:.0f}% (n={extra.get('n') or extra.get('n_items')})"
    if kind == "ms":
        return f"{v:.0f} ms"
    if kind == "num1":
        return f"{v:.1f}"
    if kind == "num2":
        return f"{v:.2f}"
    if kind == "mb":
        return f"{v / 1e6:.0f} MB"
    return value


def main(paths: list[str]) -> None:
    rows: OrderedDict[tuple, dict] = OrderedDict()
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                # Resource metrics carry no language; the dataset name ends with the language code.
                lang = r["language"] or r["dataset"].rsplit("_", 1)[-1]
                key = (r["model"], r["quantization"], lang, r["run_id"])
                rows.setdefault(key, {})[r["name"]] = (
                    r["value"],
                    json.loads(r["extra_json"] or "{}"),
                )
    header = ["Model", "Quant", "Lang", "Run"] + [c[1] for c in COLUMNS]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    # Run order is execution order (the pilot runs quantizations high to low, en/hi/bn).
    ordered = sorted(rows.items(), key=lambda kv: int(kv[0][3]) if kv[0][3].isdigit() else 0)
    for (model, quant, lang, run_id), metrics in ordered:
        cells = []
        for name, _title, kind in COLUMNS:
            value, extra = metrics.get(name, ("", {}))
            cells.append(fmt(kind, value, extra))
        print(f"| {model} | {quant} | {lang} | {run_id} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
