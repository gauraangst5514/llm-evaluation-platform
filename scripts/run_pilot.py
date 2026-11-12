#!/usr/bin/env python3
"""Launch an evaluation run from the command line and wait for it.

Examples:
  python scripts/run_pilot.py --model-id 3 --dataset global_mmlu_hi --n-items 50
  python scripts/run_pilot.py --model-id 3 --dataset global_mmlu_en --dataset global_mmlu_hi --n-items 200

Only uses the HTTP API (stdlib only, no extra deps) so it works from any machine.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(base: str, method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{base}{path}", data=data, method=method, headers={"content-type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read() or b"null")
    except urllib.error.HTTPError as exc:
        sys.exit(f"{method} {path} -> {exc.code}: {exc.read().decode(errors='replace')}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--model-id", type=int, required=True)
    ap.add_argument("--dataset", action="append", required=True, help="dataset name; repeatable")
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--dimensions", nargs="*", default=None)
    ap.add_argument("--poll", type=float, default=5.0, help="seconds between status polls")
    args = ap.parse_args()

    datasets = {d["name"]: d for d in call(args.api, "GET", "/datasets")}
    for name in args.dataset:
        if name not in datasets:
            sys.exit(f"Unknown dataset {name!r}. Known: {', '.join(datasets)}")

    for name in args.dataset:
        body = {
            "model_id": args.model_id,
            "dataset_id": datasets[name]["id"],
            "n_items": args.n_items,
            "seed": args.seed,
            "concurrency": args.concurrency,
        }
        if args.dimensions:
            body["dimensions"] = args.dimensions
        run = call(args.api, "POST", "/runs", body)
        run_id = run["id"]
        print(f"run {run_id}: {name}, {args.n_items} items, concurrency {args.concurrency}")
        while True:
            time.sleep(args.poll)
            r = call(args.api, "GET", f"/runs/{run_id}")
            p = r["progress"]
            acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "n/a"
            print(f"  {r['status']:<10} {p['n_done']}/{p['n_total']} done, "
                  f"{p['n_errors']} errors, accuracy so far {acc}", flush=True)
            if r["status"] in ("completed", "failed", "cancelled"):
                if r["error"]:
                    print(f"  error: {r['error']}")
                break


if __name__ == "__main__":
    main()
