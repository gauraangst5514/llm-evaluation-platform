#!/usr/bin/env python3
"""Export a finished run's metrics and per-item CSVs into results/.

    python scripts/export_results.py --run 3 --name smoke_qwen2.5-0.5b_q8_0_hi

Writes results/<name>_metrics.csv and results/<name>_items.csv straight from the
API's /export endpoint, so what is committed is exactly what the platform stored.
"""

import argparse
import pathlib
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--name", required=True, help="file stem, e.g. pilot_qwen2.5-7b_q4_k_m_hi")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for kind in ("metrics", "items"):
        url = f"{args.api}/export/{args.run}.csv?kind={kind}"
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"GET {url} failed: {exc}")
        path = out / f"{args.name}_{kind}.csv"
        path.write_bytes(data)
        print(f"wrote {path} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
