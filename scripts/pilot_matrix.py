#!/usr/bin/env python3
"""Run the quantization pilot end to end: for each GGUF quantization, start
llama-server, register the model, run every language, export CSVs, stop the server.

    python scripts/pilot_matrix.py --quants Q8_0 Q6_K Q4_K_M Q3_K_M --n-items 200 --delete-after

Runs one quantization at a time (a 16 GB laptop cannot hold two 7B models) and the
languages sequentially, so latency numbers are not confounded by a second run.
Stdlib only; must run on the host that runs llama-server.
"""

import argparse
import glob
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http(method: str, url: str, body: dict | None = None, timeout: float = 60) -> dict | None:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else None


def server_alias(server: str) -> str | None:
    try:
        props = http("GET", f"{server}/props", timeout=5)
        return (props or {}).get("model_alias")
    except Exception:  # noqa: BLE001
        return None


def start_server(args, quant: str) -> subprocess.Popen | None:
    tag = f"{args.hf_repo}:{quant}"
    if server_alias(args.server) == tag:
        log(f"reusing already-running server for {tag}")
        return None
    logfile = open(ROOT / "results" / f".llama_{quant}.log", "ab")  # noqa: SIM115 - lives with the process
    cmd = [
        "llama-server",
        "-hf",
        tag,
        "--port",
        str(args.port),
        "--host",
        "127.0.0.1",
        "-c",
        str(args.ctx),
        "-np",
        str(args.slots),
    ]
    log("starting: " + " ".join(cmd))
    return subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT)


def wait_healthy(server: str, timeout_s: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if http("GET", f"{server}/health", timeout=5).get("status") == "ok":  # type: ignore[union-attr]
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(10)
    sys.exit(f"server at {server} not healthy after {timeout_s:.0f}s (download still running?)")


def ensure_model(args, quant: str, build_info: str | None) -> int:
    for m in http("GET", f"{args.api}/models") or []:
        if m["name"] == args.model_name and m["quantization"] == quant:
            return m["id"]
    body = {
        "name": args.model_name,
        "hf_repo": args.tokenizer_repo,
        "backend": "llama_cpp",
        "endpoint_url": args.endpoint,
        "quantization": quant,
        "params_b": args.params_b,
        "resource_agent_url": args.agent or None,
        "notes": (
            f"llama-server {build_info or '?'} on {os.uname().machine}; "
            f"-c {args.ctx} -np {args.slots}; GGUF {args.hf_repo}:{quant}"
        ),
    }
    return http("POST", f"{args.api}/models", body)["id"]  # type: ignore[index]


def run_one(args, model_id: int, dataset: dict) -> dict:
    body = {
        "model_id": model_id,
        "dataset_id": dataset["id"],
        "n_items": args.n_items,
        "seed": args.seed,
        "concurrency": args.concurrency,
    }
    run = http("POST", f"{args.api}/runs", body)
    run_id = run["id"]  # type: ignore[index]
    log(f"run {run_id}: {dataset['name']} ({args.n_items} items)")
    last = None
    while True:
        time.sleep(args.poll)
        r = http("GET", f"{args.api}/runs/{run_id}")
        p = r["progress"]  # type: ignore[index]
        line = (
            f"  run {run_id} {r['status']} {p['n_done']}/{p['n_total']} "  # type: ignore[index]
            f"err={p['n_errors']} acc={r['accuracy']}"  # type: ignore[index]
        )
        if line != last:
            log(line)
            last = line
        if r["status"] in ("completed", "failed", "cancelled"):  # type: ignore[index]
            if r["error"]:  # type: ignore[index]
                log(f"  error: {r['error']}")  # type: ignore[index]
            return r  # type: ignore[return-value]


def export(args, run_id: int, name: str) -> None:
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    for kind in ("metrics", "items"):
        with urllib.request.urlopen(
            f"{args.api}/export/{run_id}.csv?kind={kind}", timeout=120
        ) as resp:
            (out / f"{name}_{kind}.csv").write_bytes(resp.read())
    log(f"  exported results/{name}_*.csv")


def stop_server(proc: subprocess.Popen | None, quant: str, hf_repo: str) -> None:
    subprocess.run(["pkill", "-f", f"llama-server -hf {hf_repo}:{quant}"], check=False)
    if proc is not None:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    time.sleep(2)


def delete_model_files(model_path: str) -> None:
    """Remove the GGUF shards (and the HF-cache blobs they link to) to free disk."""
    m = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", model_path)
    files = sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf")) if m else [model_path]
    freed = 0
    for f in files:
        for target in {os.path.realpath(f), f}:
            if os.path.exists(target) or os.path.islink(target):
                freed += (
                    os.path.getsize(target)
                    if os.path.isfile(target) and not os.path.islink(target)
                    else 0
                )
                os.remove(target)
    log(
        f"  deleted {len(files)} model file(s), freed ~{freed / 1e9:.1f} GB; "
        f"disk free {shutil.disk_usage('/').free / 1e9:.1f} GB"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument(
        "--server", default="http://127.0.0.1:8080", help="llama-server as seen from this host"
    )
    ap.add_argument(
        "--endpoint",
        default="http://host.docker.internal:8080",
        help="llama-server as seen from the API",
    )
    ap.add_argument("--agent", default="http://host.docker.internal:9108")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--hf-repo", default="Qwen/Qwen2.5-7B-Instruct-GGUF")
    ap.add_argument("--tokenizer-repo", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--model-name", default="Qwen2.5-7B-Instruct")
    ap.add_argument("--params-b", type=float, default=7.6)
    ap.add_argument("--quants", nargs="+", default=["Q8_0", "Q6_K", "Q4_K_M", "Q3_K_M"])
    ap.add_argument("--languages", nargs="+", default=["en", "hi", "bn"])
    ap.add_argument("--n-items", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--poll", type=float, default=15)
    ap.add_argument("--download-timeout", type=float, default=3600)
    ap.add_argument("--delete-after", action="store_true", help="delete each GGUF after its runs")
    args = ap.parse_args()

    datasets = {d["language"]: d for d in http("GET", f"{args.api}/datasets") or []}  # type: ignore[union-attr]
    stem = args.model_name.lower().replace("-instruct", "")
    summary = []
    for quant in args.quants:
        log(f"=== {quant} ===")
        proc = start_server(args, quant)
        wait_healthy(args.server, args.download_timeout)
        props = http("GET", f"{args.server}/props") or {}
        model_id = ensure_model(args, quant, props.get("build_info"))
        log(f"model id {model_id}, model_path {props.get('model_path')}")
        for lang in args.languages:
            r = run_one(args, model_id, datasets[lang])
            name = f"pilot_{stem}_{quant.lower()}_{lang}"
            if r["status"] == "completed":
                export(args, r["id"], name)
            summary.append(
                (quant, lang, r["id"], r["status"], r["accuracy"], r["progress"]["n_errors"])
            )
        stop_server(proc, quant, args.hf_repo)
        if args.delete_after and props.get("model_path"):
            delete_model_files(props["model_path"])
    log("=== summary ===")
    for row in summary:
        log("  " + " ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
