#!/usr/bin/env python3
"""Host-side resource agent for the resources dimension.

Run it on the machine that runs the model server (needs only psutil):

    uv run --with psutil scripts/resource_agent.py --process llama-server --port 9108
    # or: pip install psutil && python scripts/resource_agent.py ...

Then set the model's resource_agent_url (http://host.docker.internal:9108 when the
API runs in Docker on the same Mac). It samples the server process's RSS every
0.5 s and keeps an hour of samples, so the platform can ask for the peak since a
run started. GPU memory is read from nvidia-smi when that command exists.
Everything it cannot measure is reported as null with a reason.
"""

import argparse
import glob
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import psutil

SAMPLE_INTERVAL_S = 0.5
HISTORY_S = 3600


def find_process(name_substring: str) -> psutil.Process | None:
    """Newest process whose name or cmdline contains the substring, ignoring this agent
    and its launcher (whose cmdline also contains the substring as an argument)."""
    me = psutil.Process(os.getpid())
    skip = {me.pid}
    try:
        skip.add(me.ppid())
    except psutil.Error:
        pass
    best = None
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.pid in skip:
                continue
            cmd = " ".join(p.info["cmdline"] or [])
            if "resource_agent" in cmd:
                continue
            matches = name_substring in (p.info["name"] or "") or name_substring in cmd
            if matches and (best is None or p.create_time() > best.create_time()):
                best = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return best


def gpu_memory_bytes() -> int | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            timeout=5,
            text=True,
        )
        return sum(int(x) for x in out.split()) * 1024 * 1024
    except (subprocess.SubprocessError, ValueError):
        return None


def model_files(model_path: str) -> list[str]:
    """The given file plus sibling shards of a split GGUF (…-00001-of-00003.gguf)."""
    m = re.match(r"^(.*)-\d{5}-of-(\d{5})\.gguf$", model_path)
    if m:
        files = sorted(glob.glob(f"{m.group(1)}-*-of-{m.group(2)}.gguf"))
        return files or [model_path]
    return [model_path]


class Sampler(threading.Thread):
    def __init__(self, process_name: str):
        super().__init__(daemon=True)
        self.process_name = process_name
        self.samples: deque[tuple[float, int, int | None]] = deque()  # (ts, rss, gpu)
        self.pid: int | None = None
        self.lock = threading.Lock()

    def run(self) -> None:
        proc = None
        while True:
            if proc is None or not proc.is_running():
                proc = find_process(self.process_name)
                self.pid = proc.pid if proc else None
            rss = None
            if proc is not None:
                try:
                    rss = proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    proc = None
            gpu = gpu_memory_bytes()
            now = time.time()
            with self.lock:
                if rss is not None or gpu is not None:
                    self.samples.append((now, rss or 0, gpu))
                while self.samples and now - self.samples[0][0] > HISTORY_S:
                    self.samples.popleft()
            time.sleep(SAMPLE_INTERVAL_S)

    def peak_since(self, since_ts: float | None) -> dict:
        with self.lock:
            rows = [s for s in self.samples if since_ts is None or s[0] >= since_ts]
        rss = [r[1] for r in rows if r[1]]
        gpu = [r[2] for r in rows if r[2] is not None]
        return {
            "n_samples": len(rows),
            "peak_rss_bytes": max(rss) if rss else None,
            "rss_reason": None
            if rss
            else f"no process matching {self.process_name!r} was sampled in the window",
            "peak_gpu_memory_bytes": max(gpu) if gpu else None,
            "gpu_reason": None if gpu else "nvidia-smi not available on this host",
        }


def make_handler(sampler: Sampler):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):  # quiet
            pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path != "/sample":
                self.send_response(404)
                self.end_headers()
                return
            q = parse_qs(url.query)
            since_ts = None
            if "since" in q:
                try:
                    since_ts = datetime.fromisoformat(q["since"][0]).astimezone(UTC).timestamp()
                except ValueError:
                    since_ts = None
            data = sampler.peak_since(since_ts)
            data.update(
                {
                    "process": sampler.process_name,
                    "pid": sampler.pid,
                    "host": {
                        "platform": platform.platform(),
                        "machine": platform.machine(),
                        "total_ram_bytes": psutil.virtual_memory().total,
                    },
                    "model_file_bytes": None,
                    "model_files": None,
                    "file_reason": "no model_path given",
                }
            )
            if "model_path" in q:
                path = q["model_path"][0]
                files = [f for f in model_files(path) if os.path.exists(f)]
                if files:
                    data["model_file_bytes"] = sum(os.path.getsize(f) for f in files)
                    data["model_files"] = [{"path": f, "bytes": os.path.getsize(f)} for f in files]
                    data["file_reason"] = None
                else:
                    data["file_reason"] = f"model file not found on this host: {path}"
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--process",
        default="llama-server",
        help="substring of the server process name/cmdline",
    )
    ap.add_argument("--port", type=int, default=9108)
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="loopback is enough for Docker Desktop via host.docker.internal",
    )
    args = ap.parse_args()
    sampler = Sampler(args.process)
    sampler.start()
    print(
        f"resource agent sampling {args.process!r} every {SAMPLE_INTERVAL_S}s; GET http://{args.host}:{args.port}/sample"
    )
    HTTPServer((args.host, args.port), make_handler(sampler)).serve_forever()


if __name__ == "__main__":
    main()
