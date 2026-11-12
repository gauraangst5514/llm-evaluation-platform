"""Dimension 3: resource usage.

The API usually runs in Docker and cannot see the model server's process. A tiny
host-side agent (scripts/resource_agent.py) samples the server's RSS (psutil) and,
when present, GPU memory (nvidia-smi), and can stat the model file. This dimension
asks that agent; if there is no agent, every value is null with the reason recorded.
Nothing here is estimated.
"""

import httpx

from app.evals.base import EvalContext, Metric

DIM = "resources"


def _null(name: str, unit: str, reason: str) -> Metric:
    return Metric(DIM, name, None, unit, None, {"measured": False, "reason": reason})


class ResourcesEval:
    name = DIM

    def __init__(self, http: httpx.AsyncClient | None = None):
        self._http = http

    async def _model_path(self, ctx: EvalContext) -> str | None:
        """llama-server exposes /props with model_path; other backends do not."""
        http = self._http or httpx.AsyncClient()
        try:
            r = await http.get(f"{ctx.client.base_url}/props", timeout=10)
            if r.status_code == 200:
                return r.json().get("model_path")
        except httpx.HTTPError:
            return None
        finally:
            if self._http is None:
                await http.aclose()
        return None

    async def run(self, ctx: EvalContext) -> list[Metric]:
        agent = ctx.model.resource_agent_url
        if not agent:
            reason = (
                "no resource_agent_url on the model; "
                "run scripts/resource_agent.py on the server host"
            )
            return [
                _null("peak_rss_bytes", "bytes", reason),
                _null("peak_gpu_memory_bytes", "bytes", reason),
                _null("model_file_bytes", "bytes", reason),
            ]
        since = ctx.run.started_at.isoformat() if ctx.run.started_at else None
        model_path = await self._model_path(ctx)
        http = self._http or httpx.AsyncClient()
        try:
            r = await http.get(
                f"{agent.rstrip('/')}/sample",
                params={k: v for k, v in {"since": since, "model_path": model_path}.items() if v},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
        except (httpx.HTTPError, ValueError) as exc:
            reason = f"resource agent unreachable: {type(exc).__name__}: {exc}"
            return [
                _null("peak_rss_bytes", "bytes", reason),
                _null("peak_gpu_memory_bytes", "bytes", reason),
                _null("model_file_bytes", "bytes", reason),
            ]
        finally:
            if self._http is None:
                await http.aclose()

        common = {
            "measured": True,
            "agent": agent,
            "process": data.get("process"),
            "pid": data.get("pid"),
            "since": since,
            "n_samples": data.get("n_samples"),
            "host": data.get("host"),
        }
        metrics = []
        rss = data.get("peak_rss_bytes")
        metrics.append(
            Metric(DIM, "peak_rss_bytes", rss, "bytes", None, common)
            if rss is not None
            else _null(
                "peak_rss_bytes", "bytes", data.get("rss_reason") or "agent found no process"
            )
        )
        gpu = data.get("peak_gpu_memory_bytes")
        metrics.append(
            Metric(DIM, "peak_gpu_memory_bytes", gpu, "bytes", None, common)
            if gpu is not None
            else _null(
                "peak_gpu_memory_bytes", "bytes", data.get("gpu_reason") or "no NVIDIA GPU on host"
            )
        )
        size = data.get("model_file_bytes")
        metrics.append(
            Metric(
                DIM,
                "model_file_bytes",
                size,
                "bytes",
                None,
                {**common, "model_path": model_path, "files": data.get("model_files")},
            )
            if size is not None
            else _null("model_file_bytes", "bytes", data.get("file_reason") or "model path unknown")
        )
        return metrics
