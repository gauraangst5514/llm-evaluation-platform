# LLM Inference & Evaluation Platform

Serves open LLMs behind OpenAI-compatible endpoints (llama.cpp `llama-server` locally, vLLM on GPU), runs a fixed evaluation suite against them, scores each model on five dimensions (task accuracy, latency/throughput, resource usage, tokenizer efficiency, output reliability), stores everything in PostgreSQL and compares runs in a React dashboard.

It is also the harness for a planned study: *does GGUF quantization hurt Indic languages more than English, and does the drop track tokenizer fertility?*

**Status: Milestone 3 (all five evaluation dimensions produce real metrics; compare and CSV export endpoints).** See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full plan and milestones. No evaluation results exist yet; nothing in this repo reports numbers that were not produced by a real run.

## Quick start

```bash
cp .env.example .env        # defaults work for local Docker
docker compose up --build
```

- Web (health page): http://localhost:5173
- API: http://localhost:8000/health and http://localhost:8000/docs
- Postgres: localhost:5432 (user/db `llmeval`)

On start the API applies Alembic migrations and upserts the dataset registry, so `GET /datasets` is populated immediately. Register a model server:

```bash
curl -X POST localhost:8000/models -H 'content-type: application/json' -d '{
  "name": "Qwen2.5-7B-Instruct", "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
  "backend": "llama_cpp", "endpoint_url": "http://host.docker.internal:8080",
  "quantization": "Q4_K_M", "params_b": 7.6}'
```

Model servers are **not** bundled. They run on the host (or a remote GPU box) and are registered by URL: see [scripts/start_llama_server.md](scripts/start_llama_server.md) (macOS, GGUF) and [scripts/start_vllm_server.md](scripts/start_vllm_server.md) (Kaggle/Colab/Modal).

Launch a run from the UI (M4), the API, or the CLI:

```bash
python scripts/run_pilot.py --model-id 1 --dataset global_mmlu_en --dataset global_mmlu_hi --n-items 50
```

## How a run works

1. `POST /runs` stores the run as `pending` and starts an asyncio task in the API process.
2. The dataset's fixed subset is materialised once (seeded sample of `sample_id`s, shared across the en/hi/bn parallel group) and stored on the dataset row.
3. Each item is sent as a chat completion with `temperature=0`, `stream=true` and `stream_options.include_usage`; TTFT is measured at the first content delta, token counts come from the server's usage block or are stored as null.
4. Bounded concurrency, per-request timeout, one retry on transient failures (timeouts, connection errors, 5xx, 429). 4xx errors such as context overflow are recorded on the item and not retried.
5. Answers are parsed strictly (a single option letter). Unparseable replies count as wrong and are reported; failed requests are not scored. If more than 10% of requests fail the run is `failed`, not "low accuracy".
6. Runs can be cancelled; runs interrupted by an API restart are marked `failed` on startup rather than left `running` forever.

## The five dimensions

Each is a class in `backend/app/evals/` implementing `Eval.run(ctx) -> list[Metric]`; the executor runs the requested ones after the per-item pass and stores their metrics. Anything that cannot be measured is stored as `null` with the reason in `extra_json`.

| Dimension | What is measured | How |
|---|---|---|
| accuracy | accuracy with 95% Wilson CI, unparseable rate, request-error rate, per-category and per-subject breakdown, answer-letter distribution | from the greedy MCQ pass; strict single-letter parsing |
| latency | TTFT and total latency P50/P95 at the run's concurrency; prefill tokens/s; decode tokens/s from 5 sequential 128-token generation probes | streamed responses; token counts only from the server's usage block |
| resources | peak RSS of the server process, peak GPU memory, model file size | host-side `scripts/resource_agent.py` (psutil, nvidia-smi); null without it |
| tokenizer | fertility (tokens/word), tokens/char, decode(encode(x)) == x pass rate | the model's own `AutoTokenizer` on the exact question+option text of the run |
| reliability | MCQ valid-format rate; JSON-schema valid rate on 5 probes; consistency over 3 repeats at temperature 0.7 (unanimous rate and mean pairwise agreement) | extra requests after the accuracy pass |

Comparison and export: `GET /compare?run_ids=1,2` returns metrics side by side grouped by dimension, name and language; `GET /export/{run_id}.csv?kind=items|metrics` downloads a run.

## Development without Docker

```bash
# backend
cd backend
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/uvicorn app.main:app --reload      # needs a Postgres on localhost:5432
.venv/bin/pytest && .venv/bin/ruff check .   # tests need Postgres (separate llmeval_test db); the run tests use an in-process mock OpenAI server
.venv/bin/alembic revision --autogenerate -m "describe change"   # after editing app/db/models.py

# frontend
cd frontend
npm install && npm run dev                   # proxies /api to http://localhost:8000
```

## Layout

```
backend/   FastAPI app (app/), tests/, Alembic migrations (from M1)
frontend/  React + Vite + TypeScript dashboard
scripts/   how to start llama-server / vLLM, CLI to launch runs
results/   exported CSV/JSON of real runs (committed on purpose)
docs/      write-ups, e.g. the quantization pilot
```

## Datasets and licences

Registered in `backend/app/datasets/registry.py` and stored in the `datasets` table; nothing is committed to the repo, loaders fetch at runtime.

| Dataset | Languages | Split | Subset | Licence | Checked |
|---|---|---|---|---|---|
| [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | en, hi, bn (same questions, shared `sample_id`) | test | 200 items, seed 42 | Apache-2.0 | 2026-11-12, HF dataset card |

The parallel subset doubles as the tokenizer-efficiency sample so fertility is measured on the exact text the model is scored on. Candidates not yet registered because they are gated on the Hub and their fields could not be verified without a token: MILU (CC-BY-4.0), FLORES-200 (CC-BY-SA-4.0). IndicXNLI has no licence stated on its card and is excluded.
