# LLM Inference & Evaluation Platform

Serves open LLMs behind OpenAI-compatible endpoints (llama.cpp `llama-server` locally, vLLM on GPU), runs a fixed evaluation suite against them, scores each model on five dimensions (task accuracy, latency/throughput, resource usage, tokenizer efficiency, output reliability), stores everything in PostgreSQL and compares runs in a React dashboard.

It is also the harness for a planned study: *does GGUF quantization hurt Indic languages more than English, and does the drop track tokenizer fertility?*

**Status: Milestone 1 (data model + models/datasets API).** See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full plan and milestones. No evaluation results exist yet; nothing in this repo reports numbers that were not produced by a real run.

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

Model servers are **not** bundled. They run on the host (or a remote GPU box) and are registered by URL. Setup notes will live in `scripts/`.

## Development without Docker

```bash
# backend
cd backend
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/uvicorn app.main:app --reload      # needs a Postgres on localhost:5432
.venv/bin/pytest && .venv/bin/ruff check .   # tests need Postgres; they use a separate llmeval_test database
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
