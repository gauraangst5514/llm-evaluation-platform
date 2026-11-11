# LLM Inference & Evaluation Platform

Serves open LLMs behind OpenAI-compatible endpoints (llama.cpp `llama-server` locally, vLLM on GPU), runs a fixed evaluation suite against them, scores each model on five dimensions (task accuracy, latency/throughput, resource usage, tokenizer efficiency, output reliability), stores everything in PostgreSQL and compares runs in a React dashboard.

It is also the harness for a planned study: *does GGUF quantization hurt Indic languages more than English, and does the drop track tokenizer fertility?*

**Status: Milestone 0 (skeleton).** See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full plan and milestones. No evaluation results exist yet; nothing in this repo reports numbers that were not produced by a real run.

## Quick start

```bash
cp .env.example .env        # defaults work for local Docker
docker compose up --build
```

- Web (health page): http://localhost:5173
- API: http://localhost:8000/health and http://localhost:8000/docs
- Postgres: localhost:5432 (user/db `llmeval`)

Model servers are **not** bundled. They run on the host (or a remote GPU box) and are registered by URL. Setup notes will live in `scripts/`.

## Development without Docker

```bash
# backend
cd backend
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/uvicorn app.main:app --reload      # needs a Postgres on localhost:5432
.venv/bin/pytest && .venv/bin/ruff check .

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

## Licence of datasets

Recorded per dataset in the database and listed here once each licence has been checked (M1 onward).
