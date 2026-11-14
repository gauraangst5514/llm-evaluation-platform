# LLM Inference & Evaluation Platform

[![CI](https://github.com/gauraangst5514/llm-evaluation-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/gauraangst5514/llm-evaluation-platform/actions/workflows/ci.yml)

Serves open LLMs behind OpenAI-compatible endpoints (llama.cpp `llama-server` locally on a Mac, vLLM on a GPU box), runs a fixed, seeded evaluation suite against them, scores each model on **five dimensions**, stores everything in PostgreSQL and compares runs in a React dashboard.

It is also the harness for a pilot study: **does GGUF quantization (Q8_0 → Q3_K_M) hurt Indic languages more than English, and does the accuracy drop track tokenizer fertility?** See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full specification and milestones.

**Status: all six milestones done.** The quantization pilot (M6) has run: Qwen2.5-7B-Instruct at four GGUF quantizations × English, Hindi and Bengali × 200 items. Write-up: [docs/pilot.md](docs/pilot.md).

![Compare page: Q8_0 vs Q3_K_M in English, Hindi and Bengali](docs/screenshot-pilot-compare.png)

## Architecture

```mermaid
flowchart LR
  subgraph host["Your machine (or a GPU box)"]
    LS["llama-server (GGUF, Metal)"]
    VL["vLLM (GPU)"]
    RA["resource_agent.py<br/>psutil / nvidia-smi"]
  end
  subgraph compose["docker compose"]
    WEB["web: React + Vite<br/>runs · detail · compare"]
    API["api: FastAPI<br/>run executor · 5 eval dimensions"]
    DB[("PostgreSQL 16<br/>models · datasets · runs<br/>results_item · metrics")]
  end
  HF["Hugging Face Hub<br/>Global-MMLU parquet · tokenizers"]
  WEB -- "/api (Vite proxy)" --> API
  API --> DB
  API -- "chat/completions (SSE)" --> LS
  API -- "chat/completions (SSE)" --> VL
  API -- "/sample" --> RA
  API -- "runtime download, cached volume" --> HF
```

Model servers are deliberately **not** bundled. They run wherever the hardware is and are registered by URL.

## Quick start

```bash
cp .env.example .env            # defaults work for local Docker
docker compose up --build
```

| Service | URL |
|---|---|
| Dashboard | http://localhost:5173 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Postgres | localhost:5432, user/db `llmeval` |

On start the API applies Alembic migrations and upserts the dataset registry, so `GET /datasets` is populated immediately.

### Start a model server

**macOS, Apple Silicon (llama.cpp)** — full notes in [scripts/start_llama_server.md](scripts/start_llama_server.md):

```bash
brew install llama.cpp
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192 -np 2
# optional, for the resources dimension:
uv run --with psutil scripts/resource_agent.py --process llama-server --port 9108
```

Keep each slot's context (`-c / -np`) at 4096 or more: a Global-MMLU law question that is 363 tokens in English is 1512 tokens in Hindi.

**Kaggle / Colab / Modal (vLLM)** — see [scripts/start_vllm_server.md](scripts/start_vllm_server.md). vLLM serves the same OpenAI-compatible API; expose it through a tunnel and register the tunnel URL.

### Register it and run

From the dashboard (Runs page → Register, then New run), or:

```bash
curl -X POST localhost:8000/models -H 'content-type: application/json' -d '{
  "name": "Qwen2.5-7B-Instruct", "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
  "backend": "llama_cpp", "endpoint_url": "http://host.docker.internal:8080",
  "quantization": "Q4_K_M", "params_b": 7.6,
  "resource_agent_url": "http://host.docker.internal:9108"}'

python scripts/run_pilot.py --model-id 1 --dataset global_mmlu_en --dataset global_mmlu_hi --n-items 200
python scripts/export_results.py --run 1 --name pilot_qwen2.5-7b_q4_k_m_en   # -> results/*.csv
```

One `models` row per quantization (same `name`, different `quantization`); the dashboard compares rows.

## How a run works

1. `POST /runs` stores the run as `pending` and starts an asyncio task in the API process.
2. The dataset's fixed subset is materialised once: a seeded sample of `sample_id`s, chosen from the English file and shared by Hindi and Bengali, stored on the dataset row. Every run with the same seed sees the same items in the same order; `n_items` takes a prefix.
3. Each item is sent as a chat completion with `temperature=0`, `stream=true` and `stream_options.include_usage`. TTFT is measured at the first content delta; token counts come from the server's usage block or are stored as null.
4. Bounded concurrency, per-request timeout, one retry on transient failures (timeouts, connection errors, 5xx, 429). 4xx errors such as context overflow are recorded on the item and not retried.
5. Answers are parsed strictly (a single option letter, optionally after "Answer:"). Unparseable replies count as wrong and are reported; failed requests are not scored. More than 10% failed requests marks the run `failed`, never "low accuracy".
6. The requested dimensions run after the item pass and store their metrics. Runs can be cancelled; runs interrupted by an API restart are marked `failed` on startup.

## The five dimensions

Each is a class in `backend/app/evals/` implementing `Eval.run(ctx) -> list[Metric]`. Anything that cannot be measured is stored as `null` with the reason in `extra_json`. Nothing is estimated.

| Dimension | What is measured | How |
|---|---|---|
| accuracy | accuracy with 95% Wilson CI, unparseable rate, request-error rate, per-category and per-subject breakdown, answer-letter distribution (position bias) | the greedy MCQ pass |
| latency | TTFT and total latency P50/P95 at the run's concurrency; prefill tokens/s; decode tokens/s from 5 sequential 128-token generation probes | streamed responses, server-reported token counts |
| resources | peak RSS of the server process, peak GPU memory, model file size (split GGUFs summed) | host-side `scripts/resource_agent.py`; null without it |
| tokenizer | fertility (tokens per whitespace word), tokens per character, decode(encode(x)) == x pass rate | the model's own `AutoTokenizer` on the exact question+option text of the run |
| reliability | MCQ valid-format rate; JSON-schema valid rate on 5 probes; consistency over 3 repeats at temperature 0.7 (unanimous rate and mean pairwise agreement) | extra requests after the accuracy pass |

## API

`GET /health` · `POST/GET/PATCH/DELETE /models` · `GET /datasets` · `POST /runs` · `GET /runs` · `GET /runs/{id}` · `GET /runs/{id}/items?only_failed=` · `POST /runs/{id}/cancel` · `DELETE /runs/{id}` · `GET /compare?run_ids=1,2` · `GET /export/{id}.csv?kind=items|metrics`

## Datasets and licences

Registered in `backend/app/datasets/registry.py`. Nothing is committed to the repo; loaders fetch from the Hub at runtime into a Docker volume.

| Dataset | Languages | Split | Subset | Licence | Checked |
|---|---|---|---|---|---|
| [Global-MMLU](https://huggingface.co/datasets/CohereLabs/Global-MMLU) | en, hi, bn (same questions, shared `sample_id`) | test | 200 items, seed 42 | Apache-2.0 | 2026-11-12, Hugging Face dataset card |

The parallel subset doubles as the tokenizer-efficiency sample, so fertility is measured on the exact text the model is scored on. Not registered: MILU (CC-BY-4.0) and FLORES-200 (CC-BY-SA-4.0) are gated on the Hub and their fields could not be verified without a token; IndicXNLI states no licence on its card.

## Results

The table below is generated by `python scripts/results_table.py results/pilot_*_metrics.csv` from CSVs the platform exported (runs 6–17; all metrics in one file: [results/pilot_qwen2.5-7b.csv](results/pilot_qwen2.5-7b.csv)).

| Model | Quant | Lang | Run | Accuracy | TTFT p50 | Decode tok/s p50 | Fertility (tok/word) | JSON valid | Consistency@0.7 | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | Q8_0 | en | 6 | 69.0% (62–75%), n=200 | 1530 ms | 5.1 | 1.31 | 100% (n=5) | 100% (n=10) | 1336 MB |
| Qwen2.5-7B-Instruct | Q8_0 | hi | 7 | 46.5% (40–53%), n=200 | 8909 ms | 5.0 | 4.83 | 100% (n=5) | 90% (n=10) | 1441 MB |
| Qwen2.5-7B-Instruct | Q8_0 | bn | 8 | 38.5% (32–45%), n=200 | 8478 ms | 4.5 | 7.02 | 100% (n=5) | 70% (n=10) | 1457 MB |
| Qwen2.5-7B-Instruct | Q6_K | en | 9 | 69.0% (62–75%), n=200 | 2645 ms | 5.4 | 1.31 | 100% (n=5) | 100% (n=10) | 4330 MB |
| Qwen2.5-7B-Instruct | Q6_K | hi | 10 | 47.0% (40–54%), n=200 | 10180 ms | 3.9 | 4.83 | 100% (n=5) | 90% (n=10) | 4034 MB |
| Qwen2.5-7B-Instruct | Q6_K | bn | 11 | 39.5% (33–46%), n=200 | 6719 ms | 8.2 | 7.02 | 100% (n=5) | 60% (n=10) | 6146 MB |
| Qwen2.5-7B-Instruct | Q4_K_M | en | 12 | 70.0% (63–76%), n=200 | 1061 ms | 8.9 | 1.31 | 100% (n=5) | 100% (n=10) | 5585 MB |
| Qwen2.5-7B-Instruct | Q4_K_M | hi | 13 | 44.5% (38–51%), n=200 | 6383 ms | 9.1 | 4.83 | 100% (n=5) | 70% (n=10) | 5268 MB |
| Qwen2.5-7B-Instruct | Q4_K_M | bn | 14 | 39.0% (33–46%), n=200 | 7773 ms | 9.2 | 7.02 | 100% (n=5) | 80% (n=10) | 5230 MB |
| Qwen2.5-7B-Instruct | Q3_K_M | en | 15 | 66.5% (60–73%), n=200 | 1642 ms | 9.7 | 1.31 | 100% (n=5) | 100% (n=10) | 5054 MB |
| Qwen2.5-7B-Instruct | Q3_K_M | hi | 16 | 45.0% (38–52%), n=200 | 6538 ms | 8.8 | 4.83 | 100% (n=5) | 80% (n=10) | 4821 MB |
| Qwen2.5-7B-Instruct | Q3_K_M | bn | 17 | 40.5% (34–47%), n=200 | 7901 ms | 7.1 | 7.02 | 100% (n=5) | 90% (n=10) | 4590 MB |

Setup: MacBook Air M2 16 GB, llama-server 0.9.0 (Metal), `-c 8192 -np 2`, run concurrency 2, greedy decoding, the same 200 Global-MMLU questions in every cell.

**Finding:** at n=200, no quantization level down to Q3_K_M produced a measurable accuracy drop in any language. All paired drops against Q8_0 on the same items are between −2 and +2.5 pp, and every 95% interval includes zero. So the pilot cannot show Indic languages losing more than English, or the loss tracking tokenizer fertility. What it does show: a large language gap at every quantization (69% English, 46.5% Hindi, 38.5% Bengali at Q8_0), and 3.7× (Hindi) and 5.4× (Bengali) more tokens per word than English, which raises time to first token. Paired analysis and caveats: [docs/pilot.md](docs/pilot.md). Peak RSS undercounts memory on macOS (see Limitations); the model files are 8.10 / 6.25 / 4.68 / 3.81 GB.

The smoke-test exports of Qwen2.5-0.5B (`results/smoke_*`, n=20) are kept as pipeline checks and support no conclusion.

## Limitations

- **Sample sizes.** 200 items per language is the registered maximum. The 95% interval on a single accuracy is about ±6–7 pp at n=200 and ±20 pp at n=20; pairing items across quantizations narrows differences to roughly ±3–5 pp. Smaller effects are not detectable, and the dashboard shows the interval next to every accuracy. Consistency (10 items × 3) and JSON validity (5 probes) are too small to compare.
- **Single runs.** Greedy decoding is reproducible (runs 1 and 4 gave identical answers item by item), but latency and resource numbers are from one run on a laptop that was doing other things. The pilot cells ran on two different days, and latency does not decrease steadily with quantization. Treat these numbers as indicative.
- **English instruction prompt.** The question and options are in the target language; the instruction ("answer with only the letter") is English for every language. This is a confound for the Indic results; the pilot write-up states it.
- **One model.** The pilot covers a single 7B model family. Other families, or quantizations below Q3_K_M, may behave differently.
- **Strict parsing.** A model that answers correctly in prose is scored wrong. The unparseable rate is reported so this is visible.
- **Resource numbers on macOS.** llama.cpp uses mmap and Metal unified memory, so process RSS undercounts the memory the model actually pins. `peak_rss_bytes` is a lower bound there; `model_file_bytes` is exact. On NVIDIA hosts the agent reads `nvidia-smi`.
- **Decode throughput** comes from 5 open-ended probes at concurrency 1, not from the MCQ answers (one-letter replies are too short to time).
- **Not measured yet:** anything on GPU, any model larger than 0.5B, Bengali (registered, not yet run), MILU/FLORES.

## Development

```bash
# backend (tests need Postgres; the compose db works, tests use a separate llmeval_test database)
cd backend
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/pytest -q && .venv/bin/ruff check . ../scripts
.venv/bin/alembic revision --autogenerate -m "describe change"   # after editing app/db/models.py

# frontend
cd frontend
npm install && npm run dev        # proxies /api to http://localhost:8000
npm run lint && npm run build
```

CI (GitHub Actions) runs ruff, the backend tests against a Postgres service, and the frontend type-check and build on every push.

## Repository layout

```
backend/app/
  api/          health, models, datasets, runs, compare+export routers
  backends/     OpenAI-compatible streaming client (TTFT, usage, error classes)
  datasets/     registry (licences, subset sizes) and loader (seeded, parallel subsets)
  evals/        mcq prompting/parsing + the five dimension classes
  services/     run executor
  db/           SQLAlchemy models, session
backend/alembic/  migrations
backend/tests/    71 tests incl. an in-process mock OpenAI server
frontend/src/     pages (runs, run detail, compare), charts, API client
scripts/          start_llama_server.md, start_vllm_server.md, resource_agent.py,
                  run_pilot.py, export_results.py, results_table.py
results/          CSV exports of real runs (committed on purpose)
docs/             screenshots, pilot write-up (pilot.md)
```
