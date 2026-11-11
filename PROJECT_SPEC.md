# LLM Inference & Evaluation Platform: rebuild spec

> Hand this file to Claude Code at the root of a new, empty git repo. Save it as `CLAUDE.md` or `PROJECT_SPEC.md` and tell Claude Code: "Read PROJECT_SPEC.md and build it milestone by milestone. Stop after each milestone and show me what works."

## 0. Context and rules

I am a B.Tech IT student rebuilding a project whose code I lost. This will go on my resume and GitHub, so correctness and honesty matter more than feature count.

- Public models, public datasets and public code only. No code, data, names or keys from any employer or internship.
- Never commit secrets. Provide `.env.example`; `.env` stays in `.gitignore`. Do not commit logs, test output files, model weights or `.env.production`.
- **Commit after every working step and push to GitHub** (I lost this project once; origin must always be current).
- Every number I put on my resume must come from a run this platform actually did. Do not invent results, and do not hard-code sample numbers into the dashboard or README.
- My hardware: MacBook Air M2, 16 GB RAM, no CUDA GPU. Heavy GPU runs go to free Kaggle or Colab, or Modal. The platform must work locally with small GGUF models and also against a remote GPU server.

## 1. What it is (one paragraph)

A platform that serves open LLMs behind OpenAI-compatible endpoints (vLLM on GPU, llama.cpp server for local GGUF), runs a fixed evaluation suite against them, scores each model on five dimensions, stores everything in PostgreSQL, and shows comparisons in a React dashboard. It also acts as the harness for my planned study "Does GGUF quantization hurt Indic languages more than English?" (see section 9).

## 2. Stack

- Backend: Python 3.11, FastAPI, SQLAlchemy 2 + Alembic, httpx (async), Pydantic v2
- DB: PostgreSQL 16
- Frontend: React + Vite + TypeScript, Recharts (or similar), TanStack Query
- Model backends (all OpenAI-compatible `/v1/chat/completions`, streaming on):
  - `llama-server` from llama.cpp, for GGUF models locally
  - `vllm serve`, for GPU machines (Kaggle, Colab, Modal)
- Tokenizers: `transformers` AutoTokenizer from the model's original Hugging Face repo
- Packaging: Docker Compose for api + db + web. The model servers run outside Compose and the platform connects by URL.
- Tests: pytest; one integration test that runs end to end against a mock OpenAI-compatible server

## 3. Repo layout

```
llm-eval-platform/
  README.md
  PROJECT_SPEC.md
  docker-compose.yml
  .env.example
  backend/
    app/
      main.py
      config.py
      db/ (models.py, session.py)
      api/ (models.py, runs.py, compare.py, health.py)
      backends/ (base.py, openai_compat.py)
      evals/ (base.py, accuracy.py, latency.py, resources.py, tokenizer.py, reliability.py)
      datasets/ (loader.py, registry.py)
      services/ (run_executor.py)
    alembic/
    tests/
  frontend/
    src/ (pages, components, api client)
  scripts/
    start_llama_server.md   # how to start a GGUF model locally
    start_vllm_server.md    # how to start vLLM on Kaggle/Colab/Modal
    run_pilot.py            # CLI to launch a run without the UI
  results/                  # exported CSV/JSON of real runs, committed on purpose
```

## 4. The five evaluation dimensions

Every run produces these for a (model, quantization, dataset subset) combination.

1. **Task accuracy.** Multiple-choice QA on a fixed, seeded subset. Greedy decoding (temperature 0). Parse the chosen option strictly, count unparseable outputs as wrong and report their rate. Break results down by language and subject.
2. **Latency and throughput.** Time to first token (TTFT), tokens per second, and total latency, as P50 and P95 over the run, measured from the streaming response. Record concurrency used.
3. **Resource usage.** Peak RAM or VRAM while serving, read from the host (e.g. `psutil` for the server process on CPU/Metal, `nvidia-smi` on GPU) plus model file size on disk. If it can't be measured, store null and say so. Never estimate.
4. **Tokenizer efficiency.** Fertility (tokens per whitespace word) and tokens per character, per language, using the model's own tokenizer on the same text samples. Also record the round-trip check (decode(encode(x)) == x) pass rate.
5. **Output reliability.** (a) Valid-format rate: the share of responses that parse as the requested answer format or JSON schema. (b) Consistency: re-run a small prompt set 3 times at temperature 0.7 and report agreement rate.

Keep each dimension a separate class implementing `Eval.run(model_endpoint, dataset) -> list[Metric]` so new dimensions can be added without touching the API.

## 5. Data model (PostgreSQL)

- `models(id, name, hf_repo, backend, endpoint_url, quantization, params_b, notes, created_at)`
- `datasets(id, name, source_url, license, language, n_items, split, seed)`
- `runs(id, model_id, dataset_id, status, config_json, started_at, finished_at, error)`
- `results_item(id, run_id, item_id, prompt, response, parsed_answer, correct, ttft_ms, latency_ms, output_tokens)`
- `metrics(id, run_id, dimension, name, value, unit, language, extra_json)`

Indexes on `runs(model_id)`, `metrics(run_id, dimension)`.

## 6. API (FastAPI)

- `GET /health`
- `POST /models`, `GET /models`, `DELETE /models/{id}`
- `GET /datasets`
- `POST /runs` (body: model_id, dataset_id, dimensions[], n_items, seed, concurrency) -> returns run id; execution runs as a background task
- `GET /runs`, `GET /runs/{id}` (status, progress, metrics)
- `GET /runs/{id}/items` (paged per-item results)
- `GET /compare?run_ids=1,2,3` -> metrics side by side, grouped by dimension and language
- `GET /export/{run_id}.csv`

Run executor: async, bounded concurrency, per-request timeout, retry once on transient failure, and a clear `failed` status with the error stored. A run can be cancelled.

## 6b. Datasets

Use small, seeded, fixed subsets (store the item ids so a run is reproducible). Candidates, and I must verify each licence before use and record it in `datasets.license`:

- English baseline: an MMLU or ARC subset
- Indic: MILU, IndicXNLI, Global-MMLU (Hindi and one more Indic language)
- Tokenizer samples: public parallel text (e.g. FLORES-style) in Hindi, Tamil or Bengali, and English

If a dataset's licence forbids redistribution, download at runtime via script; do not commit the data.

## 7. Dashboard (React)

- **Runs page:** table of runs with model, quantization, dataset, status, headline accuracy.
- **Compare page:** pick 2 to 6 runs. Grouped bar chart per dimension. Per-language accuracy table. A line chart of accuracy vs quantization level for one model family, and fertility vs accuracy drop per language (scatter).
- **Run detail:** progress bar while running, then metrics and a sample of failed items.
- Show units and sample sizes on every chart. Show an empty state, not fake data, when there are no runs.

## 8. Milestones (stop and demo after each)

1. **M0 Repo and skeleton.** Git init, `.gitignore`, README stub, push to GitHub. Docker Compose with Postgres + FastAPI `/health` + Vite app. Done when `docker compose up` shows a working health page.
2. **M1 Data model and models API.** Alembic migrations, CRUD for models and datasets, tests.
3. **M2 Backend client and runner.** OpenAI-compatible streaming client with TTFT timing. Run executor with a mock server for tests. Done when a run against a real local `llama-server` stores per-item results.
4. **M3 The five dimensions.** One at a time, each with a unit test. Done when one run produces metrics for all five.
5. **M4 Dashboard.** Runs, compare and run detail pages against real data.
6. **M5 Docs and polish.** README with architecture diagram, setup for macOS (llama.cpp) and for Kaggle (vLLM), a screenshot, and a real results table. Add a CI workflow (GitHub Actions) that runs lint and tests.
7. **M6 Pilot study.** Run the quantization pilot (section 9) through the platform and export results to `results/`.

## 9. The pilot study this platform must support

Question: does GGUF quantization (Q8_0, Q6_K, Q4_K_M, Q3_K_M) hurt Indic languages more than English, and does the accuracy drop track tokenizer fertility?

- One open 7-9B instruct model first, then more if time allows.
- Languages: English, Hindi and one more Indic language.
- Same prompts, same seed, same subset for every quantization level.
- Output: `results/pilot_<model>.csv` plus the dashboard comparison, and a one-page write-up in `docs/pilot.md` that states the finding, the sample sizes and the caveats. Report confidence intervals or at least n per cell; do not claim differences that the sample size cannot support.

## 10. Definition of done

- `docker compose up` brings up api, db and web; model servers are documented, not bundled.
- All five dimensions produce real metrics from a real run on at least two models, or two quantizations of one model.
- README contains: what it does, architecture diagram, quick start, the real results table, limitations, and the licence of each dataset.
- Tests pass in CI. No secrets, logs or weights in git history.
- A short "Limitations" section is honest about sample sizes, single-run results and what was not measured.

## 11. Things to ask me instead of guessing

- Which exact models and quantization files to use (check disk space and the 16 GB RAM limit first).
- Which Indic languages and datasets, after the licence check.
- Whether to deploy a public demo (default: no).
