# Starting a GGUF model locally with llama.cpp

The platform never bundles model servers. Start one on the host, then register its URL with `POST /models`.

## Install (macOS, Apple Silicon)

```bash
brew install llama.cpp        # provides llama-server with Metal acceleration
llama-server --version
```

## Start a model straight from the Hugging Face Hub

`-hf repo:QUANT` downloads the GGUF into `~/Library/Caches/llama.cpp/` (not into this repo) and serves it:

```bash
# Smoke-test model (0.7 GB) used to validate the pipeline
llama-server -hf Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q8_0 --port 8080 --host 127.0.0.1 -c 8192 -np 2

# Pilot model, one quantization at a time (split GGUFs are handled automatically)
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --port 8080 -c 8192 -np 2
llama-server -hf Qwen/Qwen2.5-7B-Instruct-GGUF:Q8_0   --port 8080 -c 8192 -np 2
```

Flags that matter for evaluation:

| Flag | Meaning | Why it matters |
|---|---|---|
| `-c 8192` | total context, split across slots | each slot gets `-c / -np` tokens. Keep that >= 4096: a Global-MMLU law question that is 363 tokens in English was 1512 tokens in Hindi, and a 1024-token slot rejected it with HTTP 400 `exceed_context_size_error` |
| `-np N` | parallel slots | sets the real concurrency; match it to the run's `concurrency` |
| `--port` | HTTP port | the API in Docker reaches it as `http://host.docker.internal:<port>` |

Memory on a 16 GB M2: Q4_K_M of a 7B model needs about 5 GB, Q6_K about 7 GB, Q8_0 about 8.5 GB plus KV cache. Close other heavy apps for Q8_0.

## Check it

```bash
curl -s localhost:8080/health
curl -s localhost:8080/v1/models | jq '.data[].id'
```

## Register it with the platform

```bash
curl -X POST localhost:8000/models -H 'content-type: application/json' -d '{
  "name": "Qwen2.5-7B-Instruct", "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
  "backend": "llama_cpp", "endpoint_url": "http://host.docker.internal:8080",
  "quantization": "Q4_K_M", "params_b": 7.6,
  "notes": "llama-server <version>, -c 4096 -np 2, GGUF from Qwen/Qwen2.5-7B-Instruct-GGUF"}'
```

One `models` row per quantization: the comparison in the dashboard is between rows, so keep `name` identical and vary `quantization`. Record the llama-server version and flags in `notes`; resource numbers depend on them.

If the API runs outside Docker, use `http://127.0.0.1:8080` instead of `host.docker.internal`.

## Resource usage (optional but recommended)

The API cannot see host processes from inside Docker, so a small agent on the Mac samples llama-server's memory and reports the model file size:

```bash
uv run --with psutil scripts/resource_agent.py --process llama-server --port 9108
# then tell the platform where it is:
curl -X PATCH localhost:8000/models/<id> -H 'content-type: application/json' \
  -d '{"resource_agent_url": "http://host.docker.internal:9108"}'
```

Without an agent the resources dimension stores null values with the reason, never an estimate.

Caveat on Apple Silicon: llama.cpp maps the GGUF with `mmap` and runs on Metal with unified memory, so the process RSS undercounts the memory actually pinned for the model. Treat `peak_rss_bytes` on macOS as a lower bound and report `model_file_bytes` alongside it. On Linux/NVIDIA the agent also reads `nvidia-smi`, which is the number that matters there.
