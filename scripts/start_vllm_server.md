# Starting vLLM on a free GPU (Kaggle / Colab) or Modal

vLLM serves the original (unquantized or AWQ/GPTQ) weights behind the same OpenAI-compatible API, so the platform does not care which backend answers.

## Kaggle or Colab notebook (T4 / P100 / L4)

```bash
pip install -q vllm
# 7B in fp16 needs ~15 GB: fits an L4/P100 at short context; use --dtype half on T4.
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --dtype half --max-model-len 4096 \
  --gpu-memory-utilization 0.90 --port 8000 &
```

The notebook has no public IP. Expose the port with a tunnel and register the tunnel URL:

```bash
pip install -q pyngrok && python -c "from pyngrok import ngrok; print(ngrok.connect(8000).public_url)"
# or: cloudflared tunnel --url http://localhost:8000
```

Then from your laptop:

```bash
curl -X POST localhost:8000/models -H 'content-type: application/json' -d '{
  "name": "Qwen2.5-7B-Instruct", "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
  "backend": "vllm", "endpoint_url": "https://<tunnel-host>", "quantization": "fp16",
  "params_b": 7.6, "notes": "vLLM <version> on Kaggle T4, --dtype half"}'
```

Keep `concurrency` modest (2 to 4) through a tunnel; latency numbers will include tunnel overhead, so note that in the run and do not compare them with local numbers.

## Modal

```python
# modal_vllm.py — minimal sketch; see Modal's vLLM example for a production version
import modal
app = modal.App("llm-eval-vllm")
image = modal.Image.debian_slim().pip_install("vllm")

@app.function(image=image, gpu="A10G", timeout=3600)
@modal.web_server(8000)
def serve():
    import subprocess
    subprocess.Popen(["python", "-m", "vllm.entrypoints.openai.api_server",
                      "--model", "Qwen/Qwen2.5-7B-Instruct", "--port", "8000"])
```

`modal serve modal_vllm.py` prints a public URL; register it as above with `"backend": "vllm"`.

## Resource usage on GPU

Peak VRAM is read with `nvidia-smi` on the serving machine, not from the laptop. Until a remote collector exists, run this alongside the server and paste the peak into the model's `notes`, or leave the metric null. Never estimate it.

```bash
nvidia-smi --query-gpu=memory.used --format=csv -l 1 > vram.log
```
