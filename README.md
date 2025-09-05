# Novel-Job-Matching — Unified Inference API

A single FastAPI endpoint that abstracts multiple LLM backends behind a clean adapter interface.

## Features

- One endpoint: `POST /inference` for all models; `GET /models` to list model IDs.
- Clean adapter interface with a shared HuggingFace causal base.
- JSON-only contract, consistent errors, and tests.
- Dockerized with a slim Python 3.13 base and uv.

## Quickstart (local)

Prereqs: Python 3.13, uv.

```powershell
# Install deps
uv sync

# Run tests
uv run pytest -q

# Start API (localhost:8000)
uv run uvicorn app.api:app --reload
```

Try it:

```powershell
# List models
curl http://127.0.0.1:8000/models

# Echo example
curl -X POST http://127.0.0.1:8000/inference `
  -H "Content-Type: application/json" `
  -d '{
    "model": "echo",
    "input": {"text": "hello", "extra": {"k": 1}},
    "stream": false
  }'
```

Note: Non-echo adapters return 501 Not Implemented unless Transformers (and GPU-capable Torch) are available locally.
See Real HF Generation below.

## Docker (GPU)

This project includes a GPU-ready container for running real HF models with 4-bit quantization when available.

Prereqs:

- NVIDIA GPU + recent drivers on the host
- NVIDIA Container Toolkit installed (so `docker --gpus all` works)

Build and run:

```powershell
# Build image
docker build -t novel-job-matching:latest .

# Run with GPU, pass HF token, and persist HF cache (to avoid re-downloading models)
docker run --rm -p 8000:8000 --gpus all `
  -e HF_TOKEN=YOUR_HF_TOKEN `
  -v %USERPROFILE%\.cache\huggingface:/root/.cache/huggingface `
  novel-job-matching:latest
```

Linux/macOS volume example:

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  novel-job-matching:latest
```

Verify:

```bash
curl -fsS http://127.0.0.1:8000/models | jq
```

Notes:

- The image installs CUDA-enabled PyTorch (cu121) and bitsandbytes for 4-bit loading.
- The server is GPU-only for HF models; on CPU it returns 501 with a helpful message.
- Set your HF token via `HF_TOKEN` (or `HUGGINGFACE_TOKEN`/`HF_API_TOKEN`).
- VRAM guidance (approximate; varies by model and config):
    - 9B class with 4-bit: ~6–8 GB VRAM
    - 24B class with 4-bit: ~14–18 GB VRAM
- You can tune quantization in `BaseHFCausalAdapter._quantization_config()` if needed.

## API Contract

- Request

```json
{
  "model": "example-model",
  "input": {
    "text": "some input"
  },
  "params": {
    "temperature": 0.2
  },
  "stream": false
}
```

- Response

```json
{
  "model": "example-model",
  "output": {
    "result": "..."
  },
  "usage": {
    "tokens": 0
  }
}
```

- Error

```json
{
  "error": {
    "code": "string",
    "message": "string"
  }
}
```

## Repo layout

- `app/api.py` — FastAPI app with `/inference` and `/models`.
- `app/adapters/` — `echo` (working) and HF-backed adapters (base + model stubs).
- `app/core/` — contracts and registry.
- `app/utils/text_utils.py` — chunking, JSON parsing, de-dupe.
- `tests/` — API contract tests.

## Notes

- The HF adapters share common logic via `BaseHFCausalAdapter`. Override `build_messages` for model-specific chat
  templates if needed.
- For gated models, pass tokens via env vars. You can mount or bake secrets using standard Docker practices.

---

## Real HF Generation

The HF-backed adapters (GLM4-9B, GLM4-Z1-9B, Llama3.1-Nemotron-8B, Mistral-Small-24B, Qwen3-8B) are wired to run real
generation using Hugging Face Transformers when available.

- Dependencies (local):
    - Install transformers and a CUDA-enabled torch per https://pytorch.org/get-started/locally/
    - Quantization (optional): if `bitsandbytes` is installed and a CUDA GPU is available, 4-bit quantization is enabled
      automatically.

- Authentication for gated models:
    - Set an env var: `HF_TOKEN`, `HUGGINGFACE_TOKEN`, or `HF_API_TOKEN`.
    - Or place your token in a file named `huggingface-token` at the project root (not committed).

- Runtime behavior:
    - Adapters load lazily on first request. Without GPU, they return 501 with a clear message.
    - Tokenizer chat templates are used when available; otherwise a safe fallback prompt is applied.
    - Some adapters set a safe pad token where the tokenizer lacks one.
    - Outputs are parsed for a JSON object/array; common keys (`skills`, `experience`, `qualifications`, `education`)are
      flattened to a single `requirements` list and de-duplicated.

- Fast testing without models:
    - You can pass `params.mock_text` to simulate a model’s raw output; adapters will parse the JSON and return
      structured results without loading HF models:
      ```json
      {
        "model": "glm4-9b",
        "input": {"markdown": "# Title\nSome content", "chunk_size": 128},
        "params": {"mock_text": "```json\\n{\\n  \\\"skills\\\": [\\\"Python\\\"],\\n  \\\"experience\\\": [\\\"3+ years\\\"]\\n}\\n```"},
        "stream": false
      }
      ```
