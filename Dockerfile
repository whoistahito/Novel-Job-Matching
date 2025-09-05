# syntax=docker/dockerfile:1

FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/root/.local/bin:${PATH}" \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git && \
    rm -rf /var/lib/apt/lists/*

# Install uv (https://astral.sh/uv)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -y

WORKDIR /app

# Copy lockfiles first for better caching
COPY pyproject.toml uv.lock ./

# Sync dependencies (no dev deps inside runtime image)
RUN uv sync --frozen

# Install CUDA-enabled PyTorch wheels and bitsandbytes for 4-bit quantization.
# Note: Requires running the container with `--gpus all` and NVIDIA Container Toolkit on the host.
RUN uv pip uninstall -y torch torchvision torchaudio || true
RUN uv pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio && \
    uv pip install bitsandbytes

# Copy the application code
COPY app ./app
COPY models ./models
COPY README.md* ./

EXPOSE 8000

# Basic healthcheck: the app must respond on /models
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/models || exit 1

# Default command: run the API server
CMD ["uv", "run", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
