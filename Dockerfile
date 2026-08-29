FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# libgomp1 is required at runtime by the CPU build of torch (sentence-transformers'
# dependency) - without it, importing torch fails on slim/minimal base images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU-only torch build first so sentence-transformers' dependency
# resolution is satisfied by it instead of pulling the much larger default
# CUDA build - this app has no GPU in its deploy target.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch && \
    pip install -r requirements.txt

# Pre-download and cache the embedding model at build time, so it ships baked
# into the image instead of being fetched from the Hugging Face Hub on the
# first request - avoids a slow (and rate-limited) cold start in production.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# The model above is already cached in the image, so block any further
# Hugging Face Hub network calls at runtime rather than merely avoiding them.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
