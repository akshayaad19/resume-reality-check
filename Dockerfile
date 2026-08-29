FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    EMBEDDING_MODEL_DIR=/app/.cache/embedding_model

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

# Pre-download and cache the ONNX embedding model + tokenizer at build time,
# so they ship baked into the image instead of being fetched over the network
# on the first request - avoids a slow cold start in production. Embeddings
# run on ONNX Runtime rather than sentence-transformers/PyTorch: importing
# torch + sentence-transformers adds ~340MB of baseline memory (measured),
# which alone left no safety margin on a 512MB deployment target - see
# debugging-log.md.
RUN mkdir -p "$EMBEDDING_MODEL_DIR" && \
    python - <<'PYEOF'
import os, urllib.request
repo = "sentence-transformers/all-MiniLM-L6-v2"
model_dir = os.environ["EMBEDDING_MODEL_DIR"]
files = {
    "model.onnx": f"https://huggingface.co/{repo}/resolve/main/onnx/model.onnx",
    "tokenizer.json": f"https://huggingface.co/{repo}/resolve/main/tokenizer.json",
}
for name, url in files.items():
    urllib.request.urlretrieve(url, os.path.join(model_dir, name))
PYEOF

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
