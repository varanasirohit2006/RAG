FROM node:22-alpine AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache/hub \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    EMBEDDING_REVISION=1110a243fdf4706b3f48f1d95db1a4f5529b4d41 \
    EMBEDDING_DEVICE=cpu \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# CPU wheels avoid an unnecessary CUDA download for local embeddings.
RUN python -m pip install --no-cache-dir torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt requirements-ocr.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Build-time download; runtime uses this exact revision without model-network access.
RUN HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c \
    "import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ['EMBEDDING_MODEL'], revision=os.environ['EMBEDDING_REVISION'], allow_patterns=['*.json', '*.safetensors', '*.txt', '1_Pooling/*'])"

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home app
COPY --chown=app:app . /app
COPY --from=frontend --chown=app:app /frontend/dist /app/frontend/dist
RUN mkdir -p /app/docs /app/db /app/.cache /app/evaluation_runs \
    && chown -R app:app /app /opt/hf-cache

USER app
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/api/health', timeout=3)"

CMD ["python", "-B", "run_app.py", "--host", "0.0.0.0", "--port", "8501"]
