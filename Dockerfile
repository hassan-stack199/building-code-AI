FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (PDFs, app.py, build_index.py, etc.)
COPY . .

# Force fastembed/HF caches to live INSIDE /app so they get baked into the
# image and survive container restarts.
ENV HF_HOME=/app/.cache/huggingface
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed
ENV HOME=/app

# Pre-build the embedding index at IMAGE BUILD TIME so the container starts
# with everything ready. Downloads the embedding model AND embeds every
# chunk in ./regulations/. Result 