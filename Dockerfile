# CPU-only Dockerfile for Q-Structurize with StandardPdfPipeline
FROM python:3.11-slim

WORKDIR /app

# Build-time environment variables (rarely change - keeps cache valid)
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DOCLING_ARTIFACTS_PATH=/root/.cache/docling/models \
    HF_HOME=/app/.cache/huggingface \
    PATH="/usr/local/bin:${PATH}"

# System deps (lean)
RUN apt-get update && apt-get install -y --no-install-recommends \
      gcc g++ libgomp1 libgl1 libglib2.0-0 curl wget git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ---- cache-friendly boundary ----
# Install Python deps first (cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Prepare cache directories (these will be volumes at runtime)
RUN mkdir -p /root/.cache/docling/models ${HF_HOME} /app/uploads

# (Optional) Pre-download models; will fill the mounted volume and persist
# Comment out if you prefer first-run download to populate the volume.
RUN echo "Pre-downloading Docling models (optional)..." && \
    docling-tools models download || true

# Verify (non-fatal if empty on fresh volume)
RUN ls -lah /root/.cache/docling/models || true

# Runtime environment variables (change these without invalidating cache layers above)
ENV LOG_LEVEL=DEBUG \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=32 \
    MKL_NUM_THREADS=32 \
    OPENBLAS_NUM_THREADS=32 \
    NUMEXPR_NUM_THREADS=32 \
    TORCH_NUM_THREADS=32 \
    KMP_BLOCKTIME=1 \
    KMP_SETTINGS=1 \
    KMP_AFFINITY="granularity=fine,compact,1,0" \
    MALLOC_ARENA_MAX=4 \
    DOCLING_ENABLE_OCR=false \
    DOCLING_OCR_LANGUAGES=en \
    DOCLING_DO_TABLE_STRUCTURE=true \
    DOCLING_TABLE_MODE=fast \
    DOCLING_DO_CELL_MATCHING=false \
    DOCLING_DO_CODE_ENRICHMENT=false \
    DOCLING_DO_FORMULA_ENRICHMENT=false \
    DOCLING_DO_PICTURE_CLASSIFICATION=false \
    DOCLING_DO_PICTURE_DESCRIPTION=false \
    DOCLING_LAYOUT_BATCH_SIZE=64 \
    DOCLING_OCR_BATCH_SIZE=32 \
    DOCLING_TABLE_BATCH_SIZE=64 \
    DOCLING_QUEUE_MAX_SIZE=1000 \
    DOCLING_BATCH_TIMEOUT=0.5 \
    DOCLING_ACCELERATOR_DEVICE=cpu

# Finally copy app sources (changes here don't invalidate pip layer)
COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

