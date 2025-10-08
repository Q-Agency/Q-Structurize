# CPU-only Dockerfile for Q-Structurize with StandardPdfPipeline
# Includes layout analysis, OCR, and table extraction
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    # Tokenizers - disable parallelism to avoid fork warnings
    TOKENIZERS_PARALLELISM=true \
    # CPU threading optimizations for 2x Xeon 6960P (144 cores total)
    OMP_NUM_THREADS=100 \
    MKL_NUM_THREADS=100 \
    OPENBLAS_NUM_THREADS=100 \
    NUMEXPR_NUM_THREADS=100 \
    TORCH_NUM_THREADS=100 \
    # CPU-specific optimizations for Intel Xeon
    KMP_BLOCKTIME=1 \
    KMP_SETTINGS=1 \
    KMP_AFFINITY="granularity=fine,compact,1,0" \
    # Memory optimization
    MALLOC_ARENA_MAX=4 \
    # Docling Parser Configuration
    DOCLING_ENABLE_OCR=false \
    DOCLING_OCR_LANGUAGES=en \
    DOCLING_DO_TABLE_STRUCTURE=false \
    DOCLING_TABLE_MODE=fast \
    DOCLING_DO_CELL_MATCHING=false \
    DOCLING_DO_CODE_ENRICHMENT=false \
    DOCLING_DO_FORMULA_ENRICHMENT=false \
    DOCLING_DO_PICTURE_CLASSIFICATION=false \
    DOCLING_DO_PICTURE_DESCRIPTION=false \
    DOCLING_LAYOUT_BATCH_SIZE=32 \
    DOCLING_OCR_BATCH_SIZE=32 \
    DOCLING_TABLE_BATCH_SIZE=32 \
    DOCLING_QUEUE_MAX_SIZE=1000 \
    DOCLING_BATCH_TIMEOUT=0.5 \
    DOCLING_ACCELERATOR_DEVICE=cpu \
    # Docling models cache (where docling-tools downloads models)
    DOCLING_ARTIFACTS_PATH=/root/.cache/docling/models \
    # HuggingFace cache
    HF_HOME=/app/.cache/huggingface \
    PATH="/usr/local/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    gcc \
    g++ \
    # PDF and image processing libraries
    libglib2.0-0 \
    libgomp1 \
    # For OCR (EasyOCR) - OpenGL libraries
    libgl1 \
    libglib2.0-0 \
    # Utilities
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directories
RUN mkdir -p /root/.cache/docling/models ${HF_HOME}

# Pre-download Docling StandardPdfPipeline models using docling-tools CLI
# Note: Models will download automatically on first use if not pre-downloaded
RUN echo "============================================================" && \
    echo "🚀 PRE-DOWNLOADING DOCLING MODELS (Optional)" && \
    echo "============================================================" && \
    echo "" && \
    echo "Models will be downloaded on first API call:" && \
    echo "  - Layout Detection (DocLayNet) ~100MB" && \
    echo "  - Table Extraction (TableFormer) ~200MB" && \
    echo "  - OCR Models (EasyOCR) ~100MB" && \
    echo "" && \
    echo "Total size: ~400MB" && \
    echo "First run will take 3-5 minutes to download models" && \
    echo "Subsequent runs will be fast (~5 seconds)" && \
    echo "" && \
    docling-tools models download && \
    echo "" && \
    echo "✅ Models downloaded successfully!" && \
    echo "============================================================"

# Verify cache contents
RUN echo "📦 Verifying model cache:" && \
    ls -lah /root/.cache/docling/models/ 2>/dev/null || echo "⚠️  Models will download on first use" && \
    du -sh /root/.cache/docling/models 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /app/uploads

# Copy application code
COPY . .

# Expose application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

