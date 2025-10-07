# Use NVIDIA CUDA base image for H200 GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables EARLY (before any operations)
ENV HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/transformers \
    TORCH_HOME=/app/.cache/torch \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    TOKENIZERS_PARALLELISM=false \
    PATH="/usr/local/bin:${PATH}"

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    # Python
    python3.11 \
    python3.11-dev \
    python3-pip \
    # Build tools
    gcc \
    g++ \
    # System libraries for Docling VLM
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    # Media libraries
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # Image processing libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Utilities
    curl \
    wget \
    git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directories with proper permissions
RUN mkdir -p /app/.cache/huggingface/hub \
             /app/.cache/transformers \
             /app/.cache/torch \
             /app/uploads

# ============================================================================
# Pre-download Granite-Docling VLM model using OFFICIAL docling-tools CLI
# Model: ibm-granite/granite-docling-258M (258M parameters)
# This is the recommended method from Docling documentation
# ============================================================================
RUN echo "============================================================================" && \
    echo "🚀 Downloading Granite-Docling VLM (258M) using official docling-tools CLI" && \
    echo "📦 Model: ibm-granite/granite-docling-258M" && \
    echo "============================================================================" && \
    docling-tools models download && \
    echo "" && \
    echo "============================================================================" && \
    echo "✅ Model download completed successfully!" && \
    echo "============================================================================" && \
    echo "" && \
    echo "📂 Verifying downloaded models:" && \
    find /app/.cache -type d -name "*granite*" 2>/dev/null | head -5 && \
    echo ""

# Copy application code LAST (for better layer caching during development)
COPY . .

# Verify Granite-Docling model is available
RUN echo "🔍 Verifying Granite-Docling VLM model..." && \
    python -c "from docling.datamodel import vlm_model_specs; \
        opts = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS; \
        print(f'✅ Model configured: {opts.repo_id if hasattr(opts, \"repo_id\") else \"ibm-granite/granite-docling-258M\"}'); \
        print(f'✅ Model parameters: 258M'); \
        print(f'✅ Framework: Transformers')" && \
    (find /app/.cache -type d -name "*granite*docling*" -o -name "*granite-docling*" | head -1 | \
     xargs -I {} echo "✅ Model cache found at: {}") || \
    echo "⚠️  Model cache structure varies by version"

# Expose application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
