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
    libgthread-2.0-0 \
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
             /app/uploads \
             /app/scripts

# Copy preload script (before copying full application code)
COPY scripts/preload_models.py scripts/

# Pre-download and cache the GraniteDocling VLM model during build
# This eliminates download time during runtime - model will be ready instantly!
RUN echo "🚀 Starting model pre-download process..." && \
    python scripts/preload_models.py && \
    echo "✅ Model pre-download completed successfully!"

# Copy application code LAST (for better layer caching during development)
COPY . .

# Verify the model is properly cached
RUN python -c "import os; \
    cache_path = '/app/.cache/huggingface/hub/models--ibm-granite--granite-docling-258M'; \
    assert os.path.exists(cache_path), f'❌ Model cache not found at {cache_path}'; \
    print(f'✅ Model verified at {cache_path}')"

# Expose application port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]