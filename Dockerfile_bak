FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Docling VLM
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libgthread-2.0-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    python3-dev \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for CUDA (will be available via nvidia-docker runtime)
ENV CUDA_VISIBLE_DEVICES=0

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create cache directories for model storage
RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers /app/.cache/torch

# Set environment variables for model caching
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV TORCH_HOME=/app/.cache/torch
ENV HF_HUB_CACHE=/app/.cache/huggingface

# Copy application code
COPY . .

# Pre-download and cache the GraniteDocling VLM model during build
# This eliminates the "black box" download time during runtime
RUN echo "🚀 Pre-downloading GraniteDocling VLM model..." && \
    python scripts/preload_models.py

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]