# 🚀 Model Preload Guide for Q-Structurize

This guide explains how to pre-download and cache the GraniteDocling VLM model to eliminate the "black box" download time during runtime.

## 🎯 Problem Solved

**Before**: The application would get stuck for 10+ minutes during first run with no indication if it was downloading the model or processing.

**After**: The model is pre-downloaded during Docker build, so runtime startup is fast and predictable.

## 📋 What's Changed

### 1. Model Preload Script (`scripts/preload_models.py`)
- Downloads and caches the GraniteDocling VLM model during Docker build
- Provides detailed logging of the download process
- Configures cache directories properly
- Optimizes for H200 GPU performance

### 2. Updated Dockerfile
- Pre-downloads the model during `docker build`
- Sets up proper cache directories
- Eliminates runtime download delays

### 3. Enhanced Logging
- Clear distinction between model loading and document processing
- Timing information for both phases
- Progress indicators with emojis

## 🛠️ How to Use

### Option 1: Build with Model Preload (Recommended)

```bash
# Build the Docker image with model preload
docker-compose -f docker-compose.gpu.yml build

# This will show progress like:
# 🚀 Pre-downloading GraniteDocling VLM model...
# === STARTING MODEL PRE-DOWNLOAD ===
# ⏳ This may take several minutes for first-time download...
# ✅ Model download and caching completed
```

### Option 2: Test Model Preload Locally

```bash
# Test the preload script locally (requires docling installed)
python scripts/preload_models.py

# Test if the model is properly cached
python scripts/test_model_preload.py
```

## 📊 Expected Performance

### Before (Runtime Download)
- **First run**: 10-15 minutes (model download + processing)
- **Subsequent runs**: 2-5 minutes (processing only)
- **Problem**: No indication of progress, appears "stuck"

### After (Pre-downloaded)
- **Build time**: 10-15 minutes (one-time model download)
- **Runtime**: 30-60 seconds (fast model loading from cache)
- **Benefit**: Predictable startup time, clear progress logging

## 🔍 Logging Improvements

### Model Loading Phase
```
⏳ Loading VLM model (this should be fast if pre-cached)...
✅ Docling VLM converter initialized successfully in 2.34 seconds
✅ Model loaded from cache - no download required
```

### Document Processing Phase
```
⏳ Processing document (VLM inference in progress)...
✅ VLM processing completed in 4.12 seconds
```

## 🐳 Docker Build Process

The updated Dockerfile now includes:

1. **Cache Directory Setup**
   ```dockerfile
   RUN mkdir -p /app/.cache/huggingface /app/.cache/transformers /app/.cache/torch
   ```

2. **Environment Variables**
   ```dockerfile
   ENV HF_HOME=/app/.cache/huggingface
   ENV TRANSFORMERS_CACHE=/app/.cache/transformers
   ENV TORCH_HOME=/app/.cache/torch
   ENV HF_HUB_CACHE=/app/.cache/huggingface
   ```

3. **Model Pre-download**
   ```dockerfile
   RUN echo "🚀 Pre-downloading GraniteDocling VLM model..." && \
       python scripts/preload_models.py
   ```

## 🚨 Troubleshooting

### If Model Preload Fails
```bash
# Check if docling is properly installed
pip install docling

# Run preload script manually to see errors
python scripts/preload_models.py
```

### If Runtime Still Shows Long Delays
```bash
# Check cache directories
ls -la /app/.cache/huggingface/
ls -la /app/.cache/transformers/
ls -la /app/.cache/torch/

# Verify environment variables
echo $HF_HOME
echo $TRANSFORMERS_CACHE
```

### Cache Verification
The preload script will show cache contents:
```
=== CACHE VERIFICATION ===
Cache contents in /app/.cache/huggingface: ['models--ibm--granite-docling-vlm']
Cache contents in /app/.cache/transformers: ['models--ibm--granite-docling-vlm']
Cache contents in /app/.cache/torch: ['hub']
```

## 📈 Benefits

1. **Predictable Startup**: No more "black box" waiting
2. **Faster Runtime**: Model loads from cache in seconds
3. **Better UX**: Clear progress indicators and timing
4. **Offline Capable**: Works without internet after build
5. **Production Ready**: Consistent deployment experience

## 🔄 Migration from Old Setup

If you have an existing deployment:

1. **Stop current containers**:
   ```bash
   docker-compose down
   ```

2. **Rebuild with model preload**:
   ```bash
   docker-compose -f docker-compose.gpu.yml build --no-cache
   ```

3. **Start with new image**:
   ```bash
   docker-compose -f docker-compose.gpu.yml up -d
   ```

The first build will take longer (10-15 minutes) but subsequent runs will be much faster!
