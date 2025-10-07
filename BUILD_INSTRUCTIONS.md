# 🏗️ Build Instructions for Q-Structurize

## ✅ Fixed Issues

1. **Removed `libgthread-2.0-0`** - This package doesn't exist in Ubuntu 22.04 (functionality is included in `libglib2.0-0`)
2. **Added `git`** - Useful for some Python packages that may need it
3. **Added `.dockerignore`** - Speeds up Docker builds by excluding unnecessary files
4. **Fixed import statement** - Added `time` module at the top of `docling_parser.py`

## 🚀 Build and Run Commands

### Option 1: Using docker-compose (Recommended for GPU)

```bash
# Build the image with model preload
docker-compose -f docker-compose.gpu.yml build

# Start the service
docker-compose -f docker-compose.gpu.yml up -d

# Watch the logs
docker-compose -f docker-compose.gpu.yml logs -f
```

### Option 2: Using regular docker-compose

```bash
# Build and start
docker-compose up --build -d

# Watch logs
docker-compose logs -f
```

### Option 3: Direct Docker commands

```bash
# Build the image
docker build -t q-structurize:latest .

# Run the container
docker run -d \
  --name q-structurize \
  --gpus all \
  -p 8878:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/cache:/app/.cache \
  --shm-size 16g \
  q-structurize:latest
```

## 📊 Expected Build Timeline

### First Build (with model download)
```
[0-2 min]   Installing system dependencies
[2-4 min]   Installing Python packages
[4-15 min]  Downloading and caching Granite VLM model
[15-16 min] Verifying model cache
[16-17 min] Final image creation
```

**Total: ~15-17 minutes**

### Subsequent Builds (model cached)
```
[0-2 min]   Using cached layers
[2-3 min]   Copying application code
```

**Total: ~2-3 minutes** (if only app code changed)

## 🎯 What to Expect During Build

You'll see detailed progress messages like:

```
======================================================================
🚀 Q-STRUCTURIZE MODEL PRELOAD PROCESS
======================================================================

📂 Setting up cache directories...
✅ Created cache directory: /app/.cache/huggingface/hub
✅ Created cache directory: /app/.cache/transformers
✅ Created cache directory: /app/.cache/torch

🤖 Preloading Granite Docling VLM model...
======================================================================
🚀 STARTING MODEL PRE-DOWNLOAD
======================================================================
⏳ Initializing DocumentConverter (this will download the model)...
⏳ First-time download may take 5-10 minutes depending on network speed...
======================================================================

[... model downloads ...]

✅ DocumentConverter initialized in 487.23 seconds
✅ Model download and caching completed
======================================================================
🎉 MODEL PRELOAD COMPLETED SUCCESSFULLY!
======================================================================
```

## 🏃 Runtime Startup

After the build completes, runtime startup will be **fast** (30-60 seconds):

```
2025-10-07 12:00:00 - INFO - ⏳ Loading VLM model (this should be fast if pre-cached)...
2025-10-07 12:00:02 - INFO - ✅ Docling VLM converter initialized successfully in 2.34 seconds
2025-10-07 12:00:02 - INFO - ✅ Model loaded from cache - no download required
```

## 🔍 Verify Installation

Once running, test the service:

```bash
# Check health
curl http://localhost:8878/

# Check API docs
open http://localhost:8878/docs

# Check parser info
curl http://localhost:8878/parsers/info
```

## 🐛 Troubleshooting

### If build fails at package installation
```bash
# Check your internet connection
# Check Docker has enough disk space (need ~20GB)
docker system df
```

### If model download is slow
```bash
# This is normal - the Granite VLM model is ~512 MB
# Average download time: 5-10 minutes
# Be patient and watch the logs
```

### If build succeeds but runtime fails
```bash
# Check logs
docker-compose logs q-structurize

# Check GPU access
docker exec q-structurize nvidia-smi

# Verify cache
docker exec q-structurize ls -la /app/.cache/huggingface/hub/
```

## 🎉 Success Indicators

Build successful when you see:
- ✅ All packages installed
- ✅ Model pre-download completed
- ✅ Model cache verified
- ✅ Container started

Runtime successful when you see:
- ✅ Model loaded from cache in 2-5 seconds
- ✅ Uvicorn running on http://0.0.0.0:8000
- ✅ API documentation accessible

## 📝 Notes

- The cache directory will persist models between container restarts
- Rebuilds will be much faster after the first build
- Model verification ensures the build fails early if download fails
- Health check gives the service 120 seconds to start before monitoring
