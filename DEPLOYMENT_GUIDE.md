# Q-Structurize Deployment Guide

## Current Status

✅ **GPU Configuration**: Ready for Linux with H200 GPU  
✅ **Cache Optimization**: Fixed model caching issues  
✅ **H200 Optimizations**: Full precision, 32K tokens, KV cache  

## For macOS Testing (Current)

Since you're on macOS, use the macOS-compatible version:

```bash
# Use macOS version (no GPU support)
docker-compose -f docker-compose.macos.yml up -d
```

**Expected on macOS:**
- ✅ Application runs
- ❌ CPU processing only (`Accelerator device: 'cpu'`)
- ✅ Cache debugging works
- ⚠️ Slow processing (10-30 seconds per page)

## For Linux Server with H200 GPU

Use the main docker-compose.yml:

```bash
# On your Linux server with H200 GPU
docker-compose up -d
```

**Expected on Linux with H200:**
- ✅ GPU detected (`Accelerator device: 'cuda:0'`)
- ✅ Fast processing (2-5 seconds per page)
- ✅ Models cached after first run
- ✅ H200 optimizations active

## Key Differences

### macOS (docker-compose.macos.yml)
- **No GPU support** (macOS limitation)
- **CPU processing only**
- **Slower performance** (10-30 seconds per page)
- **Good for testing** cache and API functionality

### Linux with H200 (docker-compose.yml)
- **Full GPU support** with NVIDIA drivers
- **H200 optimizations** active
- **Fast processing** (2-5 seconds per page)
- **Production ready**

## Current Issue

You're running on **macOS** but the configuration is set for **Linux with GPU**. That's why you're seeing:

- ❌ `Accelerator device: 'cpu'` (should be `cuda:0` on Linux)
- ❌ Slow processing (10-30 seconds per page)
- ❌ Models downloading every time

## Solution

### For Testing on macOS:
```bash
docker-compose -f docker-compose.macos.yml up -d
```

### For Production on Linux:
```bash
# On your Linux server with H200 GPU
docker-compose up -d
```

## Next Steps

1. **Test on macOS** with `docker-compose.macos.yml` to verify cache functionality
2. **Deploy on Linux** with `docker-compose.yml` for GPU performance
3. **Monitor GPU usage** with `nvidia-smi` on Linux
4. **Check logs** for `Accelerator device: 'cuda:0'` on Linux

## Expected Results

### macOS Testing:
- Cache debugging works
- Models download to cache directories
- CPU processing (slow but functional)

### Linux Production:
- GPU detected and used
- Fast processing (2-5 seconds per page)
- Models cached after first run
- H200 optimizations active
