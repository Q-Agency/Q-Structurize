# GPU Deployment Guide

## Overview
Q-Structurize is optimized for NVIDIA H200 GPUs with CUDA 12.1 for high-performance PDF processing.

## Hardware Requirements

- **GPUs**: 2x NVIDIA H200 (141GB HBM3 each)
- **CUDA**: 12.1
- **NVIDIA Driver**: 525+
- **RAM**: 256GB
- **OS**: Ubuntu 22.04 LTS

## Prerequisites

### NVIDIA Container Toolkit

You mentioned you already have **nvidia-container-toolkit** installed. Verify it's working:

```bash
# Test GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see your H200 GPUs listed. ✅

## Deployment Commands

### Build & Start (Simple!)

```bash
# Build GPU image (first time: ~10-15 minutes)
docker-compose build

# Start service
docker-compose up -d

# View logs
docker-compose logs -f q-structurize

# Check GPU utilization
watch -n 1 nvidia-smi

# Stop service
docker-compose down
```

### Quick Restart After Code Changes

```bash
# Thanks to optimized Docker layers, rebuilds are fast (~5-10 seconds)
docker-compose build && docker-compose up -d
```

## Configuration

### GPU-Optimized Settings (in Dockerfile)

```bash
# Threading (reduced - GPU does the work)
OMP_NUM_THREADS=8
MKL_NUM_THREADS=8
TORCH_NUM_THREADS=8

# Batch sizes (increased for H200's 141GB HBM3)
DOCLING_LAYOUT_BATCH_SIZE=128
DOCLING_OCR_BATCH_SIZE=128
DOCLING_TABLE_BATCH_SIZE=128

# GPU Configuration
CUDA_VISIBLE_DEVICES=0                              # Use first H200
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True    # Better memory management
DOCLING_ACCELERATOR_DEVICE=cuda                     # Enable GPU acceleration
```

### Using Both H200 GPUs

Modify `docker-compose.yml`:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all  # Use both GPUs

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Use both GPUs
          capabilities: [gpu]
```

## Performance

### Expected Processing Speed

- **Processing time**: ~0.5-2 seconds per page
- **Throughput**: ~30-120 pages per minute (single GPU)
- **vs CPU**: 5-10x faster than CPU-only processing

### What Gets Accelerated

- ✅ **Layout detection** (biggest speedup)
- ✅ **Table structure extraction**
- ✅ **OCR processing** (if enabled)
- ✅ **Picture classification** (if enabled)
- ⚠️ PDF I/O and markdown export (CPU-bound, minimal speedup)

## Monitoring

### Real-time GPU Monitoring

```bash
# Terminal 1: GPU usage
watch -n 1 nvidia-smi

# Terminal 2: Application logs
docker-compose logs -f q-structurize
```

### Expected Startup Logs

```
🔍 Docling pipeline profiling enabled
🚀 Initializing Docling DocumentConverter (ONE-TIME SETUP)
⚙️  Configuration (from Dockerfile ENV):
   📊 Threading:
      - Threads: 8 (OMP_NUM_THREADS)
      - Device: AcceleratorDevice.CUDA  # ← Confirms GPU is active!
   🚀 Batching:
      - Layout Batch: 128
      - Table Batch: 128
✅ Converter initialized in 0.xx seconds
```

### Performance Logs (Per Request)

```
📄 Starting PDF parsing...
⏱️  Temp file write: 0.001s
⏳ Processing document...
   └─ Step 1: Document conversion
   ✅ Conversion complete: 1.234s  # ← Fast with GPU!
   └─ Step 2: Export to markdown
   ✅ Export complete: 0.012s
📊 Performance Breakdown:
   ├─ File I/O:        0.001s (0.1%)
   ├─ Conversion:      1.234s (98.9%)  # ← GPU-accelerated
   ├─ Markdown Export: 0.012s (1.0%)
   └─ TOTAL:           1.247s
```

## Troubleshooting

### GPU Not Detected

**Check if GPU is visible:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Verify Docker daemon config:**
```bash
cat /etc/docker/daemon.json
# Should contain: {"default-runtime": "nvidia"}
```

**Restart Docker if needed:**
```bash
sudo systemctl restart docker
```

### Startup Shows "Device: AcceleratorDevice.CPU"

This means GPU is not being used. Check:

1. **NVIDIA runtime available?**
   ```bash
   docker info | grep -i runtime
   # Should show: Runtimes: nvidia runc
   ```

2. **Environment variable set?**
   ```bash
   docker-compose exec q-structurize env | grep CUDA
   # Should show: CUDA_VISIBLE_DEVICES=0
   ```

3. **Rebuild and restart:**
   ```bash
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

### Out of Memory Errors

H200 has 141GB HBM3, but if you still get OOM:

**Option 1: Reduce batch sizes in `Dockerfile`:**
```bash
DOCLING_LAYOUT_BATCH_SIZE=64  # Reduce from 128
DOCLING_TABLE_BATCH_SIZE=64
```

**Option 2: Check memory usage:**
```bash
nvidia-smi
# Look at Memory-Usage column
```

### Slow Performance

1. **Verify GPU is active:**
   ```bash
   nvidia-smi
   # Should show processes under "Processes:" section
   ```

2. **Check device in logs:**
   ```bash
   docker-compose logs q-structurize | grep "Device:"
   # Must show: AcceleratorDevice.CUDA
   ```

3. **Monitor during processing:**
   ```bash
   watch -n 0.1 nvidia-smi
   # GPU utilization should spike to 80-100% during processing
   ```

## API Usage

### Test Endpoint

```bash
curl http://localhost:8878/
```

### Parse PDF

```bash
curl -X POST http://localhost:8878/parse/file \
  -F "file=@document.pdf" \
  -F "optimize_pdf=true"
```

### Get Parser Info

```bash
curl http://localhost:8878/parsers/info
```

Should show:
```json
{
  "available": true,
  "pipeline": "StandardPdfPipeline with ThreadedPdfPipelineOptions",
  "configuration": {
    "accelerator_device": "cuda",  // ← Confirms GPU
    "num_threads": 8,
    ...
  }
}
```

## Files Structure

```
QStructurize/
├── Dockerfile              # → Symlink to Dockerfile.gpu
├── Dockerfile.gpu          # Main GPU Dockerfile (CUDA 12.1)
├── Dockerfile.cpu          # Backup CPU-only version
├── docker-compose.yml      # GPU-enabled service
├── requirements.txt        # Python dependencies
├── main.py                 # FastAPI application
└── app/
    └── services/
        └── docling_parser.py  # GPU-ready parser
```

## Performance Tips

1. **Keep models warm**: The first request after startup loads models into GPU memory (~1-2 seconds). Subsequent requests are instant.

2. **Batch processing**: If processing multiple PDFs, send concurrent requests to maximize GPU utilization.

3. **Monitor GPU utilization**: Use `nvidia-smi` to ensure GPU is at 80-100% during peak processing.

4. **Use both GPUs**: For high-throughput scenarios, configure `NVIDIA_VISIBLE_DEVICES=all` and scale horizontally.

## Production Recommendations

- ✅ Use GPU for all production workloads
- ✅ Enable batch sizes of 128+ for H200
- ✅ Monitor with `nvidia-smi` and application logs
- ✅ Set up health checks (included in docker-compose.yml)
- ✅ Use persistent volumes for model caching
- ⚠️ Keep Dockerfile.cpu as backup for development/testing

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f q-structurize`
2. Verify GPU: `nvidia-smi`
3. Check application health: `curl http://localhost:8878/`
