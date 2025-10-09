# GPU Deployment Guide

## Overview
Q-Structurize now supports both CPU and GPU deployments with separate Dockerfiles optimized for each platform.

## Hardware Requirements

### CPU Deployment
- Target: 2x Intel Xeon 6960P (144 cores total)
- RAM: 256GB
- Dockerfile: `Dockerfile.cpu`

### GPU Deployment
- Target: 2x NVIDIA H200 GPUs
- CUDA: 12.1
- NVIDIA Driver: 525+
- RAM: 256GB
- Dockerfile: `Dockerfile.gpu`

## Prerequisites for GPU

### 1. Install NVIDIA Docker Runtime
```bash
# Ubuntu 22.04
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 2. Verify GPU Access
```bash
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

You should see your H200 GPUs listed.

## Deployment Commands

### CPU Deployment
```bash
# Build CPU image
docker-compose --profile cpu build

# Start CPU service
docker-compose --profile cpu up -d

# View logs
docker-compose logs -f q-structurize-cpu

# Stop service
docker-compose --profile cpu down
```

### GPU Deployment
```bash
# Build GPU image (may take 10-15 minutes first time)
docker-compose --profile gpu build

# Start GPU service
docker-compose --profile gpu up -d

# View logs
docker-compose logs -f q-structurize-gpu

# Check GPU utilization
watch -n 1 nvidia-smi

# Stop service
docker-compose --profile gpu down
```

## Configuration Differences

### CPU (Dockerfile.cpu)
```bash
OMP_NUM_THREADS=100              # High thread count for CPU processing
DOCLING_LAYOUT_BATCH_SIZE=64     # Moderate batch size
DOCLING_ACCELERATOR_DEVICE=cpu
```

### GPU (Dockerfile.gpu)
```bash
OMP_NUM_THREADS=8                # Low thread count (GPU does the work)
DOCLING_LAYOUT_BATCH_SIZE=128    # Large batch size (H200 has 141GB HBM3)
DOCLING_ACCELERATOR_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0           # Use first GPU
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Using Both GPUs

To use both H200 GPUs, modify `docker-compose.yml`:

```yaml
environment:
  - NVIDIA_VISIBLE_DEVICES=all  # Use all GPUs

deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 2  # Use both GPUs
          capabilities: [gpu]
```

## Performance Expectations

### CPU Performance
- Processing time: ~10-15 seconds per page
- Best for: Development, testing, small workloads

### GPU Performance (Single H200)
- Processing time: ~0.5-2 seconds per page
- Speedup: **5-10x faster than CPU**
- Best for: Production, high-volume processing

### What Gets Accelerated
- ✅ Layout detection (biggest speedup)
- ✅ Table structure extraction
- ✅ OCR processing (if enabled)
- ✅ Picture classification (if enabled)
- ⚠️ PDF I/O and markdown export (CPU-bound, minimal speedup)

## Monitoring GPU Usage

### Real-time GPU Monitoring
```bash
# In one terminal
watch -n 1 nvidia-smi

# In another terminal
docker-compose logs -f q-structurize-gpu
```

### Expected GPU Logs
When GPU is working, you'll see:
```
🚀 Initializing Docling DocumentConverter (ONE-TIME SETUP)
   📊 Threading:
      - Threads: 8 (OMP_NUM_THREADS)
      - Device: AcceleratorDevice.CUDA  # ← Confirms GPU is active
   🚀 Batching:
      - Layout Batch: 128
      - Table Batch: 128
```

### Performance Logs
```
📊 Performance Breakdown:
   ├─ File I/O:        0.001s (0.1%)
   ├─ Conversion:      1.234s (98.9%)  # ← Much faster with GPU!
   ├─ Markdown Export: 0.012s (1.0%)
   └─ TOTAL:           1.247s
```

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check docker daemon config
cat /etc/docker/daemon.json
# Should contain: {"default-runtime": "nvidia"}
```

### Out of Memory Errors
Reduce batch sizes in `Dockerfile.gpu`:
```bash
DOCLING_LAYOUT_BATCH_SIZE=64  # Reduce from 128
DOCLING_TABLE_BATCH_SIZE=64
```

### Slow Performance
1. Check GPU is actually being used: `nvidia-smi` should show activity
2. Verify `Device: AcceleratorDevice.CUDA` in startup logs
3. Ensure `CUDA_VISIBLE_DEVICES=0` is set

## Switching Between CPU and GPU

You can run both services simultaneously on different ports:

```yaml
# Modify docker-compose.yml
q-structurize-cpu:
  ports:
    - "8878:8000"  # CPU on port 8878

q-structurize-gpu:
  ports:
    - "8879:8000"  # GPU on port 8879
```

Then start both:
```bash
docker-compose --profile cpu --profile gpu up -d
```

## API Endpoint

Both CPU and GPU services expose the same API:
```bash
# Test endpoint
curl http://localhost:8878/

# Parse PDF (replace port 8878 with 8879 for GPU)
curl -X POST http://localhost:8878/parse/file \
  -F "file=@document.pdf" \
  -F "optimize_pdf=true"
```

## Cost-Benefit Analysis

### CPU Infrastructure
- ✅ No special hardware
- ✅ Easier to deploy
- ⚠️ Slower processing
- ⚠️ Higher latency for users

### GPU Infrastructure
- ✅ 5-10x faster
- ✅ Better user experience
- ✅ Can handle more concurrent requests
- ⚠️ Requires NVIDIA GPUs
- ⚠️ More complex setup

**Recommendation:** Use GPU for production, CPU for development/testing.

