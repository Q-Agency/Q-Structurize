# GPU Troubleshooting Guide

## Problem: Getting CPU instead of GPU on Linux

If you're getting `Accelerator device: 'cpu'` on your Linux machine, follow these steps:

## Step 1: Check nvidia-docker Installation

```bash
# Check if nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.60.13    Driver Version: 525.60.13    CUDA Version: 11.8  |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA H200         Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   45C    P0    70W / 700W |      0MiB / 81920MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

**If this fails:**
```bash
# Install nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Step 2: Test GPU Access

```bash
# Run the test script
./test_gpu.sh
```

**Expected output:**
```
PyTorch CUDA available: True
CUDA device count: 1
Current device: 0
Device name: NVIDIA H200
```

## Step 3: Try Different Docker Compose Configurations

### Option A: Simple Configuration
```bash
docker-compose -f docker-compose.simple.yml up -d
```

### Option B: Alternative GPU Configuration
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

### Option C: Manual Docker Run (for testing)
```bash
docker run --rm --gpus all -p 8878:8000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/cache:/app/.cache \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HOME=/app/.cache/huggingface \
  -e TRANSFORMERS_CACHE=/app/.cache/transformers \
  -e TORCH_HOME=/app/.cache/torch \
  qstructurize-q-structurize:latest
```

## Step 4: Check Container Logs

```bash
# Check for GPU detection
docker-compose logs | grep -E "(cuda|gpu|Accelerator device)"

# Should show: "Accelerator device: 'cuda:0'"
```

## Step 5: Verify Inside Container

```bash
# Check GPU inside container
docker-compose exec q-structurize nvidia-smi

# Check PyTorch CUDA
docker-compose exec q-structurize python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
"
```

## Common Issues and Solutions

### Issue 1: nvidia-docker not installed
**Solution:** Install nvidia-docker2 as shown above

### Issue 2: Docker daemon not restarted
**Solution:** 
```bash
sudo systemctl restart docker
```

### Issue 3: Wrong Docker Compose version
**Solution:** Use Docker Compose v2
```bash
docker compose up -d  # Note: no hyphen
```

### Issue 4: Permission issues
**Solution:**
```bash
sudo usermod -aG docker $USER
# Log out and back in
```

## Expected Results

After successful setup:
- ✅ `nvidia-smi` shows H200 GPU
- ✅ `docker run --gpus all` works
- ✅ Container logs show `Accelerator device: 'cuda:0'`
- ✅ PyTorch detects CUDA
- ✅ Fast processing (2-5 seconds per page)

## Test Commands

```bash
# 1. Test nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# 2. Test with our container
docker-compose run --rm q-structurize python -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
"

# 3. Check container logs
docker-compose logs | grep "Accelerator device"
```

## If Still Getting CPU

1. **Check Docker version:** `docker --version`
2. **Check nvidia-docker version:** `nvidia-docker --version`
3. **Check Docker daemon config:** `/etc/docker/daemon.json`
4. **Restart Docker:** `sudo systemctl restart docker`
5. **Check GPU drivers:** `nvidia-smi`
