# Q-Structurize Linux GPU Deployment Guide

## Prerequisites

### 1. NVIDIA GPU Setup
- **H200 GPU** with 80GB VRAM
- **NVIDIA drivers** installed
- **nvidia-docker** runtime installed

### 2. Install nvidia-docker (if not already installed)
```bash
# Install nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 3. Verify GPU Access
```bash
# Test nvidia-docker
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

## Deployment Steps

### 1. Clone and Setup
```bash
git clone <your-repo>
cd QStructurize
```

### 2. Build and Deploy
```bash
# Build the container
docker-compose build --no-cache

# Start the service
docker-compose up -d
```

### 3. Verify GPU Access
```bash
# Check logs for GPU detection
docker-compose logs | grep -E "(cuda|gpu|Accelerator device)"

# Should show: "Accelerator device: 'cuda:0'"
```

### 4. Test the API
```bash
# Test the API
curl http://localhost:8878/

# Test parser info
curl http://localhost:8878/parsers/info
```

## Performance Optimization

### H200 GPU Configuration
The application is optimized for H200 GPU with:
- **Full precision processing** (no quantization)
- **32K token limits** for extended context
- **KV cache acceleration** for faster processing
- **FP16 precision** for optimal speed

### Expected Performance
- **2-5 seconds per page** on H200 GPU
- **8-16GB VRAM usage** per document
- **Models cached** after first run

## Troubleshooting

### GPU Not Detected
```bash
# Check nvidia-docker installation
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check container logs
docker-compose logs | grep -E "(cuda|gpu|Accelerator device)"
```

### Models Downloading Every Time
```bash
# Check cache directories
docker-compose exec q-structurize ls -la /app/.cache/
docker-compose exec q-structurize ls -la /root/.cache/

# Check environment variables
docker-compose exec q-structurize env | grep -E "(CACHE|HF_|TRANSFORMERS|TORCH)"
```

### Performance Issues
- **Check GPU utilization**: `nvidia-smi`
- **Check memory usage**: `docker stats`
- **Check logs for errors**: `docker-compose logs --tail=100`

## Cache Management

### Clear Cache
```bash
# Clear all caches
docker-compose down
rm -rf cache/*
docker-compose up -d
```

### Check Cache Usage
```bash
# Check cache size
du -sh cache/

# Check cache contents
ls -la cache/
```

## Monitoring

### GPU Usage
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Container Stats
```bash
# Monitor container resources
docker stats qstructurize-q-structurize-1
```

### Application Logs
```bash
# Follow logs
docker-compose logs -f
```

## Production Deployment

### 1. Use Production Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  q-structurize:
    build: .
    ports:
      - "8878:8000"
    volumes:
      - ./uploads:/app/uploads
      - ./cache:/app/.cache
      - ./cache:/root/.cache
    environment:
      - PYTHONPATH=/app
      - TRANSFORMERS_CACHE=/app/.cache/transformers
      - HF_HOME=/app/.cache/huggingface
      - TORCH_HOME=/app/.cache/torch
      - HF_HUB_CACHE=/app/.cache/huggingface
      - CUDA_VISIBLE_DEVICES=0
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 2. Deploy with Production Config
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Support

For issues with:
- **GPU detection**: Check nvidia-docker installation
- **Model caching**: Check cache directory permissions
- **Performance**: Monitor GPU utilization and memory usage
- **API errors**: Check application logs

## Expected Results

After successful deployment:
- **GPU detected**: `Accelerator device: 'cuda:0'`
- **Models cached**: Cache directories populated after first run
- **Fast processing**: 2-5 seconds per page on H200
- **High accuracy**: Full precision VLM processing