# Q-Structurize Linux GPU Deployment Guide

## Prerequisites for Linux Server with H200 GPU

### 1. Install NVIDIA Drivers
```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# If not installed, install NVIDIA drivers (Ubuntu 22.04)
sudo apt update
sudo apt install nvidia-driver-535  # or latest version
sudo reboot
```

### 2. Install Docker with NVIDIA Support
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install NVIDIA Docker support
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
```

### 3. Deploy Q-Structurize
```bash
# Clone and build
git clone <your-repo>
cd QStructurize

# Build with GPU support
docker-compose build --no-cache

# Start the service
docker-compose up -d

# Check GPU is detected
docker-compose exec q-structurize python test_gpu.py

# Monitor logs
docker-compose logs -f
```

## Expected Performance on H200 GPU

- **Model Loading**: 30-60 seconds (first time)
- **PDF Processing**: 10-30 seconds per page
- **Memory Usage**: 8-16GB VRAM
- **Throughput**: 2-5 pages per minute

## Verification Commands

```bash
# Check GPU detection
docker-compose exec q-structurize python test_gpu.py

# Test API
curl http://localhost:8878/

# Test parser info
curl http://localhost:8878/parsers/info

# Test with PDF (replace with your PDF)
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@test.pdf" \
  -F "use_vlm=true" \
  -F "optimize_pdf=true"
```

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi

# Check Docker daemon config
cat /etc/docker/daemon.json
# Should contain: "default-runtime": "nvidia"
```

### Out of Memory
- Reduce batch size in VLM processing
- Use model quantization (already enabled with load_in_8bit=True)
- Monitor GPU memory: `nvidia-smi`

### Slow Processing
- Ensure GPU is being used (not CPU fallback)
- Check CUDA version compatibility
- Monitor GPU utilization: `nvidia-smi -l 1`
