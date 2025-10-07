# 🚀 Q-Structurize Deployment Guide

Complete setup guide for deploying Q-Structurize with Docling VLM on your Ubuntu machine.

## 📋 Prerequisites

### System Requirements
- **OS**: Ubuntu 20.04+ (recommended)
- **RAM**: Minimum 8GB (16GB+ recommended for VLM processing)
- **Storage**: At least 10GB free space (for models and cache)
- **GPU**: NVIDIA H200 or compatible (optional, for faster processing)
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+

### Install Docker (if not already installed)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin

# Logout and login again for group changes to take effect
```

## 🎯 Quick Deployment

### Step 1: Clone Repository
```bash
git clone https://github.com/Q-Agency/Q-Structurize.git
cd QStructurize
```

### Step 2: Build and Start
```bash
# Build and start the service
docker-compose up --build -d

# Check if running
docker-compose ps
```

### Step 3: Verify Installation
```bash
# Check logs (VLM initialization may take a few minutes)
docker-compose logs -f

# Test health endpoint
curl http://localhost:8878/

# Check VLM parser status
curl http://localhost:8878/parsers/info
```

### Step 4: Access API
- **API Base**: `http://localhost:8878`
- **Swagger UI**: `http://localhost:8878/docs`
- **ReDoc**: `http://localhost:8878/redoc`

## 🔧 Advanced Configuration

### Environment Variables
Create `.env` file for custom configuration:
```bash
# .env file
TRANSFORMERS_CACHE=/app/.cache/transformers
HF_HOME=/app/.cache/huggingface
TORCH_HOME=/app/.cache/torch
PYTHONPATH=/app
```

### GPU Support (Optional)
If you have NVIDIA GPU, install NVIDIA Docker support:
```bash
# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## 📊 Monitoring and Maintenance

### Check Service Status
```bash
# Container status
docker-compose ps

# Resource usage
docker stats

# Logs
docker-compose logs -f
```

### Model Cache Management
```bash
# Check cache size
du -sh cache/

# Clear cache (if needed)
rm -rf cache/*
```

### Restart Services
```bash
# Restart service
docker-compose restart

# Rebuild and restart
docker-compose up --build -d

# Stop service
docker-compose down
```

## 🧪 Testing the Installation

### Test PDF Parsing
```bash
# Test with a sample PDF
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@sample.pdf" \
  -F "optimize_pdf=true" \
  -F "use_vlm=true"
```

### Test Parser Info
```bash
curl -X GET "http://localhost:8878/parsers/info"
```

### Test Health Check
```bash
curl -X GET "http://localhost:8878/"
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check logs
docker-compose logs

# Check system resources
free -h
df -h

# Restart Docker
sudo systemctl restart docker
```

#### 2. VLM Model Download Issues
```bash
# Check internet connection
ping huggingface.co

# Clear cache and retry
rm -rf cache/
docker-compose up --build -d
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats

# Increase swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Port Already in Use
```bash
# Check what's using port 8878
sudo lsof -i :8878

# Kill process or change port in docker-compose.yml
```

### Log Analysis
```bash
# Follow logs in real-time
docker-compose logs -f

# Check specific service logs
docker-compose logs q-structurize

# Check for errors
docker-compose logs | grep -i error
```

## 🔄 Updates and Maintenance

### Update Application
```bash
# Pull latest changes
git pull origin main

# Rebuild and restart
docker-compose up --build -d
```

### Backup Cache
```bash
# Backup model cache
tar -czf cache-backup-$(date +%Y%m%d).tar.gz cache/
```

### Clean Up
```bash
# Remove unused Docker images
docker system prune -a

# Remove unused volumes
docker volume prune
```

## 📈 Performance Optimization

### For Production
1. **Increase Memory**: Allocate more RAM to Docker
2. **GPU Acceleration**: Use NVIDIA GPU if available
3. **SSD Storage**: Use SSD for cache directory
4. **Network**: Ensure stable internet for model downloads

### Monitoring
```bash
# Resource monitoring
docker stats --no-stream

# Disk usage
df -h

# Memory usage
free -h
```

## 🎉 Success Indicators

Your installation is successful when:
- ✅ Container starts without errors
- ✅ Health endpoint returns 200 OK
- ✅ VLM parser shows as available
- ✅ Swagger UI loads correctly
- ✅ PDF parsing works with VLM

## 📞 Support

If you encounter issues:
1. Check the logs: `docker-compose logs -f`
2. Verify system requirements
3. Check network connectivity
4. Review this troubleshooting guide

---

**Ready to process PDFs with maximum precision!** 🚀
