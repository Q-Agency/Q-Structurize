#!/bin/bash

echo "=== GPU Test Script ==="
echo "Testing GPU access on Linux machine..."

echo "1. Checking nvidia-smi..."
nvidia-smi

echo ""
echo "2. Testing nvidia-docker..."
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

echo ""
echo "3. Testing with docker-compose GPU support..."
docker-compose run --rm q-structurize python -c "
import torch
print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available')
"

echo ""
echo "=== Test Complete ==="
