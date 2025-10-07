# Q-Structurize

PDF optimization and high-precision text extraction API using Docling with GraniteDocling VLM for maximum accuracy.

## Features

- **🎯 GraniteDocling VLM**: Maximum precision PDF parsing using Vision-Language Models
- **📄 PDF Optimization**: Clean and optimize PDFs for better text extraction using pikepdf
- **🔍 Advanced Text Extraction**: Structured markdown output with visual understanding
- **⚡ Fast Processing**: Pre-cached models using official `docling-tools` CLI
- **🐳 Docker Support**: Ready for containerized deployment on Ubuntu with H200 GPU support
- **📚 FastAPI**: Modern, fast web framework with automatic API documentation

## 🚀 Quick Start

### One-Command Deployment

```bash
# 1. Build (downloads models automatically using official CLI)
docker-compose -f docker-compose.gpu.yml build

# 2. Start
docker-compose -f docker-compose.gpu.yml up -d

# 3. Access
open http://localhost:8878/docs
```

That's it! See [QUICK_START.md](QUICK_START.md) for more details.

## API Endpoints

### POST /parse/file

Parse PDF files using Docling with GraniteDocling VLM for maximum precision.

**Parameters:**
- `file` (required): PDF file upload
- `max_tokens_per_chunk` (optional, int): Maximum tokens per chunk (reserved for future use, default: 512)
- `optimize_pdf` (optional, boolean): Whether to optimize PDF for better text extraction (default: true)
- `use_vlm` (optional, boolean): Whether to use Docling VLM for maximum precision (default: true)

**Response Format:**
```json
{
  "message": "PDF parsed successfully using Docling VLM",
  "status": "success",
  "content": "# Document Title\n\nStructured markdown content..."
}
```

### GET /parsers/info

Get information about available parsers.

### GET /

Health check endpoint.

## Docling VLM Processing

The service uses **Docling with GraniteDocling VLM** for maximum precision PDF parsing:

- **🎯 Vision-Language Model**: Understands both text and visual elements
- **📄 Structured Output**: Clean markdown with visual understanding
- **⚡ Model Pre-caching**: Uses official `docling-tools` CLI for model management
- **🔍 Maximum Precision**: Advanced AI-powered text extraction
- **🚀 H200 GPU Optimized**: Full precision, 32K token limit, KV cache acceleration

## Model Management (New!)

We now use the **official `docling-tools` CLI** for model management:

```bash
# Models are automatically downloaded during Docker build
# Using: docling-tools models download

# Check downloaded models
docker exec q-structurize docling-tools models list

# Verify cache
du -sh ./cache
```

**Benefits:**
- ✅ Official Docling CLI (maintained by Docling team)
- ✅ Automatic model download during build
- ✅ Persistent cache in `./cache` directory
- ✅ Fast rebuilds (model persists)

See [OFFICIAL_CLI_SOLUTION.md](OFFICIAL_CLI_SOLUTION.md) for complete details.

## Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- **Minimum 16GB RAM** (for VLM model processing)
- **NVIDIA GPU** (H200 recommended, other GPUs supported)
- **~10GB disk space** (for models and cache)

### Build Timeline

**First Build (~10 minutes):**
```
[0-4 min]  Installing system and Python packages
[4-9 min]  Downloading VLM model (official CLI)
[9-10 min] Verification and finalization
```

**Subsequent Builds (~3 minutes):**
- Uses cached layers and pre-downloaded models

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Q-Agency/Q-Structurize.git
cd QStructurize
```

2. **Build using official method:**
```bash
# For GPU deployment (recommended)
docker-compose -f docker-compose.gpu.yml build

# Check logs during build
docker-compose -f docker-compose.gpu.yml build --progress=plain
```

3. **Start the service:**
```bash
docker-compose -f docker-compose.gpu.yml up -d
```

4. **Verify the service:**
```bash
# Check logs
docker-compose -f docker-compose.gpu.yml logs -f

# Test health endpoint
curl http://localhost:8878/

# Check VLM parser status
curl http://localhost:8878/parsers/info

# Verify models
docker exec q-structurize docling-tools models list
```

5. **Access the API:**
- **API Base URL**: `http://localhost:8878`
- **Interactive Documentation**: `http://localhost:8878/docs`
- **Alternative Documentation**: `http://localhost:8878/redoc`

### Performance Expectations

**First API Request (~60-90 seconds):**
- Model loads into GPU memory
- Subsequent requests are much faster

**Subsequent Requests (~4-5 seconds per page):**
- Model already loaded
- Fast GPU inference

### Cache Management

```bash
# ✅ Good: Rebuild keeping cache (fast)
docker-compose down      # No -v flag!
docker-compose build
docker-compose up -d

# ❌ Bad: Deletes cache, re-downloads everything
docker-compose down -v   # Removes volumes!
```

The `./cache` directory persists models between container restarts.

## Usage Examples

### Basic PDF Parsing with VLM
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "optimize_pdf=true" \
  -F "use_vlm=true"
```

### Python Example
```python
import requests

url = "http://localhost:8878/parse/file"
files = {"file": open("document.pdf", "rb")}
data = {
    "optimize_pdf": True,
    "use_vlm": True
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

## Documentation

- **[QUICK_START.md](QUICK_START.md)** - 3-command quick start guide
- **[OFFICIAL_CLI_SOLUTION.md](OFFICIAL_CLI_SOLUTION.md)** - Complete CLI guide
- **[WHATS_NEW.md](WHATS_NEW.md)** - Recent changes and improvements
- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Detailed build instructions
- **[DEPLOYMENT_LINUX.md](DEPLOYMENT_LINUX.md)** - Linux deployment guide

## Architecture

### Components

1. **FastAPI Application** (`main.py`)
   - REST API endpoints
   - Request handling and validation

2. **Docling Parser** (`app/services/docling_parser.py`)
   - GraniteDocling VLM integration
   - H200 GPU optimizations
   - Model loading and caching

3. **PDF Optimizer** (`app/services/pdf_optimizer.py`)
   - PDF preprocessing with pikepdf
   - Content normalization

4. **Model Management**
   - Official `docling-tools` CLI
   - HuggingFace model caching
   - Persistent storage in `./cache`

### Tech Stack

- **Framework**: FastAPI 0.118.0
- **PDF Processing**: Docling 2.55.1 with VLM support
- **PDF Optimization**: pikepdf 8.7.0
- **GPU**: NVIDIA CUDA 12.1 + cuDNN 8
- **Python**: 3.11
- **Container**: Docker with GPU support

## Development

### Local Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Run locally:**
```bash
python main.py
```

3. **Access API:**
- http://localhost:8000/docs

### Docker Commands

```bash
# Build
docker-compose -f docker-compose.gpu.yml build

# Start
docker-compose -f docker-compose.gpu.yml up -d

# Logs
docker-compose -f docker-compose.gpu.yml logs -f

# Stop
docker-compose -f docker-compose.gpu.yml down

# Check models
docker exec q-structurize docling-tools models list

# Shell access
docker exec -it q-structurize bash
```

## GPU Support

### H200 GPU Optimizations

- **Full Precision**: No quantization (80GB VRAM available)
- **Extended Tokens**: 32,768 token limit
- **KV Cache**: Enabled for speed
- **FP16**: Mixed precision for optimal performance

### Other GPUs

The system works with other NVIDIA GPUs:
- **A100**: Full support
- **A40/A10**: Supported
- **RTX 4090/4080**: Supported
- **CPU Fallback**: Available but slower

## Troubleshooting

### Models not found
```bash
# Check cache
ls -lh ./cache/huggingface/hub/

# Verify with CLI
docker exec q-structurize docling-tools models list

# Rebuild if needed
docker-compose build --no-cache
```

### Slow first request
This is normal! First request loads model into GPU (60-90 sec).

### Always re-downloading
Don't use `docker-compose down -v` (removes cache).

## License

MIT

## Links

- **Repository**: [github.com/Q-Agency/Q-Structurize](https://github.com/Q-Agency/Q-Structurize)
- **Docling**: [docling-project.github.io/docling](https://docling-project.github.io/docling/)
- **docling-tools CLI**: [Official Documentation](https://docling-project.github.io/docling/reference/cli/)

## Support

For issues and questions:
1. Check [TROUBLESHOOTING_GPU.md](TROUBLESHOOTING_GPU.md)
2. Review [OFFICIAL_CLI_SOLUTION.md](OFFICIAL_CLI_SOLUTION.md)
3. Open an issue on GitHub

---

**Built with ❤️ using official Docling tools**
