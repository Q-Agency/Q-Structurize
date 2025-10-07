# Q-Structurize

Advanced PDF parsing and structured text extraction API using Docling's StandardPdfPipeline with layout analysis, OCR, and table extraction.

## Features

- **📐 Layout Analysis**: Document structure understanding using DocLayNet model
- **📊 Table Extraction**: Accurate table structure preservation using TableFormer
- **🔍 OCR Support**: Text extraction from images and scanned documents using EasyOCR
- **📄 PDF Optimization**: Clean and optimize PDFs for better text extraction using pikepdf
- **🔄 Structured Output**: Clean markdown format with proper formatting
- **⚡ CPU-Optimized**: Fast processing without GPU requirements
- **🐳 Docker Support**: Ready for containerized deployment
- **📚 FastAPI**: Modern, fast web framework with automatic API documentation

## 🚀 Quick Start

### One-Command Deployment

```bash
# 1. Build (downloads models automatically during build)
docker-compose build

# 2. Start
docker-compose up -d

# 3. Access
open http://localhost:8878/docs
```

That's it! Models are pre-downloaded during build (~400MB, takes 3-5 min on first build).

## What is StandardPdfPipeline?

Docling's **StandardPdfPipeline** is a CPU-optimized PDF processing pipeline that includes:

### 1. **Layout Detection (DocLayNet Model)**
- Identifies document structure (headings, paragraphs, lists)
- Multi-column support
- Section detection
- ~100MB model size

### 2. **Table Extraction (TableFormer Model)**
- Preserves table structure and formatting
- Handles complex table layouts
- Accurate cell detection
- ~200MB model size

### 3. **OCR Processing (EasyOCR)**
- Extracts text from images and scanned documents
- Multiple language support
- High accuracy on clear scans
- ~100MB model size

### 4. **Text Extraction**
- Direct extraction from PDF text layers
- Font and formatting preservation
- Structured output generation

## API Endpoints

### POST /parse/file

Parse PDF files using Docling's StandardPdfPipeline.

**Parameters:**
- `file` (required): PDF file upload
- `max_tokens_per_chunk` (optional, int): Maximum tokens per chunk (reserved for future use, default: 512)
- `optimize_pdf` (optional, boolean): Whether to optimize PDF for better text extraction (default: true)

**Response Format:**
```json
{
  "message": "PDF parsed successfully using Docling StandardPdfPipeline",
  "status": "success",
  "content": "# Document Title\n\nStructured markdown content..."
}
```

### GET /parsers/info

Get detailed information about the StandardPdfPipeline parser:
- Models used (layout detection, table extraction, OCR)
- Features and capabilities
- Performance characteristics
- Limitations

### GET /

Health check endpoint with feature list.

## Docker Deployment

### Prerequisites
- Docker and Docker Compose installed
- **Minimum 2GB RAM** (for model loading)
- **~1GB disk space** (for models and cache)
- **No GPU required** (CPU-optimized)

### Build Timeline

**First Build (~5-7 minutes):**
```
[0-2 min]  Installing system and Python packages
[2-6 min]  Downloading StandardPdfPipeline models (~400MB)
[6-7 min]  Verification and finalization
```

**Subsequent Builds (~2 minutes):**
- Uses cached layers and pre-downloaded models

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Q-Agency/Q-Structurize.git
cd QStructurize
```

2. **Build the image:**
```bash
docker-compose build

# Check logs during build (optional)
docker-compose build --progress=plain
```

3. **Start the service:**
```bash
docker-compose up -d
```

4. **Verify the service:**
```bash
# Check logs
docker-compose logs -f

# Test health endpoint
curl http://localhost:8878/

# Check parser status
curl http://localhost:8878/parsers/info

# Check models downloaded
docker logs q-structurize 2>&1 | grep "Models downloaded"
```

5. **Access the API:**
- **API Base URL**: `http://localhost:8878`
- **Interactive Documentation**: `http://localhost:8878/docs`
- **Alternative Documentation**: `http://localhost:8878/redoc`

### Performance Expectations

**First API Request (~5-10 seconds):**
- Models load into memory
- Pipeline initialization
- Processing begins

**Subsequent Requests (~2-5 seconds per page):**
- Models already loaded
- Fast CPU processing
- Depends on document complexity

## Usage Examples

### Basic PDF Parsing
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "optimize_pdf=true"
```

### Python Example
```python
import requests

url = "http://localhost:8878/parse/file"
files = {"file": open("document.pdf", "rb")}
data = {
    "optimize_pdf": True
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Status: {result['status']}")
print(f"Content:\n{result['content']}")
```

### JavaScript Example
```javascript
const formData = new FormData();
formData.append('file', pdfFile);
formData.append('optimize_pdf', 'true');

const response = await fetch('http://localhost:8878/parse/file', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.content);
```

## Architecture

### Components

1. **FastAPI Application** (`main.py`)
   - REST API endpoints
   - Request handling and validation
   - Error handling

2. **Docling Parser** (`app/services/docling_parser.py`)
   - StandardPdfPipeline integration
   - Model loading and caching
   - Document processing

3. **PDF Optimizer** (`app/services/pdf_optimizer.py`)
   - PDF preprocessing with pikepdf
   - Content normalization
   - Size optimization

4. **Model Management**
   - Official `docling-tools` CLI for model download
   - Persistent cache in `./cache` directory
   - Automatic model loading

### Tech Stack

- **Framework**: FastAPI 0.118.0
- **PDF Processing**: Docling 2.55.1 (StandardPdfPipeline)
- **PDF Optimization**: pikepdf 8.7.0
- **Python**: 3.11
- **Container**: Docker (Python 3.11-slim base)

### Models Used

| Model | Size | Purpose | Source |
|-------|------|---------|--------|
| **DocLayNet** | ~100MB | Layout detection and structure analysis | HuggingFace Hub |
| **TableFormer** | ~200MB | Table structure recognition | HuggingFace Hub |
| **EasyOCR** | ~100MB | OCR for images and scans | HuggingFace Hub |

**Total**: ~400MB

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
docker-compose build

# Start
docker-compose up -d

# Logs
docker-compose logs -f

# Stop
docker-compose down

# Check models (during build)
docker logs q-structurize 2>&1 | grep "Downloading"

# Shell access
docker exec -it q-structurize bash
```

## Model Storage Strategy

**Important**: Models are **built into the Docker image** during `docker build`, not stored in external volumes.

### How It Works

1. **During Build** (`docker build`):
   - `docling-tools models download` downloads ~1.3GB of models
   - Models are stored in `/root/.cache/docling/models` **inside the image**
   - Models become part of the image layers

2. **During Runtime** (`docker run`):
   - Container uses models from the image (instant access)
   - No volume mounts for model cache needed
   - No risk of empty directories overwriting built-in models

### Why No Volume Mounts?

```yaml
# ❌ WRONG - Volume mount overwrites built-in models with empty directory
volumes:
  - ./cache/docling:/root/.cache/docling

# ✅ CORRECT - Use models built into the image
volumes:
  - ./uploads:/app/uploads  # Only mount what changes
```

**Key Point**: Volume mounts replace the container's directory with the host directory. If the host directory is empty, you lose the models!

### Benefits of This Approach

- ✅ **No stale cache issues** - Fresh models on every rebuild
- ✅ **Simpler deployment** - No external cache directory management
- ✅ **Instant startup** - Models are already in the image
- ✅ **Reproducible builds** - Same image always has same models
- ✅ **No volume cleanup needed** - Everything is in the image

### Trade-offs

- ⚠️ **Larger image** (~2.5GB instead of ~1.2GB)
- ⚠️ **Longer initial build** (~7 min first time, ~2 min rebuilds)
- ✅ **But faster overall** - No download wait at startup!

## Performance Tuning

### Resource Limits

Default configuration in `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '2'
```

Adjust based on your needs:
- **Light usage** (< 10 req/min): 1-2 CPUs, 1-2GB RAM
- **Medium usage** (10-50 req/min): 2-4 CPUs, 2-4GB RAM
- **Heavy usage** (> 50 req/min): 4-8 CPUs, 4-8GB RAM

### Processing Time Estimates

| Document Type | Pages | Processing Time |
|--------------|-------|----------------|
| Text-only PDF | 10 | 5-10 seconds |
| PDF with tables | 10 | 10-20 seconds |
| Scanned PDF (OCR) | 10 | 20-40 seconds |
| Complex layout | 10 | 15-30 seconds |

## Troubleshooting

### Models not found
```bash
# Check if models were downloaded during build
docker logs q-structurize 2>&1 | grep "Models downloaded"

# Check cache directory
ls -lh ./cache/

# Rebuild if needed
docker-compose build --no-cache
```

### Slow processing
- **First request**: Normal (5-10 sec for model loading)
- **Subsequent requests**: Should be faster (2-5 sec/page)
- **Always slow**: Check CPU/memory limits in docker-compose.yml

### Out of memory
```bash
# Increase memory limit in docker-compose.yml
memory: 4G  # Instead of 2G
```

### Build fails
```bash
# Clean rebuild
docker-compose down -v
docker system prune -a
docker-compose build --no-cache
```

## Limitations

1. **No Vision-Language Understanding**
   - StandardPdfPipeline uses traditional CV models, not VLM
   - For advanced visual understanding, consider upgrading to VLM pipeline

2. **OCR Accuracy**
   - Depends on scan quality
   - Works best with clear, high-resolution scans
   - May struggle with handwriting or very low quality images

3. **Complex Layouts**
   - May need manual review for very complex documents
   - Multi-column detection works well but not perfect
   - Some formatting may be lost in markdown export

## License

MIT

## Links

- **Repository**: [github.com/Q-Agency/Q-Structurize](https://github.com/Q-Agency/Q-Structurize)
- **Docling**: [docling-project.github.io/docling](https://docling-project.github.io/docling/)
- **Docling Documentation**: [https://docling-project.github.io/docling/usage](https://docling-project.github.io/docling/usage/)

## Support

For issues and questions:
1. Check logs: `docker-compose logs -f`
2. Verify models: Check build output for "Models downloaded successfully"
3. Test endpoints: `curl http://localhost:8878/parsers/info`
4. Open an issue on GitHub

---

**Built with ❤️ using Docling StandardPdfPipeline**
