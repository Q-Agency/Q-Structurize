# Q-Structurize

Advanced PDF parsing and structured text extraction API using Docling's StandardPdfPipeline with layout analysis, OCR, and table extraction.

## Features

- **📐 Layout Analysis**: Document structure understanding using DocLayNet model
- **📊 Table Extraction**: Configurable FAST or ACCURATE table structure preservation using TableFormer
- **🔍 OCR Support**: Optional multi-language text extraction from images and scanned documents using EasyOCR
- **📄 PDF Optimization**: Clean and optimize PDFs for better text extraction using pikepdf
- **🔄 Structured Output**: Clean markdown format or semantic chunks with rich metadata
- **🧩 Hybrid Chunking**: Modern chunking with native merge_peers for RAG and semantic search
- **⚡ Multi-threaded**: Optimized for high-performance CPUs (72-core Xeon 6960P)
- **⚙️ Configurable Pipeline**: Per-request configuration of OCR, table extraction, threading, and more
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

Parse PDF files using Docling's StandardPdfPipeline with configurable options. Supports both markdown output and hybrid chunking for RAG applications.

**Parameters:**
- `file` (required): PDF file upload
- `optimize_pdf` (optional, boolean): Whether to optimize PDF for better text extraction (default: true)
- `enable_chunking` (optional, boolean): Enable hybrid chunking for RAG/semantic search (default: false)
- `max_tokens_per_chunk` (optional, int): Maximum tokens per chunk, 128-2048 (default: 512)
- `merge_peers` (optional, boolean): Auto-merge undersized chunks with same headings (default: true)
- `include_markdown` (optional, boolean): Include full markdown when chunking enabled (default: false)
- `native_serialize` (optional, boolean): Use native Docling serialization via model_dump() (default: false)
- `pipeline_options` (optional, JSON string): Pipeline configuration options
  - `enable_ocr` (boolean): Enable OCR for scanned documents (default: false)
  - `ocr_languages` (array): Language codes like ["en", "es"] (default: ["en"])
  - `table_mode` (string): "fast" or "accurate" (default: "fast")
  - `do_table_structure` (boolean): Enable table extraction (default: true)
  - `do_cell_matching` (boolean): Enable cell matching (default: true)
  - `num_threads` (integer): Processing threads 1-144 (default: 8)
  - `accelerator_device` (string): "cpu", "cuda", or "auto" (default: "cpu")

**Response Format (Standard Markdown):**
```json
{
  "message": "PDF parsed successfully using Docling StandardPdfPipeline",
  "status": "success",
  "content": "# Document Title\n\nStructured markdown content..."
}
```

**Response Format (Chunked):**
```json
{
  "message": "PDF parsed and chunked successfully (42 chunks generated)",
  "status": "success",
  "chunks": [
    {
      "text": "search_document: Introduction\n\nThis document presents...",
      "section_title": "Introduction",
      "chunk_index": 0,
      "metadata": {
        "content_type": "text",
        "heading_path": "Chapter 1 > Introduction",
        "pages": [1, 2],
        "doc_items_count": 5
      }
    }
  ],
  "total_chunks": 42,
  "content": null
}
```

### GET /parsers/options

Get available pipeline configuration options with:
- Option descriptions and types
- Default values and valid ranges
- Example configurations for common scenarios
- Usage examples (curl and Python)

### GET /parsers/info

Get detailed information about the StandardPdfPipeline parser:
- Models used (layout detection, table extraction, OCR)
- Features and capabilities
- Performance characteristics
- Configuration options
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

### Basic PDF Parsing (Default Settings)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "optimize_pdf=true"
```

### Parse Scanned PDF with OCR
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@scanned.pdf" \
  -F 'pipeline_options={"enable_ocr": true, "ocr_languages": ["en"]}'
```

### Hybrid Chunking for RAG (Basic)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "enable_chunking=true"
```

### Hybrid Chunking with Custom Token Limit
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "enable_chunking=true" \
  -F "max_tokens_per_chunk=1024"
```

### Chunking with Markdown Included
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "enable_chunking=true" \
  -F "include_markdown=true"
```

### Native Serialization (Full Docling Metadata)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "enable_chunking=true" \
  -F "native_serialize=true"
```

### High-Performance Processing (Leverage 72-core Xeon)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F 'pipeline_options={"num_threads": 64, "table_mode": "fast"}'
```

### Accurate Table Extraction
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@tables.pdf" \
  -F 'pipeline_options={"table_mode": "accurate", "do_cell_matching": true, "num_threads": 24}'
```

### Multilingual Document with OCR
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@multilingual.pdf" \
  -F 'pipeline_options={"enable_ocr": true, "ocr_languages": ["en", "es", "de"], "num_threads": 16}'
```

### Get Available Configuration Options
```bash
curl -X GET "http://localhost:8878/parsers/options"
```

### Python Example with Pipeline Options
```python
import requests
import json

url = "http://localhost:8878/parse/file"
files = {"file": open("document.pdf", "rb")}

# Configure pipeline options
pipeline_config = {
    "enable_ocr": True,
    "ocr_languages": ["en"],
    "table_mode": "accurate",
    "num_threads": 16
}

data = {
    "optimize_pdf": True,
    "pipeline_options": json.dumps(pipeline_config)
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Status: {result['status']}")
print(f"Content:\n{result['content']}")
```

### Python Example with Chunking for RAG (Custom Mode)
```python
import requests

url = "http://localhost:8878/parse/file"
files = {"file": open("document.pdf", "rb")}

# Enable chunking for RAG with custom metadata
data = {
    "enable_chunking": True,
    "max_tokens_per_chunk": 1024,
    "merge_peers": True,
    "include_markdown": False,
    "native_serialize": False  # Custom mode (default)
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Status: {result['status']}")
print(f"Total chunks: {result['total_chunks']}")

# Process chunks for embedding/indexing
for chunk in result['chunks']:
    print(f"\nChunk {chunk['chunk_index']}:")
    print(f"  Section: {chunk['section_title']}")
    print(f"  Pages: {chunk['metadata']['pages']}")
    print(f"  Content type: {chunk['metadata']['content_type']}")
    print(f"  Text preview: {chunk['text'][:100]}...")
```

### Python Example with Native Serialization (Full Metadata)
```python
import requests

url = "http://localhost:8878/parse/file"
files = {"file": open("document.pdf", "rb")}

# Enable native serialization for maximum metadata
data = {
    "enable_chunking": True,
    "max_tokens_per_chunk": 1024,
    "native_serialize": True  # Native mode - full Docling metadata
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Status: {result['status']}")
print(f"Total chunks: {result['total_chunks']}")

# Native mode returns ALL Docling fields
for chunk in result['chunks']:
    print(f"\nChunk {chunk['chunk_index']}:")
    print(f"  Native fields: {list(chunk.keys())[:10]}...")  # Show first 10 fields
    print(f"  Prefixed text: {chunk['prefixed_text'][:100]}...")
    print(f"  Contextualized: {chunk['contextualized_text'][:100]}...")
```

### JavaScript Example with Pipeline Options
```javascript
const formData = new FormData();
formData.append('file', pdfFile);
formData.append('optimize_pdf', 'true');

// Configure pipeline options
const pipelineOptions = {
  enable_ocr: true,
  ocr_languages: ["en"],
  num_threads: 16,
  table_mode: "accurate"
};
formData.append('pipeline_options', JSON.stringify(pipelineOptions));

const response = await fetch('http://localhost:8878/parse/file', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result.content);
```

## Pipeline Configuration Guide

### When to Use OCR
- ✅ **Enable OCR** for: Scanned documents, images embedded in PDFs, handwritten notes
- ❌ **Disable OCR** for: Native digital PDFs with selectable text (faster processing)

### Table Extraction Modes
- **Fast Mode** (default): Good for most documents, faster processing
- **Accurate Mode**: Better for complex tables with merged cells, nested structures

### Thread Configuration for 72-core Xeon 6960P

| Scenario | Threads | Use Case |
|----------|---------|----------|
| **Light Load** | 8-16 | Multiple concurrent requests, balanced CPU usage |
| **Balanced** | 16-32 | Good balance between speed and resource availability |
| **High Performance** | 32-64 | Single high-priority document, maximum speed |
| **Maximum** | 64-144 | Dedicated processing, all resources available |

### Language Support for OCR
Common language codes:
- `en` - English
- `es` - Spanish
- `de` - German
- `fr` - French
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `zh` - Chinese
- `ja` - Japanese
- `ko` - Korean

Example: `"ocr_languages": ["en", "es"]` for English and Spanish

### Hybrid Chunking Configuration

**When to Use Chunking:**
- ✅ **RAG (Retrieval-Augmented Generation)**: Break documents into semantic chunks for embedding and retrieval
- ✅ **Semantic Search**: Create searchable chunks with rich metadata (pages, headings, content type)
- ✅ **Long Document Processing**: Split large documents into manageable pieces for LLM processing
- ✅ **Context-Aware Indexing**: Maintain document structure and context in each chunk

**Chunking Modes:**

1. **Custom Mode** (`native_serialize=false`, default)
   - Clean, curated metadata (content_type, heading_path, pages, captions)
   - Structured ChunkData models
   - Best for: Standard RAG applications, clean API responses

2. **Native Mode** (`native_serialize=true`)
   - Full Docling metadata via `model_dump()`
   - All native chunk fields from Docling
   - Best for: Maximum metadata, advanced use cases, debugging

**Chunking Parameters:**

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `enable_chunking` | false | boolean | Enable hybrid chunking mode |
| `max_tokens_per_chunk` | 512 | 128-2048 | Maximum tokens per chunk |
| `merge_peers` | true | boolean | Auto-merge undersized successive chunks with same headings |
| `include_markdown` | false | boolean | Include full markdown alongside chunks |
| `native_serialize` | false | boolean | Use native Docling serialization (model_dump) for maximum metadata |

**Chunk Metadata:**
Each chunk includes rich metadata:
- `content_type`: text, table, list, or heading
- `heading_path`: Hierarchical path (e.g., "Chapter 1 > Section 1.1")
- `pages`: List of page numbers where chunk appears
- `section_title`: Most specific heading for the chunk
- `captions`: Captions for tables/figures in the chunk

**Token Limit Guidelines:**
- **128-256 tokens**: Very small chunks, high precision retrieval
- **512 tokens** (default): Good balance for most RAG applications
- **1024 tokens**: Larger context, fewer chunks, better for summarization
- **2048 tokens**: Maximum, use for maintaining long-form context

### Pre-configured Scenarios

**Scenario 1: Fast Processing (Default)**
```json
{}
```
No OCR, fast table mode, 8 threads

**Scenario 2: Scanned Documents**
```json
{
  "enable_ocr": true,
  "ocr_languages": ["en"],
  "num_threads": 16
}
```

**Scenario 3: Complex Financial Documents**
```json
{
  "table_mode": "accurate",
  "do_cell_matching": true,
  "num_threads": 24
}
```

**Scenario 4: High-Volume Processing**
```json
{
  "num_threads": 64,
  "table_mode": "fast"
}
```

**Scenario 5: Maximum Quality**
```json
{
  "enable_ocr": true,
  "ocr_languages": ["en"],
  "table_mode": "accurate",
  "do_cell_matching": true,
  "num_threads": 32
}
```

## Architecture

### Components

1. **FastAPI Application** (`main.py`)
   - REST API endpoints
   - Request handling and validation
   - Error handling

2. **Docling Parser** (`app/services/docling_parser.py`)
   - StandardPdfPipeline integration
   - Per-request pipeline configuration
   - Dynamic model loading and caching
   - Document processing with configurable options

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
