# Q-Structurize

Advanced PDF parsing and structured text extraction API using Docling's StandardPdfPipeline with layout analysis, OCR, and table extraction.

## Features

- **📐 Layout Analysis**: Document structure understanding using DocLayNet model
- **📊 Table Extraction**: Configurable FAST or ACCURATE table structure preservation using TableFormer
- **🔍 OCR Support**: Optional multi-language text extraction from images and scanned documents using EasyOCR
- **📄 PDF Optimization**: Clean and optimize PDFs for better text extraction using pikepdf
- **🔄 Structured Output**: Clean markdown format or semantic chunks with rich metadata
- **🧩 Hybrid Chunking**: Modern chunking with native merge_peers for RAG and semantic search
- **🎯 Custom Tokenizers**: Match any HuggingFace embedding model's tokenizer for accurate chunking
- **📸 Image Description**: Optional AI-powered image description using VLM models (SmolVLM, Granite Vision) or API-based models (GPT-4 Vision)
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
- `embedding_model` (optional, string): HuggingFace embedding model name to use its tokenizer for accurate chunking (e.g., 'nomic-ai/nomic-embed-text-v1.5'). **Only loads the tokenizer (~2MB), not the full model**. Match this to your actual embedding model to ensure chunks fit perfectly. If not specified, uses HybridChunker's built-in tokenizer
- `include_markdown` (optional, boolean): Include full markdown when chunking enabled (default: false)
- `native_serialize` (optional, boolean): Use native Docling serialization via model_dump() (default: false)
- `parse_images` (optional, boolean): Enable image description for images in PDF (default: false). Requires VLM model or API key configured via environment variables
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

### Hybrid Chunking with Custom Embedding Model Tokenizer
```bash
# Match tokenizer to your embedding model for accurate chunking
# Example: Using nomic-embed-text (loads only tokenizer, ~2MB)
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "enable_chunking=true" \
  -F "max_tokens_per_chunk=512" \
  -F "embedding_model=nomic-ai/nomic-embed-text-v1.5"
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

### Parse PDF with Image Descriptions (Local VLM Model)
```bash
# Uses SmolVLM model (default) - requires model download on first use
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document_with_images.pdf" \
  -F "parse_images=true"
```

### Parse PDF with Image Descriptions (GPT-4 Vision API)
```bash
# Requires environment variables:
# DOCLING_PICTURE_DESCRIPTION_MODEL=api
# DOCLING_PICTURE_DESCRIPTION_API_URL=https://api.openai.com/v1/chat/completions
# DOCLING_PICTURE_DESCRIPTION_API_KEY=your-openai-api-key
# DOCLING_PICTURE_DESCRIPTION_API_MODEL=gpt-4o
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document_with_images.pdf" \
  -F "parse_images=true"
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

### Python Example with Custom Embedding Model Tokenizer
```python
import requests

url = "http://localhost:8878/parse/file"
files = {"file": open("document.pdf", "rb")}

# Match tokenizer to your embedding model for accurate chunking
data = {
    "enable_chunking": True,
    "max_tokens_per_chunk": 512,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Match your embedding model
    "merge_peers": True
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Status: {result['status']}")
print(f"Total chunks: {result['total_chunks']}")

# Process chunks - tokens are counted using the same tokenizer as your embedding model
for chunk in result['chunks']:
    print(f"\nChunk {chunk['chunk_index']}:")
    print(f"  Text: {chunk['text'][:100]}...")
    # Now you can safely embed this chunk without token overflow
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
| `embedding_model` | None | string | HuggingFace embedding model name for tokenization (see below) |
| `include_markdown` | false | boolean | Include full markdown alongside chunks |
| `native_serialize` | false | boolean | Use native Docling serialization (model_dump) for maximum metadata |

**Custom Embedding Model Tokenizers:**

You can specify any HuggingFace embedding model name to use its tokenizer for accurate token counting during chunking.

**⚠️ Important**: We only load the **tokenizer** (~2MB), not the full embedding model (hundreds of MB). The tokenizer is used solely to count tokens the same way your embedding model will count them when you embed the chunks later.

**Why Specify the Model Name?**

Different embedding models use different tokenizers that split text differently:
- Same text chunked with different tokenizers = different token counts
- If your chunks are "500 tokens" using a generic tokenizer, they might be "650 tokens" in your embedding model
- This causes truncation or errors when you try to embed those chunks

**Solution**: Match the tokenizer to your embedding model:
```bash
# If you use nomic-embed-text (via Ollama or API), use its tokenizer for chunking:
embedding_model=nomic-ai/nomic-embed-text-v1.5

# If you use BGE for embeddings, use its tokenizer:
embedding_model=BAAI/bge-small-en-v1.5
```

**Popular Models:**

| Model Name | Use Case | Context Window | Downloads |
|------------|----------|----------------|-----------|
| `nomic-ai/nomic-embed-text-v1.5` | High performance, long context | 8192 tokens | Tokenizer only (~2MB) |
| `BAAI/bge-small-en-v1.5` | High quality, English | 512 tokens | Tokenizer only (~2MB) |
| `sentence-transformers/all-MiniLM-L6-v2` | General purpose, fast | 256 tokens | Tokenizer only (~2MB) |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual | 128 tokens | Tokenizer only (~2MB) |
| `intfloat/e5-small-v2` | Balanced quality/speed | 512 tokens | Tokenizer only (~2MB) |

**Benefits:**
- ✅ **Accurate Token Counting**: Counts tokens exactly as your embedding model will
- ✅ **No Token Overflow**: Ensures chunks fit perfectly in your model's context window
- ✅ **Consistent Pipeline**: Same tokenizer in chunking → same tokenizer in embedding
- ✅ **Lightweight**: Only downloads tokenizer (~2MB), not full model weights (500MB+)
- ✅ **Fast**: Tokenizers are cached in memory for instant subsequent requests
- ✅ **Any HuggingFace Model**: Works with any embedding model on HuggingFace Hub

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

## Image Description Feature

### Overview

The image description feature uses Vision-Language Models (VLM) to automatically generate descriptions for images found in PDFs. This is an **optional feature** that must be explicitly enabled via the `parse_images=true` parameter.

### Supported Models

1. **Local VLM Models** (run on your server):
   - **SmolVLM** (default): Small, fast model (~256M parameters). Good for basic image descriptions.
   - **Granite Vision**: Larger, more accurate model (~2B parameters). Better for complex images.

2. **API-based Models** (remote API calls):
   - **GPT-4 Vision** (OpenAI): High-quality descriptions via OpenAI API. Requires API key.
   - **Other OpenAI-compatible APIs**: vLLM, LM Studio, Ollama, etc.

### Configuration

Image description is configured via environment variables:

**For Local Models:**
```bash
DOCLING_PICTURE_DESCRIPTION_MODEL=smolvlm  # or 'granite'
DOCLING_PICTURE_DESCRIPTION_PROMPT=""  # Optional custom prompt
```

**For API-based Models (GPT-4 Vision):**
```bash
DOCLING_PICTURE_DESCRIPTION_MODEL=api
DOCLING_PICTURE_DESCRIPTION_API_URL=https://api.openai.com/v1/chat/completions
DOCLING_PICTURE_DESCRIPTION_API_KEY=your-openai-api-key
DOCLING_PICTURE_DESCRIPTION_API_MODEL=gpt-4o  # or gpt-4-vision-preview
DOCLING_PICTURE_DESCRIPTION_API_TIMEOUT=90
```

### Usage

**Enable image descriptions:**
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "parse_images=true"
```

**Image descriptions appear in markdown output:**
- Images are automatically detected in PDFs
- Each image is processed by the selected VLM model
- Descriptions are added as alt text or captions in the exported markdown
- Descriptions are included in chunked output when `enable_chunking=true`

### Model Selection

| Model | Type | Size | Speed | Quality | Requirements |
|-------|------|------|-------|---------|--------------|
| SmolVLM | Local | ~256M | Fast | Good | Model download (~500MB) |
| Granite Vision | Local | ~2B | Medium | Better | Model download (~4GB) |
| GPT-4 Vision | API | N/A | Medium | Best | API key, network access |

### Performance Considerations

- **Local models**: First request slower (model download/loading), subsequent requests faster
- **API models**: Network latency added per image, but no local model storage needed
- **Processing time**: Adds ~1-5 seconds per image depending on model and image complexity
- **Cost**: API-based models incur API costs per image processed

### Privacy & Security

- **Local models**: All processing happens on your server, no data leaves your infrastructure
- **API models**: Images are sent to the API provider (e.g., OpenAI). Review privacy policies before use.

## Limitations

1. **Image Description is Optional**
   - Must be explicitly enabled via `parse_images=true` parameter
   - Requires VLM model configuration or API key
   - Not enabled by default to avoid unnecessary processing

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
