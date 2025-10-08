"""
Custom OpenAPI schema configuration for Swagger UI.

This module provides enhanced OpenAPI documentation with rich markdown
descriptions, examples, and metadata for the Q-Structurize API.
"""

from fastapi.openapi.utils import get_openapi


def get_custom_openapi(app):
    """
    Create custom OpenAPI schema for better Swagger UI documentation.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Enhanced OpenAPI schema dictionary
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Q-Structurize API",
        version="2.2.0",
        description="""
# Q-Structurize - Advanced PDF Parsing API

Powered by Docling StandardPdfPipeline with **ThreadedPdfPipelineOptions for batching and configurable per-request options**.

## Key Features

- 📐 **Layout Analysis** - Document structure understanding (DocLayNet)
- 📊 **Configurable Table Extraction** - FAST or ACCURATE modes (TableFormer)
- 🔍 **Optional OCR** - Multi-language support for scanned documents (EasyOCR)
- 🚀 **Batched Processing** - Process multiple pages/operations in parallel
- 🎚️ **Backpressure Control** - Queue management for large documents
- ⚡ **Multi-threaded** - Optimized for 2x 72-core Xeon 6960P (144 threads)
- ⚙️ **Per-Request Configuration** - Customize pipeline for each document
- 🔄 **Structured Output** - Clean markdown format

## Quick Examples

### 1. Default Processing (Fast, No OCR)
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@document.pdf"
```

### 2. Scanned Document with OCR
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@scanned.pdf" \\
  -F "enable_ocr=true" \\
  -F "num_threads=16"
```

### 3. High Performance (Leverage 144-core CPU)
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@document.pdf" \\
  -F "num_threads=64"
```

### 4. Maximum Throughput with Batching
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@large-document.pdf" \\
  -F "num_threads=64" \\
  -F "layout_batch_size=16" \\
  -F "table_batch_size=16" \\
  -F "queue_max_size=500"
```

### 5. Complex Tables (Accurate Mode)
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@tables.pdf" \\
  -F "table_mode=accurate" \\
  -F "do_cell_matching=true"
```

### 6. Multilingual Document
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@multilingual.pdf" \\
  -F "enable_ocr=true" \\
  -F "ocr_languages=en,es,de" \\
  -F "num_threads=16"
```

## Configuration Guide

📖 Use **GET /parsers/options** to see all available configuration options with detailed descriptions and examples.

💡 Try the examples in the dropdown menu on the `/parse/file` endpoint below!

## Hardware Optimization

This API is optimized for **2x 72-core Xeon 6960P** (144 total cores) with batching support:

**Threading:**
- Default: 8 threads (balanced)
- Light load: 8-16 threads
- High performance: 32-64 threads
- Maximum: 64-144 threads

**Batching (for high throughput):**
- Low latency: batch_size=1-2
- Balanced: batch_size=4-8 (default: 4)
- High throughput: batch_size=8-16
- Maximum: batch_size=16-32 (requires more memory)

**Queue Management:**
- Small documents: queue_max_size=50-100 (default: 100)
- Large documents: queue_max_size=300-1000

## Language Support

Supported OCR languages: English, Spanish, German, French, Italian, Portuguese, Russian, Chinese, Japanese, Korean, and more.

Example: `"ocr_languages": ["en", "es", "de"]`
        """,
        routes=app.routes,
    )
    
    # Add custom schema enhancements
    openapi_schema["info"]["x-logo"] = {
        "url": "https://docling-project.github.io/docling/assets/logo.png"
    }
    
    # Add tags metadata
    openapi_schema["tags"] = [
        {
            "name": "PDF Parsing",
            "description": "Parse PDF files with configurable pipeline options"
        },
        {
            "name": "System",
            "description": "System information and configuration endpoints"
        },
        {
            "name": "Health",
            "description": "Health check and status endpoints"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

