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
        version="2.1.0",
        description="""
# Q-Structurize - Advanced PDF Parsing API

Powered by Docling StandardPdfPipeline with **configurable per-request options**.

## Key Features

- 📐 **Layout Analysis** - Document structure understanding (DocLayNet)
- 📊 **Configurable Table Extraction** - FAST or ACCURATE modes (TableFormer)
- 🔍 **Optional OCR** - Multi-language support for scanned documents (EasyOCR)
- ⚡ **Multi-threaded** - Optimized for 72-core Xeon 6960P (1-144 threads)
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
  -F 'pipeline_options={"enable_ocr": true, "ocr_languages": ["en"], "num_threads": 16}'
```

### 3. High Performance (Leverage 72-core CPU)
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@document.pdf" \\
  -F 'pipeline_options={"num_threads": 64}'
```

### 4. Complex Tables (Accurate Mode)
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@tables.pdf" \\
  -F 'pipeline_options={"table_mode": "accurate", "do_cell_matching": true}'
```

## Configuration Guide

📖 Use **GET /parsers/options** to see all available configuration options with detailed descriptions and examples.

💡 Try the examples in the dropdown menu on the `/parse/file` endpoint below!

## Hardware Optimization

This API is optimized for **72-core Xeon 6960P** processors:
- Default: 8 threads (balanced)
- Light load: 8-16 threads
- High performance: 32-64 threads
- Maximum: 64-144 threads

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

