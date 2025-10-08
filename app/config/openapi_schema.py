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

Powered by **pre-initialized Docling converter** with ThreadedPdfPipelineOptions for **instant processing**.

## Key Features

- 🚀 **Instant Processing** - Pre-loaded models, zero initialization delay
- 📐 **Layout Analysis** - Document structure understanding (DocLayNet)
- 🎚️ **Batched Processing** - Parallel page/operation processing
- ⚡ **Multi-threaded** - Optimized for 2x 72-core Xeon 6960P (144 threads)
- 🔄 **Structured Output** - Clean markdown format
- ⚙️ **ENV-Based Config** - Set once at container startup, rebuild in ~10 seconds

## Quick Examples

### 1. Simple PDF Parsing
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@document.pdf"
```

### 2. Without PDF Optimization (Faster)
```bash
curl -X POST "http://localhost:8878/parse/file" \\
  -F "file=@document.pdf" \\
  -F "optimize_pdf=false"
```

### 3. Python Example
```python
import requests

with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8878/parse/file",
        files={"file": f}
    )
print(response.json()["content"])
```

## Configuration

Configuration is set via **Dockerfile ENV variables** at container startup. This provides:
- ✅ **Instant processing** - No per-request initialization overhead
- ✅ **Consistent performance** - Same optimized settings for all requests
- ✅ **Fast reconfiguration** - Rebuild takes only ~10 seconds (Docker cache)

### To Change Configuration:

1. **Edit Dockerfile** - Modify ENV variables or parser initialization
2. **Rebuild** - `docker-compose build` (~10 seconds with cache)
3. **Restart** - `docker-compose up`

### Key ENV Variables:

- `OMP_NUM_THREADS` - Number of processing threads (default: 100, max: 144)
- Parser initialization in `docling_parser.py` - OCR, tables, batch sizes

### Hardware Optimization

Optimized for **2x 72-core Xeon 6960P** (144 total cores):
- Default: 100 threads (leaves headroom for system)
- Maximum: 120-144 threads (full utilization)
- Batch sizes: 32 (aggressive batching for throughput)
- Queue: 1000 (large queue for big documents)
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

