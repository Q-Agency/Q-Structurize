# Q-Structurize

PDF optimization and high-precision text extraction API using Docling with GraniteDocling VLM for maximum accuracy.

## Features

- **🎯 GraniteDocling VLM**: Maximum precision PDF parsing using Vision-Language Models
- **📄 PDF Optimization**: Clean and optimize PDFs for better text extraction using pikepdf
- **🔍 Advanced Text Extraction**: Structured markdown output with visual understanding
- **⚡ Fast Processing**: Optimized for production use with model caching
- **🐳 Docker Support**: Ready for containerized deployment on Ubuntu with H200 GPU support
- **📚 FastAPI**: Modern, fast web framework with automatic API documentation

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

**Response Format:**
```json
{
  "available": true,
  "library": "docling",
  "model": "granite_docling",
  "description": "High-precision PDF parsing using GraniteDocling VLM",
  "features": [
    "Visual Language Model processing",
    "Maximum precision text extraction",
    "Structured markdown output",
    "Metadata extraction"
  ]
}
```

### GET /

Health check endpoint.

**Response Format:**
```json
{
  "message": "Q-Structurize API is running",
  "status": "healthy",
  "features": ["PDF optimization", "Docling VLM parsing"],
  "version": "1.0.0",
  "docs": "/docs",
  "redoc": "/redoc"
}
```

## Docling VLM Processing

The service uses **Docling with GraniteDocling VLM** for maximum precision PDF parsing:

- **🎯 Vision-Language Model**: Understands both text and visual elements
- **📄 Structured Output**: Clean markdown with visual understanding
- **⚡ Model Caching**: Hugging Face models cached for performance
- **🔍 Maximum Precision**: Advanced AI-powered text extraction

## PDF Optimization

The service uses `pikepdf` to optimize PDFs for better text extraction by:

- **Content Normalization**: Standardizes content streams for better parsing
- **Object Stream Optimization**: Reorganizes PDF objects for improved readability
- **Resource Cleanup**: Removes unused objects and metadata
- **Stream Compression**: Optimizes content streams while maintaining structure

### Size Tracking

Optimization results are logged with size information:
```
PDF optimization completed - Original: 265487 bytes, Optimized: 371796 bytes, Reduction: -40.04%
```

*Note: Size increase is normal and indicates successful optimization for text extraction*

## Quick Start

### Docker Deployment (Recommended)

#### Prerequisites
- Docker and Docker Compose installed
- Git (to clone the repository)
- **Minimum 8GB RAM** (for VLM model processing)
- **GPU Support** (optional, for faster processing)

#### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/Q-Agency/Q-Structurize.git
cd QStructurize
```

2. **Build and run with Docker Compose:**
```bash
# Build and start the service in detached mode
docker-compose up --build -d

# Check if the service is running
docker-compose ps
```

3. **Verify the service is running:**
```bash
# Check logs (VLM initialization may take a few minutes)
docker-compose logs -f

# Test the health endpoint
curl http://localhost:8878/

# Check VLM parser status
curl http://localhost:8878/parsers/info
```

4. **Access the API:**
- **API Base URL**: `http://localhost:8878`
- **Interactive Documentation**: `http://localhost:8878/docs`
- **Alternative Documentation**: `http://localhost:8878/redoc`

#### First Run Notes
- **Model Download**: GraniteDocling VLM (~258M) downloads automatically on first use
- **Cache Directory**: Models cached in `./cache/` directory
- **Initialization**: VLM converter initializes on startup (check logs)

#### Docker Commands Reference

```bash
# Start the service
docker-compose up -d

# Stop the service
docker-compose down

# View logs
docker-compose logs -f

# Restart the service
docker-compose restart

# Rebuild and restart
docker-compose up --build -d

# Check service status
docker-compose ps
```

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

## Usage Examples

### Basic PDF Parsing with VLM
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "optimize_pdf=true" \
  -F "use_vlm=true"
```

### Skip VLM Processing
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "optimize_pdf=true" \
  -F "use_vlm=false"
```

### Check Parser Status
```bash
curl -X GET "http://localhost:8878/parsers/info"
```

### Health Check
```bash
curl -X GET "http://localhost:8878/"
```

## Architecture

### Components

- **FastAPI Application** (`main.py`): Main API server with PDF processing endpoints
- **PDF Optimizer Service** (`app/services/pdf_optimizer.py`): pikepdf-based PDF optimization
- **Docling VLM Parser** (`app/services/docling_parser.py`): GraniteDocling VLM integration
- **Docker Configuration**: Containerized deployment with Ubuntu base image
- **Model Caching**: Hugging Face and Torch model caching for performance
- **Logging**: Comprehensive logging for optimization and VLM processing tracking

### File Structure
```
QStructurize/
├── app/
│   ├── services/
│   │   ├── pdf_optimizer.py    # PDF optimization service
│   │   └── docling_parser.py   # Docling VLM parser service
│   └── models/
│       └── schemas.py          # Pydantic models
├── cache/                      # Model cache directory (created automatically)
├── main.py                     # FastAPI application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

## Error Handling

- **415 Unsupported Media Type**: Non-PDF files
- **500 Internal Server Error**: PDF optimization errors
- **400 Bad Request**: Invalid parameters

## Dependencies

- **FastAPI**: Modern web framework for APIs
- **pikepdf**: PDF manipulation and optimization
- **docling[vlm]**: Docling with VLM support for maximum precision
- **uvicorn**: ASGI server for FastAPI
- **python-multipart**: File upload support
- **torch**: PyTorch for VLM processing
- **transformers**: Hugging Face Transformers for model loading

## Production Deployment

### Docker Environment
- **Base Image**: Python 3.11-slim
- **Target Platform**: Ubuntu with H200 GPU support
- **Port**: 8878
- **Volume Mounts**: 
  - `./uploads:/app/uploads` (file uploads)
  - `./cache:/app/.cache` (model cache)
- **Environment Variables**:
  - `TRANSFORMERS_CACHE=/app/.cache/transformers`
  - `HF_HOME=/app/.cache/huggingface`
  - `TORCH_HOME=/app/.cache/torch`

### Logging
- PDF optimization results are logged with size information
- Request/response logging for monitoring
- Error logging for debugging

## Development

### Adding New Features
1. Create new services in `app/services/`
2. Add endpoints in `main.py`
3. Update Docker configuration if needed
4. Test with Docker Compose

### Monitoring
- Check logs: `docker-compose logs -f`
- Monitor optimization results in logs
- Use health check endpoint for service status

## License

This project is part of the Q-Structurize system for PDF processing and text extraction optimization.