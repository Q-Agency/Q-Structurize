# Q-Structurize

PDF optimization and text extraction API for preparing documents for RAG (Retrieval-Augmented Generation).

## Features

- **PDF Optimization**: Clean and optimize PDFs for better text extraction using pikepdf
- **Text Extraction Ready**: Prepare PDFs for downstream text extraction tools
- **Size Tracking**: Monitor file size changes during optimization
- **Docker Support**: Ready for containerized deployment on Ubuntu with H200 GPU support
- **FastAPI**: Modern, fast web framework with automatic API documentation

## API Endpoints

### POST /parse/file

Parse and optimize PDF files for text extraction.

**Parameters:**
- `file` (required): PDF file upload
- `max_tokens_per_chunk` (optional, int): Maximum tokens per chunk (reserved for future use, default: 512)
- `optimize_pdf` (optional, boolean): Whether to optimize PDF for better text extraction (default: true)

**Response Format:**
```json
{
  "message": "Document received successfully",
  "status": "success"
}
```

### GET /

Health check endpoint.

**Response Format:**
```json
{
  "message": "Q-Structurize API is running",
  "status": "healthy"
}
```

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
# Check logs
docker-compose logs -f

# Test the health endpoint
curl http://localhost:8878/
```

4. **Access the API:**
- **API Base URL**: `http://localhost:8878`
- **Interactive Documentation**: `http://localhost:8878/docs`
- **Alternative Documentation**: `http://localhost:8878/redoc`

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

### Basic PDF Optimization
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "optimize_pdf=true"
```

### Skip Optimization
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "optimize_pdf=false"
```

### Health Check
```bash
curl -X GET "http://localhost:8878/"
```

## Architecture

### Components

- **FastAPI Application** (`main.py`): Main API server with PDF processing endpoints
- **PDF Optimizer Service** (`app/services/pdf_optimizer.py`): pikepdf-based PDF optimization
- **Docker Configuration**: Containerized deployment with Ubuntu base image
- **Logging**: Comprehensive logging for optimization tracking

### File Structure
```
QStructurize/
├── app/
│   ├── services/
│   │   └── pdf_optimizer.py    # PDF optimization service
│   └── models/
│       └── schemas.py           # Pydantic models
├── main.py                     # FastAPI application
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml         # Docker Compose setup
└── README.md                  # This file
```

## Error Handling

- **415 Unsupported Media Type**: Non-PDF files
- **500 Internal Server Error**: PDF optimization errors
- **400 Bad Request**: Invalid parameters

## Dependencies

- **FastAPI**: Modern web framework for APIs
- **pikepdf**: PDF manipulation and optimization
- **uvicorn**: ASGI server for FastAPI
- **python-multipart**: File upload support

## Production Deployment

### Docker Environment
- **Base Image**: Python 3.11-slim
- **Target Platform**: Ubuntu with H200 GPU support
- **Port**: 8878
- **Volume Mount**: `./uploads:/app/uploads`

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