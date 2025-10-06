# Q-Structurize

Upload PDF and create structured chunks ready for RAG.

## Features

- **PDF Parsing**: Extract text from PDF files using PyMuPDF
- **Document Chunking**: Split documents into semantic chunks for RAG
- **Multiple Output Formats**: Support for Markdown, HTML, and plain text
- **Flexible Configuration**: Customizable chunk sizes and output options
- **Docker Support**: Ready for containerized deployment

## API Endpoints

### POST /parse/file

Parse PDF files and return structured text output.

**Parameters:**
- `file` (required): PDF file upload
- `chunk_document` (optional, boolean): Whether to split document into chunks (default: false)
- `max_tokens_per_chunk` (optional, int): Maximum tokens per chunk (default: 512)
- `output_format` (optional, string): Output format - "markdown", "text", or "html" (default: "markdown")
- `include_json` (optional, boolean): Include raw JSON representation (default: false)

**Response Format:**
```json
{
  "message": "Document parsed successfully",
  "status": "success",
  "data": {
    "chunks": [
      {
        "text": "chunk content...",
        "section_title": "optional section name",
        "chunk_index": 0,
        "metadata": {
          "pages": [1, 2],
          "content_type": "text"
        }
      }
    ],
    "output": "full document text...",
    "json_output": {...}
  }
}
```

## Quick Start

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

3. Access the API at `http://localhost:8000`

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Access the API at `http://localhost:8000`

## Usage Examples

### Basic PDF Parsing
```bash
curl -X POST "http://localhost:8000/parse/file" \
  -F "file=@document.pdf"
```

### Chunked Output
```bash
curl -X POST "http://localhost:8000/parse/file" \
  -F "file=@document.pdf" \
  -F "chunk_document=true" \
  -F "max_tokens_per_chunk=256"
```

### HTML Output
```bash
curl -X POST "http://localhost:8000/parse/file" \
  -F "file=@document.pdf" \
  -F "output_format=html"
```

## Error Handling

- **415 Unsupported Media Type**: Non-PDF files
- **500 Internal Server Error**: PDF parsing or conversion errors
- **400 Bad Request**: Invalid parameters

## Dependencies

- FastAPI: Web framework
- PyMuPDF: PDF text extraction
- PyPDF2: Additional PDF processing
- BeautifulSoup4: HTML processing
- Markdown: Markdown formatting