# Q-Structurize API Testing Guide

## 🚀 Quick Start

The API is running at `http://localhost:8000` with automatic Swagger UI documentation.

## 📖 Testing Methods

### 1. **Swagger UI (Recommended)**

**Access:** http://localhost:8000/docs

**Steps:**
1. Open your browser and go to `http://localhost:8000/docs`
2. You'll see the interactive API documentation
3. Click on the `POST /parse/file` endpoint
4. Click "Try it out"
5. Upload a PDF file using the file input
6. Set optional parameters:
   - `chunk_document`: false (default)
   - `max_tokens_per_chunk`: 512 (default)
   - `include_json`: true (to see metadata)
7. Click "Execute"
8. View the response in the "Response body" section

### 2. **cURL Commands**

**Basic Test:**
```bash
curl -X POST "http://localhost:8000/parse/file" \
  -F "file=@your-document.pdf"
```

**With JSON Output:**
```bash
curl -X POST "http://localhost:8000/parse/file" \
  -F "file=@your-document.pdf" \
  -F "include_json=true"
```

**With All Parameters:**
```bash
curl -X POST "http://localhost:8000/parse/file" \
  -F "file=@your-document.pdf" \
  -F "chunk_document=false" \
  -F "max_tokens_per_chunk=512" \
  -F "include_json=true"
```

### 3. **Health Check**

```bash
curl -X GET "http://localhost:8000/"
```

## 📋 Expected Responses

### Success Response:
```json
{
  "message": "Document received successfully",
  "status": "success",
  "data": {
    "output": "File received successfully",
    "json_output": {
      "filename": "document.pdf",
      "content_type": "application/pdf",
      "file_size": 12345,
      "chunked": false
    }
  }
}
```

### Error Responses:

**415 - Wrong File Type:**
```json
{
  "detail": "File must be a PDF"
}
```

**500 - Server Error:**
```json
{
  "detail": "Error processing file: [error message]"
}
```

## 🧪 Test Scenarios

### 1. **Valid PDF Upload**
- Upload any PDF file
- Should return success with file metadata

### 2. **Invalid File Type**
- Upload a non-PDF file (e.g., .txt, .jpg)
- Should return 415 error

### 3. **Large File**
- Upload a large PDF file
- Should handle gracefully

### 4. **Empty File**
- Upload an empty file
- Should return appropriate error

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/parse/file` | Upload and process PDF |
| GET | `/docs` | Swagger UI documentation |
| GET | `/openapi.json` | OpenAPI specification |

## 📊 Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 415 | Unsupported Media Type (non-PDF) |
| 500 | Internal Server Error |

## 🛠️ Development Testing

### Using Python requests:
```python
import requests

# Test file upload
with open('test.pdf', 'rb') as f:
    files = {'file': f}
    data = {
        'chunk_document': False,
        'max_tokens_per_chunk': 512,
        'include_json': True
    }
    response = requests.post('http://localhost:8000/parse/file', files=files, data=data)
    print(response.json())
```

### Using JavaScript fetch:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('include_json', 'true');

fetch('http://localhost:8000/parse/file', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## 🐛 Troubleshooting

1. **Port already in use**: Kill existing processes with `pkill -f "python3 main.py"`
2. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **File upload issues**: Check file size limits and content type
4. **Connection refused**: Verify the server is running on port 8000

## 📝 Notes

- The API currently only accepts file uploads without processing
- All PDF processing functionality has been removed for minimal implementation
- Parameters are preserved for future development
- Swagger UI provides the easiest way to test the API interactively
