from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from pydantic import BaseModel
import logging
from app.services.pdf_optimizer import PDFOptimizer
from app.services.docling_parser import DoclingParser
from app.config import PIPELINE_OPTIONS_CONFIG, get_custom_openapi

# Configure logging
# Set to DEBUG to see Docling's internal pipeline profiling logs
import os
log_level = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.DEBUG),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
pdf_optimizer = PDFOptimizer()
docling_parser = DoclingParser()


class ParseResponse(BaseModel):
    """Response model for PDF parsing endpoint."""
    message: str
    status: str
    content: str | None = None  # The parsed content in markdown format


app = FastAPI(
    title="Q-Structurize",
    description="Advanced PDF parsing and structured text extraction API using Docling StandardPdfPipeline with ThreadedPdfPipelineOptions for batching and backpressure control. Features include layout analysis, optional OCR with multi-language support, configurable table extraction, batched processing, and multi-threaded processing optimized for 2x 72-core Xeon 6960P (144 cores).",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.post("/parse/file", 
          response_model=ParseResponse,
          summary="Parse PDF with Batched Processing",
          description="Upload a PDF file and get structured text output using Docling's StandardPdfPipeline with ThreadedPdfPipelineOptions for batching, backpressure control, and configurable options",
          tags=["PDF Parsing"],
          responses={
              200: {
                  "description": "PDF successfully parsed",
                  "content": {
                      "application/json": {
                          "example": {
                              "message": "PDF parsed successfully using batched processing",
                              "status": "success",
                              "content": "# Document Title\n\n## Section 1\n\nParagraph content...\n\n| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1   | Cell 2   |"
                          }
                      }
                  }
              },
              400: {"description": "Invalid request parameters"},
              415: {"description": "Invalid file type (must be PDF)"},
              500: {"description": "Server error during processing"},
              503: {"description": "Parser service unavailable"}
          })
async def parse_pdf_file(
    file: UploadFile = File(..., description="PDF file to parse", media_type="application/pdf"),
    optimize_pdf: bool = Form(True, description="Whether to optimize PDF for better text extraction")
):
    """
    Parse PDF file using pre-initialized Docling pipeline with optimized settings.
    
    **Configuration is set at container startup via Dockerfile ENV variables.**
    This approach provides:
    - 🚀 **Instant Processing**: Pre-loaded models, no initialization delay
    - ⚡ **Consistent Performance**: Same optimized settings for all requests  
    - 📐 **Layout Analysis**: Document structure (headings, paragraphs, lists)
    - 📄 **PDF Optimization**: Optional pre-processing for better extraction
    - 🔄 **Clean Output**: Structured markdown format
    
    **Parameters:**
    - **file**: PDF file to parse (required)
    - **optimize_pdf**: Pre-process PDF for better text extraction (default: true)
    
    **Configuration (Dockerfile ENV):**
    To change processing settings, update these ENV variables in Dockerfile and rebuild (takes ~10 seconds):
    - `OMP_NUM_THREADS`: Number of processing threads (default: 100, max: 144)
    - OCR, table extraction, and batching settings in parser initialization
    
    **Example Usage:**
    ```bash
    # Simple parsing
    curl -X POST "http://localhost:8878/parse/file" -F "file=@document.pdf"
    
    # Without PDF optimization (faster but may miss some text)
    curl -X POST "http://localhost:8878/parse/file" \\
      -F "file=@document.pdf" \\
      -F "optimize_pdf=false"
    ```
    
    **To Change Configuration:**
    1. Edit Dockerfile ENV variables or parser initialization
    2. Rebuild: `docker-compose build` (~10 seconds with cache)
    3. Restart: `docker-compose up`
    
    **Returns:**
    - Structured markdown content
    - Processing status and metadata
    - Error information if parsing fails
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('application/pdf'):
        raise HTTPException(status_code=415, detail="File must be a PDF")
    
    try:
        # ========================================
        # STEP 1: READ FILE CONTENT
        # ========================================
        pdf_content = await file.read()
        logger.info(f"Received PDF file: {file.filename}, size: {len(pdf_content)} bytes")
        
        # ========================================
        # STEP 2: PDF OPTIMIZATION (if requested)
        # ========================================
        if optimize_pdf:
            logger.info("Running PDF optimization...")
            pdf_content, size_info = PDFOptimizer.optimize_pdf(pdf_content)
            
            # Log optimization results for monitoring
            if size_info:
                logger.info(f"PDF optimization completed - Original: {size_info['original_size_bytes']} bytes, "
                           f"Optimized: {size_info['optimized_size_bytes']} bytes, "
                           f"Reduction: {size_info['size_reduction_percentage']}%")
        
        # ========================================
        # STEP 3: PDF PARSING WITH DOCLING BATCHED PROCESSING
        # ========================================
        if not docling_parser.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Docling parser is not available. Please check dependencies."
            )
        
        logger.info("Starting PDF parsing with pre-initialized Docling converter...")
        parse_result = docling_parser.parse_pdf(pdf_content)
        
        if not parse_result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"PDF parsing failed: {parse_result['error']}"
            )
        
        # Return successful parsing result
        return ParseResponse(
            message="PDF parsed successfully using pre-initialized converter",
            status="success",
            content=parse_result["content"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/", 
         summary="Health Check",
         description="Check if the API is running and get available features",
         tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "message": "Q-Structurize API is running", 
        "status": "healthy",
        "features": [
            "PDF optimization",
            "Pre-initialized Docling converter for instant processing",
            "Layout analysis and document structure extraction",
            "Batched processing with ThreadedPdfPipelineOptions",
            "ENV-based configuration (modify Dockerfile and rebuild)",
            "Multi-threaded processing (optimized for 2x 72-core Xeon 6960P)",
            "Structured markdown output"
        ],
        "version": "2.2.0",
        "endpoints": {
            "parse": "/parse/file - Parse PDF with configurable options",
            "parser_info": "/parsers/info - Get parser capabilities",
            "pipeline_options": "/parsers/options - Get available configuration options",
            "docs": "/docs - Swagger UI documentation",
            "redoc": "/redoc - ReDoc documentation"
        }
    }


@app.get("/parsers/info",
         summary="Get Parser Information",
         description="Get information about available PDF parsers and their status",
         tags=["System"])
async def get_parser_info():
    """
    Get information about available PDF parsers.
    
    Returns details about the Docling parser with batched processing including:
    - Availability status
    - Pipeline configuration (StandardPdfPipeline with ThreadedPdfPipelineOptions)
    - Models used (layout detection, OCR, table extraction)
    - Supported features
    - Batching capabilities
    - Performance characteristics
    - Configuration options
    """
    return docling_parser.get_parser_info()


@app.get("/parsers/options",
         summary="Get Available Pipeline Options",
         description="Get all available pipeline configuration options with defaults and descriptions",
         tags=["System"])
async def get_pipeline_options():
    """
    Get available pipeline configuration options.
    
    Returns a complete list of all configurable pipeline options including:
    - Option names and types
    - Default values
    - Valid ranges/values
    - Descriptions
    - Usage examples
    
    Use this endpoint to understand what options can be passed to the /parse/file endpoint
    via the pipeline_options parameter.
    """
    return PIPELINE_OPTIONS_CONFIG


# Configure custom OpenAPI schema for enhanced Swagger UI documentation
app.openapi = lambda: get_custom_openapi(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)