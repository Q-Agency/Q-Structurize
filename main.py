from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from pydantic import BaseModel
from enum import Enum
import logging
from app.services.pdf_optimizer import PDFOptimizer
from app.services.docling_parser import DoclingParser
from app.models.schemas import PipelineOptions, TableMode, AcceleratorDevice
from app.config import PIPELINE_OPTIONS_CONFIG, get_custom_openapi

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    description="Advanced PDF parsing and structured text extraction API using Docling ThreadedPdfPipeline with batching and backpressure control. Features include layout analysis, optional OCR with multi-language support, configurable table extraction, batched processing, and multi-threaded processing optimized for 2x 72-core Xeon 6960P (144 cores).",
    version="2.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.post("/parse/file", 
          response_model=ParseResponse,
          summary="Parse PDF with Docling ThreadedPipeline",
          description="Upload a PDF file and get structured text output using Docling's ThreadedPdfPipeline with batching, backpressure control, and configurable options",
          tags=["PDF Parsing"],
          responses={
              200: {
                  "description": "PDF successfully parsed",
                  "content": {
                      "application/json": {
                          "example": {
                              "message": "PDF parsed successfully using Docling ThreadedPdfPipeline",
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
    max_tokens_per_chunk: int = Form(512, description="Maximum tokens per chunk (reserved for future use)"),
    optimize_pdf: bool = Form(True, description="Whether to optimize PDF for better text extraction"),
    # Pipeline options as individual parameters
    enable_ocr: bool = Form(False, description="Enable OCR for scanned documents and images"),
    ocr_languages: str = Form("en", description="Comma-separated OCR language codes (e.g., 'en', 'en,es,de')"),
    table_mode: TableMode = Form(TableMode.FAST, description="Table extraction mode: 'fast' or 'accurate'"),
    do_table_structure: bool = Form(True, description="Enable table structure extraction"),
    do_cell_matching: bool = Form(True, description="Enable cell matching for better table accuracy"),
    num_threads: int = Form(8, ge=1, le=144, description="Number of processing threads (1-144, optimized for 72-core Xeon)"),
    accelerator_device: AcceleratorDevice = Form(AcceleratorDevice.CPU, description="Accelerator device: 'cpu', 'cuda', or 'auto'"),
    # Enrichment options (advanced features, increase processing time)
    do_code_enrichment: bool = Form(False, description="Enable code block language detection and parsing"),
    do_formula_enrichment: bool = Form(False, description="Enable formula analysis and LaTeX extraction"),
    do_picture_classification: bool = Form(False, description="Enable image classification (charts, diagrams, logos, etc.)"),
    do_picture_description: bool = Form(False, description="Enable AI-powered image description (requires VLM, significantly increases processing time)"),
    # ThreadedPdfPipeline batching options (performance tuning)
    layout_batch_size: int = Form(4, ge=1, le=32, description="Batch size for layout detection (1-32, higher = more throughput, more memory)"),
    ocr_batch_size: int = Form(4, ge=1, le=32, description="Batch size for OCR processing (1-32, higher = more throughput, more memory)"),
    table_batch_size: int = Form(4, ge=1, le=32, description="Batch size for table extraction (1-32, higher = more throughput, more memory)"),
    queue_max_size: int = Form(100, ge=10, le=1000, description="Maximum queue size for backpressure control (10-1000)"),
    batch_timeout_seconds: float = Form(2.0, ge=0.1, le=30.0, description="Batch processing timeout in seconds (0.1-30.0)")
):
    """
    Parse PDF file using Docling's ThreadedPdfPipeline with batching and configurable options.
    
    This endpoint processes PDF files using Docling's high-performance threaded pipeline which includes:
    - **Layout Detection**: Document structure analysis using DocLayNet model
    - **Text Extraction**: High-quality text extraction from PDF layers
    - **OCR Processing**: Optional text extraction from images and scanned documents using EasyOCR
    - **Table Extraction**: Accurate table structure preservation using TableFormer
    - **Batched Processing**: Process multiple pages/operations in parallel
    - **Backpressure Control**: Queue management to prevent memory overflow
    - **Structured Output**: Clean markdown with proper formatting
    
    **Features:**
    - 📐 **Layout Analysis**: Understands document structure (headings, paragraphs, lists)
    - 📊 **Table Extraction**: Configurable FAST or ACCURATE modes
    - 🔍 **OCR Support**: Optional with multi-language support
    - 📄 **PDF Optimization**: Pre-processing for better results  
    - 🔄 **Structured Output**: Clean markdown format
    - ⚡ **Multi-threading**: Optimized for 2x 72-core Xeon 6960P (144 cores)
    - 🚀 **Batched Processing**: Parallel processing for better throughput
    - 🎚️ **Backpressure Control**: Queue management for large documents
    - ⚙️ **Per-request Configuration**: Customize pipeline per document
    
    **Parameters:**
    - **file**: PDF file to parse (required)
    - **optimize_pdf**: Whether to optimize PDF for better text extraction (default: true)
    - **enable_ocr**: Enable OCR for scanned documents (default: false)
    - **ocr_languages**: Comma-separated language codes (default: "en", e.g., "en,es,de")
    - **table_mode**: Table extraction mode "fast" or "accurate" (default: "fast")
    - **do_table_structure**: Enable table extraction (default: true)
    - **do_cell_matching**: Enable cell matching (default: true)
    - **num_threads**: Number of threads 1-144 (default: 8)
    - **accelerator_device**: Device selection "cpu", "cuda", or "auto" (default: "cpu")
    - **layout_batch_size**: Batch size for layout detection 1-32 (default: 4)
    - **ocr_batch_size**: Batch size for OCR processing 1-32 (default: 4)
    - **table_batch_size**: Batch size for table extraction 1-32 (default: 4)
    - **queue_max_size**: Queue size for backpressure control 10-1000 (default: 100)
    - **batch_timeout_seconds**: Batch timeout 0.1-30.0 seconds (default: 2.0)
    
    **Example Usage:**
    
    Default (fast processing):
    ```bash
    curl -X POST "http://localhost:8878/parse/file" -F "file=@document.pdf"
    ```
    
    With OCR enabled:
    ```bash
    curl -X POST "http://localhost:8878/parse/file" \\
      -F "file=@document.pdf" \\
      -F "enable_ocr=true" \\
      -F "num_threads=16"
    ```
    
    High performance with accurate tables:
    ```bash
    curl -X POST "http://localhost:8878/parse/file" \\
      -F "file=@document.pdf" \\
      -F "table_mode=accurate" \\
      -F "num_threads=64"
    ```
    
    Maximum throughput with batching:
    ```bash
    curl -X POST "http://localhost:8878/parse/file" \\
      -F "file=@document.pdf" \\
      -F "num_threads=64" \\
      -F "layout_batch_size=16" \\
      -F "table_batch_size=16" \\
      -F "queue_max_size=500"
    ```
    
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
        # BUILD PIPELINE OPTIONS from individual parameters
        # ========================================
        # Parse comma-separated language codes into list
        ocr_lang_list = [lang.strip() for lang in ocr_languages.split(',') if lang.strip()]
        
        # Build options dictionary
        options_dict = {
            "enable_ocr": enable_ocr,
            "ocr_languages": ocr_lang_list,
            "table_mode": table_mode.value,  # Convert enum to string
            "do_table_structure": do_table_structure,
            "do_cell_matching": do_cell_matching,
            "num_threads": num_threads,
            "accelerator_device": accelerator_device.value,  # Convert enum to string
            # Enrichment options
            "do_code_enrichment": do_code_enrichment,
            "do_formula_enrichment": do_formula_enrichment,
            "do_picture_classification": do_picture_classification,
            "do_picture_description": do_picture_description,
            # ThreadedPdfPipeline batching options
            "layout_batch_size": layout_batch_size,
            "ocr_batch_size": ocr_batch_size,
            "table_batch_size": table_batch_size,
            "queue_max_size": queue_max_size,
            "batch_timeout_seconds": batch_timeout_seconds
        }
        
        # Validate using Pydantic model
        try:
            validated_options = PipelineOptions(**options_dict)
            parsed_options = validated_options.model_dump()
            logger.info(f"Using pipeline options: OCR={enable_ocr}, threads={num_threads}, table_mode={table_mode.value}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid pipeline options: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building pipeline options: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error building pipeline options: {str(e)}")
    
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
        # STEP 3: PDF PARSING WITH DOCLING THREADED PIPELINE
        # ========================================
        if not docling_parser.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Docling parser is not available. Please check dependencies."
            )
        
        logger.info("Starting PDF parsing with Docling ThreadedPdfPipeline...")
        parse_result = docling_parser.parse_pdf(pdf_content, options=parsed_options)
        
        if not parse_result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"PDF parsing failed: {parse_result['error']}"
            )
        
        # Return successful parsing result
        return ParseResponse(
            message="PDF parsed successfully using Docling ThreadedPdfPipeline",
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
            "Docling ThreadedPdfPipeline with batching",
            "Configurable pipeline options per request",
            "Layout analysis",
            "Optional OCR with multi-language support",
            "Configurable table extraction (fast/accurate)",
            "Batched processing for better throughput",
            "Backpressure control for large documents",
            "Multi-threaded processing (optimized for 2x 72-core Xeon 6960P)"
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
    
    Returns details about the Docling ThreadedPdfPipeline parser including:
    - Availability status
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