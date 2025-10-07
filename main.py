from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Dict, Any
from pydantic import BaseModel
import logging
from app.services.pdf_optimizer import PDFOptimizer
from app.services.docling_parser import DoclingParser

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
    description="Advanced PDF parsing and structured text extraction API using Docling StandardPdfPipeline with layout analysis, OCR, and table extraction",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.post("/parse/file", 
          response_model=ParseResponse,
          summary="Parse PDF with Docling StandardPipeline",
          description="Upload a PDF file and get structured text output using Docling's StandardPdfPipeline with layout analysis, OCR, and table extraction",
          tags=["PDF Parsing"])
async def parse_pdf_file(
    file: UploadFile = File(..., description="PDF file to parse", media_type="application/pdf"),
    max_tokens_per_chunk: int = Form(512, description="Maximum tokens per chunk (reserved for future use)"),
    optimize_pdf: bool = Form(True, description="Whether to optimize PDF for better text extraction")
):
    """
    Parse PDF file using Docling's StandardPdfPipeline.
    
    This endpoint processes PDF files using Docling's robust standard pipeline which includes:
    - **Layout Detection**: Document structure analysis using DocLayNet model
    - **Text Extraction**: High-quality text extraction from PDF layers
    - **OCR Processing**: Text extraction from images and scanned documents using EasyOCR
    - **Table Extraction**: Accurate table structure preservation using TableFormer
    - **Structured Output**: Clean markdown with proper formatting
    
    **Features:**
    - 📐 **Layout Analysis**: Understands document structure (headings, paragraphs, lists)
    - 📊 **Table Extraction**: Preserves table structure and formatting
    - 🔍 **OCR Support**: Handles scanned documents and images
    - 📄 **PDF Optimization**: Pre-processing for better results  
    - 🔄 **Structured Output**: Clean markdown format
    - ⚡ **CPU-Optimized**: Fast processing without GPU
    
    **Parameters:**
    - **file**: PDF file to parse (required)
    - **max_tokens_per_chunk**: Maximum tokens per chunk (reserved for future use)
    - **optimize_pdf**: Whether to optimize PDF for better text extraction (default: true)
    
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
        # STEP 3: PDF PARSING WITH DOCLING STANDARD PIPELINE
        # ========================================
        if not docling_parser.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Docling parser is not available. Please check dependencies."
            )
        
        logger.info("Starting PDF parsing with Docling StandardPdfPipeline...")
        parse_result = docling_parser.parse_pdf(pdf_content)
        
        if not parse_result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"PDF parsing failed: {parse_result['error']}"
            )
        
        # Return successful parsing result
        return ParseResponse(
            message="PDF parsed successfully using Docling StandardPdfPipeline",
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
            "Docling StandardPdfPipeline",
            "Layout analysis",
            "OCR processing",
            "Table extraction"
        ],
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/parsers/info",
         summary="Get Parser Information",
         description="Get information about available PDF parsers and their status",
         tags=["System"])
async def get_parser_info():
    """
    Get information about available PDF parsers.
    
    Returns details about the Docling StandardPdfPipeline parser including:
    - Availability status
    - Models used (layout detection, OCR, table extraction)
    - Supported features
    - Performance characteristics
    - Cache information
    """
    return docling_parser.get_parser_info()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)