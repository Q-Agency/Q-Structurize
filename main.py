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


app = FastAPI(
    title="Q-Structurize",
    description="High-precision PDF parsing and structured text extraction API using Docling VLM",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


@app.post("/parse/file", 
          response_model=ParseResponse,
          summary="Parse PDF with GraniteDocling VLM",
          description="Upload a PDF file and get structured text output using Docling VLM for maximum precision",
          tags=["PDF Parsing"])
async def parse_pdf_file(
    file: UploadFile = File(..., description="PDF file to parse", media_type="application/pdf"),
    max_tokens_per_chunk: int = Form(512, description="Maximum tokens per chunk (reserved for future use)"),
    optimize_pdf: bool = Form(True, description="Whether to optimize PDF for better text extraction"),
    use_vlm: bool = Form(True, description="Whether to use Docling VLM for maximum precision")
):
    """
    Parse PDF file using Docling VLM for maximum precision.
    
    This endpoint processes PDF files using advanced Vision-Language Model (VLM) technology
    to extract structured text with maximum accuracy. The VLM can understand both text and
    visual elements in the document.
    
    **Features:**
    - 🎯 **GraniteDocling VLM**: Maximum precision PDF parsing
    - 📄 **PDF Optimization**: Pre-processing for better results  
    - 🔄 **Structured Output**: Clean markdown with visual understanding
    - ⚡ **Fast Processing**: Optimized for production use
    
    **Parameters:**
    - **file**: PDF file to parse (required)
    - **max_tokens_per_chunk**: Maximum tokens per chunk (reserved for future use)
    - **optimize_pdf**: Whether to optimize PDF for better text extraction (default: true)
    - **use_vlm**: Whether to use Docling VLM for maximum precision (default: true)
    
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
        # STEP 3: PDF PARSING WITH DOCLING VLM
        # ========================================
        if use_vlm:
            if not docling_parser.is_available():
                raise HTTPException(
                    status_code=503, 
                    detail="Docling VLM parser is not available. Please check dependencies."
                )
            
            logger.info("Starting PDF parsing with GraniteDocling VLM...")
            parse_result = docling_parser.parse_pdf(pdf_content)
            
            if not parse_result["success"]:
                raise HTTPException(
                    status_code=500, 
                    detail=f"PDF parsing failed: {parse_result['error']}"
                )
            
            # Return successful parsing result
            return ParseResponse(
                message="PDF parsed successfully using GraniteDocling VLM",
                status="success",
                content=parse_result["content"]
            )
        else:
            # Fallback to basic processing
            return ParseResponse(
                message="PDF received successfully (VLM parsing disabled)",
                status="success",
                content=None
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
        "features": ["PDF optimization", "Docling VLM parsing"],
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)