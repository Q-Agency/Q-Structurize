from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Dict, Any
from pydantic import BaseModel
import logging
from app.services.pdf_optimizer import PDFOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ParseResponse(BaseModel):
    """Response model for PDF parsing endpoint."""
    message: str
    status: str


app = FastAPI(
    title="Q-Structurize",
    description="PDF parsing and structured text extraction API",
    version="1.0.0"
)


@app.post("/parse/file", response_model=ParseResponse)
async def parse_pdf_file(
    file: UploadFile = File(...),
    max_tokens_per_chunk: int = Form(512),
    optimize_pdf: bool = Form(True)
):
    """
    Parse PDF file and return structured text output.
    
    - **file**: PDF file to parse
    - **max_tokens_per_chunk**: Maximum tokens per chunk (reserved for future use)
    - **optimize_pdf**: Whether to optimize PDF for better text extraction (default: true)
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('application/pdf'):
        raise HTTPException(status_code=415, detail="File must be a PDF")
    
    try:
        # ========================================
        # STEP 1: READ FILE CONTENT
        # ========================================
        pdf_content = await file.read()
        
        # ========================================
        # STEP 2: PDF OPTIMIZATION (if requested)
        # ========================================
        if optimize_pdf:
            # Run PDF optimization using pikepdf
            pdf_content, size_info = PDFOptimizer.optimize_pdf(pdf_content)
            
            # Log optimization results for monitoring
            if size_info:
                logger.info(f"PDF optimization completed - Original: {size_info['original_size_bytes']} bytes, "
                           f"Optimized: {size_info['optimized_size_bytes']} bytes, "
                           f"Reduction: {size_info['size_reduction_percentage']}%")
        
        # ========================================
        # STEP 3: RETURN SUCCESS RESPONSE
        # ========================================
        return ParseResponse(
            message="Document received successfully",
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Q-Structurize API is running", "status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)