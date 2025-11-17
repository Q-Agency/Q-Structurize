from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import Optional
from pydantic import BaseModel
import logging
import inspect
from app.services.pdf_optimizer import PDFOptimizer
from app.services.docling_parser import DoclingParser
from app.services import hybrid_chunker
from app.services.tokenizer_manager import get_tokenizer_manager
from app.models.schemas import ParseResponse, ChunkData, ChunkMetadata, ChunkingData
from app.config import PIPELINE_OPTIONS_CONFIG, get_custom_openapi

# Configure logging
import os
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize services
pdf_optimizer = PDFOptimizer()
docling_parser = DoclingParser()

# Lazy initialization for image description parser (only created when needed)
docling_parser_images = None

def get_image_parser():
    """Lazy initialization of image description parser."""
    global docling_parser_images
    if docling_parser_images is None:
        try:
            from app.services.docling_parser_images import DoclingParserImages
            docling_parser_images = DoclingParserImages()
            if not docling_parser_images.is_available():
                logger.warning("⚠️  Image description parser is not available")
                docling_parser_images = None
            else:
                logger.info("✅ Image description parser initialized")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize image description parser: {str(e)}")
            docling_parser_images = None
    return docling_parser_images


app = FastAPI(
    title="Q-Structurize",
    description="Advanced PDF parsing and structured text extraction API using Docling StandardPdfPipeline with ThreadedPdfPipelineOptions for batching and backpressure control. Features include layout analysis, hybrid chunking for RAG with custom HuggingFace tokenizers, optional OCR with multi-language support, configurable table extraction, batched processing, and multi-threaded processing optimized for 2x 72-core Xeon 6960P (144 cores).",
    version="2.4.0",
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
    optimize_pdf: bool = Form(True, description="Whether to optimize PDF for better text extraction"),
    enable_chunking: bool = Form(False, description="Enable hybrid chunking for RAG and semantic search"),
    max_tokens_per_chunk: int = Form(512, ge=128, le=2048, description="Maximum tokens per chunk (128-2048)"),
    merge_peers: bool = Form(True, description="Merge undersized successive chunks with same headings"),
    embedding_model: Optional[str] = Form(None, description="HuggingFace embedding model name for tokenization (e.g., 'sentence-transformers/all-MiniLM-L6-v2'). If not specified, uses HybridChunker's built-in tokenizer"),
    include_markdown: bool = Form(False, description="Include full markdown content when chunking is enabled"),
    include_full_metadata: bool = Form(False, description="Include complete Docling metadata (model_dump) in addition to curated metadata"),
    serialize_tables: bool = Form(False, description="Serialize table chunks as key-value pairs optimized for embeddings (extracts tables from document structure)"),
    semantic_refinement: bool = Form(False, description="Apply LlamaIndex semantic chunking refinement to improve chunk boundaries. Requires embedding_model to be specified."),
    parse_images: bool = Form(False, description="Enable image description for images in PDF (requires VLM model or API key configured via environment variables)"),
    image_description_prompt: Optional[str] = Form(None, description="Custom prompt for image description (e.g., 'Describe the image in one word' or 'Describe the image in detail.'). If not provided, uses prompt from environment variable DOCLING_PICTURE_DESCRIPTION_PROMPT or model default.")
):
    # Validate file type
    if not file.content_type or not file.content_type.startswith('application/pdf'):
        raise HTTPException(status_code=415, detail="File must be a PDF")
    
    try:
        # Read file content
        pdf_content = await file.read()
        logger.info(f"📄 Received: {file.filename} ({len(pdf_content):,} bytes)")
        
        # PDF optimization (if requested)
        if optimize_pdf:
            pdf_content, size_info = PDFOptimizer.optimize_pdf(pdf_content)
        
        # Select parser based on parse_images flag
        if parse_images:
            # Use image description parser
            image_parser = get_image_parser()
            if image_parser is None or not image_parser.is_available():
                logger.warning("⚠️  Image description requested but parser not available, falling back to standard parser")
                parser = docling_parser
            else:
                parser = image_parser
                logger.info("📸 Using image description parser")
        else:
            # Use standard parser
            parser = docling_parser
        
        # PDF parsing availability check
        if not parser.is_available():
            raise HTTPException(
                status_code=503, 
                detail="Docling parser is not available. Please check dependencies."
            )
        
        # Prepare config override for image description prompt if provided
        config_override = None
        if parse_images and image_description_prompt:
            config_override = {"prompt": image_description_prompt.strip()}
            logger.info(f"📝 Using custom image description prompt from API: '{image_description_prompt}'")
        
        # Branch: Chunking vs Standard Markdown
        if enable_chunking:
            # Parse to DoclingDocument object
            # Check if parser supports config_override (image parser)
            if hasattr(parser, 'parse_pdf_to_document') and config_override:
                # Check if method signature accepts config_override
                sig = inspect.signature(parser.parse_pdf_to_document)
                if 'config_override' in sig.parameters:
                    document = parser.parse_pdf_to_document(pdf_content, config_override=config_override)
                else:
                    document = parser.parse_pdf_to_document(pdf_content)
            else:
                document = parser.parse_pdf_to_document(pdf_content)
            
            # Load tokenizer if embedding model is specified
            tokenizer = None
            if embedding_model:
                try:
                    tokenizer_manager = get_tokenizer_manager()
                    tokenizer = tokenizer_manager.get_tokenizer(embedding_model)
                    logger.info(f"Using custom tokenizer: {embedding_model}")
                except Exception as e:
                    logger.error(f"❌ Failed to load tokenizer '{embedding_model}': {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid embedding model: {str(e)}"
                    )
            
            # Chunk document with optional full metadata, table serialization, and semantic refinement
            chunks = hybrid_chunker.chunk_document(
                document=document,
                max_tokens=max_tokens_per_chunk,
                merge_peers=merge_peers,
                tokenizer=tokenizer,
                include_full_metadata=include_full_metadata,
                serialize_tables=serialize_tables,
                semantic_refinement=semantic_refinement
            )
            
            # Convert chunk dicts to ChunkData models
            chunk_models = [
                ChunkData(
                    section_title=chunk["section_title"],
                    text=chunk["text"],
                    chunk_index=chunk["chunk_index"],
                    metadata=ChunkMetadata(**chunk["metadata"]),
                    full_metadata=chunk.get("full_metadata")  # Include if present
                )
                for chunk in chunks
            ]
            
            logger.info(f"✅ Generated {len(chunk_models)} chunks")
            
            return ParseResponse(
                message=f"Document chunked successfully ({len(chunk_models)} chunks)",
                status="success",
                data=ChunkingData(chunks=chunk_models)
            )
        else:
            # Standard markdown parsing
            # Check if parser supports config_override (image parser)
            if hasattr(parser, 'parse_pdf') and config_override:
                # Check if method signature accepts config_override
                sig = inspect.signature(parser.parse_pdf)
                if 'config_override' in sig.parameters:
                    parse_result = parser.parse_pdf(pdf_content, config_override=config_override)
                else:
                    parse_result = parser.parse_pdf(pdf_content)
            else:
                parse_result = parser.parse_pdf(pdf_content)
            
            if not parse_result["success"]:
                raise HTTPException(
                    status_code=500, 
                    detail=f"PDF parsing failed: {parse_result['error']}"
                )
            
            # Return successful result
            return ParseResponse(
                message="PDF parsed successfully",
                status="success",
                content=parse_result["content"]
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing file: {str(e)}")
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
            "Hybrid chunking with native merge_peers for RAG",
            "Table serialization for embeddings (key-value format)",
            "Custom embedding model tokenizers (any HuggingFace model)",
            "Batched processing with ThreadedPdfPipelineOptions",
            "ENV-based configuration (modify Dockerfile and rebuild)",
            "Multi-threaded processing (optimized for 2x 72-core Xeon 6960P)",
            "Structured markdown or semantic chunks output",
            "Image description support (optional, via parse_images parameter)"
        ],
        "version": "2.4.0",
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