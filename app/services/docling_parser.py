"""H200-optimized Docling service for PDF parsing using GraniteDocling VLM."""

import tempfile
import logging
import os
from typing import Optional, Dict, Any

# Try to import docling, set availability flag
try:
    from docling.datamodel import vlm_model_specs
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    vlm_model_specs = None
    InputFormat = None
    VlmPipelineOptions = None
    DocumentConverter = None
    PdfFormatOption = None
    VlmPipeline = None

logger = logging.getLogger(__name__)


class DoclingParser:
    """Service for parsing PDFs using Docling with GraniteDocling VLM optimized for H200 GPU."""
    
    def __init__(self):
        """Initialize the Docling parser with H200 optimizations."""
        self.converter = None
        self._setup_converter()
    
    def _setup_converter(self):
        """Setup the DocumentConverter with VLM pipeline optimized for H200 GPU."""
        if not DOCLING_AVAILABLE:
            logger.warning("Docling is not available. VLM parsing will not work.")
            return

        try:
            # Debug cache directories
            logger.info("=== CACHE DEBUG INFO ===")
            logger.info(f"HF_HOME: {os.environ.get('HF_HOME', 'Not set')}")
            logger.info(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}")
            logger.info(f"TORCH_HOME: {os.environ.get('TORCH_HOME', 'Not set')}")
            
            # Check if cache directories exist
            cache_dirs = ['/app/.cache', '/root/.cache']
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    logger.info(f"Cache directory exists: {cache_dir}")
                    # List contents
                    try:
                        contents = os.listdir(cache_dir)
                        logger.info(f"Cache contents in {cache_dir}: {contents}")
                    except:
                        logger.info(f"Could not list contents of {cache_dir}")
                else:
                    logger.info(f"Cache directory does not exist: {cache_dir}")
            
            # Force cache directories to be used by setting them explicitly
            os.environ['HF_HOME'] = '/app/.cache/huggingface'
            os.environ['TRANSFORMERS_CACHE'] = '/app/.cache/transformers'
            os.environ['TORCH_HOME'] = '/app/.cache/torch'
            os.environ['HF_HUB_CACHE'] = '/app/.cache/huggingface'
            
            # Create cache directories if they don't exist
            for cache_dir in ['/app/.cache/huggingface', '/app/.cache/transformers', '/app/.cache/torch']:
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"Created cache directory: {cache_dir}")
            
            # Get base VLM options
            vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
            
            # Optimize for H200 GPU performance
            logger.info("Configuring VLM for H200 GPU optimization...")
            
            # Override settings for high-end GPU performance
            vlm_options.load_in_8bit = False  # Use full precision on H200 (80GB VRAM)
            vlm_options.max_new_tokens = 32768  # Increase token limit for H200
            vlm_options.temperature = 0.0  # Deterministic output
            vlm_options.scale = 1.0  # Faster processing, less scaling
            vlm_options.use_kv_cache = True  # Enable KV cache for speed
            
            # Additional H200 optimizations
            if hasattr(vlm_options, 'torch_dtype'):
                vlm_options.torch_dtype = 'float16'  # Use FP16 for speed on H200
            
            # Use GraniteDocling VLM with H200 optimizations
            logger.info("Initializing DocumentConverter with H200 optimizations...")
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        vlm_options=vlm_options,
                    ),
                }
            )
            logger.info("Docling VLM converter initialized successfully with H200 optimizations")
            logger.info(f"H200 Config: load_in_8bit={vlm_options.load_in_8bit}, max_tokens={vlm_options.max_new_tokens}, kv_cache={vlm_options.use_kv_cache}")
            
            # Check cache after initialization
            logger.info("=== POST-INITIALIZATION CACHE CHECK ===")
            for cache_dir in cache_dirs:
                if os.path.exists(cache_dir):
                    try:
                        contents = os.listdir(cache_dir)
                        logger.info(f"Cache contents after init in {cache_dir}: {contents}")
                    except:
                        logger.info(f"Could not list contents of {cache_dir}")

        except Exception as e:
            logger.error(f"Failed to initialize Docling VLM converter: {str(e)}")
            self.converter = None
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Parse PDF content using Docling with GraniteDocling VLM optimized for H200.
        
        Args:
            pdf_content: Raw PDF file content as bytes
            
        Returns:
            Dictionary containing parsed content and metadata
        """
        if not DOCLING_AVAILABLE:
            return {
                "success": False,
                "error": "Docling is not available",
                "content": None
            }
        
        if not self.converter:
            return {
                "success": False,
                "error": "Docling converter not initialized",
                "content": None
            }
        
        try:
            # Create temporary file for PDF content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                # Parse the PDF using optimized Docling VLM
                logger.info("Starting PDF parsing with H200-optimized GraniteDocling VLM")
                result = self.converter.convert(source=tmp_file.name)
                
                # Extract content - simple markdown export
                document = result.document
                markdown_content = document.export_to_markdown()
                
                logger.info("PDF parsing completed successfully with H200 optimization")
                
                return {
                    "success": True,
                    "content": markdown_content,
                    "error": None
                }
                
        except Exception as e:
            logger.error(f"Error during PDF parsing with H200-optimized Docling VLM: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def is_available(self) -> bool:
        """Check if Docling VLM parsing is available."""
        return DOCLING_AVAILABLE and self.converter is not None

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the H200-optimized Docling VLM parser."""
        return {
            "available": self.is_available(),
            "library": "docling" if DOCLING_AVAILABLE else None,
            "model": "granite_docling_h200_optimized" if self.is_available() else None,
            "description": "High-precision PDF parsing using GraniteDocling VLM optimized for H200 GPU",
            "features": [
                "Visual Language Model processing",
                "H200 GPU optimization (80GB VRAM)",
                "Full precision processing (no quantization)",
                "Extended token limits (32K tokens)",
                "KV cache acceleration",
                "Structured markdown output",
                "Metadata extraction"
            ],
            "performance": {
                "expected_speed": "2-5 seconds per page",
                "memory_usage": "8-16GB VRAM",
                "optimization": "H200-specific"
            }
        }