"""Simple Docling service for PDF parsing using GraniteDocling VLM."""

import tempfile
import logging
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
    """Service for parsing PDFs using Docling with GraniteDocling VLM for maximum precision."""
    
    def __init__(self):
        """Initialize the Docling parser."""
        self.converter = None
        self._setup_converter()
    
    def _setup_converter(self):
        """Setup the DocumentConverter with VLM pipeline (simple version)."""
        if not DOCLING_AVAILABLE:
            logger.warning("Docling is not available. VLM parsing will not work.")
            return

        try:
            # Use GraniteDocling VLM explicitly for maximum precision
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        vlm_options=vlm_model_specs.GRANITEDOCLING_TRANSFORMERS,
                    ),
                }
            )
            logger.info("Docling VLM converter initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Docling VLM converter: {str(e)}")
            self.converter = None
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Parse PDF content using Docling with GraniteDocling VLM for maximum precision.
        
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
                
                # Parse the PDF using Docling VLM (simple version)
                logger.info("Starting PDF parsing with GraniteDocling VLM")
                result = self.converter.convert(source=tmp_file.name)
                
                # Extract content - simple markdown export
                document = result.document
                markdown_content = document.export_to_markdown()
                
                logger.info("PDF parsing completed successfully")
                
                return {
                    "success": True,
                    "content": markdown_content,
                    "error": None
                }
                
        except Exception as e:
            logger.error(f"Error during PDF parsing with Docling VLM: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def is_available(self) -> bool:
        """Check if Docling VLM parsing is available."""
        return DOCLING_AVAILABLE and self.converter is not None

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the Docling VLM parser."""
        return {
            "available": self.is_available(),
            "library": "docling" if DOCLING_AVAILABLE else None,
            "model": "granite_docling" if self.is_available() else None,
            "description": "High-precision PDF parsing using GraniteDocling VLM",
            "features": [
                "Visual Language Model processing",
                "Maximum precision text extraction",
                "Structured markdown output",
                "Metadata extraction"
            ]
        }
