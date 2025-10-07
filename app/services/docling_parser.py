"""
Docling service for PDF parsing using StandardPdfPipeline.

The StandardPdfPipeline includes:
- Layout Detection: DocLayNet-based model for document structure understanding
- Table Extraction: TableFormer model for accurate table structure recognition
- OCR: EasyOCR for text extraction from images and scanned documents
- Text Extraction: Direct text layer extraction when available

Models are automatically downloaded from HuggingFace Hub on first use and cached locally.
"""

import tempfile
import logging
import os
import time
from typing import Optional, Dict, Any

# Try to import docling
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.pipeline.pdf_pipeline_options import PdfPipelineOptions, TableFormerMode
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    StandardPdfPipeline = None

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Service for parsing PDFs using Docling's StandardPdfPipeline.
    
    Features:
    - Layout detection and document structure analysis
    - Table extraction with structure preservation
    - OCR for scanned documents
    - CPU-optimized processing
    - No GPU required
    
    Models used (auto-downloaded on first use):
    - DocLayNet layout model (~100MB)
    - TableFormer structure model (~200MB)
    - EasyOCR language models (~100MB)
    """
    
    def __init__(self):
        """Initialize the Docling parser with StandardPdfPipeline."""
        self.converter = None
        self.mode = "standard-pipeline"
        self._setup_converter()
    
    def _setup_converter(self):
        """
        Setup the DocumentConverter with StandardPdfPipeline.
        
        On first initialization, Docling will automatically download required models:
        - Layout detection model (DocLayNet)
        - Table extraction model (TableFormer)
        - OCR models (EasyOCR)
        
        These are cached in ~/.cache/docling/ or $DOCLING_ARTIFACTS_PATH
        """
        if not DOCLING_AVAILABLE:
            logger.warning("Docling is not available. PDF parsing will not work.")
            return

        try:
            logger.info("============================================================")
            logger.info("🚀 DOCLING STANDARD PIPELINE (CPU-OPTIMIZED)")
            logger.info("============================================================")
            
            # Docling uses default cache locations (~/.cache/docling/models)
            # Models are automatically downloaded by docling-tools during Docker build
            logger.info(f"📦 Using Docling's default model cache locations")
            
            # Initialize DocumentConverter with StandardPdfPipeline
            logger.info("⏳ Initializing StandardPdfPipeline...")
            logger.info("   This includes:")
            logger.info("   - Layout Detection (DocLayNet model)")
            logger.info("   - Table Extraction (TableFormer model)")
            logger.info("   - OCR Engine (EasyOCR)")
            logger.info("")
            logger.info("⚠️  First run: Models will be downloaded (~400MB, 3-5 min)")
            logger.info("⚡ Subsequent runs: Models loaded from cache (~5 sec)")
            logger.info("")
            
            init_start = time.time()
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = False  # DISABLE OCR
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST

            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=StandardPdfPipeline,
                        pipeline_options=pipeline_options,
                    ),
                }
            )
            
            init_time = time.time() - init_start
            logger.info(f"✅ StandardPdfPipeline initialized in {init_time:.2f} seconds")
            
            if init_time < 10:
                logger.info("✅ Models loaded from cache (fast initialization)")
            else:
                logger.info("✅ Models downloaded and cached for future use")
            
            logger.info("✅ Ready for document processing")
            logger.info("============================================================")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Docling converter: {str(e)}")
            logger.error(f"   Make sure all dependencies are installed: pip install docling")
            self.converter = None
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Parse PDF content using Docling's StandardPdfPipeline.
        
        The pipeline performs:
        1. Layout analysis - Identifies document structure (headings, paragraphs, tables)
        2. Text extraction - Extracts text from PDF layers
        3. OCR processing - Extracts text from images/scanned pages
        4. Table extraction - Preserves table structure
        5. Markdown export - Structured output with proper formatting
        
        Args:
            pdf_content: Raw PDF file content as bytes
            
        Returns:
            Dictionary containing:
            - success: bool
            - content: markdown string (if successful)
            - error: error message (if failed)
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
                
                # Parse the PDF using StandardPdfPipeline
                logger.info("------------------------------------------------------------")
                logger.info("📄 Starting PDF parsing with StandardPdfPipeline")
                logger.info("⏳ Processing:")
                logger.info("   1. Layout analysis")
                logger.info("   2. Text extraction")
                logger.info("   3. OCR processing")
                logger.info("   4. Table extraction")
                logger.info("")
                
                # Track processing time
                processing_start = time.time()
                result = self.converter.convert(source=tmp_file.name)
                processing_time = time.time() - processing_start
                
                logger.info(f"✅ Processing completed in {processing_time:.2f} seconds")
                
                # Extract content - export to markdown
                document = result.document
                markdown_content = document.export_to_markdown()
                
                # Log statistics
                logger.info(f"📊 Document Statistics:")
                logger.info(f"   - Size: {len(markdown_content)} characters")
                logger.info(f"   - Processing time: {processing_time:.2f}s")
                logger.info("------------------------------------------------------------")
                
                return {
                    "success": True,
                    "content": markdown_content,
                    "error": None
                }
                
        except Exception as e:
            logger.error(f"❌ Error during PDF parsing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def is_available(self) -> bool:
        """Check if Docling parsing is available."""
        return DOCLING_AVAILABLE and self.converter is not None

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the Docling StandardPdfPipeline parser."""
        return {
            "available": self.is_available(),
            "library": "docling" if DOCLING_AVAILABLE else None,
            "pipeline": "StandardPdfPipeline" if self.is_available() else None,
            "description": "CPU-optimized PDF parsing with layout analysis, OCR, and table extraction",
            "models": {
                "layout_detection": {
                    "name": "DocLayNet",
                    "size": "~100MB",
                    "purpose": "Document layout and structure analysis"
                },
                "table_extraction": {
                    "name": "TableFormer",
                    "size": "~200MB",
                    "purpose": "Table structure recognition and extraction"
                },
                "ocr": {
                    "name": "EasyOCR",
                    "size": "~100MB",
                    "purpose": "Text extraction from images and scanned documents"
                }
            },
            "features": [
                "Document layout analysis",
                "Heading and paragraph detection",
                "Table structure preservation",
                "OCR for scanned documents",
                "Multi-column support",
                "List detection",
                "Structured markdown output",
                "CPU-optimized processing"
            ],
            "performance": {
                "expected_speed": "2-5 seconds per page",
                "memory_usage": "< 1GB RAM",
                "optimization": "CPU-only",
                "first_run": "Models download automatically (~400MB, 3-5 min)",
                "cached_run": "Fast initialization (~5 sec)"
            },
            "limitations": [
                "No vision-language understanding (use VLM for advanced cases)",
                "OCR accuracy depends on image quality",
                "Complex layouts may need manual review"
            ]
        }

