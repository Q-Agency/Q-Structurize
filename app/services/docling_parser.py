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
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    StandardPdfPipeline = None
    AcceleratorOptions = None
    AcceleratorDevice = None

logger = logging.getLogger(__name__)


class DoclingParser:
    """
    Service for parsing PDFs using Docling's StandardPdfPipeline.
    
    Features:
    - Layout detection and document structure analysis
    - Table extraction with structure preservation
    - OCR for scanned documents (configurable)
    - CPU-optimized processing with configurable threading
    - Dynamic pipeline configuration per request
    
    Models used (auto-downloaded on first use):
    - DocLayNet layout model (~100MB)
    - TableFormer structure model (~200MB)
    - EasyOCR language models (~100MB, loaded only if OCR enabled)
    """
    
    def __init__(self):
        """Initialize the Docling parser."""
        self.mode = "standard-pipeline"
        # No longer initialize converter at startup - will be created per request
    
    def _create_pipeline_options(self, user_options: Optional[Dict[str, Any]] = None) -> PdfPipelineOptions:
        """
        Create PdfPipelineOptions based on user configuration.
        
        Args:
            user_options: Dictionary with pipeline configuration options:
                - enable_ocr: bool (default: False)
                - ocr_languages: List[str] (default: ["en"])
                - table_mode: str "fast" or "accurate" (default: "fast")
                - do_table_structure: bool (default: True)
                - do_cell_matching: bool (default: True)
                - num_threads: int (default: 8)
                - accelerator_device: str "cpu", "cuda", or "auto" (default: "cpu")
        
        Returns:
            Configured PdfPipelineOptions instance
        """
        # Default options
        defaults = {
            "enable_ocr": False,
            "ocr_languages": ["en"],
            "table_mode": "fast",
            "do_table_structure": True,
            "do_cell_matching": True,
            "num_threads": 8,
            "accelerator_device": "cpu"
        }
        
        # Merge with user options
        options = {**defaults, **(user_options or {})}
        
        # Create pipeline options
        pipeline_options = PdfPipelineOptions()
        
        # OCR configuration
        pipeline_options.do_ocr = options["enable_ocr"]
        if options["enable_ocr"]:
            pipeline_options.ocr_options.lang = options["ocr_languages"]
        
        # Table extraction configuration
        pipeline_options.do_table_structure = options["do_table_structure"]
        if options["do_table_structure"]:
            if options["table_mode"] == "accurate":
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            else:
                pipeline_options.table_structure_options.mode = TableFormerMode.FAST
            pipeline_options.table_structure_options.do_cell_matching = options["do_cell_matching"]
        
        # Accelerator configuration
        device_map = {
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "auto": AcceleratorDevice.AUTO
        }
        device = device_map.get(options["accelerator_device"], AcceleratorDevice.CPU)
        
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=options["num_threads"],
            device=device
        )
        
        return pipeline_options
    
    def _create_converter(self, pipeline_options: PdfPipelineOptions) -> DocumentConverter:
        """
        Create a DocumentConverter with the specified pipeline options.
        
        Args:
            pipeline_options: Configured PdfPipelineOptions
            
        Returns:
            Initialized DocumentConverter instance
        """
        if not DOCLING_AVAILABLE:
            raise RuntimeError("Docling is not available")
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        
        return converter
    
    def parse_pdf(self, pdf_content: bytes, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse PDF content using Docling's StandardPdfPipeline with configurable options.
        
        The pipeline performs:
        1. Layout analysis - Identifies document structure (headings, paragraphs, tables)
        2. Text extraction - Extracts text from PDF layers
        3. OCR processing - Extracts text from images/scanned pages (if enabled)
        4. Table extraction - Preserves table structure
        5. Markdown export - Structured output with proper formatting
        
        Args:
            pdf_content: Raw PDF file content as bytes
            options: Optional dictionary with pipeline configuration:
                - enable_ocr: bool (default: False)
                - ocr_languages: List[str] (default: ["en"])
                - table_mode: str "fast" or "accurate" (default: "fast")
                - do_table_structure: bool (default: True)
                - do_cell_matching: bool (default: True)
                - num_threads: int (default: 8)
                - accelerator_device: str "cpu", "cuda", or "auto" (default: "cpu")
            
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
        
        try:
            # Create pipeline options based on user input
            pipeline_options = self._create_pipeline_options(options)
            
            # Log configuration
            logger.info("============================================================")
            logger.info("📄 Starting PDF parsing with StandardPdfPipeline")
            logger.info("⚙️  Configuration:")
            logger.info(f"   - OCR: {'Enabled' if pipeline_options.do_ocr else 'Disabled'}")
            if pipeline_options.do_ocr:
                logger.info(f"   - OCR Languages: {pipeline_options.ocr_options.lang}")
            logger.info(f"   - Table Extraction: {'Enabled' if pipeline_options.do_table_structure else 'Disabled'}")
            if pipeline_options.do_table_structure:
                logger.info(f"   - Table Mode: {pipeline_options.table_structure_options.mode}")
            logger.info(f"   - Threads: {pipeline_options.accelerator_options.num_threads}")
            logger.info(f"   - Device: {pipeline_options.accelerator_options.device}")
            logger.info("------------------------------------------------------------")
            
            # Create converter with specified options
            converter = self._create_converter(pipeline_options)
            
            # Create temporary file for PDF content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                # Parse the PDF
                logger.info("⏳ Processing document...")
                processing_start = time.time()
                result = converter.convert(source=tmp_file.name)
                processing_time = time.time() - processing_start
                
                logger.info(f"✅ Processing completed in {processing_time:.2f} seconds")
                
                # Extract content - export to markdown
                document = result.document
                markdown_content = document.export_to_markdown()
                
                # Log statistics
                logger.info(f"📊 Document Statistics:")
                logger.info(f"   - Size: {len(markdown_content)} characters")
                logger.info(f"   - Processing time: {processing_time:.2f}s")
                logger.info("============================================================")
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass
                
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
        return DOCLING_AVAILABLE

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the Docling StandardPdfPipeline parser."""
        return {
            "available": self.is_available(),
            "library": "docling" if DOCLING_AVAILABLE else None,
            "pipeline": "StandardPdfPipeline" if self.is_available() else None,
            "description": "Configurable PDF parsing with layout analysis, optional OCR, and table extraction",
            "configuration": {
                "note": "Pipeline options can be configured per request via API",
                "options": {
                    "enable_ocr": "Enable/disable OCR processing (default: False)",
                    "ocr_languages": "OCR language codes (default: ['en'])",
                    "table_mode": "Table extraction mode: 'fast' or 'accurate' (default: 'fast')",
                    "do_table_structure": "Enable table structure extraction (default: True)",
                    "do_cell_matching": "Enable cell matching for better table accuracy (default: True)",
                    "num_threads": "Number of processing threads, 1-144 (default: 8, optimized for 72-core Xeon)",
                    "accelerator_device": "Device selection: 'cpu', 'cuda', 'auto' (default: 'cpu')"
                }
            },
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
                    "purpose": "Text extraction from images and scanned documents (loaded only when enabled)"
                }
            },
            "features": [
                "Document layout analysis",
                "Heading and paragraph detection",
                "Table structure preservation",
                "Configurable OCR for scanned documents",
                "Multi-language OCR support",
                "Multi-column support",
                "List detection",
                "Structured markdown output",
                "Multi-threaded processing",
                "Per-request pipeline configuration"
            ],
            "performance": {
                "expected_speed": "2-5 seconds per page (varies with configuration)",
                "memory_usage": "< 1GB RAM (varies with OCR and threads)",
                "threading": "Optimized for 72-core Xeon 6960P (configurable 1-144 threads)",
                "first_run": "Models download automatically (~400MB, 3-5 min)",
                "cached_run": "Fast initialization (<5 sec)"
            },
            "limitations": [
                "No vision-language understanding (use VLM for advanced cases)",
                "OCR accuracy depends on image quality",
                "Complex layouts may need manual review"
            ]
        }

