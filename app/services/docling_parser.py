"""
Docling service for PDF parsing using ThreadedPdfPipeline.

The ThreadedPdfPipeline includes:
- Layout Detection: DocLayNet-based model for document structure understanding
- Table Extraction: TableFormer model for accurate table structure recognition
- OCR: EasyOCR for text extraction from images and scanned documents
- Text Extraction: Direct text layer extraction when available
- Batching: Process multiple pages/operations in parallel for better performance
- Backpressure Control: Queue management to prevent memory overflow

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
    from docling.pipeline.standard_pdf_pipeline import ThreadedPdfPipeline
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    ThreadedPdfPipeline = None
    PdfPipelineOptions = None
    AcceleratorOptions = None
    AcceleratorDevice = None

logger = logging.getLogger(__name__)


class DoclingParser:
 
    
    def __init__(self):
        """Initialize the Docling parser."""
        self.mode = "threaded-pipeline"
        # No longer initialize converter at startup - will be created per request
    
    def _create_pipeline_options(self, user_options: Optional[Dict[str, Any]] = None) -> PdfPipelineOptions:

        # Default options
        defaults = {
            "enable_ocr": False,
            "ocr_languages": ["en"],
            "table_mode": "fast",
            "do_table_structure": True,
            "do_cell_matching": True,
            "num_threads": 8,
            "accelerator_device": "cpu",
            # Enrichment options
            "do_code_enrichment": False,
            "do_formula_enrichment": False,
            "do_picture_classification": False,
            "do_picture_description": False,
            # ThreadedPdfPipeline batching options
            "layout_batch_size": 4,
            "ocr_batch_size": 4,
            "table_batch_size": 4,
            "queue_max_size": 100,
            "batch_timeout_seconds": 2.0
        }
        
        # Merge with user options
        options = {**defaults, **(user_options or {})}
        
        # Create PdfPipelineOptions (works with ThreadedPdfPipeline)
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
        
        # Enrichment options (advanced features)
        pipeline_options.do_code_enrichment = options["do_code_enrichment"]
        pipeline_options.do_formula_enrichment = options["do_formula_enrichment"]
        pipeline_options.do_picture_classification = options["do_picture_classification"]
        
        # Note: do_picture_description might require additional VLM setup
        # Only enable if explicitly requested and VLM is available
        if options["do_picture_description"]:
            # This feature requires Vision-Language Model (VLM) support
            # May need additional configuration or models
            pipeline_options.do_picture_description = True
        
        # ThreadedPdfPipeline batching configuration
        pipeline_options.layout_batch_size = options["layout_batch_size"]
        pipeline_options.ocr_batch_size = options["ocr_batch_size"]
        pipeline_options.table_batch_size = options["table_batch_size"]
        pipeline_options.queue_max_size = options["queue_max_size"]
        pipeline_options.batch_timeout_seconds = options["batch_timeout_seconds"]
        
        return pipeline_options
    
    def _create_converter(self, pipeline_options: PdfPipelineOptions) -> DocumentConverter:

        if not DOCLING_AVAILABLE:
            raise RuntimeError("Docling is not available")
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedPdfPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        
        return converter
    
    def parse_pdf(self, pdf_content: bytes, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            logger.info("📄 Starting PDF parsing with ThreadedPdfPipeline")
            logger.info("⚙️  Configuration:")
            logger.info(f"   - OCR: {'Enabled' if pipeline_options.do_ocr else 'Disabled'}")
            if pipeline_options.do_ocr:
                logger.info(f"   - OCR Languages: {pipeline_options.ocr_options.lang}")
                logger.info(f"   - OCR Batch Size: {pipeline_options.ocr_batch_size}")
            logger.info(f"   - Table Extraction: {'Enabled' if pipeline_options.do_table_structure else 'Disabled'}")
            if pipeline_options.do_table_structure:
                logger.info(f"   - Table Mode: {pipeline_options.table_structure_options.mode}")
                logger.info(f"   - Table Batch Size: {pipeline_options.table_batch_size}")
            logger.info(f"   - Layout Batch Size: {pipeline_options.layout_batch_size}")
            logger.info(f"   - Queue Max Size: {pipeline_options.queue_max_size}")
            logger.info(f"   - Batch Timeout: {pipeline_options.batch_timeout_seconds}s")
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
        """Get information about the Docling ThreadedPdfPipeline parser."""
        return {
            "available": self.is_available(),
            "library": "docling" if DOCLING_AVAILABLE else None,
            "pipeline": "ThreadedPdfPipeline" if self.is_available() else None,
            "description": "High-performance PDF parsing with batching, layout analysis, optional OCR, and table extraction",
            "configuration": {
                "note": "Pipeline options can be configured per request via API",
                "options": {
                    "enable_ocr": "Enable/disable OCR processing (default: False)",
                    "ocr_languages": "OCR language codes (default: ['en'])",
                    "table_mode": "Table extraction mode: 'fast' or 'accurate' (default: 'fast')",
                    "do_table_structure": "Enable table structure extraction (default: True)",
                    "do_cell_matching": "Enable cell matching for better table accuracy (default: True)",
                    "num_threads": "Number of processing threads, 1-144 (default: 8, optimized for 72-core Xeon)",
                    "accelerator_device": "Device selection: 'cpu', 'cuda', 'auto' (default: 'cpu')",
                    "layout_batch_size": "Batch size for layout detection, 1-32 (default: 4)",
                    "ocr_batch_size": "Batch size for OCR processing, 1-32 (default: 4)",
                    "table_batch_size": "Batch size for table extraction, 1-32 (default: 4)",
                    "queue_max_size": "Maximum queue size for backpressure control, 10-1000 (default: 100)",
                    "batch_timeout_seconds": "Batch processing timeout in seconds, 0.1-30.0 (default: 2.0)"
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
                "Batched processing for layout, OCR, and tables",
                "Backpressure control with queue management",
                "Per-request pipeline configuration"
            ],
            "performance": {
                "expected_speed": "2-5 seconds per page (varies with configuration)",
                "memory_usage": "< 1GB RAM (varies with OCR, threads, and batch sizes)",
                "threading": "Optimized for 72-core Xeon 6960P (configurable 1-144 threads)",
                "batching": "Process multiple pages/operations in parallel for better throughput",
                "backpressure": "Queue management prevents memory overflow on large documents",
                "first_run": "Models download automatically (~400MB, 3-5 min)",
                "cached_run": "Fast initialization (<5 sec)"
            },
            "limitations": [
                "No vision-language understanding (use VLM for advanced cases)",
                "OCR accuracy depends on image quality",
                "Complex layouts may need manual review"
            ]
        }

