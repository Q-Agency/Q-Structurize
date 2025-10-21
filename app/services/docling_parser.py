"""
Docling service for PDF parsing using StandardPdfPipeline with ThreadedPdfPipelineOptions.
Pre-initializes the converter once for optimal performance.
"""

# ============================================================================
# Threading Configuration
# Note: OMP/MKL/OPENBLAS thread counts are set in Dockerfile ENV
# We only set PyTorch-specific threading here as it's not always respected
# from environment variables.
# ============================================================================
import os
import tempfile
import logging
import time
import sys
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)
# Only attach once to avoid duplicate logs on reload
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    h.setLevel(level)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    h.setFormatter(fmt)
    logger.addHandler(h)
    logger.setLevel(level)
    # prevent double propagation into root if uvicorn/root also handles
    logger.propagate = False


# Configure PyTorch threading and GPU optimizations (must be done before importing docling)
# PyTorch doesn't always respect TORCH_NUM_THREADS env var, so we set it explicitly
try:
    import torch
    num_threads = int(os.environ.get('OMP_NUM_THREADS', '64'))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 10))
    logger.info("✅ PyTorch threading configured: %s intra-op, %s inter-op threads",
                torch.get_num_threads(), torch.get_num_interop_threads())

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
            logger.info("✅ GPU optimizations enabled: TF32 matmul and cuDNN")
        except Exception:
            logger.info("✅ GPU optimizations enabled: TF32 matmul and cuDNN (float32 precision not available)")
except ImportError:
    logger.warning("⚠️  PyTorch not available, skipping torch threading configuration")
except Exception as e:
    logger.warning("⚠️  Could not configure PyTorch threading: %s", e)

# Try to import docling
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, TableFormerMode
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.settings import settings
    from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
    settings.perf.page_batch_size = int(os.environ.get('DOCLING_PAGE_BATCH_SIZE', '12'))
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    settings = None
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    StandardPdfPipeline = None
    ThreadedPdfPipelineOptions = None
    AcceleratorOptions = None
    AcceleratorDevice = None
    logger.error(f"Failed to import docling: {e}")


class DoclingParser:
    """
    High-performance PDF parser using Docling with pre-initialized converter.
    
    CRITICAL: The converter should be initialized ONCE at startup and reused
    for all requests to avoid expensive model reloading.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Docling parser with pre-loaded models.
        
        Args:
            config: Default configuration for the parser
        """
        self.mode = "threaded-pipeline-optimized"
        
        if not DOCLING_AVAILABLE:
            logger.error("Docling is not available")
            self.converter = None
            return
        
        # Read configuration from environment variables (set in Dockerfile)
        # This allows changing all settings by editing Dockerfile and rebuilding
        self.default_config = config or {
            # OCR Configuration
            "enable_ocr": os.environ.get('DOCLING_ENABLE_OCR', 'false').lower() == 'true',
            "ocr_languages": [lang.strip() for lang in os.environ.get('DOCLING_OCR_LANGUAGES', 'en').split(',')],
            
            # Table Extraction Configuration
            "do_table_structure": os.environ.get('DOCLING_DO_TABLE_STRUCTURE', 'false').lower() == 'true',
            "table_mode": os.environ.get('DOCLING_TABLE_MODE', 'fast'),
            "do_cell_matching": os.environ.get('DOCLING_DO_CELL_MATCHING', 'false').lower() == 'true',
            
            # Enrichment Options
            "do_code_enrichment": os.environ.get('DOCLING_DO_CODE_ENRICHMENT', 'false').lower() == 'true',
            "do_formula_enrichment": os.environ.get('DOCLING_DO_FORMULA_ENRICHMENT', 'false').lower() == 'true',
            "do_picture_classification": os.environ.get('DOCLING_DO_PICTURE_CLASSIFICATION', 'false').lower() == 'true',
            "do_picture_description": os.environ.get('DOCLING_DO_PICTURE_DESCRIPTION', 'false').lower() == 'true',
            
            # Threading and Acceleration
            "num_threads": int(os.environ.get('OMP_NUM_THREADS', '100')),
            "accelerator_device": os.environ.get('DOCLING_ACCELERATOR_DEVICE', 'cpu'),
            
            # Batching Configuration (ThreadedPdfPipelineOptions)
            "layout_batch_size": int(os.environ.get('DOCLING_LAYOUT_BATCH_SIZE', '32')),
            "ocr_batch_size": int(os.environ.get('DOCLING_OCR_BATCH_SIZE', '32')),
            "table_batch_size": int(os.environ.get('DOCLING_TABLE_BATCH_SIZE', '32')),
            "queue_max_size": int(os.environ.get('DOCLING_QUEUE_MAX_SIZE', '1000')),
            "batch_timeout_seconds": float(os.environ.get('DOCLING_BATCH_TIMEOUT', '0.5'))
        }
        
        # Enable Docling's built-in pipeline profiling for detailed timing
        if settings:
            settings.debug.profile_pipeline_timings = True
            logger.debug("🔍 Docling pipeline profiling enabled")
        
        # Create pipeline options with default configuration
        pipeline_options = self._create_pipeline_options(self.default_config)
        
        # CRITICAL: Initialize converter ONCE at startup
        logger.info("🚀 Initializing Docling DocumentConverter (ONE-TIME SETUP)")
        logger.info(f"⚙️  Threads: {pipeline_options.accelerator_options.num_threads}, Device: {pipeline_options.accelerator_options.device}")
        table_status = 'Enabled' if pipeline_options.do_table_structure else 'Disabled'
        logger.info(f"📝 Tables: {table_status}, OCR: {'Enabled' if pipeline_options.do_ocr else 'Disabled'}")
        
        init_start = time.time()
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=ThreadedStandardPdfPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        
        # Pre-initialize the pipeline (loads models into memory)
        self.converter.initialize_pipeline(InputFormat.PDF)
        
        init_time = time.time() - init_start
        logger.info(f"✅ Converter initialized in {init_time:.2f}s (models cached in memory)")
    
    def _create_pipeline_options(self, user_options: Dict[str, Any]) -> ThreadedPdfPipelineOptions:
        """
        Create ThreadedPdfPipelineOptions from configuration.
        
        Note: This is only used during initialization. Runtime changes to
        thread count or batch sizes require re-initialization of the converter.
        """
        if not DOCLING_AVAILABLE:
            raise RuntimeError("Docling is not available")
        
        # Create ThreadedPdfPipelineOptions for batched processing
        pipeline_options = ThreadedPdfPipelineOptions()
        
        # OCR configuration
        pipeline_options.do_ocr = user_options.get("enable_ocr", False)
        if pipeline_options.do_ocr:
            pipeline_options.ocr_options.lang = user_options.get("ocr_languages", ["en"])
        
        # Table extraction configuration
        pipeline_options.do_table_structure = user_options.get("do_table_structure", False)
        if pipeline_options.do_table_structure:
            table_mode = user_options.get("table_mode", "fast")
            if table_mode == "accurate":
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            else:
                pipeline_options.table_structure_options.mode = TableFormerMode.FAST
            pipeline_options.table_structure_options.do_cell_matching = user_options.get("do_cell_matching", False)
        
        # Accelerator configuration
        device_str = user_options.get("accelerator_device", "cpu")
        device_map = {
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "auto": AcceleratorDevice.AUTO
        }
        device = device_map.get(device_str, AcceleratorDevice.CPU)
        
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=user_options.get("num_threads", 120),
            device=device
        )
        
        # Enrichment options
        pipeline_options.do_code_enrichment = user_options.get("do_code_enrichment", False)
        pipeline_options.do_formula_enrichment = user_options.get("do_formula_enrichment", False)
        pipeline_options.do_picture_classification = user_options.get("do_picture_classification", False)
        pipeline_options.do_picture_description = user_options.get("do_picture_description", False)
        
        # ThreadedPdfPipeline batching configuration
        pipeline_options.layout_batch_size = user_options.get("layout_batch_size", 32)
        pipeline_options.ocr_batch_size = user_options.get("ocr_batch_size", 32)
        pipeline_options.table_batch_size = user_options.get("table_batch_size", 32)
        pipeline_options.queue_max_size = user_options.get("queue_max_size", 1000)
        pipeline_options.batch_timeout_seconds = user_options.get("batch_timeout_seconds", 0.5)
        
        return pipeline_options
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Parse PDF content using the pre-initialized converter.
        
        Configuration is set at container startup. To change settings, update 
        Dockerfile ENV variables and rebuild (takes ~10 seconds with cache).
        """
        if not DOCLING_AVAILABLE or self.converter is None:
            return {
                "success": False,
                "error": "Docling is not available",
                "content": None
            }
        
        try:
            # Create temporary file for PDF content
            file_write_start = time.time()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                file_write_time = time.time() - file_write_start
                
                # Parse the PDF using pre-initialized converter
                logger.info(f"📄 Processing PDF ({len(pdf_content):,} bytes)")
                processing_start = time.time()
                
                # CRITICAL: Reuse the pre-initialized converter
                result = self.converter.convert(source=tmp_file.name)
                conversion_time = time.time() - processing_start
                
                # Extract content - export to markdown
                export_start = time.time()
                document = result.document
                markdown_content = document.export_to_markdown()
                export_time = time.time() - export_start
                
                total_time = time.time() - processing_start
                
                # Log summary
                logger.info(f"✅ Parsed in {total_time:.2f}s: {len(markdown_content):,} chars ({len(markdown_content)/total_time:.0f} chars/sec)")
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass
                
                return {
                    "success": True,
                    "content": markdown_content,
                    "error": None,
                    "processing_time": total_time,
                    "timings": {
                        "file_write": file_write_time,
                        "conversion": conversion_time,
                        "markdown_export": export_time,
                        "total": total_time
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error during PDF parsing: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def parse_pdf_to_document(self, pdf_content: bytes):
        """
        Parse PDF content and return DoclingDocument object for further processing.
        
        This method is similar to parse_pdf() but returns the DoclingDocument object
        instead of markdown string. This is useful for chunking and other document
        processing workflows that need access to the structured document.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            DoclingDocument object if successful, None otherwise
            
        Raises:
            RuntimeError: If Docling is not available or conversion fails
        """
        if not DOCLING_AVAILABLE or self.converter is None:
            raise RuntimeError("Docling is not available")
        
        try:
            # Create temporary file for PDF content
            parse_start = time.time()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                tmp_path = tmp_file.name
                
                logger.info(f"📄 Parsing PDF to document ({len(pdf_content):,} bytes)")
                
                # Convert using pre-initialized converter
                conversion_start = time.time()
                result = self.converter.convert(source=tmp_path)
                conversion_time = time.time() - conversion_start
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                
                total_time = time.time() - parse_start
                logger.info(f"✅ Document parsed in {total_time:.2f}s (conversion: {conversion_time:.2f}s)")
                
                # Return the document object
                return result.document
                
        except Exception as e:
            logger.error(f"❌ Error parsing PDF to document: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to parse PDF: {str(e)}") from e
    
    def is_available(self) -> bool:
        """Check if Docling parsing is available."""
        return DOCLING_AVAILABLE and self.converter is not None

    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the Docling parser."""
        return {
            "available": self.is_available(),
            "library": "docling" if DOCLING_AVAILABLE else None,
            "pipeline": "StandardPdfPipeline with ThreadedPdfPipelineOptions (pre-initialized)" if self.is_available() else None,
            "description": "High-performance PDF parsing with pre-initialized models, batching, and multi-threading",
            "performance_mode": "optimized_with_preinitialization",
            "configuration": self.default_config if hasattr(self, 'default_config') else {},
            "performance": {
                "initialization": "One-time at startup (~5-30 seconds depending on cache)",
                "per_request": "Fast (<1 second per page with warm cache)",
                "threading": f"{self.default_config.get('num_threads', 120)} threads configured",
                "batching": "Enabled with aggressive batch sizes"
            }
        }