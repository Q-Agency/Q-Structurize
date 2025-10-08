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
from typing import Optional, Dict, Any

# Configure PyTorch threading (must be done before importing docling)
# PyTorch doesn't always respect TORCH_NUM_THREADS env var, so we set it explicitly
try:
    import torch
    # Read thread count from Dockerfile ENV (default to 100 if not set)
    num_threads = int(os.environ.get('OMP_NUM_THREADS', '100'))
    torch.set_num_threads(num_threads)              # Intra-op parallelism
    torch.set_num_interop_threads(max(1, num_threads // 10))  # Inter-op (10% of threads)
    logging.getLogger(__name__).info(f"✅ PyTorch threading configured: {torch.get_num_threads()} intra-op, {torch.get_num_interop_threads()} inter-op threads")
except ImportError:
    logging.getLogger(__name__).warning("⚠️  PyTorch not available, skipping torch threading configuration")
except Exception as e:
    logging.getLogger(__name__).warning(f"⚠️  Could not configure PyTorch threading: {e}")

# Try to import docling
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, TableFormerMode
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    StandardPdfPipeline = None
    ThreadedPdfPipelineOptions = None
    AcceleratorOptions = None
    AcceleratorDevice = None
    import logging
    logging.error(f"Failed to import docling: {e}")

logger = logging.getLogger(__name__)


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
        
        # Set default configuration
        self.default_config = config or {
            "enable_ocr": False,
            "ocr_languages": ["en"],
            "table_mode": "fast",
            "do_table_structure": False,  # Disabled by default for speed
            "do_cell_matching": False,
            "num_threads": int(os.environ.get('OMP_NUM_THREADS', '100')),  # Use most of your 144 cores
            "accelerator_device": "cpu",
            "do_code_enrichment": False,
            "do_formula_enrichment": False,
            "do_picture_classification": False,
            "do_picture_description": False,
            "layout_batch_size": 32,
            "ocr_batch_size": 32,
            "table_batch_size": 32,
            "queue_max_size": 1000,
            "batch_timeout_seconds": 0.5
        }
        
        # Create pipeline options with default configuration
        pipeline_options = self._create_pipeline_options(self.default_config)
        
        # CRITICAL: Initialize converter ONCE at startup
        logger.info("============================================================")
        logger.info("🚀 Initializing Docling DocumentConverter (ONE-TIME SETUP)")
        logger.info("⚙️  Default Configuration:")
        logger.info(f"   - Threads: {pipeline_options.accelerator_options.num_threads}")
        logger.info(f"   - Layout Batch Size: {pipeline_options.layout_batch_size}")
        logger.info(f"   - Queue Max Size: {pipeline_options.queue_max_size}")
        logger.info(f"   - OCR: {'Enabled' if pipeline_options.do_ocr else 'Disabled'}")
        logger.info(f"   - Tables: {'Enabled' if pipeline_options.do_table_structure else 'Disabled'}")
        
        init_start = time.time()
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options,
                ),
            }
        )
        
        # Pre-initialize the pipeline (loads models into memory)
        logger.info("📦 Pre-loading models into memory...")
        self.converter.initialize_pipeline(InputFormat.PDF)
        
        init_time = time.time() - init_start
        logger.info(f"✅ Converter initialized in {init_time:.2f} seconds")
        logger.info("📝 Models are now cached in memory for fast processing")
        logger.info("============================================================")
    
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
    
    def parse_pdf(self, pdf_content: bytes, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Parse PDF content using the pre-initialized converter.
        
        IMPORTANT: Runtime options are currently ignored because the converter
        is pre-initialized. To support dynamic options, the converter would need
        to be re-initialized (expensive) or we'd need to create a pool of converters
        with different configurations.
        
        Args:
            pdf_content: Raw PDF bytes
            options: Parsing options (currently ignored - uses default config)
        
        Returns:
            Dictionary with success status, content, and error info
        """
        if not DOCLING_AVAILABLE or self.converter is None:
            return {
                "success": False,
                "error": "Docling is not available",
                "content": None
            }
        
        # Log warning if user provided options (they will be ignored)
        if options:
            logger.warning("⚠️  Runtime options are ignored. Using default configuration from initialization.")
            logger.warning("    To change configuration, restart the service with new defaults.")
        
        try:
            # Log start
            logger.info("============================================================")
            logger.info("📄 Starting PDF parsing with pre-initialized converter")
            logger.info("------------------------------------------------------------")
            
            # Create temporary file for PDF content
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                # Parse the PDF using pre-initialized converter
                logger.info("⏳ Processing document...")
                processing_start = time.time()
                
                # CRITICAL: Reuse the pre-initialized converter
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
                logger.info(f"   - Throughput: {len(markdown_content)/processing_time:.0f} chars/sec")
                logger.info("============================================================")
                
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except Exception:
                    pass
                
                return {
                    "success": True,
                    "content": markdown_content,
                    "error": None,
                    "processing_time": processing_time
                }
                
        except Exception as e:
            logger.error(f"❌ Error during PDF parsing: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
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