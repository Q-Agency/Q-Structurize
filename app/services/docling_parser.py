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
import logging
import time
import sys
from io import BytesIO
from typing import Optional, Dict, Any

from pydantic import ValidationError

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
    from docling.datamodel.base_models import InputFormat, DocumentStream, ConversionStatus
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, TableFormerMode
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.settings import settings
    from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
    from docling.utils.model_downloader import download_models
    from docling.datamodel.layout_model_specs import (
        DOCLING_LAYOUT_HERON,
        DOCLING_LAYOUT_HERON_101,
        DOCLING_LAYOUT_EGRET_MEDIUM,
        DOCLING_LAYOUT_EGRET_LARGE,
        DOCLING_LAYOUT_EGRET_XLARGE,
    )
    from docling.models.layout_model import LayoutModel
    # Configure global performance settings
    if settings:
        settings.perf.page_batch_size = int(os.environ.get('DOCLING_PAGE_BATCH_SIZE', '12'))
        settings.perf.doc_batch_size = int(os.environ.get('DOCLING_DOC_BATCH_SIZE', '1'))
        settings.perf.doc_batch_concurrency = int(os.environ.get('DOCLING_DOC_BATCH_CONCURRENCY', '1'))
        settings.perf.elements_batch_size = int(os.environ.get('DOCLING_ELEMENTS_BATCH_SIZE', '16'))
    DOCLING_AVAILABLE = True
except ImportError as e:
    DOCLING_AVAILABLE = False
    settings = None
    InputFormat = None
    DocumentStream = None
    DocumentConverter = None
    PdfFormatOption = None
    StandardPdfPipeline = None
    ThreadedPdfPipelineOptions = None
    AcceleratorOptions = None
    AcceleratorDevice = None
    download_models = None
    ConversionStatus = None
    DOCLING_LAYOUT_HERON = None
    DOCLING_LAYOUT_HERON_101 = None
    DOCLING_LAYOUT_EGRET_MEDIUM = None
    DOCLING_LAYOUT_EGRET_LARGE = None
    DOCLING_LAYOUT_EGRET_XLARGE = None
    LayoutModel = None
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
            "table_mode": os.environ.get('DOCLING_TABLE_MODE', 'accurate'),
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
            "batch_polling_interval_seconds": float(os.environ.get('DOCLING_BATCH_POLLING_INTERVAL', '0.5')),
            
            # Document Safety Limits
            "max_num_pages": int(os.environ.get('DOCLING_MAX_NUM_PAGES', '1000')),
            "max_file_size": int(os.environ.get('DOCLING_MAX_FILE_SIZE', '104857600')),  # 100MB default
            "page_range": os.environ.get('DOCLING_PAGE_RANGE', ''),  # Format: "1-100" or empty for all
            "document_timeout": float(os.environ.get('DOCLING_DOCUMENT_TIMEOUT', '300')),  # 5 minutes default
            
            # Layout Model Selection
            "layout_model": os.environ.get('DOCLING_LAYOUT_MODEL', 'heron').lower(),
            
            # Image Generation Options
            "generate_page_images": os.environ.get('DOCLING_GENERATE_PAGE_IMAGES', 'false').lower() == 'true',
            "generate_picture_images": os.environ.get('DOCLING_GENERATE_PICTURE_IMAGES', 'false').lower() == 'true',
            "images_scale": float(os.environ.get('DOCLING_IMAGES_SCALE', '1.0')),
            
            # VLM Integration
            "force_backend_text": os.environ.get('DOCLING_FORCE_BACKEND_TEXT', 'false').lower() == 'true',
        }
        
        # Enable Docling's built-in pipeline profiling for detailed timing (configurable)
        if settings:
            enable_profiling = os.environ.get('DOCLING_ENABLE_PROFILING', 'false').lower() == 'true'
            settings.debug.profile_pipeline_timings = enable_profiling
            if enable_profiling:
                logger.debug("🔍 Docling pipeline profiling enabled")
            else:
                logger.debug("🔍 Docling pipeline profiling disabled (set DOCLING_ENABLE_PROFILING=true to enable)")
        
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
        
        # Try to pre-initialize the pipeline (loads models into memory)
        # If models aren't available, initialization will happen lazily on first use
        try:
            self.converter.initialize_pipeline(InputFormat.PDF)
            init_time = time.time() - init_start
            logger.info(f"✅ Converter initialized in {init_time:.2f}s (models cached in memory)")
            self._pipeline_initialized = True
        except (FileNotFoundError, OSError) as e:
            # Models not available yet - will initialize lazily on first use
            logger.warning(f"⚠️  Models not available at startup: {str(e)}")
            logger.info("📦 Pipeline will initialize lazily on first PDF conversion (models will be downloaded if needed)")
            self._pipeline_initialized = False
        except Exception as e:
            # Other errors - log but don't fail startup
            logger.warning(f"⚠️  Pipeline initialization failed: {str(e)}")
            logger.info("📦 Pipeline will initialize lazily on first PDF conversion")
            self._pipeline_initialized = False
    
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
        num_threads = user_options.get("num_threads", 120)
        try:
            accelerator_options = AcceleratorOptions(
                num_threads=num_threads,
                device=device_str
            )
        except ValidationError as exc:
            logger.warning(
                "Invalid accelerator configuration '%s': %s. Falling back to CPU.",
                device_str,
                exc,
            )
            accelerator_options = AcceleratorOptions(
                num_threads=num_threads,
                device=AcceleratorDevice.CPU,
            )
        
        pipeline_options.accelerator_options = accelerator_options
        
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
        pipeline_options.batch_polling_interval_seconds = user_options.get("batch_polling_interval_seconds", 0.5)
        
        # Document timeout protection
        pipeline_options.document_timeout = user_options.get("document_timeout", 300)
        
        # Layout model selection
        layout_model = user_options.get("layout_model", "heron").lower()
        if DOCLING_AVAILABLE and DOCLING_LAYOUT_HERON:
            if layout_model == "heron_101":
                pipeline_options.layout_options.model_spec = DOCLING_LAYOUT_HERON_101
            elif layout_model == "egret_medium":
                pipeline_options.layout_options.model_spec = DOCLING_LAYOUT_EGRET_MEDIUM
            elif layout_model == "egret_large":
                pipeline_options.layout_options.model_spec = DOCLING_LAYOUT_EGRET_LARGE
            elif layout_model == "egret_xlarge":
                pipeline_options.layout_options.model_spec = DOCLING_LAYOUT_EGRET_XLARGE
            # Default is already DOCLING_LAYOUT_HERON
        
        # Image generation options
        pipeline_options.generate_page_images = user_options.get("generate_page_images", False)
        pipeline_options.generate_picture_images = user_options.get("generate_picture_images", False)
        pipeline_options.images_scale = float(user_options.get("images_scale", 1.0))
        
        # VLM integration option
        pipeline_options.force_backend_text = user_options.get("force_backend_text", False)
        
        return pipeline_options
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Parse PDF content using the pre-initialized converter.
        
        Uses DocumentStream to avoid temporary file overhead.
        Configuration is set at container startup. To change settings, update 
        Dockerfile ENV variables and rebuild (takes ~10 seconds with cache).
        """
        if not DOCLING_AVAILABLE or self.converter is None:
            return {
                "success": False,
                "error": "Docling is not available",
                "content": None
            }
        
        # Ensure pipeline is initialized (lazy initialization if models weren't ready at startup)
        if not getattr(self, '_pipeline_initialized', True):
            try:
                logger.info("🔄 Initializing pipeline on first use...")
                self.converter.initialize_pipeline(InputFormat.PDF)
                self._pipeline_initialized = True
                logger.info("✅ Pipeline initialized successfully")
            except (FileNotFoundError, OSError) as e:
                # Models missing - try to download them
                error_msg = str(e)
                if "safetensors" in error_msg.lower() or "model" in error_msg.lower():
                    logger.warning(f"⚠️  Models missing: {error_msg}")
                    logger.info("📥 Attempting to download required models...")
                    try:
                        if download_models is not None and LayoutModel is not None:
                            # Download the specific layout model that was selected
                            layout_model = self.default_config.get("layout_model", "heron").lower()
                            layout_model_config = DOCLING_LAYOUT_HERON  # default
                            if layout_model == "heron_101" and DOCLING_LAYOUT_HERON_101:
                                layout_model_config = DOCLING_LAYOUT_HERON_101
                            elif layout_model == "egret_medium" and DOCLING_LAYOUT_EGRET_MEDIUM:
                                layout_model_config = DOCLING_LAYOUT_EGRET_MEDIUM
                            elif layout_model == "egret_large" and DOCLING_LAYOUT_EGRET_LARGE:
                                layout_model_config = DOCLING_LAYOUT_EGRET_LARGE
                            elif layout_model == "egret_xlarge" and DOCLING_LAYOUT_EGRET_XLARGE:
                                layout_model_config = DOCLING_LAYOUT_EGRET_XLARGE
                            
                            # Download the specific layout model
                            if settings:
                                output_dir = settings.cache_dir / "models"
                                logger.info(f"📥 Downloading layout model: {layout_model_config.name}")
                                LayoutModel.download_models(
                                    local_dir=output_dir / layout_model_config.model_repo_folder,
                                    force=False,
                                    progress=True,
                                    layout_model_config=layout_model_config,
                                )
                            
                            # Download other models if needed
                            download_models(
                                output_dir=None,  # Uses settings.cache_dir / "models"
                                force=False,
                                progress=True,
                                with_layout=False,  # Already downloaded above
                                with_tableformer=self.default_config.get("do_table_structure", False),
                                with_code_formula=self.default_config.get("do_code_enrichment", False),
                                with_picture_classifier=self.default_config.get("do_picture_classification", False),
                                with_rapidocr=self.default_config.get("enable_ocr", False),
                                with_easyocr=False,
                            )
                            logger.info("✅ Models downloaded, retrying pipeline initialization...")
                            self.converter.initialize_pipeline(InputFormat.PDF)
                            self._pipeline_initialized = True
                            logger.info("✅ Pipeline initialized successfully after model download")
                        else:
                            raise RuntimeError("Model downloader not available")
                    except Exception as download_error:
                        logger.error(f"❌ Failed to download models: {str(download_error)}")
                        return {
                            "success": False,
                            "error": f"Models missing and download failed: {str(download_error)}. Please run 'docling-tools models download' or ensure models are available.",
                            "content": None
                        }
                else:
                    raise
            except Exception as e:
                logger.error(f"❌ Failed to initialize pipeline: {str(e)}")
                return {
                    "success": False,
                    "error": f"Pipeline initialization failed: {str(e)}",
                    "content": None
                }
        
        try:
            # Create DocumentStream from bytes (no temp file needed)
            processing_start = time.time()
            doc_stream = DocumentStream(name="document.pdf", stream=BytesIO(pdf_content))
            
            # Parse the PDF using pre-initialized converter
            logger.info(f"📄 Processing PDF ({len(pdf_content):,} bytes)")
            
            # Parse page range if specified (format: "1-100" or empty for all)
            page_range = None
            page_range_str = self.default_config.get("page_range", "")
            if page_range_str:
                try:
                    parts = page_range_str.split("-")
                    if len(parts) == 2:
                        start = int(parts[0].strip())
                        end = int(parts[1].strip())
                        page_range = (start, end)
                        logger.info(f"📄 Processing pages {start}-{end}")
                except (ValueError, IndexError):
                    logger.warning(f"⚠️  Invalid page_range format '{page_range_str}', processing all pages")
            
            # CRITICAL: Reuse the pre-initialized converter with safety limits
            result = self.converter.convert(
                source=doc_stream,
                max_num_pages=self.default_config.get("max_num_pages", 1000),
                max_file_size=self.default_config.get("max_file_size", 104857600),
                page_range=page_range if page_range else (1, sys.maxsize)
            )
            conversion_time = time.time() - processing_start
            
            # Check conversion status and handle warnings/errors
            status = result.status
            warnings = []
            errors = []
            
            if hasattr(result, 'errors') and result.errors:
                for error_item in result.errors:
                    error_msg = f"{error_item.component_type}: {error_item.error_message}"
                    if status == ConversionStatus.PARTIAL_SUCCESS:
                        warnings.append(error_msg)
                        logger.warning(f"⚠️  Conversion warning: {error_msg}")
                    else:
                        errors.append(error_msg)
                        logger.error(f"❌ Conversion error: {error_msg}")
            
            # Extract content - export to markdown only if successful or partially successful
            export_start = time.time()
            markdown_content = None
            if status in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                document = result.document
                markdown_content = document.export_to_markdown()
            export_time = time.time() - export_start
            
            total_time = time.time() - processing_start
            
            # Log summary with status
            if status == ConversionStatus.SUCCESS:
                logger.info(f"✅ Parsed in {total_time:.2f}s: {len(markdown_content):,} chars ({len(markdown_content)/total_time:.0f} chars/sec)")
            elif status == ConversionStatus.PARTIAL_SUCCESS:
                logger.warning(f"⚠️  Partially parsed in {total_time:.2f}s: {len(markdown_content):,} chars (warnings: {len(warnings)})")
            else:
                logger.error(f"❌ Conversion failed in {total_time:.2f}s")
            
            return {
                "success": status == ConversionStatus.SUCCESS,
                "status": status.value if status else "unknown",
                "content": markdown_content,
                "error": errors[0] if errors else None,
                "warnings": warnings,
                "errors": errors,
                "processing_time": total_time,
                "timings": {
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
        
        Uses DocumentStream to avoid temporary file overhead.
        This method is similar to parse_pdf() but returns the DoclingDocument object
        instead of markdown string. This is useful for chunking and other document
        processing workflows that need access to the structured document.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            DoclingDocument object if successful
            
        Raises:
            RuntimeError: If Docling is not available or conversion fails
        """
        if not DOCLING_AVAILABLE or self.converter is None:
            raise RuntimeError("Docling is not available")
        
        # Ensure pipeline is initialized (lazy initialization if models weren't ready at startup)
        if not getattr(self, '_pipeline_initialized', True):
            try:
                logger.info("🔄 Initializing pipeline on first use...")
                self.converter.initialize_pipeline(InputFormat.PDF)
                self._pipeline_initialized = True
                logger.info("✅ Pipeline initialized successfully")
            except (FileNotFoundError, OSError) as e:
                # Models missing - try to download them
                error_msg = str(e)
                if "safetensors" in error_msg.lower() or "model" in error_msg.lower():
                    logger.warning(f"⚠️  Models missing: {error_msg}")
                    logger.info("📥 Attempting to download required models...")
                    try:
                        if download_models is not None and LayoutModel is not None:
                            # Download the specific layout model that was selected
                            layout_model = self.default_config.get("layout_model", "heron").lower()
                            layout_model_config = DOCLING_LAYOUT_HERON  # default
                            if layout_model == "heron_101" and DOCLING_LAYOUT_HERON_101:
                                layout_model_config = DOCLING_LAYOUT_HERON_101
                            elif layout_model == "egret_medium" and DOCLING_LAYOUT_EGRET_MEDIUM:
                                layout_model_config = DOCLING_LAYOUT_EGRET_MEDIUM
                            elif layout_model == "egret_large" and DOCLING_LAYOUT_EGRET_LARGE:
                                layout_model_config = DOCLING_LAYOUT_EGRET_LARGE
                            elif layout_model == "egret_xlarge" and DOCLING_LAYOUT_EGRET_XLARGE:
                                layout_model_config = DOCLING_LAYOUT_EGRET_XLARGE
                            
                            # Download the specific layout model
                            if settings:
                                output_dir = settings.cache_dir / "models"
                                logger.info(f"📥 Downloading layout model: {layout_model_config.name}")
                                LayoutModel.download_models(
                                    local_dir=output_dir / layout_model_config.model_repo_folder,
                                    force=False,
                                    progress=True,
                                    layout_model_config=layout_model_config,
                                )
                            
                            # Download other models if needed
                            download_models(
                                output_dir=None,  # Uses settings.cache_dir / "models"
                                force=False,
                                progress=True,
                                with_layout=False,  # Already downloaded above
                                with_tableformer=self.default_config.get("do_table_structure", False),
                                with_code_formula=self.default_config.get("do_code_enrichment", False),
                                with_picture_classifier=self.default_config.get("do_picture_classification", False),
                                with_rapidocr=self.default_config.get("enable_ocr", False),
                                with_easyocr=False,
                            )
                            logger.info("✅ Models downloaded, retrying pipeline initialization...")
                            self.converter.initialize_pipeline(InputFormat.PDF)
                            self._pipeline_initialized = True
                            logger.info("✅ Pipeline initialized successfully after model download")
                        else:
                            raise RuntimeError("Model downloader not available")
                    except Exception as download_error:
                        logger.error(f"❌ Failed to download models: {str(download_error)}")
                        raise RuntimeError(f"Models missing and download failed: {str(download_error)}. Please run 'docling-tools models download' or ensure models are available.") from download_error
                else:
                    raise
            except Exception as e:
                logger.error(f"❌ Failed to initialize pipeline: {str(e)}")
                raise RuntimeError(f"Pipeline initialization failed: {str(e)}") from e
        
        try:
            # Create DocumentStream from bytes (no temp file needed)
            parse_start = time.time()
            doc_stream = DocumentStream(name="document.pdf", stream=BytesIO(pdf_content))
            
            logger.info(f"📄 Parsing PDF to document ({len(pdf_content):,} bytes)")
            
            # Parse page range if specified (format: "1-100" or empty for all)
            page_range = None
            page_range_str = self.default_config.get("page_range", "")
            if page_range_str:
                try:
                    parts = page_range_str.split("-")
                    if len(parts) == 2:
                        start = int(parts[0].strip())
                        end = int(parts[1].strip())
                        page_range = (start, end)
                        logger.info(f"📄 Processing pages {start}-{end}")
                except (ValueError, IndexError):
                    logger.warning(f"⚠️  Invalid page_range format '{page_range_str}', processing all pages")
            
            # Convert using pre-initialized converter with safety limits
            conversion_start = time.time()
            result = self.converter.convert(
                source=doc_stream,
                max_num_pages=self.default_config.get("max_num_pages", 1000),
                max_file_size=self.default_config.get("max_file_size", 104857600),
                page_range=page_range if page_range else (1, sys.maxsize)
            )
            conversion_time = time.time() - conversion_start
            
            total_time = time.time() - parse_start
            
            # Check conversion status and handle errors
            status = result.status
            if status == ConversionStatus.FAILURE:
                error_msgs = []
                if hasattr(result, 'errors') and result.errors:
                    error_msgs = [f"{e.component_type}: {e.error_message}" for e in result.errors]
                error_msg = "; ".join(error_msgs) if error_msgs else "Conversion failed"
                logger.error(f"❌ Document conversion failed: {error_msg}")
                raise RuntimeError(f"Failed to parse PDF: {error_msg}")
            elif status == ConversionStatus.PARTIAL_SUCCESS:
                warnings = []
                if hasattr(result, 'errors') and result.errors:
                    warnings = [f"{e.component_type}: {e.error_message}" for e in result.errors]
                logger.warning(f"⚠️  Document partially parsed (warnings: {len(warnings)})")
                for warning in warnings:
                    logger.warning(f"⚠️  {warning}")
            
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