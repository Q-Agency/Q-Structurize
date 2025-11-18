"""
Docling service for PDF parsing with image description support.
Modular extension that wraps DoclingParser to add image description capabilities.
Only used when parse_images=true is passed to the API.
"""

import os
import logging
import time
import sys
import re
from io import BytesIO
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
    logger.propagate = False

# Configure PyTorch threading (must be done before importing docling)
try:
    import torch
    num_threads = int(os.environ.get('OMP_NUM_THREADS', '64'))
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(max(1, num_threads // 10))
    logger.info("✅ PyTorch threading configured: %s intra-op, %s inter-op threads",
                torch.get_num_threads(), torch.get_num_interop_threads())
except ImportError:
    logger.warning("⚠️  PyTorch not available, skipping torch threading configuration")
except Exception as e:
    logger.warning("⚠️  Could not configure PyTorch threading: %s", e)

# Try to import docling
try:
    from docling.datamodel.base_models import InputFormat, DocumentStream, ConversionStatus
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        ThreadedPdfPipelineOptions,
        TableFormerMode,
        PictureDescriptionVlmOptions,
        PictureDescriptionApiOptions,
        granite_picture_description,
        smolvlm_picture_description,
    )
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
    from pydantic import AnyUrl
    from docling_core.types.doc import PictureItem
    from docling_core.types.doc.document import PictureDescriptionData
    
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
    ThreadedPdfPipelineOptions = None
    PictureDescriptionVlmOptions = None
    PictureDescriptionApiOptions = None
    granite_picture_description = None
    smolvlm_picture_description = None
    AcceleratorOptions = None
    AcceleratorDevice = None
    download_models = None
    ConversionStatus = None
    DOCLING_LAYOUT_HERON = None
    DOCLING_LAYOUT_HERON_101 = None
    DOCLING_LAYOUT_EGRET_MEDIUM = None
    DOCLING_LAYOUT_EGRET_LARGE = None
    DOCLING_LAYOUT_EGRET_XLARGE = None
    AnyUrl = None
    PictureItem = None
    PictureDescriptionData = None
    logger.error(f"Failed to import docling: {e}")


class DoclingParserImages:
    """
    PDF parser with image description support using Docling.
    
    This is a modular extension that adds image description capabilities
    without modifying the base DoclingParser. Only initialized when
    parse_images=true is passed to the API.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Docling parser with image description enabled.
        
        Args:
            config: Optional configuration override
        """
        self.mode = "threaded-pipeline-with-image-description"
        
        if not DOCLING_AVAILABLE:
            logger.error("Docling is not available for image description")
            self.converter = None
            return
        
        # Read image description configuration from environment variables
        self.image_config = {
            # Model selection: 'smolvlm', 'granite', 'api', or 'custom'
            "model": os.environ.get('DOCLING_PICTURE_DESCRIPTION_MODEL', 'smolvlm').lower(),
            
            # Custom prompt (optional) - strip whitespace to handle Docker ENV formatting
            "prompt": os.environ.get('DOCLING_PICTURE_DESCRIPTION_PROMPT', '').strip(),
            
            # API configuration (for API-based models like GPT-4 Vision)
            "api_url": os.environ.get('DOCLING_PICTURE_DESCRIPTION_API_URL', ''),
            "api_key": os.environ.get('DOCLING_PICTURE_DESCRIPTION_API_KEY', ''),
            "api_model": os.environ.get('DOCLING_PICTURE_DESCRIPTION_API_MODEL', 'gpt-4o'),
            "api_timeout": float(os.environ.get('DOCLING_PICTURE_DESCRIPTION_API_TIMEOUT', '90')),
            
            # Picture description advanced settings
            "picture_area_threshold": float(os.environ.get('DOCLING_PICTURE_AREA_THRESHOLD', '0.05')),  # Min 5% of page area
            "batch_size": int(os.environ.get('DOCLING_PICTURE_BATCH_SIZE', '8')),  # Images per batch
            "max_new_tokens": int(os.environ.get('DOCLING_PICTURE_MAX_NEW_TOKENS', '200')),  # Max tokens for VLM
            "scale": float(os.environ.get('DOCLING_PICTURE_SCALE', '2.0')),  # Image upscaling factor
            "provenance": os.environ.get('DOCLING_PICTURE_PROVENANCE', '').strip(),  # Custom provenance label
            "concurrency": int(os.environ.get('DOCLING_PICTURE_CONCURRENCY', '4')),  # Parallel API calls (API mode only)
            
            # Inherit other settings from base parser config
            "enable_ocr": os.environ.get('DOCLING_ENABLE_OCR', 'false').lower() == 'true',
            "ocr_languages": [lang.strip() for lang in os.environ.get('DOCLING_OCR_LANGUAGES', 'en').split(',')],
            "do_table_structure": os.environ.get('DOCLING_DO_TABLE_STRUCTURE', 'false').lower() == 'true',
            "table_mode": os.environ.get('DOCLING_TABLE_MODE', 'fast'),
            "do_cell_matching": os.environ.get('DOCLING_DO_CELL_MATCHING', 'false').lower() == 'true',
            "num_threads": int(os.environ.get('OMP_NUM_THREADS', '100')),
            "accelerator_device": os.environ.get('DOCLING_ACCELERATOR_DEVICE', 'cpu'),
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
            
            # Picture Classification (separate from description - adds category labels)
            "do_picture_classification": os.environ.get('DOCLING_DO_PICTURE_CLASSIFICATION', 'false').lower() == 'true',
        }
        
        # Override with provided config if any
        if config:
            self.image_config.update(config)
        
        # Enable Docling's built-in pipeline profiling for detailed timing (configurable)
        if settings:
            enable_profiling = os.environ.get('DOCLING_ENABLE_PROFILING', 'false').lower() == 'true'
            settings.debug.profile_pipeline_timings = enable_profiling
            if enable_profiling:
                logger.debug("🔍 Docling pipeline profiling enabled")
            else:
                logger.debug("🔍 Docling pipeline profiling disabled (set DOCLING_ENABLE_PROFILING=true to enable)")
        
        # Create pipeline options with image description enabled
        pipeline_options = self._create_pipeline_options(self.image_config)
        
        # Initialize converter with image description enabled
        logger.info("🚀 Initializing Docling DocumentConverter with image description support")
        logger.info(f"📸 Image description model: {self.image_config['model']}")
        if self.image_config.get('prompt'):
            logger.info(f"📝 Custom prompt configured: '{self.image_config['prompt']}'")
        else:
            logger.info("📝 Using default model prompt")
        
        init_start = time.time()
        
        try:
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=ThreadedStandardPdfPipeline,
                        pipeline_options=pipeline_options,
                    ),
                }
            )
            
            # Try to pre-initialize the pipeline
            try:
                self.converter.initialize_pipeline(InputFormat.PDF)
                init_time = time.time() - init_start
                logger.info(f"✅ Image-enabled converter initialized in {init_time:.2f}s")
                self._pipeline_initialized = True
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"⚠️  Models not available at startup: {str(e)}")
                logger.info("📦 Pipeline will initialize lazily on first PDF conversion")
                self._pipeline_initialized = False
            except Exception as e:
                logger.warning(f"⚠️  Pipeline initialization failed: {str(e)}")
                logger.info("📦 Pipeline will initialize lazily on first PDF conversion")
                self._pipeline_initialized = False
        except Exception as e:
            logger.error(f"❌ Failed to create image-enabled converter: {str(e)}")
            self.converter = None
            self._pipeline_initialized = False
    
    def _create_pipeline_options(self, user_options: Dict[str, Any]) -> ThreadedPdfPipelineOptions:
        """
        Create ThreadedPdfPipelineOptions with image description enabled.
        """
        if not DOCLING_AVAILABLE:
            raise RuntimeError("Docling is not available")
        
        # Create base pipeline options
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
        
        # Picture classification (categorizes images: diagram, photo, chart, etc.)
        # This is separate from picture description - classification adds category labels
        pipeline_options.do_picture_classification = user_options.get("do_picture_classification", False)
        
        # ============================================================================
        # IMAGE DESCRIPTION CONFIGURATION
        # ============================================================================
        pipeline_options.do_picture_description = True
        
        model_type = user_options.get("model", "smolvlm").lower()
        
        if model_type == "api":
            # API-based model (e.g., GPT-4 Vision via OpenAI)
            api_url = user_options.get("api_url", "")
            api_key = user_options.get("api_key", "")
            api_model = user_options.get("api_model", "gpt-4o")
            api_timeout = user_options.get("api_timeout", 90.0)
            custom_prompt = user_options.get("prompt", "Describe this image in a few sentences.")
            
            logger.info(f"🔧 API model prompt: '{custom_prompt}'")
            
            if not api_url:
                logger.warning("⚠️  API URL not provided, falling back to SmolVLM")
                pipeline_options.picture_description_options = smolvlm_picture_description
            else:
                # Enable remote services for API calls
                pipeline_options.enable_remote_services = True
                
                headers = {}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                pipeline_options.picture_description_options = PictureDescriptionApiOptions(
                    url=AnyUrl(api_url),
                    params={"model": api_model},
                    headers=headers,
                    prompt=custom_prompt if custom_prompt else "Describe this image in a few sentences.",
                    timeout=api_timeout,
                    picture_area_threshold=user_options.get("picture_area_threshold", 0.05),
                    batch_size=user_options.get("batch_size", 8),
                    scale=user_options.get("scale", 2.0),
                    provenance=user_options.get("provenance", "") or f"api-{api_model}",
                    concurrency=user_options.get("concurrency", 4),
                )
                logger.info(f"✅ Configured API-based image description: {api_model} at {api_url}")
        
        elif model_type == "granite":
            # Granite Vision model
            custom_prompt = user_options.get("prompt", "")
            generation_config = {
                "max_new_tokens": user_options.get("max_new_tokens", 200),
                "do_sample": False
            }
            if custom_prompt:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=granite_picture_description.repo_id,
                    prompt=custom_prompt,
                    picture_area_threshold=user_options.get("picture_area_threshold", 0.05),
                    batch_size=user_options.get("batch_size", 8),
                    scale=user_options.get("scale", 2.0),
                    generation_config=generation_config,
                )
                logger.info(f"✅ Configured Granite Vision model with custom prompt: '{custom_prompt}'")
            else:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=granite_picture_description.repo_id,
                    prompt=granite_picture_description.prompt,
                    picture_area_threshold=user_options.get("picture_area_threshold", 0.05),
                    batch_size=user_options.get("batch_size", 8),
                    scale=user_options.get("scale", 2.0),
                    generation_config=generation_config,
                )
                logger.info("✅ Configured Granite Vision model with default prompt")
            
            # Download model if needed (will happen lazily on first use)
            try:
                if download_models is not None:
                    download_models(
                        output_dir=None,
                        force=False,
                        progress=True,
                        with_granite_vision=True,
                    )
            except Exception as e:
                logger.warning(f"⚠️  Could not pre-download Granite Vision model: {e}")
        
        elif model_type == "custom":
            # Custom VLM model
            repo_id = user_options.get("custom_repo_id", "")
            custom_prompt = user_options.get("prompt", "Describe this image in a few sentences.")
            generation_config = {
                "max_new_tokens": user_options.get("max_new_tokens", 200),
                "do_sample": False
            }
            
            if not repo_id:
                logger.warning("⚠️  Custom repo_id not provided, falling back to SmolVLM")
                pipeline_options.picture_description_options = smolvlm_picture_description
            else:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=repo_id,
                    prompt=custom_prompt,
                    picture_area_threshold=user_options.get("picture_area_threshold", 0.05),
                    batch_size=user_options.get("batch_size", 8),
                    scale=user_options.get("scale", 2.0),
                    generation_config=generation_config,
                )
                logger.info(f"✅ Configured custom VLM model: {repo_id}")
        
        else:
            # Default: SmolVLM model
            custom_prompt = user_options.get("prompt", "")
            generation_config = {
                "max_new_tokens": user_options.get("max_new_tokens", 200),
                "do_sample": False
            }
            if custom_prompt:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=smolvlm_picture_description.repo_id,
                    prompt=custom_prompt,
                    picture_area_threshold=user_options.get("picture_area_threshold", 0.05),
                    batch_size=user_options.get("batch_size", 8),
                    scale=user_options.get("scale", 2.0),
                    generation_config=generation_config,
                )
                logger.info(f"✅ Configured SmolVLM model with custom prompt: '{custom_prompt}'")
            else:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=smolvlm_picture_description.repo_id,
                    prompt=smolvlm_picture_description.prompt,
                    picture_area_threshold=user_options.get("picture_area_threshold", 0.05),
                    batch_size=user_options.get("batch_size", 8),
                    scale=user_options.get("scale", 2.0),
                    generation_config=generation_config,
                )
                logger.info("✅ Configured SmolVLM model with default prompt")
            
            # Download model if needed (will happen lazily on first use)
            try:
                if download_models is not None:
                    download_models(
                        output_dir=None,
                        force=False,
                        progress=True,
                        with_smolvlm=True,
                    )
            except Exception as e:
                logger.warning(f"⚠️  Could not pre-download SmolVLM model: {e}")
        
        return pipeline_options
    
    def _deduplicate_markdown(self, markdown_content: str) -> str:
        """
        Remove repetitive sentences/phrases from markdown content.
        
        This handles cases where image descriptions or other content
        gets duplicated in the markdown export.
        
        Args:
            markdown_content: Original markdown content
            
        Returns:
            Deduplicated markdown content
        """
        if not markdown_content:
            return markdown_content
        
        # Split into sentences (handle . ! ? endings)
        # Use a regex to split on sentence boundaries while preserving punctuation
        sentences = re.split(r'([.!?]\s+)', markdown_content)
        
        # Reconstruct sentences (alternating between text and separators)
        seen_sentences = set()
        deduplicated_parts = []
        
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
                # We have text + separator
                sentence_text = sentences[i].strip()
                separator = sentences[i + 1]
                
                if sentence_text:
                    # Normalize for comparison: lowercase, remove extra whitespace
                    normalized = ' '.join(sentence_text.lower().split())
                    
                    # Only add if we haven't seen this exact sentence before
                    if normalized not in seen_sentences:
                        deduplicated_parts.append(sentence_text + separator)
                        seen_sentences.add(normalized)
                        # Keep a sliding window of last 20 sentences to avoid memory growth
                        if len(seen_sentences) > 20:
                            # Convert to list, keep last 15, convert back to set
                            seen_list = list(seen_sentences)
                            seen_sentences = set(seen_list[-15:])
                    # else: skip this duplicate sentence
                else:
                    # Empty text, keep separator if it exists
                    deduplicated_parts.append(separator)
                
                i += 2
            else:
                # Last item (no separator after it)
                sentence_text = sentences[i].strip()
                if sentence_text:
                    normalized = ' '.join(sentence_text.lower().split())
                    if normalized not in seen_sentences:
                        deduplicated_parts.append(sentence_text)
                        seen_sentences.add(normalized)
                break
        
        result = ''.join(deduplicated_parts)
        
        # Log if we removed significant content
        original_len = len(markdown_content)
        new_len = len(result)
        if original_len > new_len * 1.1:  # More than 10% reduction
            reduction_pct = ((original_len - new_len) / original_len) * 100
            logger.info(f"🔧 Removed {reduction_pct:.1f}% duplicate content ({original_len:,} -> {new_len:,} chars)")
        
        return result
    
    def _log_image_descriptions(self, document):
        """
        Log information about images and their descriptions in the document.
        
        Args:
            document: DoclingDocument object
        """
        if not DOCLING_AVAILABLE or PictureItem is None:
            return
        
        try:
            picture_count = 0
            described_count = 0
            descriptions = []
            
            # Iterate through all items in the document
            for element, _level in document.iterate_items():
                if isinstance(element, PictureItem):
                    picture_count += 1
                    
                    # Get description/caption if available
                    caption = None
                    try:
                        if hasattr(element, 'caption_text'):
                            caption = element.caption_text(doc=document)
                    except Exception:
                        pass
                    
                    # Extract text from annotations (PictureDescriptionData objects from docling)
                    annotation_texts = []
                    annotations = getattr(element, 'annotations', [])
                    annotation_types = []
                    for ann in annotations:
                        annotation_types.append(type(ann).__name__)
                        # Check for PictureDescriptionData first (docling's picture description type)
                        if PictureDescriptionData is not None and isinstance(ann, PictureDescriptionData):
                            if ann.text and ann.text.strip():
                                annotation_texts.append(ann.text)
                        elif isinstance(ann, str):
                            annotation_texts.append(ann)
                        else:
                            # Fallback: Try common attributes for other annotation types
                            for attr in ['text', 'content', 'description', 'value', 'annotation']:
                                if hasattr(ann, attr):
                                    text = getattr(ann, attr)
                                    if isinstance(text, str) and text.strip():
                                        annotation_texts.append(text)
                                        break
                            # If no text attribute found, try converting to string
                            if not annotation_texts:
                                try:
                                    text = str(ann)
                                    if text and text.strip() and text != str(type(ann)):
                                        annotation_texts.append(text)
                                except Exception:
                                    pass
                    
                    # Combine caption and annotations
                    desc_text = None
                    if caption:
                        desc_text = caption
                    elif annotation_texts:
                        desc_text = annotation_texts[0]  # Use first annotation
                    
                    # Check if image has description
                    has_description = bool(desc_text)
                    if has_description:
                        described_count += 1
                    
                    # Get image reference/ID
                    image_ref = getattr(element, 'self_ref', f'image_{picture_count}')
                    
                    # Get page number if available
                    page_num = None
                    try:
                        if hasattr(element, 'page'):
                            page_num = element.page
                        elif hasattr(element, 'meta') and hasattr(element.meta, 'page'):
                            page_num = element.meta.page
                    except Exception:
                        pass
                    
                    # Calculate image size fraction of page (for diagnostics)
                    area_fraction = None
                    below_threshold = False
                    try:
                        if len(element.prov) > 0:
                            prov = element.prov[0]
                            page = document.pages.get(prov.page_no)
                            if page is not None and page.size is not None:
                                page_area = page.size.width * page.size.height
                                if page_area > 0:
                                    area_fraction = prov.bbox.area() / page_area
                                    threshold = self.image_config.get("picture_area_threshold", 0.05)
                                    below_threshold = area_fraction < threshold
                    except Exception:
                        pass
                    
                    # Log individual image info
                    if has_description:
                        # Log full description text (no truncation)
                        page_info = f" (page {page_num})" if page_num is not None else ""
                        size_info = f" [{area_fraction*100:.1f}% of page]" if area_fraction is not None else ""
                        logger.info(f"📸 Image {picture_count} ({image_ref}){page_info}{size_info}: {desc_text}")
                        descriptions.append({
                            'ref': str(image_ref),
                            'description': desc_text,
                            'page': page_num
                        })
                    else:
                        page_info = f" (page {page_num})" if page_num is not None else ""
                        size_info = f" [{area_fraction*100:.1f}% of page]" if area_fraction is not None else ""
                        # Log diagnostic info for images without descriptions
                        has_annotations = len(annotations) > 0
                        annotation_info = f" ({len(annotations)} annotations: {', '.join(annotation_types[:3])})" if has_annotations else " (no annotations)"
                        
                        # Add warning if image is too small
                        if below_threshold:
                            logger.warning(f"📸 Image {picture_count} ({image_ref}){page_info}{size_info}: Likely filtered by area threshold (< {self.image_config.get('picture_area_threshold', 0.05)*100:.0f}%){annotation_info}")
                        else:
                            logger.info(f"📸 Image {picture_count} ({image_ref}){page_info}{size_info}: No description available{annotation_info}")
                        
                        # Log additional diagnostic info at DEBUG level
                        logger.debug(f"   - Caption available: {caption is not None}")
                        logger.debug(f"   - Annotations count: {len(annotations)}")
                        if annotations:
                            logger.debug(f"   - Annotation types: {[type(a).__name__ for a in annotations]}")
                        # Check for image properties that might affect description
                        try:
                            if hasattr(element, 'bbox'):
                                bbox = element.bbox
                                logger.debug(f"   - Bounding box: {bbox}")
                            if hasattr(element, 'image'):
                                logger.debug(f"   - Has image data: {element.image is not None}")
                        except Exception:
                            pass
            
            # Log summary
            if picture_count > 0:
                logger.info(f"📊 Image description summary: {described_count}/{picture_count} images described")
                if described_count > 0:
                    logger.info(f"✅ Successfully described {described_count} image(s)")
                if described_count < picture_count:
                    logger.warning(f"⚠️  {picture_count - described_count} image(s) without descriptions")
            else:
                logger.info("📊 No images found in document")
                
        except Exception as e:
            logger.warning(f"⚠️  Could not log image descriptions: {str(e)}")
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Parse PDF content with image descriptions enabled.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Dictionary with success status, content (markdown), and error if any
        """
        if not DOCLING_AVAILABLE or self.converter is None:
            return {
                "success": False,
                "error": "Docling image description is not available",
                "content": None
            }
        
        # Ensure pipeline is initialized
        if not getattr(self, '_pipeline_initialized', True):
            try:
                logger.info("🔄 Initializing image-enabled pipeline on first use...")
                self.converter.initialize_pipeline(InputFormat.PDF)
                self._pipeline_initialized = True
                logger.info("✅ Image-enabled pipeline initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize image-enabled pipeline: {str(e)}")
                return {
                    "success": False,
                    "error": f"Pipeline initialization failed: {str(e)}",
                    "content": None
                }
        
        try:
            processing_start = time.time()
            doc_stream = DocumentStream(name="document.pdf", stream=BytesIO(pdf_content))
            
            # Log configuration being used
            model_info = f"model={self.image_config.get('model', 'N/A')}"
            prompt_info = f"prompt='{self.image_config.get('prompt', 'default')}'" if self.image_config.get('prompt') else "prompt=default"
            logger.info(f"📄 Processing PDF with image descriptions ({len(pdf_content):,} bytes) - {model_info}, {prompt_info}")
            
            # Parse page range if specified (format: "1-100" or empty for all)
            page_range = None
            page_range_str = self.image_config.get("page_range", "")
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
            
            # Convert with safety limits
            result = self.converter.convert(
                source=doc_stream,
                max_num_pages=self.image_config.get("max_num_pages", 1000),
                max_file_size=self.image_config.get("max_file_size", 104857600),
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
                
                # Log image descriptions
                self._log_image_descriptions(document)
                
                markdown_content = document.export_to_markdown()
            
            # Deduplicate repetitive content (especially image descriptions)
            if markdown_content:
                markdown_content = self._deduplicate_markdown(markdown_content)
            
            export_time = time.time() - export_start
            
            total_time = time.time() - processing_start
            
            # Log summary with status
            if status == ConversionStatus.SUCCESS:
                logger.info(f"✅ Parsed with image descriptions in {total_time:.2f}s: {len(markdown_content):,} chars ({len(markdown_content)/total_time:.0f} chars/sec)" if markdown_content else f"✅ Parsed with image descriptions in {total_time:.2f}s")
            elif status == ConversionStatus.PARTIAL_SUCCESS:
                logger.warning(f"⚠️  Partially parsed with image descriptions in {total_time:.2f}s: {len(markdown_content):,} chars (warnings: {len(warnings)})" if markdown_content else f"⚠️  Partially parsed with image descriptions in {total_time:.2f}s (warnings: {len(warnings)})")
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
            logger.error(f"❌ Error during PDF parsing with image descriptions: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def parse_pdf_to_document(self, pdf_content: bytes):
        """
        Parse PDF content and return DoclingDocument object with image descriptions.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            DoclingDocument object if successful
            
        Raises:
            RuntimeError: If Docling is not available or conversion fails
        """
        if not DOCLING_AVAILABLE or self.converter is None:
            raise RuntimeError("Docling image description is not available")
        
        # Ensure pipeline is initialized
        if not getattr(self, '_pipeline_initialized', True):
            try:
                logger.info("🔄 Initializing image-enabled pipeline on first use...")
                self.converter.initialize_pipeline(InputFormat.PDF)
                self._pipeline_initialized = True
                logger.info("✅ Image-enabled pipeline initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize image-enabled pipeline: {str(e)}")
                raise RuntimeError(f"Pipeline initialization failed: {str(e)}") from e
        
        try:
            parse_start = time.time()
            doc_stream = DocumentStream(name="document.pdf", stream=BytesIO(pdf_content))
            
            logger.info(f"📄 Parsing PDF to document with image descriptions ({len(pdf_content):,} bytes)")
            
            # Parse page range if specified (format: "1-100" or empty for all)
            page_range = None
            page_range_str = self.image_config.get("page_range", "")
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
            
            # Convert with safety limits
            conversion_start = time.time()
            result = self.converter.convert(
                source=doc_stream,
                max_num_pages=self.image_config.get("max_num_pages", 1000),
                max_file_size=self.image_config.get("max_file_size", 104857600),
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
            
            # Log image descriptions
            document = result.document
            self._log_image_descriptions(document)
            
            logger.info(f"✅ Document parsed with image descriptions in {total_time:.2f}s (conversion: {conversion_time:.2f}s)")
            
            return document
                
        except Exception as e:
            logger.error(f"❌ Error parsing PDF to document with image descriptions: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to parse PDF: {str(e)}") from e
    
    def is_available(self) -> bool:
        """Check if image description parsing is available."""
        return DOCLING_AVAILABLE and self.converter is not None
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the image-enabled parser."""
        return {
            "available": self.is_available(),
            "library": "docling" if DOCLING_AVAILABLE else None,
            "pipeline": "ThreadedStandardPdfPipeline with image description (pre-initialized)" if self.is_available() else None,
            "description": "High-performance PDF parsing with image description support using VLM models",
            "performance_mode": "optimized_with_image_description",
            "image_configuration": self.image_config if hasattr(self, 'image_config') else {},
            "performance": {
                "initialization": "One-time at startup (~5-30 seconds depending on cache)",
                "per_request": "Fast (<1 second per page with warm cache, +VLM inference time for images)",
                "threading": f"{self.image_config.get('num_threads', 120)} threads configured" if hasattr(self, 'image_config') else "N/A",
                "batching": "Enabled with aggressive batch sizes",
                "image_description": f"Enabled with model: {self.image_config.get('model', 'N/A')}" if hasattr(self, 'image_config') else "N/A"
            }
        }

