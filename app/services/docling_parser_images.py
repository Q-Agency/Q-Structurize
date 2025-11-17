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
    from docling.datamodel.base_models import InputFormat, DocumentStream
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
    from pydantic import AnyUrl
    from docling_core.types.doc import PictureItem
    
    settings.perf.page_batch_size = int(os.environ.get('DOCLING_PAGE_BATCH_SIZE', '12'))
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
    AnyUrl = None
    PictureItem = None
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
            "queue_max_size": int(os.environ.get('DOCLING_QUEUE_MAX_SIZE', '1000'))
        }
        
        # Override with provided config if any
        if config:
            self.image_config.update(config)
        
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
                )
                logger.info(f"✅ Configured API-based image description: {api_model} at {api_url}")
        
        elif model_type == "granite":
            # Granite Vision model
            custom_prompt = user_options.get("prompt", "")
            if custom_prompt:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=granite_picture_description.repo_id,
                    prompt=custom_prompt,
                )
                logger.info(f"✅ Configured Granite Vision model with custom prompt: '{custom_prompt}'")
            else:
                pipeline_options.picture_description_options = granite_picture_description
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
            
            if not repo_id:
                logger.warning("⚠️  Custom repo_id not provided, falling back to SmolVLM")
                pipeline_options.picture_description_options = smolvlm_picture_description
            else:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=repo_id,
                    prompt=custom_prompt,
                )
                logger.info(f"✅ Configured custom VLM model: {repo_id}")
        
        else:
            # Default: SmolVLM model
            custom_prompt = user_options.get("prompt", "")
            if custom_prompt:
                pipeline_options.picture_description_options = PictureDescriptionVlmOptions(
                    repo_id=smolvlm_picture_description.repo_id,
                    prompt=custom_prompt,
                )
                logger.info(f"✅ Configured SmolVLM model with custom prompt: '{custom_prompt}'")
            else:
                pipeline_options.picture_description_options = smolvlm_picture_description
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
                    
                    # Extract text from annotations (may be DescriptionAnnotation objects)
                    annotation_texts = []
                    annotations = getattr(element, 'annotations', [])
                    annotation_types = []
                    for ann in annotations:
                        annotation_types.append(type(ann).__name__)
                        if isinstance(ann, str):
                            annotation_texts.append(ann)
                        else:
                            # Try common attributes for DescriptionAnnotation objects
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
                    
                    # Log individual image info
                    if has_description:
                        # Truncate long descriptions for logging
                        desc_preview = desc_text[:150] + "..." if len(desc_text) > 150 else desc_text
                        page_info = f" (page {page_num})" if page_num is not None else ""
                        logger.info(f"📸 Image {picture_count} ({image_ref}){page_info}: {desc_preview}")
                        descriptions.append({
                            'ref': str(image_ref),
                            'description': desc_text,
                            'page': page_num
                        })
                    else:
                        page_info = f" (page {page_num})" if page_num is not None else ""
                        # Log diagnostic info for images without descriptions
                        has_annotations = len(annotations) > 0
                        annotation_info = f" ({len(annotations)} annotations: {', '.join(annotation_types[:3])})" if has_annotations else " (no annotations)"
                        logger.info(f"📸 Image {picture_count} ({image_ref}){page_info}: No description available{annotation_info}")
                        
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
            
            result = self.converter.convert(source=doc_stream)
            conversion_time = time.time() - processing_start
            
            # Extract content - export to markdown
            export_start = time.time()
            document = result.document
            
            # Log image descriptions
            self._log_image_descriptions(document)
            
            markdown_content = document.export_to_markdown()
            
            # Deduplicate repetitive content (especially image descriptions)
            markdown_content = self._deduplicate_markdown(markdown_content)
            
            export_time = time.time() - export_start
            
            total_time = time.time() - processing_start
            
            logger.info(f"✅ Parsed with image descriptions in {total_time:.2f}s: {len(markdown_content):,} chars")
            
            return {
                "success": True,
                "content": markdown_content,
                "error": None,
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
            
            conversion_start = time.time()
            result = self.converter.convert(source=doc_stream)
            conversion_time = time.time() - conversion_start
            
            # Log image descriptions
            document = result.document
            self._log_image_descriptions(document)
            
            total_time = time.time() - parse_start
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

