"""
VLM (Vision Language Model) service for end-to-end PDF parsing using remote models.
Pre-initializes the converter once for optimal performance.

This service uses remote VLM APIs (like vLLM-served Granite Docling) for 
document understanding without traditional layout analysis pipelines.
"""

import os
import tempfile
import logging
import time
from typing import Optional, Dict, Any

# Try to import docling VLM components
try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
    VLM_AVAILABLE = True
except ImportError as e:
    VLM_AVAILABLE = False
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    VlmPipeline = None
    VlmPipelineOptions = None
    ApiVlmOptions = None
    ResponseFormat = None
    logging.error(f"Failed to import docling VLM components: {e}")

logger = logging.getLogger(__name__)


class VlmParser:
    """
    High-performance VLM PDF parser using Docling with remote VLM services.
    
    CRITICAL: The converter should be initialized ONCE at startup and reused
    for all requests to avoid expensive initialization overhead.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VLM parser with remote model configuration.
        
        Args:
            config: Optional configuration for the parser (defaults to ENV vars)
        """
        self.mode = "vlm-remote-api"
        
        if not VLM_AVAILABLE:
            logger.error("Docling VLM components are not available")
            self.converter = None
            return
        
        # Read configuration from environment variables (set in Dockerfile)
        self.default_config = config or {
            "vlm_url": os.environ.get('DOCLING_VLM_URL', 'http://192.168.20.74:8004/v1/chat/completions'),
            "vlm_model": os.environ.get('DOCLING_VLM_MODEL', 'ibm-granite/granite-docling-258M'),
            "vlm_api_key": os.environ.get('DOCLING_VLM_API_KEY', 'blabla'),
            "vlm_timeout": int(os.environ.get('DOCLING_VLM_TIMEOUT', '90')),
            "vlm_temperature": float(os.environ.get('DOCLING_VLM_TEMPERATURE', '0.7')),
            "vlm_max_tokens": int(os.environ.get('DOCLING_VLM_MAX_TOKENS', '4096')),
            "vlm_scale": float(os.environ.get('DOCLING_VLM_SCALE', '2.0')),
            "vlm_prompt": os.environ.get('DOCLING_VLM_PROMPT', 'Convert this page to docling.'),
        }
        
        # Create VLM pipeline options
        logger.info("============================================================")
        logger.info("🚀 Initializing VLM DocumentConverter (ONE-TIME SETUP)")
        logger.info("⚙️  Configuration (from Dockerfile ENV):")
        logger.info(f"   🌐 Remote VLM Service:")
        logger.info(f"      - URL: {self.default_config['vlm_url']}")
        logger.info(f"      - Model: {self.default_config['vlm_model']}")
        logger.info(f"      - Timeout: {self.default_config['vlm_timeout']}s")
        logger.info(f"      - Temperature: {self.default_config['vlm_temperature']}")
        logger.info(f"      - Max Tokens: {self.default_config['vlm_max_tokens']}")
        logger.info(f"      - Scale: {self.default_config['vlm_scale']}")
        logger.info(f"   📝 Prompt: \"{self.default_config['vlm_prompt']}\"")
        
        init_start = time.time()
        
        try:
            # Create ApiVlmOptions for remote model
            vlm_options = self._create_vlm_options(self.default_config)
            
            # Create VlmPipelineOptions with remote services enabled
            pipeline_options = VlmPipelineOptions(
                enable_remote_services=True  # Required for remote VLM APIs
            )
            pipeline_options.vlm_options = vlm_options
            
            # Initialize the converter with VLM pipeline
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=pipeline_options,
                    )
                }
            )
            
            # Pre-initialize the pipeline
            logger.info("📦 Pre-loading VLM pipeline...")
            self.converter.initialize_pipeline(InputFormat.PDF)
            
            init_time = time.time() - init_start
            logger.info(f"✅ VLM Converter initialized in {init_time:.2f} seconds")
            logger.info("📝 VLM pipeline ready for remote processing")
            logger.info("============================================================")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize VLM converter: {str(e)}", exc_info=True)
            self.converter = None
    
    def _create_vlm_options(self, config: Dict[str, Any]) -> ApiVlmOptions:
        """
        Create ApiVlmOptions for remote VLM model.
        
        Args:
            config: Configuration dictionary with VLM settings
            
        Returns:
            Configured ApiVlmOptions instance
        """
        if not VLM_AVAILABLE:
            raise RuntimeError("Docling VLM components are not available")
        
        # Prepare headers with API key
        headers = {}
        if config['vlm_api_key']:
            headers["Authorization"] = f"Bearer {config['vlm_api_key']}"
        
        # Create ApiVlmOptions following Docling's OpenAI-compatible pattern
        options = ApiVlmOptions(
            url=config['vlm_url'],
            params=dict(
                model=config['vlm_model'],
                max_tokens=config['vlm_max_tokens'],
                top_p=0.95,
                frequency_penalty=0.2,
                presence_penalty=0.2,
                stop=["<|end_of_text|>", "</document>", "\n\n\n\n"],
            ),
            headers=headers,
            prompt=config['vlm_prompt'],
            timeout=config['vlm_timeout'],
            scale=config['vlm_scale'],
            temperature=config['vlm_temperature'],
            response_format=ResponseFormat.DOCTAGS,  # Granite Docling uses DOCTAGS format
        )
        
        logger.info(
            "VLM options configured: url=%s, model=%s, timeout=%ss",
            config['vlm_url'],
            config['vlm_model'],
            config['vlm_timeout']
        )
        
        return options
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        """
        Parse PDF content using the pre-initialized VLM converter.
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Dictionary with parsing results:
            - success: bool
            - content: str (markdown content)
            - error: Optional[str]
            - processing_time: float
            - timings: dict with detailed timing breakdown
        """
        if not VLM_AVAILABLE or self.converter is None:
            return {
                "success": False,
                "error": "VLM parser is not available",
                "content": None
            }
        
        try:
            # Log start
            logger.info("============================================================")
            logger.info("📄 Starting VLM PDF parsing with pre-initialized converter")
            logger.info("------------------------------------------------------------")
            
            # Create temporary file for PDF content
            file_write_start = time.time()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                file_write_time = time.time() - file_write_start
                logger.info(f"⏱️  Temp file write: {file_write_time:.3f}s")
                
                # Parse the PDF using pre-initialized VLM converter
                logger.info("⏳ Processing document with VLM...")
                logger.info("   └─ Step 1: VLM document conversion (remote API call)")
                processing_start = time.time()
                
                # CRITICAL: Reuse the pre-initialized converter
                result = self.converter.convert(source=tmp_file.name)
                
                conversion_time = time.time() - processing_start
                logger.info(f"   ✅ VLM conversion complete: {conversion_time:.3f}s")
                
                # Extract content - export to markdown
                logger.info("   └─ Step 2: Export to markdown")
                export_start = time.time()
                document = result.document
                markdown_content = document.export_to_markdown()
                export_time = time.time() - export_start
                logger.info(f"   ✅ Export complete: {export_time:.3f}s")
                
                total_time = time.time() - processing_start
                
                # Detailed timing breakdown
                logger.info("📊 VLM Performance Breakdown:")
                logger.info(f"   ├─ File I/O:        {file_write_time:.3f}s ({file_write_time/total_time*100:.1f}%)")
                logger.info(f"   ├─ VLM Conversion:  {conversion_time:.3f}s ({conversion_time/total_time*100:.1f}%)")
                logger.info(f"   ├─ Markdown Export: {export_time:.3f}s ({export_time/total_time*100:.1f}%)")
                logger.info(f"   └─ TOTAL:           {total_time:.3f}s")
                
                # Document statistics
                logger.info(f"📄 Document Statistics:")
                logger.info(f"   ├─ Content size:    {len(markdown_content):,} characters")
                logger.info(f"   ├─ Input size:      {len(pdf_content):,} bytes")
                logger.info(f"   └─ Throughput:      {len(markdown_content)/total_time:.0f} chars/sec")
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
                    "processing_time": total_time,
                    "timings": {
                        "file_write": file_write_time,
                        "vlm_conversion": conversion_time,
                        "markdown_export": export_time,
                        "total": total_time
                    }
                }
                
        except Exception as e:
            logger.error(f"❌ Error during VLM PDF parsing: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def is_available(self) -> bool:
        """Check if VLM parsing is available."""
        return VLM_AVAILABLE and self.converter is not None
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get information about the VLM parser."""
        return {
            "available": self.is_available(),
            "library": "docling-vlm" if VLM_AVAILABLE else None,
            "pipeline": "VlmPipeline with remote API (pre-initialized)" if self.is_available() else None,
            "description": "End-to-end PDF parsing using remote Vision-Language Model",
            "performance_mode": "optimized_with_preinitialization",
            "configuration": self.default_config if hasattr(self, 'default_config') else {},
            "performance": {
                "initialization": "One-time at startup (~1-5 seconds)",
                "per_request": "Depends on remote VLM service latency",
                "remote_service": f"{self.default_config.get('vlm_url', 'N/A')}" if hasattr(self, 'default_config') else "N/A"
            },
            "limitations": [
                "Chunking is not supported with VLM parsing",
                "Returns full markdown only",
                "Requires remote VLM service to be running"
            ]
        }

