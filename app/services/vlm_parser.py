"""VLM (Vision Language Model) parser service for end-to-end PDF parsing."""

import os
import tempfile
import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

try:
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    VLM_AVAILABLE = True
except ImportError as e:
    VLM_AVAILABLE = False
    InputFormat = None
    DocumentConverter = None
    PdfFormatOption = None
    VlmPipeline = None
    logger.error(f"Failed to import docling VLM components: {e}")


class VlmParser:
    def __init__(self):
        self.mode = "vlm-pipeline"
        
        if not VLM_AVAILABLE:
            logger.error("Docling VLM components are not available")
            self.converter = None
            return
        
        logger.info("============================================================")
        logger.info("🚀 Initializing VLM DocumentConverter (MINIMAL)")
        logger.info("============================================================")
        
        init_start = time.time()
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                ),
            }
        )
        
        logger.info("📦 Pre-loading VLM model...")
        self.converter.initialize_pipeline(InputFormat.PDF)
        
        init_time = time.time() - init_start
        logger.info(f"✅ VLM converter initialized in {init_time:.2f} seconds")
        logger.info("============================================================")
    
    def parse_pdf(self, pdf_content: bytes) -> Dict[str, Any]:
        if not VLM_AVAILABLE or self.converter is None:
            return {"success": False, "error": "VLM parser is not available", "content": None}
        
        try:
            logger.info("🤖 Starting VLM PDF parsing...")
            processing_start = time.time()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                tmp_file.flush()
                
                result = self.converter.convert(source=tmp_file.name)
                markdown_content = result.document.export_to_markdown()
                
                os.unlink(tmp_file.name)
                
                total_time = time.time() - processing_start
                logger.info(f"✅ VLM parsing complete in {total_time:.2f}s")
                
                return {"success": True, "content": markdown_content, "error": None}
        except Exception as e:
            logger.error(f"❌ Error during VLM PDF parsing: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e), "content": None}
    
    def is_available(self) -> bool:
        return VLM_AVAILABLE and self.converter is not None
    
    def get_parser_info(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "library": "docling" if VLM_AVAILABLE else None,
            "pipeline": "VlmPipeline (GraniteDocling default)",
            "description": "Vision Language Model for end-to-end PDF parsing",
            "model": "ibm-granite/granite-docling-258M (default)",
            "backend": "Transformers",
        }
