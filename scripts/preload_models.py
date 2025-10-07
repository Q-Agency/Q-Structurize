#!/usr/bin/env python3
"""
Model pre-download script for Q-Structurize.
Downloads and caches the GraniteDocling VLM model during Docker build.
"""

import os
import sys
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_cache_directories():
    """Setup cache directories for model storage."""
    cache_dirs = [
        '/app/.cache/huggingface',
        '/app/.cache/transformers', 
        '/app/.cache/torch'
    ]
    
    for cache_dir in cache_dirs:
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Created cache directory: {cache_dir}")
    
    # Set environment variables
    os.environ['HF_HOME'] = '/app/.cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/app/.cache/transformers'
    os.environ['TORCH_HOME'] = '/app/.cache/torch'
    os.environ['HF_HUB_CACHE'] = '/app/.cache/huggingface'
    
    logger.info("Cache directories configured")

def preload_docling_model():
    """Pre-download the GraniteDocling VLM model."""
    try:
        logger.info("=== STARTING MODEL PRE-DOWNLOAD ===")
        start_time = time.time()
        
        # Import docling components
        logger.info("Importing docling components...")
        from docling.datamodel import vlm_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
        
        logger.info("✅ Docling components imported successfully")
        
        # Get VLM options
        logger.info("Getting VLM model specifications...")
        vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
        
        # Configure for H200 GPU
        logger.info("Configuring VLM for H200 GPU optimization...")
        vlm_options.load_in_8bit = False
        vlm_options.max_new_tokens = 32768
        vlm_options.temperature = 0.0
        vlm_options.scale = 1.0
        vlm_options.use_kv_cache = True
        
        if hasattr(vlm_options, 'torch_dtype'):
            vlm_options.torch_dtype = 'float16'
        
        logger.info(f"VLM Config: load_in_8bit={vlm_options.load_in_8bit}, max_tokens={vlm_options.max_new_tokens}")
        
        # Initialize converter to trigger model download
        logger.info("Initializing DocumentConverter to trigger model download...")
        logger.info("⏳ This may take several minutes for first-time download...")
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    vlm_options=vlm_options,
                ),
            }
        )
        
        logger.info("✅ DocumentConverter initialized successfully")
        logger.info("✅ Model download and caching completed")
        
        # Check cache contents
        logger.info("=== CACHE VERIFICATION ===")
        cache_dirs = ['/app/.cache/huggingface', '/app/.cache/transformers', '/app/.cache/torch']
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    contents = os.listdir(cache_dir)
                    logger.info(f"Cache contents in {cache_dir}: {contents}")
                except Exception as e:
                    logger.warning(f"Could not list contents of {cache_dir}: {e}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"=== MODEL PRE-DOWNLOAD COMPLETED in {elapsed_time:.2f} seconds ===")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Failed to import docling: {e}")
        logger.error("Make sure docling is installed: pip install docling")
        return False
        
    except Exception as e:
        logger.error(f"❌ Failed to preload model: {e}")
        return False

def main():
    """Main preload function."""
    logger.info("🚀 Starting Q-Structurize model preload process...")
    
    # Setup cache directories
    setup_cache_directories()
    
    # Preload the model
    success = preload_docling_model()
    
    if success:
        logger.info("🎉 Model preload completed successfully!")
        logger.info("The model is now cached and ready for use.")
        sys.exit(0)
    else:
        logger.error("💥 Model preload failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
