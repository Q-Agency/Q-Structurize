#!/usr/bin/env python3
"""
Model pre-download script for Q-Structurize.
Downloads and caches the GraniteDocling VLM model during Docker build.
This ensures instant startup with no download delays at runtime.
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
        '/app/.cache/huggingface/hub',
        '/app/.cache/transformers', 
        '/app/.cache/torch'
    ]
    
    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created cache directory: {cache_dir}")
    
    # Set environment variables (double-check they're set)
    env_vars = {
        'HF_HOME': '/app/.cache/huggingface',
        'HF_HUB_CACHE': '/app/.cache/huggingface/hub',
        'TRANSFORMERS_CACHE': '/app/.cache/transformers',
        'TORCH_HOME': '/app/.cache/torch'
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"📌 Set {key}={value}")
    
    logger.info("✅ Cache directories configured")


def verify_cache_structure():
    """Verify that the cache has the expected structure."""
    expected_model_path = Path('/app/.cache/huggingface/hub/models--ibm-granite--granite-docling-258M')
    
    if expected_model_path.exists():
        logger.info(f"✅ Model cache found at: {expected_model_path}")
        
        # Count files in cache
        try:
            files = list(expected_model_path.rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            dir_count = len([f for f in files if f.is_dir()])
            
            logger.info(f"📊 Cache statistics:")
            logger.info(f"   - {file_count} files")
            logger.info(f"   - {dir_count} directories")
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            size_mb = total_size / (1024 * 1024)
            logger.info(f"   - Total size: {size_mb:.2f} MB")
            
            return True
        except Exception as e:
            logger.warning(f"Could not calculate cache statistics: {e}")
            return True  # Cache exists, just can't get stats
    else:
        logger.warning(f"⚠️  Model cache not found at expected location: {expected_model_path}")
        return False


def preload_docling_model():
    """Pre-download the GraniteDocling VLM model."""
    try:
        logger.info("=" * 70)
        logger.info("🚀 STARTING MODEL PRE-DOWNLOAD")
        logger.info("=" * 70)
        start_time = time.time()
        
        # Import docling components
        logger.info("📦 Importing docling components...")
        from docling.datamodel import vlm_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import VlmPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.pipeline.vlm_pipeline import VlmPipeline
        
        logger.info("✅ Docling components imported successfully")
        
        # Get VLM options
        logger.info("⚙️  Getting VLM model specifications...")
        vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
        
        logger.info(f"📝 Model repo: {vlm_options.repo_id if hasattr(vlm_options, 'repo_id') else 'ibm-granite/granite-docling-258M'}")
        
        # Configure for H200 GPU
        logger.info("🔧 Configuring VLM for H200 GPU optimization...")
        vlm_options.load_in_8bit = False
        vlm_options.max_new_tokens = 32768
        vlm_options.temperature = 0.0
        vlm_options.scale = 1.0
        vlm_options.use_kv_cache = True
        
        if hasattr(vlm_options, 'torch_dtype'):
            vlm_options.torch_dtype = 'float16'
        
        logger.info(f"✅ VLM Config:")
        logger.info(f"   - load_in_8bit: {vlm_options.load_in_8bit}")
        logger.info(f"   - max_new_tokens: {vlm_options.max_new_tokens}")
        logger.info(f"   - use_kv_cache: {vlm_options.use_kv_cache}")
        
        # Initialize converter to trigger model download
        logger.info("=" * 70)
        logger.info("⏳ Initializing DocumentConverter (this will download the model)...")
        logger.info("⏳ First-time download may take 5-10 minutes depending on network speed...")
        logger.info("=" * 70)
        
        init_start = time.time()
        
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=VlmPipelineOptions(
                        vlm_options=vlm_options,
                    ),
                ),
            }
        )
        
        init_time = time.time() - init_start
        
        logger.info("=" * 70)
        logger.info(f"✅ DocumentConverter initialized in {init_time:.2f} seconds")
        logger.info("✅ Model download and caching completed")
        logger.info("=" * 70)
        
        # Verify cache structure
        logger.info("🔍 Verifying cache structure...")
        cache_valid = verify_cache_structure()
        
        # Check cache contents
        logger.info("=" * 70)
        logger.info("📂 CACHE VERIFICATION")
        logger.info("=" * 70)
        
        cache_dirs = [
            '/app/.cache/huggingface/hub',
            '/app/.cache/transformers',
            '/app/.cache/torch'
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    contents = os.listdir(cache_dir)
                    logger.info(f"📁 {cache_dir}:")
                    if contents:
                        for item in contents[:10]:  # Show first 10 items
                            logger.info(f"   - {item}")
                        if len(contents) > 10:
                            logger.info(f"   ... and {len(contents) - 10} more items")
                    else:
                        logger.info(f"   (empty)")
                except Exception as e:
                    logger.warning(f"   ⚠️  Could not list contents: {e}")
            else:
                logger.warning(f"   ⚠️  Directory does not exist")
        
        elapsed_time = time.time() - start_time
        logger.info("=" * 70)
        logger.info(f"🎉 MODEL PRE-DOWNLOAD COMPLETED in {elapsed_time:.2f} seconds")
        logger.info("=" * 70)
        
        return cache_valid
        
    except ImportError as e:
        logger.error("=" * 70)
        logger.error(f"❌ Failed to import docling: {e}")
        logger.error("Make sure docling is installed: pip install docling")
        logger.error("=" * 70)
        return False
        
    except Exception as e:
        logger.error("=" * 70)
        logger.error(f"❌ Failed to preload model: {e}")
        logger.exception("Full traceback:")
        logger.error("=" * 70)
        return False


def main():
    """Main preload function."""
    logger.info("=" * 70)
    logger.info("🚀 Q-STRUCTURIZE MODEL PRELOAD PROCESS")
    logger.info("=" * 70)
    
    # Setup cache directories
    logger.info("\n📂 Setting up cache directories...")
    setup_cache_directories()
    
    # Preload the model
    logger.info("\n🤖 Preloading Granite Docling VLM model...")
    success = preload_docling_model()
    
    if success:
        logger.info("\n" + "=" * 70)
        logger.info("🎉 MODEL PRELOAD COMPLETED SUCCESSFULLY!")
        logger.info("=" * 70)
        logger.info("✅ The model is now cached and ready for instant use")
        logger.info("✅ Subsequent container starts will be nearly instantaneous")
        logger.info("=" * 70)
        sys.exit(0)
    else:
        logger.error("\n" + "=" * 70)
        logger.error("💥 MODEL PRELOAD FAILED!")
        logger.error("=" * 70)
        logger.error("❌ Check the logs above for error details")
        logger.error("❌ Common issues:")
        logger.error("   - Network connectivity problems")
        logger.error("   - Insufficient disk space")
        logger.error("   - Missing dependencies")
        logger.error("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()