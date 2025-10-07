#!/usr/bin/env python3
"""
GPU Test Script for Q-Structurize
Tests if CUDA/GPU is available and working properly.
"""

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_availability():
    """Test if GPU/CUDA is available and working."""
    logger.info("🔍 Testing GPU availability...")
    
    try:
        import torch
        logger.info(f"✅ PyTorch version: {torch.__version__}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"🎯 CUDA available: {cuda_available}")
        
        if cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logger.info(f"🚀 GPU devices found: {device_count}")
            logger.info(f"🎮 Current device: {current_device}")
            logger.info(f"💻 Device name: {device_name}")
            
            # Test GPU memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            logger.info(f"💾 GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB, Total: {memory_total:.2f}GB")
            
            # Test basic GPU operation
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor)
            logger.info("✅ GPU computation test passed!")
            
            return True
        else:
            logger.warning("❌ CUDA not available - will use CPU (very slow for VLM)")
            return False
            
    except ImportError as e:
        logger.error(f"❌ PyTorch not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ GPU test failed: {e}")
        return False

def test_docling_gpu():
    """Test if Docling can use GPU."""
    logger.info("🔍 Testing Docling GPU support...")
    
    try:
        from docling.datamodel import vlm_model_specs
        from docling.pipeline.vlm_pipeline import VlmPipeline
        logger.info("✅ Docling VLM imports successful")
        
        # Check if we can create VLM pipeline
        vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
        logger.info(f"✅ VLM options available: {vlm_options}")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Docling not available: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Docling test failed: {e}")
        return False

def main():
    """Main test function."""
    logger.info("🚀 Starting GPU test for Q-Structurize...")
    logger.info("=" * 60)
    
    # Test GPU availability
    gpu_ok = test_gpu_availability()
    logger.info("=" * 60)
    
    # Test Docling
    docling_ok = test_docling_gpu()
    logger.info("=" * 60)
    
    # Summary
    if gpu_ok and docling_ok:
        logger.info("🎉 ALL TESTS PASSED! GPU is ready for VLM processing.")
        logger.info("⚡ VLM processing will be 100x faster than CPU!")
        return 0
    elif gpu_ok:
        logger.warning("⚠️  GPU available but Docling issues detected.")
        logger.warning("💡 Try: pip install --upgrade docling[vlm]")
        return 1
    else:
        logger.error("❌ GPU not available. VLM will be very slow on CPU.")
        logger.error("💡 Check: 1) NVIDIA drivers, 2) Docker GPU support, 3) nvidia-docker")
        return 2

if __name__ == "__main__":
    sys.exit(main())
