#!/usr/bin/env python3
"""
Test script to verify model preload works correctly.
"""

import os
import sys
import time
import logging

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_preload():
    """Test if the model preload worked correctly."""
    try:
        logger.info("🧪 Testing model preload...")
        
        # Import the docling parser
        from app.services.docling_parser import DoclingParser
        
        # Initialize parser
        logger.info("Initializing DoclingParser...")
        start_time = time.time()
        
        parser = DoclingParser()
        
        init_time = time.time() - start_time
        logger.info(f"✅ DoclingParser initialized in {init_time:.2f} seconds")
        
        # Check if parser is available
        if parser.is_available():
            logger.info("✅ Parser is available and ready")
            logger.info(f"Parser info: {parser.get_parser_info()}")
        else:
            logger.error("❌ Parser is not available")
            return False
            
        # Test with a simple PDF (if available)
        test_pdf_path = os.path.join(os.path.dirname(__file__), '..', 'test.txt')
        if os.path.exists(test_pdf_path):
            logger.info("📄 Test file found, but skipping PDF test (requires actual PDF)")
        else:
            logger.info("📄 No test PDF found, skipping PDF processing test")
        
        logger.info("🎉 Model preload test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model preload test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_preload()
    sys.exit(0 if success else 1)
