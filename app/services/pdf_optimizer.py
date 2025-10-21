"""PDF optimization service for better text extraction."""

import os
import time
import tempfile
import logging
from typing import Optional

# Try to import pikepdf, set availability flag
try:
    import pikepdf
    from pikepdf import Pdf
    PDF_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PDF_OPTIMIZATION_AVAILABLE = False
    pikepdf = None
    Pdf = None

logger = logging.getLogger(__name__)


class PDFOptimizer:
    """Service for optimizing PDF files for better text extraction."""
    
    @staticmethod
    def optimize_pdf(binary_data: bytes) -> tuple[bytes, dict]:
        """
        Optimize PDF for better text extraction.
        
        Args:
            binary_data: Raw PDF file content as bytes
            
        Returns:
            Tuple of (optimized PDF content as bytes, size information dict)
        """
        # Track original size
        original_size = len(binary_data)
        
        if not PDF_OPTIMIZATION_AVAILABLE:
            logger.debug("pikepdf not available, skipping optimization")
            size_info = {
                "original_size_bytes": original_size,
                "optimized_size_bytes": original_size,
                "size_reduction_bytes": 0,
                "size_reduction_percentage": 0.0
            }
            return binary_data, size_info
        
        start_time = time.time()
        
        try:
            # Create temporary files for input and output
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_in, \
                 tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_out:
                
                # Write the input binary data to the temp file
                tmp_in.write(binary_data)
                tmp_in.flush()
                
                # Open the PDF with pikepdf
                with Pdf.open(tmp_in.name) as pdf:
                    # Remove unnecessary metadata that might interfere with parsing
                    pdf.remove_unreferenced_resources()
                    
                    # Save with optimization options
                    pdf.save(
                        tmp_out.name,
                        linearize=False,  # Avoid conflict with normalize_content
                        object_stream_mode=pikepdf.ObjectStreamMode.generate,
                        compress_streams=True,
                        normalize_content=True,
                        qdf=False
                    )
                
                # Read the optimized PDF
                with open(tmp_out.name, 'rb') as f:
                    optimized_data = f.read()
                
                # Calculate size information
                optimized_size = len(optimized_data)
                size_reduction_bytes = original_size - optimized_size
                size_reduction_percentage = (size_reduction_bytes / original_size * 100) if original_size > 0 else 0.0
                
                size_info = {
                    "original_size_bytes": original_size,
                    "optimized_size_bytes": optimized_size,
                    "size_reduction_bytes": size_reduction_bytes,
                    "size_reduction_percentage": round(size_reduction_percentage, 2)
                }
                
                # Clean up temporary files
                os.unlink(tmp_in.name)
                os.unlink(tmp_out.name)
                
                optimization_time = time.time() - start_time
                logger.info(f"✅ PDF optimized in {optimization_time:.2f}s: {original_size:,} → {optimized_size:,} bytes ({size_reduction_percentage:.1f}% reduction)")
                return optimized_data, size_info
                
        except Exception as e:
            logger.warning(f"⚠️  PDF optimization failed: {str(e)}, using original")
            size_info = {
                "original_size_bytes": original_size,
                "optimized_size_bytes": original_size,
                "size_reduction_bytes": 0,
                "size_reduction_percentage": 0.0,
                "error": str(e)
            }
            return binary_data, size_info
    
    @staticmethod
    def is_optimization_available() -> bool:
        """
        Check if PDF optimization is available.
        
        Returns:
            True if pikepdf is available, False otherwise
        """
        return PDF_OPTIMIZATION_AVAILABLE
    
    @staticmethod
    def get_optimization_info() -> dict:
        """
        Get information about PDF optimization capabilities.
        
        Returns:
            Dictionary with optimization status and details
        """
        return {
            "available": PDF_OPTIMIZATION_AVAILABLE,
            "library": "pikepdf" if PDF_OPTIMIZATION_AVAILABLE else None,
            "description": "PDF optimization for better text extraction"
        }
