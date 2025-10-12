"""
Tokenizer Manager Service for managing HuggingFace tokenizers with caching.

This module provides a centralized service for loading and caching tokenizers
from HuggingFace models to be used with the HybridChunker for document chunking.

FEATURES:
- Load tokenizers from any HuggingFace model name
- LRU cache to avoid reloading tokenizers on each request
- Graceful error handling for invalid models or network issues
- Thread-safe caching implementation

USAGE:
    tokenizer_manager = TokenizerManager()
    tokenizer = tokenizer_manager.get_tokenizer("sentence-transformers/all-MiniLM-L6-v2")
"""

import logging
from typing import Optional, Any
from functools import lru_cache
import threading

logger = logging.getLogger(__name__)

# Thread lock for tokenizer loading
_tokenizer_lock = threading.Lock()


class TokenizerManager:
    """
    Manages loading and caching of HuggingFace tokenizers.
    
    Uses LRU cache to store up to 10 tokenizers in memory to avoid
    reloading the same tokenizer on subsequent requests.
    """
    
    def __init__(self, cache_size: int = 10):
        """
        Initialize TokenizerManager.
        
        Args:
            cache_size: Maximum number of tokenizers to cache (default: 10)
        """
        self.cache_size = cache_size
        self._load_tokenizer_cached = lru_cache(maxsize=cache_size)(self._load_tokenizer)
        logger.info(f"TokenizerManager initialized with cache_size={cache_size}")
    
    def _load_tokenizer(self, model_name: str) -> Any:
        """
        Load a tokenizer from HuggingFace (internal, cached method).
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Loaded tokenizer instance
            
        Raises:
            ValueError: If model name is invalid
            Exception: If tokenizer loading fails
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library is required for custom tokenizers. "
                "Install with: pip install transformers"
            )
        
        with _tokenizer_lock:
            logger.info(f"Loading tokenizer from HuggingFace: {model_name}")
            
            try:
                # Load tokenizer from HuggingFace
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir="/Users/zmatokanovic/development/QStructurize/cache/huggingface"
                )
                logger.info(f"Successfully loaded tokenizer: {model_name}")
                return tokenizer
                
            except Exception as e:
                logger.error(f"Failed to load tokenizer '{model_name}': {str(e)}")
                raise ValueError(
                    f"Failed to load tokenizer '{model_name}'. "
                    f"Error: {str(e)}. "
                    f"Please ensure the model name is correct and available on HuggingFace."
                )
    
    def get_tokenizer(self, model_name: str) -> Any:
        """
        Get a tokenizer by model name (cached).
        
        This method uses LRU cache to avoid reloading the same tokenizer.
        
        Args:
            model_name: HuggingFace model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
            
        Returns:
            Loaded tokenizer instance
            
        Raises:
            ValueError: If model name is invalid or empty
            Exception: If tokenizer loading fails
        """
        # Validate model name
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Model name must be a non-empty string")
        
        model_name = model_name.strip()
        
        if not model_name:
            raise ValueError("Model name cannot be empty or whitespace")
        
        if len(model_name) > 200:
            raise ValueError("Model name is too long (max 200 characters)")
        
        # Check for obviously invalid patterns
        if model_name.startswith('/') or model_name.endswith('/'):
            raise ValueError("Model name cannot start or end with '/'")
        
        logger.debug(f"Requesting tokenizer: {model_name}")
        
        # Load from cache or fetch new
        return self._load_tokenizer_cached(model_name)
    
    def clear_cache(self):
        """Clear the tokenizer cache."""
        self._load_tokenizer_cached.cache_clear()
        logger.info("Tokenizer cache cleared")
    
    def get_cache_info(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache hits, misses, and size
        """
        cache_info = self._load_tokenizer_cached.cache_info()
        return {
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "size": cache_info.currsize,
            "max_size": cache_info.maxsize
        }


# Global singleton instance
_tokenizer_manager: Optional[TokenizerManager] = None


def get_tokenizer_manager() -> TokenizerManager:
    """
    Get the global TokenizerManager singleton instance.
    
    Returns:
        TokenizerManager instance
    """
    global _tokenizer_manager
    if _tokenizer_manager is None:
        _tokenizer_manager = TokenizerManager()
    return _tokenizer_manager

