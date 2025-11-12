"""
Semantic Chunker Refinement Service using LlamaIndex.

This module provides post-processing refinement of Docling chunks using
LlamaIndex's SemanticSplitterNodeParser to improve chunk boundaries and
prevent overly large chunks that combine multiple sections.

ARCHITECTURE:
- Takes Docling chunks as input (from hybrid chunker output)
- Uses semantic similarity to identify better split points
- Preserves ALL metadata including bounding boxes from parent chunks
- Respects max_tokens constraint
- Returns chunks in the same format as hybrid chunker output

USAGE:
    refiner = SemanticChunkerRefiner(
        max_tokens=768,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
    refined_chunks = refiner.refine_chunks(chunks)
"""

import logging
import time
from typing import List, Dict, Any, Optional
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    'SemanticChunkerRefiner',
    'refine_chunks',
]


class SemanticChunkerRefiner:
    """
    Refines Docling chunks using LlamaIndex semantic chunking.
    
    This class applies semantic chunking as a post-processing step to improve
    chunk boundaries, especially for chunks that are too large or combine
    multiple semantic sections. All metadata from parent chunks is preserved.
    """
    
    def __init__(
        self,
        max_tokens: int = 768,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        breakpoint_percentile_threshold: float = 95,
        similarity_threshold: Optional[float] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize semantic chunker refiner.
        
        Args:
            max_tokens: Maximum tokens per refined chunk
            embedding_model: HuggingFace embedding model for semantic similarity
            breakpoint_percentile_threshold: Percentile threshold for breakpoint detection (default: 95)
            similarity_threshold: Optional fixed similarity threshold (overrides percentile)
            tokenizer: Optional tokenizer for counting tokens (uses embedding model's tokenizer if None)
        """
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        
        logger.info(f"Initializing SemanticChunkerRefiner: max_tokens={max_tokens}, embedding_model={embedding_model}")
        
        # Initialize embedding model
        try:
            import os
            cache_dir = os.environ.get('HF_CACHE_DIR', './cache/huggingface')
            self.embedding_model = HuggingFaceEmbedding(
                model_name=embedding_model,
                cache_folder=cache_dir
            )
            logger.info(f"✅ Embedding model loaded: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model '{embedding_model}': {str(e)}")
            raise RuntimeError(f"Failed to initialize embedding model: {str(e)}") from e
        
        # Initialize semantic splitter
        try:
            self.semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,  # Small buffer for fine-grained splitting
                breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                similarity_threshold=similarity_threshold,
                embed_model=self.embedding_model
            )
            logger.info(f"✅ SemanticSplitterNodeParser initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize SemanticSplitterNodeParser: {str(e)}")
            raise RuntimeError(f"Failed to initialize semantic splitter: {str(e)}") from e
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text using tokenizer or rough estimate.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4
    
    def refine_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Refine chunks using semantic chunking.
        
        Applies semantic refinement to ALL chunks from Docling. When a chunk
        is split, all metadata (including full_metadata with bounding boxes)
        is preserved in each child chunk.
        
        Args:
            chunks: List of chunk dictionaries from Docling hybrid chunker
            
        Returns:
            List of refined chunk dictionaries in the same format as input
        """
        logger.info(f"Starting semantic chunk refinement: {len(chunks)} input chunks")
        start_time = time.time()
        
        refined_chunks = []
        chunks_refined = 0
        chunks_split = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            # Extract text (remove "search_document: " prefix if present)
            text = chunk.get("text", "")
            if text.startswith("search_document: "):
                text = text[len("search_document: "):]
            
            # Skip if text is empty or too short
            if not text or len(text.strip()) < 10:
                refined_chunks.append(chunk)
                continue
            
            # Apply semantic chunking
            try:
                # Create LlamaIndex document with parent metadata
                llama_doc = LlamaDocument(
                    text=text,
                    metadata={
                        "original_chunk_index": chunk.get("chunk_index", chunk_idx),
                        "section_title": chunk.get("section_title"),
                        "original_metadata": chunk.get("metadata", {}),
                        "original_full_metadata": chunk.get("full_metadata")
                    }
                )
                
                # Split using semantic chunker
                nodes = self.semantic_splitter.get_nodes_from_documents([llama_doc])
                
                if len(nodes) == 1:
                    # No split occurred, keep original chunk
                    refined_chunks.append(chunk)
                else:
                    # Chunk was split into multiple nodes
                    chunks_refined += 1
                    chunks_split += len(nodes)
                    
                    # Create refined chunks from nodes
                    for node_idx, node in enumerate(nodes):
                        node_text = node.get_content()
                        
                        # Skip nodes that are too small
                        if len(node_text.strip()) < 10:
                            continue
                        
                        # Re-add "search_document: " prefix
                        prefixed_text = f"search_document: {node_text}"
                        
                        # Get original metadata from parent chunk
                        original_metadata = node.metadata.get("original_metadata", {})
                        original_full_metadata = node.metadata.get("original_full_metadata")
                        section_title = node.metadata.get("section_title")
                        
                        # Create refined chunk with ALL parent metadata preserved
                        refined_chunk = {
                            "text": prefixed_text,
                            "section_title": section_title,  # Preserve section title from parent
                            "chunk_index": len(refined_chunks),  # Sequential index
                            "metadata": {
                                **original_metadata,  # Preserve all curated metadata
                                "semantic_refined": True,
                                "original_chunk_index": node.metadata.get("original_chunk_index"),
                                "refined_chunk_index": node_idx,
                                "total_refined_chunks": len(nodes)
                            }
                        }
                        
                        # Preserve full_metadata (including bounding boxes) if present
                        if original_full_metadata is not None:
                            refined_chunk["full_metadata"] = original_full_metadata
                        
                        refined_chunks.append(refined_chunk)
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to refine chunk {chunk_idx}: {str(e)}, keeping original")
                refined_chunks.append(chunk)
        
        refinement_time = time.time() - start_time
        
        logger.info(
            f"✅ Semantic refinement completed in {refinement_time:.2f}s: "
            f"{len(chunks)} → {len(refined_chunks)} chunks "
            f"({chunks_refined} chunks refined, {chunks_split} total splits)"
        )
        
        return refined_chunks


def refine_chunks(
    chunks: List[Dict[str, Any]],
    max_tokens: int = 768,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    tokenizer: Optional[Any] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to refine chunks using semantic chunking.
    
    Args:
        chunks: List of chunk dictionaries from Docling
        max_tokens: Maximum tokens per refined chunk
        embedding_model: HuggingFace embedding model for semantic similarity
        tokenizer: Optional tokenizer for token counting
        **kwargs: Additional arguments passed to SemanticChunkerRefiner
        
    Returns:
        List of refined chunk dictionaries in the same format as input
    """
    refiner = SemanticChunkerRefiner(
        max_tokens=max_tokens,
        embedding_model=embedding_model,
        tokenizer=tokenizer,
        **kwargs
    )
    return refiner.refine_chunks(chunks)

