"""
Semantic Chunker Refinement Service using LlamaIndex.

This module provides post-processing refinement of Docling chunks using
LlamaIndex's SemanticSplitterNodeParser to improve chunk boundaries and
prevent overly large chunks that combine multiple sections.

ARCHITECTURE:
- Takes Docling chunks as input (from hybrid chunker output)
- Uses semantic similarity to identify better split points
- Special handling for numbered lists (splits by logical groupings)
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
import re
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
    
    Special handling for numbered lists ensures logical groupings without
    mid-item splits.
    """
    
    def __init__(
        self,
        max_tokens: int = 768,
        embedding_model: Optional[str] = None,
        breakpoint_percentile_threshold: float = 95,
        similarity_threshold: Optional[float] = None,
        tokenizer: Optional[Any] = None,
        max_list_items_per_chunk: int = 15,
        max_list_tokens_per_chunk: int = 350
    ):
        """
        Initialize semantic chunker refiner.
        
        Args:
            max_tokens: Maximum tokens per refined chunk
            embedding_model: Optional HuggingFace embedding model for semantic similarity.
                           If None, uses lightweight default 'sentence-transformers/all-MiniLM-L6-v2'.
                           Note: This is separate from the tokenizer model - semantic chunking only
                           needs to identify similarity breakpoints, not match your exact embedding model.
            breakpoint_percentile_threshold: Percentile threshold for breakpoint detection (default: 95)
            similarity_threshold: Optional fixed similarity threshold (overrides percentile)
            tokenizer: Optional tokenizer for counting tokens (uses embedding model's tokenizer if None)
            max_list_items_per_chunk: Maximum number of items in numbered lists per chunk
            max_list_tokens_per_chunk: Maximum tokens for numbered list chunks
        """
        # Validate parameters
        if max_tokens < 100 or max_tokens > 2000:
            raise ValueError(f"max_tokens must be between 100-2000, got {max_tokens}")
        
        if breakpoint_percentile_threshold < 50 or breakpoint_percentile_threshold > 99:
            raise ValueError(
                f"breakpoint_percentile_threshold must be 50-99, "
                f"got {breakpoint_percentile_threshold}"
            )
        
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.max_list_items_per_chunk = max_list_items_per_chunk
        self.max_list_tokens_per_chunk = max_list_tokens_per_chunk
        
        # Use lightweight default model if not specified (avoids loading heavy models twice)
        if embedding_model is None:
            embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            logger.info(f"Using lightweight default embedding model for semantic chunking: {embedding_model}")
        
        logger.info(f"Initializing SemanticChunkerRefiner: max_tokens={max_tokens}, embedding_model={embedding_model}")
        
        # Initialize embedding model
        try:
            import os
            cache_dir = os.environ.get('HF_CACHE_DIR', './cache/huggingface')
            # Only use trust_remote_code for models that need it (like nomic)
            use_trust_remote_code = "nomic" in embedding_model.lower() or "bert-2048" in embedding_model.lower()
            self.embedding_model = HuggingFaceEmbedding(
                model_name=embedding_model,
                cache_folder=cache_dir,
                trust_remote_code=use_trust_remote_code
            )
            logger.info(f"✅ Embedding model loaded: {embedding_model}")
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model '{embedding_model}': {str(e)}")
            # If the specified model fails, try falling back to lightweight default
            if embedding_model != "sentence-transformers/all-MiniLM-L6-v2":
                logger.warning(f"⚠️  Falling back to lightweight default model")
                try:
                    self.embedding_model = HuggingFaceEmbedding(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        cache_folder=cache_dir
                    )
                    logger.info(f"✅ Fallback embedding model loaded: sentence-transformers/all-MiniLM-L6-v2")
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback model also failed: {str(fallback_error)}")
                    raise RuntimeError(f"Failed to initialize embedding model: {str(e)}") from e
            else:
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
    
    def _is_numbered_list(self, text: str) -> bool:
        """
        Detect if text is a numbered list.
        
        Args:
            text: Text to check
            
        Returns:
            True if text contains a numbered list with at least 3 items
        """
        # Look for pattern: multiple lines starting with numbers
        numbered_lines = re.findall(r'^\d+\.\s+', text, re.MULTILINE)
        return len(numbered_lines) >= 3  # At least 3 numbered items
    
    def _split_numbered_list(
        self,
        text: str,
        max_items_per_chunk: Optional[int] = None,
        max_tokens_per_chunk: Optional[int] = None
    ) -> List[str]:
        """
        Split numbered list by logical groupings.
        
        Strategy:
        1. Split every N items (max_items_per_chunk)
        2. Respect token limit (max_tokens_per_chunk)
        3. Keep complete items together (don't split mid-item)
        
        Args:
            text: Text containing numbered list
            max_items_per_chunk: Maximum items per chunk (uses instance default if None)
            max_tokens_per_chunk: Maximum tokens per chunk (uses instance default if None)
            
        Returns:
            List of chunk texts
        """
        if max_items_per_chunk is None:
            max_items_per_chunk = self.max_list_items_per_chunk
        if max_tokens_per_chunk is None:
            max_tokens_per_chunk = self.max_list_tokens_per_chunk
        
        # Split text into parts at numbered item boundaries
        parts = re.split(r'(\n\d+\.\s+)', text)
        
        # Extract header (text before first numbered item)
        header = ""
        numbered_items = []
        
        i = 0
        while i < len(parts):
            part = parts[i]
            
            # Check if this is the header (before any numbered items)
            if i == 0 and not re.match(r'\n?\d+\.\s+', part):
                header = part.strip()
                i += 1
                continue
            
            # Check if this is a number prefix
            if re.match(r'\n?\d+\.\s+', part):
                # Combine number with its content
                number = part.strip()
                content = ""
                if i + 1 < len(parts):
                    content = parts[i + 1].strip()
                    i += 2
                else:
                    i += 1
                
                # Only add if there's actual content
                if content:
                    numbered_items.append(f"{number} {content}")
            else:
                i += 1
        
        if not numbered_items:
            # Fallback if parsing fails
            logger.warning("⚠️  Failed to parse numbered list, returning original text")
            return [text]
        
        logger.info(f"📋 Parsed numbered list: {len(numbered_items)} items, header: '{header[:50]}...'")
        
        # Split into chunks based on item count and token limits
        chunks = []
        current_chunk_items = []
        current_token_count = self._count_tokens(header)
        
        for idx, item in enumerate(numbered_items):
            item_tokens = self._count_tokens(item)
            
            # Check if adding this item would exceed limits
            would_exceed_items = len(current_chunk_items) >= max_items_per_chunk
            would_exceed_tokens = (current_token_count + item_tokens) > max_tokens_per_chunk
            
            if current_chunk_items and (would_exceed_items or would_exceed_tokens):
                # Finalize current chunk
                chunk_text = header + "\n" + "\n".join(current_chunk_items)
                chunks.append(chunk_text.strip())
                
                # Start new chunk
                current_chunk_items = [item]
                current_token_count = self._count_tokens(header) + item_tokens
            else:
                # Add to current chunk
                current_chunk_items.append(item)
                current_token_count += item_tokens
        
        # Add final chunk
        if current_chunk_items:
            chunk_text = header + "\n" + "\n".join(current_chunk_items)
            chunks.append(chunk_text.strip())
        
        logger.info(f"📋 Split numbered list into {len(chunks)} chunks")
        return chunks
    
    def refine_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Refine chunks using semantic chunking.
        
        Applies semantic refinement to ALL chunks from Docling. When a chunk
        is split, all metadata (including full_metadata with bounding boxes)
        is preserved in each child chunk.
        
        Special handling for numbered lists ensures logical groupings without
        mid-item splits.
        
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
        list_chunks_processed = 0
        token_counts = []
        
        for chunk_idx, chunk in enumerate(chunks):
            # Extract text (remove "search_document: " prefix if present)
            text = chunk.get("text", "")
            if text.startswith("search_document: "):
                text = text[len("search_document: "):]
            
            # Skip if text is empty or too short
            if not text or len(text.strip()) < 10:
                refined_chunks.append(chunk)
                continue
            
            # Check if this is a numbered list
            is_list = self._is_numbered_list(text)
            
            if is_list:
                logger.info(f"📋 Chunk {chunk_idx} detected as numbered list, using list-aware splitting")
                list_chunks_processed += 1
                
                try:
                    # Use list-aware splitting
                    list_chunk_texts = self._split_numbered_list(text)
                    
                    if len(list_chunk_texts) == 1:
                        # No split needed
                        refined_chunks.append(chunk)
                        token_counts.append(self._count_tokens(text))
                    else:
                        # Create refined chunks
                        chunks_refined += 1
                        chunks_split += len(list_chunk_texts)
                        
                        for node_idx, chunk_text in enumerate(list_chunk_texts):
                            # Re-add "search_document: " prefix
                            prefixed_text = f"search_document: {chunk_text}"
                            
                            # Track token count
                            token_count = self._count_tokens(chunk_text)
                            token_counts.append(token_count)
                            
                            # Create refined chunk with ALL parent metadata preserved
                            refined_chunk = {
                                "text": prefixed_text,
                                "section_title": chunk.get("section_title"),
                                "chunk_index": len(refined_chunks),
                                "metadata": {
                                    **chunk.get("metadata", {}),
                                    "list_refined": True,
                                    "original_chunk_index": chunk.get("chunk_index", chunk_idx),
                                    "refined_chunk_index": node_idx,
                                    "total_refined_chunks": len(list_chunk_texts),
                                    "token_count": token_count
                                }
                            }
                            
                            # Flag if still needs LLM subchunking
                            if token_count > self.max_tokens:
                                refined_chunk["metadata"]["needs_llm_subchunking"] = True
                            
                            # Preserve full_metadata (including bounding boxes) if present
                            if chunk.get("full_metadata") is not None:
                                refined_chunk["full_metadata"] = chunk["full_metadata"]
                            
                            refined_chunks.append(refined_chunk)
                    
                    continue  # Skip semantic chunking for lists
                    
                except Exception as e:
                    logger.warning(
                        f"⚠️  List-aware splitting failed for chunk {chunk_idx}: {str(e)}, "
                        f"falling back to semantic chunking"
                    )
                    # Fall through to semantic chunking
            
            # Apply semantic chunking (for non-lists or if list splitting failed)
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
                    token_counts.append(self._count_tokens(text))
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
                        
                        # Track token count
                        token_count = self._count_tokens(node_text)
                        token_counts.append(token_count)
                        
                        # Get original metadata from parent chunk
                        original_metadata = node.metadata.get("original_metadata", {})
                        original_full_metadata = node.metadata.get("original_full_metadata")
                        section_title = node.metadata.get("section_title")
                        
                        # Create refined chunk with ALL parent metadata preserved
                        refined_chunk = {
                            "text": prefixed_text,
                            "section_title": section_title,
                            "chunk_index": len(refined_chunks),
                            "metadata": {
                                **original_metadata,
                                "semantic_refined": True,
                                "original_chunk_index": node.metadata.get("original_chunk_index"),
                                "refined_chunk_index": node_idx,
                                "total_refined_chunks": len(nodes),
                                "token_count": token_count
                            }
                        }
                        
                        # Flag if still needs LLM subchunking
                        if token_count > self.max_tokens:
                            refined_chunk["metadata"]["needs_llm_subchunking"] = True
                        
                        # Preserve full_metadata (including bounding boxes) if present
                        if original_full_metadata is not None:
                            refined_chunk["full_metadata"] = original_full_metadata
                        
                        refined_chunks.append(refined_chunk)
                
            except Exception as e:
                logger.warning(f"⚠️  Failed to refine chunk {chunk_idx}: {str(e)}, keeping original")
                refined_chunks.append(chunk)
                token_counts.append(self._count_tokens(text))
        
        refinement_time = time.time() - start_time
        
        # Log statistics
        logger.info(
            f"✅ Semantic refinement completed in {refinement_time:.2f}s: "
            f"{len(chunks)} → {len(refined_chunks)} chunks "
            f"({chunks_refined} chunks refined, {chunks_split} total splits, "
            f"{list_chunks_processed} lists processed)"
        )
        
        # Log token distribution
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            max_tokens_found = max(token_counts)
            min_tokens_found = min(token_counts)
            logger.info(
                f"📊 Chunk size stats: avg={avg_tokens:.0f}, "
                f"min={min_tokens_found}, max={max_tokens_found} tokens"
            )
            
            # Count how many chunks exceed target
            oversized = sum(1 for t in token_counts if t > self.max_tokens)
            if oversized > 0:
                logger.warning(
                    f"⚠️  {oversized}/{len(token_counts)} chunks exceed max_tokens ({self.max_tokens}), "
                    f"flagged for LLM subchunking"
                )
        
        return refined_chunks


def refine_chunks(
    chunks: List[Dict[str, Any]],
    max_tokens: int = 768,
    embedding_model: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to refine chunks using semantic chunking.
    
    Args:
        chunks: List of chunk dictionaries from Docling
        max_tokens: Maximum tokens per refined chunk
        embedding_model: Optional HuggingFace embedding model for semantic similarity.
                        If None, uses lightweight default. Note: This is separate from
                        the tokenizer model - semantic chunking only needs similarity detection.
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