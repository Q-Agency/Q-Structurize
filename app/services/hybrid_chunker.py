"""
Hybrid chunking service using Docling's native HybridChunker.

This module provides flexible chunking with optional metadata levels:

FEATURES:
- Curated metadata extraction via extract_chunk_metadata()
- Optional complete Docling metadata via chunk.model_dump()
- Flexible output: choose between curated or full metadata
- Clean structure: {text, section_title, chunk_index, metadata, [full_metadata]}

USAGE:
- Default: Returns curated metadata for common use cases
- Full metadata: Set include_full_metadata=True for maximum information

ARCHITECTURE:
- Shared helpers for DRY code (_create_chunker, _process_chunk_text, _log_chunk_statistics)
- Single function handles both metadata approaches
- Clean separation between curated and complete metadata
"""

import logging
import time
from typing import List, Dict, Any, Optional, Set
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker import BaseChunk
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    'chunk_document',
    'extract_chunk_metadata',
]


def _create_chunker(
    max_tokens: int,
    merge_peers: bool,
    tokenizer: Optional[Any]
) -> HybridChunker:
    """
    Initialize HybridChunker with common parameters.
    
    Args:
        max_tokens: Maximum tokens per chunk
        merge_peers: Whether to merge undersized successive chunks with same headings
        tokenizer: Optional tokenizer (uses HybridChunker's built-in if None)
        
    Returns:
        Initialized HybridChunker instance
    """
    # Build chunker parameters
    chunker_params = {
        "max_tokens": max_tokens,
        "merge_peers": merge_peers,
    }
    
    # Only pass tokenizer if explicitly provided
    if tokenizer is not None:
        chunker_params["tokenizer"] = tokenizer
        logger.info("Using provided tokenizer")
    else:
        logger.info("Using HybridChunker's built-in tokenizer")
    
    chunker = HybridChunker(**chunker_params)
    logger.debug(f"HybridChunker initialized with params: {chunker_params}")
    
    return chunker


def _process_chunk_text(chunker: HybridChunker, chunk: BaseChunk) -> tuple[str, str]:
    contextualized_text = chunker.contextualize(chunk)
    prefixed_text = f"search_document: {contextualized_text}"
    return contextualized_text, prefixed_text


def _log_chunk_statistics(
    chunks: List[Dict[str, Any]],
    start_time: float,
    text_field: str = "text",
    is_native: bool = False
):
    """
    Log chunking statistics.
    
    Args:
        chunks: List of chunk dictionaries
        start_time: Start time of chunking process
        text_field: Field name containing the text to measure
        is_native: Whether this is native chunking (for log messages)
    """
    chunking_time = time.time() - start_time
    chunk_type = "Native chunking" if is_native else "Document chunking"
    
    logger.info(
        f"{chunk_type} completed in {chunking_time:.2f}s. "
        f"Generated {len(chunks)} chunks (avg {len(chunks)/chunking_time:.1f} chunks/s)"
    )
    
    # Log statistics
    if chunks:
        avg_length = sum(len(c.get(text_field, '')) for c in chunks) / len(chunks)
        logger.info(f"Chunk statistics: avg_length={avg_length:.0f} chars")
        
        # For native chunks, show available fields
        if is_native:
            logger.info(f"Native chunk fields: {list(chunks[0].keys())[:10]}...")


def extract_chunk_metadata(chunk: BaseChunk) -> Dict[str, Any]:
    """
    Extract metadata from a BaseChunk object.
    
    Args:
        chunk: BaseChunk object from HybridChunker
        
    Returns:
        Dictionary with metadata including content_type, heading_path, pages, etc.
    """
    metadata = {}
    
    # Extract content type from doc_items
    content_type = "text"  # default
    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
        types = set()
        for item in chunk.meta.doc_items:
            if hasattr(item, "label"):
                types.add(item.label)
        
        if "table" in types:
            content_type = "table"
        elif "list_item" in types:
            content_type = "list"
        elif "section_header" in types:
            content_type = "heading"
    
    metadata["content_type"] = content_type
    
    # Extract heading path (breadcrumb)
    if hasattr(chunk.meta, "headings") and chunk.meta.headings:
        metadata["heading_path"] = " > ".join(chunk.meta.headings)
    
    # Extract page numbers
    page_numbers: Set[int] = set()
    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
        for item in chunk.meta.doc_items:
            if hasattr(item, "prov"):
                for prov in item.prov:
                    if hasattr(prov, "page_no") and prov.page_no is not None:
                        page_numbers.add(prov.page_no)
    
    if page_numbers:
        metadata["pages"] = sorted(list(page_numbers))
    
    # Extract captions from tables and figures (commented out for now)
    # captions = []
    # if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
    #     for item in chunk.meta.doc_items:
    #         if hasattr(item, "label") and item.label == "table":
    #             if hasattr(item, "captions") and item.captions:
    #                 captions.extend(item.captions)
    # 
    # if captions:
    #     metadata["captions"] = captions
    
    # Extract table data if present (simplified - just mark as table)
    if content_type == "table":
        # Could extract detailed table structure here if needed
        metadata["has_table_structure"] = True
    
    # Count doc items for debugging
    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
        metadata["doc_items_count"] = len(chunk.meta.doc_items)
    
    return metadata


def chunk_document(
    document: DoclingDocument,
    max_tokens: int = 512,
    merge_peers: bool = True,
    tokenizer: Optional[Any] = None,
    include_full_metadata: bool = False
) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument with flexible metadata options.
    
    APPROACH:
    - Always includes curated metadata via extract_chunk_metadata()
    - Optionally includes complete Docling metadata via chunk.model_dump()
    
    METADATA OPTIONS:
    - Default (include_full_metadata=False):
      Returns: {text, section_title, chunk_index, metadata}
      - Clean, curated metadata fields
      - Simplified structure for common use cases
      
    - Full metadata (include_full_metadata=True):
      Returns: {text, section_title, chunk_index, metadata, full_metadata}
      - Includes all curated metadata PLUS complete Docling metadata
      - Maximum information preservation
    
    Args:
        document: DoclingDocument to chunk
        max_tokens: Maximum tokens per chunk (default: 512)
        merge_peers: Whether to merge undersized successive chunks with same headings (default: True)
        tokenizer: Optional tokenizer (uses HybridChunker's built-in if None)
        include_full_metadata: Include complete Docling metadata via model_dump() (default: False)
        
    Returns:
        List of chunk dictionaries with text, metadata, and optional full metadata
        
    Example:
        >>> # Standard chunking with curated metadata
        >>> chunks = chunk_document(document, max_tokens=1024)
        >>> print(chunks[0].keys())  # ['text', 'section_title', 'chunk_index', 'metadata']
        
        >>> # With complete Docling metadata
        >>> chunks = chunk_document(document, max_tokens=1024, include_full_metadata=True)
        >>> print(chunks[0].keys())  # Includes 'full_metadata' with all Docling fields
    """
    logger.info(f"Starting document chunking: max_tokens={max_tokens}, merge_peers={merge_peers}")
    start_time = time.time()
    
    # Initialize chunker (shared logic)
    chunker = _create_chunker(max_tokens, merge_peers, tokenizer)
    
    chunks = []
    
    # Process document with HybridChunker
    for chunk_idx, chunk in enumerate(chunker.chunk(document)):
        # Process text (shared logic)
        contextualized_text, prefixed_text = _process_chunk_text(chunker, chunk)
        
        # Extract section title from headings (most specific heading)
        section_title = None
        if hasattr(chunk.meta, "headings") and chunk.meta.headings:
            section_title = chunk.meta.headings[-1]  # Use last (most specific) heading
        
        # Extract curated metadata
        metadata = extract_chunk_metadata(chunk)
        
        # Create chunk data dictionary
        chunk_data = {
            "text": prefixed_text,
            "section_title": section_title,
            "chunk_index": chunk_idx,
            "metadata": metadata
        }
        
        # Optionally include complete Docling metadata
        if include_full_metadata:
            chunk_data["full_metadata"] = chunk.model_dump()
        
        chunks.append(chunk_data)
        
        if chunk_idx % 10 == 0 and chunk_idx > 0:
            logger.debug(f"Processed {chunk_idx} chunks...")
    
    # Log statistics (shared logic)
    _log_chunk_statistics(chunks, start_time, text_field="text", is_native=False)
    
    return chunks

