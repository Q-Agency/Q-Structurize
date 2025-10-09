"""
Hybrid chunking service using Docling's native HybridChunker.

This module provides two chunking approaches:
1. Custom metadata extraction (clean, curated metadata)
2. Native serialization (full Docling metadata via model_dump())
"""

import logging
import time
from typing import List, Dict, Any, Optional, Set
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker import BaseChunk
from docling_core.types.doc.document import DoclingDocument

# Try alternative import path for newer Docling versions
try:
    from docling.chunking import HybridChunker as DoclingHybridChunker
    DOCLING_CHUNKING_AVAILABLE = True
except ImportError:
    DoclingHybridChunker = HybridChunker
    DOCLING_CHUNKING_AVAILABLE = False

logger = logging.getLogger(__name__)


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
    
    # Extract captions from tables and figures
    captions = []
    if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
        for item in chunk.meta.doc_items:
            if hasattr(item, "label") and item.label == "table":
                if hasattr(item, "captions") and item.captions:
                    captions.extend(item.captions)
    
    if captions:
        metadata["captions"] = captions
    
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
    tokenizer: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument using HybridChunker with modern features.
    
    This function uses Docling's native HybridChunker with built-in features like
    merge_peers for efficient chunk consolidation. It extracts rich metadata from
    each chunk for better retrieval and context understanding.
    
    Args:
        document: DoclingDocument to chunk
        max_tokens: Maximum tokens per chunk (default: 512)
        merge_peers: Whether to merge undersized successive chunks with same headings (default: True)
        tokenizer: Optional tokenizer (uses HybridChunker's built-in if None)
        
    Returns:
        List of chunk dictionaries with text, metadata, and section info
        
    Example:
        >>> chunks = chunk_document(document, max_tokens=1024, merge_peers=True)
        >>> print(f"Generated {len(chunks)} chunks")
        >>> print(chunks[0]['text'])
    """
    logger.info(f"Starting document chunking: max_tokens={max_tokens}, merge_peers={merge_peers}")
    start_time = time.time()
    
    # Initialize HybridChunker with modern parameters
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
    
    chunks = []
    
    # Process document with HybridChunker
    for chunk_idx, chunk in enumerate(chunker.chunk(document)):
        # Add contextualization to each chunk (adds heading context)
        contextualized_text = chunker.contextualize(chunk)
        
        # Add search prefix for optimal embedding (Nomic convention)
        # Even without Nomic tokenizer, this prefix helps with semantic search
        prefixed_text = f"search_document: {contextualized_text}"
        
        # Extract section title from headings (most specific heading)
        section_title = None
        if hasattr(chunk.meta, "headings") and chunk.meta.headings:
            section_title = chunk.meta.headings[-1]  # Use last (most specific) heading
        
        # Extract metadata
        metadata = extract_chunk_metadata(chunk)
        
        # Create chunk data dictionary
        chunk_data = {
            "text": prefixed_text,
            "section_title": section_title,
            "chunk_index": chunk_idx,
            "metadata": metadata
        }
        
        chunks.append(chunk_data)
        
        if chunk_idx % 10 == 0 and chunk_idx > 0:
            logger.debug(f"Processed {chunk_idx} chunks...")
    
    chunking_time = time.time() - start_time
    logger.info(
        f"Document chunking completed in {chunking_time:.2f}s. "
        f"Generated {len(chunks)} chunks (avg {len(chunks)/chunking_time:.1f} chunks/s)"
    )
    
    # Log statistics
    if chunks:
        avg_length = sum(len(c["text"]) for c in chunks) / len(chunks)
        logger.info(f"Chunk statistics: avg_length={avg_length:.0f} chars")
    
    return chunks


def chunk_document_native(
    document: DoclingDocument,
    max_tokens: int = 512,
    merge_peers: bool = True,
    tokenizer: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument using native serialization (model_dump).
    
    This is the SIMPLE approach that returns ALL native Docling metadata
    using Pydantic's model_dump() method. This gives you the full chunk
    structure as defined by Docling without custom filtering.
    
    Use this when you need:
    - Complete metadata from Docling
    - Native chunk structure
    - Maximum information preservation
    
    Args:
        document: DoclingDocument to chunk
        max_tokens: Maximum tokens per chunk (default: 512)
        merge_peers: Whether to merge undersized successive chunks with same headings (default: True)
        tokenizer: Optional tokenizer (uses HybridChunker's built-in if None)
        
    Returns:
        List of chunk dictionaries with full native metadata via model_dump()
        
    Example:
        >>> chunks = chunk_document_native(document, max_tokens=1024)
        >>> print(chunks[0].keys())  # All native Docling fields
    """
    logger.info(f"Native chunking: max_tokens={max_tokens}, merge_peers={merge_peers}")
    start_time = time.time()
    
    # Use the appropriate HybridChunker import
    ChunkerClass = DoclingHybridChunker if DOCLING_CHUNKING_AVAILABLE else HybridChunker
    
    # Initialize chunker
    chunker_params = {
        "max_tokens": max_tokens,
        "merge_peers": merge_peers,
    }
    
    if tokenizer is not None:
        chunker_params["tokenizer"] = tokenizer
        logger.info("Using provided tokenizer for native chunking")
    else:
        logger.info("Using HybridChunker's built-in tokenizer for native chunking")
    
    chunker = ChunkerClass(**chunker_params)
    logger.debug(f"Native HybridChunker initialized with params: {chunker_params}")
    
    chunks = []
    
    # Process document - try dl_doc parameter first, fall back to positional
    try:
        chunk_iterator = chunker.chunk(dl_doc=document)
    except TypeError:
        # Fallback for older versions that don't use dl_doc parameter
        chunk_iterator = chunker.chunk(document)
    
    for chunk_idx, chunk in enumerate(chunk_iterator):
        # Get ALL metadata automatically via model_dump()
        chunk_dict = chunk.model_dump()
        
        # Add contextualized text for embeddings
        contextualized_text = chunker.contextualize(chunk)
        chunk_dict['contextualized_text'] = contextualized_text
        
        # Add search prefix for optimal embedding (Nomic convention)
        chunk_dict['prefixed_text'] = f"search_document: {contextualized_text}"
        
        # Add chunk index
        chunk_dict['chunk_index'] = chunk_idx
        
        chunks.append(chunk_dict)
        
        if chunk_idx % 10 == 0 and chunk_idx > 0:
            logger.debug(f"Processed {chunk_idx} native chunks...")
    
    chunking_time = time.time() - start_time
    logger.info(
        f"Native chunking completed in {chunking_time:.2f}s. "
        f"Generated {len(chunks)} chunks (avg {len(chunks)/chunking_time:.1f} chunks/s)"
    )
    
    # Log statistics
    if chunks:
        avg_length = sum(len(c.get('prefixed_text', '')) for c in chunks) / len(chunks)
        logger.info(f"Native chunk statistics: avg_length={avg_length:.0f} chars")
        logger.info(f"Native chunk fields: {list(chunks[0].keys())[:10]}...")  # Show first 10 keys
    
    return chunks

