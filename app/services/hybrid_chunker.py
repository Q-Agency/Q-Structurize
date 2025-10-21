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
from app.services.table_serializer import extract_tables_from_document

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
        # Try to get tokenizer info for logging
        tokenizer_info = "custom tokenizer"
        if hasattr(tokenizer, 'name_or_path'):
            tokenizer_info = f"custom tokenizer (model: {tokenizer.name_or_path})"
        elif hasattr(tokenizer, '__class__'):
            tokenizer_info = f"custom tokenizer (type: {tokenizer.__class__.__name__})"
        logger.info(f"Using {tokenizer_info}")
    else:
        logger.info("Using HybridChunker's built-in tokenizer")
    
    chunker = HybridChunker(**chunker_params)
    logger.debug(f"HybridChunker initialized with max_tokens={max_tokens}, merge_peers={merge_peers}")
    
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
    include_full_metadata: bool = False,
    serialize_tables: bool = False
) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument with flexible metadata options.
    
    APPROACH:
    - Always includes curated metadata via extract_chunk_metadata()
    - Optionally includes complete Docling metadata via chunk.model_dump()
    - Optionally serializes table chunks into embedding-optimized format
    
    METADATA OPTIONS:
    - Default (include_full_metadata=False):
      Returns: {text, section_title, chunk_index, metadata}
      - Clean, curated metadata fields
      - Simplified structure for common use cases
      
    - Full metadata (include_full_metadata=True):
      Returns: {text, section_title, chunk_index, metadata, full_metadata}
      - Includes all curated metadata PLUS complete Docling metadata
      - Maximum information preservation
    
    TABLE SERIALIZATION:
    - When serialize_tables=True, table chunks are reformatted as key-value pairs
    - Format: "Column1: Value1, Column2: Value2, ..."
    - Includes table caption as prefix (if available)
    - Optimized for embedding models and semantic search
    
    Args:
        document: DoclingDocument to chunk
        max_tokens: Maximum tokens per chunk (default: 512)
        merge_peers: Whether to merge undersized successive chunks with same headings (default: True)
        tokenizer: Optional tokenizer (uses HybridChunker's built-in if None)
        include_full_metadata: Include complete Docling metadata via model_dump() (default: False)
        serialize_tables: Serialize table chunks as key-value pairs for embeddings (default: False)
        
    Returns:
        List of chunk dictionaries with text, metadata, and optional full metadata
        
    Example:
        >>> # Standard chunking with curated metadata
        >>> chunks = chunk_document(document, max_tokens=1024)
        >>> print(chunks[0].keys())  # ['text', 'section_title', 'chunk_index', 'metadata']
        
        >>> # With complete Docling metadata
        >>> chunks = chunk_document(document, max_tokens=1024, include_full_metadata=True)
        >>> print(chunks[0].keys())  # Includes 'full_metadata' with all Docling fields
        
        >>> # With table serialization for embeddings
        >>> chunks = chunk_document(document, max_tokens=1024, serialize_tables=True)
        >>> # Table chunks will have key-value format text
    """
    logger.info(f"Starting document chunking: max_tokens={max_tokens}, merge_peers={merge_peers}, serialize_tables={serialize_tables}")
    start_time = time.time()
    
    # Extract tables from document BEFORE chunking if serialization is enabled
    extracted_tables = []
    table_texts = {}  # Map from table text to serialized version
    
    if serialize_tables:
        logger.info("Extracting tables from document structure...")
        table_extract_start = time.time()
        extracted_tables = extract_tables_from_document(document)
        table_extract_time = time.time() - table_extract_start
        
        logger.info(f"Extracted {len(extracted_tables)} tables in {table_extract_time:.2f}s")
        
        # Build mapping from table item to serialized text for later matching
        for table_info in extracted_tables:
            item = table_info['item']
            serialized = table_info['serialized_text']
            
            # Get original text representation to use as key
            if hasattr(item, 'text'):
                orig_text = item.text
                # Store first 200 chars as key (enough to match)
                table_texts[orig_text[:200]] = serialized
            
            logger.debug(f"Table mapping created: {table_info.get('caption', 'no caption')}")
    
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
        
        # Handle table serialization if enabled
        final_text = prefixed_text
        if serialize_tables and metadata.get("content_type") == "table":
            # Try to match this chunk to a pre-extracted table
            matched = False
            
            # Get chunk's raw text for matching
            chunk_raw_text = chunk.text if hasattr(chunk, 'text') else contextualized_text
            
            # Try to find matching serialized table
            for text_key, serialized in table_texts.items():
                # Check if this chunk contains part of the table
                if text_key in chunk_raw_text or chunk_raw_text[:200] in text_key:
                    final_text = f"search_document: {serialized}"
                    matched = True
                    logger.debug(f"Matched and serialized table chunk {chunk_idx}")
                    break
            
            if not matched:
                logger.debug(f"Table chunk {chunk_idx} could not be matched to extracted tables")
        
        # Create chunk data dictionary
        chunk_data = {
            "text": final_text,
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

