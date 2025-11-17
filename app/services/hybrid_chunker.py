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
from app.services.table_serializer import serialize_table_from_chunk
from app.services.semantic_chunker_refiner import refine_chunks as semantic_refine_chunks
import re

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
        logger.debug(f"Using {tokenizer_info}")
    else:
        logger.debug("Using HybridChunker's built-in tokenizer")
    
    chunker = HybridChunker(**chunker_params)
    logger.debug(f"HybridChunker initialized with max_tokens={max_tokens}, merge_peers={merge_peers}")
    
    return chunker


def _process_chunk_text(chunker: HybridChunker, chunk: BaseChunk) -> tuple[str, str]:
    contextualized_text = chunker.contextualize(chunk)
    prefixed_text = f"search_document: {contextualized_text}"
    return contextualized_text, prefixed_text


def _preserve_image_descriptions(chunks: List[Dict[str, Any]], document: DoclingDocument) -> List[Dict[str, Any]]:
    """
    Post-process chunks to ensure complete image descriptions are preserved.
    
    This is a fallback function that checks for truncated descriptions that might
    have been missed during chunk processing. The main preservation happens during
    chunk creation, but this provides an additional safety check.
    
    Args:
        chunks: List of chunk dictionaries
        document: DoclingDocument containing image descriptions
        
    Returns:
        List of chunks with preserved image descriptions
    """
    try:
        # Import PictureItem only if available
        try:
            from docling_core.types.doc import PictureItem
        except ImportError:
            # If PictureItem not available, return chunks as-is
            return chunks
        
        # Build a map of image descriptions from the document
        image_descriptions = {}
        for element, _level in document.iterate_items():
            if isinstance(element, PictureItem):
                image_ref = getattr(element, 'self_ref', None)
                if image_ref:
                    # Get description/caption
                    desc_text = None
                    try:
                        if hasattr(element, 'caption_text'):
                            caption = element.caption_text(doc=document)
                            if caption:
                                desc_text = caption
                    except Exception:
                        pass
                    
                    # Extract from annotations if no caption
                    if not desc_text:
                        annotations = getattr(element, 'annotations', [])
                        for ann in annotations:
                            if isinstance(ann, str):
                                desc_text = ann
                                break
                            else:
                                for attr in ['text', 'content', 'description', 'value', 'annotation']:
                                    if hasattr(ann, attr):
                                        text = getattr(ann, attr)
                                        if isinstance(text, str) and text.strip():
                                            desc_text = text
                                            break
                                if desc_text:
                                    break
                    
                    if desc_text:
                        image_descriptions[str(image_ref)] = desc_text
        
        if not image_descriptions:
            return chunks
        
        # Process each chunk to ensure image descriptions are complete (fallback check)
        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue
            
            # For each image description, check if it appears truncated in the chunk
            for image_ref, full_description in image_descriptions.items():
                # Use a shorter prefix (first 50 chars) to catch more cases
                prefix_len = min(50, len(full_description))
                description_prefix = full_description[:prefix_len].lower().strip()
                
                # Skip if prefix is too short (might cause false matches)
                if len(description_prefix) < 20:
                    continue
                
                # Check if the prefix appears in the chunk (case-insensitive)
                chunk_lower = chunk_text.lower()
                if description_prefix in chunk_lower:
                    # Check if the full description is present (case-insensitive)
                    full_desc_lower = full_description.lower()
                    if full_desc_lower not in chunk_lower:
                        # Description is present but truncated - find where it starts
                        desc_start_idx = chunk_lower.find(description_prefix)
                        if desc_start_idx >= 0:
                            # Get the remaining text from where description starts
                            remaining_in_chunk = chunk_text[desc_start_idx:]
                            
                            # More aggressive detection: if remaining text is shorter than full description
                            # OR if chunk ends abruptly (doesn't end with sentence-ending punctuation)
                            chunk_ends_abruptly = not chunk_text.rstrip().endswith(('.', '!', '?', '。', '」'))
                            
                            # Check if truncated (use 90% threshold to be more aggressive)
                            is_truncated = (
                                len(remaining_in_chunk) < len(full_description) * 0.9 or
                                (chunk_ends_abruptly and len(remaining_in_chunk) < len(full_description))
                            )
                            
                            if is_truncated:
                                # Likely truncated - replace with full description
                                # Keep text before description
                                text_before = chunk_text[:desc_start_idx].rstrip()
                                
                                # Handle "search_document:" prefix
                                if text_before.startswith("search_document:"):
                                    prefix = "search_document:"
                                    text_after_prefix = text_before[len(prefix):].strip()
                                    if text_after_prefix:
                                        chunk["text"] = f"{prefix} {text_after_prefix}\n\n{full_description}"
                                    else:
                                        chunk["text"] = f"{prefix} {full_description}"
                                else:
                                    # Append full description
                                    if text_before:
                                        chunk["text"] = f"{text_before}\n\n{full_description}"
                                    else:
                                        chunk["text"] = full_description
                                
                                logger.info(f"🔧 Preserved complete image description for {image_ref} in chunk {chunk.get('chunk_index', '?')} (was {len(remaining_in_chunk)} chars, now {len(full_description)} chars)")
                                break  # Only process one description per chunk to avoid conflicts
        
        return chunks
    except Exception as e:
        logger.warning(f"⚠️  Could not preserve image descriptions: {str(e)}")
        return chunks


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
    serialize_tables: bool = False,
    semantic_refinement: bool = False
) -> List[Dict[str, Any]]:
    """
    Chunk a DoclingDocument with flexible metadata options.
    
    APPROACH:
    - Always includes curated metadata via extract_chunk_metadata()
    - Optionally includes complete Docling metadata via chunk.model_dump()
    - Optionally serializes table chunks into embedding-optimized format
    - Optionally applies semantic chunking refinement using LlamaIndex
    
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
    
    SEMANTIC REFINEMENT:
    - When semantic_refinement=True, applies LlamaIndex semantic chunking as post-processing
    - Improves chunk boundaries for chunks that combine multiple sections
    - Uses semantic similarity to identify better split points
    - Preserves ALL metadata (including full_metadata with bounding boxes) from parent chunks
    - Requires embedding_model to be specified (extracted from tokenizer.name_or_path)
    - Applied to ALL chunks from Docling
    
    Args:
        document: DoclingDocument to chunk
        max_tokens: Maximum tokens per chunk (default: 512)
        merge_peers: Whether to merge undersized successive chunks with same headings (default: True)
        tokenizer: Optional tokenizer (uses HybridChunker's built-in if None)
        include_full_metadata: Include complete Docling metadata via model_dump() (default: False)
        serialize_tables: Serialize table chunks as key-value pairs for embeddings (default: False)
        semantic_refinement: Apply semantic chunking refinement using LlamaIndex (default: False)
        
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
        
        >>> # With semantic refinement (requires embedding_model parameter in API)
        >>> chunks = chunk_document(document, max_tokens=1024, semantic_refinement=True, tokenizer=tokenizer)
        >>> # Chunks are refined using semantic similarity
    """
    logger.info(f"Starting document chunking: max_tokens={max_tokens}, merge_peers={merge_peers}, serialize_tables={serialize_tables}, semantic_refinement={semantic_refinement}")
    start_time = time.time()
    
    # Initialize chunker (shared logic)
    chunker = _create_chunker(max_tokens, merge_peers, tokenizer)
    
    chunks = []
    tables_serialized = 0
    tables_failed = 0
    
    # Build image descriptions map first (needed for both processing and preservation)
    image_descriptions_map = {}
    try:
        from docling_core.types.doc import PictureItem
        for element, _level in document.iterate_items():
            if isinstance(element, PictureItem):
                image_ref = getattr(element, 'self_ref', None)
                if image_ref:
                    # Get description/caption
                    desc_text = None
                    try:
                        if hasattr(element, 'caption_text'):
                            caption = element.caption_text(doc=document)
                            if caption:
                                desc_text = caption
                    except Exception:
                        pass
                    
                    # Extract from annotations if no caption
                    if not desc_text:
                        annotations = getattr(element, 'annotations', [])
                        for ann in annotations:
                            if isinstance(ann, str):
                                desc_text = ann
                                break
                            else:
                                for attr in ['text', 'content', 'description', 'value', 'annotation']:
                                    if hasattr(ann, attr):
                                        text = getattr(ann, attr)
                                        if isinstance(text, str) and text.strip():
                                            desc_text = text
                                            break
                                if desc_text:
                                    break
                    
                    if desc_text:
                        image_descriptions_map[str(image_ref)] = desc_text
    except Exception:
        pass  # PictureItem not available, continue without image description preservation
    
    # Process document with HybridChunker
    for chunk_idx, chunk in enumerate(chunker.chunk(document)):
        # Process text (shared logic)
        contextualized_text, prefixed_text = _process_chunk_text(chunker, chunk)
        
        # Check if this chunk contains any PictureItems and ensure their descriptions are complete
        # First, check doc_items for picture references
        chunk_has_image = False
        image_refs_in_chunk = []
        if image_descriptions_map and hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
            for item in chunk.meta.doc_items:
                # Check if item is a picture by label
                if hasattr(item, 'label') and item.label == 'picture':
                    chunk_has_image = True
                    # Try to get image reference from item
                    image_ref = None
                    if hasattr(item, 'self_ref'):
                        image_ref = str(item.self_ref)
                    elif hasattr(item, 'ref'):
                        image_ref = str(item.ref)
                    else:
                        # Try to find ref in item attributes
                        for attr in ['self_ref', 'ref', 'id', 'item_id']:
                            if hasattr(item, attr):
                                image_ref = str(getattr(item, attr))
                                break
                    
                    if image_ref:
                        image_refs_in_chunk.append(image_ref)
        
        # Also check the contextualized text for partial image descriptions
        # This catches cases where the description is in the text but truncated
        contextualized_lower = contextualized_text.lower()
        for image_ref, full_description in image_descriptions_map.items():
            # Check if any part of this description appears in the chunk text
            # Use first 50-100 chars as prefix (shorter for better matching)
            desc_prefix_len = min(100, max(50, len(full_description) // 4))
            desc_prefix = full_description[:desc_prefix_len].lower().strip()
            
            if len(desc_prefix) >= 30 and desc_prefix in contextualized_lower:
                # Description appears in chunk - check if it's complete
                if full_description.lower() not in contextualized_lower:
                    # Description is truncated - find where it starts
                    desc_start = contextualized_lower.find(desc_prefix)
                    if desc_start >= 0:
                        # Get the truncated portion
                        truncated_portion = contextualized_text[desc_start:]
                        # Get text before the description
                        text_before = contextualized_text[:desc_start].rstrip()
                        
                        # Replace truncated portion with full description
                        if text_before:
                            contextualized_text = f"{text_before}\n\n{full_description}"
                        else:
                            contextualized_text = full_description
                        prefixed_text = f"search_document: {contextualized_text}"
                        
                        logger.info(f"🔧 Preserved complete image description for {image_ref} in chunk {chunk_idx}")
                        logger.debug(f"   - Truncated portion was: {truncated_portion[:100]}...")
                        logger.debug(f"   - Full description length: {len(full_description)} chars")
                        break  # Only handle one per chunk
        
        # Also check doc_items-based detection
        if image_refs_in_chunk:
            for image_ref in image_refs_in_chunk:
                if image_ref in image_descriptions_map:
                    full_description = image_descriptions_map[image_ref]
                    # Check if the full description is in the contextualized text
                    if full_description.lower() not in contextualized_text.lower():
                        # Description is missing or truncated - append it
                        if contextualized_text.strip():
                            contextualized_text = f"{contextualized_text}\n\n{full_description}"
                        else:
                            contextualized_text = full_description
                        prefixed_text = f"search_document: {contextualized_text}"
                        logger.info(f"🔧 Added complete image description for {image_ref} to chunk {chunk_idx} (from doc_items)")
                        break  # Only handle one per chunk
        
        # Extract section title from headings (most specific heading)
        section_title = None
        if hasattr(chunk.meta, "headings") and chunk.meta.headings:
            section_title = chunk.meta.headings[-1]  # Use last (most specific) heading
        
        # Extract curated metadata
        metadata = extract_chunk_metadata(chunk)
        
        # Handle table serialization if enabled
        final_text = prefixed_text
        if serialize_tables:
            if metadata.get("content_type") == "table":
                # Serialize table from chunk's doc_items (pass document for reference resolution)
                serialized = serialize_table_from_chunk(chunk, document=document)
                if serialized:
                    # Include section title with serialized table (with blank line for readability)
                    if section_title:
                        final_text = f"search_document: {section_title}\n\n{serialized}"
                    else:
                        final_text = f"search_document: {serialized}"
                    tables_serialized += 1
                else:
                    tables_failed += 1
        
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
    
    # Preserve complete image descriptions in chunks
    chunks = _preserve_image_descriptions(chunks, document)
    
    # Log statistics (shared logic)
    _log_chunk_statistics(chunks, start_time, text_field="text", is_native=False)
    
    # Log table serialization results (one line summary)
    if serialize_tables and (tables_serialized > 0 or tables_failed > 0):
        if tables_failed > 0:
            logger.info(f"📊 Tables: {tables_serialized} serialized, {tables_failed} failed")
        else:
            logger.info(f"📊 Tables: {tables_serialized} serialized")
    
    # Apply semantic refinement if requested
    if semantic_refinement:
        # Use lightweight default embedding model for semantic chunking
        # (semantic chunking only needs similarity detection, not exact model matching)
        # This avoids loading heavy models twice (once for tokenization, once for embeddings)
        logger.info("Applying semantic refinement with lightweight default embedding model")
        chunks = semantic_refine_chunks(
            chunks=chunks,
            max_tokens=max_tokens,
            embedding_model=None,  # None = use lightweight default (all-MiniLM-L6-v2)
            tokenizer=tokenizer
        )
    
    return chunks

