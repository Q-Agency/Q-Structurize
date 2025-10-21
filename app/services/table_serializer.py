"""
Table serialization helper for Docling 2.57.0

This module extracts tables from chunk.meta.doc_items and serializes them
into embedding-optimized text format.

ARCHITECTURE:
- Tables are accessed from chunks (HybridChunker already extracts them)
- Each chunk has chunk.meta.doc_items containing table references
- Access TableData directly from doc_items
- Serialize to key-value format optimized for embeddings

SERIALIZATION FORMAT:
- Entire table as one chunk (not per-row)
- Key-value pairs: "Column1: Value1, Column2: Value2, ..."
- Table caption included as prefix (if available)
- Optimized for semantic search and embedding models

USAGE:
    from app.services.table_serializer import serialize_table_from_chunk
    
    # In chunking workflow
    for chunk in chunker.chunk(document):
        if chunk contains table:
            serialized_text = serialize_table_from_chunk(chunk)
"""

import logging
from typing import List, Dict, Any, Optional
from docling_core.transforms.chunker import BaseChunk

logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    'serialize_table_from_chunk',
    'extract_table_structure',
    'format_table_as_keyvalue',
]


def extract_table_structure(table_data: Any) -> Optional[Dict[str, Any]]:
    """
    Extract structured data from Docling's TableData object.
    
    Accesses the grid structure directly to get headers and rows.
    
    Args:
        table_data: TableData object from item.data
        
    Returns:
        Dictionary with 'headers', 'rows', and grid information
        Returns None if extraction fails
    """
    if not table_data:
        return None
    
    result = {
        'headers': None,
        'rows': [],
        'num_rows': 0,
        'num_cols': 0,
    }
    
    try:
        # Access grid structure
        if hasattr(table_data, 'grid') and table_data.grid:
            grid = table_data.grid
            
            # Try method 1: num_rows and num_cols
            if hasattr(grid, 'num_rows') and hasattr(grid, 'num_cols'):
                result['num_rows'] = grid.num_rows
                result['num_cols'] = grid.num_cols
                logger.debug(f"Grid size: {grid.num_rows} rows x {grid.num_cols} cols")
            
            # Try method 2: export_to_dataframe (if available)
            if hasattr(grid, 'export_to_dataframe'):
                try:
                    df = grid.export_to_dataframe()
                    if df is not None and not df.empty:
                        result['headers'] = df.columns.tolist()
                        result['rows'] = df.values.tolist()
                        logger.debug(f"Extracted via dataframe: {len(result['rows'])} rows")
                        return result
                except Exception as e:
                    logger.debug(f"export_to_dataframe failed: {e}")
            
            # Try method 3: export_to_list
            if hasattr(grid, 'export_to_list'):
                try:
                    rows = grid.export_to_list()
                    if rows and len(rows) > 0:
                        result['headers'] = rows[0] if rows else None
                        result['rows'] = rows[1:] if len(rows) > 1 else []
                        logger.debug(f"Extracted via list: {len(result['rows'])} rows")
                        return result
                except Exception as e:
                    logger.debug(f"export_to_list failed: {e}")
            
            # Try method 4: Iterate cells
            if hasattr(grid, 'cells'):
                try:
                    cells = grid.cells
                    rows_dict = {}
                    
                    for cell in cells:
                        if hasattr(cell, 'row') and hasattr(cell, 'col'):
                            row_idx = cell.row
                            col_idx = cell.col
                            text = cell.text if hasattr(cell, 'text') else str(cell)
                            
                            if row_idx not in rows_dict:
                                rows_dict[row_idx] = {}
                            rows_dict[row_idx][col_idx] = text
                    
                    if rows_dict:
                        # Convert to list format
                        sorted_rows = sorted(rows_dict.items())
                        all_rows = []
                        for _, row_cells in sorted_rows:
                            sorted_cells = sorted(row_cells.items())
                            row = [text for _, text in sorted_cells]
                            all_rows.append(row)
                        
                        if all_rows:
                            result['headers'] = all_rows[0]
                            result['rows'] = all_rows[1:] if len(all_rows) > 1 else []
                            logger.debug(f"Extracted via cells: {len(result['rows'])} rows")
                            return result
                            
                except Exception as e:
                    logger.debug(f"Cell iteration failed: {e}")
        
        # Try method 5: Direct table text via export_to_markdown
        if hasattr(table_data, 'export_to_markdown'):
            try:
                markdown = table_data.export_to_markdown()
                if markdown and '|' in markdown:
                    # Parse markdown table
                    lines = [l.strip() for l in markdown.strip().split('\n') if l.strip()]
                    data_lines = [l for l in lines if l.count('|') > 1 and not all(c in '|-: ' for c in l)]
                    
                    if data_lines:
                        rows = []
                        for line in data_lines:
                            cells = [c.strip() for c in line.strip('|').split('|')]
                            rows.append(cells)
                        
                        if rows:
                            result['headers'] = rows[0]
                            result['rows'] = rows[1:] if len(rows) > 1 else []
                            logger.debug(f"Extracted via markdown: {len(result['rows'])} rows")
                            return result
            except Exception as e:
                logger.debug(f"export_to_markdown failed: {e}")
        
    except Exception as e:
        logger.warning(f"Failed to extract table structure: {e}")
    
    return result if result['headers'] or result['rows'] else None


def format_table_as_keyvalue(
    headers: List[str],
    rows: List[List[str]],
    caption: Optional[str] = None
) -> str:
    """
    Format table data as key-value pairs for embedding.
    
    Converts structured table data into a text format optimized for
    semantic search and embedding models. Each row is formatted as:
    "Column1: Value1, Column2: Value2, Column3: Value3"
    
    Args:
        headers: List of column headers
        rows: List of rows (each row is a list of values)
        caption: Optional table caption
        
    Returns:
        Formatted string with caption (if any) and rows as key-value pairs
        
    Example:
        >>> headers = ['Region', 'Q1', 'Q2']
        >>> rows = [['North', '100', '150'], ['South', '120', '180']]
        >>> print(format_table_as_keyvalue(headers, rows, 'Sales Data'))
        Table: Sales Data
        Region: North, Q1: 100, Q2: 150
        Region: South, Q1: 120, Q2: 180
    """
    lines = []
    
    # Add caption as prefix if available
    if caption:
        lines.append(f"Table: {caption}")
    
    # Format each row as key-value pairs
    for row in rows:
        # Match headers with row values
        pairs = []
        for i, header in enumerate(headers):
            value = row[i] if i < len(row) else ''
            # Clean header and value
            header_clean = str(header).strip()
            value_clean = str(value).strip()
            
            if header_clean and value_clean:  # Skip empty headers or values
                pairs.append(f"{header_clean}: {value_clean}")
        
        if pairs:
            lines.append(', '.join(pairs))
    
    return '\n'.join(lines)


def serialize_table_from_chunk(chunk: BaseChunk) -> Optional[str]:
    """
    Serialize table from a chunk's doc_items.
    
    This is the main entry point for table serialization. HybridChunker
    already extracts tables and includes them in chunk.meta.doc_items.
    We access that data and re-serialize it to key-value format.
    
    Args:
        chunk: BaseChunk object from HybridChunker
        
    Returns:
        Serialized table text in key-value format, or None if no table found
        
    Example:
        >>> for chunk in chunker.chunk(document):
        ...     if has_table(chunk):
        ...         serialized = serialize_table_from_chunk(chunk)
        ...         if serialized:
        ...             print(serialized)
    """
    if not hasattr(chunk, 'meta') or not hasattr(chunk.meta, 'doc_items'):
        logger.debug("Chunk has no doc_items")
        return None
    
    # Find table items in doc_items
    table_item = None
    for item in chunk.meta.doc_items:
        if hasattr(item, 'label') and item.label == 'table':
            table_item = item
            break
    
    if not table_item:
        logger.debug("No table found in chunk doc_items")
        return None
    
    logger.debug(f"Found table item in chunk")
    
    # Extract caption
    caption = None
    if hasattr(table_item, 'captions') and table_item.captions:
        caption = ' '.join(str(cap) for cap in table_item.captions)
        logger.debug(f"Table caption: {caption}")
    
    # Extract table structure from item.data
    table_struct = None
    if hasattr(table_item, 'data') and table_item.data:
        logger.debug(f"Extracting table structure from item.data")
        table_struct = extract_table_structure(table_item.data)
    else:
        logger.debug("Table item has no data attribute")
    
    if not table_struct or not table_struct.get('headers'):
        logger.debug("Could not extract table structure")
        return None
    
    # Format as key-value pairs
    serialized = format_table_as_keyvalue(
        headers=table_struct['headers'],
        rows=table_struct['rows'],
        caption=caption
    )
    
    logger.debug(f"Serialized table: {len(table_struct['rows'])} rows, {len(table_struct['headers'])} columns")
    
    return serialized
