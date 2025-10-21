"""
Table serialization helper for Docling 2.57.0

This module provides utilities to extract and serialize table structures from
Docling's BaseChunk objects into embedding-optimized text format.

SERIALIZATION FORMAT:
- Entire table as one chunk (not per-row)
- Key-value pairs: "Column1: Value1, Column2: Value2, ..."
- Table caption included as prefix (if available)
- Optimized for semantic search and embedding models

USAGE:
    from app.services.table_serializer import serialize_table_chunk
    
    # In chunking workflow
    if chunk contains table:
        serialized_text = serialize_table_chunk(chunk)
        if serialized_text:
            # Use serialized text instead of raw chunk text
            chunk_text = serialized_text

DOCLING 2.57.0 TABLE STRUCTURE:
- Tables are identified in chunk.meta.doc_items where item.label == "table"
- Table data accessed through item.data or item.export_to_markdown()
- Captions available in item.captions (list)
- Grid structure with cells containing text values
"""

import logging
from typing import Optional, Dict, List, Any
from docling_core.transforms.chunker import BaseChunk

logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    'serialize_table_chunk',
    'extract_table_from_doc_items',
    'format_table_as_keyvalue',
]


def extract_table_from_doc_items(doc_items: List[Any]) -> Optional[Dict[str, Any]]:
    """
    Extract table structure (headers, rows, caption) from doc_items.
    
    Iterates through doc_items to find table elements and extracts:
    - Table data (grid/cells)
    - Headers (first row or dedicated header)
    - Caption text
    - Raw markdown representation
    
    Args:
        doc_items: List of document items from chunk.meta.doc_items
        
    Returns:
        Dictionary with table data structure:
        {
            'caption': str or None,
            'headers': List[str] or None,
            'rows': List[List[str]],
            'markdown': str (fallback representation)
        }
        Returns None if no table found.
        
    Example:
        >>> doc_items = chunk.meta.doc_items
        >>> table_data = extract_table_from_doc_items(doc_items)
        >>> print(table_data['caption'])
        'Table 1: Sales by Region'
    """
    if not doc_items:
        return None
    
    table_data = {
        'caption': None,
        'headers': None,
        'rows': [],
        'markdown': ''
    }
    
    found_table = False
    
    for item in doc_items:
        # Check if this is a table item
        if not hasattr(item, 'label') or item.label != 'table':
            continue
            
        found_table = True
        
        # Debug: Log what attributes the table item has
        logger.debug(f"Found table item with attributes: {[attr for attr in dir(item) if not attr.startswith('_')][:20]}")
        
        # Extract caption if available
        if hasattr(item, 'captions') and item.captions:
            # Join multiple captions with space
            table_data['caption'] = ' '.join(str(cap) for cap in item.captions)
            logger.debug(f"Table caption: {table_data['caption']}")
        
        # Try to extract structured table data
        if hasattr(item, 'data') and item.data is not None:
            # Docling's TableData structure
            table_obj = item.data
            logger.debug(f"Table data type: {type(table_obj)}, attributes: {[attr for attr in dir(table_obj) if not attr.startswith('_')][:15]}")
            
            # Extract grid data if available
            if hasattr(table_obj, 'grid'):
                grid = table_obj.grid
                logger.debug(f"Found grid structure")
                
                # Try to parse grid into rows
                if hasattr(grid, 'export_to_list'):
                    # Some versions have export_to_list method
                    try:
                        rows = grid.export_to_list()
                        if rows:
                            # First row might be headers
                            table_data['headers'] = rows[0]
                            table_data['rows'] = rows[1:] if len(rows) > 1 else []
                    except Exception as e:
                        logger.debug(f"Could not export grid to list: {e}")
                        
                # Fallback: try to access cells directly
                elif hasattr(grid, 'cells') or hasattr(grid, '_cells'):
                    try:
                        cells = grid.cells if hasattr(grid, 'cells') else grid._cells
                        # Parse cells into row/column structure
                        # This is implementation-specific to Docling's grid structure
                        parsed_rows = _parse_grid_cells(cells)
                        if parsed_rows:
                            table_data['headers'] = parsed_rows[0]
                            table_data['rows'] = parsed_rows[1:]
                    except Exception as e:
                        logger.debug(f"Could not parse grid cells: {e}")
        
        # Fallback: use markdown export
        if hasattr(item, 'export_to_markdown'):
            try:
                table_data['markdown'] = item.export_to_markdown()
            except Exception as e:
                logger.debug(f"Could not export table to markdown: {e}")
        
        # Alternative: get text representation
        elif hasattr(item, 'text'):
            table_data['markdown'] = item.text
            
        # If we found a table, we're done (assuming one table per chunk)
        break
    
    if not found_table:
        return None
        
    # If we couldn't extract structured data, try parsing fallbacks
    if not table_data['headers'] and table_data['markdown']:
        logger.debug(f"Attempting to parse table text (length: {len(table_data['markdown'])} chars)")
        
        # Try standard markdown table parser first
        parsed = _parse_markdown_table(table_data['markdown'])
        if parsed:
            logger.debug("Successfully parsed as markdown table")
            table_data['headers'] = parsed.get('headers')
            table_data['rows'] = parsed.get('rows', [])
        else:
            # Try Docling's text-based format parser
            logger.debug("Markdown parser failed, trying text-based parser")
            parsed = _parse_docling_text_table(table_data['markdown'])
            if parsed:
                logger.debug("Successfully parsed as text-based table")
                table_data['headers'] = parsed.get('headers')
                table_data['rows'] = parsed.get('rows', [])
            else:
                logger.debug("All parsers failed, will use raw text fallback")
    
    return table_data


def _parse_grid_cells(cells: Any) -> List[List[str]]:
    """
    Parse Docling grid cells into row/column structure.
    
    This is a helper function to extract text from Docling's internal
    grid cell structure. Implementation may vary by Docling version.
    
    Args:
        cells: Grid cells object from Docling
        
    Returns:
        List of rows, where each row is a list of cell values
    """
    rows = []
    
    try:
        # Attempt to iterate cells and organize by row/col
        # This is a simplified implementation - adjust based on actual structure
        if isinstance(cells, list):
            # If cells is a flat list with row/col indices
            cell_dict = {}
            for cell in cells:
                if hasattr(cell, 'row_idx') and hasattr(cell, 'col_idx'):
                    row_idx = cell.row_idx
                    col_idx = cell.col_idx
                    text = cell.text if hasattr(cell, 'text') else str(cell)
                    
                    if row_idx not in cell_dict:
                        cell_dict[row_idx] = {}
                    cell_dict[row_idx][col_idx] = text
            
            # Convert dict to list of rows
            for row_idx in sorted(cell_dict.keys()):
                row_cells = cell_dict[row_idx]
                row = [row_cells.get(col_idx, '') for col_idx in sorted(row_cells.keys())]
                rows.append(row)
                
    except Exception as e:
        logger.debug(f"Error parsing grid cells: {e}")
    
    return rows


def _parse_markdown_table(markdown: str) -> Optional[Dict[str, Any]]:
    """
    Parse a markdown table string into headers and rows.
    
    Fallback parser for when structured table data is not available.
    Handles standard markdown table format.
    
    Args:
        markdown: Markdown table string
        
    Returns:
        Dictionary with 'headers' and 'rows' keys, or None if parsing fails
    """
    if not markdown or '|' not in markdown:
        return None
    
    try:
        lines = [line.strip() for line in markdown.strip().split('\n') if line.strip()]
        
        # Filter out separator lines (e.g., |---|---|)
        data_lines = [line for line in lines if not all(c in '|-: ' for c in line)]
        
        if not data_lines:
            return None
        
        # Parse rows by splitting on |
        parsed_rows = []
        for line in data_lines:
            # Remove leading/trailing pipes
            line = line.strip('|')
            # Split by pipe and clean
            cells = [cell.strip() for cell in line.split('|')]
            parsed_rows.append(cells)
        
        if not parsed_rows:
            return None
            
        return {
            'headers': parsed_rows[0] if parsed_rows else None,
            'rows': parsed_rows[1:] if len(parsed_rows) > 1 else []
        }
        
    except Exception as e:
        logger.debug(f"Error parsing markdown table: {e}")
        return None


def _parse_docling_text_table(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse Docling's text-based table format.
    
    Docling sometimes exports tables as prose text with patterns like:
    "Column1 = Value1. Column2 = Value2. Column1 = Value3. Column2 = Value4."
    or
    "Column1: Value1, Column2: Value2. Column1: Value3, Column2: Value4."
    
    This parser extracts these patterns and reconstructs the table structure.
    
    Args:
        text: Text representation of table from Docling
        
    Returns:
        Dictionary with 'headers' and 'rows' keys, or None if parsing fails
        
    Example:
        >>> text = "Name = John, Age = 30. Name = Jane, Age = 25."
        >>> parsed = _parse_docling_text_table(text)
        >>> print(parsed['headers'])  # ['Age', 'Name']
        >>> print(parsed['rows'])     # [['30', 'John'], ['25', 'Jane']]
    """
    import re
    
    if not text or len(text.strip()) < 5:
        return None
    
    try:
        # Split by periods first to get potential rows
        # Handle both ". " and ".\n" as row separators
        segments = re.split(r'\.\s+|\.\n', text)
        
        rows = []
        all_headers = set()
        
        for segment in segments:
            if not segment.strip():
                continue
            
            row_data = {}
            
            # Try to extract key=value or key: value patterns
            # Pattern 1: "Key = Value" or "Key=Value"
            pairs_eq = re.findall(r'([^,=:\.]+?)\s*=\s*([^,=:\.]+?)(?=\s*,|\s*$)', segment)
            
            # Pattern 2: "Key: Value" or "Key : Value"
            pairs_colon = re.findall(r'([^,=:\.]+?)\s*:\s*([^,=:\.]+?)(?=\s*,|\s*\.|$)', segment)
            
            # Use whichever pattern found more matches
            pairs = pairs_eq if len(pairs_eq) >= len(pairs_colon) else pairs_colon
            
            for key, value in pairs:
                key = key.strip()
                value = value.strip()
                
                # Clean up bullet points and special chars
                key = re.sub(r'^[•\-\*]\s*', '', key)
                value = re.sub(r'^[•\-\*]\s*', '', value)
                
                if key and value and len(key) < 100:  # Sanity check
                    all_headers.add(key)
                    row_data[key] = value
            
            if row_data:
                rows.append(row_data)
        
        if not rows or not all_headers:
            logger.debug("Could not extract structured data from text table")
            return None
        
        # Convert to consistent format with sorted headers
        headers_list = sorted(list(all_headers))
        rows_list = []
        
        for row_data in rows:
            row = [row_data.get(h, '') for h in headers_list]
            rows_list.append(row)
        
        logger.debug(f"Parsed text table: {len(headers_list)} columns, {len(rows_list)} rows")
        
        return {
            'headers': headers_list,
            'rows': rows_list
        }
        
    except Exception as e:
        logger.debug(f"Error parsing text-based table: {e}")
        return None


def format_table_as_keyvalue(table_data: Dict[str, Any]) -> str:
    """
    Format table data as key-value pairs for embedding.
    
    Converts structured table data into a text format optimized for
    semantic search and embedding models. Each row is formatted as:
    "Column1: Value1, Column2: Value2, Column3: Value3"
    
    Args:
        table_data: Dictionary with 'caption', 'headers', 'rows', 'markdown'
        
    Returns:
        Formatted string with caption (if any) and rows as key-value pairs
        
    Example:
        >>> table_data = {
        ...     'caption': 'Sales Data',
        ...     'headers': ['Region', 'Q1', 'Q2'],
        ...     'rows': [['North', '100', '150'], ['South', '120', '180']]
        ... }
        >>> print(format_table_as_keyvalue(table_data))
        Table: Sales Data
        Region: North, Q1: 100, Q2: 150
        Region: South, Q1: 120, Q2: 180
    """
    lines = []
    
    # Add caption as prefix if available
    if table_data.get('caption'):
        lines.append(f"Table: {table_data['caption']}")
    
    headers = table_data.get('headers')
    rows = table_data.get('rows', [])
    
    if headers and rows:
        # Format each row as key-value pairs
        for row in rows:
            # Match headers with row values
            pairs = []
            for i, header in enumerate(headers):
                value = row[i] if i < len(row) else ''
                if header and value:  # Skip empty headers or values
                    pairs.append(f"{header}: {value}")
            
            if pairs:
                lines.append(', '.join(pairs))
    
    elif table_data.get('markdown'):
        # Fallback to markdown if structured data not available
        # Clean up the markdown for better embedding
        markdown = table_data['markdown']
        # Remove separator lines
        cleaned_lines = [
            line.strip() 
            for line in markdown.split('\n') 
            if line.strip() and not all(c in '|-: ' for c in line.strip())
        ]
        lines.extend(cleaned_lines)
    
    return '\n'.join(lines)


def serialize_table_chunk(chunk: BaseChunk) -> Optional[str]:
    """
    Serialize a table from BaseChunk into embedding-optimized text.
    
    This is the main entry point for table serialization. It extracts table
    data from the chunk and formats it as key-value pairs suitable for
    embedding and semantic search.
    
    PROCESS:
    1. Extract table data from chunk.meta.doc_items
    2. Format as key-value pairs with caption
    3. Return None if chunk doesn't contain valid table data
    
    Args:
        chunk: BaseChunk object from Docling's HybridChunker
        
    Returns:
        Formatted table text string, or None if no table found
        
    Example:
        >>> chunk = next(chunker.chunk(document))
        >>> serialized = serialize_table_chunk(chunk)
        >>> if serialized:
        ...     print("Found table:", serialized[:100])
        
    Usage in chunking workflow:
        >>> for chunk in chunker.chunk(document):
        ...     if has_table(chunk):
        ...         text = serialize_table_chunk(chunk) or chunk.text
        ...     else:
        ...         text = chunk.text
    """
    if not hasattr(chunk, 'meta') or not hasattr(chunk.meta, 'doc_items'):
        return None
    
    # Extract table structure
    table_data = extract_table_from_doc_items(chunk.meta.doc_items)
    
    if not table_data:
        return None
    
    # Format as key-value pairs
    serialized_text = format_table_as_keyvalue(table_data)
    
    if not serialized_text or not serialized_text.strip():
        return None
    
    logger.debug(f"Serialized table with {len(table_data.get('rows', []))} rows")
    
    return serialized_text

