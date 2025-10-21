"""
Table serialization helper for Docling 2.57.0

This module extracts tables directly from DoclingDocument structure and
serializes them into embedding-optimized text format.

ARCHITECTURE:
- Extract tables from document.body.iterate_items() BEFORE chunking
- Access TableData.grid directly for structured data
- No text parsing - work with native Docling structures
- Serialize to key-value format optimized for embeddings

SERIALIZATION FORMAT:
- Entire table as one chunk (not per-row)
- Key-value pairs: "Column1: Value1, Column2: Value2, ..."
- Table caption included as prefix (if available)
- Optimized for semantic search and embedding models

USAGE:
    from app.services.table_serializer import extract_tables_from_document
    
    # Extract tables from document
    document = parser.parse_pdf_to_document(pdf_content)
    tables = extract_tables_from_document(document)
    
    # Use in chunking
    chunks = chunk_document(document, serialize_tables=True)
"""

import logging
from typing import List, Dict, Any, Optional
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)

# Export public API
__all__ = [
    'extract_tables_from_document',
    'serialize_table_item',
    'format_table_as_keyvalue',
]


def extract_table_structure(table_data: Any) -> Optional[Dict[str, Any]]:
    """
    Extract structured data from Docling's TableData object.
    
    Accesses the grid structure directly to get headers and rows.
    
    Args:
        table_data: TableData object from item.data
        
    Returns:
        Dictionary with 'headers', 'rows', and 'grid' information
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
        logger.warning(f"Failed to extract table structure: {e}", exc_info=True)
    
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


def serialize_table_item(item: Any) -> Optional[str]:
    """
    Serialize a single table item from document.body.iterate_items().
    
    Extracts table structure and formats as key-value pairs.
    
    Args:
        item: Table item from DoclingDocument
        
    Returns:
        Serialized table text, or None if extraction fails
    """
    if not hasattr(item, 'label') or item.label != 'table':
        return None
    
    # Extract caption
    caption = None
    if hasattr(item, 'captions') and item.captions:
        caption = ' '.join(str(cap) for cap in item.captions)
    
    # Extract table structure
    table_struct = None
    if hasattr(item, 'data') and item.data:
        table_struct = extract_table_structure(item.data)
    
    if not table_struct or not table_struct.get('headers'):
        logger.debug("Could not extract table structure, skipping serialization")
        return None
    
    # Format as key-value pairs
    serialized = format_table_as_keyvalue(
        headers=table_struct['headers'],
        rows=table_struct['rows'],
        caption=caption
    )
    
    logger.debug(f"Serialized table: {len(table_struct['rows'])} rows, caption='{caption}'")
    
    return serialized


def extract_tables_from_document(document: DoclingDocument) -> List[Dict[str, Any]]:
    """
    Extract all tables from DoclingDocument structure.
    
    Iterates through document.body to find table items and extracts
    their structured data directly (no text parsing).
    
    Args:
        document: DoclingDocument from Docling parser
        
    Returns:
        List of dictionaries with table information:
        {
            'caption': str,
            'serialized_text': str,
            'headers': List[str],
            'num_rows': int,
            'item': original table item (for matching with chunks)
        }
        
    Example:
        >>> document = parser.parse_pdf_to_document(pdf_content)
        >>> tables = extract_tables_from_document(document)
        >>> for table in tables:
        ...     print(f"Table: {table['caption']}")
        ...     print(table['serialized_text'])
    """
    tables = []
    
    if not hasattr(document, 'body'):
        logger.warning("Document has no body attribute")
        return tables
    
    if not hasattr(document.body, 'iterate_items'):
        logger.warning("Document body has no iterate_items method")
        return tables
    
    logger.info("Extracting tables from document...")
    
    try:
        for item, level in document.body.iterate_items():
            if hasattr(item, 'label') and item.label == 'table':
                logger.debug(f"Found table at level {level}")
                
                # Serialize the table
                serialized = serialize_table_item(item)
                
                if serialized:
                    # Extract additional metadata
                    caption = None
                    if hasattr(item, 'captions') and item.captions:
                        caption = ' '.join(str(cap) for cap in item.captions)
                    
                    # Get structure info
                    table_struct = None
                    if hasattr(item, 'data') and item.data:
                        table_struct = extract_table_structure(item.data)
                    
                    tables.append({
                        'caption': caption,
                        'serialized_text': serialized,
                        'headers': table_struct.get('headers') if table_struct else None,
                        'num_rows': len(table_struct.get('rows', [])) if table_struct else 0,
                        'item': item,  # Keep reference for matching with chunks
                        'level': level,
                    })
                    
                    logger.info(f"Extracted table: caption='{caption}', rows={tables[-1]['num_rows']}")
                else:
                    logger.debug("Table serialization failed, skipping")
    
    except Exception as e:
        logger.error(f"Error extracting tables from document: {e}", exc_info=True)
    
    logger.info(f"Extracted {len(tables)} tables from document")
    
    return tables
