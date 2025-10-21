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
        # Check if table_data has grid
        if not hasattr(table_data, 'grid'):
            logger.info("TableData has no 'grid' attribute")
            logger.info(f"TableData attributes: {[attr for attr in dir(table_data) if not attr.startswith('_')][:20]}")
            # Try direct markdown export as fallback
        elif not table_data.grid:
            logger.info("TableData.grid is None")
        else:
            # Access grid structure
            grid = table_data.grid
            logger.info(f"Found grid structure (type: {type(grid)})")
            
            # Try method 1: num_rows and num_cols
            if hasattr(grid, 'num_rows') and hasattr(grid, 'num_cols'):
                result['num_rows'] = grid.num_rows
                result['num_cols'] = grid.num_cols
                logger.info(f"Grid size: {grid.num_rows} rows x {grid.num_cols} cols")
            
            # Try method 2: export_to_dataframe (if available)
            if hasattr(grid, 'export_to_dataframe'):
                logger.info("Trying export_to_dataframe...")
                try:
                    df = grid.export_to_dataframe()
                    if df is not None and not df.empty:
                        result['headers'] = df.columns.tolist()
                        result['rows'] = df.values.tolist()
                        logger.info(f"✅ Extracted via dataframe: {len(result['rows'])} rows")
                        return result
                    else:
                        logger.info("export_to_dataframe returned empty/None")
                except Exception as e:
                    logger.info(f"export_to_dataframe failed: {e}")
            else:
                logger.info("Grid has no export_to_dataframe method")
            
            # Try method 3: export_to_list
            if hasattr(grid, 'export_to_list'):
                logger.info("Trying export_to_list...")
                try:
                    rows = grid.export_to_list()
                    if rows and len(rows) > 0:
                        result['headers'] = rows[0] if rows else None
                        result['rows'] = rows[1:] if len(rows) > 1 else []
                        logger.info(f"✅ Extracted via list: {len(result['rows'])} rows")
                        return result
                    else:
                        logger.info("export_to_list returned empty")
                except Exception as e:
                    logger.info(f"export_to_list failed: {e}")
            else:
                logger.info("Grid has no export_to_list method")
            
            # Try method 4: Iterate cells
            if hasattr(grid, 'cells'):
                logger.info("Trying cell iteration...")
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
                            logger.info(f"✅ Extracted via cells: {len(result['rows'])} rows")
                            return result
                    else:
                        logger.info("Cell iteration produced no rows")
                            
                except Exception as e:
                    logger.info(f"Cell iteration failed: {e}")
            else:
                logger.info("Grid has no cells attribute")
        
        # Try method 5: Direct table text via export_to_markdown
        if hasattr(table_data, 'export_to_markdown'):
            logger.info("Trying export_to_markdown...")
            try:
                markdown = table_data.export_to_markdown()
                if markdown and '|' in markdown:
                    logger.info(f"Got markdown (length: {len(markdown)})")
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
                            logger.info(f"✅ Extracted via markdown: {len(result['rows'])} rows")
                            return result
                        else:
                            logger.info("Markdown parsing produced no rows")
                    else:
                        logger.info("No data lines found in markdown")
                else:
                    logger.info("export_to_markdown returned no markdown or no pipes")
            except Exception as e:
                logger.info(f"export_to_markdown failed: {e}")
        else:
            logger.info("TableData has no export_to_markdown method")
        
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


def serialize_table_from_chunk(chunk: BaseChunk, document: Any = None) -> Optional[str]:
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
    if not hasattr(chunk, 'meta'):
        logger.warning("Chunk has no meta attribute")
        return None
        
    if not hasattr(chunk.meta, 'doc_items'):
        logger.warning("Chunk.meta has no doc_items attribute")
        return None
    
    if not chunk.meta.doc_items:
        logger.info("Chunk.meta.doc_items is empty")
        return None
    
    logger.info(f"Chunk has {len(chunk.meta.doc_items)} doc_items, checking for tables...")
    
    # Find table items in doc_items
    table_item = None
    for i, item in enumerate(chunk.meta.doc_items):
        logger.debug(f"  Item {i}: label={getattr(item, 'label', 'NO_LABEL')}")
        if hasattr(item, 'label') and item.label == 'table':
            table_item = item
            logger.info(f"  Found table at item {i}!")
            break
    
    if not table_item:
        logger.info("No table item found in doc_items (checked all items)")
        return None
    
    logger.info(f"Processing table item...")
    
    # Extract caption
    caption = None
    if hasattr(table_item, 'captions') and table_item.captions:
        caption = ' '.join(str(cap) for cap in table_item.captions)
        logger.info(f"Table caption: {caption}")
    else:
        logger.info("No caption found")
    
    # Check if table item has data
    if not hasattr(table_item, 'data'):
        logger.warning("Table item has no 'data' attribute!")
        logger.info(f"Table item attributes: {[attr for attr in dir(table_item) if not attr.startswith('_')][:20]}")
        
        # Try to get data via model_dump()
        if hasattr(table_item, 'model_dump'):
            logger.info("Trying to get data via model_dump()...")
            try:
                item_data = table_item.model_dump()
                logger.info(f"model_dump keys: {list(item_data.keys())}")
                
                # Check if there's table data in the dump
                if 'data' in item_data and item_data['data']:
                    logger.info("Found 'data' in model_dump!")
                    # Try to reconstruct table data
                    # For now, just log what we find
                    logger.info(f"Data content: {str(item_data['data'])[:200]}")
                    
                # Check for other potential table data fields
                for key in ['content', 'table', 'grid', 'cells']:
                    if key in item_data:
                        logger.info(f"Found '{key}' in model_dump: {type(item_data[key])}")
                        
            except Exception as e:
                logger.warning(f"model_dump() failed: {e}")
        
        # Try get_ref() to get reference and resolve it from document
        if hasattr(table_item, 'get_ref'):
            try:
                ref = table_item.get_ref()
                logger.info(f"get_ref() returned: {ref}")
                
                # Parse the reference (e.g., '#/tables/0')
                if document and hasattr(ref, 'cref'):
                    ref_str = ref.cref
                    logger.info(f"Reference string: {ref_str}")
                    
                    # Parse reference like '#/tables/0'
                    if ref_str.startswith('#/tables/'):
                        table_index = int(ref_str.split('/')[-1])
                        logger.info(f"Trying to access document.tables[{table_index}]")
                        
                        # Access the actual table from document
                        if hasattr(document, 'tables') and document.tables:
                            if table_index < len(document.tables):
                                actual_table = document.tables[table_index]
                                logger.info(f"✅ Found actual table in document.tables[{table_index}]!")
                                logger.info(f"Table type: {type(actual_table)}")
                                
                                # Now extract from the actual table
                                if hasattr(actual_table, 'data'):
                                    logger.info("Actual table has 'data' attribute!")
                                    table_struct = extract_table_structure(actual_table.data)
                                    
                                    if table_struct and table_struct.get('headers'):
                                        logger.info(f"Successfully extracted from resolved table: {len(table_struct.get('headers', []))} headers, {len(table_struct.get('rows', []))} rows")
                                        
                                        # Format and return
                                        serialized = format_table_as_keyvalue(
                                            headers=table_struct['headers'],
                                            rows=table_struct['rows'],
                                            caption=caption
                                        )
                                        return serialized
                                else:
                                    logger.warning("Actual table has no 'data' attribute")
                            else:
                                logger.warning(f"Table index {table_index} out of range (document has {len(document.tables)} tables)")
                        else:
                            logger.warning("Document has no 'tables' attribute or it's empty")
                else:
                    logger.warning("No document provided or ref has no 'cref' attribute")
                    
            except Exception as e:
                logger.warning(f"Reference resolution failed: {e}", exc_info=True)
        
        return None
    
    if not table_item.data:
        logger.warning("Table item.data is None!")
        return None
    
    # Extract table structure from item.data
    logger.info(f"Extracting table structure from item.data (type: {type(table_item.data)})")
    table_struct = extract_table_structure(table_item.data)
    
    if not table_struct:
        logger.warning("extract_table_structure returned None")
        return None
    
    if not table_struct.get('headers'):
        logger.warning(f"Table structure has no headers. Structure: {table_struct}")
        return None
    
    logger.info(f"Successfully extracted table structure: {len(table_struct.get('headers', []))} headers, {len(table_struct.get('rows', []))} rows")
    
    # Format as key-value pairs
    serialized = format_table_as_keyvalue(
        headers=table_struct['headers'],
        rows=table_struct['rows'],
        caption=caption
    )
    
    logger.debug(f"Serialized table: {len(table_struct['rows'])} rows, {len(table_struct['headers'])} columns")
    
    return serialized
