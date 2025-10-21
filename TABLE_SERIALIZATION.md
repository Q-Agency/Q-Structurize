# Table Serialization for Embeddings

## Overview

Table serialization helper for Docling 2.57.0 that converts table structures into embedding-optimized text format. This feature enhances semantic search and RAG applications by transforming tables into key-value pair text suitable for embedding models.

## Implementation

### Files Created/Modified

1. **`app/services/table_serializer.py`** (NEW)
   - Core table serialization module
   - Extracts table data from Docling's BaseChunk objects
   - Formats tables as key-value pairs

2. **`app/services/hybrid_chunker.py`** (UPDATED)
   - Added `serialize_tables` parameter to `chunk_document()`
   - Integrates table serialization into chunking workflow
   - Backward compatible (default: disabled)

3. **`app/services/__init__.py`** (UPDATED)
   - Exports table serialization functions
   - Clean module interface

## Features

### Serialization Format

Tables are serialized as key-value pairs optimized for embeddings:

```
Table: Sales Data Q1 2024
Region: North, Q1: 100, Q2: 150
Region: South, Q1: 120, Q2: 180
Region: East, Q1: 110, Q2: 160
```

### Key Characteristics

- **Entire table as one chunk**: Not split per-row
- **Key-value format**: "Column1: Value1, Column2: Value2, ..."
- **Caption inclusion**: Table captions prefixed (if available)
- **Embedding optimized**: Format designed for semantic search

## Usage

### Basic Usage

```python
from app.services.hybrid_chunker import chunk_document
from app.services.docling_parser import DoclingParser

# Parse PDF
parser = DoclingParser()
document = parser.parse_pdf_to_document(pdf_content)

# Chunk with table serialization enabled
chunks = chunk_document(
    document,
    max_tokens=512,
    serialize_tables=True  # Enable table serialization
)

# Tables will be in key-value format
for chunk in chunks:
    if chunk['metadata']['content_type'] == 'table':
        print(chunk['text'])
        # Output: search_document: Table: Caption\nColumn1: Value1, Column2: Value2...
```

### Advanced Usage

```python
# Direct table serialization
from app.services import serialize_table_chunk

# For a single chunk
if chunk_is_table(chunk):
    serialized_text = serialize_table_chunk(chunk)
    if serialized_text:
        # Use serialized text for embedding
        embedding = embed_model.encode(serialized_text)
```

### API Functions

#### `serialize_table_chunk(chunk: BaseChunk) -> Optional[str]`

Main entry point for table serialization.

**Args:**
- `chunk`: BaseChunk object from Docling's HybridChunker

**Returns:**
- Formatted table text string, or None if no table found

**Example:**
```python
serialized = serialize_table_chunk(chunk)
if serialized:
    print("Table found:", serialized[:100])
```

#### `extract_table_from_doc_items(doc_items: List) -> Optional[Dict]`

Extracts table structure from Docling document items.

**Returns:**
```python
{
    'caption': 'Table 1: Sales Data',
    'headers': ['Region', 'Q1', 'Q2'],
    'rows': [['North', '100', '150'], ...],
    'markdown': '| Region | Q1 | Q2 |\n...'
}
```

#### `format_table_as_keyvalue(table_data: Dict) -> str`

Formats table data as key-value pairs for embedding.

**Example:**
```python
table_data = {
    'caption': 'Sales Data',
    'headers': ['Region', 'Q1'],
    'rows': [['North', '100']]
}
text = format_table_as_keyvalue(table_data)
# Output: "Table: Sales Data\nRegion: North, Q1: 100"
```

## Integration with Chunking Workflow

The table serialization is seamlessly integrated into the hybrid chunking workflow:

1. **Document is chunked** using HybridChunker
2. **Table chunks are detected** via metadata extraction
3. **Tables are serialized** (if `serialize_tables=True`)
4. **Serialized text replaces** default chunk text
5. **Metadata preserved** with `content_type: "table"`

## Backward Compatibility

The implementation is fully backward compatible:

- **Default behavior unchanged**: `serialize_tables=False` by default
- **No breaking changes**: All existing code continues to work
- **Opt-in feature**: Enable when needed for embeddings

## Benefits for Embeddings

### Why Key-Value Format?

1. **Semantic clarity**: "Region: North, Sales: 100" is more meaningful than raw table text
2. **Context preservation**: Column headers repeated for each row
3. **Better embeddings**: Embedding models understand the relationship between keys and values
4. **Search optimization**: Queries like "sales in north region" match naturally

### Comparison

**Without serialization:**
```
| Region | Sales |
| North  | 100   |
| South  | 120   |
```

**With serialization:**
```
Table: Q1 Sales Summary
Region: North, Sales: 100
Region: South, Sales: 120
```

The serialized format provides better semantic representation for embedding models.

## Docling 2.57.0 Compatibility

This implementation is designed for Docling 2.57.0 and handles:

- **TableData structures**: Extracts grid/cells from Docling's internal format
- **Multiple fallbacks**: Grid cells → Markdown → Text representation
- **Caption extraction**: From item.captions attribute
- **Robust parsing**: Handles various table structures gracefully

## Configuration

### Environment Variables

Table extraction must be enabled in Docling configuration:

```bash
DOCLING_DO_TABLE_STRUCTURE=true
DOCLING_TABLE_MODE=accurate  # or 'fast'
```

### Runtime Configuration

Enable table serialization per request:

```python
chunks = chunk_document(
    document,
    max_tokens=512,
    serialize_tables=True  # Enable for this request
)
```

## Testing

To test the implementation:

```python
# 1. Parse a PDF with tables
parser = DoclingParser()
document = parser.parse_pdf_to_document(pdf_with_tables)

# 2. Chunk with serialization enabled
chunks = chunk_document(document, serialize_tables=True)

# 3. Find table chunks
table_chunks = [c for c in chunks if c['metadata']['content_type'] == 'table']

# 4. Verify serialization
for chunk in table_chunks:
    print(chunk['text'])
    # Should see key-value format
```

## Future Enhancements

Potential improvements for future versions:

1. **Per-row chunking**: Option to split large tables into row-level chunks
2. **Natural language format**: "The Region is North and the Sales is 100"
3. **Table summarization**: Add table summaries before detailed rows
4. **Custom formatters**: Plugin system for custom serialization formats
5. **Hierarchical tables**: Special handling for nested/hierarchical table structures

## Support

For questions or issues:
- Check logs for table serialization debug messages
- Verify `DOCLING_DO_TABLE_STRUCTURE=true` is set
- Ensure Docling 2.57.0 is installed: `pip list | grep docling`

## Version History

- **v1.0** (2025-10-21): Initial implementation
  - Key-value serialization format
  - Integration with hybrid_chunker
  - Docling 2.57.0 support

