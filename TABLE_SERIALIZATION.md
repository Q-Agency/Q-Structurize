# Table Serialization for Embeddings - v2.0

## Overview

Table serialization for Docling 2.57.0 that extracts tables **directly from DoclingDocument structure** and converts them into embedding-optimized text format. This approach is more reliable than text parsing as it accesses structured table data before it's flattened by the chunker.

## New Architecture (v2.0)

### Extraction Flow

```
PDF → DoclingDocument → Extract Tables (structured) → Serialize → Match to Chunks
              ↓
      document.body.iterate_items()
      Access TableData.grid directly
```

**Key Improvements:**
- ✅ Extract tables from `document.body.iterate_items()` BEFORE chunking
- ✅ Access `TableData.grid` directly for structured data
- ✅ No text parsing - work with native Docling structures
- ✅ More reliable and maintainable

### Files

1. **`app/services/table_serializer.py`** - Core table extraction and serialization
2. **`app/services/document_inspector.py`** - Debug utilities to inspect document structure
3. **`app/services/hybrid_chunker.py`** - Integration with chunking workflow
4. **`test_document_tables.py`** - Test script to verify extraction

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

- **Direct structure access**: Extract from DoclingDocument.body, not chunk text
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

### Direct Table Extraction

```python
from app.services import extract_tables_from_document

# Extract tables directly from document
document = parser.parse_pdf_to_document(pdf_content)
tables = extract_tables_from_document(document)

# Each table contains:
for table in tables:
    print(f"Caption: {table['caption']}")
    print(f"Headers: {table['headers']}")
    print(f"Rows: {table['num_rows']}")
    print(f"Serialized:\n{table['serialized_text']}")
```

### Debugging with Inspector

```python
from app.services import inspect_document_structure, inspect_table_data

# Inspect entire document
inspect_document_structure(document)

# Inspect specific table
for item, level in document.body.iterate_items():
    if item.label == 'table':
        inspect_table_data(item)
```

## API Reference

### `extract_tables_from_document(document: DoclingDocument) -> List[Dict]`

Extract all tables from DoclingDocument structure.

**Returns:** List of dictionaries:
```python
{
    'caption': str,              # Table caption (if available)
    'serialized_text': str,      # Key-value formatted text
    'headers': List[str],        # Column headers
    'num_rows': int,            # Number of data rows
    'item': Any,                # Original table item
    'level': int                # Nesting level in document
}
```

### `serialize_table_item(item: Any) -> Optional[str]`

Serialize a single table item from document.body.iterate_items().

**Args:** Table item from DoclingDocument
**Returns:** Serialized table text, or None if extraction fails

### `format_table_as_keyvalue(headers: List[str], rows: List[List[str]], caption: Optional[str]) -> str`

Format table data as key-value pairs for embedding.

**Example:**
```python
headers = ['Region', 'Q1', 'Q2']
rows = [['North', '100', '150'], ['South', '120', '180']]
text = format_table_as_keyvalue(headers, rows, 'Sales Data')
# Output:
# Table: Sales Data
# Region: North, Q1: 100, Q2: 150
# Region: South, Q1: 120, Q2: 180
```

### `inspect_document_structure(document: DoclingDocument) -> None`

Print detailed information about document structure (for debugging).

### `inspect_table_data(item: Any) -> None`

Deep inspection of a single table item (for debugging).

## How It Works

### 1. Document-Level Extraction

```python
# In chunk_document() when serialize_tables=True:

# Step 1: Extract tables from document BEFORE chunking
extracted_tables = extract_tables_from_document(document)

# Step 2: Build mapping from table text to serialized version
table_texts = {}
for table_info in extracted_tables:
    orig_text = table_info['item'].text
    table_texts[orig_text[:200]] = table_info['serialized_text']
```

### 2. Table Structure Access

```python
# Inside extract_tables_from_document():

for item, level in document.body.iterate_items():
    if item.label == 'table':
        # Access structured data directly
        table_data = item.data  # TableData object
        grid = table_data.grid  # Grid structure
        
        # Try multiple extraction methods:
        # 1. grid.export_to_dataframe()
        # 2. grid.export_to_list()
        # 3. grid.cells iteration
        # 4. Markdown export (fallback)
```

### 3. Chunk Matching

```python
# Step 3: After chunking, match table chunks to serialized versions

for chunk in chunker.chunk(document):
    if chunk is table:
        # Match chunk text to pre-extracted table
        for text_key, serialized in table_texts.items():
            if matches(chunk.text, text_key):
                chunk.text = serialized  # Replace with key-value format
```

## Testing

### Test with PDF

```bash
python test_document_tables.py your_file.pdf
```

This will:
1. Parse PDF to DoclingDocument
2. Inspect document structure
3. Extract tables directly
4. Show serialized output

### Test with API

```bash
curl -X POST "http://localhost:8878/parse/chunk" \
  -F "file=@your_pdf_with_tables.pdf" \
  -F 'chunk_options={"serialize_tables": true}'
```

Check output for tables with `content_type: "table"`.

## Benefits of New Architecture

### vs Old Text Parsing Approach

| Aspect | Old (v1.x) | New (v2.0) |
|--------|-----------|-----------|
| **Extraction** | Parse flattened text | Direct structure access |
| **Reliability** | Regex patterns, fragile | Native Docling API |
| **Code** | Complex parsers | Simple extraction |
| **Quality** | Lossy, pattern-dependent | Full table structure |
| **Maintenance** | Hard to update | Easy to maintain |

### Comparison

**Old approach (v1.x):**
```python
# Parse text like: "Activity, Column = Value. Activity, Column = Value."
result = _parse_docling_text_table(chunk.text)  # Fragile!
```

**New approach (v2.0):**
```python
# Access structure directly
for item in document.body.iterate_items():
    if item.label == 'table':
        grid = item.data.grid  # Clean!
        headers, rows = extract_from_grid(grid)
```

## Troubleshooting

### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG
```

Look for:
- `"Extracting tables from document structure..."`
- `"Extracted N tables in X.XXs"`
- `"Matched and serialized table chunk N"`

### Common Issues

**Issue: No tables extracted**
- Check that `DOCLING_DO_TABLE_STRUCTURE=true` is set
- Use `inspect_document_structure()` to see what's in the document
- Verify PDF actually contains tables

**Issue: Tables not serialized**
- Check logs for "could not be matched to extracted tables"
- The chunk text might not match the table item text
- Try with `LOG_LEVEL=DEBUG` to see matching attempts

**Issue: Empty table data**
- Some tables might not have extractable grid structure
- Check `inspect_table_data(item)` to see what's available
- Fallback to markdown export if grid is empty

### Inspect Your Document

```python
from app.services import inspect_document_structure

document = parser.parse_pdf_to_document(pdf_content)
inspect_document_structure(document)
```

This shows:
- What attributes are available
- How many tables exist
- What data each table has
- Available extraction methods

## Configuration

### Environment Variables

Table extraction must be enabled in Docling:

```bash
DOCLING_DO_TABLE_STRUCTURE=true
DOCLING_TABLE_MODE=accurate  # or 'fast'
```

### Runtime Configuration

Enable per request:

```python
chunks = chunk_document(
    document,
    max_tokens=512,
    serialize_tables=True  # Enable for this request
)
```

## Version History

- **v2.0** (2025-10-21): Document-level extraction architecture
  - Complete refactor to extract from DoclingDocument
  - Added `extract_tables_from_document()`
  - Added `document_inspector` module for debugging
  - Removed text parsing approach
  - Direct access to TableData.grid
  - More reliable and maintainable

- **v1.1** (2025-10-21): Text-based parser (deprecated)
  - Added text parsing for prose-style tables
  - Complex regex patterns

- **v1.0** (2025-10-21): Initial implementation (deprecated)
  - Chunk-level text parsing
  - Markdown table parser

## Migration from v1.x

If you were using v1.x:

**Old code:**
```python
from app.services import serialize_table_chunk

serialized = serialize_table_chunk(chunk)
```

**New code:**
```python
# Now done automatically when serialize_tables=True
chunks = chunk_document(document, serialize_tables=True)
```

The new approach is automatic - just enable `serialize_tables=True` in `chunk_document()`.

## Support

For questions or issues:
- Enable DEBUG logging
- Use `inspect_document_structure()` to see document contents
- Check that Docling table extraction is enabled
- Share debug logs for troubleshooting
