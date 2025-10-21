# Table Serialization for Embeddings

## Overview

Table serialization for Docling 2.57.0 that extracts tables from **chunk.meta.doc_items** (after HybridChunker processes the document) and converts them into embedding-optimized text format.

## Architecture

### How It Works

```
PDF → DoclingDocument → HybridChunker → Chunks (with tables in doc_items)
                                            ↓
                                    Serialize tables from chunk.meta.doc_items
                                            ↓
                                    Key-value format for embeddings
```

**Key Points:**
- HybridChunker already extracts tables during chunking
- Tables are accessible via `chunk.meta.doc_items`
- We re-serialize them to key-value format for embeddings
- No pre-processing needed - work with chunks directly

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

- **Chunk-level extraction**: Access tables from chunk.meta.doc_items
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

### API Usage

```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@your_pdf.pdf" \
  -F "enable_chunking=true" \
  -F "serialize_tables=true" \
  -F "max_tokens_per_chunk=2048"
```

### Direct Table Serialization

```python
from app.services import serialize_table_from_chunk

# For a specific chunk
for chunk in chunker.chunk(document):
    if chunk_is_table(chunk):
        serialized = serialize_table_from_chunk(chunk)
        if serialized:
            print(serialized)
```

## API Reference

### `serialize_table_from_chunk(chunk: BaseChunk) -> Optional[str]`

Serialize table from a chunk's doc_items.

**Args:** BaseChunk object from HybridChunker
**Returns:** Serialized table text in key-value format, or None if no table found

**Example:**
```python
serialized = serialize_table_from_chunk(chunk)
if serialized:
    print("Table:", serialized[:100])
```

### `extract_table_structure(table_data: Any) -> Optional[Dict]`

Extract structured data from Docling's TableData object.

**Returns:** Dictionary with headers, rows, and grid information

### `format_table_as_keyvalue(headers, rows, caption) -> str`

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

## How It Works Internally

### 1. HybridChunker Processes Document

```python
# HybridChunker extracts tables during chunking
chunker = HybridChunker(tokenizer=tokenizer)
for chunk in chunker.chunk(document):
    # Chunk has tables in chunk.meta.doc_items
    # Each table has label='table' and item.data (TableData)
```

### 2. Detect Table Chunks

```python
# Check if chunk contains a table
if chunk.meta.doc_items:
    for item in chunk.meta.doc_items:
        if item.label == 'table':
            # Found a table!
```

### 3. Extract Table Structure

```python
# Access table data from item
table_data = item.data  # TableData object
grid = table_data.grid  # Grid structure

# Try multiple extraction methods:
# 1. grid.export_to_dataframe()
# 2. grid.export_to_list()
# 3. grid.cells iteration
# 4. Markdown export (fallback)
```

### 4. Serialize to Key-Value Format

```python
# Format as key-value pairs
for row in rows:
    pairs = [f"{header}: {value}" for header, value in zip(headers, row)]
    formatted_row = ', '.join(pairs)
```

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

## Example Output

### Before Serialization

```
"Apple M3 Max, Thread budget. = 4. Apple M3 Max, native backend.TTS = 177 s 167 s..."
```

### After Serialization

```
Table: Runtime characteristics
CPU: Apple M3 Max, Thread budget: 4, native backend TTS: 177 s, Pages/s: 1.27
CPU: Intel Xeon E5-2690, Thread budget: 16, native backend TTS: 375 s, Pages/s: 0.60
```

Much cleaner for embeddings!

## Troubleshooting

### Enable Debug Logging

```bash
LOG_LEVEL=DEBUG
```

Look for:
- `"Serialized table chunk N"` - Success
- `"Table chunk N could not be serialized"` - Failure
- `"Table serialization: X tables successfully serialized"` - Summary

### Common Issues

**Issue: No tables serialized**
- Check that `DOCLING_DO_TABLE_STRUCTURE=true` is set
- Verify `serialize_tables=True` is passed to chunk_document()
- Enable debug logging to see what's happening

**Issue: Tables not in key-value format**
- Check logs for serialization attempts
- Verify table has extractable structure (headers and rows)
- Some tables might not have grid data

**Issue: Extraction fails**
- Table might not have accessible grid structure
- Check if table.data exists
- Fallback to markdown export might be needed

### Debug Example

```python
# Check if chunk has table
if chunk.meta.doc_items:
    for item in chunk.meta.doc_items:
        if item.label == 'table':
            print(f"Found table!")
            print(f"Has data: {hasattr(item, 'data')}")
            if item.data:
                print(f"Has grid: {hasattr(item.data, 'grid')}")
```

## Benefits

| Benefit | Description |
|---------|-------------|
| **Correct API** | Uses actual Docling API (chunk.meta.doc_items) |
| **No Pre-processing** | Tables extracted during normal chunking |
| **Reliable** | Access structured data from chunks |
| **Simple** | Just enable serialize_tables=True |
| **Better Embeddings** | Key-value format more semantic than raw text |

## Version History

- **v2.1** (2025-10-21): Correct chunk-level extraction
  - Fixed to use `chunk.meta.doc_items` (correct Docling API)
  - Removed incorrect `document.body.iterate_items()` approach
  - Simplified architecture - work with chunks directly
  - More reliable and aligned with Docling docs

- **v2.0** (2025-10-21): Document-level extraction (incorrect)
  - Attempted to use non-existent document.body.iterate_items()
  - Over-complicated with pre-extraction

- **v1.x** (2025-10-21): Text parsing approach (deprecated)
  - Tried to parse flattened text back into structure

## Support

For questions or issues:
- Enable DEBUG logging to see extraction details
- Check that `DOCLING_DO_TABLE_STRUCTURE=true` is set
- Verify tables appear in chunk.meta.doc_items
- Share debug logs for troubleshooting
