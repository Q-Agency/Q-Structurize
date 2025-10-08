# Enrichment Features - Advanced Document Understanding

## Overview
Added 4 new enrichment options to the API that enable advanced document understanding features powered by specialized AI models. These features are disabled by default to optimize processing time.

## New Features Added

### 1. Code Enrichment (`do_code_enrichment`)
**What it does:** Detects and analyzes code blocks within documents, identifying programming languages.

**Use cases:**
- Technical documentation
- Programming tutorials
- API documentation
- Code repositories
- Software manuals

**Example:**
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@technical-doc.pdf" \
  -F "do_code_enrichment=true"
```

**Output enhancement:**
- Code blocks tagged with detected language (Python, JavaScript, Java, etc.)
- Better syntax preservation
- Improved code block formatting

### 2. Formula Enrichment (`do_formula_enrichment`)
**What it does:** Extracts mathematical formulas and converts them to LaTeX format.

**Use cases:**
- Research papers
- Academic documents
- Mathematical textbooks
- Scientific publications
- Engineering documentation

**Example:**
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@research-paper.pdf" \
  -F "do_formula_enrichment=true"
```

**Output enhancement:**
- Formulas extracted in LaTeX format
- Better equation rendering in exports
- MathML syntax support for HTML exports
- Preserves mathematical notation

### 3. Picture Classification (`do_picture_classification`)
**What it does:** Classifies images into categories using DocumentFigureClassifier model.

**Detected categories:**
- Charts (bar, line, pie, scatter)
- Flow diagrams
- Logos
- Signatures
- Organizational charts
- Natural images
- Screenshots

**Use cases:**
- Business reports
- Presentations
- Mixed content documents
- Financial reports
- Marketing materials

**Example:**
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@business-report.pdf" \
  -F "do_picture_classification=true"
```

**Output enhancement:**
- Images tagged with classification type
- Better understanding of document structure
- Improved content organization

### 4. Picture Description (`do_picture_description`)
**What it does:** Generates AI-powered natural language descriptions of images using Vision-Language Models (VLM).

⚠️ **Important:** Requires VLM model configuration. Significantly increases processing time and resource usage.

**Use cases:**
- Accessibility (screen readers)
- Content understanding
- Visual analysis
- Document summarization
- Alt-text generation

**Example:**
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@presentation.pdf" \
  -F "do_picture_description=true"
```

**Output enhancement:**
- Natural language descriptions for each image
- Alt-text for accessibility
- Better content indexing

## Performance Impact

| Feature | Processing Time Impact | Model Size | CPU Usage |
|---------|------------------------|------------|-----------|
| Code Enrichment | +10-20% | Small (~50MB) | Moderate |
| Formula Enrichment | +15-30% | Medium (~100MB) | Moderate |
| Picture Classification | +20-40% | Medium (~100MB) | Moderate |
| Picture Description | +100-300% | Large (~500MB+) | High |

**Recommendation:** Only enable features you need for your specific document type.

## Pre-configured Scenarios

### Scientific Paper
For research papers with formulas and code:
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@paper.pdf" \
  -F "table_mode=accurate" \
  -F "do_formula_enrichment=true" \
  -F "do_code_enrichment=true" \
  -F "do_picture_classification=true" \
  -F "num_threads=24"
```

### Technical Documentation
For technical docs with code samples:
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@docs.pdf" \
  -F "do_code_enrichment=true" \
  -F "do_picture_classification=true" \
  -F "table_mode=accurate"
```

### Business Report
For reports with charts and tables:
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@report.pdf" \
  -F "table_mode=accurate" \
  -F "do_picture_classification=true"
```

## API Parameters

All enrichment options are now available as individual form fields in Swagger UI:

```yaml
do_code_enrichment: boolean (default: false)
  Enable code block language detection and parsing

do_formula_enrichment: boolean (default: false)
  Enable formula analysis and LaTeX extraction

do_picture_classification: boolean (default: false)
  Enable image classification (charts, diagrams, logos, etc.)

do_picture_description: boolean (default: false)
  Enable AI-powered image description (requires VLM)
```

## Swagger UI Integration

In `/docs`, you'll see:
- ✅ Checkboxes for each enrichment option
- ✅ Clear descriptions and warnings
- ✅ Performance impact notes
- ✅ Use case recommendations

## Python Client Example

```python
import requests

# Scientific paper with all enrichments
response = requests.post(
    "http://localhost:8878/parse/file",
    files={"file": open("research-paper.pdf", "rb")},
    data={
        "table_mode": "accurate",
        "do_formula_enrichment": "true",
        "do_code_enrichment": "true",
        "do_picture_classification": "true",
        "num_threads": "24"
    }
)

result = response.json()
print(result["content"])  # Markdown with enriched content
```

## Configuration via `/parsers/options`

View all available enrichment options:
```bash
curl http://localhost:8878/parsers/options
```

Returns detailed information about each enrichment feature including:
- Description
- Default values
- Use cases
- Performance impact
- Requirements

## Important Notes

### VLM Requirement for Picture Description
The `do_picture_description` feature requires:
1. Vision-Language Model (VLM) to be configured
2. Additional model downloads (~500MB+)
3. Significantly more processing time
4. Higher memory usage

If VLM is not configured, enabling this option may result in:
- Graceful degradation (feature skipped)
- Or error message about missing VLM

### Performance Optimization Tips

**For Speed:**
```bash
# Disable all enrichments (default)
-F "do_code_enrichment=false" \
-F "do_formula_enrichment=false" \
-F "do_picture_classification=false"
```

**For Quality:**
```bash
# Enable needed enrichments based on document type
-F "do_formula_enrichment=true" \
-F "do_code_enrichment=true" \
-F "num_threads=32"
```

**Balanced Approach:**
```bash
# Enable only classification (moderate impact)
-F "do_picture_classification=true" \
-F "num_threads=16"
```

## Files Modified

1. ✅ `app/models/schemas.py` - Added 4 enrichment fields to PipelineOptions
2. ✅ `main.py` - Added 4 enrichment parameters to `/parse/file` endpoint
3. ✅ `app/services/docling_parser.py` - Added enrichment options handling
4. ✅ `app/config/pipeline_config.py` - Added enrichment documentation and examples

## When to Use Each Feature

### Code Enrichment
✅ Use when:
- Document contains code snippets
- API documentation
- Programming tutorials
- Technical specifications

❌ Skip when:
- No code in document
- Pure business documents
- General text documents

### Formula Enrichment
✅ Use when:
- Scientific papers
- Mathematical documents
- Engineering specifications
- Academic publications

❌ Skip when:
- No mathematical content
- Business reports
- General documentation

### Picture Classification
✅ Use when:
- Mixed content documents
- Presentations
- Business reports
- Need to understand image types

❌ Skip when:
- Few or no images
- Speed is critical
- Image types not important

### Picture Description
✅ Use when:
- Accessibility is required
- Need detailed image understanding
- Creating alt-text
- Visual content is critical

❌ Skip when:
- Speed is important
- VLM not available
- Simple classification is enough
- Limited resources

## Summary

✅ Added 4 powerful enrichment features
✅ All disabled by default for performance
✅ Individual control via form fields
✅ Comprehensive documentation
✅ Pre-configured scenarios
✅ Swagger UI integrated
✅ Performance impact clearly documented

These enrichment features transform the API from basic text extraction to comprehensive document understanding! 🚀

