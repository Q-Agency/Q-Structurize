# API Parameter Improvement - Individual Form Fields

## Overview
Changed the pipeline options from a single JSON string to **individual form fields**, significantly improving the API user experience.

## Changes Made

### Before (JSON String Approach)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F 'pipeline_options={"enable_ocr": true, "num_threads": 16, "table_mode": "accurate"}'
```

**Problems:**
- ❌ Hard to read and write
- ❌ Easy to make JSON syntax errors
- ❌ Poor Swagger UI experience (single text box)
- ❌ No automatic type validation
- ❌ No field-level descriptions in UI

### After (Individual Form Fields)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "enable_ocr=true" \
  -F "num_threads=16" \
  -F "table_mode=accurate"
```

**Benefits:**
- ✅ Clean and readable
- ✅ No JSON syntax errors possible
- ✅ Excellent Swagger UI (individual fields with dropdowns)
- ✅ Automatic FastAPI type validation
- ✅ Field-level descriptions and defaults visible

## New API Parameters

All parameters are now top-level form fields:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | PDF file to parse |
| `optimize_pdf` | Boolean | `true` | Pre-process PDF for better extraction |
| `enable_ocr` | Boolean | `false` | Enable OCR for scanned documents |
| `ocr_languages` | String | `"en"` | Comma-separated language codes (e.g., "en,es,de") |
| `table_mode` | String | `"fast"` | Table mode: "fast" or "accurate" |
| `do_table_structure` | Boolean | `true` | Enable table structure extraction |
| `do_cell_matching` | Boolean | `true` | Enable cell matching |
| `num_threads` | Integer | `8` | Processing threads (1-144) |
| `accelerator_device` | String | `"cpu"` | Device: "cpu", "cuda", or "auto" |

## Swagger UI Improvements

### Before
![Before: Single text field for JSON](docs/swagger-before.png)
- One text field for all options
- Manual JSON typing required
- No autocomplete
- No validation hints

### After
![After: Individual fields with descriptions](docs/swagger-after.png)
- Individual field for each option
- Dropdowns for enums (table_mode, accelerator_device)
- Checkboxes for booleans
- Numeric input with min/max for threads
- Each field shows description and default value
- **Try it out** works perfectly!

## Usage Examples

### 1. Default Processing (Minimal)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf"
```

### 2. OCR Enabled (Simple)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@scanned.pdf" \
  -F "enable_ocr=true"
```

### 3. High Performance (Focused)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "num_threads=64"
```

### 4. Complete Configuration (Advanced)
```bash
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "enable_ocr=true" \
  -F "ocr_languages=en,es,de" \
  -F "table_mode=accurate" \
  -F "do_cell_matching=true" \
  -F "num_threads=32"
```

### 5. Python Client (Cleaner)
```python
import requests

response = requests.post(
    "http://localhost:8878/parse/file",
    files={"file": open("document.pdf", "rb")},
    data={
        "enable_ocr": "true",
        "ocr_languages": "en,es",
        "num_threads": "16",
        "table_mode": "accurate"
    }
)
```

### 6. JavaScript/Fetch (Intuitive)
```javascript
const formData = new FormData();
formData.append('file', pdfFile);
formData.append('enable_ocr', 'true');
formData.append('num_threads', '16');
formData.append('table_mode', 'accurate');

const response = await fetch('http://localhost:8878/parse/file', {
  method: 'POST',
  body: formData
});
```

## Implementation Details

### Parameter Parsing
Language codes are now comma-separated strings:
```python
# In main.py
ocr_lang_list = [lang.strip() for lang in ocr_languages.split(',') if lang.strip()]

# Examples:
"en" → ["en"]
"en,es" → ["en", "es"]
"en, es, de" → ["en", "es", "de"]
```

### Validation
FastAPI automatically validates:
- `num_threads`: Must be 1-144 (via `ge=1, le=144`)
- `enable_ocr`: Must be boolean
- `table_mode`: Must be string
- Type mismatches return 422 with helpful error messages

### Backward Compatibility

⚠️ **Breaking Change**: This is **NOT backward compatible** with the old JSON string approach.

**Migration Guide for Users:**

**Old way (no longer works):**
```bash
-F 'pipeline_options={"enable_ocr": true, "num_threads": 16}'
```

**New way:**
```bash
-F "enable_ocr=true" -F "num_threads=16"
```

## Benefits Summary

### For End Users
1. **Easier to use** - No JSON syntax to remember
2. **Fewer errors** - Automatic type validation
3. **Better discovery** - All options visible in Swagger UI
4. **Cleaner code** - More readable in all languages

### For API Testing
1. **Swagger UI** - Individual fields with descriptions
2. **Postman** - Better form field support
3. **cURL** - Cleaner, more readable commands

### For Developers
1. **Type safety** - FastAPI validates everything
2. **Less parsing** - No JSON deserialization needed
3. **Clear contracts** - OpenAPI schema auto-generated
4. **Better docs** - Each field self-documenting

## Files Modified

1. ✅ `main.py`
   - Changed parameters from JSON string to individual Form fields
   - Updated parameter parsing logic
   - Updated docstring with new examples

2. ✅ `app/config/pipeline_config.py`
   - Updated usage examples
   - Added advanced curl example

3. ✅ `app/config/openapi_schema.py`
   - Updated all example commands
   - Added multilingual example

## Testing

### Test with Swagger UI
1. Go to `http://localhost:8878/docs`
2. Expand `/parse/file` endpoint
3. Click "Try it out"
4. See individual fields for each option! 🎉
5. Upload a PDF and test different configurations

### Test with cURL
```bash
# Test 1: Default
curl -X POST "http://localhost:8878/parse/file" -F "file=@test.pdf"

# Test 2: With OCR
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@test.pdf" \
  -F "enable_ocr=true" \
  -F "num_threads=16"

# Test 3: High performance
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@test.pdf" \
  -F "num_threads=64"
```

## Recommendations

### For Shared Server (Multiple Services)
```bash
# Use conservative defaults
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "num_threads=32"   # Leave ~100 cores for other services
```

### For Dedicated Server
```bash
# Use maximum performance
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@document.pdf" \
  -F "num_threads=144"  # Use all available cores
```

### For Scanned Documents
```bash
# Enable OCR with appropriate threading
curl -X POST "http://localhost:8878/parse/file" \
  -F "file=@scanned.pdf" \
  -F "enable_ocr=true" \
  -F "ocr_languages=en" \
  -F "num_threads=16"
```

## Version Information

- **API Version**: 2.1.0
- **Change Type**: Breaking change (new parameter format)
- **Migration Required**: Yes - update API calls to use individual form fields
- **Benefits**: Significantly improved UX and developer experience

## Next Steps

1. Update client applications to use new parameter format
2. Test thoroughly with different parameter combinations
3. Update any API documentation or tutorials
4. Announce breaking change to API consumers
5. Consider adding request examples to help with migration

## Summary

This change transforms the API from a developer-unfriendly JSON string approach to a clean, intuitive form field design that:
- ✅ Works better in Swagger UI
- ✅ Is easier to use in curl/Postman
- ✅ Provides better validation
- ✅ Is more discoverable
- ✅ Reduces errors
- ✅ Improves overall developer experience

The new design aligns with REST API best practices and makes the API much more accessible to users of all skill levels! 🚀

