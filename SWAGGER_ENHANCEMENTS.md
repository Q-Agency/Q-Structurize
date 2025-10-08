# Swagger UI Enhancements

## Overview
Enhanced the Swagger/OpenAPI documentation to provide a comprehensive, user-friendly interface for the configurable pipeline options.

## What Was Added

### 1. Enhanced Landing Page Description
The main Swagger UI page now includes:
- **Key Features** overview with emojis for visual clarity
- **Quick Examples** with ready-to-use curl commands
- **Configuration Guide** pointer to `/parsers/options` endpoint
- **Hardware Optimization** tips for 72-core Xeon
- **Language Support** information

### 2. Interactive Pipeline Options Examples
Added **6 pre-configured example scenarios** in a dropdown menu for the `pipeline_options` parameter:

1. **Default (Fast, No OCR)** - Fastest processing with default settings
2. **Enable OCR** - For scanned documents with English text
3. **Accurate Tables** - For complex table structures
4. **High Performance** - Maximum speed using 64 threads
5. **Multilingual OCR** - For documents with multiple languages
6. **Complete Extraction** - OCR + accurate tables + multi-threading

Users can select any example from the dropdown and it will populate the field automatically!

### 3. Detailed Response Codes
Added comprehensive HTTP response documentation:
- **200**: Success with example response
- **400**: Invalid request parameters
- **415**: Invalid file type
- **500**: Server error
- **503**: Parser service unavailable

### 4. Response Examples
Included a sample response showing markdown output format with:
- Headings
- Paragraphs
- Table structure

### 5. Tag Descriptions
Organized endpoints into logical groups:
- **PDF Parsing** - Main parsing functionality
- **System** - Configuration and info endpoints
- **Health** - Status checks

### 6. Custom OpenAPI Function
Created `custom_openapi()` function that:
- Provides rich markdown documentation
- Adds logo reference for Docling
- Enhances tag metadata
- Adds examples and usage instructions

## Benefits

### For Users:
- 🎯 **One-Click Examples** - Select pre-configured scenarios from dropdown
- 📖 **Rich Documentation** - Comprehensive guides right in Swagger UI
- 💡 **Visual Examples** - See exactly how to format requests
- 🔍 **Discoverable Options** - Easy to find all configuration parameters

### For Developers:
- 📝 **Self-Documenting API** - Less need for external docs
- 🚀 **Faster Onboarding** - New users can experiment immediately
- 🎨 **Professional Appearance** - Polished, production-ready look

## How to Use

### Access Swagger UI
```
http://localhost:8878/docs
```

### Try the Examples
1. Navigate to `/parse/file` endpoint
2. Click "Try it out"
3. Upload a PDF file
4. Click the dropdown on `pipeline_options`
5. Select any pre-configured example
6. Click "Execute"

### View Available Options
Navigate to `/parsers/options` endpoint for complete reference with:
- All available configuration parameters
- Default values and ranges
- Detailed descriptions
- Example configurations
- Usage examples in multiple languages

## Comparison: Before vs After

### Before
```
pipeline_options: string (optional)
Example: {"enable_ocr": true, "num_threads": 16}
```

### After
```
pipeline_options: string (optional)

📋 EXAMPLES DROPDOWN:
  - Default (Fast, No OCR)
  - Enable OCR
  - Accurate Tables
  - High Performance
  - Multilingual OCR
  - Complete Extraction

Each with full JSON ready to use!
```

## Testing the Enhancements

1. **Start the API**:
```bash
docker-compose up -d
```

2. **Access Swagger UI**:
```
http://localhost:8878/docs
```

3. **Explore the Documentation**:
   - Read the enhanced landing page description
   - Try the example scenarios
   - Test different pipeline configurations
   - View response examples

4. **Check ReDoc (Alternative View)**:
```
http://localhost:8878/redoc
```

## OpenAPI Schema
The enhanced OpenAPI schema is now available at:
```
http://localhost:8878/openapi.json
```

## Notes

- All examples are functional and tested
- The dropdown menu makes it incredibly easy for users to experiment
- Documentation is inline and always up-to-date
- No external documentation files needed for basic usage
- Professional appearance suitable for production deployment

## Future Enhancements (Optional)

Could add in the future:
- More example scenarios
- Performance benchmarks in documentation
- Visual diagrams of the pipeline
- Video tutorials embedded in Swagger
- Cost/time estimates for each configuration

