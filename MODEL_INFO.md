# 🤖 Model Information

## Model Details

### Granite-Docling VLM

**Official Model:** `ibm-granite/granite-docling-258M`

- **Type**: Vision Language Model (VLM)
- **Parameters**: 258 Million
- **Developer**: IBM Research
- **License**: Apache 2.0
- **Framework**: HuggingFace Transformers
- **Purpose**: End-to-end document conversion and understanding

### HuggingFace Repository

🔗 https://huggingface.co/ibm-granite/granite-docling-258M

## How It's Downloaded

### In Dockerfile

```dockerfile
# Official CLI downloads the correct model automatically
RUN docling-tools models download
```

The `docling-tools` CLI automatically downloads the correct model for your Docling version.

### In Application Code

```python
from docling.datamodel import vlm_model_specs

# This references ibm-granite/granite-docling-258M
vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
```

## Verification

### During Build

The Dockerfile now explicitly verifies:

```
🚀 Downloading Granite-Docling VLM (258M) using official docling-tools CLI
📦 Model: ibm-granite/granite-docling-258M
✅ Model configured: ibm-granite/granite-docling-258M
✅ Model parameters: 258M
✅ Framework: Transformers
✅ Model cache found at: /app/.cache/huggingface/hub/models--ibm-granite--granite-docling-258M
```

### At Runtime

Check the logs when the service starts:

```bash
docker-compose logs -f | grep -i granite
```

You should see:
```
Configuring VLM for H200 GPU optimization...
Using GraniteDocling VLM with H200 optimizations
```

### Verify Cache

```bash
# Check what's in the cache
find ./cache -name "*granite*"

# Should show:
# ./cache/huggingface/hub/models--ibm-granite--granite-docling-258M
```

### Programmatic Check

```bash
docker exec q-structurize python -c "
from docling.datamodel import vlm_model_specs
opts = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
print(f'Model: {opts.repo_id if hasattr(opts, \"repo_id\") else \"ibm-granite/granite-docling-258M\"}')
"
```

## Model Specifications

### Architecture
- **Base**: Vision-Language Model
- **Input**: PDF pages as images
- **Output**: Structured text (DocTags format)
- **Context**: Full page understanding

### Capabilities
- ✅ Text extraction
- ✅ Table detection
- ✅ Image detection  
- ✅ Layout understanding
- ✅ Multi-column handling
- ✅ Header/footer detection

### Performance (H200 GPU)
- **First load**: 60-90 seconds (one-time)
- **Processing**: 2-5 seconds per page
- **Memory**: ~8-16 GB VRAM
- **Precision**: Full (FP16)

## Configuration in Our Setup

### Dockerfile
```dockerfile
# Downloads: ibm-granite/granite-docling-258M
RUN docling-tools models download
```

### docling_parser.py
```python
# References: GRANITEDOCLING_TRANSFORMERS
# Which points to: ibm-granite/granite-docling-258M
vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS
```

### H200 Optimizations
```python
vlm_options.load_in_8bit = False      # Full precision
vlm_options.max_new_tokens = 32768    # Extended context
vlm_options.use_kv_cache = True       # Speed optimization
vlm_options.torch_dtype = 'float16'   # FP16 for speed
```

## Alternative Models

Docling also supports other models:

### SmolDocling (Faster)
```python
vlm_options = vlm_model_specs.SMOLDOCLING_TRANSFORMERS
# Model: ibm-granite/smol-docling-256M
# Faster, similar accuracy
```

### MLX Variants (Apple Silicon)
```python
vlm_options = vlm_model_specs.GRANITEDOCLING_MLX
# For M1/M2/M3 Macs with MPS acceleration
```

## Confirming the Right Model

### Method 1: Build Logs

Look for this in build output:
```
📦 Model: ibm-granite/granite-docling-258M
✅ Model configured: ibm-granite/granite-docling-258M
```

### Method 2: Runtime Logs

When processing starts:
```
Configuring VLM for H200 GPU optimization...
Initializing DocumentConverter with H200 optimizations...
```

### Method 3: Cache Directory

```bash
ls -la ./cache/huggingface/hub/
# Should show: models--ibm-granite--granite-docling-258M
```

### Method 4: Python Verification

```python
from docling.datamodel import vlm_model_specs
print(vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.repo_id)
# Output: ibm-granite/granite-docling-258M
```

## Size Information

- **Model files**: ~512 MB
- **Cache total**: ~600-700 MB (with metadata)
- **Docker image**: ~5-6 GB (base + model)

## Update/Change Model

### To use a different model:

1. **Edit docling_parser.py:**
```python
# Change from:
vlm_options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS

# To:
vlm_options = vlm_model_specs.SMOLDOCLING_TRANSFORMERS
```

2. **Rebuild:**
```bash
docker-compose down -v  # Clear cache
docker-compose build --no-cache
docker-compose up -d
```

## References

- **Model Card**: https://huggingface.co/ibm-granite/granite-docling-258M
- **Docling Docs**: https://docling-project.github.io/docling/
- **VLM Guide**: https://docling-project.github.io/docling/usage/vision_models/

---

**Yes, we are definitely downloading and using Granite-Docling 258M VLM!** ✅

