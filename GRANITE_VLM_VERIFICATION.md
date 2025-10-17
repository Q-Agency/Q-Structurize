# Granite VLM Verification Guide

This document explains how to verify that your QStructurize deployment is using the IBM Granite VLM model (`ibm-granite/granite-docling-258M`).

## Quick Verification

### 1. Check Docker Logs on Startup

When your container starts, look for these logs in the VLM initialization:

```
============================================================
🚀 Initializing VLM DocumentConverter (MINIMAL)
============================================================
🔍 DOCLING_VLM_MODEL environment variable: /opt/models/granite-docling-258M
✅ Granite VLM model path exists: /opt/models/granite-docling-258M
📁 Model directory contains: config.json, model.safetensors, tokenizer.json, ...
🌐 HF_HUB_OFFLINE: 1, TRANSFORMERS_OFFLINE: 1
📦 Pre-loading VLM model (ibm-granite/granite-docling-258M)...
✅ VLM Pipeline type: VlmPipeline
✅ VLM converter initialized in X.XX seconds
🎯 Using IBM Granite Docling VLM (granite-docling-258M)
============================================================
```

### 2. Use the Verification API Endpoint

Once your service is running, call the dedicated verification endpoint:

```bash
curl http://localhost:8000/parsers/vlm/verify-granite
```

**Expected Response (Success):**
```json
{
  "status": "verified",
  "message": "Granite VLM model verified successfully",
  "verification": {
    "is_granite": true,
    "model_path": "/opt/models/granite-docling-258M",
    "model_exists": true,
    "env_configured": true,
    "model_files": [
      "config.json",
      "model.safetensors",
      "tokenizer.json",
      "tokenizer_config.json",
      "special_tokens_map.json",
      "preprocessor_config.json",
      "generation_config.json"
    ],
    "offline_mode": true,
    "model_type": "mllama",
    "architectures": ["MllamaForConditionalGeneration"]
  },
  "expected_model": "ibm-granite/granite-docling-258M",
  "recommendation": null
}
```

### 3. Check General Parser Info

Get comprehensive parser information including Granite verification:

```bash
curl http://localhost:8000/parsers/info
```

The response will include a `vlm_parser` section with `granite_verification` embedded:

```json
{
  "standard_parser": { ... },
  "vlm_parser": {
    "available": true,
    "library": "docling",
    "pipeline": "VlmPipeline (GraniteDocling default)",
    "description": "Vision Language Model for end-to-end PDF parsing",
    "model": "ibm-granite/granite-docling-258M (default)",
    "backend": "Transformers",
    "granite_verification": {
      "is_granite": true,
      "model_path": "/opt/models/granite-docling-258M",
      "model_exists": true,
      "env_configured": true,
      ...
    }
  }
}
```

## Manual Verification Inside Container

If you want to manually inspect the model inside your running container:

```bash
# Exec into the container
docker exec -it qstructurize-app bash

# Check environment variable
echo $DOCLING_VLM_MODEL
# Expected: /opt/models/granite-docling-258M

# List model files
ls -lah /opt/models/granite-docling-258M/
# Should show: config.json, model.safetensors, tokenizer files, etc.

# Read model config to confirm architecture
cat /opt/models/granite-docling-258M/config.json | grep -E "model_type|architectures"
# Expected: "model_type": "mllama" and "architectures": ["MllamaForConditionalGeneration"]
```

## Configuration Files

The Granite VLM model is configured in several places:

### 1. Dockerfile.gpu (lines 42-45)
```dockerfile
RUN huggingface-cli download ibm-granite/granite-docling-258M \
      --local-dir /opt/models/granite-docling-258M \
      --local-dir-use-symlinks False
```

### 2. Dockerfile.gpu (line 90)
```dockerfile
ENV DOCLING_VLM_MODEL=/opt/models/granite-docling-258M
```

### 3. app/services/vlm_parser.py
The VLM parser automatically uses the model specified by `DOCLING_VLM_MODEL` environment variable.

## Troubleshooting

### Model Not Found
If verification shows `model_exists: false`:
1. Ensure Docker image was built completely
2. Check that the model download step in Dockerfile succeeded
3. Rebuild with: `docker-compose build --no-cache`

### Wrong Model Being Used
If you suspect a different model is being used:
1. Check `DOCLING_VLM_MODEL` environment variable
2. Review Docker build logs for download errors
3. Verify HuggingFace cache isn't interfering (check HF_HOME)

### Offline Mode Issues
If `offline_mode: false` but you expect `true`:
1. Check `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` environment variables
2. These should be set to `1` in production for offline operation

## Performance Indicators

When using Granite VLM:
- **Initialization**: 10-30 seconds (model loading)
- **First inference**: May take longer due to model warmup
- **Subsequent inferences**: Faster, especially on GPU
- **Model size**: ~258MB for granite-docling-258M

## References

- Model on HuggingFace: https://huggingface.co/ibm-granite/granite-docling-258M
- Docling Documentation: https://github.com/DS4SD/docling
- VLM Pipeline: Uses `VlmPipeline` from docling library

