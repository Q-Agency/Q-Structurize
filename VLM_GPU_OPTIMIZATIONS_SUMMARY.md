# VLM GPU Optimizations - Implementation Summary

## Overview

Successfully implemented **7 optimization steps** to accelerate Granite VLM PDF parsing on your NVIDIA H200 GPU.

**Current Performance:** 2 minutes for 9 pages (~13 sec/page) with 40-70% GPU utilization
**Target Performance:** 5-10 seconds for 9 pages (~0.5-1 sec/page) with 90%+ GPU utilization
**Expected Speedup:** 4-6x faster (combined optimizations)

---

## ✅ Completed Optimizations

### Step 1: PyTorch GPU Optimizations (TF32, cuDNN)
**File:** `app/services/vlm_parser.py` (lines 11-48)

**What was added:**
- TF32 matmul enabled for H200 (~2x speedup on matrix operations)
- cuDNN TF32 enabled
- cuDNN benchmark mode (auto-select optimal algorithms)
- GPU detection and logging

**Expected Impact:** ~2x speedup on matrix operations

**Logs to look for:**
```
🎮 GPU Configuration for VLM
✅ GPU detected: NVIDIA H200
💾 GPU memory: 141.0 GB
✅ TF32 enabled for matmul and cuDNN (balanced speed/quality)
✅ cuDNN benchmark mode enabled (optimal kernel selection)
```

---

### Step 2: VLM Pipeline GPU Acceleration
**File:** `app/services/vlm_parser.py` (lines 100-121)

**What was added:**
- PdfPipelineOptions with AcceleratorOptions configured
- Device set to CUDA
- Thread count optimized for GPU workloads (8 threads)

**Expected Impact:** Proper GPU utilization, better resource management

**Logs to look for:**
```
⚙️  Configuring VLM pipeline options...
✅ Accelerator configured: device=AcceleratorDevice.CUDA, threads=8
```

---

### Step 3: BF16 Mixed Precision
**File:** `app/services/vlm_parser.py` (lines 123-165)

**What was added:**
- BFloat16 precision configuration (better than FP16 for numerical stability)
- Model dtype configuration via pipeline_options.model_kwargs
- Device map to cuda:0

**Expected Impact:** ~2x speedup with minimal quality loss

**Logs to look for:**
```
🔢 VLM precision: bfloat16 (balanced speed/quality for H200)
✅ Model dtype configured: torch.bfloat16
```

---

### Step 4: Batch Processing Configuration
**File:** `app/services/vlm_parser.py` (lines 167-184)

**What was added:**
- Page batch size configuration (default: 4 pages)
- Parallel processing of multiple pages

**Expected Impact:** ~1.5x throughput improvement

**Logs to look for:**
```
✅ VLM batch size configured: 4 pages
```

---

### Step 5: Flash Attention 2
**Files:** 
- `requirements.txt` (line 14): Uncommented flash-attn
- `app/services/vlm_parser.py` (lines 148-156): Flash Attention configuration
- `Dockerfile.gpu` (lines 38-40): Build configuration

**What was added:**
- Flash Attention 2 dependency enabled
- Automatic detection and configuration
- Build environment variables for CUDA compilation

**Expected Impact:** 2-3x speedup on attention layers (significant for transformers)

**Logs to look for:**
```
⚡ Flash Attention 2 enabled (flash-attn 2.5.9.post1)
```

**Note:** Flash Attention requires compilation during Docker build (~5-10 minutes longer build time)

---

### Step 6: GPU Memory Monitoring
**File:** `app/services/vlm_parser.py` (lines 217-285)

**What was added:**
- GPU memory tracking before/after parsing
- Peak memory monitoring
- Memory usage per parsing operation
- Detailed timing breakdowns (conversion vs export)

**Expected Impact:** Better debugging and optimization tracking

**Logs to look for:**
```
🎮 GPU memory before parsing: X.XX GB
⏱️  Document conversion: X.XXs
⏱️  Markdown export: X.XXs
🎮 GPU memory after parsing: X.XX GB
🎮 GPU memory peak: X.XX GB
🎮 GPU memory used for this parsing: X.XX GB
✅ VLM parsing complete in X.XXs total
```

---

### Step 7: Dockerfile Environment Variables
**File:** `Dockerfile.gpu`

**What was added:**
- `FLASH_ATTENTION_FORCE_BUILD=TRUE` (line 39)
- `MAX_JOBS=8` (line 40)
- `DOCLING_VLM_BATCH_SIZE=4` (line 95)
- `DOCLING_VLM_DTYPE=bfloat16` (line 96)
- `DOCLING_USE_FLASH_ATTENTION=1` (line 97)

**Expected Impact:** Consolidated configuration, easier to adjust settings

---

## 🧪 Testing Instructions

### 1. Rebuild Docker Image

**Important:** First build will take ~15-20 minutes due to Flash Attention compilation.

```bash
cd /Users/zmatokanovic/development/QStructurize

# Build the GPU image
docker-compose build

# Start the service
docker-compose up -d

# Watch the logs for optimization confirmations
docker-compose logs -f q-structurize
```

### 2. Verify GPU Optimizations in Logs

Look for these sections in the startup logs:

```
🎮 GPU Configuration for VLM
============================================================
✅ GPU detected: NVIDIA H200
💾 GPU memory: 141.0 GB
✅ TF32 enabled for matmul and cuDNN
✅ cuDNN benchmark mode enabled
============================================================

============================================================
🚀 Initializing VLM DocumentConverter (MINIMAL)
============================================================
⚙️  Configuring VLM pipeline options...
✅ Accelerator configured: device=AcceleratorDevice.CUDA, threads=8
🔢 VLM precision: bfloat16 (balanced speed/quality for H200)
⚡ Flash Attention 2 enabled (flash-attn 2.5.9.post1)
✅ Model dtype configured: torch.bfloat16
✅ VLM batch size configured: 4 pages
```

### 3. Test with Your 9-Page PDF

```bash
curl -X POST "http://localhost:8000/parse/file" \
  -F "file=@your-9-page-pdf.pdf" \
  -F "use_vlm=true"
```

### 4. Monitor GPU Utilization

In a separate terminal:

```bash
watch -n 1 nvidia-smi
```

**What to look for:**
- GPU utilization should spike to 80-100% during parsing
- GPU memory usage should be visible
- Should see activity on GPU 0

### 5. Check Performance Improvement

Look for the timing logs:

```
✅ VLM parsing complete in X.XXs total
```

**Expected results:**
- **Before:** ~120 seconds (2 minutes)
- **After:** ~10-30 seconds (4-12x improvement)
- **Best case:** ~5-10 seconds with all optimizations working

---

## 📊 Performance Breakdown

| Optimization | Expected Speedup | Cumulative |
|-------------|------------------|------------|
| TF32 matmul | 2x | 2x |
| BF16 precision | 2x | 4x |
| Flash Attention 2 | 2-3x | 8-12x |
| Batch processing | 1.5x | 12-18x |
| GPU configuration | Better utilization | - |

**Note:** Actual speedup will depend on:
- Whether docling's VlmPipeline supports all these features
- PDF complexity and content
- CPU preprocessing bottlenecks

---

## 🔧 Troubleshooting

### Flash Attention Build Fails

If Flash Attention compilation fails during Docker build:

**Option 1:** Build without Flash Attention (still get 3-4x improvement)
```bash
# Comment out line 14 in requirements.txt
sed -i 's/^flash-attn/#flash-attn/' requirements.txt
docker-compose build
```

**Option 2:** Try different build flags
```dockerfile
# In Dockerfile.gpu, change line 40
ENV MAX_JOBS=4  # Reduce if running out of memory during build
```

### GPU Not Detected

Check Docker GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this fails, reinstall nvidia-container-toolkit.

### Lower Than Expected Speedup

**Check:**
1. Flash Attention actually loaded (check logs for "⚡ Flash Attention 2 enabled")
2. BF16 configured (check logs for "🔢 VLM precision: bfloat16")
3. GPU utilization reaches 80%+ (use `nvidia-smi`)
4. No CPU bottlenecks (check if conversion time << total time)

**Possible causes:**
- docling's VlmPipeline may not support all configuration options
- PDF preprocessing still happening on CPU
- Model loading from disk instead of GPU memory

### Quality Issues with BF16

If output quality degrades:

```dockerfile
# In Dockerfile.gpu, change line 96
DOCLING_VLM_DTYPE=float32
```

This will be slower but maintain full precision.

---

## 🎯 Next Steps

1. **Test with your 9-page PDF** and measure actual improvement
2. **Monitor GPU utilization** to ensure it's at 80-100%
3. **Compare output quality** to ensure BF16 doesn't affect accuracy
4. **If needed:** Adjust batch size (DOCLING_VLM_BATCH_SIZE) for your use case

### If You Need MORE Speed

If 10-30 seconds isn't fast enough:

1. **Dual-GPU pipeline parallelism** - split pages across both H200s (~50% faster)
2. **Larger batch size** - increase DOCLING_VLM_BATCH_SIZE to 8 or 16
3. **TensorRT-LLM** - compile model with TensorRT for maximum speed
4. **Custom VLM implementation** - direct PyTorch/transformers control

Let me know the results and we can further optimize! 🚀

---

## 📝 Files Modified

- ✅ `app/services/vlm_parser.py` - GPU optimizations, monitoring
- ✅ `requirements.txt` - Flash Attention enabled
- ✅ `Dockerfile.gpu` - Build config and env variables

No other files were changed. Your standard PDF parser is unaffected.

