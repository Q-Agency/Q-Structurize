# Environment-Based Configuration Guide

Q-Structurize uses **ENV-based configuration** for optimal performance with pre-initialized models.

## Why ENV-Based Configuration?

- 🚀 **Instant Processing** - Models loaded once at startup, zero per-request overhead
- ⚡ **Consistent Performance** - Same optimized settings for all requests
- 🔄 **Fast Reconfiguration** - Docker rebuild takes only ~10 seconds (with cache)

## How to Change Configuration

### 1. Edit Dockerfile

Modify ENV variables in `Dockerfile` (lines 15-30):

```dockerfile
ENV OMP_NUM_THREADS=120 \
    MKL_NUM_THREADS=120 \
    OPENBLAS_NUM_THREADS=120 \
    NUMEXPR_NUM_THREADS=120 \
    TORCH_NUM_THREADS=120
```

### 2. Rebuild Container

```bash
docker-compose build  # Takes ~10 seconds with cache!
```

### 3. Restart Service

```bash
docker-compose up
```

## All Configuration Options (Dockerfile ENV)

**All settings are in Dockerfile lines 14-40.** Edit these ENV variables and rebuild.

### Threading Configuration

```dockerfile
# Number of CPU threads for parallel processing
OMP_NUM_THREADS=100          # OpenMP threads (default: 100)
MKL_NUM_THREADS=100          # Intel MKL threads
OPENBLAS_NUM_THREADS=100     # OpenBLAS threads
NUMEXPR_NUM_THREADS=100      # NumExpr threads
TORCH_NUM_THREADS=100        # PyTorch threads (also set in Python)
```

**Recommendations:**
- **Conservative (70%)**: 100 threads - Shared servers
- **Balanced (85%)**: 120 threads - **Recommended**
- **Maximum (95%)**: 140 threads - Dedicated server

### OCR Configuration

```dockerfile
DOCLING_ENABLE_OCR=false     # Enable/disable OCR (true/false)
DOCLING_OCR_LANGUAGES=en     # Comma-separated language codes (e.g., "en,es,de")
```

### Table Extraction Configuration

```dockerfile
DOCLING_DO_TABLE_STRUCTURE=false   # Enable/disable table extraction (true/false)
DOCLING_TABLE_MODE=fast            # Table mode: "fast" or "accurate"
DOCLING_DO_CELL_MATCHING=false     # Enable cell matching (true/false)
```

### Enrichment Features

```dockerfile
DOCLING_DO_CODE_ENRICHMENT=false         # Code block detection (true/false)
DOCLING_DO_FORMULA_ENRICHMENT=false      # LaTeX formula extraction (true/false)
DOCLING_DO_PICTURE_CLASSIFICATION=false  # Image classification (true/false)
DOCLING_DO_PICTURE_DESCRIPTION=false     # AI image description (true/false, requires VLM)
```

### Batching Configuration (Performance Tuning)

```dockerfile
DOCLING_LAYOUT_BATCH_SIZE=32    # Layout detection batch size (1-32)
DOCLING_OCR_BATCH_SIZE=32       # OCR processing batch size (1-32)
DOCLING_TABLE_BATCH_SIZE=32     # Table extraction batch size (1-32)
DOCLING_QUEUE_MAX_SIZE=1000     # Queue size for backpressure (10-1000)
DOCLING_BATCH_TIMEOUT=0.5       # Batch timeout in seconds (0.1-30.0)
```

### Accelerator Configuration

```dockerfile
DOCLING_ACCELERATOR_DEVICE=cpu   # Device: "cpu", "cuda", or "auto"
```

## Example Configurations

### Fast Processing (No OCR, No Tables) - DEFAULT

**Best for:** Clean PDFs with good text layer, maximum speed

**Edit Dockerfile:**
```dockerfile
# Lines 14-40 - Already configured as default!
OMP_NUM_THREADS=100
DOCLING_ENABLE_OCR=false
DOCLING_DO_TABLE_STRUCTURE=false
DOCLING_LAYOUT_BATCH_SIZE=32
DOCLING_QUEUE_MAX_SIZE=1000
DOCLING_BATCH_TIMEOUT=0.5
```

**Rebuild:**
```bash
docker-compose build && docker-compose up
```

---

### Full Processing (OCR + Tables)

**Best for:** Scanned documents, documents with tables

**Edit Dockerfile (lines 14-40):**
```dockerfile
OMP_NUM_THREADS=120                      # More threads for OCR
DOCLING_ENABLE_OCR=true                  # ← Enable OCR
DOCLING_OCR_LANGUAGES=en                 # Language
DOCLING_DO_TABLE_STRUCTURE=true          # ← Enable tables
DOCLING_TABLE_MODE=accurate              # ← Accurate mode
DOCLING_DO_CELL_MATCHING=true            # ← Better accuracy
DOCLING_LAYOUT_BATCH_SIZE=16             # Lower batch for heavy processing
DOCLING_OCR_BATCH_SIZE=16
DOCLING_TABLE_BATCH_SIZE=16
DOCLING_QUEUE_MAX_SIZE=500
DOCLING_BATCH_TIMEOUT=1.0
```

**Rebuild:**
```bash
docker-compose build && docker-compose up
```

---

### Multilingual OCR

**Best for:** Documents in multiple languages

**Edit Dockerfile (lines 14-40):**
```dockerfile
OMP_NUM_THREADS=140
DOCLING_ENABLE_OCR=true
DOCLING_OCR_LANGUAGES=en,es,de,fr        # ← Multiple languages
DOCLING_DO_TABLE_STRUCTURE=true
DOCLING_LAYOUT_BATCH_SIZE=12             # Lower for multilingual
DOCLING_OCR_BATCH_SIZE=12
DOCLING_TABLE_BATCH_SIZE=12
DOCLING_QUEUE_MAX_SIZE=500
```

**Rebuild:**
```bash
docker-compose build && docker-compose up
```

---

### Maximum Throughput

**Best for:** High-volume processing, batch workloads

**Edit Dockerfile (lines 14-40):**
```dockerfile
OMP_NUM_THREADS=140                      # Maximum CPU
DOCLING_ENABLE_OCR=false
DOCLING_DO_TABLE_STRUCTURE=false
DOCLING_LAYOUT_BATCH_SIZE=32             # Maximum batching
DOCLING_QUEUE_MAX_SIZE=1000              # Large queue
DOCLING_BATCH_TIMEOUT=0.1                # Minimal wait
```

**Also add to Dockerfile (line 20):**
```dockerfile
KMP_BLOCKTIME=0                          # Minimal thread wait
```

**Rebuild:**
```bash
docker-compose build && docker-compose up
```

---

### Scientific Papers (Formulas + Code)

**Best for:** Research papers, technical documentation

**Edit Dockerfile (lines 14-40):**
```dockerfile
OMP_NUM_THREADS=120
DOCLING_ENABLE_OCR=false
DOCLING_DO_TABLE_STRUCTURE=true
DOCLING_TABLE_MODE=accurate
DOCLING_DO_CODE_ENRICHMENT=true          # ← Code detection
DOCLING_DO_FORMULA_ENRICHMENT=true       # ← Formula extraction
DOCLING_DO_PICTURE_CLASSIFICATION=true   # ← Image classification
DOCLING_LAYOUT_BATCH_SIZE=16
DOCLING_TABLE_BATCH_SIZE=16
```

**Rebuild:**
```bash
docker-compose build && docker-compose up
```

## Performance Tuning Guide

### Thread Count (`OMP_NUM_THREADS`)

| Cores Used | Thread Count | Use Case |
|-----------|--------------|----------|
| 70% | 100 | Shared server, leave headroom |
| 85% | 120 | **Recommended default** |
| 95% | 140 | Dedicated server, maximum performance |

### Batch Sizes

| Batch Size | Memory | Latency | Throughput | Use Case |
|-----------|--------|---------|------------|----------|
| 1-4 | Low | Low | Low | Interactive, low-latency |
| 8-16 | Medium | Medium | Medium | **Balanced default** |
| 24-32 | High | High | High | Batch processing, high-volume |

### Queue Size

| Queue Size | Memory | Use Case |
|-----------|--------|----------|
| 50-100 | Low | Small documents |
| 200-500 | Medium | **Default** |
| 500-1000 | High | Large multi-page documents |

## Monitoring & Validation

After rebuilding, check the startup logs to verify your ENV configuration was loaded:

```bash
docker-compose up
```

Look for the configuration summary:
```
✅ PyTorch threading configured: 100 intra-op, 10 inter-op threads
============================================================
🚀 Initializing Docling DocumentConverter (ONE-TIME SETUP)
⚙️  Configuration (from Dockerfile ENV):
   📊 Threading:
      - Threads: 100 (OMP_NUM_THREADS)
      - Device: cpu
   🚀 Batching:
      - Layout Batch: 32
      - OCR Batch: 32
      - Table Batch: 32
      - Queue Max: 1000
      - Batch Timeout: 0.5s
   📝 Features:
      - OCR: ❌ Disabled
      - Tables: ❌ Disabled
      - Code Enrichment: ❌
      - Formula Enrichment: ❌
      - Picture Classification: ❌
📦 Pre-loading models into memory...
✅ Converter initialized in 12.34 seconds
📝 Models are now cached in memory for fast processing
============================================================
```

**This confirms all your ENV variables were read correctly!**

## Supported OCR Languages

Common language codes for `ocr_languages`:

- `en` - English
- `es` - Spanish
- `de` - German
- `fr` - French
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `zh` - Chinese (Simplified)
- `ja` - Japanese
- `ko` - Korean
- `ar` - Arabic
- `hi` - Hindi

[See full list of supported languages](https://www.jaided.ai/easyocr/)

## Troubleshooting

### "Out of memory" errors

**Solution:** Reduce batch sizes and/or thread count

```python
"layout_batch_size": 8,   # Down from 32
"queue_max_size": 200,    # Down from 1000
```

```dockerfile
ENV OMP_NUM_THREADS=80    # Down from 120
```

### "Processing too slow"

**Solution:** Increase thread count and batch sizes

```dockerfile
ENV OMP_NUM_THREADS=140   # Up from 100
```

```python
"layout_batch_size": 32,  # Up from 16
```

### "First request takes long time"

**Expected:** First request initializes models (~5-30 seconds depending on cache)

**Solution:** Pre-download models during build (already configured in Dockerfile)

### Models not found

**Solution:** Models are downloaded automatically on first use. Check:
```bash
docker exec -it q-structurize ls -lah /root/.cache/docling/models/
```

If empty, models will download on first API call (~400MB, 3-5 minutes).

