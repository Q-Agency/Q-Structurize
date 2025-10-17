# Quick Test Guide - VLM GPU Optimizations

## 🚀 Quick Start (3 Commands)

```bash
# 1. Rebuild with optimizations
docker-compose build

# 2. Start service
docker-compose up -d

# 3. Test with your PDF
curl -X POST "http://localhost:8000/parse/file" \
  -F "file=@your-pdf.pdf" \
  -F "use_vlm=true"
```

---

## 📊 Monitor Performance

### Terminal 1: Watch Logs
```bash
docker-compose logs -f q-structurize | grep -E "(GPU|VLM|⏱️|✅)"
```

### Terminal 2: Watch GPU
```bash
watch -n 1 nvidia-smi
```

---

## ✅ Success Indicators

**In startup logs, you should see:**
```
✅ GPU detected: NVIDIA H200
✅ TF32 enabled
⚡ Flash Attention 2 enabled
✅ Model dtype configured: torch.bfloat16
✅ VLM batch size configured: 4 pages
```

**During parsing:**
```
🎮 GPU memory before parsing: X.XX GB
⏱️  Document conversion: X.XXs
🎮 GPU memory peak: X.XX GB
✅ VLM parsing complete in X.XXs total
```

**GPU monitoring:**
- GPU utilization: 80-100% during parsing
- GPU memory: Should show usage
- Power: Should increase during parsing

---

## 📈 Expected Results

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| **Time (9 pages)** | 120s | 10-30s | 5-10s |
| **Time per page** | 13s | 1-3s | 0.5-1s |
| **GPU utilization** | 40-70% | 80-100% | 90%+ |

---

## ⚠️ If Build Fails

Flash Attention compilation can be tricky. If it fails:

```bash
# Disable Flash Attention temporarily
sed -i 's/^flash-attn/#flash-attn/' requirements.txt

# Rebuild
docker-compose build

# You'll still get 3-4x improvement without it!
```

---

## 🔍 Detailed Logs

```bash
# Full startup sequence
docker-compose logs q-structurize | head -100

# Just GPU config
docker-compose logs q-structurize | grep "🎮\|GPU"

# Just timing info
docker-compose logs q-structurize | grep "⏱️\|complete in"

# Just errors
docker-compose logs q-structurize | grep "❌\|ERROR"
```

---

## 🎯 What to Report

After testing, please share:

1. **Timing improvement:**
   ```
   Before: XXX seconds
   After: XXX seconds
   Speedup: XX.Xx
   ```

2. **GPU utilization:**
   ```
   Peak: XX%
   Average: XX%
   ```

3. **Which optimizations loaded:**
   ```
   TF32: ✅/❌
   BF16: ✅/❌
   Flash Attention: ✅/❌
   Batch processing: ✅/❌
   ```

4. **Any errors or warnings** in the logs

This will help determine if further optimization is needed! 🚀

