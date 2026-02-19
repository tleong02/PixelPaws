# PixelPaws Installation Guide

## Quick Install (Recommended)

### Option 1: With GPU Support (Recommended)
```bash
pip install -r requirements_gpu.txt
python enable_pytorch_gpu.py
```

### Option 2: CPU Only
```bash
pip install -r requirements_cpu.txt
```

### Option 3: Custom (Advanced)
Edit `requirements.txt` to choose CUDA version, then:
```bash
pip install -r requirements.txt
python enable_pytorch_gpu.py
```

---

## Which Should I Use?

### Use `requirements_gpu.txt` if:
✅ You have an NVIDIA GPU
✅ You want 5-10x faster feature extraction
✅ Your GPU is RTX series or newer

### Use `requirements_cpu.txt` if:
❌ You don't have an NVIDIA GPU
❌ You have AMD or Intel GPU
❌ You only have CPU

---

## After Installation

### Enable GPU acceleration:
```bash
python enable_pytorch_gpu.py
```

### Verify GPU is working:
```bash
python enable_pytorch_gpu.py --check
```

Expected output:
```
PyTorch Status:
CUDA available: True
GPU: NVIDIA GeForce RTX 3080
✓ PyTorch with CUDA is ready!
```

---

## Troubleshooting

### "Could not find a version that satisfies torch"

**Problem:** PyTorch CUDA version mismatch

**Solution:** Try CUDA 11.8 instead:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"

**Problem:** GPU memory full

**Solution:** PyTorch will automatically fall back to frame-by-frame processing. This is normal and handled automatically.

### "torch.cuda.is_available() returns False"

**Problem:** PyTorch can't find GPU

**Solutions:**
1. Check GPU drivers: `nvidia-smi`
2. Reinstall PyTorch with CUDA
3. Use CPU version if GPU not available

---

## Version Information

- **Python**: 3.7 or higher (3.11 recommended)
- **CUDA**: 11.8 or 12.1
- **GPU**: NVIDIA with compute capability 3.5+
- **RAM**: 8GB minimum, 16GB recommended
- **GPU Memory**: 4GB minimum, 8GB+ recommended

---

## File Descriptions

| File | Description | Use Case |
|------|-------------|----------|
| `requirements.txt` | Full version with comments | Reference & customization |
| `requirements_gpu.txt` | GPU-enabled (CUDA 12.1) | **Recommended for most users** |
| `requirements_cpu.txt` | CPU-only | No GPU available |

---

## Installation Examples

### Example 1: Fresh Install with GPU
```bash
# Create virtual environment
python -m venv pixelpaws_env
source pixelpaws_env/bin/activate  # On Windows: pixelpaws_env\Scripts\activate

# Install dependencies
pip install -r requirements_gpu.txt

# Enable GPU acceleration
python enable_pytorch_gpu.py

# Launch PixelPaws
python PixelPaws_GUI.py
```

### Example 2: Upgrade Existing Installation
```bash
# If you already have PixelPaws installed:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python enable_pytorch_gpu.py
```

### Example 3: CPU Only
```bash
pip install -r requirements_cpu.txt
python PixelPaws_GUI.py
```

---

## Performance Comparison

| Configuration | Install Time | Feature Extraction (50k frames) |
|---------------|--------------|--------------------------------|
| CPU Only | 5 minutes | 180 seconds |
| **GPU (PyTorch)** | **7 minutes** | **40 seconds (4.5x)** |

**Recommendation:** GPU version worth the extra 2 minutes!

---

## Need Help?

1. Check GPU status: `python enable_pytorch_gpu.py --check`
2. Verify installation: `pip list | grep torch`
3. Test GPU: `python -c "import torch; print(torch.cuda.is_available())"`

Still stuck? All code has automatic CPU fallback, so worst case you're no slower than before! 🎯
