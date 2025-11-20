# GPU Acceleration Setup Guide

This guide covers GPU setup for all supported platforms. ML-Suite supports GPU acceleration for faster model training using PyTorch and XGBoost.

---

## Quick Platform Selection

- **[Apple Silicon (M1/M2/M3/M4)](#apple-silicon-m1m2m3m4)** - MacBook, Mac Mini, Mac Studio
- **[NVIDIA GPUs](#nvidia-gpus-cuda)** - Windows, Linux with NVIDIA graphics cards
- **[AMD GPUs](#amd-gpus-rocm)** - Linux with AMD graphics cards
- **[Intel GPUs](#intel-gpus)** - Limited support on Windows/Linux

---

## Apple Silicon (M1/M2/M3/M4)

### Quick Installation

```bash
# Navigate to ML-Suite directory
cd /path/to/ML-suite-1

# Activate virtual environment
source venv/bin/activate

# Install PyTorch with Apple Silicon (MPS) support
pip install torch torchvision torchaudio

# Install XGBoost
pip install xgboost

# Install NVIDIA monitoring tools (optional, for compatibility)
pip install nvidia-ml-py3

# Test GPU detection
python test_gpu.py
```

### Requirements
- **macOS 12.3+** (for MPS support)
- **Apple Silicon chip** (M1, M2, M3, M4, or later)
- **8GB+ RAM** recommended

### Expected Output

```
==================================================
ML-Suite GPU Detection Test
==================================================
Testing PyTorch GPU support...
✓ PyTorch version: 2.x.x
✓ MPS available (Apple Silicon): True
✓ GPU acceleration available via mps

Testing XGBoost GPU support...
✗ XGBoost GPU not available: XGBoost requires CUDA

SUMMARY
==================================================
🎉 GPU acceleration is available!
  ✓ PyTorch GPU support (MPS)

ML-Suite will automatically use GPU acceleration for compatible models.
```

### What's Accelerated on Apple Silicon?

| Model Type | GPU Support | Expected Speedup |
|------------|-------------|------------------|
| Neural Networks | ✅ Full MPS | 2-4x faster |
| XGBoost | ❌ CPU only | N/A |
| Random Forest | ❌ CPU only | N/A |
| Other scikit-learn | ❌ CPU only | N/A |

### Performance Expectations (M3 MacBook Air)

| Dataset Size | CPU Time | GPU Time | Speedup |
|--------------|----------|----------|---------|
| 1K rows | 4s | 4s | 1.0x |
| 10K rows | 50s | 15s | 3.3x |
| 50K rows | 250s | 70s | 3.6x |

**Note**: First training run may be slower (cold start). Later runs will be faster.

### Troubleshooting

**MPS shows as unavailable:**
1. Check macOS version: `sw_vers` (need 12.3+)
2. Verify PyTorch: `python -c "import torch; print(torch.backends.mps.is_available())"`
3. Reinstall if needed: `pip uninstall torch && pip install torch`

**GPU utilization shows 0%:**
- This is normal on Apple Silicon - MPS doesn't provide real-time metrics
- Check console logs instead: `[GPU] Neural Network using MPS acceleration`

---

## NVIDIA GPUs (CUDA)

### Quick Installation (Windows/Linux)

```bash
# Navigate to ML-Suite directory
cd /path/to/ML-suite-1

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux:
source venv/bin/activate

# Install PyTorch with CUDA support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install XGBoost with GPU support
pip install xgboost[gpu]

# Install NVIDIA monitoring tools
pip install nvidia-ml-py3

# Test GPU detection
python test_gpu.py
```

### Requirements
- **NVIDIA GPU** with compute capability 3.5+ (GTX 700 series or newer)
- **CUDA Toolkit** 11.8 or 12.1
- **cuDNN** (bundled with PyTorch)
- **NVIDIA Driver** 450.80+ (Linux) or 452.39+ (Windows)
- **4GB+ VRAM** recommended

### Installing CUDA Toolkit

**Windows:**
1. Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Run installer and follow prompts
3. Verify: `nvcc --version`

**Linux (Ubuntu/Debian):**
```bash
# CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### Expected Output

```
==================================================
ML-Suite GPU Detection Test
==================================================
Testing PyTorch GPU support...
✓ PyTorch version: 2.x.x
✓ CUDA available: True
✓ CUDA version: 11.8
✓ GPU acceleration available via cuda

Testing XGBoost GPU support...
✓ XGBoost version: 2.x.x
✓ XGBoost GPU available

Testing NVIDIA GPU utilization...
✓ Found 1 NVIDIA GPU(s)
  GPU 0: NVIDIA GeForce RTX 3080 | 10GB

SUMMARY
==================================================
🎉 Full GPU acceleration is available!
  ✓ PyTorch GPU support (CUDA)
  ✓ XGBoost GPU support
```

### What's Accelerated on NVIDIA?

| Model Type | GPU Support | Expected Speedup |
|------------|-------------|------------------|
| Neural Networks | ✅ Full CUDA | 5-20x faster |
| XGBoost | ✅ Full CUDA | 3-10x faster |
| Random Forest | ❌ CPU only | N/A |
| Other scikit-learn | ❌ CPU only | N/A |

### Troubleshooting

**CUDA not available:**
1. Check GPU: `nvidia-smi`
2. Check CUDA: `nvcc --version`
3. Verify PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
4. Reinstall PyTorch with correct CUDA version

**Out of Memory errors:**
- Reduce batch size in neural network settings
- Close other GPU applications
- Use smaller datasets for testing

---

## AMD GPUs (ROCm)

### Quick Installation (Linux Only)

```bash
# Navigate to ML-Suite directory
cd /path/to/ML-suite-1

# Activate virtual environment
source venv/bin/activate

# Install PyTorch with ROCm support (ROCm 5.7)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# Install XGBoost (CPU version - no ROCm support)
pip install xgboost

# Test GPU detection
python test_gpu.py
```

### Requirements
- **AMD GPU** with ROCm support (see [ROCm compatibility list](https://github.com/RadeonOpenCompute/ROCm#hardware-and-software-support))
- **Linux** (Ubuntu 20.04/22.04, RHEL 8/9, SLES 15 SP4)
- **ROCm 5.7+**
- **4GB+ VRAM** recommended

### Installing ROCm (Ubuntu 22.04)

```bash
# Add ROCm repository
wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo apt-get install ./amdgpu-install_5.7.50700-1_all.deb

# Install ROCm
sudo amdgpu-install --usecase=rocm

# Add user to video/render groups
sudo usermod -a -G render,video $LOGNAME

# Reboot
sudo reboot
```

### What's Accelerated on AMD?

| Model Type | GPU Support | Expected Speedup |
|------------|-------------|------------------|
| Neural Networks | ✅ ROCm | 3-10x faster |
| XGBoost | ❌ CPU only | N/A |
| Random Forest | ❌ CPU only | N/A |
| Other scikit-learn | ❌ CPU only | N/A |

**Note**: XGBoost doesn't support ROCm. Only CUDA is supported for XGBoost GPU acceleration.

---

## Intel GPUs

### Limited Support

Intel GPU support is experimental and limited:

- **Intel Arc GPUs** - Partial PyTorch support via Intel Extension for PyTorch
- **Integrated GPUs** - Not recommended for ML workloads

### Installation (Windows/Linux)

```bash
# Install Intel Extension for PyTorch
pip install torch torchvision torchaudio
pip install intel-extension-for-pytorch

# Note: Support is limited and may not work with all models
```

**We recommend using CPU mode on Intel GPUs** for stability.

---

## Verifying GPU Setup in ML-Suite

After installation:

1. Start ML-Suite: `python backend/app.py`
2. Open browser: `http://localhost:5000`
3. Go to **Settings → Performance**
4. Check GPU Status:
   - ✅ **Available (CUDA)** - NVIDIA GPU detected
   - ✅ **Available (MPS)** - Apple Silicon detected
   - ✅ **Available (ROCm)** - AMD GPU detected
   - ❌ **Not Available** - No GPU or drivers not installed

### During Training

Watch the console logs:
```
[GPU] Neural Network using CUDA acceleration  ← NVIDIA GPU
[GPU] Neural Network using MPS acceleration   ← Apple Silicon
[GPU] XGBoost using GPU acceleration          ← NVIDIA only
[CPU] Random Forest (CPU optimized)           ← CPU-only models
```

### System Monitor

Check real-time GPU utilization in the System Stats panel:
- **NVIDIA**: Shows real-time GPU % and memory usage
- **Apple Silicon**: Shows "N/A" (platform limitation)
- **AMD**: Shows real-time GPU % (if supported)

---

## Performance Tips

### Cold Start vs Warm Start
- **First training run**: Slower due to initialization, compilation, caching
- **Subsequent runs**: Much faster (50-70% faster is normal)
- **Recommendation**: Don't judge GPU performance on first run

### When GPU Helps Most
- **Large datasets** (10K+ rows)
- **Neural networks** (any size)
- **Deep models** (many layers/estimators)

### When CPU May Be Faster
- **Small datasets** (<1K rows)
- **Simple models** (few features)
- **GPU overhead** exceeds computation time

### Optimal Settings
1. Enable GPU in Settings → Performance
2. Use "Auto" backend (recommended)
3. For neural networks: Use larger batch sizes with GPU
4. For XGBoost (NVIDIA only): GPU automatically used

---

## Troubleshooting General Issues

### GPU not being used even though detected

1. Check Settings → Performance → Enable GPU Acceleration
2. Verify GPU backend is set to "Auto" or specific backend
3. Restart ML-Suite after changing settings
4. Check console logs during training for `[GPU]` prefix

### Training slower with GPU than CPU

- First run: Normal due to cold start
- Later runs: Check dataset size (may be too small)
- Check GPU utilization in system stats (should be >0%)

### Import errors after installation

```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Verify installations
python -c "import torch; print(torch.__version__)"
python -c "import xgboost; print(xgboost.__version__)"
```

---

## Benchmark Testing

Generate test datasets to compare CPU vs GPU performance:

```bash
cd test_datasets
python generate_benchmark_datasets.py
```

This creates datasets of various sizes (1K, 5K, 10K, 50K rows) for testing.

**Test procedure:**
1. Train with CPU: Settings → Performance → Force CPU Mode
2. Note training time
3. Train with GPU: Settings → Performance → Enable GPU, disable Force CPU
4. Note training time and speedup

**Important**: Run each test 2-3 times to account for cold start overhead.

---

## Getting Help

If you encounter issues:

1. Run `python test_gpu.py` and share the output
2. Check console logs during training
3. Verify driver/toolkit versions
4. Open an issue with:
   - Your GPU model
   - Operating system
   - Output of `test_gpu.py`
   - Console logs from training

---

## Summary Table

| Platform | PyTorch | XGBoost | Difficulty | Speedup |
|----------|---------|---------|------------|---------|
| **Apple Silicon** | ✅ MPS | ❌ CPU | Easy | 2-4x |
| **NVIDIA CUDA** | ✅ CUDA | ✅ CUDA | Medium | 5-20x |
| **AMD ROCm** | ✅ ROCm | ❌ CPU | Hard | 3-10x |
| **Intel** | ⚠️ Limited | ❌ CPU | Hard | 1-2x |

**Recommendation**: NVIDIA GPUs provide the best overall acceleration with both PyTorch and XGBoost support.

