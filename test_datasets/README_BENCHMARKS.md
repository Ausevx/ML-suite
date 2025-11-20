# Benchmark Datasets for GPU vs CPU Testing

## 📁 Available Datasets

### Regression Datasets (Predict continuous target)
- ✅ `gpu_bench_tiny_500.csv` - 500 rows, 20 features
- ✅ `gpu_bench_small_1k.csv` - 1,000 rows, 20 features
- ✅ `gpu_bench_small_2k.csv` - 2,000 rows, 20 features
- ✅ `gpu_bench_medium_5k.csv` - 5,000 rows, 20 features
- ✅ `gpu_bench_medium_10k.csv` - 10,000 rows, 20 features
- ✅ `gpu_bench_large_20k.csv` - 20,000 rows, 20 features
- ✅ `gpu_bench_large_50k.csv` - 50,000 rows, 20 features
- ✅ `gpu_bench_xlarge_100k.csv` - 100,000 rows, 20 features

### Classification Datasets (Predict class labels)
- ✅ `gpu_bench_class_tiny_500.csv` - 500 rows, 3 classes
- ✅ `gpu_bench_class_small_1k.csv` - 1,000 rows, 3 classes
- ✅ `gpu_bench_class_small_2k.csv` - 2,000 rows, 3 classes
- ✅ `gpu_bench_class_medium_5k.csv` - 5,000 rows, 3 classes
- ✅ `gpu_bench_class_medium_10k.csv` - 10,000 rows, 3 classes

## 🎯 Quick Start

### 1. Test Small Dataset (GPU NOT beneficial)
```
Dataset: gpu_bench_small_1k.csv
Model: Neural Network
Settings: 64 neurons, 50 epochs
Result: CPU faster or same as GPU
```

### 2. Test Medium Dataset (GPU beneficial)
```
Dataset: gpu_bench_medium_10k.csv
Model: Neural Network
Settings: 128 neurons, 100 epochs
Result: GPU 3-5x faster than CPU
```

### 3. Test Large Dataset (GPU excellent)
```
Dataset: gpu_bench_large_50k.csv
Model: Neural Network
Settings: 256 neurons, 100 epochs
Result: GPU 8-12x faster than CPU
```

## 📊 Expected Performance (M3 MacBook Air)

| Dataset | CPU Time | GPU Time | Speedup |
|---------|----------|----------|---------|
| 500 rows | ~3s | ~3s | ~1x |
| 1K rows | ~5s | ~5s | ~1x |
| 5K rows | ~35s | ~15s | ~2.3x |
| 10K rows | ~75s | ~18s | ~4x |
| 50K rows | ~450s | ~45s | ~10x |

## 🔧 How to Use

1. **Upload dataset** in Model Trainer
2. **Select target**: Use column named "target"
3. **Select features**: Use all feature_1 through feature_20
4. **Choose model**: Neural Network (only model using GPU)
5. **Configure**: Set hidden size, epochs as recommended
6. **Train twice**: Once with GPU, once with CPU (Force CPU in Settings)
7. **Compare times**: Check Model Management for training times

## 💡 Key Insights

- **< 2K rows**: GPU has overhead, CPU competitive
- **2-5K rows**: GPU starts showing benefit (1.5-3x)
- **5-20K rows**: GPU clearly faster (3-7x)
- **20K+ rows**: GPU excellent (7-15x)

See `../GPU_BENCHMARK_GUIDE.md` for complete testing guide!

