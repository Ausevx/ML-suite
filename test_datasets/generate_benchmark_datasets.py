#!/usr/bin/env python3
"""
Generate benchmark datasets for GPU vs CPU testing
Creates datasets of varying sizes to demonstrate GPU performance benefits
"""

import numpy as np
import pandas as pd
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_regression_dataset(n_samples, n_features=20, noise=0.1, filename=None):
    """
    Generate synthetic regression dataset
    
    Args:
        n_samples: Number of samples/rows
        n_features: Number of input features
        noise: Amount of noise in target variable
        filename: Output CSV filename
    """
    print(f"Generating regression dataset: {n_samples} samples, {n_features} features...")
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target as non-linear combination of features
    # This makes neural networks beneficial
    weights = np.random.randn(n_features)
    target = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Non-linear transformation
        target[i] = (
            np.sum(X[i] * weights) +
            0.5 * np.sum(np.sin(X[i] * 2)) +
            0.3 * np.sum(X[i] ** 2) +
            noise * np.random.randn()
        )
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    
    # Save to CSV
    if filename:
        filepath = os.path.join('test_datasets', filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath} ({df.memory_usage(deep=True).sum() / 1024:.1f} KB)")
    
    return df

def generate_classification_dataset(n_samples, n_features=20, n_classes=3, filename=None):
    """
    Generate synthetic classification dataset
    
    Args:
        n_samples: Number of samples/rows
        n_features: Number of input features
        n_classes: Number of classes
        filename: Output CSV filename
    """
    print(f"Generating classification dataset: {n_samples} samples, {n_features} features, {n_classes} classes...")
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate class labels based on feature combinations
    weights = np.random.randn(n_features)
    scores = X @ weights
    
    # Convert scores to classes
    percentiles = np.percentile(scores, [100/n_classes * i for i in range(1, n_classes)])
    target = np.digitize(scores, percentiles)
    
    # Create DataFrame
    feature_names = [f'feature_{i+1}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = target
    
    # Save to CSV
    if filename:
        filepath = os.path.join('test_datasets', filename)
        df.to_csv(filepath, index=False)
        print(f"✓ Saved to {filepath} ({df.memory_usage(deep=True).sum() / 1024:.1f} KB)")
    
    return df

def main():
    """Generate all benchmark datasets"""
    print("=" * 60)
    print("GPU vs CPU Benchmark Dataset Generator")
    print("=" * 60)
    print()
    
    # Create test_datasets directory if it doesn't exist
    os.makedirs('test_datasets', exist_ok=True)
    
    # Generate datasets of various sizes
    datasets = [
        # Small datasets (CPU should be competitive or faster)
        {'n_samples': 500, 'filename': 'gpu_bench_tiny_500.csv', 'desc': 'Tiny (CPU faster due to overhead)'},
        {'n_samples': 1000, 'filename': 'gpu_bench_small_1k.csv', 'desc': 'Small (CPU still competitive)'},
        {'n_samples': 2000, 'filename': 'gpu_bench_small_2k.csv', 'desc': 'Small-Medium (GPU starts helping)'},
        
        # Medium datasets (GPU should show 2-5x improvement)
        {'n_samples': 5000, 'filename': 'gpu_bench_medium_5k.csv', 'desc': 'Medium (GPU 2-3x faster)'},
        {'n_samples': 10000, 'filename': 'gpu_bench_medium_10k.csv', 'desc': 'Medium (GPU 3-5x faster)'},
        {'n_samples': 20000, 'filename': 'gpu_bench_large_20k.csv', 'desc': 'Large (GPU 5-7x faster)'},
        
        # Large datasets (GPU should show 5-15x improvement)
        {'n_samples': 50000, 'filename': 'gpu_bench_large_50k.csv', 'desc': 'Large (GPU 7-10x faster)'},
        {'n_samples': 100000, 'filename': 'gpu_bench_xlarge_100k.csv', 'desc': 'X-Large (GPU 10-15x faster)'},
    ]
    
    print("Generating REGRESSION datasets:")
    print("-" * 60)
    for config in datasets:
        print(f"\n{config['desc']}:")
        generate_regression_dataset(
            n_samples=config['n_samples'],
            n_features=20,
            filename=config['filename']
        )
    
    print("\n" + "=" * 60)
    print("Generating CLASSIFICATION datasets:")
    print("-" * 60)
    for config in datasets[:5]:  # Generate fewer classification datasets
        filename = config['filename'].replace('bench', 'bench_class')
        print(f"\n{config['desc']}:")
        generate_classification_dataset(
            n_samples=config['n_samples'],
            n_features=20,
            n_classes=3,
            filename=filename
        )
    
    print("\n" + "=" * 60)
    print("✓ All datasets generated successfully!")
    print("=" * 60)
    print()
    print("📊 TESTING GUIDE:")
    print("-" * 60)
    print()
    print("1. TEST SMALL DATASETS (500-2000 rows):")
    print("   - Use: gpu_bench_tiny_500.csv or gpu_bench_small_1k.csv")
    print("   - Model: Neural Network")
    print("   - Settings: 2 hidden layers, 64 neurons, 50 epochs")
    print("   - Expected: CPU competitive or faster (GPU overhead)")
    print()
    print("2. TEST MEDIUM DATASETS (5000-20000 rows):")
    print("   - Use: gpu_bench_medium_5k.csv or gpu_bench_medium_10k.csv")
    print("   - Model: Neural Network")
    print("   - Settings: 3 hidden layers, 128 neurons, 100 epochs")
    print("   - Expected: GPU 2-5x faster")
    print()
    print("3. TEST LARGE DATASETS (50000+ rows):")
    print("   - Use: gpu_bench_large_50k.csv or gpu_bench_xlarge_100k.csv")
    print("   - Model: Neural Network")
    print("   - Settings: 4 hidden layers, 256 neurons, 100 epochs")
    print("   - Expected: GPU 5-15x faster")
    print()
    print("💡 TIPS:")
    print("   - First run may be slower (MPS kernel compilation)")
    print("   - Watch console for [GPU] vs [CPU] tags")
    print("   - Compare training times in Model Management")
    print("   - Toggle GPU in Settings → Performance to compare")
    print()
    print("🔧 BENCHMARK PROCEDURE:")
    print("   1. Train with GPU enabled → note time")
    print("   2. Settings → Performance → Force CPU Mode")
    print("   3. Train same model again → note time")
    print("   4. Compare: GPU speedup = CPU_time / GPU_time")
    print()

if __name__ == '__main__':
    main()

