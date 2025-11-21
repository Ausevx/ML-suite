#!/usr/bin/env python3
"""
GPU Detection Test Script for ML-Suite
Tests GPU availability for PyTorch and XGBoost
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_pytorch_gpu():
    """Test PyTorch GPU availability"""
    print("Testing PyTorch GPU support...")
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")

        # Check CUDA
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA available: {cuda_available}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"✓ CUDA devices: {device_count}")

            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                print(f"  Device {i}: {props.name} ({props.total_memory / (1024**3):.1f} GB)")

        # Check MPS (Apple Silicon)
        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        print(f"✓ MPS available (Apple Silicon): {mps_available}")

        if mps_available:
            print("  Apple Silicon GPU detected")

        # Overall GPU availability
        gpu_available = cuda_available or mps_available
        if gpu_available:
            backend = 'cuda' if cuda_available else 'mps'
            print(f"✓ GPU acceleration available via {backend}")
        else:
            print("✗ No GPU acceleration available - using CPU only")

        return gpu_available

    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ PyTorch GPU test failed: {e}")
        return False

def test_xgboost_gpu():
    """Test XGBoost GPU availability"""
    print("\nTesting XGBoost GPU support...")
    try:
        import xgboost as xgb
        print(f"✓ XGBoost version: {xgb.__version__}")

        # Try to create XGBoost model with GPU
        try:
            model = xgb.XGBRegressor(tree_method='gpu_hist')
            print("✓ XGBoost GPU support available (gpu_hist)")
            return True
        except Exception as e:
            print(f"✗ XGBoost GPU not available: {e}")
            # Try CPU fallback
            try:
                model = xgb.XGBRegressor(tree_method='hist')
                print("✓ XGBoost CPU support available (hist)")
                return False
            except Exception as e2:
                print(f"✗ XGBoost CPU support failed: {e2}")
                return False

    except ImportError:
        print("✗ XGBoost not installed")
        return False
    except Exception as e:
        print(f"✗ XGBoost test failed: {e}")
        return False

def test_system_stats():
    """Test system stats GPU detection"""
    print("\nTesting system stats GPU detection...")
    try:
        from modules.system_stats import get_gpu_info
        gpu_info = get_gpu_info()

        print("GPU Detection Results:")
        print(f"  Available: {gpu_info['available']}")
        print(f"  PyTorch CUDA: {gpu_info['pytorch_cuda']}")
        print(f"  PyTorch MPS: {gpu_info.get('pytorch_mps', False)}")
        print(f"  XGBoost GPU: {gpu_info['xgboost_gpu']}")
        print(f"  GPU Backend: {gpu_info.get('gpu_backend', 'cpu')}")
        print(f"  Devices: {len(gpu_info.get('devices', []))}")

        if gpu_info.get('devices'):
            for device in gpu_info['devices']:
                if device.get('type') == 'cuda':
                    print(f"    CUDA {device['name']}: {device['total_memory_gb']} GB")
                elif device.get('type') == 'mps':
                    print(f"    {device['name']} (Apple Silicon)")
                else:
                    print(f"    {device['name']}")

        return gpu_info['available']

    except Exception as e:
        print(f"✗ System stats test failed: {e}")
        return False

def main():
    """Run all GPU tests"""
    print("=" * 50)
    print("ML-Suite GPU Detection Test")
    print("=" * 50)

    pytorch_gpu = test_pytorch_gpu()
    xgboost_gpu = test_xgboost_gpu()
    system_stats_gpu = test_system_stats()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    gpu_available = pytorch_gpu or xgboost_gpu

    if gpu_available:
        print("🎉 GPU acceleration is available!")
        if pytorch_gpu:
            print("  ✓ PyTorch GPU support")
        if xgboost_gpu:
            print("  ✓ XGBoost GPU support")
        print("\nML-Suite will automatically use GPU acceleration for compatible models.")
    else:
        print("⚠️  GPU acceleration not available")
        print("  ML-Suite will use CPU-only training")
        print("\nTo enable GPU acceleration:")
        print("  1. Install NVIDIA CUDA toolkit")
        print("  2. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("  3. XGBoost will automatically detect CUDA")

    return 0 if gpu_available else 1

if __name__ == "__main__":
    sys.exit(main())
