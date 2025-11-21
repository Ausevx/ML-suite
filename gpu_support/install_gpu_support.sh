#!/bin/bash
# Install GPU Support for ML-Suite
# Detects platform and installs appropriate packages

echo "=================================================="
echo "ML-Suite GPU Support Installer"
echo "=================================================="

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected: $OS $ARCH"

# Check if in virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Not in virtual environment!"
    echo "Activating venv..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "❌ Virtual environment not found. Run install.sh first."
        exit 1
    fi
fi

echo ""
echo "Installing GPU support packages..."
echo ""

# Install based on platform
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    # Apple Silicon (M1/M2/M3)
    echo "🍎 Apple Silicon detected - Installing PyTorch with MPS support..."
    pip install torch torchvision torchaudio
    
elif [ "$OS" = "Linux" ]; then
    # Linux - check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 NVIDIA GPU detected - Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install nvidia-ml-py3
    else
        echo "💻 No NVIDIA GPU - Installing CPU-only PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
    # Intel Mac
    echo "💻 Intel Mac detected - Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
    
else
    # Other platforms (Windows via Git Bash, etc.)
    echo "💻 Installing PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install XGBoost (GPU support automatic if CUDA available)
echo ""
echo "Installing XGBoost..."
pip install xgboost

echo ""
echo "=================================================="
echo "Testing GPU Detection..."
echo "=================================================="
python test_gpu.py

echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "🎉 GPU support packages installed successfully!"
echo ""
echo "Next steps:"
echo "1. Check the test results above"
echo "2. Start the ML-Suite: python backend/app.py"
echo "3. Go to Settings → Performance to configure GPU"
echo ""

