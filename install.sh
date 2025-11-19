#!/bin/bash
# ML-Suite Installation Script for macOS and Linux
# This script installs all dependencies and sets up the environment

set -e  # Exit on error

echo "ML-Suite Installation Script"
echo "============================"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python 3.8 or higher is required. Found Python $PYTHON_VERSION"
    exit 1
fi

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
    echo "Note: Python 3.13+ detected. Using latest compatible package versions."
fi

echo "Python version: $PYTHON_VERSION"
echo ""

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    echo "Detected: macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Try to detect Linux distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS=$ID
        echo "Detected: $PRETTY_NAME"
    else
        OS="linux"
        echo "Detected: Linux (generic)"
    fi
else
    echo "Error: Unsupported operating system"
    exit 1
fi

# Install system dependencies based on OS
echo ""
echo "Installing system dependencies..."

if [ "$OS" == "macos" ]; then
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        echo "Error: Homebrew is not installed. Please install it from https://brew.sh"
        exit 1
    fi
    brew install tesseract poppler libomp
elif [ "$OS" == "ubuntu" ] || [ "$OS" == "debian" ]; then
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr poppler-utils python3-pip python3-venv
elif [ "$OS" == "fedora" ] || [ "$OS" == "rhel" ] || [ "$OS" == "centos" ]; then
    sudo dnf install -y tesseract poppler-utils python3-pip python3-virtualenv
else
    echo "Warning: Unknown Linux distribution. Please install tesseract-ocr and poppler-utils manually."
    read -p "Press enter to continue..."
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip and build tools (important for Python 3.13+)
echo ""
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r backend/requirements.txt

echo ""
echo "============================"
echo "Installation complete!"
echo ""
echo "To run the application:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run: python backend/app.py"
echo "  3. Open http://localhost:5000 in your browser"
echo ""

