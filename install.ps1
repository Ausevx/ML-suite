# ML-Suite Installation Script for Windows PowerShell
# Run this script as Administrator: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Write-Host "ML-Suite Installation Script" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan
Write-Host ""

# Check if Chocolatey is installed
if (!(Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey is not installed. Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    Write-Host "Chocolatey installed successfully!" -ForegroundColor Green
}

# Install system dependencies
Write-Host ""
Write-Host "Installing system dependencies..." -ForegroundColor Cyan
choco install tesseract -y
choco install poppler -y

# Check if Python is installed
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python is not installed. Installing Python..." -ForegroundColor Yellow
    choco install python -y
    Write-Host "Please restart your terminal and run this script again." -ForegroundColor Yellow
    exit
}

# Check Python version
Write-Host ""
Write-Host "Checking Python version..." -ForegroundColor Cyan
$pythonVersion = python --version 2>&1 | Out-String
Write-Host $pythonVersion -ForegroundColor Green

$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
        Write-Host "Error: Python 3.8 or higher is required." -ForegroundColor Red
        exit 1
    }
    if ($major -eq 3 -and $minor -ge 13) {
        Write-Host "Note: Python 3.13+ detected. Using latest compatible package versions." -ForegroundColor Yellow
    }
}

# Create virtual environment
Write-Host ""
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

# Activate virtual environment
Write-Host ""
Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

# Upgrade pip and build tools (important for Python 3.13+)
Write-Host ""
Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel

# Install Python dependencies
Write-Host ""
Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
pip install -r backend/requirements.txt

Write-Host ""
Write-Host "============================" -ForegroundColor Cyan
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the application:" -ForegroundColor Yellow
Write-Host "  1. Activate the virtual environment: .\venv\Scripts\Activate.ps1"
Write-Host "  2. Run: python backend/app.py"
Write-Host "  3. Open http://localhost:5000 in your browser"
Write-Host ""
Write-Host "GPU Support:" -ForegroundColor Cyan
Write-Host "  - PyTorch with GPU support has been installed"
Write-Host "  - NVIDIA GPU: See gpu_support\GPU_SETUP_GUIDE.md for CUDA setup"
Write-Host "  - Test GPU: python test_gpu.py"
Write-Host ""

