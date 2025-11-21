# ML-Suite

Local desktop ML application with OCR, model training, and comprehensive ML workflows. Runs entirely on localhost.

## Features

- **OCR Module**: Extract text from images and PDFs
- **Model Trainer**: Train regression, classification, clustering, and PCA models with visual interface
- **GPU Acceleration**: Automatic GPU support for Neural Networks (Apple Silicon MPS, NVIDIA CUDA)
- **Model Management**: Save, load, download, and manage trained models with training times
- **Hyperparameter Tuning**: Grid Search, Random Search, Bayesian Optimization
- **Visualizations**: Matplotlib charts (ROC, confusion matrix, residuals, feature importance, etc.)
- **Notebook Export**: Generate Jupyter notebooks with complete training code and visualizations
- **Console/Logs**: Real-time logging with filtering and hardware usage tracking
- **System Monitoring**: CPU, RAM, and GPU usage tracking
- **Customizable Themes**: 8 terminal themes with CRT effects and light modern theme

## Quick Start

### Prerequisites
- Python 3.8-3.13
- Tesseract OCR (for text extraction)
- Poppler (for PDF processing)

### Install

**macOS:**
```bash
brew install tesseract poppler libomp
./install.sh
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils python3-pip python3-venv
./install.sh
```

**Windows:**
```cmd
choco install tesseract poppler -y
install.bat
```

*Alternatively, use `install.ps1` for PowerShell.*

## Run

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate.bat  # Windows

# Start application
python backend/app.py
```

** GPU Acceleration:**
- See [GPU Setup Guide](gpu_support/GPU_SETUP_GUIDE.md) for detailed instructions
- Supports Apple Silicon, NVIDIA, and AMD GPUs

Open `http://localhost:5000` in your browser.



## Features Overview

### 🔍 OCR Module
Extract text from images and PDFs using Tesseract OCR. Simply drag and drop your files and get instant text extraction.

![OCR Interface](images/ocr-interface.png)

*Upload images or PDFs and extract text with one click*

###  Model Training
Train machine learning models with an intuitive interface. Supports regression, classification, clustering, and dimensionality reduction.

#### How Model Training Works:
1. **Upload CSV Data** - Drag and drop your dataset
2. **Select Target & Features** - Choose what to predict and which columns to use
3. **Pick Model Type** - Choose from 15+ algorithms including XGBoost, Random Forest, SVM
4. **Configure Parameters** - Set hyperparameters or use auto-tuning
5. **Train & Evaluate** - Get performance metrics, visualizations, and downloadable models

![Model Training Interface](images/model-training.png)

*Complete ML workflow from data upload to trained model*

### Data Preprocessing & Feature Selection
Clean and prepare your data for optimal model performance with built-in preprocessing tools.

#### Feature & Target Selection:
- **Target Variable**: Choose which column you want to predict (the "Y" variable)
- **Feature Columns**: Select which columns to use as predictors (the "X" variables)
- **Data Types**: Automatic detection of numerical vs categorical features
- **Missing Values**: Handle missing data with various strategies (drop, fill, impute)

#### Data Cleaning Options:
- **Remove Duplicates**: Automatically identify and remove duplicate rows
- **Handle Outliers**: Statistical methods to detect and treat outliers
- **Scale Features**: Standardize or normalize numerical features
- **Encode Categories**: Convert categorical variables to numerical format

![Data Preprocessing](images/data-preprocessing.png)

*Smart data cleaning and feature selection for better model results*

###  Hyperparameter Tuning
Optimize model performance with automated hyperparameter tuning using Grid Search, Random Search, or Bayesian Optimization.

![Hyperparameter Tuning](images/hyperparameter-tuning.png)

*Automated parameter optimization for better model performance*

###  Customizable Themes
Choose from multiple themes including retro terminal styles with CRT effects and modern light themes.

![Theme Selection](images/themes.png)

*8 different themes from classic terminal to modern light*

### 📊 Model Management
View, download, and manage all your trained models. Export complete Jupyter notebooks with training code and visualizations.

###  Notebook Export
Generate fully functional Jupyter notebooks with your complete ML workflow, including data preprocessing, model training, and evaluation plots.

## Demo

A screen recording is attached for help and understanding of the application's features and workflows.

## Project Structure

```
├── backend/          # Flask API and ML modules
├── frontend/         # Web interface (HTML/CSS/JS)
├── models/           # Trained models (auto-created)
├── uploads/          # Uploaded files (auto-created)
├── test_datasets/    # Sample datasets
└── install.*         # Platform-specific installers
```

## Getting Started

1. **Launch the app** and open `http://localhost:5000`
2. **Try OCR**: Upload an image or PDF to extract text
3. **Train a model**: Use the sample dataset in `test_datasets/creditcard_sample.csv`
4. **Explore features**: Switch between modules using the sidebar navigation

### Sample Datasets

**Included:**
- `test_datasets/creditcard_sample.csv` - Credit card fraud detection (classification)

**Generate GPU Benchmarks:**
```bash
cd test_datasets
python generate_benchmark_datasets.py
```

This creates datasets from 500 to 100K rows for testing GPU vs CPU performance.

###  GPU Acceleration

ML-Suite automatically uses GPU when beneficial for faster training.

**Supported Platforms:**
- **Apple Silicon** (M1/M2/M3/M4) - MPS acceleration for Neural Networks
- **NVIDIA GPUs** - CUDA acceleration for Neural Networks & XGBoost
- **AMD GPUs** - ROCm acceleration for Neural Networks (Linux only)

**Quick Setup:**
```bash
# Apple Silicon (M1/M2/M3)
pip install torch torchvision torchaudio xgboost

# NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xgboost nvidia-ml-py3

# AMD GPU (Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
pip install xgboost
```

📖 **[Complete GPU Setup Guide](gpu_support/GPU_SETUP_GUIDE.md)** - Detailed instructions for all platforms

**Performance Notes:**
- **First Run**: May be slow (cold start overhead)
- **Later Runs**: Much faster (warm cache)
- **Small Datasets** (< 5K rows): CPU may be faster
- **Large Datasets** (50K+ rows): GPU shows 2-10x speedup
- **Verify**: Check console logs for `[GPU]` or `[CPU]` tags
- **NOTE:** For Apple silicon, GPU acceleration might not provide faster computation because of multithreading compatibility issues with MPD and neural networks (PyTorch), which may need to be limited to a single thread.



## Troubleshooting

- **Tesseract not found**: Install Tesseract OCR and ensure it's in your system PATH
- **XGBoost issues on macOS**: Install `libomp` with `brew install libomp`
- **Port already in use**: The app runs on port 5000 by default. Change it in `backend/config.py` if needed
- **Python dependency issues**: Try `pip install --upgrade pip setuptools wheel`

## License

MIT License - see project for details.
