# ML-Suite

Local desktop ML application with OCR, model training, and comprehensive ML workflows. Runs entirely on localhost.

## Features

- **OCR Module**: Extract text from images and PDFs
- **Model Trainer**: Train regression, classification, clustering, and PCA models with visual interface
- **Model Management**: Save, load, download, and manage trained models
- **Hyperparameter Tuning**: Grid Search, Random Search, Bayesian Optimization
- **Visualizations**: Matplotlib charts (ROC, confusion matrix, residuals, feature importance, etc.)
- **Notebook Export**: Generate Jupyter notebooks with complete training code and visualizations
- **Console/Logs**: Real-time logging with filtering
- **System Monitoring**: CPU and RAM usage tracking
- **Customizable Themes**: 8 terminal themes with CRT effects and light modern theme

## Requirements

- Python 3.8-3.13
- Tesseract OCR
- Poppler (for PDF support)
- `libomp` (macOS only, for XGBoost)

## Quick Install

### macOS
```bash
brew install tesseract poppler libomp
./install.sh
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils python3-pip python3-venv
./install.sh
```

### Windows
```cmd
choco install tesseract poppler -y
install.bat
```

Or use `install.ps1` for PowerShell.

## Run

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate.bat  # Windows

# Start application
python backend/app.py
```

Open `http://localhost:5000` in your browser.

## Download Path Setup

ML-Suite stores models and uploads in configurable directories. To use exported Jupyter notebooks:

1. **Check Storage Paths**:
   - Go to Settings → Storage tab in the app
   - Note the paths for "Models Directory" and "Uploads Directory"

2. **Set Default Paths**:
   - Default models: `./models/`
   - Default uploads: `./uploads/`

3. **Use in Exported Notebooks**:
   - Each exported notebook has a configuration cell at the top:
   ```python
   # CONFIGURATION
   DATA_PATH = 'your_data.csv'
   MODEL_PATH = 'model_xyz.pkl'
   ```
   - Update `DATA_PATH` to your CSV location
   - Update `MODEL_PATH` to the downloaded model path
   - Example: `MODEL_PATH = '/Users/username/Downloads/model_20231215_123456.pkl'`

4. **Copy Paths**: Use the paths from Settings → Storage to ensure notebooks can locate models.

## Project Structure

```
ml-suite/
├── backend/          # Flask application and modules
│   ├── app.py
│   ├── config.py
│   ├── modules/
│   └── requirements.txt
├── frontend/         # HTML, CSS, JavaScript
│   ├── static/
│   └── templates/
├── models/          # Saved models (auto-created)
├── uploads/         # Uploaded files (auto-created)
└── install.*        # Installation scripts
```

## Usage

1. **Train Models**: Upload CSV → Select features/target → Choose model type → Train
2. **Manage Models**: View, download, export notebooks, or delete trained models
3. **Export Notebooks**: Download complete Jupyter notebooks with training code and visualizations
4. **OCR**: Upload images/PDFs to extract text

**Demo Dataset**: Sample credit card fraud dataset for classification available in `test_datasets/creditcard_sample.csv`

## Troubleshooting

- **Tesseract not found**: Add to system PATH or reinstall
- **XGBoost error (macOS)**: Install `libomp` via `brew install libomp`
- **scikit-learn fails (Python 3.13)**: Run `pip install --upgrade pip setuptools wheel` first
- **Port in use**: Change port in `backend/config.py`

## License

MIT License - see project for details.
