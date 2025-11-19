"""
Configuration file for ML-Suite application
"""
import os

# Server Configuration
PORT = 5000
HOST = '127.0.0.1'
DEBUG = True

# File Upload Configuration
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
ALLOWED_CSV_EXTENSIONS = {'csv'}
ALLOWED_NOTEBOOK_EXTENSIONS = {'ipynb'}

# Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# Create directories if they don't exist
for directory in [MODELS_DIR, UPLOADS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model Configuration
DEFAULT_TRAIN_TEST_SPLIT = 0.8
MAX_INPUT_TOKENS = 512

# OCR Configuration
OCR_ENGINE = 'pytesseract'  # or 'easyocr'

# Text Generation Configuration
TEXT_MODEL_NAME = 'google/flan-t5-small'  # Lightweight model for local use

