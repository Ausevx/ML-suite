"""
Model Downloader Module - Handles AI model downloads for text generation
"""
from flask import Blueprint, request, jsonify
from transformers import pipeline
import threading
import os

model_downloader_bp = Blueprint('model_downloader', __name__)

# Available models with descriptions
AVAILABLE_MODELS = {
    'google/flan-t5-small': {
        'name': 'FLAN-T5 Small',
        'size': '~300 MB',
        'speed': 'Fast',
        'quality': 'Good',
        'description': 'Lightweight and fast, good for basic rephrasing'
    },
    'google/flan-t5-base': {
        'name': 'FLAN-T5 Base',
        'size': '~900 MB',
        'speed': 'Medium',
        'quality': 'Better',
        'description': 'Balanced performance, recommended for most users'
    },
    'facebook/bart-large-cnn': {
        'name': 'BART Large',
        'size': '~1.6 GB',
        'speed': 'Slower',
        'quality': 'Best',
        'description': 'High quality text generation, requires more RAM'
    }
}

# Download status tracking
download_status = {
    'is_downloading': False,
    'current_model': None,
    'progress': 0,
    'status_message': '',
    'error': None
}

def download_model_background(model_name):
    """Download model in background thread"""
    global download_status
    
    try:
        download_status['is_downloading'] = True
        download_status['current_model'] = model_name
        download_status['progress'] = 10
        download_status['status_message'] = f'Initializing download of {model_name}...'
        download_status['error'] = None
        
        # Download the model
        download_status['progress'] = 30
        download_status['status_message'] = 'Downloading model files...'
        
        pipeline(
            "text2text-generation",
            model=model_name,
            device=-1
        )
        
        download_status['progress'] = 100
        download_status['status_message'] = 'Download complete!'
        
    except Exception as e:
        download_status['error'] = str(e)
        download_status['status_message'] = f'Download failed: {str(e)}'
    finally:
        download_status['is_downloading'] = False

@model_downloader_bp.route('/models/available', methods=['GET'])
def get_available_models():
    """Get list of available models"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'success': True
    })

@model_downloader_bp.route('/models/download', methods=['POST'])
def start_model_download():
    """Start downloading a model"""
    data = request.get_json()
    model_name = data.get('model_name')
    
    if not model_name or model_name not in AVAILABLE_MODELS:
        return jsonify({
            'error': 'Invalid model name',
            'success': False
        }), 400
    
    if download_status['is_downloading']:
        return jsonify({
            'error': 'A download is already in progress',
            'success': False
        }), 409
    
    # Start download in background thread
    thread = threading.Thread(target=download_model_background, args=(model_name,))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Download started',
        'model': model_name,
        'success': True
    })

@model_downloader_bp.route('/models/status', methods=['GET'])
def get_download_status():
    """Get current download status"""
    return jsonify({
        'status': download_status,
        'success': True
    })

@model_downloader_bp.route('/models/installed', methods=['GET'])
def get_installed_models():
    """Check which models are already installed"""
    installed = []
    
    # Check if models are in cache
    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
    
    for model_key in AVAILABLE_MODELS.keys():
        # Simplified check - just see if we can import it
        model_slug = model_key.replace('/', '--')
        model_dirs = []
        
        if os.path.exists(cache_dir):
            for item in os.listdir(cache_dir):
                if model_slug in item:
                    model_dirs.append(item)
        
        if model_dirs:
            installed.append(model_key)
    
    return jsonify({
        'installed': installed,
        'success': True
    })

