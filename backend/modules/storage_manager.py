"""
Storage Manager Module - Handles storage paths and usage information
"""
from flask import Blueprint, request, jsonify
import os
import config

storage_bp = Blueprint('storage', __name__)

@storage_bp.route('/paths', methods=['GET'])
def get_storage_paths():
    """Get current storage paths"""
    paths = {
        'uploads': os.path.abspath(config.UPLOADS_DIR),
        'models': os.path.abspath(config.MODELS_DIR),
        'cache': os.path.expanduser('~/.cache/huggingface/'),
        'project_root': os.path.abspath('.')
    }
    
    return jsonify({'paths': paths, 'success': True})

@storage_bp.route('/usage', methods=['GET'])
def get_storage_usage():
    """Get storage usage information"""
    try:
        uploads_size = get_directory_size(config.UPLOADS_DIR)
        models_size = get_directory_size(config.MODELS_DIR)
        
        uploads_count = count_files(config.UPLOADS_DIR)
        models_count = count_files(config.MODELS_DIR)
        
        usage = {
            'uploads': {
                'size_bytes': uploads_size,
                'size_mb': round(uploads_size / (1024 * 1024), 2),
                'file_count': uploads_count
            },
            'models': {
                'size_bytes': models_size,
                'size_mb': round(models_size / (1024 * 1024), 2),
                'file_count': models_count
            }
        }
        
        return jsonify({'usage': usage, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

def get_directory_size(path):
    """Calculate total size of directory"""
    total = 0
    try:
        if os.path.exists(path):
            for entry in os.scandir(path):
                if entry.is_file():
                    total += entry.stat().st_size
                elif entry.is_dir():
                    total += get_directory_size(entry.path)
    except:
        pass
    return total

def count_files(path):
    """Count files in directory"""
    count = 0
    try:
        if os.path.exists(path):
            for entry in os.scandir(path):
                if entry.is_file():
                    count += 1
                elif entry.is_dir():
                    count += count_files(entry.path)
    except:
        pass
    return count

