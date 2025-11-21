"""
Model Management Module - Handles saved models listing, loading, deletion, and inference
"""
from flask import Blueprint, request, jsonify, send_file
import os
import json
import pandas as pd
import joblib
import config

models_bp = Blueprint('models', __name__)

def get_model_metadata(model_id):
    """Load model metadata from JSON file"""
    metadata_path = os.path.join(config.MODELS_DIR, f'{model_id}.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

@models_bp.route('', methods=['GET'])
def list_models():
    """List all saved models"""
    models = []
    
    # Scan models directory for JSON metadata files
    for filename in os.listdir(config.MODELS_DIR):
        if filename.endswith('.json'):
            model_id = filename[:-5]  # Remove .json extension
            metadata = get_model_metadata(model_id)
            if metadata:
                models.append({
                    'model_id': model_id,
                    'model_name': metadata.get('model_name', 'Unnamed'),
                    'model_type': metadata.get('model_type', 'unknown'),
                    'created_at': metadata.get('created_at', ''),
                    'dataset_name': metadata.get('dataset_name', ''),
                    'metrics': metadata.get('metrics', {}),
                    'features': metadata.get('features', [])
                })
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return jsonify({'models': models})

@models_bp.route('/<model_id>', methods=['GET'])
def get_model_details(model_id):
    """Get detailed information about a specific model"""
    metadata = get_model_metadata(model_id)
    
    if not metadata:
        return jsonify({'error': 'Model not found'}), 404
    
    return jsonify(metadata)

@models_bp.route('/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a model and its metadata"""
    model_path = os.path.join(config.MODELS_DIR, f'{model_id}.pkl')
    metadata_path = os.path.join(config.MODELS_DIR, f'{model_id}.json')
    
    if not os.path.exists(model_path) and not os.path.exists(metadata_path):
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        return jsonify({'success': True, 'message': 'Model deleted successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to delete model: {str(e)}'}), 500

@models_bp.route('/<model_id>/download', methods=['GET'])
def download_model(model_id):
    """Download model file"""
    model_path = os.path.join(config.MODELS_DIR, f'{model_id}.pkl')
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    
    return send_file(model_path, as_attachment=True, download_name=f'{model_id}.pkl')

@models_bp.route('/<model_id>/predict', methods=['POST'])
def predict(model_id):
    """Make predictions using a saved model"""
    metadata = get_model_metadata(model_id)
    
    if not metadata:
        return jsonify({'error': 'Model not found'}), 404
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Load CSV
        df = pd.read_csv(file)
        
        # Load model
        model_path = os.path.join(config.MODELS_DIR, f'{model_id}.pkl')
        model = joblib.load(model_path)
        
        # Get required features
        required_features = metadata.get('features', [])
        
        # Check if all features are present
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {", ".join(missing_features)}'
            }), 400
        
        # Prepare features
        X = df[required_features].select_dtypes(include=['number'])
        
        # Make predictions
        predictions = model.predict(X)
        
        # Add predictions to dataframe
        result_df = df.copy()
        result_df['prediction'] = predictions
        
        # Convert to records for JSON response
        results = result_df.to_dict('records')
        
        return jsonify({
            'predictions': results,
            'model_name': metadata.get('model_name', 'Unnamed'),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

