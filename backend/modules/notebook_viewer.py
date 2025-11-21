"""
Notebook Viewer Module - Handles .ipynb file upload and display
"""
from flask import Blueprint, request, jsonify
import json
import config
from werkzeug.utils import secure_filename

notebook_bp = Blueprint('notebook', __name__)

def allowed_notebook_file(filename):
    """Check if file is .ipynb"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_NOTEBOOK_EXTENSIONS

@notebook_bp.route('/upload', methods=['POST'])
def upload_notebook():
    """Upload .ipynb file and return parsed structure"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_notebook_file(file.filename):
        return jsonify({'error': 'Unsupported file format. Please upload .ipynb file.'}), 400
    
    try:
        # Parse notebook JSON
        notebook_data = json.load(file)
        
        # Validate notebook structure
        if 'cells' not in notebook_data:
            return jsonify({'error': 'Invalid notebook format'}), 400
        
        # Extract cell information
        cells = []
        for idx, cell in enumerate(notebook_data.get('cells', [])):
            cell_type = cell.get('cell_type', 'unknown')
            cell_info = {
                'index': idx,
                'type': cell_type,
                'source': ''.join(cell.get('source', []))
            }
            
            # Add output if present
            if 'outputs' in cell and cell['outputs']:
                outputs = []
                for output in cell['outputs']:
                    if 'text' in output:
                        outputs.append({
                            'type': 'text',
                            'data': ''.join(output.get('text', []))
                        })
                    elif 'data' in output:
                        # Handle image or other data
                        outputs.append({
                            'type': 'data',
                            'data': output.get('data', {})
                        })
                cell_info['outputs'] = outputs
            
            cells.append(cell_info)
        
        return jsonify({
            'filename': secure_filename(file.filename),
            'cells': cells,
            'metadata': notebook_data.get('metadata', {}),
            'nbformat': notebook_data.get('nbformat', 0),
            'nbformat_minor': notebook_data.get('nbformat_minor', 0),
            'success': True
        })
        
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to process notebook: {str(e)}'}), 500

