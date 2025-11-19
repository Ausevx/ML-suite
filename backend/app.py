"""
ML-Suite Local Desktop Application
Main Flask application entry point
"""
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import config

# Import modules
from modules.ocr import ocr_bp
from modules.model_trainer import trainer_bp
from modules.model_management import models_bp
from modules.notebook_viewer import notebook_bp
from modules.system_stats import stats_bp
from modules.storage_manager import storage_bp

app = Flask(__name__,
                static_folder='../frontend/static',
                template_folder='../frontend/templates')
CORS(app)

# Register blueprints
app.register_blueprint(ocr_bp, url_prefix='/api/ocr')
app.register_blueprint(trainer_bp, url_prefix='/api')
app.register_blueprint(models_bp, url_prefix='/api/models')
app.register_blueprint(notebook_bp, url_prefix='/api/notebook')
app.register_blueprint(stats_bp, url_prefix='/api/system')
app.register_blueprint(storage_bp, url_prefix='/api/storage')

@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Clear screen for better visibility
    os.system('clear' if os.name != 'nt' else 'cls')
    
    print("\n" + "="*60)
    print("🚀 ML-Suite Local Desktop Application")
    print("="*60)
    print("\n✅ Server starting...")
    print(f"📂 Workspace: {os.path.abspath('.')}")
    print(f"📁 Uploads: {os.path.abspath(config.UPLOADS_DIR)}")
    print(f"🤖 Models: {os.path.abspath(config.MODELS_DIR)}")
    print("\n" + "="*60)
    print("🌐 ML-Suite is ready!")
    print("="*60)
    print(f"\n👉 Open your browser and go to:")
    print(f"\n   🔗 http://localhost:{config.PORT}")
    print(f"   🔗 http://127.0.0.1:{config.PORT}")
    print("\n💡 Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)

