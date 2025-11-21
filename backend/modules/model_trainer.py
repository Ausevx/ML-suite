"""
Model Trainer Module - Handles CSV upload, data preparation, and model training
"""
from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import uuid
import requests
from io import StringIO, BytesIO
from datetime import datetime

# CRITICAL: Set threading environment variables BEFORE any ML library imports
# This prevents OpenMP crashes when using PyTorch on macOS
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
try:
    from sklearn.experimental import enable_halving_search_cv
    from sklearn.model_selection import HalvingGridSearchCV
    HALVING_AVAILABLE = True
except ImportError:
    HALVING_AVAILABLE = False
    print("Warning: HalvingGridSearchCV not available (requires scikit-learn >=1.2)")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    print("Warning: Bayesian optimization not available. Install with: pip install scikit-optimize")

# GPU support detection
try:
    import torch
    PYTORCH_AVAILABLE = True

    # Check for different GPU backends
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

    # Check GPU configuration settings
    import config
    gpu_master_enabled = getattr(config, 'GPU_ENABLED', True)
    gpu_force_cpu = getattr(config, 'GPU_FORCE_CPU', False)
    gpu_preferred_backend = getattr(config, 'GPU_PREFERRED_BACKEND', 'auto')

    # Determine GPU availability based on hardware and settings
    hardware_gpu_available = CUDA_AVAILABLE or MPS_AVAILABLE

    if not gpu_master_enabled or gpu_force_cpu:
        GPU_AVAILABLE = False
        GPU_BACKEND = 'cpu'
        print("GPU acceleration disabled by configuration")
    elif gpu_preferred_backend == 'cpu':
        GPU_AVAILABLE = False
        GPU_BACKEND = 'cpu'
        print("GPU acceleration forced to CPU by configuration")
    elif gpu_preferred_backend == 'cuda' and CUDA_AVAILABLE:
        GPU_AVAILABLE = True
        GPU_BACKEND = 'cuda'
        print("GPU acceleration enabled: CUDA (preferred)")
    elif gpu_preferred_backend == 'mps' and MPS_AVAILABLE:
        GPU_AVAILABLE = True
        GPU_BACKEND = 'mps'
        print("GPU acceleration enabled: MPS (preferred)")
    elif hardware_gpu_available:
        GPU_AVAILABLE = True
        GPU_BACKEND = 'cuda' if CUDA_AVAILABLE else 'mps'
        print(f"GPU acceleration enabled: {GPU_BACKEND} (auto-detected)")
    else:
        GPU_AVAILABLE = False
        GPU_BACKEND = 'cpu'
        print("No GPU acceleration available")

except ImportError:
    PYTORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    CUDA_AVAILABLE = False
    MPS_AVAILABLE = False
    GPU_BACKEND = 'cpu'

# XGBoost GPU support detection (CUDA only, not MPS)
try:
    # Test XGBoost GPU support (only works with CUDA and respects GPU settings)
    import xgboost as xgb
    if CUDA_AVAILABLE and GPU_AVAILABLE and GPU_BACKEND == 'cuda':
        test_model = xgb.XGBRegressor(tree_method='gpu_hist')
        XGBOOST_GPU_AVAILABLE = True
        print("XGBoost GPU support available and enabled")
    else:
        XGBOOST_GPU_AVAILABLE = False
        if not CUDA_AVAILABLE:
            print("XGBoost GPU support not available (no CUDA)")
        elif not GPU_AVAILABLE:
            print("XGBoost GPU support disabled by configuration")
        else:
            print("XGBoost GPU support not available (MPS backend)")
except Exception as e:
    XGBOOST_GPU_AVAILABLE = False
    print(f"XGBoost GPU support not available: {e}")

if GPU_AVAILABLE:
    print("GPU acceleration available via PyTorch")
if XGBOOST_GPU_AVAILABLE:
    print("XGBoost GPU acceleration available")
if not GPU_AVAILABLE and not XGBOOST_GPU_AVAILABLE:
    print("GPU acceleration not available - using CPU only")

import joblib
import config
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import base64

trainer_bp = Blueprint('trainer', __name__)

# Store uploaded datasets in memory (simple approach)
_uploaded_datasets = {}

def get_current_gpu_settings():
    """Get current GPU settings from config (allows dynamic updates)"""
    import config
    return {
        'gpu_enabled': getattr(config, 'GPU_ENABLED', True),
        'gpu_force_cpu': getattr(config, 'GPU_FORCE_CPU', False),
        'gpu_preferred_backend': getattr(config, 'GPU_PREFERRED_BACKEND', 'auto'),
        'cuda_available': CUDA_AVAILABLE if PYTORCH_AVAILABLE else False,
        'mps_available': MPS_AVAILABLE if PYTORCH_AVAILABLE else False
    }

def should_use_gpu():
    """Determine if GPU should be used based on current settings"""
    settings = get_current_gpu_settings()
    
    if not settings['gpu_enabled'] or settings['gpu_force_cpu']:
        return False, 'cpu'
    
    preferred = settings['gpu_preferred_backend']
    
    if preferred == 'cpu':
        return False, 'cpu'
    elif preferred == 'cuda' and settings['cuda_available']:
        return True, 'cuda'
    elif preferred == 'mps' and settings['mps_available']:
        return True, 'mps'
    elif preferred == 'auto':
        if settings['cuda_available']:
            return True, 'cuda'
        elif settings['mps_available']:
            return True, 'mps'
    
    return False, 'cpu'

# Define PyTorch Neural Network classes at module level (required for pickling)
if PYTORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from threading import Lock
    from sklearn.base import BaseEstimator, RegressorMixin

    class AdvancedNN(nn.Module):
        """Advanced Neural Network with dropout and batch normalization"""
        def __init__(self, input_size, hidden_layers=None, output_size=1, dropout_rate=0.2):
            super(AdvancedNN, self).__init__()
            if hidden_layers is None:
                hidden_layers = [128, 64, 32]
            
            layers = []
            prev_size = input_size
            
            # Build hidden layers with batch norm and dropout
            for hidden_size in hidden_layers:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                prev_size = hidden_size
            
            # Output layer
            layers.append(nn.Linear(prev_size, output_size))
            
            self.layers = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.layers(x)

    class PyTorchRegressor(BaseEstimator, RegressorMixin):
        """Thread-safe PyTorch Neural Network Regressor"""
        def __init__(self, hidden_layers=None, epochs=100, learning_rate=0.001, 
                     batch_size=32, dropout_rate=0.2, weight_decay=1e-5):
            if hidden_layers is None:
                hidden_layers = [128, 64, 32]
            
            self.hidden_layers = hidden_layers
            self.epochs = epochs
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.dropout_rate = dropout_rate
            self.weight_decay = weight_decay
            self._model_lock = Lock()  # Thread safety for model access
            self.model_ = None
            self.device = None

            # Check GPU settings dynamically
            use_gpu, backend = should_use_gpu()
            
            if use_gpu:
                self.device = torch.device(backend)
                print(f"[GPU] Neural Network using {backend.upper()} acceleration")
            else:
                self.device = torch.device('cpu')
                print(f"[CPU] Neural Network using CPU (GPU disabled in settings)")

        def __getstate__(self):
            """Handle pickling by excluding unpicklable Lock object"""
            state = self.__dict__.copy()
            # Remove the unpicklable Lock object
            state.pop('_model_lock', None)
            # Also exclude the model since it contains torch modules
            state.pop('model_', None)
            return state

        def __setstate__(self, state):
            """Handle unpickling by recreating Lock object"""
            self.__dict__.update(state)
            # Recreate the Lock object
            self._model_lock = Lock()
            # model_ will be None and recreated during fit()

        def fit(self, X, y):
            """Train the neural network (thread-safe)"""
            with self._model_lock:
                # Convert to tensors
                X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
                y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).unsqueeze(1).to(self.device)

                # Create model
                self.model_ = AdvancedNN(
                    X.shape[1], 
                    self.hidden_layers, 
                    1, 
                    self.dropout_rate
                ).to(self.device)
                
                criterion = nn.MSELoss()
                optimizer = optim.Adam(
                    self.model_.parameters(), 
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
                
                # Training loop with early stopping check
                self.model_.train()
                for epoch in range(self.epochs):
                    total_loss = 0
                    num_batches = 0
                    
                    for i in range(0, len(X_tensor), self.batch_size):
                        batch_X = X_tensor[i:i+self.batch_size]
                        batch_y = y_tensor[i:i+self.batch_size]

                        optimizer.zero_grad()
                        outputs = self.model_(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        
                        # Gradient clipping to prevent exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        total_loss += loss.item()
                        num_batches += 1
                    
                    avg_loss = total_loss / num_batches
                    if (epoch + 1) % max(1, self.epochs // 10) == 0:
                        print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")

            return self

        def predict(self, X):
            """Make predictions (thread-safe)"""
            with self._model_lock:
                if self.model_ is None:
                    raise ValueError("Model not fitted yet")
                
                self.model_.eval()
                X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
                
                with torch.no_grad():
                    predictions = self.model_(X_tensor).cpu().numpy().flatten()
            
            return predictions

def allowed_csv_file(filename):
    """Check if file is CSV"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_CSV_EXTENSIONS

def get_model_category(model_type):
    """Determine the category of a model type"""
    regression_models = ['linear_regression', 'ridge', 'lasso', 'polynomial',
                        'random_forest', 'gradient_boosting', 'xgboost', 'svr', 'neural_network']
    classification_models = ['logistic_regression', 'random_forest_classifier',
                            'gradient_boosting_classifier', 'xgboost_classifier', 'svc']
    clustering_models = ['kmeans', 'dbscan', 'hierarchical']
    dim_reduction_models = ['pca']
    deep_learning_models = ['neural_network']

    if model_type in regression_models:
        return 'regression'
    elif model_type in classification_models:
        return 'classification'
    elif model_type in clustering_models:
        return 'clustering'
    elif model_type in dim_reduction_models:
        return 'dim_reduction'
    elif model_type in deep_learning_models:
        return 'regression'  # Neural network returns regression predictions
    else:
        return 'regression'  # default

@trainer_bp.route('/data/upload', methods=['POST'])
def upload_csv():
    """Upload CSV file and return preview"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_csv_file(file.filename):
        return jsonify({'error': 'Unsupported file format. Please upload CSV.'}), 400
    
    try:
        # Read CSV
        df = pd.read_csv(file)
        
        # Validation
        if len(df.columns) < 2:
            return jsonify({'error': 'CSV must have at least 2 columns'}), 400
        
        if len(df) < 10:
            return jsonify({'error': 'CSV must have at least 10 rows'}), 400
        
        # Store dataset
        dataset_id = str(uuid.uuid4())
        _uploaded_datasets[dataset_id] = {
            'dataframe': df,
            'filename': secure_filename(file.filename),
            'uploaded_at': datetime.now().isoformat()
        }
        
        # Get row/column range from form data (if provided)
        row_start = int(request.form.get('row_start', 0))
        row_end = int(request.form.get('row_end', min(10, len(df))))
        col_start = int(request.form.get('col_start', 0))
        col_end = int(request.form.get('col_end', len(df.columns)))
        
        # Validate ranges
        row_start = max(0, min(row_start, len(df) - 1))
        row_end = max(row_start + 1, min(row_end, len(df)))
        col_start = max(0, min(col_start, len(df.columns) - 1))
        col_end = max(col_start + 1, min(col_end, len(df.columns)))
        
        # Get subset of columns for preview
        preview_columns = df.columns[col_start:col_end]
        
        # Get preview with specified range - AGGRESSIVE JSON sanitization
        preview = []
        for idx in range(row_start, row_end):
            row_dict = {}
            for col in preview_columns:
                val = df.iloc[idx][col]
                try:
                    if pd.isna(val):
                        row_dict[str(col)] = None
                    elif isinstance(val, (int, np.integer)):
                        row_dict[str(col)] = int(val)
                    elif isinstance(val, (float, np.floating)):
                        if np.isinf(val):
                            row_dict[str(col)] = None
                        else:
                            row_dict[str(col)] = float(val)
                    elif isinstance(val, str):
                        # Truncate and clean strings
                        clean_val = val.encode('utf-8', 'ignore').decode('utf-8')[:150]
                        row_dict[str(col)] = clean_val
                    else:
                        row_dict[str(col)] = str(val)[:150]
                except:
                    row_dict[str(col)] = None
            preview.append(row_dict)
        
        # Get column info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isnull().sum())
            
            # Get sample values, sanitize them
            sample_vals = df[col].dropna().head(3)
            # Convert to list and handle special values
            sample_list = []
            for val in sample_vals:
                if pd.isna(val):
                    sample_list.append(None)
                elif isinstance(val, (np.integer, np.floating)):
                    if np.isinf(val):
                        sample_list.append(None)
                    else:
                        sample_list.append(float(val) if isinstance(val, np.floating) else int(val))
                else:
                    # Convert to string and truncate if too long
                    str_val = str(val)
                    sample_list.append(str_val[:100] if len(str_val) > 100 else str_val)
            
            columns_info.append({
                'name': col,
                'type': dtype,
                'null_count': null_count,
                'sample_values': sample_list
            })
        
        return jsonify({
            'dataset_id': dataset_id,
            'filename': secure_filename(file.filename),
            'preview': preview,
            'columns': columns_info,
            'row_count': len(df),
            'column_count': len(df.columns),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to process CSV: {str(e)}'}), 500

@trainer_bp.route('/data/import-api', methods=['POST'])
def import_from_api():
    """Import dataset from API endpoint"""
    data = request.get_json()
    
    if not data or 'url' not in data:
        return jsonify({'error': 'API URL is required'}), 400
    
    api_url = data['url']
    method = data.get('method', 'GET').upper()
    headers = data.get('headers', {})
    body = data.get('body', {})
    
    try:
        # Make API request
        if method == 'GET':
            response = requests.get(api_url, headers=headers, timeout=30)
        elif method == 'POST':
            response = requests.post(api_url, headers=headers, json=body, timeout=30)
        else:
            return jsonify({'error': 'Invalid HTTP method'}), 400
        
        response.raise_for_status()
        
        # Try to parse as CSV
        try:
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            
            if 'json' in content_type:
                # If JSON, try to convert to DataFrame
                json_data = response.json()
                if isinstance(json_data, list):
                    df = pd.DataFrame(json_data)
                elif isinstance(json_data, dict):
                    df = pd.DataFrame([json_data])
                else:
                    return jsonify({'error': 'JSON response format not supported'}), 400
            else:
                # Assume CSV
                df = pd.read_csv(StringIO(response.text))
        except Exception as e:
            return jsonify({'error': f'Failed to parse response as CSV/JSON: {str(e)}'}), 400
        
        # Validation
        if len(df.columns) < 2:
            return jsonify({'error': 'Data must have at least 2 columns'}), 400
        
        if len(df) < 10:
            return jsonify({'error': 'Data must have at least 10 rows'}), 400
        
        # Store dataset
        dataset_id = str(uuid.uuid4())
        _uploaded_datasets[dataset_id] = {
            'dataframe': df,
            'filename': f'api_import_{dataset_id[:8]}.csv',
            'uploaded_at': datetime.now().isoformat(),
            'source': 'api',
            'api_url': api_url
        }
        
        # Get preview (first 10 rows) - AGGRESSIVE JSON sanitization
        preview = []
        for idx in range(min(10, len(df))):
            row_dict = {}
            for col in df.columns:
                val = df.iloc[idx][col]
                try:
                    if pd.isna(val):
                        row_dict[str(col)] = None
                    elif isinstance(val, (int, np.integer)):
                        row_dict[str(col)] = int(val)
                    elif isinstance(val, (float, np.floating)):
                        if np.isinf(val):
                            row_dict[str(col)] = None
                        else:
                            row_dict[str(col)] = float(val)
                    elif isinstance(val, str):
                        # Truncate and clean strings
                        clean_val = val.encode('utf-8', 'ignore').decode('utf-8')[:150]
                        row_dict[str(col)] = clean_val
                    else:
                        row_dict[str(col)] = str(val)[:150]
                except:
                    row_dict[str(col)] = None
            preview.append(row_dict)
        
        # Get column info
        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_count = int(df[col].isnull().sum())
            
            # Get sample values, sanitize them
            sample_vals = df[col].dropna().head(3)
            # Convert to list and handle special values
            sample_list = []
            for val in sample_vals:
                if pd.isna(val):
                    sample_list.append(None)
                elif isinstance(val, (np.integer, np.floating)):
                    if np.isinf(val):
                        sample_list.append(None)
                    else:
                        sample_list.append(float(val) if isinstance(val, np.floating) else int(val))
                else:
                    # Convert to string and truncate if too long
                    str_val = str(val)
                    sample_list.append(str_val[:100] if len(str_val) > 100 else str_val)
            
            columns_info.append({
                'name': col,
                'type': dtype,
                'null_count': null_count,
                'sample_values': sample_list
            })
        
        return jsonify({
            'dataset_id': dataset_id,
            'filename': f'api_import_{dataset_id[:8]}.csv',
            'preview': preview,
            'columns': columns_info,
            'row_count': len(df),
            'column_count': len(df.columns),
            'source': 'api',
            'success': True
        })
        
    except requests.exceptions.Timeout:
        return jsonify({'error': 'API request timed out'}), 408
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'API request failed: {str(e)}'}), 500
    except Exception as e:
        return jsonify({'error': f'Failed to import data: {str(e)}'}), 500

@trainer_bp.route('/data/analyze', methods=['POST'])
def analyze_data():
    """Analyze uploaded CSV and return statistics"""
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    
    if dataset_id not in _uploaded_datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    df = _uploaded_datasets[dataset_id]['dataframe']
    
    # Calculate statistics
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicate_rows': int(df.duplicated().sum())
    }
    
    for col in df.columns:
        stats['missing_values'][col] = int(df[col].isnull().sum())
    
    return jsonify(stats)

@trainer_bp.route('/model/train', methods=['POST'])
def train_model():
    """Train a regression model with provided configuration"""
    import time
    start_time = time.time()
    
    data = request.get_json()
    
    # Extract configuration
    dataset_id = data.get('dataset_id')
    target_column = data.get('target_column')
    feature_columns = data.get('feature_columns', [])
    missing_value_strategy = data.get('missing_value_strategy', {})
    remove_duplicates = data.get('remove_duplicates', False)
    train_size = data.get('train_size', config.DEFAULT_TRAIN_TEST_SPLIT)
    model_name = data.get('model_name', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    model_type = data.get('model_type', 'linear_regression')  # New: model type selection
    model_params = data.get('model_params', {})  # New: model-specific parameters
    
    # Extract GPU settings from request (overrides config defaults)
    request_gpu_enabled = data.get('gpu_enabled', getattr(config, 'GPU_ENABLED', True))
    request_gpu_force_cpu = data.get('gpu_force_cpu', getattr(config, 'GPU_FORCE_CPU', False))
    
    # Disable GPU entirely for neural networks or when CPU is forced
    is_deep_learning = model_type == 'neural_network'
    if is_deep_learning or request_gpu_force_cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU entirely
    
    if dataset_id not in _uploaded_datasets:
        return jsonify({'error': 'Dataset not found'}), 404
    
    try:
        df = _uploaded_datasets[dataset_id]['dataframe'].copy()
        
        # Validate columns
        if target_column not in df.columns:
            return jsonify({'error': f'Target column "{target_column}" not found'}), 400
        
        for col in feature_columns:
            if col not in df.columns:
                return jsonify({'error': f'Feature column "{col}" not found'}), 400
        
        # Track preprocessing steps
        preprocessing_steps = {
            'removed_duplicates': remove_duplicates,
            'missing_value_strategies': {},
            'dropped_columns': [],
            'encoded_columns': {},
            'original_feature_count': len(feature_columns),
            'original_row_count': len(df)
        }
        
        # Remove duplicates if requested
        if remove_duplicates:
            initial_rows = len(df)
            df = df.drop_duplicates()
            preprocessing_steps['duplicates_removed_count'] = initial_rows - len(df)
        
        # Handle missing values and track strategies
        for col, strategy in missing_value_strategy.items():
            if col not in df.columns:
                continue
            
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                preprocessing_steps['missing_value_strategies'][col] = {
                    'strategy': strategy,
                    'missing_count': int(missing_count)
                }
            
            if strategy == 'drop':
                df = df.dropna(subset=[col])
            elif strategy == 'mean':
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
                preprocessing_steps['missing_value_strategies'][col]['fill_value'] = float(fill_value)
            elif strategy == 'median':
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                preprocessing_steps['missing_value_strategies'][col]['fill_value'] = float(fill_value)
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(fill_value)
                preprocessing_steps['missing_value_strategies'][col]['fill_value'] = float(fill_value)
            elif strategy == 'zero':
                df[col] = df[col].fillna(0)
                preprocessing_steps['missing_value_strategies'][col]['fill_value'] = 0
        
        # Determine model category
        model_category = get_model_category(model_type)
        
        # Prepare features and target
        X = df[feature_columns].select_dtypes(include=[np.number])
        
        # Track which columns were dropped (non-numeric)
        dropped_cols = set(feature_columns) - set(X.columns)
        if dropped_cols:
            preprocessing_steps['dropped_columns'] = list(dropped_cols)
            preprocessing_steps['drop_reason'] = 'non-numeric'
        
        y = df[target_column]
        
        # Validate target based on model category
        label_classes = None
        if model_category == 'regression':
            if not pd.api.types.is_numeric_dtype(y):
                return jsonify({'error': 'Target column must be numeric for regression'}), 400
        elif model_category == 'classification':
            # Classification can handle both numeric and categorical targets
            # Encode categorical targets if needed
            if not pd.api.types.is_numeric_dtype(y):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_original = y.copy()
                y = pd.Series(le.fit_transform(y), index=y.index)
                # Store label encoder info for later
                label_classes = list(le.classes_)
                preprocessing_steps['encoded_columns'][target_column] = {
                    'type': 'target',
                    'encoding': 'LabelEncoder',
                    'classes': label_classes,
                    'original_values': list(y_original.unique())
                }
            else:
                label_classes = list(y.unique())
        elif model_category in ['clustering', 'dim_reduction']:
            # Unsupervised learning: no target needed (but we'll use it for visualization)
            pass
        
        # Track final dataset shape after preprocessing
        preprocessing_steps['final_row_count'] = len(df)
        preprocessing_steps['final_feature_count'] = len(X.columns)
        
        if len(X.columns) == 0:
            return jsonify({'error': 'No numeric feature columns available'}), 400
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-train_size, random_state=42
        )
        
        # Override config GPU settings with request parameters for this training run
        # This ensures neural networks get the CPU-only settings sent from frontend
        original_gpu_enabled = getattr(config, 'GPU_ENABLED', True)
        original_gpu_force_cpu = getattr(config, 'GPU_FORCE_CPU', False)
        
        config.GPU_ENABLED = request_gpu_enabled
        config.GPU_FORCE_CPU = request_gpu_force_cpu
        
        # Create model based on type
        model, X_train_transformed, X_test_transformed = create_model(
            model_type, model_params, X_train, X_test
        )
        
        # Hyperparameter tuning if enabled
        tuning_config = data.get('hyperparameter_tuning', None)
        best_params = {}
        
        if tuning_config:
            print(f"Starting hyperparameter tuning: {tuning_config['method']}")
            model, best_params = perform_hyperparameter_tuning(
                model, 
                X_train_transformed, 
                y_train, 
                model_type,
                tuning_config
            )
            print(f"Tuning complete. Best params: {best_params}")
        else:
            # Train model normally
            model.fit(X_train_transformed, y_train)
        
        # Predictions and metrics based on model category
        metrics = {}
        predictions_data = {}
        
        if model_category == 'regression':
            # Regression: predictions are continuous values
            y_train_pred = model.predict(X_train_transformed)
            y_test_pred = model.predict(X_test_transformed)
            
            # Calculate regression metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            # Calculate adjusted R2
            n = len(y_test)
            p = len(X.columns)
            adjusted_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1) if n > p + 1 else test_r2
            
            # Calculate residuals
            residuals = y_test - y_test_pred
            
            metrics = {
                'r2_score': float(test_r2),
                'rmse': float(test_rmse),
                'mae': float(test_mae),
                'mse': float(test_mse),
                'adjusted_r2': float(adjusted_r2)
            }
            
            predictions_data = {
                'actual': y_test.tolist(),
                'predicted': y_test_pred.tolist(),
                'residuals': residuals.tolist()
            }
        
        elif model_category == 'classification':
            # Classification: predictions are class labels
            y_train_pred = model.predict(X_train_transformed)
            y_test_pred = model.predict(X_test_transformed)
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                y_test_proba = model.predict_proba(X_test_transformed)
            else:
                y_test_proba = None
            
            # Calculate classification metrics
            test_accuracy = accuracy_score(y_test, y_test_pred)
            
            # Handle binary and multi-class classification
            n_classes = len(np.unique(y))
            if n_classes == 2:
                # Binary classification
                test_precision = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='binary', zero_division=0)
                
                # ROC-AUC for binary classification
                if y_test_proba is not None:
                    test_roc_auc = roc_auc_score(y_test, y_test_proba[:, 1])
                else:
                    test_roc_auc = None
            else:
                # Multi-class classification
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                test_roc_auc = None  # ROC-AUC for multi-class is more complex
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_test_pred).tolist()
            
            metrics = {
                'accuracy': float(test_accuracy),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1_score': float(test_f1),
                'confusion_matrix': conf_matrix,
                'n_classes': n_classes
            }
            
            if test_roc_auc is not None:
                metrics['roc_auc'] = float(test_roc_auc)
            
            predictions_data = {
                'actual': y_test.tolist(),
                'predicted': y_test_pred.tolist()
            }
            
            if y_test_proba is not None:
                predictions_data['probabilities'] = y_test_proba.tolist()
        
        elif model_category == 'clustering':
            # Clustering: no train-test split needed, fit on all data
            labels = model.fit_predict(X)
            
            # Calculate clustering metrics
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters
                silhouette = silhouette_score(X, labels)
                davies_bouldin = davies_bouldin_score(X, labels)
            else:
                silhouette = None
                davies_bouldin = None
            
            metrics = {
                'n_clusters': len(np.unique(labels)),
                'silhouette_score': float(silhouette) if silhouette is not None else None,
                'davies_bouldin_score': float(davies_bouldin) if davies_bouldin is not None else None
            }
            
            predictions_data = {
                'labels': labels.tolist()
            }
            
            # For clustering, we don't have train/test predictions
            y_train_pred = None
            y_test_pred = labels
            
        elif model_category == 'dim_reduction':
            # Dimensionality reduction: transform data
            X_transformed = model.fit_transform(X)
            
            metrics = {
                'n_components': model.n_components,
                'explained_variance': model.explained_variance_.tolist() if hasattr(model, 'explained_variance_') else None,
                'explained_variance_ratio': model.explained_variance_ratio_.tolist() if hasattr(model, 'explained_variance_ratio_') else None
            }
            
            predictions_data = {
                'transformed': X_transformed.tolist()
            }
            
            y_train_pred = None
            y_test_pred = None
        
        # Execute custom code if provided
        custom_code = data.get('custom_code', None)
        custom_output = None
        if custom_code:
            try:
                # Create safe execution environment
                exec_globals = {
                    'model': model,
                    'X_train': X_train_transformed,
                    'X_test': X_test_transformed,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_train_pred': y_train_pred,
                    'y_test_pred': y_test_pred,
                    'residuals': residuals,
                    'np': np,
                    'pd': pd,
                    'plt': plt
                }
                exec_locals = {}
                
                # Capture print output
                import io
                import sys
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                
                # Execute custom code
                exec(custom_code, exec_globals, exec_locals)
                
                # Get captured output
                custom_output = sys.stdout.getvalue()
                sys.stdout = old_stdout
                
                if not custom_output:
                    custom_output = "Custom code executed successfully (no output)"
                    
                print(f"Custom code executed: {custom_output[:100]}...")
            except Exception as e:
                custom_output = f"Error executing custom code: {str(e)}"
                print(f"Custom code error: {e}")
        
        # Save model
        model_id = str(uuid.uuid4())
        model_path = os.path.join(config.MODELS_DIR, f'{model_id}.pkl')
        joblib.dump(model, model_path)
        
        # Save metadata
        # Get coefficients if available
        coefficients = {}
        if hasattr(model, 'coef_'):
            if hasattr(model.coef_, '__iter__') and len(model.coef_.shape) > 0:
                if len(model.coef_.shape) == 1 and len(X.columns) == len(model.coef_):
                    coefficients = {col: float(coef) for col, coef in zip(X.columns, model.coef_)}
                elif len(model.coef_.shape) == 2:
                    # Multi-class classification coefficients
                    coefficients = {col: model.coef_[:, i].tolist() for i, col in enumerate(X.columns)}
        elif hasattr(model, 'feature_importances_'):
            coefficients = {col: float(imp) for col, imp in zip(X.columns, model.feature_importances_)}
        
        intercept = float(model.intercept_) if hasattr(model, 'intercept_') and np.isscalar(model.intercept_) else (model.intercept_.tolist() if hasattr(model, 'intercept_') else 0.0)
        
        # Calculate training time early for metadata
        training_time = time.time() - start_time
        
        # Get hardware info early for metadata
        use_gpu, backend = should_use_gpu()
        hardware_used = {
            'device': 'cpu',
            'backend': 'cpu',
            'gpu_accelerated': False
        }
        
        if model_type in ['xgboost', 'xgboost_classifier']:
            if XGBOOST_GPU_AVAILABLE and use_gpu and backend == 'cuda':
                hardware_used = {'device': 'gpu', 'backend': 'cuda', 'gpu_accelerated': True}
        elif model_type == 'neural_network':
            if use_gpu and PYTORCH_AVAILABLE:
                hardware_used = {'device': 'gpu', 'backend': backend, 'gpu_accelerated': True}
        
        metadata = {
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'model_category': model_category,
            'created_at': datetime.now().isoformat(),
            'dataset_name': _uploaded_datasets[dataset_id]['filename'],
            'features': list(X.columns),
            'target': target_column,
            'train_test_split': train_size,
            'metrics': metrics,
            'file_path': model_path,
            'coefficients': coefficients,
            'intercept': intercept,
            'preprocessing': preprocessing_steps,
            'label_classes': label_classes,
            'training_time': round(training_time, 2),
            'hardware_used': hardware_used
        }
        
        metadata_path = os.path.join(config.MODELS_DIR, f'{model_id}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Generate charts based on request settings and model category
        charts = {}
        chart_settings = data.get('chart_settings', {})
        
        if model_category == 'regression':
            # Regression charts
            if chart_settings.get('actualVsPredicted', True):
                charts['actualVsPredicted'] = generate_actual_vs_predicted_chart(y_test, y_test_pred)
            
            if chart_settings.get('residualPlot', True):
                charts['residualPlot'] = generate_residual_plot(y_test_pred, residuals)
            
            if chart_settings.get('featureImportance', True):
                # Use coefficients if available, otherwise feature importances for tree models
                if hasattr(model, 'coef_') and len(model.coef_.shape) == 1 and len(X.columns) == len(model.coef_):
                    feature_values = model.coef_
                elif hasattr(model, 'feature_importances_'):
                    feature_values = model.feature_importances_
                else:
                    feature_values = None

                if feature_values is not None:
                    charts['featureImportance'] = generate_feature_importance_chart(X.columns, feature_values)
            
            if chart_settings.get('distribution', False):
                charts['distribution'] = generate_distribution_chart(y_test, y_test_pred)
        
        elif model_category == 'classification':
            # Classification charts
            if chart_settings.get('confusionMatrix', True):
                charts['confusionMatrix'] = generate_confusion_matrix_chart(metrics['confusion_matrix'], list(range(metrics['n_classes'])))
            
            if chart_settings.get('featureImportance', True) and hasattr(model, 'feature_importances_'):
                charts['featureImportance'] = generate_feature_importance_chart(X.columns, model.feature_importances_)
            
            if chart_settings.get('rocCurve', False) and metrics.get('roc_auc') is not None and y_test_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_test_proba[:, 1])
                charts['rocCurve'] = generate_roc_curve_chart(fpr, tpr, metrics['roc_auc'])
        
        elif model_category == 'clustering':
            # Clustering charts
            if chart_settings.get('clusterScatter', True) and X.shape[1] >= 2:
                # Use first 2 features for visualization
                charts['clusterScatter'] = generate_cluster_scatter_chart(X.iloc[:, :2], y_test_pred)
        
        elif model_category == 'dim_reduction':
            # Dimensionality reduction charts
            if chart_settings.get('scree', True) and metrics.get('explained_variance_ratio') is not None:
                charts['screePlot'] = generate_scree_plot(metrics['explained_variance_ratio'])
            
            if chart_settings.get('transformedScatter', True) and predictions_data.get('transformed') is not None:
                X_transformed = np.array(predictions_data['transformed'])
                if X_transformed.shape[1] >= 2:
                    charts['transformedScatter'] = generate_transformed_scatter(X_transformed[:, :2])
        
        # Calculate training time and hardware info
        training_time = time.time() - start_time

        # Get hardware acceleration info (use dynamic settings check)
        use_gpu, backend = should_use_gpu()
        
        hardware_info = {
            'device_used': 'cpu',
            'gpu_accelerated': False,
            'gpu_backend': None,
            'gpu_memory_used': None
        }

        # Check if GPU was actually used for this model
        if model_type in ['xgboost', 'xgboost_classifier']:
            if XGBOOST_GPU_AVAILABLE and use_gpu and backend == 'cuda':
                hardware_info['device_used'] = 'cuda'
                hardware_info['gpu_accelerated'] = True
                hardware_info['gpu_backend'] = 'cuda'
        elif model_type == 'neural_network':
            if use_gpu and PYTORCH_AVAILABLE:
                hardware_info['gpu_accelerated'] = True
                hardware_info['device_used'] = backend
                hardware_info['gpu_backend'] = backend
                # Try to get CUDA GPU memory info
                if backend == 'cuda':
                    try:
                        hardware_info['gpu_memory_used'] = round(torch.cuda.memory_allocated() / (1024**3), 2)
                    except:
                        pass

        return jsonify({
            'model_id': model_id,
            'model_name': model_name,
            'model_type': model_type,
            'metrics': metadata['metrics'],
            'predictions': predictions_data,
            'coefficients': metadata['coefficients'],
            'intercept': metadata['intercept'],
            'train_size': len(X_train) if model_category != 'clustering' else len(X),
            'test_size': len(X_test) if model_category != 'clustering' else 0,
            'charts': charts,
            'custom_output': custom_output,
            'best_params': best_params if best_params else None,
            'training_time': round(training_time, 2),
            'hardware_info': hardware_info,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500
    finally:
        # Restore original GPU config settings (always executed)
        try:
            config.GPU_ENABLED = original_gpu_enabled
            config.GPU_FORCE_CPU = original_gpu_force_cpu
        except:
            pass

def generate_actual_vs_predicted_chart(y_actual, y_predicted):
    """Generate actual vs predicted scatter plot"""
    try:
        plt.figure(figsize=(10, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        # Scatter plot
        plt.scatter(y_actual, y_predicted, alpha=0.6, color='#00ff00', edgecolors='#00ff00', linewidths=0.5)
        
        # Perfect prediction line
        min_val = min(y_actual.min(), y_predicted.min())
        max_val = max(y_actual.max(), y_predicted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit', color='#ff0000')
        
        plt.xlabel('Actual Values', color='#00ff00', fontsize=12)
        plt.ylabel('Predicted Values', color='#00ff00', fontsize=12)
        plt.title('Actual vs Predicted Values', color='#00ff00', fontsize=14, pad=20)
        plt.legend(facecolor='#000000', edgecolor='#00ff00', labelcolor='#00ff00')
        plt.grid(True, alpha=0.2, color='#00ff00')
        
        # Style ticks
        ax.tick_params(colors='#00ff00', which='both')
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['top'].set_color('#00ff00')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['right'].set_color('#00ff00')
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#000000', edgecolor='#00ff00', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{img_base64}'
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def generate_residual_plot(y_predicted, residuals):
    """Generate residual plot"""
    try:
        plt.figure(figsize=(10, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        # Scatter plot of residuals
        plt.scatter(y_predicted, residuals, alpha=0.6, color='#00ff00', edgecolors='#00ff00', linewidths=0.5)
        plt.axhline(y=0, color='#ff0000', linestyle='--', linewidth=2, label='Zero Residual')
        
        plt.xlabel('Predicted Values', color='#00ff00', fontsize=12)
        plt.ylabel('Residuals', color='#00ff00', fontsize=12)
        plt.title('Residual Plot', color='#00ff00', fontsize=14, pad=20)
        plt.legend(facecolor='#000000', edgecolor='#00ff00', labelcolor='#00ff00')
        plt.grid(True, alpha=0.2, color='#00ff00')
        
        # Style ticks
        ax.tick_params(colors='#00ff00', which='both')
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['top'].set_color('#00ff00')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['right'].set_color('#00ff00')
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#000000', edgecolor='#00ff00', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{img_base64}'
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def generate_feature_importance_chart(feature_names, coefficients):
    """Generate feature importance bar chart"""
    try:
        plt.figure(figsize=(10, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        # Sort features by coefficient magnitude
        indices = np.argsort(np.abs(coefficients))[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_coefs = [coefficients[i] for i in indices]
        
        # Bar chart
        bars = plt.barh(sorted_features, sorted_coefs, color='#00ff00', edgecolor='#00ff00', linewidth=1)
        
        # Color negative bars differently
        for i, coef in enumerate(sorted_coefs):
            if coef < 0:
                bars[i].set_color('#ff0000')
                bars[i].set_edgecolor('#ff0000')
        
        plt.xlabel('Coefficient Value', color='#00ff00', fontsize=12)
        plt.ylabel('Features', color='#00ff00', fontsize=12)
        plt.title('Feature Importance (Coefficients)', color='#00ff00', fontsize=14, pad=20)
        plt.grid(True, alpha=0.2, color='#00ff00', axis='x')
        
        # Style ticks
        ax.tick_params(colors='#00ff00', which='both')
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['top'].set_color('#00ff00')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['right'].set_color('#00ff00')
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#000000', edgecolor='#00ff00', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{img_base64}'
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def generate_distribution_chart(y_actual, y_predicted):
    """Generate distribution comparison histogram"""
    try:
        plt.figure(figsize=(10, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        # Histograms
        plt.hist(y_actual, bins=30, alpha=0.5, label='Actual', color='#00ff00', edgecolor='#00ff00')
        plt.hist(y_predicted, bins=30, alpha=0.5, label='Predicted', color='#00ffff', edgecolor='#00ffff')
        
        plt.xlabel('Value', color='#00ff00', fontsize=12)
        plt.ylabel('Frequency', color='#00ff00', fontsize=12)
        plt.title('Distribution: Actual vs Predicted', color='#00ff00', fontsize=14, pad=20)
        plt.legend(facecolor='#000000', edgecolor='#00ff00', labelcolor='#00ff00')
        plt.grid(True, alpha=0.2, color='#00ff00')
        
        # Style ticks
        ax.tick_params(colors='#00ff00', which='both')
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['top'].set_color('#00ff00')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['right'].set_color('#00ff00')
        
        # Convert to base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, facecolor='#000000', edgecolor='#00ff00', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{img_base64}'
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

def create_model(model_type, params, X_train, X_test):
    """Create and configure a model based on type"""
    X_train_transformed = X_train.copy()
    X_test_transformed = X_test.copy()
    
    if model_type == 'linear_regression':
        model = LinearRegression()
    
    elif model_type == 'ridge':
        alpha = params.get('alpha', 1.0)
        model = Ridge(alpha=alpha)
    
    elif model_type == 'lasso':
        alpha = params.get('alpha', 1.0)
        model = Lasso(alpha=alpha)
    
    elif model_type == 'polynomial':
        degree = params.get('degree', 2)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_transformed = poly.fit_transform(X_train)
        X_test_transformed = poly.transform(X_test)
        model = LinearRegression()
    
    elif model_type == 'random_forest':
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    elif model_type == 'gradient_boosting':
        n_estimators = params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        max_depth = params.get('max_depth', 3)
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    elif model_type == 'xgboost':
        if not XGBOOST_AVAILABLE:
            raise Exception("XGBoost not available. Please install: pip install xgboost")
        n_estimators = params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        max_depth = params.get('max_depth', 6)

        # Check GPU settings dynamically
        use_gpu, backend = should_use_gpu()
        
        # Use GPU acceleration if available and enabled
        if XGBOOST_GPU_AVAILABLE and use_gpu and backend == 'cuda':
            tree_method = 'gpu_hist'
            print(f"[GPU] XGBoost using CUDA acceleration (gpu_hist)")
        else:
            tree_method = 'hist'
            if not use_gpu:
                print("[CPU] XGBoost using CPU (GPU disabled in settings)")
            elif backend != 'cuda':
                print(f"[CPU] XGBoost using CPU (XGBoost requires CUDA, current: {backend})")
            else:
                print("[CPU] XGBoost using CPU (CUDA not available)")

        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            tree_method=tree_method,
            n_jobs=-1 if tree_method != 'gpu_hist' else None  # n_jobs not used with GPU
        )
    
    elif model_type == 'svr':
        kernel = params.get('kernel', 'rbf')
        C = params.get('C', 1.0)
        model = SVR(kernel=kernel, C=C)
    
    # Classification Models
    elif model_type == 'logistic_regression':
        C = params.get('C', 1.0)
        max_iter = params.get('max_iter', 1000)
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    
    elif model_type == 'random_forest_classifier':
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    
    elif model_type == 'gradient_boosting_classifier':
        n_estimators = params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        max_depth = params.get('max_depth', 3)
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    elif model_type == 'xgboost_classifier':
        if not XGBOOST_AVAILABLE:
            raise Exception("XGBoost not available. Please install: pip install xgboost")
        n_estimators = params.get('n_estimators', 100)
        learning_rate = params.get('learning_rate', 0.1)
        max_depth = params.get('max_depth', 6)

        # Check GPU settings dynamically
        use_gpu, backend = should_use_gpu()
        
        # Use GPU acceleration if available and enabled
        if XGBOOST_GPU_AVAILABLE and use_gpu and backend == 'cuda':
            tree_method = 'gpu_hist'
            print(f"[GPU] XGBoost Classifier using CUDA acceleration (gpu_hist)")
        else:
            tree_method = 'hist'
            if not use_gpu:
                print("[CPU] XGBoost Classifier using CPU (GPU disabled in settings)")
            elif backend != 'cuda':
                print(f"[CPU] XGBoost Classifier using CPU (XGBoost requires CUDA, current: {backend})")
            else:
                print("[CPU] XGBoost Classifier using CPU (CUDA not available)")

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            tree_method=tree_method,
            n_jobs=-1 if tree_method != 'gpu_hist' else None  # n_jobs not used with GPU
        )
    
    elif model_type == 'svc':
        kernel = params.get('kernel', 'rbf')
        C = params.get('C', 1.0)
        model = SVC(kernel=kernel, C=C, random_state=42, probability=True)
    
    # Unsupervised Models
    elif model_type == 'kmeans':
        n_clusters = params.get('n_clusters', 3)
        max_iter = params.get('max_iter', 300)
        model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=42)
    
    elif model_type == 'dbscan':
        eps = params.get('eps', 0.5)
        min_samples = params.get('min_samples', 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    
    elif model_type == 'hierarchical':
        n_clusters = params.get('n_clusters', 3)
        linkage = params.get('linkage', 'ward')
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    
    elif model_type == 'pca':
        n_components = params.get('n_components', 2)
        model = PCA(n_components=n_components, random_state=42)

    # Deep Learning Models (PyTorch-based)
    elif model_type == 'neural_network':
        if not PYTORCH_AVAILABLE:
            raise Exception("PyTorch not available. Please install: pip install torch torchvision torchaudio")

        hidden_layers = params.get('hidden_layers', [128, 64, 32])
        epochs = params.get('epochs', 100)
        learning_rate = params.get('learning_rate', 0.001)
        batch_size = params.get('batch_size', 32)
        dropout_rate = params.get('dropout_rate', 0.2)
        weight_decay = params.get('weight_decay', 1e-5)

        model = PyTorchRegressor(
            hidden_layers=hidden_layers,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            weight_decay=weight_decay
        )


    else:
        # Default to linear regression
        model = LinearRegression()
    
    return model, X_train_transformed, X_test_transformed

def perform_hyperparameter_tuning(base_model, X_train, y_train, model_type, tuning_config):
    """Perform hyperparameter tuning using specified method"""
    method = tuning_config.get('method', 'random')
    cv_folds = tuning_config.get('cv_folds', 3)
    n_iter = tuning_config.get('n_iter', 10)
    
    # Define parameter grids for each model type
    param_grids = {
        'linear_regression': {},  # No hyperparameters to tune
        'ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        'lasso': {
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'polynomial': {},  # Degree is fixed by user
        'random_forest': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        },
        'gradient_boosting': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10]
        },
        'xgboost': {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9, 12],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        },
        'svr': {
            'C': [0.1, 1.0, 10.0, 100.0],
            'epsilon': [0.01, 0.1, 0.2],
            'kernel': ['linear', 'rbf', 'poly']
        }
    }
    
    # Define Bayesian search spaces (for scikit-optimize)
    bayesian_spaces = {
        'ridge': {'alpha': Real(0.001, 100.0, prior='log-uniform')},
        'lasso': {'alpha': Real(0.0001, 10.0, prior='log-uniform')},
        'random_forest': {
            'n_estimators': Integer(50, 300),
            'max_depth': Integer(5, 50),
            'min_samples_split': Integer(2, 10),
            'min_samples_leaf': Integer(1, 4)
        },
        'gradient_boosting': {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 9),
            'subsample': Real(0.7, 1.0)
        },
        'xgboost': {
            'n_estimators': Integer(50, 300),
            'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'max_depth': Integer(3, 12),
            'subsample': Real(0.7, 1.0),
            'colsample_bytree': Real(0.7, 1.0)
        },
        'svr': {
            'C': Real(0.1, 100.0, prior='log-uniform'),
            'epsilon': Real(0.01, 0.5, prior='log-uniform')
        }
    }
    
    param_grid = param_grids.get(model_type, {})
    
    if not param_grid and model_type not in ['linear_regression', 'polynomial']:
        print(f"No parameter grid defined for {model_type}, skipping tuning")
        base_model.fit(X_train, y_train)
        return base_model, {}
    
    if model_type in ['linear_regression', 'polynomial']:
        # No tuning needed
        base_model.fit(X_train, y_train)
        return base_model, {}
    
    try:
        if method == 'grid':
            # Grid Search
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                verbose=1
            )
            
        elif method == 'random':
            # Random Search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
        elif method == 'halving':
            # Halving Grid Search
            if not HALVING_AVAILABLE:
                print("Halving search not available, falling back to RandomizedSearchCV")
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            else:
                search = HalvingGridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
                
        elif method == 'bayesian':
            # Bayesian Optimization
            if not BAYESIAN_AVAILABLE:
                print("Bayesian optimization not available, falling back to RandomizedSearchCV")
                search = RandomizedSearchCV(
                    base_model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            else:
                search_space = bayesian_spaces.get(model_type, param_grid)
                search = BayesSearchCV(
                    base_model,
                    search_space,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='r2',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
        else:
            # Default to random search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv_folds,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
        
        # Perform the search
        search.fit(X_train, y_train)
        
        print(f"Best score: {search.best_score_:.4f}")
        print(f"Best parameters: {search.best_params_}")
        
        return search.best_estimator_, search.best_params_
        
    except Exception as e:
        print(f"Hyperparameter tuning failed: {e}")
        print("Training with default parameters instead")
        base_model.fit(X_train, y_train)
        return base_model, {}

@trainer_bp.route('/models/categories', methods=['GET'])
def get_model_categories():
    """Get model categories and their types"""
    categories = {
        'supervised_regression': {
            'name': 'Supervised - Regression',
            'description': 'Predict continuous numerical values',
            'use_cases': 'House prices, sales forecasting, temperature prediction',
            'models': [
                'linear_regression', 'ridge', 'lasso', 'polynomial',
                'random_forest', 'gradient_boosting', 'xgboost', 'svr'
            ]
        },
        'supervised_classification': {
            'name': 'Supervised - Classification',
            'description': 'Predict categorical labels or classes',
            'use_cases': 'Spam detection, disease diagnosis, customer churn',
            'models': [
                'logistic_regression', 'random_forest_classifier',
                'gradient_boosting_classifier', 'xgboost_classifier', 'svc'
            ]
        },
        'unsupervised_clustering': {
            'name': 'Unsupervised - Clustering',
            'description': 'Group similar data points together',
            'use_cases': 'Customer segmentation, image compression, anomaly detection',
            'models': ['kmeans', 'dbscan', 'hierarchical']
        },
        'unsupervised_dim_reduction': {
            'name': 'Unsupervised - Dimensionality Reduction',
            'description': 'Reduce feature space while preserving information',
            'use_cases': 'Data visualization, feature extraction, noise reduction',
            'models': ['pca']
        },
        'deep_learning': {
            'name': 'Deep Learning',
            'description': 'Neural networks and advanced machine learning models',
            'use_cases': 'Complex pattern recognition, image processing, sequential data',
            'models': ['neural_network']
        }
    }
    
    return jsonify({'categories': categories, 'success': True})

@trainer_bp.route('/models/available', methods=['GET'])
def get_available_models():
    """Get list of available model types with full details"""
    models = {
        # REGRESSION MODELS
        'linear_regression': {
            'name': 'Linear Regression',
            'category': 'supervised_regression',
            'description': 'Simple linear relationship between features and target',
            'speed': 'Very Fast',
            'complexity': 'Low',
            'parameters': {}
        },
        'ridge': {
            'name': 'Ridge Regression',
            'category': 'supervised_regression',
            'description': 'Linear regression with L2 regularization (prevents overfitting)',
            'speed': 'Very Fast',
            'complexity': 'Low',
            'parameters': {
                'alpha': {'type': 'float', 'default': 1.0, 'min': 0.01, 'max': 100, 'description': 'Regularization strength'}
            }
        },
        'lasso': {
            'name': 'Lasso Regression',
            'category': 'supervised_regression',
            'description': 'Linear regression with L1 regularization (feature selection)',
            'speed': 'Fast',
            'complexity': 'Low',
            'parameters': {
                'alpha': {'type': 'float', 'default': 1.0, 'min': 0.01, 'max': 100, 'description': 'Regularization strength'}
            }
        },
        'polynomial': {
            'name': 'Polynomial Regression',
            'category': 'supervised_regression',
            'description': 'Non-linear relationships using polynomial features',
            'speed': 'Fast',
            'complexity': 'Medium',
            'parameters': {
                'degree': {'type': 'int', 'default': 2, 'min': 2, 'max': 5, 'description': 'Polynomial degree (higher = more complex)'}
            }
        },
        'random_forest': {
            'name': 'Random Forest Regressor',
            'category': 'supervised_regression',
            'description': 'Ensemble of decision trees (handles non-linear patterns well)',
            'speed': 'Medium',
            'complexity': 'High',
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'description': 'Number of trees'},
                'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 50, 'description': 'Maximum tree depth (None = unlimited)'}
            }
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting Regressor',
            'category': 'supervised_regression',
            'description': 'Sequential tree building (often highest accuracy)',
            'speed': 'Slow',
            'complexity': 'High',
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'description': 'Number of boosting stages'},
                'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'description': 'Learning rate'},
                'max_depth': {'type': 'int', 'default': 3, 'min': 1, 'max': 10, 'description': 'Maximum tree depth'}
            }
        },
        'xgboost': {
            'name': 'XGBoost Regressor',
            'category': 'supervised_regression',
            'description': 'Optimized gradient boosting (fastest boosting algorithm)',
            'speed': 'Medium',
            'complexity': 'High',
            'available': XGBOOST_AVAILABLE,
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'description': 'Number of boosting rounds'},
                'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'description': 'Learning rate'},
                'max_depth': {'type': 'int', 'default': 6, 'min': 1, 'max': 15, 'description': 'Maximum tree depth'}
            }
        },
        'svr': {
            'name': 'Support Vector Regression',
            'category': 'supervised_regression',
            'description': 'Finds optimal hyperplane for regression',
            'speed': 'Slow',
            'complexity': 'Medium',
            'parameters': {
                'kernel': {'type': 'select', 'default': 'rbf', 'options': ['linear', 'poly', 'rbf', 'sigmoid'], 'description': 'Kernel type'},
                'C': {'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 100, 'description': 'Regularization parameter'}
            }
        },
        
        # CLASSIFICATION MODELS
        'logistic_regression': {
            'name': 'Logistic Regression',
            'category': 'supervised_classification',
            'description': 'Binary and multi-class classification with linear decision boundary',
            'speed': 'Very Fast',
            'complexity': 'Low',
            'parameters': {
                'C': {'type': 'float', 'default': 1.0, 'min': 0.01, 'max': 100, 'description': 'Inverse regularization strength'},
                'max_iter': {'type': 'int', 'default': 1000, 'min': 100, 'max': 5000, 'description': 'Maximum iterations'}
            }
        },
        'random_forest_classifier': {
            'name': 'Random Forest Classifier',
            'category': 'supervised_classification',
            'description': 'Ensemble of decision trees for classification',
            'speed': 'Medium',
            'complexity': 'High',
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'description': 'Number of trees'},
                'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 50, 'description': 'Maximum tree depth'}
            }
        },
        'gradient_boosting_classifier': {
            'name': 'Gradient Boosting Classifier',
            'category': 'supervised_classification',
            'description': 'Sequential tree building for classification',
            'speed': 'Slow',
            'complexity': 'High',
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'description': 'Number of boosting stages'},
                'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'description': 'Learning rate'},
                'max_depth': {'type': 'int', 'default': 3, 'min': 1, 'max': 10, 'description': 'Maximum tree depth'}
            }
        },
        'xgboost_classifier': {
            'name': 'XGBoost Classifier',
            'category': 'supervised_classification',
            'description': 'Optimized gradient boosting for classification',
            'speed': 'Medium',
            'complexity': 'High',
            'available': XGBOOST_AVAILABLE,
            'parameters': {
                'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500, 'description': 'Number of boosting rounds'},
                'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0, 'description': 'Learning rate'},
                'max_depth': {'type': 'int', 'default': 6, 'min': 1, 'max': 15, 'description': 'Maximum tree depth'}
            }
        },
        'svc': {
            'name': 'Support Vector Classifier',
            'category': 'supervised_classification',
            'description': 'Finds optimal hyperplane for classification',
            'speed': 'Slow',
            'complexity': 'Medium',
            'parameters': {
                'kernel': {'type': 'select', 'default': 'rbf', 'options': ['linear', 'poly', 'rbf', 'sigmoid'], 'description': 'Kernel type'},
                'C': {'type': 'float', 'default': 1.0, 'min': 0.1, 'max': 100, 'description': 'Regularization parameter'}
            }
        },
        
        # CLUSTERING MODELS
        'kmeans': {
            'name': 'K-Means Clustering',
            'category': 'unsupervised_clustering',
            'description': 'Partition data into K distinct clusters',
            'speed': 'Fast',
            'complexity': 'Low',
            'parameters': {
                'n_clusters': {'type': 'int', 'default': 3, 'min': 2, 'max': 20, 'description': 'Number of clusters'},
                'max_iter': {'type': 'int', 'default': 300, 'min': 100, 'max': 1000, 'description': 'Maximum iterations'}
            }
        },
        'dbscan': {
            'name': 'DBSCAN',
            'category': 'unsupervised_clustering',
            'description': 'Density-based clustering (finds arbitrary shaped clusters)',
            'speed': 'Medium',
            'complexity': 'Medium',
            'parameters': {
                'eps': {'type': 'float', 'default': 0.5, 'min': 0.1, 'max': 10.0, 'description': 'Maximum distance between samples'},
                'min_samples': {'type': 'int', 'default': 5, 'min': 2, 'max': 20, 'description': 'Minimum samples in neighborhood'}
            }
        },
        'hierarchical': {
            'name': 'Hierarchical Clustering',
            'category': 'unsupervised_clustering',
            'description': 'Build a hierarchy of clusters',
            'speed': 'Slow',
            'complexity': 'Medium',
            'parameters': {
                'n_clusters': {'type': 'int', 'default': 3, 'min': 2, 'max': 20, 'description': 'Number of clusters'},
                'linkage': {'type': 'select', 'default': 'ward', 'options': ['ward', 'complete', 'average', 'single'], 'description': 'Linkage criterion'}
            }
        },
        
        # DIMENSIONALITY REDUCTION
        'pca': {
            'name': 'PCA (Principal Component Analysis)',
            'category': 'unsupervised_dim_reduction',
            'description': 'Reduce dimensions while preserving variance',
            'speed': 'Fast',
            'complexity': 'Low',
            'parameters': {
                'n_components': {'type': 'int', 'default': 2, 'min': 1, 'max': 10, 'description': 'Number of components to keep'}
            }
        },

        # Deep Learning Models
        'neural_network': {
            'name': 'Neural Network (PyTorch)',
            'category': 'supervised_regression',
            'description': 'Deep learning neural network for regression tasks',
            'speed': 'Medium' if GPU_AVAILABLE else 'Slow',
            'complexity': 'High',
            'available': PYTORCH_AVAILABLE,
            'gpu_accelerated': GPU_AVAILABLE,
            'parameters': {
                'hidden_size': {'type': 'int', 'default': 64, 'min': 16, 'max': 512, 'description': 'Number of neurons in hidden layers'},
                'epochs': {'type': 'int', 'default': 100, 'min': 10, 'max': 1000, 'description': 'Number of training epochs'},
                'learning_rate': {'type': 'float', 'default': 0.001, 'min': 0.0001, 'max': 0.1, 'description': 'Learning rate for optimization'},
                'batch_size': {'type': 'int', 'default': 32, 'min': 8, 'max': 256, 'description': 'Batch size for training'}
            }
        }
    }
    
    return jsonify({'models': models, 'success': True})

def generate_confusion_matrix_chart(conf_matrix, class_labels):
    """Generate confusion matrix heatmap"""
    try:
        plt.figure(figsize=(8, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        # Create heatmap
        import seaborn as sns
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=class_labels, yticklabels=class_labels,
                   cbar_kw={'label': 'Count'})
        
        plt.xlabel('Predicted Label', color='#00ff00', fontsize=12)
        plt.ylabel('True Label', color='#00ff00', fontsize=12)
        plt.title('Confusion Matrix', color='#00ff00', fontsize=14, pad=20)
        
        # Style
        plt.xticks(color='#00ff00')
        plt.yticks(color='#00ff00')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#000000', edgecolor='none', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        print(f"Error generating confusion matrix chart: {e}")
        return None

def generate_roc_curve_chart(fpr, tpr, roc_auc):
    """Generate ROC curve"""
    try:
        plt.figure(figsize=(8, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='#00ff00', linewidth=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='#666666', linestyle='--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', color='#00ff00', fontsize=12)
        plt.ylabel('True Positive Rate', color='#00ff00', fontsize=12)
        plt.title('ROC Curve', color='#00ff00', fontsize=14, pad=20)
        plt.legend(loc='lower right', fontsize=10, facecolor='#000000', edgecolor='#00ff00', labelcolor='#00ff00')
        
        # Style
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#00ff00')
        plt.grid(True, alpha=0.2, color='#00ff00')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#000000', edgecolor='none', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        print(f"Error generating ROC curve: {e}")
        return None

def generate_cluster_scatter_chart(X_subset, labels):
    """Generate cluster visualization scatter plot"""
    try:
        plt.figure(figsize=(10, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        # Scatter plot with different colors for each cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(X_subset.iloc[mask, 0], X_subset.iloc[mask, 1], 
                       c=[color], label=f'Cluster {label}', alpha=0.6, s=50)
        
        plt.xlabel(X_subset.columns[0], color='#00ff00', fontsize=12)
        plt.ylabel(X_subset.columns[1], color='#00ff00', fontsize=12)
        plt.title('Cluster Visualization', color='#00ff00', fontsize=14, pad=20)
        plt.legend(loc='best', fontsize=10, facecolor='#000000', edgecolor='#00ff00', labelcolor='#00ff00')
        
        # Style
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#00ff00')
        plt.grid(True, alpha=0.2, color='#00ff00')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#000000', edgecolor='none', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        print(f"Error generating cluster scatter chart: {e}")
        return None

def generate_scree_plot(explained_variance_ratio):
    """Generate scree plot for PCA"""
    try:
        plt.figure(figsize=(10, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        components = list(range(1, len(explained_variance_ratio) + 1))
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Plot explained variance
        plt.bar(components, explained_variance_ratio, color='#00ff00', alpha=0.6, label='Individual')
        plt.plot(components, cumulative_variance, color='#ff00ff', marker='o', linewidth=2, label='Cumulative')
        
        plt.xlabel('Principal Component', color='#00ff00', fontsize=12)
        plt.ylabel('Explained Variance Ratio', color='#00ff00', fontsize=12)
        plt.title('Scree Plot', color='#00ff00', fontsize=14, pad=20)
        plt.legend(loc='best', fontsize=10, facecolor='#000000', edgecolor='#00ff00', labelcolor='#00ff00')
        
        # Style
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#00ff00')
        plt.grid(True, alpha=0.2, color='#00ff00')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#000000', edgecolor='none', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        print(f"Error generating scree plot: {e}")
        return None

def generate_transformed_scatter(X_transformed):
    """Generate scatter plot of transformed data (e.g., PCA)"""
    try:
        plt.figure(figsize=(10, 6), facecolor='#000000')
        ax = plt.gca()
        ax.set_facecolor('#000000')
        
        # Scatter plot
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                   c='#00ff00', alpha=0.6, s=50)
        
        plt.xlabel('Component 1', color='#00ff00', fontsize=12)
        plt.ylabel('Component 2', color='#00ff00', fontsize=12)
        plt.title('Transformed Data Visualization', color='#00ff00', fontsize=14, pad=20)
        
        # Style
        ax.spines['bottom'].set_color('#00ff00')
        ax.spines['left'].set_color('#00ff00')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#00ff00')
        plt.grid(True, alpha=0.2, color='#00ff00')
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='#000000', edgecolor='none', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return f'data:image/png;base64,{image_base64}'
    except Exception as e:
        print(f"Error generating transformed scatter plot: {e}")
        return None

