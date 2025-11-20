"""
System Stats Module - Provides system resource usage information
"""
from flask import Blueprint, jsonify, request
import psutil
import platform

# GPU detection imports (optional)
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

stats_bp = Blueprint('system_stats', __name__)

def get_gpu_info():
    """Get GPU information and availability"""
    gpu_info = {
        'available': False,
        'pytorch_cuda': False,
        'pytorch_mps': False,
        'xgboost_gpu': False,
        'gpu_backend': 'cpu',
        'devices': [],
        'utilization': 0,
        'memory_used_gb': 0,
        'memory_total_gb': 0
    }

    # Check PyTorch GPU availability (CUDA and MPS)
    if PYTORCH_AVAILABLE:
        try:
            # Check CUDA
            cuda_available = torch.cuda.is_available()
            gpu_info['pytorch_cuda'] = cuda_available

            # Check MPS (Apple Silicon)
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            gpu_info['pytorch_mps'] = mps_available

            # Set overall availability and backend
            gpu_info['available'] = cuda_available or mps_available
            if cuda_available:
                gpu_info['gpu_backend'] = 'cuda'
                gpu_info['device_count'] = torch.cuda.device_count()
                gpu_info['devices'] = []
                
                # Get CUDA utilization using nvidia-ml-py3 if available
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_info['utilization'] = utilization.gpu
                    
                    # Get memory info
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info['memory_used_gb'] = round(mem_info.used / (1024**3), 2)
                    gpu_info['memory_total_gb'] = round(mem_info.total / (1024**3), 2)
                    pynvml.nvmlShutdown()
                except:
                    # Fallback to memory allocated by PyTorch
                    try:
                        gpu_info['memory_used_gb'] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
                        gpu_info['memory_total_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                        if gpu_info['memory_total_gb'] > 0:
                            gpu_info['utilization'] = round((gpu_info['memory_used_gb'] / gpu_info['memory_total_gb']) * 100, 1)
                    except:
                        pass
                
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    gpu_info['devices'].append({
                        'id': i,
                        'name': device_props.name,
                        'total_memory_gb': round(device_props.total_memory / (1024**3), 2),
                        'major': device_props.major,
                        'minor': device_props.minor,
                        'type': 'cuda'
                    })
            elif mps_available:
                gpu_info['gpu_backend'] = 'mps'
                # MPS doesn't provide detailed device info like CUDA
                gpu_info['devices'] = [{'id': 0, 'name': 'Apple Silicon GPU', 'type': 'mps'}]
                gpu_info['device_count'] = 1
                
                # Try to get Apple Silicon GPU activity using powermetrics (requires elevated privileges)
                # Since we can't easily access this, we'll estimate based on memory allocation
                try:
                    # Check if there's any PyTorch memory allocated on MPS
                    if hasattr(torch.mps, 'current_allocated_memory'):
                        allocated = torch.mps.current_allocated_memory() / (1024**3)
                        if allocated > 0:
                            gpu_info['utilization'] = min(int(allocated * 20), 100)  # Rough estimate
                            gpu_info['memory_used_gb'] = round(allocated, 2)
                        else:
                            gpu_info['utilization'] = 0
                    else:
                        # MPS is available but not actively in use
                        gpu_info['utilization'] = 0
                except:
                    # Fallback: show 0 utilization for MPS (can't measure accurately)
                    gpu_info['utilization'] = 0
                    pass

        except Exception as e:
            print(f"Error checking PyTorch GPU: {e}")

    # Check XGBoost GPU support (only CUDA, not MPS)
    if XGBOOST_AVAILABLE:
        try:
            # XGBoost GPU only works with CUDA
            if gpu_info['pytorch_cuda']:
                test_model = xgb.XGBRegressor(tree_method='gpu_hist')
                gpu_info['xgboost_gpu'] = True
            else:
                gpu_info['xgboost_gpu'] = False
        except Exception as e:
            print(f"XGBoost GPU not available: {e}")
            gpu_info['xgboost_gpu'] = False

    return gpu_info

@stats_bp.route('/stats', methods=['GET'])
def get_system_stats():
    """Get current system resource usage"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)

        # Memory usage
        memory = psutil.virtual_memory()
        ram_percent = memory.percent
        ram_used_gb = memory.used / (1024 ** 3)
        ram_total_gb = memory.total / (1024 ** 3)

        # Disk usage (current directory)
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent

        # System info
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version()
        }

        # GPU info
        gpu_info = get_gpu_info()

        return jsonify({
            'cpu': {
                'percent': cpu_percent,
                'count': psutil.cpu_count()
            },
            'ram': {
                'percent': ram_percent,
                'used_gb': round(ram_used_gb, 2),
                'total_gb': round(ram_total_gb, 2)
            },
            'disk': {
                'percent': disk_percent
            },
            'gpu': gpu_info,
            'system': system_info,
            'status': 'online'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@stats_bp.route('/gpu/settings', methods=['GET'])
def get_gpu_settings():
    """Get current GPU configuration settings"""
    try:
        import config
        settings = {
            'gpu_enabled': getattr(config, 'GPU_ENABLED', True),
            'gpu_force_cpu': getattr(config, 'GPU_FORCE_CPU', False),
            'gpu_preferred_backend': getattr(config, 'GPU_PREFERRED_BACKEND', 'auto')
        }
        return jsonify({'settings': settings, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

@stats_bp.route('/gpu/settings', methods=['POST'])
def update_gpu_settings():
    """Update GPU configuration settings"""
    try:
        import config
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({'error': 'No data provided', 'success': False}), 400

        # Update settings if provided
        if 'gpu_enabled' in data:
            config.GPU_ENABLED = bool(data['gpu_enabled'])
        if 'gpu_force_cpu' in data:
            config.GPU_FORCE_CPU = bool(data['gpu_force_cpu'])
        if 'gpu_preferred_backend' in data:
            valid_backends = ['auto', 'cuda', 'mps', 'cpu']
            if data['gpu_preferred_backend'] in valid_backends:
                config.GPU_PREFERRED_BACKEND = data['gpu_preferred_backend']
            else:
                return jsonify({'error': f'Invalid backend. Must be one of: {valid_backends}', 'success': False}), 400

        # Return updated settings
        settings = {
            'gpu_enabled': config.GPU_ENABLED,
            'gpu_force_cpu': config.GPU_FORCE_CPU,
            'gpu_preferred_backend': config.GPU_PREFERRED_BACKEND
        }

        return jsonify({
            'settings': settings,
            'message': 'GPU settings updated. Restart may be required for changes to take effect.',
            'success': True
        })

    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

