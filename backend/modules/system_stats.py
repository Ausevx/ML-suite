"""
System Stats Module - Provides system resource usage information
"""
from flask import Blueprint, jsonify
import psutil
import platform

stats_bp = Blueprint('system_stats', __name__)

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
            'system': system_info,
            'status': 'online'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

