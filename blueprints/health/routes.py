"""
Health check routes for ChordMini Flask application.

This module provides endpoints for health monitoring and basic API status.
"""

from flask import Blueprint, jsonify
from extensions import limiter
from config import get_config

# Create blueprint
health_bp = Blueprint('health', __name__)

# Get configuration for rate limits
config = get_config()


@health_bp.route('/')
@limiter.limit(config.get_rate_limit('health'))
def index():
    """Root endpoint - basic health check."""
    return jsonify({
        "status": "healthy",
        "message": "Audio analysis API is running"
    })


@health_bp.route('/health')
def health():
    """Simple health check endpoint for Cloud Run and load balancers."""
    return jsonify({"status": "healthy"}), 200


@health_bp.route('/api/service-status')
@limiter.limit(config.get_rate_limit('light_processing'))
def service_status():
    """Check availability of all services with error details."""
    status = {}

    # Check demucs (stem separation)
    try:
        import torch
        import torchaudio
        from demucs.pretrained import get_model
        status["demucs"] = {"available": True, "torch": torch.__version__, "torchaudio": torchaudio.__version__}
    except Exception as e:
        status["demucs"] = {"available": False, "error": str(e)}

    # Check faster-whisper (lyrics)
    try:
        from faster_whisper import WhisperModel
        status["faster_whisper"] = {"available": True}
    except Exception as e:
        status["faster_whisper"] = {"available": False, "error": str(e)}

    # Check services from app extensions
    from flask import current_app
    services = current_app.extensions.get('services', {})
    for name, svc in services.items():
        if svc is None:
            status[name] = {"available": False, "error": "Service is None"}
        elif hasattr(svc, 'is_available'):
            status[name] = {"available": svc.is_available()}
        elif hasattr(svc, 'spleeter_service'):
            status[name] = {
                "available": True,
                "stem_separation": svc.spleeter_service.is_available(),
            }

    return jsonify(status)