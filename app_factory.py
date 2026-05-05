"""
Flask application factory for ChordMini.

This module implements the application factory pattern, creating and configuring
Flask applications with proper separation of concerns.
"""

from flask import Flask
from typing import Optional

# Import compatibility patches first
import compat

# Import configuration
from config import get_config

# Import extensions
from extensions import init_extensions

# Import error handlers
from error_handlers import register_error_handlers, register_custom_error_handlers

# Import utilities
from utils.logging import log_info, log_debug, is_debug_enabled


def create_app(config_name: Optional[str] = None) -> Flask:
    """
    Create and configure Flask application using the application factory pattern.

    Args:
        config_name: Configuration name ('development', 'production', 'testing')
                    If None, auto-detect from environment

    Returns:
        Configured Flask application instance
    """
    # Apply compatibility patches before any heavy imports
    compat.apply_all()

    # Create Flask application
    app = Flask(__name__, template_folder='templates')

    # Load configuration
    config = get_config(config_name)
    app.config.from_object(config)

    log_info(f"Creating Flask app with config: {config.__class__.__name__}")

    # Initialize extensions
    init_extensions(app, config)

    # Register error handlers
    register_error_handlers(app)
    register_custom_error_handlers(app)

    # Register blueprints
    register_blueprints(app, config)

    # Initialize service container
    init_services(app, config)

    log_info("Flask application created successfully")

    return app


def register_blueprints(app: Flask, config) -> None:
    """
    Register all blueprints with the Flask application.

    Args:
        app: Flask application instance
        config: Configuration object
    """
    # Import blueprints
    from blueprints.health import health_bp
    from blueprints.docs import docs_bp
    from blueprints.beats import beats_bp
    from blueprints.chords import chords_bp
    from blueprints.lyrics import lyrics_bp
    from blueprints.songformer import songformer_bp
    from blueprints.analyze import analyze_bp
    from blueprints.youtube import youtube_bp
    from blueprints.jobs import jobs_bp
    from blueprints.debug import debug_bp

    # Register blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(docs_bp)
    app.register_blueprint(beats_bp)
    app.register_blueprint(chords_bp)
    app.register_blueprint(lyrics_bp)
    app.register_blueprint(songformer_bp)
    app.register_blueprint(analyze_bp)
    app.register_blueprint(youtube_bp)
    app.register_blueprint(jobs_bp)

    # Register debug blueprint only in non-production mode
    if not config.PRODUCTION_MODE:
        app.register_blueprint(debug_bp)
        log_info("Debug blueprint registered (non-production mode)")
    else:
        log_info("Debug blueprint skipped (production mode)")

    log_info("Blueprints registered successfully")


def init_services(app: Flask, config) -> None:
    """
    Initialize service container with dependency injection.

    Args:
        app: Flask application instance
        config: Configuration object
    """
    # Setup model paths for imports
    from utils.paths import setup_model_paths
    setup_model_paths()

    # Create a simple service container
    services = {}

    # Initialize beat detection service
    try:
        from services.audio.beat_detection_service import BeatDetectionService
        services['beat_detection'] = BeatDetectionService()
        log_info("Beat detection service initialized")
    except Exception as e:
        log_info(f"Failed to initialize beat detection service: {e}")
        # Create a dummy service that returns errors
        services['beat_detection'] = None

    # Initialize chord recognition service
    try:
        from services.audio.chord_recognition_service import ChordRecognitionService
        services['chord_recognition'] = ChordRecognitionService()
        log_info("Chord recognition service initialized")
    except Exception as e:
        log_info(f"Failed to initialize chord recognition service: {e}")
        # Create a dummy service that returns errors
        services['chord_recognition'] = None

    # Initialize lyrics service
    try:
        from services.lyrics.orchestrator import LyricsOrchestrator
        services['lyrics'] = LyricsOrchestrator(config)
        log_info("Lyrics service initialized")
    except Exception as e:
        log_info(f"Failed to initialize lyrics service: {e}")
        services['lyrics'] = None

    # Initialize lyrics transcription service (server-side Whisper)
    try:
        from services.audio.lyrics_transcription_service import LyricsTranscriptionService
        lyrics_transcription = LyricsTranscriptionService(model_size="small")
        if lyrics_transcription.is_available():
            services['lyrics_transcription'] = lyrics_transcription
            log_info("Lyrics transcription service initialized (small model)")
        else:
            services['lyrics_transcription'] = None
            log_info("Lyrics transcription service unavailable (faster-whisper not installed)")
    except Exception as e:
        log_info(f"Failed to initialize lyrics transcription service: {e}")
        services['lyrics_transcription'] = None

    # Initialize SongFormer service
    try:
        from services.audio.songformer_service import SongFormerService
        services['songformer'] = SongFormerService()
        log_info("SongFormer service initialized")
    except Exception as e:
        log_info(f"Failed to initialize SongFormer service: {e}")
        services['songformer'] = None

    # Store services in app extensions
    app.extensions['services'] = services

    # Initialize job service (requires Redis)
    redis_url = config.REDIS_URL
    if redis_url:
        try:
            from services.job_service import JobService
            app.extensions['job_service'] = JobService(redis_url)
            log_info("Job service initialized (Redis connected)")
        except Exception as e:
            app.extensions['job_service'] = None
            log_info(f"Job service unavailable (Redis not reachable): {e}")
    else:
        app.extensions['job_service'] = None
        log_info("Job service skipped (REDIS_URL not set)")

    log_info("Service container initialized")