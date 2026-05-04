"""
Combined analysis endpoint for Riff iOS app.

Runs Spleeter stem separation + chord recognition and beat detection
in parallel on a single uploaded file.
"""

import os
import time
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Blueprint, request, jsonify, current_app
from extensions import limiter
from config import get_config
from utils.logging import log_info, log_error, log_debug

# Create blueprint
analyze_bp = Blueprint('analyze', __name__)

# Get configuration for rate limits
config = get_config()


@analyze_bp.route('/api/analyze', methods=['POST'])
@limiter.limit(config.get_rate_limit('heavy_processing'))
def analyze():
    """
    Combined analysis: Spleeter separation + chord recognition + beat detection.

    Chord recognition (Spleeter + model) and beat detection run in parallel
    since they're independent — chords use the accompaniment stem, beats use
    the full mix.

    Parameters:
    - file: Audio file (multipart/form-data)
    - model: Chord model (default 'chord-cnn-lstm')
    - detector: Beat detector (default 'madmom')
    - use_spleeter: 'true'/'false' (default 'true')
    - chord_dict: Chord dictionary (optional)

    Returns:
    - JSON with chords, beats, bpm, duration, and metadata
    """
    temp_file_path = None
    start_time = time.time()

    try:
        # Validate file upload
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No audio file provided"}), 400

        # Parse parameters
        model = request.form.get('model', 'chord-cnn-lstm').lower()
        detector = request.form.get('detector', 'madmom').lower()
        use_spleeter_param = request.form.get('use_spleeter', 'true').lower()
        use_spleeter = use_spleeter_param == 'true'
        chord_dict = request.form.get('chord_dict', None)

        # Save uploaded file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        file.save(temp_file.name)
        temp_file_path = temp_file.name

        log_info(f"Combined analysis request: model={model}, detector={detector}, "
                 f"use_spleeter={use_spleeter}, chord_dict={chord_dict}")

        # Get services
        chord_service = current_app.extensions['services']['chord_recognition']
        beat_service = current_app.extensions['services']['beat_detection']

        if not chord_service:
            return jsonify({"error": "Chord recognition service unavailable"}), 503
        if not beat_service:
            return jsonify({"error": "Beat detection service unavailable"}), 503

        # Run chord recognition and beat detection IN PARALLEL.
        # They're independent: chords use Spleeter accompaniment stem,
        # beats use the full mix (needs drums for rhythm tracking).
        chord_result = None
        beat_result = None
        chord_error = None
        beat_error = None

        def run_chords():
            return chord_service.recognize_chords(
                file_path=temp_file_path,
                detector=model,
                chord_dict=chord_dict,
                force=False,
                use_spleeter=use_spleeter
            )

        def run_beats():
            return beat_service.detect_beats(
                file_path=temp_file_path,
                detector=detector,
                force=False
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            chord_future = executor.submit(run_chords)
            beat_future = executor.submit(run_beats)

            try:
                chord_result = chord_future.result()
            except Exception as e:
                chord_error = str(e)
                log_error(f"Chord recognition thread failed: {e}")

            try:
                beat_result = beat_future.result()
            except Exception as e:
                beat_error = str(e)
                log_error(f"Beat detection thread failed: {e}")

        if chord_error or not chord_result or not chord_result.get('success'):
            error = chord_error or (chord_result.get('error') if chord_result else 'Unknown')
            return jsonify({"success": False, "error": f"Chord recognition failed: {error}"}), 500

        if beat_error or not beat_result or not beat_result.get('success'):
            error = beat_error or (beat_result.get('error') if beat_result else 'Unknown')
            return jsonify({"success": False, "error": f"Beat detection failed: {error}"}), 500

        processing_time = time.time() - start_time

        response = {
            "success": True,
            "chords": chord_result.get("chords", []),
            "beats": beat_result.get("beats", []),
            "bpm": beat_result.get("bpm", 0.0),
            "duration": chord_result.get("duration", beat_result.get("duration", 0.0)),
            "total_chords": chord_result.get("total_chords", 0),
            "model_used": chord_result.get("model_used", model),
            "chord_dict": chord_result.get("chord_dict", "submission"),
            "used_spleeter": chord_result.get("spleeter", {}).get("used", False) if isinstance(chord_result.get("spleeter"), dict) else use_spleeter,
            "processing_time": round(processing_time, 1)
        }

        log_info(f"Combined analysis complete: {response['total_chords']} chords, "
                 f"{len(response['beats'])} beats, BPM {response['bpm']}, "
                 f"{response['processing_time']}s")

        return jsonify(response)

    except Exception as e:
        error_msg = f"Combined analysis error: {str(e)}"
        log_error(error_msg)
        log_error(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": error_msg,
            "traceback": traceback.format_exc() if not config.PRODUCTION_MODE else None
        }), 500
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                log_debug(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                log_error(f"Failed to clean up temporary file {temp_file_path}: {cleanup_error}")
