"""
Combined analysis endpoint for Riff iOS app.

Runs chord recognition + beat detection in parallel, then lyrics transcription
sequentially. Running all three ML models simultaneously exceeds Railway's
memory limit, so lyrics runs after chords+beats complete.
"""

import os
import time
import tempfile
import traceback
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
    Combined analysis: chord recognition + beat detection + lyrics transcription.

    Phase 1: Chords + beats run in parallel (both are lightweight).
    Phase 2: Lyrics runs after phase 1 completes (avoids OOM from 3 concurrent models).

    Parameters:
    - file: Audio file (multipart/form-data)
    - model: Chord model (default 'chord-cnn-lstm')
    - detector: Beat detector (default 'madmom')
    - use_spleeter: 'true'/'false' (default 'false' — Demucs too heavy for Railway CPU)
    - chord_dict: Chord dictionary (optional)

    Returns:
    - JSON with chords, beats, bpm, duration, lyrics, and metadata
    """
    temp_file_path = None
    spleeter_result = None
    start_time = time.time()

    try:
        # Validate file upload
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No audio file provided"}), 400

        # Parse parameters
        model = request.form.get('model', 'chord-cnn-lstm').lower()
        detector = request.form.get('detector', 'madmom').lower()
        use_spleeter_param = request.form.get('use_spleeter', 'false').lower()
        use_spleeter = use_spleeter_param == 'true'
        chord_dict = request.form.get('chord_dict', None)

        # Save uploaded file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.m4a')
        file.save(temp_file.name)
        temp_file_path = temp_file.name

        log_info(f"Combined analysis request: model={model}, detector={detector}, "
                 f"use_spleeter={use_spleeter}, chord_dict={chord_dict}")

        # Get services
        chord_service = current_app.extensions['services']['chord_recognition']
        beat_service = current_app.extensions['services']['beat_detection']
        lyrics_service = current_app.extensions['services'].get('lyrics_transcription')

        if not chord_service:
            return jsonify({"error": "Chord recognition service unavailable"}), 503
        if not beat_service:
            return jsonify({"error": "Beat detection service unavailable"}), 503

        # --- Optional: Demucs stem separation ---
        # Disabled by default — htdemucs needs ~6GB RAM on CPU which causes OOM on Railway.
        # When enabled (use_spleeter=true), chords run on accompaniment, lyrics on vocals.
        accompaniment_path = None
        vocals_path = None
        spleeter_info = {"used": False}

        if use_spleeter and chord_service.spleeter_service.is_available():
            log_info("Running Demucs htdemucs stem separation")
            spleeter_start = time.time()
            spleeter_result = chord_service.spleeter_service.extract_vocals(temp_file_path)

            if spleeter_result.get("success"):
                accompaniment_path = spleeter_result.get("accompaniment_path")
                vocals_path = spleeter_result.get("vocals_path")
                spleeter_time = time.time() - spleeter_start
                spleeter_info = {
                    "used": True,
                    "model": spleeter_result.get("model_used", "htdemucs"),
                    "processing_time": round(spleeter_time, 1),
                }
                log_info(f"Demucs complete in {spleeter_time:.1f}s")
            else:
                log_error(f"Demucs failed: {spleeter_result.get('error')}. "
                          f"Proceeding without separation.")
                spleeter_info = {"used": False, "error": spleeter_result.get("error")}

        chord_audio = accompaniment_path or temp_file_path
        lyrics_audio = vocals_path or temp_file_path

        # --- Run chords, beats, lyrics SEQUENTIALLY ---
        # Railway's memory is too constrained for parallel ML model inference.
        # Sequential execution ensures only one model is under heavy load at a time.

        # Step 1: Beat detection
        log_info("Step 1/3: Beat detection")
        beat_result = None
        try:
            beat_result = beat_service.detect_beats(
                file_path=temp_file_path,
                detector=detector,
                force=False,
            )
        except Exception as e:
            log_error(f"Beat detection failed: {e}")

        if not beat_result or not beat_result.get('success'):
            error = beat_result.get('error') if beat_result else 'Unknown'
            return jsonify({"success": False, "error": f"Beat detection failed: {error}"}), 500

        # Step 2: Chord recognition
        log_info("Step 2/3: Chord recognition")
        chord_result = None
        try:
            chord_result = chord_service.recognize_chords(
                file_path=chord_audio,
                detector=model,
                chord_dict=chord_dict,
                force=False,
                use_spleeter=False,
            )
        except Exception as e:
            log_error(f"Chord recognition failed: {e}")

        if not chord_result or not chord_result.get('success'):
            error = chord_result.get('error') if chord_result else 'Unknown'
            return jsonify({"success": False, "error": f"Chord recognition failed: {error}"}), 500

        # Step 3: Lyrics transcription
        log_info("Step 3/3: Lyrics transcription")
        lyrics_words = []
        if lyrics_service and lyrics_service.is_available():
            try:
                lyrics_result = lyrics_service.transcribe(audio_path=lyrics_audio)
                if lyrics_result and lyrics_result.get("success"):
                    lyrics_words = lyrics_result.get("lyrics", [])
                    log_info(f"Lyrics transcription: {len(lyrics_words)} words")
                else:
                    lyrics_err = lyrics_result.get("error") if lyrics_result else "Unknown"
                    log_info(f"Lyrics transcription failed: {lyrics_err}")
            except Exception as e:
                log_error(f"Lyrics transcription error: {e}")
        else:
            log_info("Lyrics transcription service not available, skipping")

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
            "used_spleeter": spleeter_info.get("used", False),
            "lyrics": lyrics_words,
            "total_words": len(lyrics_words),
            "processing_time": round(processing_time, 1),
        }

        log_info(f"Combined analysis complete: {response['total_chords']} chords, "
                 f"{len(response['beats'])} beats, BPM {response['bpm']}, "
                 f"{response['total_words']} lyrics words, "
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
        # Clean up temp files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                log_debug(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                log_error(f"Failed to clean up temp file {temp_file_path}: {e}")

        # Clean up Demucs stems
        if spleeter_result and spleeter_result.get("success"):
            try:
                chord_service = current_app.extensions['services']['chord_recognition']
                chord_service.spleeter_service.cleanup_stems(spleeter_result)
                log_debug("Cleaned up Demucs stems")
            except Exception as e:
                log_error(f"Failed to clean up Demucs stems: {e}")
