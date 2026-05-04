"""
Combined analysis endpoint for Riff iOS app.

Runs Spleeter stem separation first, then chord recognition, beat detection,
and lyrics transcription all in parallel on the appropriate stems:
  - Accompaniment stem → chord recognition (no vocal interference)
  - Vocals stem → lyrics transcription (clean vocals, no music)
  - Full mix → beat detection (needs drums for rhythm tracking)
"""

import os
import time
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
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
    Combined analysis: Spleeter separation + chord recognition + beat detection + lyrics.

    Spleeter runs first to separate vocals and accompaniment, then three tasks
    run in parallel on the appropriate audio:
    - Chords: accompaniment stem (no vocals/drums polluting chroma)
    - Beats: full mix (drums needed for rhythm tracking)
    - Lyrics: vocals stem (clean vocals for accurate transcription)

    Parameters:
    - file: Audio file (multipart/form-data)
    - model: Chord model (default 'chord-cnn-lstm')
    - detector: Beat detector (default 'madmom')
    - use_spleeter: 'true'/'false' (default 'true')
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
        use_spleeter_param = request.form.get('use_spleeter', 'true').lower()
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

        # --- Step 1: Run Demucs stem separation ---
        # Produces vocals + accompaniment (drums+bass+other combined).
        # Do this first so we can distribute stems to parallel tasks.
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
                log_info(f"Demucs complete in {spleeter_time:.1f}s — "
                         f"accompaniment: {accompaniment_path}, vocals: {vocals_path}")
            else:
                log_error(f"Demucs failed: {spleeter_result.get('error')}. "
                          f"Proceeding without separation.")
                spleeter_info = {"used": False, "error": spleeter_result.get("error")}

        # --- Step 2: Run chords, beats, and lyrics IN PARALLEL ---
        # Each uses the optimal audio source.
        chord_audio = accompaniment_path or temp_file_path
        # Prefer vocals stem for lyrics (clean vocals); fall back to full mix
        lyrics_audio = vocals_path or temp_file_path

        chord_result = None
        beat_result = None
        lyrics_result = None
        chord_error = None
        beat_error = None
        lyrics_error = None

        def run_chords():
            # Pass use_spleeter=False since we already separated above
            return chord_service.recognize_chords(
                file_path=chord_audio,
                detector=model,
                chord_dict=chord_dict,
                force=False,
                use_spleeter=False,
            )

        def run_beats():
            return beat_service.detect_beats(
                file_path=temp_file_path,
                detector=detector,
                force=False,
            )

        def run_lyrics():
            if not lyrics_service or not lyrics_service.is_available():
                return {"success": False, "error": "Lyrics service unavailable", "lyrics": []}
            return lyrics_service.transcribe(audio_path=lyrics_audio)

        with ThreadPoolExecutor(max_workers=3) as executor:
            chord_future = executor.submit(run_chords)
            beat_future = executor.submit(run_beats)
            lyrics_future = executor.submit(run_lyrics)

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

            try:
                lyrics_result = lyrics_future.result()
            except Exception as e:
                lyrics_error = str(e)
                log_error(f"Lyrics transcription thread failed: {e}")

        # --- Validate required results (chords + beats) ---
        if chord_error or not chord_result or not chord_result.get('success'):
            error = chord_error or (chord_result.get('error') if chord_result else 'Unknown')
            return jsonify({"success": False, "error": f"Chord recognition failed: {error}"}), 500

        if beat_error or not beat_result or not beat_result.get('success'):
            error = beat_error or (beat_result.get('error') if beat_result else 'Unknown')
            return jsonify({"success": False, "error": f"Beat detection failed: {error}"}), 500

        # Lyrics are optional — don't fail the whole request if they fail
        lyrics_words = []
        if lyrics_result and lyrics_result.get("success"):
            lyrics_words = lyrics_result.get("lyrics", [])
            log_info(f"Lyrics transcription: {len(lyrics_words)} words")
        else:
            lyrics_err = lyrics_error or (lyrics_result.get("error") if lyrics_result else "N/A")
            log_info(f"Lyrics transcription skipped/failed: {lyrics_err}")

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
