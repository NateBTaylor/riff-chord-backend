"""
Combined analysis endpoint for Riff iOS app.

Pipeline:
    beats (local, ~2s)
       → Spleeter (Replicate, ~3s warm)
       → chord recognition  ┐
       → whisper lyrics     ┘  run in parallel (~6s warm, dominated by chord)

Private Replicate Deployments (RIFF_DEPLOY_* env vars) carry their own
warm pool, so we no longer need the sleep-between-calls workaround that
existed when we hit shared public models.
"""

import gc
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
    Combined analysis: beat detection + chord recognition + lyrics transcription.

    Pipeline (warm path):
    1. Beat detection (local librosa, ~2s)
    2. Spleeter stem separation (Replicate deployment, ~3s)
    3. Chord recognition + lyrics transcription run in parallel (~6s warm)

    Parameters:
    - file: Audio file (multipart/form-data)
    - model: Chord model (default 'auto')
    - detector: Beat detector (default 'librosa')
    - chord_dict: Chord dictionary (optional)

    Returns:
    - JSON with chords, beats, bpm, duration, lyrics, and metadata
    """
    temp_file_path = None
    stems_info = None
    start_time = time.time()

    try:
        # Validate file upload
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No audio file provided"}), 400

        # Parse parameters
        model = request.form.get('model', 'auto').lower()
        detector = request.form.get('detector', 'librosa').lower()
        chord_dict = request.form.get('chord_dict', None)

        # Save uploaded file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.m4a')
        file.save(temp_file.name)
        temp_file_path = temp_file.name

        log_info(f"Combined analysis request: model={model}, detector={detector}, "
                 f"chord_dict={chord_dict}")

        # Get services
        chord_service = current_app.extensions['services']['chord_recognition']
        beat_service = current_app.extensions['services']['beat_detection']
        lyrics_service = current_app.extensions['services'].get('lyrics_transcription')
        spleeter_service = current_app.extensions['services'].get('spleeter')

        if not chord_service:
            return jsonify({"error": "Chord recognition service unavailable"}), 503
        if not beat_service:
            return jsonify({"error": "Beat detection service unavailable"}), 503

        # --- Step 1: Beat detection (local librosa, ~2s) ---
        log_info("Step 1/4: Beat detection")
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
            log_error(f"Beat detection failed: {error} — continuing with chord recognition")
            beat_result = {"success": True, "beats": [], "bpm": 120.0, "duration": 0.0}

        gc.collect()

        # --- Step 2: Spleeter stem separation (Replicate, ~3s) ---
        audio_for_lyrics = temp_file_path
        stems_info = None

        if spleeter_service and spleeter_service.is_available():
            log_info("Step 2/3: Stem separation (vocals for lyrics)")
            try:
                stems_info = spleeter_service.extract_stems(temp_file_path)
                if stems_info.get('success'):
                    audio_for_lyrics = stems_info.get('vocals_path', temp_file_path)
                    log_info(f"Stems separated in {stems_info.get('processing_time', 0):.1f}s")
                else:
                    log_error(f"Stem separation failed, using full mix: {stems_info.get('error')}")
                    stems_info = None
            except Exception as e:
                log_error(f"Stem separation error: {e}")
                stems_info = None
        else:
            log_info("Step 2/3: Skipping stem separation (unavailable)")

        # --- Step 3: Chord recognition || Whisper lyrics (parallel) ---
        # Private deployments don't share the public rate-limit pool, so we
        # can fire both at once. Worst-case wall time becomes max(chord, whisper)
        # instead of sum(chord, whisper) + sleep delays.
        log_info("Step 3/3: Chord recognition + lyrics (parallel)" +
                 (" (lyrics on vocals stem)" if stems_info else " (lyrics on full mix)"))

        def _run_chord():
            try:
                return chord_service.recognize_chords(
                    file_path=temp_file_path,
                    detector=model,
                    chord_dict=chord_dict,
                    force=False,
                    use_spleeter=False,
                )
            except Exception as e:
                log_error(f"Chord recognition failed: {e}")
                return None

        def _run_lyrics():
            if not lyrics_service:
                return None
            try:
                return lyrics_service.transcribe(audio_path=audio_for_lyrics)
            except Exception as e:
                log_error(f"Lyrics transcription failed: {e}")
                return None

        chord_result = None
        lyrics_result = None
        with ThreadPoolExecutor(max_workers=2) as pool:
            chord_future = pool.submit(_run_chord)
            lyrics_future = pool.submit(_run_lyrics)
            chord_result = chord_future.result()
            lyrics_result = lyrics_future.result()

        if not chord_result or not chord_result.get('success'):
            error = chord_result.get('error') if chord_result else 'Unknown'
            return jsonify({"success": False, "error": f"Chord recognition failed: {error}"}), 500

        # Extract lyrics (non-fatal if missing)
        lyrics = []
        total_words = 0
        if lyrics_result and lyrics_result.get('success'):
            lyrics = lyrics_result.get('lyrics', [])
            total_words = lyrics_result.get('total_words', 0)
            log_info(f"Lyrics: {total_words} words in {lyrics_result.get('processing_time', 0)}s")

        gc.collect()

        processing_time = time.time() - start_time

        response = {
            "success": True,
            "chords": chord_result.get("chords", []),
            "beats": beat_result.get("beats", []),
            "bpm": beat_result.get("bpm", 0.0),
            "duration": chord_result.get("duration", beat_result.get("duration", 0.0)),
            "total_chords": chord_result.get("total_chords", 0),
            "model_used": chord_result.get("model_used", model),
            "chord_dict": chord_result.get("chord_dict", "ismir2017"),
            "used_spleeter": stems_info is not None,
            "lyrics": lyrics,
            "total_words": total_words,
            "processing_time": round(processing_time, 1),
        }

        log_info(f"Combined analysis complete: {response['total_chords']} chords, "
                 f"{len(response['beats'])} beats, BPM {response['bpm']}, "
                 f"{total_words} lyrics words, {response['processing_time']}s")

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
        if stems_info:
            try:
                spleeter_service.cleanup_stems(stems_info)
            except Exception as e:
                log_error(f"Failed to cleanup stems: {e}")

        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                log_debug(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                log_error(f"Failed to clean up temp file {temp_file_path}: {e}")


@analyze_bp.route('/api/warmup', methods=['POST', 'GET'])
@limiter.limit("60/minute")
def warmup():
    """
    Wake idle Replicate deployment containers before the user uploads audio.

    iOS fires this when the import flow opens so the GPU is already booted
    by the time analyze starts. We fire predictions in background threads
    and return immediately (~50ms) — the actual warm-up happens on Replicate.

    Replicate bills only for GPU run-time, not boot, and these empty-input
    predictions fail validation before any GPU work happens, so warm-up is
    effectively free.
    """
    from utils.replicate_utils import warmup_deployment

    deploy_vars = [
        "RIFF_DEPLOY_SPLEETER",
        "RIFF_DEPLOY_CHORD",
        "RIFF_DEPLOY_WHISPER",
    ]
    configured = [v for v in deploy_vars if os.environ.get(v)]
    if not configured:
        return jsonify({
            "success": False,
            "warmed": [],
            "reason": "No RIFF_DEPLOY_* env vars set — running against public models",
        }), 200

    # Fire-and-forget: each warm-up takes 200-500ms to hand off to Replicate.
    # We could block but the iOS client is already async; returning quick
    # lets it move on.
    pool = ThreadPoolExecutor(max_workers=len(configured))
    for var in configured:
        pool.submit(warmup_deployment, var)
    pool.shutdown(wait=False)

    return jsonify({
        "success": True,
        "warmed": configured,
    }), 200
