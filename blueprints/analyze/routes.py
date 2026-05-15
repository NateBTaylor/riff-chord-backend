"""
Combined analysis endpoint for Riff iOS app.

Two execution paths:

1. Modal (fast path)
   When MODAL_APP_NAME is set, we send the audio to the riff-pipeline
   Modal app which runs demucs + chord-cnn-lstm + faster-whisper in one
   container. Memory snapshots make cold starts ~10-20s; warm ~10-15s.
   Beat detection still runs locally (librosa, ~2s) in parallel.

2. Replicate (fallback path)
   Original serial pipeline: beats → spleeter → whisper → chord, with
   sleep pauses between calls to dodge 429s on public models.
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
from utils import modal_client

# Create blueprint
analyze_bp = Blueprint('analyze', __name__)

# Get configuration for rate limits
config = get_config()


@analyze_bp.route('/api/analyze', methods=['POST'])
@limiter.limit(config.get_rate_limit('heavy_processing'))
def analyze():
    """
    Combined analysis: beat detection + chord recognition + lyrics transcription.

    Steps run sequentially to avoid Replicate 429 rate-limit collisions:
    1. Beat detection (local librosa, ~2s)
    2. Spleeter stem separation (Replicate, ~3s) — vocals for lyrics
    3. Whisper lyrics transcription (Replicate, ~3s) — on vocals stem
    4. Chord recognition (Replicate CNN-LSTM, ~6s) — on original audio

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

        # --- Modal fast path ---
        # When configured, Modal handles the entire pipeline including
        # beat detection (parallel with demucs inside the container). We
        # skip the local librosa step entirely on this path — saves ~2s
        # round-trip.
        if modal_client.is_enabled():
            log_info("Modal path: combined beats + demucs + chord + whisper")
            with open(temp_file_path, 'rb') as f:
                audio_bytes = f.read()
            modal_result = modal_client.analyze_audio(
                audio_bytes,
                chord_dict=chord_dict or "submission",
            )
            if modal_result and modal_result.get('success'):
                processing_time = time.time() - start_time
                response = {
                    "success": True,
                    "chords": modal_result.get("chords", []),
                    "beats": modal_result.get("beats", []),
                    "bpm": modal_result.get("bpm", 0.0),
                    "duration": modal_result.get("duration", 0.0),
                    "total_chords": len(modal_result.get("chords", [])),
                    "model_used": "modal-combined",
                    "chord_dict": chord_dict or "submission",
                    "used_spleeter": True,
                    "lyrics": modal_result.get("lyrics", []),
                    "total_words": len(modal_result.get("lyrics", [])),
                    "processing_time": round(processing_time, 1),
                }
                log_info(f"Modal pipeline complete: {response['total_chords']} chords, "
                         f"{response['total_words']} lyrics words, "
                         f"BPM {response['bpm']:.1f}, "
                         f"{response['processing_time']}s")
                return jsonify(response)
            log_error("Modal call failed or returned no success — falling back to Replicate")

        # --- Fallback path: local beats + Replicate chord/lyrics ---
        log_info("Step 1: Beat detection (fallback)")
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
            log_info("Step 2/4: Stem separation (vocals for lyrics)")
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
            log_info("Step 2/4: Skipping stem separation (unavailable)")

        # Brief pause between Replicate calls to avoid 429 rate limits
        if stems_info:
            log_info("Waiting 3s before Whisper to avoid Replicate rate limit...")
            time.sleep(3)

        # --- Step 3: Lyrics transcription (Replicate Whisper, ~3s) ---
        lyrics_result = None
        if lyrics_service:
            log_info("Step 3/4: Lyrics transcription" +
                     (" (on vocals stem)" if stems_info else " (on full mix)"))
            try:
                lyrics_result = lyrics_service.transcribe(audio_path=audio_for_lyrics)
            except Exception as e:
                log_error(f"Lyrics transcription failed: {e}")

        # Brief pause between Replicate calls to avoid 429 rate limits
        # Only sleep if a Replicate call actually ran above (Spleeter or Whisper)
        ran_replicate = stems_info is not None or (lyrics_result and lyrics_result.get('success'))
        if ran_replicate:
            log_info("Waiting 5s before chord Replicate call to avoid rate limit...")
            time.sleep(5)
        else:
            log_info("No prior Replicate calls — skipping rate-limit pause")

        # --- Step 4: Chord recognition (Replicate CNN-LSTM on original audio, ~6s) ---
        log_info("Step 4/4: Chord recognition (on original audio)")
        chord_result = None
        try:
            chord_result = chord_service.recognize_chords(
                file_path=temp_file_path,
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
