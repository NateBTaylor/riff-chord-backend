"""
Combined analysis endpoint for Riff iOS app.

Pipeline: beat detection (librosa) → chord recognition + lyrics transcription in parallel.
"""

import gc
import json
import os
import threading
import time
import tempfile
import traceback
import uuid
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
    Combined analysis: beat detection + chord recognition + lyrics transcription.

    Beat detection runs first (fast with librosa), then chord recognition and
    lyrics transcription run in parallel to minimize total wall time.

    Parameters:
    - file: Audio file (multipart/form-data)
    - model: Chord model (default 'chord-cnn-lstm')
    - detector: Beat detector (default 'librosa')
    - chord_dict: Chord dictionary (optional)

    Returns:
    - JSON with chords, beats, bpm, duration, lyrics, and metadata
    """
    temp_file_path = None
    start_time = time.time()

    try:
        # Validate file upload
        file = request.files.get('file')
        if not file:
            return jsonify({"error": "No audio file provided"}), 400

        # Parse parameters
        model = request.form.get('model', 'chroma').lower()
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

        if not chord_service:
            return jsonify({"error": "Chord recognition service unavailable"}), 503
        if not beat_service:
            return jsonify({"error": "Beat detection service unavailable"}), 503

        # --- Step 1: Beat detection (fast with librosa ~2-5s) ---
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

        gc.collect()

        # --- Step 2+3: Chord recognition + lyrics transcription in parallel ---
        chord_result = None
        lyrics_result = None

        def run_chords():
            log_info("Step 2/3: Chord recognition")
            return chord_service.recognize_chords(
                file_path=temp_file_path,
                detector=model,
                chord_dict=chord_dict,
                force=False,
                use_spleeter=False,
            )

        def run_lyrics():
            log_info("Step 3/3: Lyrics transcription")
            return lyrics_service.transcribe(audio_path=temp_file_path)

        lyrics_job_id = None

        if lyrics_service:
            executor = ThreadPoolExecutor(max_workers=2)
            chord_future = executor.submit(run_chords)
            lyrics_future = executor.submit(run_lyrics)

            try:
                chord_result = chord_future.result(timeout=120)
            except Exception as e:
                log_error(f"Chord recognition failed: {e}")

            # Wait up to 20s for lyrics with the initial response
            try:
                lyrics_result = lyrics_future.result(timeout=20)
            except Exception as e:
                log_info(f"Lyrics not ready in 20s, continuing in background")
                # Let lyrics finish in background and store result in Redis
                job_service = current_app.extensions.get('job_service')
                if job_service:
                    lyrics_job_id = uuid.uuid4().hex[:12]
                    job_service.redis.hset(f"lyrics:{lyrics_job_id}", mapping={
                        "status": "processing",
                    })
                    job_service.redis.expire(f"lyrics:{lyrics_job_id}", 600)

                    redis_client = job_service.redis
                    def finish_lyrics():
                        try:
                            result = lyrics_future.result(timeout=300)
                            if result and result.get('success'):
                                words = result.get('lyrics', [])
                                redis_client.hset(f"lyrics:{lyrics_job_id}", mapping={
                                    "status": "complete",
                                    "result": json.dumps(words),
                                    "total_words": len(words),
                                })
                                log_info(f"[Lyrics {lyrics_job_id}] Background complete: {len(words)} words")
                            else:
                                error = result.get('error', 'Unknown') if result else 'Unknown'
                                redis_client.hset(f"lyrics:{lyrics_job_id}", mapping={
                                    "status": "failed", "error": error,
                                })
                        except Exception as ex:
                            log_error(f"[Lyrics {lyrics_job_id}] Background failed: {ex}")
                            redis_client.hset(f"lyrics:{lyrics_job_id}", mapping={
                                "status": "failed", "error": str(ex),
                            })
                        finally:
                            # Clean up temp file if it still exists
                            if temp_file_path and os.path.exists(temp_file_path):
                                try:
                                    os.unlink(temp_file_path)
                                except Exception:
                                    pass

                    threading.Thread(target=finish_lyrics, daemon=True).start()
                    # Prevent the finally block from deleting the temp file
                    # while the background thread still needs it
                    temp_file_path = None

            # Don't wait for still-running lyrics task to finish
            executor.shutdown(wait=False)
        else:
            log_info("Step 2/2: Chord recognition (lyrics service unavailable)")
            try:
                chord_result = run_chords()
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
            "chord_dict": chord_result.get("chord_dict", "submission"),
            "used_spleeter": False,
            "lyrics": lyrics,
            "total_words": total_words,
            "processing_time": round(processing_time, 1),
        }

        # Include lyrics job ID so the client can poll for results
        if lyrics_job_id:
            response["lyrics_job_id"] = lyrics_job_id

        log_info(f"Combined analysis complete: {response['total_chords']} chords, "
                 f"{len(response['beats'])} beats, BPM {response['bpm']}, "
                 f"{total_words} lyrics words, {response['processing_time']}s"
                 f"{f', lyrics pending: {lyrics_job_id}' if lyrics_job_id else ''}")

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
        # Clean up temp file (skipped if background lyrics thread owns it)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                log_debug(f"Cleaned up temp file: {temp_file_path}")
            except Exception as e:
                log_error(f"Failed to clean up temp file {temp_file_path}: {e}")


@analyze_bp.route('/api/lyrics/<lyrics_id>', methods=['GET'])
def get_lyrics(lyrics_id):
    """Poll for background lyrics transcription results."""
    job_service = current_app.extensions.get('job_service')
    if not job_service:
        return jsonify({"error": "Service unavailable"}), 503

    data = job_service.redis.hgetall(f"lyrics:{lyrics_id}")
    if not data:
        return jsonify({"error": "Not found"}), 404

    status = data.get("status", "unknown")
    if status == "processing":
        return jsonify({"status": "processing"}), 202

    if status == "complete":
        lyrics = json.loads(data.get("result", "[]"))
        total_words = int(data.get("total_words", 0))
        return jsonify({
            "status": "complete",
            "lyrics": lyrics,
            "total_words": total_words,
        })

    return jsonify({"status": "failed", "error": data.get("error", "Unknown")}), 500
