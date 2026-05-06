"""
Async job queue service using Redis for state and threading for background processing.

Jobs flow: pending → processing → complete | failed
"""

import gc
import json
import os
import tempfile
import threading
import time
import traceback
import uuid

from utils.logging import log_info, log_error


class JobService:
    """Manages async analysis jobs backed by Redis."""

    JOB_TTL = 3600  # 1 hour

    def __init__(self, redis_url: str):
        import redis
        self.redis = redis.from_url(redis_url, decode_responses=True)
        # Verify connection
        self.redis.ping()
        log_info("JobService connected to Redis")

    def create_job(self, params: dict, temp_file_path: str) -> str:
        """Create a new job in Redis. Returns job_id."""
        job_id = uuid.uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "status": "pending",
            "step": "queued",
            "params": json.dumps(params),
            "temp_file_path": temp_file_path,
            "created_at": time.time(),
            "error": "",
            "result": "",
        }
        self.redis.hset(f"job:{job_id}", mapping=job)
        self.redis.expire(f"job:{job_id}", self.JOB_TTL)
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        """Read job state from Redis."""
        data = self.redis.hgetall(f"job:{job_id}")
        if not data:
            return None
        return data

    def update_job(self, job_id: str, **fields):
        """Update specific fields on a job."""
        if fields:
            self.redis.hset(f"job:{job_id}", mapping=fields)

    def start_processing(self, job_id: str, app):
        """Spawn a daemon thread to process the job."""
        t = threading.Thread(
            target=self._process_job,
            args=(job_id, app),
            daemon=True,
        )
        t.start()

    def _process_job(self, job_id: str, app):
        """Run analysis in background thread with app context."""
        with app.app_context():
            try:
                self.update_job(job_id, status="processing", step="beat_detection")

                job = self.get_job(job_id)
                if not job:
                    return

                params = json.loads(job["params"])
                temp_file_path = job["temp_file_path"]
                model = params.get("model", "chord-cnn-lstm")
                detector = params.get("detector", "librosa")
                chord_dict = params.get("chord_dict")

                from flask import current_app
                chord_service = current_app.extensions['services']['chord_recognition']
                beat_service = current_app.extensions['services']['beat_detection']

                if not beat_service or not chord_service:
                    self.update_job(job_id, status="failed", error="Services unavailable")
                    return

                # Step 1: Beat detection
                log_info(f"[Job {job_id}] Step 1/2: Beat detection")
                beat_result = beat_service.detect_beats(
                    file_path=temp_file_path,
                    detector=detector,
                    force=False,
                )

                if not beat_result or not beat_result.get('success'):
                    error = beat_result.get('error') if beat_result else 'Unknown'
                    self.update_job(job_id, status="failed", error=f"Beat detection failed: {error}")
                    return

                gc.collect()

                # Step 2: Chord recognition
                self.update_job(job_id, step="chord_recognition")
                log_info(f"[Job {job_id}] Step 2/2: Chord recognition")
                chord_result = chord_service.recognize_chords(
                    file_path=temp_file_path,
                    detector=model,
                    chord_dict=chord_dict,
                    force=False,
                    use_spleeter=False,
                )

                if not chord_result or not chord_result.get('success'):
                    error = chord_result.get('error') if chord_result else 'Unknown'
                    self.update_job(job_id, status="failed", error=f"Chord recognition failed: {error}")
                    return

                gc.collect()

                # Build result (same shape as /api/analyze response)
                processing_time = time.time() - float(job["created_at"])
                result = {
                    "success": True,
                    "chords": chord_result.get("chords", []),
                    "beats": beat_result.get("beats", []),
                    "bpm": beat_result.get("bpm", 0.0),
                    "duration": chord_result.get("duration", beat_result.get("duration", 0.0)),
                    "total_chords": chord_result.get("total_chords", 0),
                    "model_used": chord_result.get("model_used", model),
                    "chord_dict": chord_result.get("chord_dict", "submission"),
                    "used_spleeter": False,
                    "lyrics": [],
                    "total_words": 0,
                    "processing_time": round(processing_time, 1),
                }

                self.update_job(
                    job_id,
                    status="complete",
                    step="done",
                    result=json.dumps(result),
                )
                log_info(f"[Job {job_id}] Complete: {result['total_chords']} chords, "
                         f"{len(result['beats'])} beats, BPM {result['bpm']}, "
                         f"{result['processing_time']}s")

            except Exception as e:
                log_error(f"[Job {job_id}] Failed: {e}")
                log_error(traceback.format_exc())
                self.update_job(job_id, status="failed", error=str(e))
            finally:
                # Clean up temp file
                temp_path = self.get_job(job_id).get("temp_file_path", "") if self.get_job(job_id) else ""
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
