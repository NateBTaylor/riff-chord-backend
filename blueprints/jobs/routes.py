"""
Async job queue endpoints for song analysis.

POST /api/jobs/analyze — submit a job, returns job_id (HTTP 202)
GET  /api/jobs/<job_id> — poll job status and retrieve results
"""

import json
import os
import tempfile
from flask import Blueprint, request, jsonify, current_app
from extensions import limiter
from config import get_config
from utils.logging import log_info, log_error

jobs_bp = Blueprint('jobs', __name__)
config = get_config()


def _get_job_service():
    """Get job service, returns None if Redis unavailable."""
    return current_app.extensions.get('job_service')


@jobs_bp.route('/api/jobs/analyze', methods=['POST'])
@limiter.limit(config.get_rate_limit('heavy_processing'))
def submit_job():
    """
    Submit an async analysis job.

    Accepts the same multipart upload as /api/analyze.
    Returns HTTP 202 with {"job_id": "..."} immediately.
    """
    job_service = _get_job_service()
    if not job_service:
        return jsonify({"error": "Job queue unavailable (no Redis)"}), 503

    # Validate file upload
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No audio file provided"}), 400

    # Parse parameters
    params = {
        "model": request.form.get('model', 'auto').lower(),
        "detector": request.form.get('detector', 'librosa').lower(),
        "chord_dict": request.form.get('chord_dict', None),
    }

    # Save uploaded file to a temp location that persists beyond this request
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.m4a')
    file.save(temp_file.name)
    temp_file_path = temp_file.name

    try:
        job_id = job_service.create_job(params, temp_file_path)
        job_service.start_processing(job_id, current_app._get_current_object())
        log_info(f"Job {job_id} created: model={params['model']}, detector={params['detector']}")
        return jsonify({"job_id": job_id}), 202
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        log_error(f"Failed to create job: {e}")
        return jsonify({"error": "Failed to create job"}), 500


@jobs_bp.route('/api/jobs/<job_id>', methods=['GET'])
@limiter.limit(config.get_rate_limit('light_processing'))
def get_job(job_id):
    """
    Poll job status.

    Returns:
    - {job_id, status:"pending"|"processing", step} while running
    - {job_id, status:"complete", result:{...}} when done
    - {job_id, status:"failed", error:"..."} on failure
    - 404 if job_id not found or expired
    """
    job_service = _get_job_service()
    if not job_service:
        return jsonify({"error": "Job queue unavailable (no Redis)"}), 503

    job = job_service.get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    response = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "step": job.get("step", "unknown"),
    }

    if job["status"] == "complete" and job.get("result"):
        response["result"] = json.loads(job["result"])
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return jsonify(response)
