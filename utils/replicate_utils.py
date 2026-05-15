"""
Shared Replicate API helpers.

`replicate_run_with_retry` calls a public model with retry on 429.
`run_deployment_or_model` prefers a configured private deployment
(`owner/name`) and falls back to a public model:version when the
deployment env var is unset — letting us roll out deployments
without breaking the deployment-less local dev path.
"""

import os
import random
import re
import time
from typing import Optional
from utils.logging import log_info, log_error


def replicate_run_with_retry(model_id: str, input: dict,
                             max_retries: int = 3, base_wait: float = 8.0):
    """
    Call replicate.run() with automatic retry on 429 rate-limit errors.

    Args:
        model_id: Model identifier ("owner/model:version")
        input: Input dict for the model
        max_retries: Number of retries on 429
        base_wait: Seconds to wait before retry (parsed from error if possible)

    Returns:
        The replicate.run() output
    """
    import replicate

    for attempt in range(max_retries + 1):
        try:
            log_info(f"Running Replicate model: {model_id.split(':')[0]}")
            return replicate.run(model_id, input=input)
        except Exception as e:
            error_str = str(e)
            if "429" not in error_str or attempt >= max_retries:
                raise

            # Parse wait time from error: "resets in ~7s"
            wait = base_wait
            match = re.search(r"resets in ~(\d+)s", error_str)
            if match:
                wait = int(match.group(1)) + 1
            wait += random.uniform(0, 3)

            log_info(f"Replicate 429 rate limit — retrying in {wait:.1f}s "
                     f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)


def run_deployment_or_model(deployment_env: str, fallback_model_id: str,
                            input: dict, *, max_retries: int = 3,
                            base_wait: float = 8.0):
    """
    Run a Replicate prediction, preferring a private deployment when configured.

    If the env var `deployment_env` is set to "owner/name", call that
    deployment (warm pool, no public-tier queueing). Otherwise fall back
    to `replicate.run(fallback_model_id)` against the public model.

    Returns the same shape as replicate.run() (FileOutput / dict / iterable),
    so callers don't need to change how they consume output.
    """
    import replicate

    slug = os.environ.get(deployment_env)
    if not slug:
        return replicate_run_with_retry(fallback_model_id, input,
                                        max_retries=max_retries,
                                        base_wait=base_wait)

    if "/" not in slug:
        log_error(f"{deployment_env}={slug!r} is not in 'owner/name' form — "
                  f"falling back to public model")
        return replicate_run_with_retry(fallback_model_id, input,
                                        max_retries=max_retries,
                                        base_wait=base_wait)

    owner, name = slug.split("/", 1)
    for attempt in range(max_retries + 1):
        try:
            log_info(f"Running Replicate deployment: {slug}")
            deployment = replicate.deployments.get(slug)
            prediction = deployment.predictions.create(input=input)
            prediction.wait()
            if prediction.status != "succeeded":
                raise RuntimeError(
                    f"Deployment {slug} prediction {prediction.id} "
                    f"ended with status {prediction.status}: {prediction.error}"
                )
            return prediction.output
        except Exception as e:
            error_str = str(e)
            if "429" not in error_str or attempt >= max_retries:
                raise
            wait = base_wait
            match = re.search(r"resets in ~(\d+)s", error_str)
            if match:
                wait = int(match.group(1)) + 1
            wait += random.uniform(0, 3)
            log_info(f"Replicate 429 on deployment — retrying in {wait:.1f}s "
                     f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)


def _ensure_silence_wav() -> str:
    """Create a 3-second 16kHz mono 16-bit PCM silence WAV on disk and return
    its path. Written once and reused for all warmup calls.

    Why 3 seconds: shorter clips (1s) were letting predict() finish in
    milliseconds, so the autoscaler may have marked the container idle
    immediately. 3 seconds gives the container enough work to register as
    "active" while still being tiny (~96KB).

    Why on disk (not BytesIO): Replicate's Python SDK occasionally chokes on
    pure in-memory file objects for models that probe filename or
    content-type. A real file path is the reliable choice.
    """
    import tempfile
    import wave
    path = os.path.join(tempfile.gettempdir(), "riff_silence_3s.wav")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * (16000 * 3))  # 3s of silence
    return path


def warmup_deployment(deployment_env: str) -> Optional[str]:
    """
    Fire a real silence-audio prediction against a deployment to keep the
    container warm. We use a tiny valid audio file (not empty input) so
    predict() actually runs — Replicate's autoscaler only counts containers
    serving real predictions as "active" and will scale them down otherwise.

    Returns the prediction id (or None if the env var isn't set / call fails).
    Does not wait for completion — caller should not block on the result.
    """
    slug = os.environ.get(deployment_env)
    if not slug or "/" not in slug:
        log_info(f"Warmup skipped: {deployment_env} not set in env")
        return None
    try:
        import replicate
        deployment = replicate.deployments.get(slug)
        silence_path = _ensure_silence_wav()
        with open(silence_path, "rb") as f:
            prediction = deployment.predictions.create(input={"audio": f})
        log_info(f"Warmup fired for {deployment_env}={slug} → prediction {prediction.id}")
        return prediction.id
    except Exception as e:
        log_error(f"Warmup failed for {deployment_env}={slug}: {e}")
        return None
