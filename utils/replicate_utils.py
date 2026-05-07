"""
Shared Replicate API helpers.

Uses manual prediction creation + polling instead of replicate.run()
to support timeouts and detailed status logging.

Predictions run concurrently — 429 rate-limit errors are handled by
automatic retry with jitter rather than a global lock (which caused
head-of-line blocking when one model had a long queue).
"""

import random
import re
import time
from utils.logging import log_info, log_error

# Terminal statuses for Replicate predictions
_TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


def replicate_run_with_retry(model_id: str, input: dict,
                             max_retries: int = 3, base_wait: float = 8.0,
                             timeout: float = 300.0):
    """
    Create a Replicate prediction, poll until complete, with timeout.

    Predictions are NOT serialized — spleeter, whisper, and chord
    detection can run concurrently.  If Replicate returns 429, we
    retry with exponential back-off and jitter.

    Args:
        model_id: Model identifier ("owner/model:version")
        input: Input dict for the model
        max_retries: Number of retries on 429
        base_wait: Seconds to wait before retry (parsed from error if possible)
        timeout: Maximum seconds to wait for prediction completion

    Returns:
        The prediction output
    """
    for attempt in range(max_retries + 1):
        try:
            return _run_with_timeout(model_id, input, timeout)
        except Exception as e:
            error_str = str(e)
            if "429" not in error_str or attempt >= max_retries:
                raise

            # Parse wait time from error: "resets in ~7s"
            wait = base_wait
            match = re.search(r"resets in ~(\d+)s", error_str)
            if match:
                wait = int(match.group(1)) + 1
            # Add jitter to avoid thundering-herd retries
            wait += random.uniform(0, 3)

            log_info(f"Replicate 429 rate limit — retrying in {wait:.1f}s "
                     f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)


def _run_with_timeout(model_id: str, input: dict, timeout: float):
    """Create a Replicate prediction and poll with timeout."""
    import replicate

    # Parse version hash from "owner/model:version"
    if ":" in model_id:
        version = model_id.split(":", 1)[1]
    else:
        version = model_id

    log_info(f"Creating Replicate prediction for {model_id}")

    prediction = replicate.predictions.create(
        version=version,
        input=input,
    )

    log_info(f"Prediction {prediction.id} created (status: {prediction.status})")

    deadline = time.time() + timeout
    last_status = prediction.status

    while prediction.status not in _TERMINAL_STATUSES:
        if time.time() > deadline:
            log_error(
                f"Prediction {prediction.id} timed out after {timeout:.0f}s "
                f"(status: {prediction.status})"
            )
            try:
                prediction.cancel()
                log_info(f"Prediction {prediction.id} canceled")
            except Exception as cancel_err:
                log_error(f"Failed to cancel prediction {prediction.id}: {cancel_err}")
            raise TimeoutError(
                f"Replicate prediction timed out after {timeout:.0f}s "
                f"(last status: {prediction.status})"
            )

        time.sleep(1)
        prediction.reload()

        if prediction.status != last_status:
            log_info(f"Prediction {prediction.id}: {last_status} -> {prediction.status}")
            last_status = prediction.status

    if prediction.status == "failed":
        error_msg = getattr(prediction, "error", None) or "Unknown error"
        raise RuntimeError(f"Replicate prediction failed: {error_msg}")

    if prediction.status == "canceled":
        raise RuntimeError("Replicate prediction was canceled")

    log_info(f"Prediction {prediction.id} succeeded")
    return prediction.output
