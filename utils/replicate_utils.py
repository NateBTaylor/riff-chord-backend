"""
Shared Replicate API helpers.

Uses a threading lock to serialize prediction creation, preventing
concurrent requests from hitting Replicate's rate limits.
"""

import re
import threading
import time
from utils.logging import log_info, log_error

# Serialize all Replicate calls so concurrent threads (chords + whisper)
# don't fire predictions simultaneously and get 429'd.
_replicate_lock = threading.Lock()


def replicate_run_with_retry(model_id: str, input: dict,
                             max_retries: int = 2, base_wait: float = 8.0):
    """
    Call replicate.run() with a lock to prevent concurrent requests
    and automatic retry on 429 rate-limit errors.

    The lock ensures only one replicate.run() executes at a time.
    Since replicate.run() blocks until the prediction completes,
    this serializes both creation and polling.

    Args:
        model_id: Model identifier (with or without version hash)
        input: Input dict for the model
        max_retries: Number of retries on 429
        base_wait: Seconds to wait before retry (parsed from error if possible)

    Returns:
        The replicate.run() output
    """
    import replicate

    with _replicate_lock:
        for attempt in range(max_retries + 1):
            try:
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

                log_info(f"Replicate 429 rate limit — retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
