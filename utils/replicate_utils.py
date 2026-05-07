"""
Shared Replicate API helpers.

Handles rate-limit (429) retries for accounts with < $5 credit,
which are limited to burst=1 concurrent prediction.
"""

import re
import time
from utils.logging import log_info, log_error


def replicate_run_with_retry(model_id: str, input: dict,
                             max_retries: int = 2, base_wait: float = 8.0):
    """
    Call replicate.run() with automatic retry on 429 rate-limit errors.

    Args:
        model_id: Model identifier (with or without version hash)
        input: Input dict for the model
        max_retries: Number of retries on 429
        base_wait: Seconds to wait before retry (parsed from error if possible)

    Returns:
        The replicate.run() output
    """
    import replicate

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
