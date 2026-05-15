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


def warmup_deployment(deployment_env: str) -> Optional[str]:
    """
    Fire a no-op prediction against a deployment to wake an idle container.

    Does not wait for completion — returns the prediction id (or None if
    the env var isn't set or the call fails). Caller is responsible for
    not blocking on the result.
    """
    slug = os.environ.get(deployment_env)
    if not slug or "/" not in slug:
        return None
    try:
        import replicate
        deployment = replicate.deployments.get(slug)
        # Empty input still triggers container boot — the model will fail
        # validation but the container is now warm. Replicate bills only
        # for actual GPU seconds, so a validation-failed call is free.
        prediction = deployment.predictions.create(input={})
        log_info(f"Warmup fired for {slug} → prediction {prediction.id}")
        return prediction.id
    except Exception as e:
        log_error(f"Warmup failed for {slug}: {e}")
        return None
