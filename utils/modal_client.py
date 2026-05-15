"""
Thin client wrapper around the Modal `riff-pipeline` app.

Why this file exists: the backend services (spleeter / chord / whisper) used
to each make a separate Replicate call. Modal combines them into one call,
so we don't fit cleanly into the per-service abstraction — we want a
single helper the analyze route can use directly.

Gated by MODAL_APP_NAME env var. When unset, callers should fall back to
Replicate. When set (e.g. "riff-pipeline"), this module's `analyze_audio`
function dispatches to the Modal class method.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from utils.logging import log_info, log_error


def is_enabled() -> bool:
    return bool(os.environ.get("MODAL_APP_NAME"))


def analyze_audio(audio_bytes: bytes, chord_dict: str = "submission") -> Optional[dict]:
    """Run the Modal combined pipeline. Returns the raw dict from
    RiffPipeline.analyze, or None if Modal isn't configured / the call fails.

    The Modal class is identified by:
        MODAL_APP_NAME=riff-pipeline         (the modal.App name)
        MODAL_CLASS_NAME=RiffPipeline        (defaults to "RiffPipeline")
        MODAL_METHOD=analyze                 (defaults to "analyze")
    """
    app_name = os.environ.get("MODAL_APP_NAME")
    if not app_name:
        return None

    class_name = os.environ.get("MODAL_CLASS_NAME", "RiffPipeline")
    method_name = os.environ.get("MODAL_METHOD", "analyze")

    try:
        import modal
    except ImportError:
        log_error("MODAL_APP_NAME set but `modal` package not installed")
        return None

    try:
        log_info(f"Calling Modal {app_name}.{class_name}.{method_name} "
                 f"({len(audio_bytes) / 1024:.0f}KB)")
        t0 = time.time()
        cls = modal.Cls.lookup(app_name, class_name)
        method = getattr(cls(), method_name)
        result = method.remote(audio_bytes, chord_dict)
        log_info(f"Modal call complete in {time.time() - t0:.1f}s")
        return result
    except Exception as e:
        log_error(f"Modal call failed: {e}")
        return None
