"""
Lyrics transcription service.

Priority order:
1. Replicate incredibly-fast-whisper — $0.0027/run, ~3s, GPU large-v3
2. Local faster-whisper              — free, ~30-60s, CPU base model
"""

import json
import os
import time
from typing import Dict, Any, Optional, List
from utils.logging import log_info, log_error, log_debug


class LyricsTranscriptionService:
    """
    Transcribes lyrics from audio with word-level timestamps.

    Uses Replicate GPU Whisper when REPLICATE_API_TOKEN is set,
    otherwise falls back to local faster-whisper on CPU.
    """

    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        self._model = None
        self._local_available = None
        self._replicate_available = None

    def is_available(self) -> bool:
        return self._check_replicate() or self._check_local()

    def _check_replicate(self) -> bool:
        if self._replicate_available is not None:
            return self._replicate_available
        try:
            import replicate  # noqa: F401
            self._replicate_available = bool(os.environ.get('REPLICATE_API_TOKEN'))
            return self._replicate_available
        except ImportError:
            self._replicate_available = False
            return False

    def _check_local(self) -> bool:
        if self._local_available is not None:
            return self._local_available
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            self._local_available = True
            return True
        except ImportError:
            self._local_available = False
            return False

    def _get_local_model(self):
        if self._model is not None:
            return self._model
        if not self._check_local():
            raise RuntimeError("faster-whisper not available")
        from faster_whisper import WhisperModel
        log_info(f"Loading local Whisper model: {self._model_size} (CPU, int8)")
        start = time.time()
        self._model = WhisperModel(self._model_size, device="cpu", compute_type="int8")
        log_info(f"Whisper model loaded in {time.time() - start:.1f}s")
        return self._model

    # ------------------------------------------------------------------
    # Replicate incredibly-fast-whisper (~$0.0027, ~3s)
    # ------------------------------------------------------------------

    def _transcribe_replicate(self, audio_path: str,
                              language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe using Replicate GPU Whisper (large-v3, ~3s)."""
        start_time = time.time()

        try:
            from utils.replicate_utils import replicate_run_with_retry

            log_info(f"Transcribing via Replicate incredibly-fast-whisper: {audio_path}")

            inputs = {
                "audio": open(audio_path, 'rb'),
                "timestamp": "word",
                "batch_size": 24,
            }
            if language:
                inputs["language"] = language

            output = replicate_run_with_retry(
                "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
                input=inputs,
            )

            words = self._parse_replicate_output(output)
            processing_time = time.time() - start_time

            log_info(f"Replicate Whisper: {len(words)} words in {processing_time:.1f}s")

            return {
                "success": True,
                "lyrics": words,
                "total_words": len(words),
                "processing_time": round(processing_time, 1),
            }

        except Exception as e:
            error_msg = f"Replicate Whisper failed: {str(e)}"
            log_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "lyrics": [],
                "processing_time": round(time.time() - start_time, 1),
            }

    def _parse_replicate_output(self, output) -> List[Dict[str, Any]]:
        """Parse incredibly-fast-whisper output into word list."""
        # The output can be a dict, JSON string, or other format
        data = output
        if isinstance(data, str):
            data = json.loads(data)

        # Output format: {"text": "...", "chunks": [{"text": "word", "timestamp": [start, end]}, ...]}
        # or: {"segments": [{"words": [{"word": "...", "start": 0.0, "end": 0.5}, ...]}]}
        words = []

        # Format 1: chunks with timestamps
        chunks = None
        if isinstance(data, dict):
            chunks = data.get("chunks")
        if chunks:
            is_first = True
            for chunk in chunks:
                text = chunk.get("text", "").strip()
                if not text:
                    continue
                ts = chunk.get("timestamp", [0, 0])
                start = float(ts[0]) if ts and ts[0] is not None else 0.0
                end = float(ts[1]) if ts and len(ts) > 1 and ts[1] is not None else start + 0.5

                is_line_start = is_first
                if not is_first and words:
                    prev = words[-1]["word"]
                    gap = start - words[-1].get("end", 0)
                    if prev.endswith((".", "?", "!")) or (prev.endswith(",") and gap > 1.0):
                        is_line_start = True

                words.append({
                    "word": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "is_line_start": is_line_start,
                })
                is_first = False
            return words

        # Format 2: segments with words
        segments = data.get("segments", []) if isinstance(data, dict) else []
        is_first = True
        for seg in segments:
            seg_words = seg.get("words", [])
            for w in seg_words:
                text = w.get("word", w.get("text", "")).strip()
                if not text:
                    continue
                start = float(w.get("start", 0))
                end = float(w.get("end", start + 0.5))

                is_line_start = is_first
                if not is_first and words:
                    prev = words[-1]["word"]
                    gap = start - words[-1].get("end", 0)
                    if prev.endswith((".", "?", "!")) or (prev.endswith(",") and gap > 1.0):
                        is_line_start = True

                words.append({
                    "word": text,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "is_line_start": is_line_start,
                })
                is_first = False

        # Format 3: plain text (no timestamps — shouldn't happen with timestamp="word")
        if not words and isinstance(data, dict) and "text" in data:
            log_error("Replicate Whisper returned text without word timestamps")

        return words

    # ------------------------------------------------------------------
    # Local faster-whisper (free, ~30-60s)
    # ------------------------------------------------------------------

    def _transcribe_local(self, audio_path: str,
                          language: Optional[str] = None) -> Dict[str, Any]:
        """Transcribe using local faster-whisper on CPU."""
        start_time = time.time()

        try:
            model = self._get_local_model()

            log_info(f"Transcribing locally ({self._model_size}): {audio_path}")

            segments, info = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=300,
                    speech_pad_ms=600,
                    threshold=0.15,
                ),
                initial_prompt="Song lyrics, verse and chorus:",
            )

            words: List[Dict[str, Any]] = []
            is_first = True

            for segment in segments:
                if not segment.words:
                    continue
                for w in segment.words:
                    text = w.word.strip()
                    if not text:
                        continue
                    is_line_start = is_first
                    if not is_first and words:
                        prev = words[-1]["word"]
                        if prev.endswith((".", "?", "!")) or \
                           (prev.endswith(",") and (w.start - words[-1]["end"]) > 1.0):
                            is_line_start = True
                    words.append({
                        "word": text,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "is_line_start": is_line_start,
                    })
                    is_first = False

            processing_time = time.time() - start_time
            log_info(f"Local Whisper: {len(words)} words in {processing_time:.1f}s, "
                     f"lang: {info.language}")

            return {
                "success": True,
                "lyrics": words,
                "language": info.language,
                "total_words": len(words),
                "processing_time": round(processing_time, 1),
            }

        except Exception as e:
            error_msg = f"Local Whisper error: {str(e)}"
            log_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "lyrics": [],
                "processing_time": round(time.time() - start_time, 1),
            }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe lyrics. Tries Replicate GPU first, falls back to local CPU.
        """
        if self._check_replicate():
            result = self._transcribe_replicate(audio_path, language)
            if result.get("success"):
                return result
            log_error(f"Replicate Whisper failed, trying local: {result.get('error')}")

        if self._check_local():
            return self._transcribe_local(audio_path, language)

        return {
            "success": False,
            "error": "No transcription service available",
            "lyrics": [],
            "processing_time": 0,
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "replicate_gpu": self._check_replicate(),
            "local_cpu": self._check_local(),
            "local_model_size": self._model_size,
        }
