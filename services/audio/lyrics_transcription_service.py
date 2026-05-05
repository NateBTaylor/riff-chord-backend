"""
Lyrics transcription service using faster-whisper.

Runs Whisper on a vocals stem (from Spleeter separation) to produce
word-level timestamped lyrics. Clean vocals without background music
dramatically improve transcription accuracy.
"""

import time
from typing import Dict, Any, Optional, List
from utils.logging import log_info, log_error, log_debug


class LyricsTranscriptionService:
    """
    Service for transcribing lyrics from audio using faster-whisper.
    Designed to run on isolated vocals from Spleeter separation.
    """

    def __init__(self, model_size: str = "small"):
        """
        Initialize the lyrics transcription service.

        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3')
                       'small' is a good balance of accuracy vs RAM on Railway.
        """
        self._model_size = model_size
        self._model = None
        self._available = None

    def is_available(self) -> bool:
        """Check if faster-whisper is available."""
        if self._available is not None:
            return self._available

        try:
            from faster_whisper import WhisperModel
            self._available = True
            log_debug("faster-whisper is available")
            return True
        except ImportError as e:
            log_error(f"faster-whisper import failed: {e}")
            self._available = False
            return False

    def _get_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is not None:
            return self._model

        if not self.is_available():
            raise RuntimeError("faster-whisper is not available")

        from faster_whisper import WhisperModel

        log_info(f"Loading Whisper model: {self._model_size} (CPU, int8)")
        start = time.time()
        self._model = WhisperModel(
            self._model_size,
            device="cpu",
            compute_type="int8",
        )
        log_info(f"Whisper model loaded in {time.time() - start:.1f}s")
        return self._model

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe lyrics from an audio file with word-level timestamps.

        Args:
            audio_path: Path to audio file (ideally isolated vocals from Spleeter)
            language: Language code (e.g. 'en'). None for auto-detection.

        Returns:
            Dict with:
                success: bool
                lyrics: list of {word, start, end, is_line_start}
                language: detected language
                processing_time: float
        """
        start_time = time.time()

        try:
            model = self._get_model()

            log_info(f"Transcribing lyrics from: {audio_path}")

            segments, info = model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400,
                    threshold=0.3,
                ),
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

                    # Detect line starts: first word, or after sentence-ending punctuation
                    is_line_start = is_first
                    if not is_first and words:
                        prev = words[-1]["word"]
                        if prev.endswith((".","?","!")) or prev.endswith((",")) and (w.start - words[-1]["end"]) > 1.0:
                            is_line_start = True

                    words.append({
                        "word": text,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "is_line_start": is_line_start,
                    })
                    is_first = False

            processing_time = time.time() - start_time

            log_info(f"Transcribed {len(words)} words in {processing_time:.1f}s, "
                     f"language: {info.language} ({info.language_probability:.0%})")

            return {
                "success": True,
                "lyrics": words,
                "language": info.language,
                "language_probability": round(info.language_probability, 2),
                "total_words": len(words),
                "processing_time": round(processing_time, 1),
            }

        except Exception as e:
            error_msg = f"Lyrics transcription error: {str(e)}"
            log_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "lyrics": [],
                "processing_time": round(time.time() - start_time, 1),
            }

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the transcription model."""
        return {
            "available": self.is_available(),
            "model_size": self._model_size,
            "model_loaded": self._model is not None,
            "device": "cpu",
            "compute_type": "int8",
        }
