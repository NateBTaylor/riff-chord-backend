"""
Replicate GPU-backed Chord-CNN-LSTM detector.

Calls the triadmusic/chord-detection-cnn-lstm model on Replicate
for fast, accurate chord recognition on GPU hardware.
"""

import json
import os
import tempfile
import time
from typing import Dict, Any, List, Optional
from utils.logging import log_info, log_error, log_debug


class ReplicateChordDetectorService:
    """
    Chord detection via Replicate's GPU-backed CNN-LSTM model.

    Requires REPLICATE_API_TOKEN environment variable.
    """

    MODEL_ID = "triadmusic/chord-detection-cnn-lstm:be95be03"

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Check if Replicate API is available."""
        if self._available is not None:
            return self._available

        try:
            import replicate  # noqa: F401
            token = os.environ.get('REPLICATE_API_TOKEN')
            self._available = bool(token)
            if self._available:
                log_debug("Replicate chord detector available")
            return self._available
        except ImportError:
            self._available = False
            return False

    def recognize_chords(self, file_path: str, chord_dict: str = 'submission',
                         **kwargs) -> Dict[str, Any]:
        """
        Recognize chords via Replicate GPU.

        Returns the same normalized format as other detectors.
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Replicate API not available",
                "model_used": "chord-cnn-lstm (replicate)",
                "model_name": "Chord-CNN-LSTM (Replicate GPU)",
            }

        start_time = time.time()

        try:
            import replicate

            log_info(f"Running Chord-CNN-LSTM via Replicate GPU on: {file_path}")

            with open(file_path, 'rb') as f:
                output = replicate.run(
                    self.MODEL_ID,
                    input={"audio": f},
                )

            chords = self._parse_output(output)
            duration = chords[-1]["end"] if chords else 0.0
            processing_time = time.time() - start_time

            log_info(f"Replicate chord detection: {len(chords)} chords in {processing_time:.1f}s")

            return {
                "success": True,
                "chords": chords,
                "total_chords": len(chords),
                "duration": round(duration, 3),
                "model_used": "chord-cnn-lstm (replicate)",
                "model_name": "Chord-CNN-LSTM (Replicate GPU)",
                "chord_dict": chord_dict,
                "processing_time": round(processing_time, 1),
            }

        except Exception as e:
            error_msg = f"Replicate chord detection error: {str(e)}"
            log_error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "model_used": "chord-cnn-lstm (replicate)",
                "model_name": "Chord-CNN-LSTM (Replicate GPU)",
                "chord_dict": chord_dict,
                "processing_time": round(time.time() - start_time, 1),
            }

    def _parse_output(self, output) -> List[Dict[str, Any]]:
        """Parse Replicate model output into normalized chord list."""

        # Case 1: JSON string
        if isinstance(output, str):
            return self._parse_json_or_lab(output)

        # Case 2: dict with chords key
        if isinstance(output, dict):
            if "chords" in output:
                return self._normalize_chord_list(output["chords"])
            # Might be the full result dict
            return self._normalize_chord_list([output])

        # Case 3: list of chord dicts
        if isinstance(output, list):
            return self._normalize_chord_list(output)

        # Case 4: FileOutput — read content
        if hasattr(output, 'read'):
            content = output.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            return self._parse_json_or_lab(content)

        # Case 5: iterable (e.g. streaming output)
        try:
            chunks = []
            for chunk in output:
                if isinstance(chunk, bytes):
                    chunks.append(chunk.decode('utf-8'))
                elif isinstance(chunk, str):
                    chunks.append(chunk)
            content = ''.join(chunks)
            if content:
                return self._parse_json_or_lab(content)
        except TypeError:
            pass

        # Last resort: try string conversion
        content = str(output)
        return self._parse_json_or_lab(content)

    def _parse_json_or_lab(self, content: str) -> List[Dict[str, Any]]:
        """Parse content as JSON or tab-separated lab format."""
        content = content.strip()

        # Try JSON first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return self._normalize_chord_list(data)
            if isinstance(data, dict) and "chords" in data:
                return self._normalize_chord_list(data["chords"])
            if isinstance(data, dict):
                return self._normalize_chord_list([data])
        except (json.JSONDecodeError, ValueError):
            pass

        # Try tab-separated lab format: start\tend\tchord
        chords = []
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()  # Try whitespace
            if len(parts) >= 3:
                try:
                    chords.append({
                        "start": round(float(parts[0]), 3),
                        "end": round(float(parts[1]), 3),
                        "chord": parts[2],
                        "confidence": float(parts[3]) if len(parts) > 3 else 1.0,
                    })
                except (ValueError, IndexError):
                    continue

        if chords:
            return chords

        log_error(f"Could not parse Replicate chord output: {content[:200]}")
        return []

    def _normalize_chord_list(self, raw_chords: list) -> List[Dict[str, Any]]:
        """Normalize a list of chord dicts to our standard format."""
        chords = []
        for item in raw_chords:
            if not isinstance(item, dict):
                continue
            chord = {
                "start": round(float(item.get("start", item.get("start_time", 0))), 3),
                "end": round(float(item.get("end", item.get("end_time", 0))), 3),
                "chord": item.get("chord", item.get("label", item.get("name", "N"))),
                "confidence": round(float(item.get("confidence", item.get("score", 1.0))), 3),
            }
            chords.append(chord)
        return chords

    def get_supported_chord_dicts(self) -> List[str]:
        return ['full', 'ismir2017', 'submission', 'extended']

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Chord-CNN-LSTM (Replicate GPU)",
            "description": "GPU-accelerated chord recognition via Replicate API",
            "supported_chord_dicts": self.get_supported_chord_dicts(),
            "available": self.is_available(),
            "replicate_model": self.MODEL_ID,
        }
