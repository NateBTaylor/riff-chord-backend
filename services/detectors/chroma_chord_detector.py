"""
Fast chroma-based chord detector.

Uses librosa chroma features + chord template matching for real-time chord
recognition. Processes a 3-minute song in ~5-10 seconds on CPU.
Handles major, minor, and 7th chords — sufficient for most pop/rock songs.
"""

import time
import numpy as np
from typing import Dict, Any, List
from utils.logging import log_info, log_error, log_debug


# 12 pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Chord templates: each is a 12-element binary vector (pitch class profile)
# Intervals relative to root: major=[0,4,7], minor=[0,3,7], 7=[0,4,7,10], m7=[0,3,7,10]
_INTERVALS = {
    'maj': [0, 4, 7],
    'min': [0, 3, 7],
    '7':   [0, 4, 7, 10],
    'm7':  [0, 3, 7, 10],
}


def _build_templates() -> tuple:
    """Build chord name list and template matrix."""
    names = []
    templates = []
    for root_idx, root_name in enumerate(NOTE_NAMES):
        for quality, intervals in _INTERVALS.items():
            template = np.zeros(12, dtype=np.float32)
            for iv in intervals:
                template[(root_idx + iv) % 12] = 1.0
            template /= np.linalg.norm(template)
            if quality == 'maj':
                names.append(root_name)
            elif quality == 'min':
                names.append(f"{root_name}m")
            elif quality == '7':
                names.append(f"{root_name}7")
            elif quality == 'm7':
                names.append(f"{root_name}m7")
            templates.append(template)
    # Add "N" (no chord) — zero vector handled separately
    names.append('N')
    return names, np.array(templates)


CHORD_NAMES, CHORD_TEMPLATES = _build_templates()


class ChromaChordDetectorService:
    """
    Fast chord detector using chroma features and template matching.
    """

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import librosa
            self._available = True
            return True
        except ImportError:
            self._available = False
            return False

    def recognize_chords(self, file_path: str, chord_dict: str = 'submission',
                         **kwargs) -> Dict[str, Any]:
        """
        Recognize chords using chroma feature template matching.

        Returns the same normalized format as ChordCNNLSTMDetectorService.
        """
        if not self.is_available():
            return {"success": False, "error": "librosa not available",
                    "model_used": "chroma", "model_name": "Chroma"}

        start_time = time.time()
        try:
            import librosa

            log_info(f"Running chroma chord detection on: {file_path}")

            # Load audio at 22050 Hz (standard for music analysis)
            y, sr = librosa.load(file_path, sr=22050)
            duration = len(y) / sr

            # Compute chroma features using CQT (better for music than STFT)
            # hop_length=2048 gives ~10 frames/sec — good resolution, fast
            hop_length = 2048
            chroma = librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=hop_length, n_chroma=12
            )
            # chroma shape: (12, n_frames)
            n_frames = chroma.shape[1]
            frame_duration = hop_length / sr  # ~0.093s per frame

            log_info(f"Chroma: {n_frames} frames, {frame_duration:.3f}s/frame, "
                     f"duration: {duration:.1f}s")

            # Normalize each frame
            norms = np.linalg.norm(chroma, axis=0, keepdims=True)
            norms[norms < 1e-6] = 1.0
            chroma_norm = chroma / norms  # (12, n_frames)

            # Template matching: cosine similarity for each frame
            # similarities shape: (n_templates, n_frames)
            similarities = CHORD_TEMPLATES @ chroma_norm  # (n_templates, n_frames)

            # Detect low-energy frames as "N" (no chord)
            energy = np.sum(chroma, axis=0)
            energy_threshold = np.percentile(energy, 10) + 0.01

            # Pick best chord per frame
            best_idx = np.argmax(similarities, axis=0)
            best_score = np.max(similarities, axis=0)

            # Mark low-energy or low-confidence frames as N
            n_idx = len(CHORD_NAMES) - 1  # "N" index
            no_chord_mask = (energy < energy_threshold) | (best_score < 0.5)
            best_idx[no_chord_mask] = n_idx

            # Smooth with median filter (removes single-frame glitches)
            from scipy.ndimage import median_filter
            smoothed = median_filter(best_idx.astype(np.float64), size=7)
            smoothed = smoothed.astype(int)

            # Merge consecutive frames with the same chord into segments
            chords = []
            if n_frames > 0:
                seg_start = 0
                seg_chord = smoothed[0]
                for i in range(1, n_frames):
                    if smoothed[i] != seg_chord:
                        chord_name = CHORD_NAMES[seg_chord]
                        start_sec = seg_start * frame_duration
                        end_sec = i * frame_duration
                        if chord_name != 'N' and (end_sec - start_sec) >= 0.3:
                            chords.append({
                                "start": round(start_sec, 3),
                                "end": round(end_sec, 3),
                                "chord": chord_name,
                                "confidence": round(float(
                                    np.mean(best_score[seg_start:i])
                                ), 3),
                            })
                        seg_start = i
                        seg_chord = smoothed[i]
                # Final segment
                chord_name = CHORD_NAMES[seg_chord]
                start_sec = seg_start * frame_duration
                end_sec = n_frames * frame_duration
                if chord_name != 'N' and (end_sec - start_sec) >= 0.3:
                    chords.append({
                        "start": round(start_sec, 3),
                        "end": round(end_sec, 3),
                        "chord": chord_name,
                        "confidence": round(float(
                            np.mean(best_score[seg_start:n_frames])
                        ), 3),
                    })

            processing_time = time.time() - start_time
            log_info(f"Chroma chord detection: {len(chords)} chords in {processing_time:.1f}s")

            return {
                "success": True,
                "chords": chords,
                "total_chords": len(chords),
                "duration": round(duration, 3),
                "model_used": "chroma",
                "model_name": "Chroma",
                "chord_dict": chord_dict,
                "processing_time": processing_time,
            }

        except Exception as e:
            log_error(f"Chroma chord detection error: {e}")
            return {
                "success": False,
                "error": str(e),
                "model_used": "chroma",
                "model_name": "Chroma",
                "processing_time": time.time() - start_time,
            }

    def get_supported_chord_dicts(self) -> List[str]:
        return ['submission']

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": "Chroma",
            "description": "Fast chroma-based chord detection using template matching",
            "supported_chord_dicts": self.get_supported_chord_dicts(),
            "available": self.is_available(),
        }
