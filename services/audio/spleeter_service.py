"""
Audio stem separation service using Demucs (Meta/Facebook Research).

Replaces the original Spleeter-based implementation with Demucs, which produces
significantly better separation quality. Uses the htdemucs model (Hybrid Transformer)
for best vocal/accompaniment isolation.

The public interface (extract_vocals, cleanup_stems, etc.) is unchanged so that
chord_recognition_service.py and the analyze endpoint work without modification.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from utils.logging import log_info, log_error, log_debug


class SpleeterService:
    """
    Audio source separation service backed by Demucs.

    Keeps the SpleeterService class name for backward compatibility with
    chord_recognition_service and the analyze endpoint.
    """

    def __init__(self):
        """Initialize the separation service."""
        self._available = None
        self._model = None

    def is_available(self) -> bool:
        """Check if Demucs is available."""
        if self._available is not None:
            return self._available

        try:
            import torch
            import torchaudio
            from demucs.pretrained import get_model
            self._available = True
            log_info("Demucs stem separation is available")
            return True
        except ImportError as e:
            log_error(f"Demucs import failed: {e}")
            self._available = False
            return False

    def _get_model(self):
        """Lazy-load the Demucs model on first use."""
        if self._model is not None:
            return self._model

        if not self.is_available():
            raise RuntimeError("Demucs is not available")

        import torch
        from demucs.pretrained import get_model

        log_info("Loading Demucs htdemucs model...")
        start = time.time()
        model = get_model("htdemucs")
        model.eval()
        model.to(torch.device("cpu"))
        self._model = model
        log_info(f"Demucs model loaded in {time.time() - start:.1f}s")
        return model

    def separate_audio(self, audio_path: str, model_name: str = 'htdemucs',
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Separate audio into stems using Demucs.

        Demucs produces 4 stems: drums, bass, other, vocals.
        For 2-stem mode, drums+bass+other are combined into 'accompaniment'.

        Args:
            audio_path: Path to the input audio file
            model_name: Ignored (always uses htdemucs), kept for interface compat
            output_dir: Output directory (if None, uses temporary directory)

        Returns:
            Dict with success, stems dict, output_dir, processing_time, etc.
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Demucs is not available",
                "model_used": "htdemucs",
            }

        start_time = time.time()
        temp_dir_created = False

        try:
            import torch
            import torchaudio
            from demucs.apply import apply_model

            model = self._get_model()

            # Create output directory
            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="demucs_")
                temp_dir_created = True

            log_info(f"Running Demucs separation on: {audio_path}")

            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Resample to model's sample rate if needed
            if sr != model.samplerate:
                log_debug(f"Resampling from {sr}Hz to {model.samplerate}Hz")
                waveform = torchaudio.functional.resample(
                    waveform, sr, model.samplerate
                )
                sr = model.samplerate

            # Ensure stereo
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            # Normalize
            ref = waveform.mean(0)
            ref_mean = ref.mean()
            ref_std = ref.std()
            waveform = (waveform - ref_mean) / (ref_std + 1e-8)

            # Run separation
            # split=True processes in chunks to limit memory usage
            with torch.no_grad():
                sources = apply_model(
                    model,
                    waveform[None],  # Add batch dimension
                    split=True,
                    overlap=0.25,
                    device=torch.device("cpu"),
                )[0]  # Remove batch dimension

            # Denormalize
            sources = sources * ref_std + ref_mean

            # model.sources = ['drums', 'bass', 'other', 'vocals']
            stem_names = model.sources
            audio_name = Path(audio_path).stem

            stems = {}
            for i, name in enumerate(stem_names):
                stem_path = os.path.join(output_dir, f"{audio_name}_{name}.wav")
                torchaudio.save(stem_path, sources[i].cpu(), sr)
                stems[name] = stem_path
                log_debug(f"Saved stem '{name}' to: {stem_path}")

            # Create combined accompaniment (drums + bass + other)
            accompaniment = sources[0] + sources[1] + sources[2]
            acc_path = os.path.join(output_dir, f"{audio_name}_accompaniment.wav")
            torchaudio.save(acc_path, accompaniment.cpu(), sr)
            stems["accompaniment"] = acc_path
            log_debug(f"Saved combined accompaniment to: {acc_path}")

            processing_time = time.time() - start_time
            log_info(f"Demucs separation complete: {len(stems)} stems in {processing_time:.1f}s")

            return {
                "success": True,
                "stems": stems,
                "output_dir": output_dir,
                "model_used": "htdemucs",
                "processing_time": processing_time,
                "temp_dir_created": temp_dir_created,
            }

        except Exception as e:
            error_msg = f"Demucs separation error: {str(e)}"
            log_error(error_msg)

            if temp_dir_created and output_dir:
                try:
                    import shutil
                    shutil.rmtree(output_dir)
                except Exception:
                    pass

            return {
                "success": False,
                "error": error_msg,
                "model_used": "htdemucs",
                "processing_time": time.time() - start_time,
            }

    def extract_vocals(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract vocals and accompaniment from audio.

        Returns dict with vocals_path (isolated vocals) and
        accompaniment_path (drums + bass + other instruments combined).
        """
        result = self.separate_audio(audio_path, output_dir=output_dir)

        if result.get("success"):
            stems = result.get("stems", {})
            result["vocals_path"] = stems.get("vocals")
            result["accompaniment_path"] = stems.get("accompaniment")

        return result

    def extract_instruments(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract individual instruments (4-stem separation).
        """
        result = self.separate_audio(audio_path, output_dir=output_dir)

        if result.get("success"):
            stems = result.get("stems", {})
            result["vocals_path"] = stems.get("vocals")
            result["drums_path"] = stems.get("drums")
            result["bass_path"] = stems.get("bass")
            result["other_path"] = stems.get("other")

        return result

    def cleanup_stems(self, stems_info: Dict[str, Any]) -> bool:
        """Clean up separated stem files."""
        try:
            if stems_info.get("temp_dir_created") and stems_info.get("output_dir"):
                import shutil
                shutil.rmtree(stems_info["output_dir"])
                log_debug(f"Cleaned up output directory: {stems_info['output_dir']}")
                return True
            elif stems_info.get("stems"):
                for stem_path in stems_info["stems"].values():
                    if os.path.exists(stem_path):
                        os.unlink(stem_path)
                return True
            return True
        except Exception as e:
            log_error(f"Failed to cleanup stems: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        if not self.is_available():
            return []
        return ["htdemucs"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about separation models."""
        return {
            "available": self.is_available(),
            "backend": "demucs",
            "models": {
                "htdemucs": {
                    "description": "Hybrid Transformer Demucs — high-quality 4-stem separation (44.1kHz)",
                    "stems": ["drums", "bass", "other", "vocals", "accompaniment"],
                }
            },
        }
