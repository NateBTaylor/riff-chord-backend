"""
Audio stem separation service using Demucs (Meta/Facebook Research).

Tries Replicate GPU first (~60s) for fast vocal separation,
falls back to local CPU Demucs (~2-5min) if unavailable.

The public interface (extract_vocals, cleanup_stems, etc.) is unchanged so that
chord_recognition_service.py and the analyze endpoint work without modification.
"""

import os
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional, List
from utils.logging import log_info, log_error, log_debug


class SpleeterService:
    """
    Audio source separation service.

    Uses Replicate GPU when REPLICATE_API_TOKEN is set (fast, ~60s),
    otherwise falls back to local Demucs CPU (slow, ~2-5min).
    """

    def __init__(self):
        """Initialize the separation service."""
        self._local_available = None
        self._replicate_available = None
        self._model = None

    def is_available(self) -> bool:
        """Check if any separation method is available."""
        return self._check_replicate() or self._check_local()

    def _check_replicate(self) -> bool:
        """Check if Replicate API is available for GPU separation."""
        if self._replicate_available is not None:
            return self._replicate_available

        try:
            import replicate  # noqa: F401
            token = os.environ.get('REPLICATE_API_TOKEN')
            self._replicate_available = bool(token)
            if self._replicate_available:
                log_info("Replicate API available for GPU stem separation")
            return self._replicate_available
        except ImportError:
            self._replicate_available = False
            return False

    def _check_local(self) -> bool:
        """Check if local Demucs is available."""
        if self._local_available is not None:
            return self._local_available

        try:
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            from demucs.pretrained import get_model  # noqa: F401
            self._local_available = True
            log_info("Local Demucs stem separation is available")
            return True
        except ImportError as e:
            log_debug(f"Local Demucs not available: {e}")
            self._local_available = False
            return False

    def _get_model(self):
        """Lazy-load the local Demucs model on first use."""
        if self._model is not None:
            return self._model

        if not self._check_local():
            raise RuntimeError("Local Demucs is not available")

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

    # ------------------------------------------------------------------
    # Replicate GPU separation
    # ------------------------------------------------------------------

    def _extract_vocals_replicate(self, audio_path: str,
                                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract vocals using Replicate's GPU-backed Demucs (~60s)."""
        start_time = time.time()
        temp_dir_created = False

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="demucs_rep_")
            temp_dir_created = True

        try:
            import replicate

            log_info("Running Demucs vocal separation via Replicate GPU...")

            with open(audio_path, 'rb') as f:
                output = replicate.run(
                    "cjwbw/demucs",
                    input={"audio": f},
                )

            vocals_path = os.path.join(output_dir, "vocals.wav")
            self._download_replicate_vocals(output, vocals_path)

            processing_time = time.time() - start_time
            log_info(f"Replicate vocal separation complete in {processing_time:.1f}s")

            return {
                "success": True,
                "vocals_path": vocals_path,
                "stems": {"vocals": vocals_path},
                "output_dir": output_dir,
                "model_used": "htdemucs (replicate GPU)",
                "processing_time": processing_time,
                "temp_dir_created": temp_dir_created,
            }

        except Exception as e:
            error_msg = f"Replicate separation failed: {str(e)}"
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
                "processing_time": time.time() - start_time,
            }

    @staticmethod
    def _download_replicate_vocals(output, vocals_path: str):
        """Download the vocals stem from Replicate output (handles multiple formats)."""
        # Case 1: dict with stem URLs (e.g. {"vocals": "https://...", "drums": ...})
        if isinstance(output, dict):
            vocals_url = output.get('vocals') or output.get('Vocals')
            if vocals_url:
                urllib.request.urlretrieve(str(vocals_url), vocals_path)
                return
            # If no vocals key, try first value
            for key, val in output.items():
                if 'vocal' in key.lower():
                    urllib.request.urlretrieve(str(val), vocals_path)
                    return
            raise ValueError(f"No vocals stem in output keys: {list(output.keys())}")

        # Case 2: string URL
        if isinstance(output, str):
            urllib.request.urlretrieve(output, vocals_path)
            return

        # Case 3: FileOutput with .read()
        if hasattr(output, 'read'):
            with open(vocals_path, 'wb') as f:
                f.write(output.read())
            return

        # Case 4: iterable of FileOutputs (multi-stem)
        try:
            items = list(output)
            # Find the vocals stem — typically the last one in [drums, bass, other, vocals]
            if len(items) >= 4:
                # Demucs order: drums, bass, other, vocals
                vocal_item = items[3]
            elif len(items) == 1:
                vocal_item = items[0]
            else:
                vocal_item = items[-1]

            if hasattr(vocal_item, 'read'):
                with open(vocals_path, 'wb') as f:
                    f.write(vocal_item.read())
            elif isinstance(vocal_item, str):
                urllib.request.urlretrieve(vocal_item, vocals_path)
            else:
                # Try URL conversion
                urllib.request.urlretrieve(str(vocal_item), vocals_path)
            return
        except (TypeError, IndexError):
            pass

        raise ValueError(f"Could not extract vocals from Replicate output: {type(output)}")

    # ------------------------------------------------------------------
    # Local CPU separation
    # ------------------------------------------------------------------

    def separate_audio(self, audio_path: str, model_name: str = 'htdemucs',
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Separate audio into stems using local Demucs CPU.

        Demucs produces 4 stems: drums, bass, other, vocals.
        For 2-stem mode, drums+bass+other are combined into 'accompaniment'.
        """
        if not self._check_local():
            return {
                "success": False,
                "error": "Local Demucs is not available",
                "model_used": "htdemucs",
            }

        start_time = time.time()
        temp_dir_created = False

        try:
            import torch
            import torchaudio
            from demucs.apply import apply_model

            model = self._get_model()

            if output_dir is None:
                output_dir = tempfile.mkdtemp(prefix="demucs_")
                temp_dir_created = True

            log_info(f"Running local Demucs separation on: {audio_path}")

            waveform, sr = torchaudio.load(audio_path)

            if sr != model.samplerate:
                log_debug(f"Resampling from {sr}Hz to {model.samplerate}Hz")
                waveform = torchaudio.functional.resample(
                    waveform, sr, model.samplerate
                )
                sr = model.samplerate

            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            ref = waveform.mean(0)
            ref_mean = ref.mean()
            ref_std = ref.std()
            waveform = (waveform - ref_mean) / (ref_std + 1e-8)

            with torch.no_grad():
                sources = apply_model(
                    model,
                    waveform[None],
                    split=True,
                    overlap=0.25,
                    device=torch.device("cpu"),
                )[0]

            sources = sources * ref_std + ref_mean

            stem_names = model.sources
            audio_name = Path(audio_path).stem

            stems = {}
            for i, name in enumerate(stem_names):
                stem_path = os.path.join(output_dir, f"{audio_name}_{name}.wav")
                torchaudio.save(stem_path, sources[i].cpu(), sr)
                stems[name] = stem_path

            accompaniment = sources[0] + sources[1] + sources[2]
            acc_path = os.path.join(output_dir, f"{audio_name}_accompaniment.wav")
            torchaudio.save(acc_path, accompaniment.cpu(), sr)
            stems["accompaniment"] = acc_path

            processing_time = time.time() - start_time
            log_info(f"Local Demucs separation complete: {len(stems)} stems in {processing_time:.1f}s")

            return {
                "success": True,
                "stems": stems,
                "output_dir": output_dir,
                "model_used": "htdemucs (local CPU)",
                "processing_time": processing_time,
                "temp_dir_created": temp_dir_created,
            }

        except Exception as e:
            error_msg = f"Local Demucs separation error: {str(e)}"
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_vocals(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract vocals from audio. Tries Replicate GPU first, falls back to local CPU.

        Returns dict with vocals_path and metadata.
        """
        # Try Replicate GPU first (~60s vs ~2-5min on CPU)
        if self._check_replicate():
            result = self._extract_vocals_replicate(audio_path, output_dir)
            if result.get("success"):
                return result
            log_error(f"Replicate failed, falling back to local Demucs: {result.get('error')}")

        # Fall back to local Demucs CPU
        result = self.separate_audio(audio_path, output_dir=output_dir)
        if result.get("success"):
            stems = result.get("stems", {})
            result["vocals_path"] = stems.get("vocals")
            result["accompaniment_path"] = stems.get("accompaniment")

        return result

    def extract_instruments(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract individual instruments (4-stem separation). Always uses local Demucs."""
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
            "replicate_gpu": self._check_replicate(),
            "local_cpu": self._check_local(),
            "models": {
                "htdemucs": {
                    "description": "Hybrid Transformer Demucs — high-quality 4-stem separation",
                    "stems": ["drums", "bass", "other", "vocals", "accompaniment"],
                }
            },
        }
