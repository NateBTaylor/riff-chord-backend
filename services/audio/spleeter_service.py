"""
Audio stem separation service.

Priority order for vocal extraction:
1. Replicate Spleeter  — $0.00022/run, ~1s   (cheapest + fastest)
2. Replicate Demucs     — $0.023/run,  ~60s   (best quality)
3. Local Demucs CPU     — free,        ~2-5min (no API needed)
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
    Audio source separation service with multiple backends.
    """

    def __init__(self):
        self._local_available = None
        self._replicate_available = None
        self._model = None

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
            import torch  # noqa: F401
            import torchaudio  # noqa: F401
            from demucs.pretrained import get_model  # noqa: F401
            self._local_available = True
            return True
        except ImportError:
            self._local_available = False
            return False

    # ------------------------------------------------------------------
    # Replicate Spleeter (~$0.00022, ~1s)
    # ------------------------------------------------------------------

    def _extract_vocals_spleeter(self, audio_path: str,
                                 output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract vocals using Replicate Spleeter (cheapest: ~$0.00022, ~1s)."""
        start_time = time.time()
        temp_dir_created = False

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="spleeter_")
            temp_dir_created = True

        try:
            import replicate

            log_info("Separating vocals via Replicate Spleeter...")

            with open(audio_path, 'rb') as f:
                output = replicate.run(
                    "soykertje/spleeter",
                    input={"audio": f},
                )

            vocals_path = os.path.join(output_dir, "vocals.wav")
            self._download_stem(output, "vocals", vocals_path)

            processing_time = time.time() - start_time
            log_info(f"Spleeter vocal separation: {processing_time:.1f}s")

            return {
                "success": True,
                "vocals_path": vocals_path,
                "stems": {"vocals": vocals_path},
                "output_dir": output_dir,
                "model_used": "spleeter (replicate)",
                "processing_time": processing_time,
                "temp_dir_created": temp_dir_created,
            }

        except Exception as e:
            error_msg = f"Replicate Spleeter failed: {str(e)}"
            log_error(error_msg)
            if temp_dir_created and output_dir:
                try:
                    import shutil
                    shutil.rmtree(output_dir)
                except Exception:
                    pass
            return {"success": False, "error": error_msg,
                    "processing_time": time.time() - start_time}

    # ------------------------------------------------------------------
    # Replicate Demucs (~$0.023, ~60s)
    # ------------------------------------------------------------------

    def _extract_vocals_demucs_replicate(self, audio_path: str,
                                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Extract vocals using Replicate Demucs (higher quality: ~$0.023, ~60s)."""
        start_time = time.time()
        temp_dir_created = False

        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="demucs_rep_")
            temp_dir_created = True

        try:
            import replicate

            log_info("Separating vocals via Replicate Demucs...")

            with open(audio_path, 'rb') as f:
                output = replicate.run(
                    "cjwbw/demucs",
                    input={"audio": f},
                )

            vocals_path = os.path.join(output_dir, "vocals.wav")
            self._download_stem(output, "vocals", vocals_path)

            processing_time = time.time() - start_time
            log_info(f"Demucs vocal separation: {processing_time:.1f}s")

            return {
                "success": True,
                "vocals_path": vocals_path,
                "stems": {"vocals": vocals_path},
                "output_dir": output_dir,
                "model_used": "htdemucs (replicate)",
                "processing_time": processing_time,
                "temp_dir_created": temp_dir_created,
            }

        except Exception as e:
            error_msg = f"Replicate Demucs failed: {str(e)}"
            log_error(error_msg)
            if temp_dir_created and output_dir:
                try:
                    import shutil
                    shutil.rmtree(output_dir)
                except Exception:
                    pass
            return {"success": False, "error": error_msg,
                    "processing_time": time.time() - start_time}

    # ------------------------------------------------------------------
    # Local Demucs CPU (free, ~2-5min)
    # ------------------------------------------------------------------

    def _get_model(self):
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

    def separate_audio(self, audio_path: str, model_name: str = 'htdemucs',
                       output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Full 4-stem separation using local Demucs CPU."""
        if not self._check_local():
            return {"success": False, "error": "Local Demucs not available"}

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

            log_info(f"Running local Demucs on: {audio_path}")
            waveform, sr = torchaudio.load(audio_path)

            if sr != model.samplerate:
                waveform = torchaudio.functional.resample(waveform, sr, model.samplerate)
                sr = model.samplerate
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)

            ref = waveform.mean(0)
            ref_mean, ref_std = ref.mean(), ref.std()
            waveform = (waveform - ref_mean) / (ref_std + 1e-8)

            with torch.no_grad():
                sources = apply_model(model, waveform[None], split=True,
                                      overlap=0.25, device=torch.device("cpu"))[0]
            sources = sources * ref_std + ref_mean

            audio_name = Path(audio_path).stem
            stems = {}
            for i, name in enumerate(model.sources):
                p = os.path.join(output_dir, f"{audio_name}_{name}.wav")
                torchaudio.save(p, sources[i].cpu(), sr)
                stems[name] = p

            acc = sources[0] + sources[1] + sources[2]
            acc_path = os.path.join(output_dir, f"{audio_name}_accompaniment.wav")
            torchaudio.save(acc_path, acc.cpu(), sr)
            stems["accompaniment"] = acc_path

            processing_time = time.time() - start_time
            log_info(f"Local Demucs: {len(stems)} stems in {processing_time:.1f}s")

            return {
                "success": True, "stems": stems, "output_dir": output_dir,
                "model_used": "htdemucs (local CPU)",
                "processing_time": processing_time, "temp_dir_created": temp_dir_created,
            }
        except Exception as e:
            log_error(f"Local Demucs error: {e}")
            if temp_dir_created and output_dir:
                try:
                    import shutil
                    shutil.rmtree(output_dir)
                except Exception:
                    pass
            return {"success": False, "error": str(e),
                    "processing_time": time.time() - start_time}

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _download_stem(output, stem_name: str, dest_path: str):
        """Download a specific stem from Replicate output (handles multiple formats)."""
        # dict with stem URLs
        if isinstance(output, dict):
            for key, val in output.items():
                if stem_name.lower() in key.lower():
                    urllib.request.urlretrieve(str(val), dest_path)
                    return
            raise ValueError(f"No '{stem_name}' in output: {list(output.keys())}")

        # string URL
        if isinstance(output, str):
            urllib.request.urlretrieve(output, dest_path)
            return

        # FileOutput
        if hasattr(output, 'read'):
            with open(dest_path, 'wb') as f:
                f.write(output.read())
            return

        # iterable of FileOutputs / URLs (multi-stem)
        try:
            items = list(output)
            # Spleeter 2-stem: [vocals, accompaniment] or [accompaniment, vocals]
            # Demucs 4-stem: [drums, bass, other, vocals]
            target = None
            for item in items:
                item_str = str(item).lower()
                if stem_name.lower() in item_str:
                    target = item
                    break
            if target is None:
                # Spleeter typically returns vocals first for 2-stem
                if stem_name == "vocals":
                    target = items[0] if len(items) >= 1 else None
                elif len(items) >= 4:
                    # Demucs order: drums, bass, other, vocals
                    target = items[3]
                else:
                    target = items[-1]

            if target is not None:
                if hasattr(target, 'read'):
                    with open(dest_path, 'wb') as f:
                        f.write(target.read())
                else:
                    urllib.request.urlretrieve(str(target), dest_path)
                return
        except (TypeError, IndexError):
            pass

        raise ValueError(f"Could not extract '{stem_name}' from output: {type(output)}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_vocals(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract vocals. Priority: Spleeter (cheap) → Demucs Replicate → local Demucs.
        """
        if self._check_replicate():
            # Try Spleeter first (cheapest: $0.00022, ~1s)
            result = self._extract_vocals_spleeter(audio_path, output_dir)
            if result.get("success"):
                return result
            log_error(f"Spleeter failed, trying Demucs: {result.get('error')}")

            # Try Replicate Demucs (better quality: $0.023, ~60s)
            result = self._extract_vocals_demucs_replicate(audio_path, output_dir)
            if result.get("success"):
                return result
            log_error(f"Replicate Demucs failed, trying local: {result.get('error')}")

        # Fall back to local Demucs CPU
        if self._check_local():
            result = self.separate_audio(audio_path, output_dir=output_dir)
            if result.get("success"):
                stems = result.get("stems", {})
                result["vocals_path"] = stems.get("vocals")
                result["accompaniment_path"] = stems.get("accompaniment")
            return result

        return {"success": False, "error": "No separation method available"}

    def extract_instruments(self, audio_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Full 4-stem separation (always local Demucs)."""
        result = self.separate_audio(audio_path, output_dir=output_dir)
        if result.get("success"):
            stems = result.get("stems", {})
            result["vocals_path"] = stems.get("vocals")
            result["drums_path"] = stems.get("drums")
            result["bass_path"] = stems.get("bass")
            result["other_path"] = stems.get("other")
        return result

    def cleanup_stems(self, stems_info: Dict[str, Any]) -> bool:
        try:
            if stems_info.get("temp_dir_created") and stems_info.get("output_dir"):
                import shutil
                shutil.rmtree(stems_info["output_dir"])
                return True
            elif stems_info.get("stems"):
                for p in stems_info["stems"].values():
                    if os.path.exists(p):
                        os.unlink(p)
            return True
        except Exception as e:
            log_error(f"Failed to cleanup stems: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "available": self.is_available(),
            "replicate_gpu": self._check_replicate(),
            "local_cpu": self._check_local(),
        }
