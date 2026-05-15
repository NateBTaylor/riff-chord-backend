"""
Modal app: combined chord + lyrics + stem-separation pipeline.

One container loads three models into memory and runs them as a single
analyze() call. Memory snapshots checkpoint the loaded state to disk, so
cold restarts skip the 60-90s model-load cost — they restore in 5-15s.

Pipeline (internal to the container):
    audio bytes → demucs (htdemucs)
                      ├── vocals.wav  → faster-whisper large-v3
                      └── other.wav   → chord-cnn-lstm
    → {chords, lyrics, duration}

Deploy:
    cd riff-chord-backend
    modal deploy modal_app/pipeline.py

Call from backend:
    f = modal.Function.lookup("riff-pipeline", "RiffPipeline.analyze")
    result = f.remote(audio_bytes)
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
# Chord-CNN-LSTM was written for Python ~3.7-3.9 + librosa 0.7-0.8. We pin
# numpy<2 because pumpp / older librosa break on numpy 2.x. PyTorch
# version matches what demucs + faster-whisper expect on CUDA 12.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        # Core
        "numpy>=1.23,<2.0",
        "scipy>=1.10,<1.14",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        # Stem separation
        "demucs==4.0.1",
        # Lyrics transcription
        "faster-whisper==1.0.3",
        # Chord-CNN-LSTM deps
        "librosa==0.10.1",
        "h5py>=3.0",
        "pretty_midi>=0.2.9",
        "mir_eval>=0.7",
        "pumpp==0.6.0",
        "jams==0.3.4",
        "joblib>=1.0",
        "pydub>=0.25",
        "scikit-learn>=1.2",
        "matplotlib>=3.5",
        # Server / utility
        "requests>=2.28",
    )
    # Bundle the local Chord-CNN-LSTM checkpoint + code into the image.
    .add_local_dir(
        "models/Chord-CNN-LSTM",
        "/root/chord_cnn_lstm",
        copy=True,
    )
)

# Pre-download demucs + whisper weights into the image so they're part of
# the snapshot (no runtime download on cold start).
image = image.run_commands(
    # Demucs htdemucs weights
    "python -c \"from demucs.pretrained import get_model; get_model('htdemucs')\"",
    # Faster-whisper large-v3 weights (CTranslate2-converted)
    "python -c \"from faster_whisper import WhisperModel; "
    "WhisperModel('large-v3', device='cpu', compute_type='int8')\"",
)


app = modal.App("riff-pipeline", image=image)


# ---------------------------------------------------------------------------
# Pipeline class — one container holds all 3 models
# ---------------------------------------------------------------------------

@app.cls(
    gpu="T4",                    # 16GB VRAM — plenty for demucs+whisper+chord
    enable_memory_snapshot=True, # snapshot CPU state, restore on cold start
    scaledown_window=600,        # stay warm 10 min after each call
    timeout=600,                 # 10 min per analyze call (generous)
    min_containers=0,            # scale to zero when idle ($0 idle cost)
    max_containers=4,            # cap on parallel scale-out
)
class RiffPipeline:
    """Combined demucs + chord-cnn-lstm + faster-whisper pipeline."""

    @modal.enter(snap=True)
    def load_on_cpu(self):
        """Runs once. Loads everything onto CPU — gets snapshotted to disk.
        Subsequent cold starts skip this step and restore from snapshot."""
        import sys
        import torch
        from demucs.pretrained import get_model
        from faster_whisper import WhisperModel

        # 1. Demucs — load weights to CPU
        print("[setup] Loading demucs htdemucs...")
        self.demucs_model = get_model("htdemucs")
        self.demucs_model.eval()
        self.demucs_sources = self.demucs_model.sources  # ['drums','bass','other','vocals']

        # 2. Faster-Whisper — load to CPU first; will re-instantiate on GPU
        #    in post-snapshot enter. (CTranslate2 doesn't support .to("cuda")
        #    after construction; the device is baked in.)
        print("[setup] Loading whisper large-v3 (CPU placeholder)...")
        self.whisper_cpu = WhisperModel("large-v3", device="cpu", compute_type="int8")
        self.whisper_gpu = None  # populated post-snapshot

        # 3. Chord-CNN-LSTM — make its module importable, but don't run the
        #    function yet (it loads weights lazily inside chord_recognition()).
        #    We bake the directory into sys.path so the import works from
        #    anywhere.
        chord_dir = "/root/chord_cnn_lstm"
        if chord_dir not in sys.path:
            sys.path.insert(0, chord_dir)
        self.chord_dir = chord_dir
        print("[setup] CPU-side load complete. Snapshot will be taken now.")

    @modal.enter(snap=False)
    def move_to_gpu(self):
        """Runs every container start (cold OR after snapshot restore).
        Moves models from CPU memory to GPU. ~3-5s for demucs, whisper
        is re-instantiated directly on GPU since CTranslate2 device is fixed."""
        import torch
        from faster_whisper import WhisperModel

        print("[gpu-init] Moving demucs to CUDA...")
        self.demucs_model.to("cuda")

        print("[gpu-init] Re-instantiating whisper on CUDA...")
        # WhisperModel can be re-created cheaply because the weights are
        # already cached in /root/.cache/huggingface from the image build.
        self.whisper_gpu = WhisperModel(
            "large-v3", device="cuda", compute_type="float16"
        )
        # Drop the CPU placeholder to free RAM.
        del self.whisper_cpu
        print("[gpu-init] Ready.")

    # ----------- Internal model wrappers -----------

    def _separate_stems(self, audio_path: str, out_dir: str) -> dict:
        """Run demucs on the audio file. Returns {vocals_path, other_path}.

        We use demucs.apply directly to avoid the demucs CLI overhead and
        keep control of the output paths.
        """
        import torch
        import torchaudio
        from demucs.apply import apply_model

        waveform, sr = torchaudio.load(audio_path)
        target_sr = self.demucs_model.samplerate
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
            sr = target_sr
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)  # demucs expects stereo

        # Normalize like demucs CLI does
        ref = waveform.mean(0)
        ref_mean, ref_std = ref.mean(), ref.std()
        waveform_norm = (waveform - ref_mean) / (ref_std + 1e-8)

        with torch.no_grad():
            sources = apply_model(
                self.demucs_model,
                waveform_norm[None].to("cuda"),
                split=True,
                overlap=0.25,
                device="cuda",
            )[0]
        sources = sources * ref_std + ref_mean  # un-normalize

        # Demucs source order: drums, bass, other, vocals
        vocals = sources[3]
        other = sources[0] + sources[1] + sources[2]  # drums+bass+other = accompaniment

        vocals_path = os.path.join(out_dir, "vocals.wav")
        other_path = os.path.join(out_dir, "other.wav")
        torchaudio.save(vocals_path, vocals.cpu(), sr)
        torchaudio.save(other_path, other.cpu(), sr)

        return {"vocals_path": vocals_path, "other_path": other_path}

    def _transcribe_lyrics(self, vocals_path: str) -> list[dict]:
        """faster-whisper word-timestamped transcription on the vocals stem."""
        segments, _info = self.whisper_gpu.transcribe(
            vocals_path,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                speech_pad_ms=600,
                threshold=0.15,
            ),
            initial_prompt="Song lyrics, verse and chorus:",
        )

        words: list[dict] = []
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
                    if prev.endswith((".", "?", "!")) or (
                        prev.endswith(",") and (w.start - words[-1]["end"]) > 1.0
                    ):
                        is_line_start = True
                words.append({
                    "word": text,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "is_line_start": is_line_start,
                })
                is_first = False
        return words

    def _recognize_chords(self, audio_path: str, chord_dict: str = "submission") -> list[dict]:
        """Run Chord-CNN-LSTM on the accompaniment stem.

        The model code expects to run from its own directory (it loads pkl
        weight files via relative paths). We chdir into it, call the
        chord_recognition() function which writes a .lab file, parse the
        .lab back into our normalized format, and chdir back.
        """
        import sys

        # The model module path lives in /root/chord_cnn_lstm and was
        # added to sys.path in load_on_cpu. Import lazily so we don't
        # interfere with snapshot creation.
        original_cwd = os.getcwd()
        try:
            os.chdir(self.chord_dir)
            from chord_recognition import chord_recognition  # noqa: E402

            lab_path = tempfile.mktemp(suffix=".lab")
            success = chord_recognition(audio_path, lab_path, chord_dict)
            if not success:
                return []
            chords = self._parse_lab(lab_path)
            try:
                os.unlink(lab_path)
            except OSError:
                pass
            return chords
        finally:
            os.chdir(original_cwd)

    @staticmethod
    def _parse_lab(lab_path: str) -> list[dict]:
        """Parse a tab- or whitespace-separated .lab file:  start\\tend\\tchord."""
        chords: list[dict] = []
        try:
            with open(lab_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split("\t") if "\t" in line else line.split()
                    if len(parts) < 3:
                        continue
                    try:
                        chords.append({
                            "start": round(float(parts[0]), 3),
                            "end": round(float(parts[1]), 3),
                            "chord": parts[2],
                            "confidence": 1.0,
                        })
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
        return chords

    # ----------- Public RPC method -----------

    @modal.method()
    def analyze(self, audio_bytes: bytes, chord_dict: str = "submission") -> dict[str, Any]:
        """Run the full pipeline on a song. Returns chords + lyrics + duration.

        Caller passes the raw audio bytes (any format ffmpeg can decode);
        we write to a temp file because demucs/whisper want a path on disk.

        Internal flow:
            1. write audio bytes → /tmp/in.audio
            2. demucs → vocals.wav + other.wav
            3. parallel:
                 - chord-cnn-lstm on other.wav
                 - whisper on vocals.wav
            4. return merged result
        """
        import time
        from concurrent.futures import ThreadPoolExecutor

        t_start = time.time()

        with tempfile.TemporaryDirectory(prefix="riff_") as workdir:
            in_path = os.path.join(workdir, "in.audio")
            with open(in_path, "wb") as f:
                f.write(audio_bytes)

            print(f"[analyze] Input audio: {len(audio_bytes) / 1024:.0f}KB")

            # 1. Demucs (~3-8s on T4 for a 3-min song)
            t0 = time.time()
            stems = self._separate_stems(in_path, workdir)
            print(f"[analyze] Demucs: {time.time() - t0:.1f}s")

            # 2. Chord + Whisper in parallel
            with ThreadPoolExecutor(max_workers=2) as pool:
                t1 = time.time()
                fut_chords = pool.submit(self._recognize_chords, stems["other_path"], chord_dict)
                fut_lyrics = pool.submit(self._transcribe_lyrics, stems["vocals_path"])
                chords = fut_chords.result()
                lyrics = fut_lyrics.result()
                print(f"[analyze] Chord+Whisper (parallel): {time.time() - t1:.1f}s")

        duration = chords[-1]["end"] if chords else 0.0
        total = time.time() - t_start
        print(f"[analyze] Total: {total:.1f}s — "
              f"{len(chords)} chords, {len(lyrics)} words")

        return {
            "success": True,
            "chords": chords,
            "lyrics": lyrics,
            "duration": duration,
            "processing_time": round(total, 2),
        }


# ---------------------------------------------------------------------------
# Local entry point for `modal run modal_app/pipeline.py --file audio.mp3`
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(file: str):
    """Smoke-test the pipeline locally. Usage:
        modal run modal_app/pipeline.py --file path/to/audio.mp3
    """
    p = Path(file)
    if not p.exists():
        raise SystemExit(f"File not found: {file}")
    audio_bytes = p.read_bytes()
    result = RiffPipeline().analyze.remote(audio_bytes)
    print(f"Got {len(result.get('chords', []))} chords, "
          f"{len(result.get('lyrics', []))} words, "
          f"duration {result.get('duration', 0):.1f}s, "
          f"processing_time {result.get('processing_time', 0)}s")
