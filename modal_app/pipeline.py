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
        # Lyrics transcription — 1.1.0+ adds BatchedInferencePipeline
        "faster-whisper>=1.1.0",
        # Chord-CNN-LSTM deps — 0.10.2+ fixes scipy.signal.hann removed in scipy 1.13
        "librosa>=0.10.2,<0.11",
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
#
# Whisper: distil-medium.en is ~280MB (vs 600MB large-v3) and 2-3× faster
# on T4. English-only — fine for our use case since we pin language="en"
# anyway. Quality drop on song lyrics is minimal; large-v3 was overkill.
WHISPER_MODEL = "Systran/faster-distil-whisper-medium.en"

image = image.run_commands(
    # Demucs htdemucs weights
    "python -c \"from demucs.pretrained import get_model; get_model('htdemucs')\"",
    # Faster-whisper distil-large-v3 weights
    f"python -c \"from faster_whisper import WhisperModel; "
    f"WhisperModel('{WHISPER_MODEL}', device='cpu', compute_type='int8')\"",
)


app = modal.App("riff-pipeline", image=image)


# ---------------------------------------------------------------------------
# Pipeline class — one container holds all 3 models
# ---------------------------------------------------------------------------

@app.cls(
    gpu="T4",                    # 16GB VRAM — plenty for demucs+whisper+chord
    enable_memory_snapshot=True, # snapshot CPU state, restore on cold start
    # 60s scaledown: memory snapshots make cold restart cheap (~10-15s) so
    # there's no reason to pay for long warm idle. Each unbilled minute of
    # idle = ~$0.01 of T4 time we don't owe Modal. Drop to 60s and rely on
    # snapshots for "warm enough" cold starts.
    scaledown_window=60,
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
        print(f"[setup] Loading whisper ({WHISPER_MODEL}) on CPU placeholder...")
        self.whisper_cpu = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        self.whisper_gpu = None  # populated post-snapshot

        # 3. Chord-CNN-LSTM — pre-load all 5 ensemble nets so each analyze()
        #    call doesn't re-instantiate them. The original chord_recognition
        #    function builds 5 NetworkInterfaces per call, which prints
        #    "moved to cuda" 5× per request and wastes a few seconds.
        chord_dir = "/root/chord_cnn_lstm"
        if chord_dir not in sys.path:
            sys.path.insert(0, chord_dir)
        self.chord_dir = chord_dir

        # Importing chord_recognition has the side effect of resolving
        # MODEL_NAMES (the 5 checkpoint paths). Use those to build cached nets.
        os.chdir(chord_dir)  # so relative cache_data/ paths resolve
        from chord_recognition import MODEL_NAMES  # noqa: E402
        from chordnet_ismir_naive import ChordNet  # noqa: E402
        from mir.nn.train import NetworkInterface  # noqa: E402

        print(f"[setup] Loading {len(MODEL_NAMES)} chord ensemble nets on CPU...")
        self.chord_nets = [
            NetworkInterface(ChordNet(None), name, load_checkpoint=False)
            for name in MODEL_NAMES
        ]
        os.chdir("/")

        # ---- Warm librosa's numba JIT ----
        # librosa.beat.beat_track relies on numba-compiled functions that
        # take 10-30s to JIT-compile on first call. Running them here
        # (pre-snapshot) bakes the compiled state into the Python process
        # memory, which Modal captures in the CPU snapshot. Every restored
        # container starts with librosa fully warmed up.
        print("[setup] Warming librosa numba JIT...")
        import wave
        import librosa
        warmup_wav = "/tmp/_riff_librosa_warmup.wav"
        with wave.open(warmup_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            # 2 seconds of barely-audible tone so beat_track has something
            # non-degenerate to chew on (pure silence can shortcut).
            import struct
            samples = bytearray()
            for i in range(22050 * 2):
                samples.extend(struct.pack("<h", (i % 100) - 50))
            wf.writeframes(bytes(samples))
        y, sr = librosa.load(warmup_wav, sr=22050, mono=True)
        _ = librosa.beat.beat_track(y=y, sr=sr)
        print("[setup] librosa numba JIT warmed.")

        print("[setup] CPU-side load complete. Snapshot will be taken now.")

    @modal.enter(snap=False)
    def move_to_gpu(self):
        """Runs every container start (cold OR after snapshot restore).
        Moves models from CPU memory to GPU. Whisper is re-instantiated
        directly on GPU since CTranslate2's device is fixed at construction.
        """
        from faster_whisper import WhisperModel, BatchedInferencePipeline

        print("[gpu-init] Moving demucs to CUDA...")
        self.demucs_model.to("cuda")

        print(f"[gpu-init] Re-instantiating whisper ({WHISPER_MODEL}) on CUDA...")
        whisper_model = WhisperModel(
            WHISPER_MODEL, device="cuda", compute_type="float16"
        )
        self.whisper_gpu = BatchedInferencePipeline(model=whisper_model)
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

    def _detect_beats(self, audio_path: str) -> dict:
        """Librosa beat detection. Runs on CPU so it parallelizes with
        demucs (which is GPU-bound) and chord+whisper after.

        Returns the same shape the backend route used to build itself.
        """
        import librosa

        # 22050 Hz mono is the standard rate for beat tracking — small,
        # fast, and accurate. Cap at 6 minutes so a sneaky-long song
        # can't OOM the container.
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=360)
        duration = float(librosa.get_duration(y=y, sr=sr))
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr).tolist()
        return {
            "beats": beat_times,
            "downbeats": beat_times[::4],
            "bpm": float(tempo),
            "duration": duration,
            "time_signature": "4/4",
        }

    def _transcribe_lyrics(self, vocals_path: str) -> list[dict]:
        """faster-whisper word-timestamped transcription on the vocals stem.

        Uses BatchedInferencePipeline (set up in move_to_gpu) with greedy
        decoding (beam_size=1) for max speed. Distil-large-v3 with batched
        inference + greedy decoding hits ~5-8s on T4 for a 3-min song,
        down from ~30s with standard transcribe().

        We pin language="en" because distil-large-v3 is English-only and
        crashes with `max() arg is an empty sequence` if asked to detect.
        """
        segments, _info = self.whisper_gpu.transcribe(
            vocals_path,
            language="en",
            word_timestamps=True,
            batch_size=16,                       # parallel chunks on GPU
            beam_size=1,                         # greedy decoding (2-3x faster than beam=5)
            condition_on_previous_text=False,    # skip slow context-tracking step
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
        """Run Chord-CNN-LSTM on the accompaniment stem using the 5 cached
        ensemble nets pre-loaded in load_on_cpu.

        We bypass the original chord_recognition() wrapper because it
        builds 5 NetworkInterfaces per call (which re-loads weights + moves
        them to CUDA every single time). Here we replicate the same logic
        but with the cached nets — saves ~3-5s per call and quiets the
        "moved to cuda" spam.
        """
        import numpy as np

        original_cwd = os.getcwd()
        try:
            os.chdir(self.chord_dir)
            # All these imports resolve from /root/chord_cnn_lstm (in sys.path).
            from mir import io, DataEntry  # noqa: E402
            from extractors.cqt import CQTV2  # noqa: E402
            from extractors.xhmm_ismir import XHMMDecoder  # noqa: E402
            from io_new.chordlab_io import ChordLabIO  # noqa: E402
            from settings import DEFAULT_SR, DEFAULT_HOP_LENGTH  # noqa: E402

            # Build the audio entry + CQT features (same as original)
            entry = DataEntry()
            entry.prop.set("sr", DEFAULT_SR)
            entry.prop.set("hop_length", DEFAULT_HOP_LENGTH)
            try:
                entry.append_file(audio_path, io.MusicIO, "music")
            except Exception:
                import librosa
                y, sr = librosa.load(audio_path, sr=DEFAULT_SR)
                entry.music = y
                entry.prop.set("sr", sr)
            entry.append_extractor(CQTV2, "cqt")

            # Inference with each cached net — no re-loading, no re-moving
            # to CUDA. The first call per container does the GPU migration
            # inside NetworkInterface; subsequent calls reuse it.
            probs_list = []
            for net in self.chord_nets:
                try:
                    probs_list.append(net.inference(entry.cqt))
                except Exception as e:
                    print(f"[chord] net {net} inference failed: {e}")
            if not probs_list:
                return []

            # Average probabilities across the 5 nets
            avg_probs = [
                np.mean([p[i] for p in probs_list], axis=0)
                for i in range(len(probs_list[0]))
            ]

            # HMM decode
            template_file = os.path.join(
                self.chord_dir, "data", f"{chord_dict}_chord_list.txt"
            )
            hmm = XHMMDecoder(template_file=template_file)
            chordlab = hmm.decode_to_chordlab(entry, avg_probs, False)

            # Normalize directly — no need to round-trip through a .lab file.
            return [
                {
                    "start": round(float(c[0]), 3),
                    "end": round(float(c[1]), 3),
                    "chord": c[2],
                    "confidence": 1.0,
                }
                for c in chordlab
            ]
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

            # Pre-decode to 44.1kHz stereo WAV with ffmpeg. Skips librosa's
            # audioread fallback (which decoded m4a in ~30s) and gives
            # demucs its native format — both downstream consumers load
            # this in < 100ms.
            import subprocess
            wav_path = os.path.join(workdir, "in.wav")
            t_decode = time.time()
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", in_path,
                 "-ac", "2", "-ar", "44100", "-acodec", "pcm_s16le", wav_path],
                check=True,
            )
            print(f"[analyze] Decode to WAV: {time.time() - t_decode:.1f}s")

            # Stage 1: demucs (GPU) || librosa beats (CPU)
            # Different compute resources, true parallelism. Both read the
            # decoded WAV; beats finishes in ~1-2s, demucs in ~3-5s, so
            # demucs bounds the wall time.
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_stems = pool.submit(self._separate_stems, wav_path, workdir)
                fut_beats = pool.submit(self._detect_beats, wav_path)
                stems = fut_stems.result()
                beats_result = fut_beats.result()
            print(f"[analyze] Demucs+Beats (parallel): {time.time() - t0:.1f}s")

            # Stage 2: chord recognition (on accompaniment) || whisper (on vocals)
            with ThreadPoolExecutor(max_workers=2) as pool:
                t1 = time.time()
                fut_chords = pool.submit(self._recognize_chords, stems["other_path"], chord_dict)
                fut_lyrics = pool.submit(self._transcribe_lyrics, stems["vocals_path"])
                chords = fut_chords.result()
                lyrics = fut_lyrics.result()
                print(f"[analyze] Chord+Whisper (parallel): {time.time() - t1:.1f}s")

        # Prefer chord-derived duration when chords are detected, otherwise
        # fall back to librosa's measurement.
        duration = chords[-1]["end"] if chords else beats_result["duration"]
        total = time.time() - t_start
        print(f"[analyze] Total: {total:.1f}s — "
              f"{len(chords)} chords, {len(lyrics)} words, "
              f"{len(beats_result['beats'])} beats, BPM {beats_result['bpm']:.1f}")

        return {
            "success": True,
            "chords": chords,
            "lyrics": lyrics,
            "beats": beats_result["beats"],
            "downbeats": beats_result["downbeats"],
            "bpm": beats_result["bpm"],
            "time_signature": beats_result["time_signature"],
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
