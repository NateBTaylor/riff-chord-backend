"""
Force Modal to create the riff-pipeline snapshot right after deploy, so
the first real user request doesn't pay the ~60s snapshot-creation cost.

Run this immediately after `modal deploy modal_app/pipeline.py`:

    python modal_app/warm_snapshot.py

The first time this is called against a newly-deployed app, Modal has
to run setup() in full (loading models, warming numba JIT) before
taking the snapshot. That takes ~60-90s — but it happens HERE rather
than blocking a real user.

After this completes, every subsequent cold-from-snapshot container
restore is ~5-10s.
"""

import io
import struct
import time
import wave
from pathlib import Path


def _silence_audio_bytes(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a tiny silent WAV in-memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * int(sample_rate * seconds))
    return buf.getvalue()


def main() -> int:
    try:
        import modal
    except ImportError:
        print("ERROR: `modal` package not installed. Run `pip install modal` first.")
        return 1

    print("Looking up deployed RiffPipeline...")
    try:
        cls = modal.Cls.from_name("riff-pipeline", "RiffPipeline")
    except Exception as e:
        print(f"ERROR: could not find deployed riff-pipeline: {e}")
        print("Did you run `modal deploy modal_app/pipeline.py` first?")
        return 1

    print("Triggering snapshot creation (this is the slow ~60-90s part)...")
    print("  → If snapshot exists, this returns in ~10s.")
    print("  → If snapshot needs to be built, this blocks until done.")
    audio = _silence_audio_bytes(seconds=1.0)
    t0 = time.time()
    try:
        result = cls().analyze.remote(audio)
        elapsed = time.time() - t0
        print(f"\n✓ Done in {elapsed:.1f}s")
        chords = len(result.get("chords", []))
        words = len(result.get("lyrics", []))
        print(f"   ({chords} chords, {words} lyric words on silence — both should be ~0)")
        if elapsed > 30:
            print("\nSnapshot was just created. Future cold starts will be ~5-10s.")
        else:
            print("\nLooks like the snapshot was already warm. You're good to go.")
        return 0
    except Exception as e:
        print(f"\n✗ Modal call failed: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
