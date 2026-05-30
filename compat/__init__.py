"""
Compatibility patches for ChordMini Flask application.

This package contains patches for compatibility issues with:
- NumPy (deprecated attributes)
- SciPy (signal module changes)
- Madmom (collections.MutableSequence)
- Librosa (beat tracker issues)

Import order is critical - these patches must be applied before
importing the affected libraries.
"""

from .numpy_patch import patch_numpy_compatibility
from .scipy_patch import apply_scipy_patches
from .madmom_patch import patch_madmom_compatibility
from .librosa_patch import patch_librosa_beat_tracker, monkey_patch_beat_track
from utils.logging import log_debug, is_debug_enabled


_LIBROSA_PATCHES_APPLIED = False


def apply_all():
    """
    Apply the boot-time compatibility patches.

    Called early in startup. Intentionally excludes the librosa/scipy
    beat-tracking patches: those import librosa (which pulls in numba + scipy,
    ~150MB resident) purely to monkey-patch librosa.beat.beat_track, and that
    function only runs on the local beat-detection fallback. On this
    deployment the analyze pipeline goes through Modal by default, so loading
    librosa at boot just pins RAM that Railway bills by the GB-minute. The
    librosa patches are applied lazily via apply_librosa_patches() right
    before the fallback uses librosa — see services/detectors/librosa_detector.
    """
    # numpy + madmom patches are cheap (they don't import librosa/scipy) and
    # may be needed before any numpy/madmom use, so keep them at boot.
    patch_numpy_compatibility()
    patch_madmom_compatibility()

    if is_debug_enabled():
        log_debug("Boot compatibility patches applied (numpy, madmom)")


def apply_librosa_patches():
    """
    Apply the librosa/scipy beat-tracking compatibility patches.

    This imports librosa + scipy, so it must be called lazily — right before
    librosa.beat.beat_track runs on the local fallback path, never at boot.
    Idempotent: safe to call repeatedly; the heavy import happens only once.
    """
    global _LIBROSA_PATCHES_APPLIED
    if _LIBROSA_PATCHES_APPLIED:
        return
    # Order matters: scipy shim first, then the two librosa.beat patches.
    apply_scipy_patches()
    patch_librosa_beat_tracker()
    monkey_patch_beat_track()
    _LIBROSA_PATCHES_APPLIED = True

    if is_debug_enabled():
        log_debug("Librosa/scipy beat-tracking patches applied (lazy)")