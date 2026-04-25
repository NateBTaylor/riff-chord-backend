import os
import numpy as np
import librosa
import warnings
import tempfile

# Force any GPU-based processing to use CPU to avoid memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Try to import Spleeter, but don't fail if it's not available
SPLEETER_AVAILABLE = False
try:
    # Force import to check if Spleeter is really available
    import spleeter
    from spleeter.separator import Separator
    from spleeter.audio.adapter import AudioAdapter

    # Try to get version, but don't fail if not available
    try:
        version = spleeter.__version__
    except:
        version = "unknown"

    SPLEETER_AVAILABLE = True
    print(f"Spleeter is available (version {version}) and will be used for audio separation")
except Exception as e:
    print(f"Spleeter import error: {e}")
    warnings.warn(f"Spleeter not available: {e}. Will use simple spectrogram instead.")

def demix_audio_to_spectrogram(audio_file, output_file, sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000):
    """
    Generate a demixed spectrogram from an audio file using Spleeter
    Falls back to a simple spectrogram if Spleeter is not available or fails
    """
    print(f"demix_audio_to_spectrogram called with:")
    print(f"  - audio_file: {audio_file}")
    print(f"  - output_file: {output_file}")
    print(f"  - sr: {sr}")
    print(f"  - Spleeter available: {SPLEETER_AVAILABLE}")

    try:
        if SPLEETER_AVAILABLE:
            # Try with Spleeter first
            print("Attempting to use Spleeter for demixing")
            return demix_with_spleeter(audio_file, output_file, sr, n_fft, n_mels, fmin, fmax)
        else:
            # Use fallback if Spleeter not available
            print("Spleeter not available, using simple spectrogram")
            return simple_spectrogram(audio_file, output_file, sr, n_fft, n_mels, fmin, fmax)
    except Exception as e:
        print(f"Error in Spleeter demixing: {e}. Falling back to simple spectrogram.")
        return simple_spectrogram(audio_file, output_file, sr, n_fft, n_mels, fmin, fmax)

def demix_with_spleeter(audio_file, output_file, sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000):
    """Enhanced Spleeter-based implementation with better error handling"""
    try:
        print(f"Starting Spleeter separation for {audio_file}")

        # Create a temporary directory for Spleeter output
        temp_dir = tempfile.mkdtemp()
        print(f"Created temporary directory for Spleeter: {temp_dir}")

        # Load audio using Spleeter's adapter
        audio_loader = AudioAdapter.default()
        waveform, _ = audio_loader.load(audio_file, sample_rate=sr)
        print(f"Loaded audio with shape: {waveform.shape}")

        # Initialize Spleeter for 5-stems demixing
        print("Initializing Spleeter 5-stems separator")
        separator = Separator('spleeter:5stems')

        # Separate the audio
        print("Separating audio with Spleeter...")
        demixed = separator.separate(waveform)
        print(f"Separation complete. Got {len(demixed)} stems: {list(demixed.keys())}")

        # Create Mel filter bank
        mel_f = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T

        # Process each stem
        spectrograms = []
        for key in demixed:
            print(f"Processing stem: {key}")

            # FIXED: Use librosa with exact Beat-Transformer parameters instead of Spleeter's STFT
            # Beat-Transformer was trained with hop_length=1024 (44100/1024 = 43.07 fps)
            # This ensures consistent timing with the model's expectations
            stem_audio = demixed[key]
            if len(stem_audio.shape) > 1:
                # Convert to mono if stereo
                stem_audio = np.mean(stem_audio, axis=1)

            # Use librosa mel spectrogram with Beat-Transformer parameters
            mel_spec = librosa.feature.melspectrogram(
                y=stem_audio,
                sr=sr,
                n_fft=n_fft,
                hop_length=1024,  # CRITICAL: Match Beat-Transformer training (44100/1024 fps)
                n_mels=n_mels,
                fmin=fmin,
                fmax=fmax
            )
            print(f"Mel spec shape for {key}: {mel_spec.shape}")

            # Convert to dB scale
            spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            print(f"Final spec shape for {key}: {spec_db.shape}")

            # Ensure the spectrogram has the correct shape (time, frequency)
            if spec_db.shape[0] == n_mels:
                spec_db = spec_db.T
                print(f"Transposed spec shape: {spec_db.shape}")

            spectrograms.append(spec_db)

        # Stack all channel spectrograms (shape: num_channels x time x mel_bins)
        demixed_spec = np.stack(spectrograms, axis=0)
        print(f"Final stacked spectrogram shape: {demixed_spec.shape}")

        # Save the result
        np.save(output_file, demixed_spec)
        print(f"Demixed spectrogram saved to {output_file}")

        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temporary directory: {temp_dir}")

        return True
    except Exception as e:
        print(f"Error in Spleeter processing: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to simple spectrogram")
        return simple_spectrogram(audio_file, output_file, sr, n_fft, n_mels, fmin, fmax)

def simple_spectrogram(audio_file, output_file, sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000):
    """
    Create a simplified spectrogram for Beat-Transformer without using Spleeter
    Creates a stack of 5 copies of the same spectrogram to match the 5-stem format
    """
    print(f"Using simple spectrogram for {audio_file}")
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_file, sr=sr)

        # Create mel spectrogram with Beat-Transformer parameters
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=1024, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Ensure the spectrogram has the correct shape for the model
        # The model expects (batch, instr, time, mel_bins)
        # For the spectrogram, we need (instr, time, mel_bins)

        # First, make sure we have the frequency bins as the second dimension
        if spec_db.shape[0] == n_mels:
            # If first dimension is n_mels, we need to transpose
            spec_db = spec_db.T  # Now shape is (time, frequency)

        # Print shape information for debugging
        print(f"Original spectrogram shape: {spec_db.shape}")

        # Create a stack of 5 copies to simulate the 5-stem format
        # Shape will be (5, time, frequency)
        stacked_spec = np.stack([spec_db] * 5, axis=0)

        print(f"Stacked spectrogram shape: {stacked_spec.shape}")

        # Save the result
        np.save(output_file, stacked_spec)
        print(f"Simple spectrogram saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error creating simple spectrogram: {e}")
        raise

if __name__ == "__main__":
    # Use your own audio file path here
    AUDIO_FILE = "test_audio/ocean.mp3"
    OUTPUT_FILE = "./demixed_spectrogram.npy"
    demix_audio_to_spectrogram(AUDIO_FILE, OUTPUT_FILE)
