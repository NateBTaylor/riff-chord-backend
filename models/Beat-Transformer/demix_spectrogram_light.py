import os
import numpy as np
import librosa
import warnings
import tempfile

# Force any GPU-based processing to use CPU to avoid memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def demix_audio_to_spectrogram_light(audio_file, output_file, sr=44100, n_fft=4096, n_mels=128, fmin=30, fmax=11000):
    """
    Generate a single-channel spectrogram for Beat-Transformer light version
    Creates a stack with a single channel instead of 5 channels to reduce processing time
    """
    print(f"demix_audio_to_spectrogram_light called with:")
    print(f"  - audio_file: {audio_file}")
    print(f"  - output_file: {output_file}")
    print(f"  - sr: {sr}")

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

        # Create a stack with a single channel instead of 5 channels
        # Shape will be (1, time, frequency) instead of (5, time, frequency)
        single_channel_spec = np.expand_dims(spec_db, axis=0)

        print(f"Single-channel spectrogram shape: {single_channel_spec.shape}")

        # Save the result
        np.save(output_file, single_channel_spec)
        print(f"Lightweight spectrogram saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error creating lightweight spectrogram: {e}")
        raise

if __name__ == "__main__":
    # Use your own audio file path here
    AUDIO_FILE = "test_audio/ocean.mp3"
    OUTPUT_FILE = "./demixed_spectrogram_light.npy"
    demix_audio_to_spectrogram_light(AUDIO_FILE, OUTPUT_FILE)
