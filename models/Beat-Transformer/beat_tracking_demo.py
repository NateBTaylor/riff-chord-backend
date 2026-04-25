from pathlib import Path
import sys
import warnings

# Fix for Python 3.10+ compatibility with madmom
# MUST come before any madmom imports
import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence

# Insert the "code" folder into sys.path
sys.path.insert(0, str(Path(__file__).parent / "code"))

import numpy as np
# Fix for NumPy 1.20+ compatibility
# These attributes are deprecated in newer NumPy versions
try:
    np.float = float  # Use built-in float instead
    np.int = int      # Use built-in int instead
    print("Applied NumPy compatibility fixes for np.float and np.int")
except Exception as e:
    print(f"Note: NumPy compatibility patch not needed: {e}")

import torch
import librosa
import IPython.display as ipd

# Now it's safe to import madmom
try:
    from madmom.features.beats import DBNBeatTrackingProcessor
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    MADMOM_AVAILABLE = True
except ImportError as e:
    MADMOM_AVAILABLE = False
    warnings.warn(f"Madmom import failed: {e}. Will use librosa fallback for beat tracking.")

import json

# Try to import from the correct path
try:
    from code.DilatedTransformer import Demixed_DilatedTransformerModel
    print("Successfully imported Demixed_DilatedTransformerModel from code.DilatedTransformer at module level")
except ImportError as e:
    print(f"Error importing Demixed_DilatedTransformerModel at module level: {e}")
    try:
        # Try alternative import path
        from DilatedTransformer import Demixed_DilatedTransformerModel
        print("Successfully imported Demixed_DilatedTransformerModel from root at module level")
    except ImportError as e2:
        print(f"Error importing from root at module level: {e2}")
        print("Will try to import when needed")

# Filter the specific RuntimeWarning from madmom
warnings.filterwarnings('ignore', category=RuntimeWarning,
                      message='divide by zero encountered in log')

def fallback_beat_tracking(beat_activation, downbeat_activation, sr=44100, hop_length=1024):
    """Simple beat tracking fallback when madmom is not available"""
    print("Using librosa fallback for beat tracking")

    try:
        # Convert activation to frames for librosa
        frames = np.where(beat_activation > 0.5)[0]
        beat_frames = librosa.util.fix_frames(frames)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

        # For downbeats, use a higher threshold
        downbeat_frames = np.where(downbeat_activation > 0.6)[0]
        downbeat_frames = librosa.util.fix_frames(downbeat_frames)
        downbeat_times = librosa.frames_to_time(downbeat_frames, sr=sr, hop_length=hop_length)

        # If no downbeats detected, try to estimate the time signature
        if len(downbeat_times) == 0 and len(beat_times) >= 4:
            print("No downbeats detected, estimating time signature from beat intervals")

            # Calculate beat intervals
            beat_intervals = np.diff(beat_times)

            # Look for patterns in the beat intervals
            # Try different time signatures (2/4, 3/4, 4/4, etc.)
            possible_signatures = [2, 3, 4, 6, 8]
            best_signature = 4  # Default to 4/4
            best_score = float('inf')

            for signature in possible_signatures:
                # Calculate score based on how well beats fit into measures of this length
                score = 0
                for i in range(len(beat_times) - signature):
                    measure_length = beat_times[i + signature] - beat_times[i]
                    expected_beat_length = measure_length / signature

                    # Check if intermediate beats are evenly spaced
                    for j in range(1, signature):
                        expected_time = beat_times[i] + j * expected_beat_length
                        actual_time = beat_times[i + j]
                        score += abs(expected_time - actual_time)

                score /= (len(beat_times) - signature)

                if score < best_score:
                    best_score = score
                    best_signature = signature

            print(f"Estimated time signature: {best_signature}/4 (score: {best_score:.4f})")

            # Use the estimated time signature to place downbeats
            downbeat_times = beat_times[::best_signature]
            print(f"Generated {len(downbeat_times)} downbeats using {best_signature}/4 time signature")

        # Ensure we have at least one downbeat for processing
        if len(downbeat_times) == 0:
            print("Adding default downbeat at time 0")
            downbeat_times = np.array([0.0])

        return beat_times, downbeat_times

    except Exception as e:
        print(f"Error in fallback beat tracking: {e}")
        # Return minimal valid data
        return np.array([0.0]), np.array([0.0])

def run_beat_tracking(demixed_spec_file, audio_file, param_path, device="cuda:0", max_time=None):
    """
    Run beat tracking on a demixed spectrogram

    Parameters:
    - demixed_spec_file: Path to the demixed spectrogram file
    - audio_file: Path to the audio file
    - param_path: Path to the model parameters
    - device: Device to run the model on (default: "cuda:0")
    - max_time: Maximum time in seconds to process (for chunking)
    """
    # Debug logging commented out
    # print(f"Beat-Transformer run_beat_tracking called with:")
    # print(f"  - demixed_spec_file: {demixed_spec_file}")
    # print(f"  - audio_file: {audio_file}")
    # print(f"  - param_path: {param_path}")
    # print(f"  - device: {device}")
    # print(f"  - max_time: {max_time}")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    # print(f"CUDA available: {cuda_available}")

    # use CPU if CUDA is not available
    device = torch.device("cuda:0" if cuda_available else "cpu")
    # print(f"Using device: {device}")

    try:
        # Load pre-saved demixed spectrogram
        demixed_spec = np.load(demixed_spec_file)

        # Get spectrogram dimensions
        instr, time_frames, mel_bins = demixed_spec.shape
        # print(f"Spectrogram shape: {demixed_spec.shape}")

        # Initialize Beat Transformer model
        try:
            # Try to import from the correct path
            try:
                # First try with the code package
                from code.DilatedTransformer import Demixed_DilatedTransformerModel
                # print("Successfully imported Demixed_DilatedTransformerModel from code.DilatedTransformer")
            except ImportError as e:
                # Try with direct import from the code directory
                import sys
                import os
                # Get the directory of the current file
                current_dir = os.path.dirname(os.path.abspath(__file__))
                # Add the code directory to the path
                code_dir = os.path.join(current_dir, "code")
                if code_dir not in sys.path:
                    sys.path.append(code_dir)
                    # print(f"Added {code_dir} to Python path")

                # Now try to import from the code directory
                from DilatedTransformer import Demixed_DilatedTransformerModel
                # print("Successfully imported Demixed_DilatedTransformerModel from code directory")
        except ImportError as e:
            print(f"Error importing Demixed_DilatedTransformerModel: {e}")
            try:
                # Try alternative import path
                from DilatedTransformer import Demixed_DilatedTransformerModel
                # print("Successfully imported Demixed_DilatedTransformerModel from root")
            except ImportError as e2:
                print(f"Error importing from root: {e2}")
                raise ImportError(f"Could not import Demixed_DilatedTransformerModel: {e}, {e2}")

        model = Demixed_DilatedTransformerModel(attn_len=5, instr=5, ntoken=2,
                                                dmodel=256, nhead=8, d_hid=1024,
                                                nlayers=9, norm_first=True)
        model.load_state_dict(torch.load(param_path, map_location=device)['state_dict'])
        model.to(device)
        model.eval()

        # Check if the audio is longer than the model can handle in one go
        # The model has a limitation of approximately 167 seconds (which is around 7200 frames)
        MAX_FRAMES = 6000  # Reduced to be well below the limit
        OVERLAP_FRAMES = 1500  # Increased overlap for better continuity

        # Calculate frames to seconds conversion for logging
        FRAMES_PER_SECOND = 44100 / 1024  # Based on the model's fps parameter
        # print(f"Model limitation: ~167 seconds ({MAX_FRAMES} frames)")
        # print(f"Audio duration: ~{time_frames / FRAMES_PER_SECOND:.2f} seconds ({time_frames} frames)")

        if time_frames > MAX_FRAMES:
            # print(f"Long audio detected ({time_frames} frames). Processing in chunks...")

            # Process in overlapping chunks
            beat_activation_chunks = []
            downbeat_activation_chunks = []
            frame_positions = []

            # Process each chunk
            for start_frame in range(0, time_frames, MAX_FRAMES - OVERLAP_FRAMES):
                end_frame = min(start_frame + MAX_FRAMES, time_frames)
                # print(f"Processing chunk from frame {start_frame} to {end_frame} ({end_frame - start_frame} frames)")

                # Extract chunk
                spec_chunk = demixed_spec[:, start_frame:end_frame, :]

                # Run model inference on chunk
                with torch.no_grad():
                    # Print shape information for debugging
                    # print(f"Chunk shape before model: {spec_chunk.shape}")

                    # Ensure the input has the correct shape (batch, instr, time, mel_bins)
                    # The model expects (batch, instr, time, mel_bins) but our spectrogram might be (instr, time, mel_bins)
                    model_input = torch.from_numpy(spec_chunk).float()
                    if len(model_input.shape) == 3:  # (instr, time, mel_bins)
                        model_input = model_input.unsqueeze(0)  # Add batch dimension

                    # FIXED: Add context padding for chunked processing too
                    # The model needs ~30 frames of context due to dilated attention (attn_len=5, 9 layers)
                    batch, instr, time, mel_bins = model_input.shape
                    context_frames = 32  # Same as full processing

                    # For the first chunk, add padding at the beginning
                    if start_frame == 0:
                        padding = torch.zeros((batch, instr, context_frames, mel_bins), dtype=model_input.dtype)
                        model_input = torch.cat([padding, model_input], dim=2)
                        chunk_has_padding = True
                    else:
                        chunk_has_padding = False

                    # Ensure the model input has the correct shape
                    # print(f"Model input shape: {model_input.shape}")

                    # Check if the input shape is valid for the model
                    if model_input.shape[2] > 6000:
                        # print(f"Warning: Input time dimension ({model_input.shape[2]}) exceeds model limit (6000)")
                        # print("Truncating input to 6000 frames")
                        model_input = model_input[:, :, :6000, :]
                        # print(f"Truncated model input shape: {model_input.shape}")

                    # Move to device
                    model_input = model_input.to(device)

                    try:
                        activation, _ = model(model_input)
                    except RuntimeError as e:
                        # print(f"Error in model inference: {e}")
                        # print("Trying with reshaped input...")

                        # Try reshaping the input to match the expected shape
                        if len(model_input.shape) == 4:  # (batch, instr, time, mel_bins)
                            # Reshape to ensure dimensions are compatible
                            batch, instr, time, mel_bins = model_input.shape

                            # Ensure time dimension is a multiple of 256 (common requirement for transformer models)
                            pad_time = (256 - (time % 256)) % 256
                            if pad_time > 0:
                                # print(f"Padding time dimension by {pad_time} frames")
                                pad = torch.zeros((batch, instr, pad_time, mel_bins), device=device)
                                model_input = torch.cat([model_input, pad], dim=2)
                                # print(f"Padded model input shape: {model_input.shape}")

                            activation, _ = model(model_input)

                # Get activations
                beat_act_chunk = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
                downbeat_act_chunk = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()

                # Remove padding from first chunk if it was added
                if start_frame == 0 and chunk_has_padding:
                    beat_act_chunk = beat_act_chunk[context_frames:]
                    downbeat_act_chunk = downbeat_act_chunk[context_frames:]

                # Store chunk activations and positions
                beat_activation_chunks.append(beat_act_chunk)
                downbeat_activation_chunks.append(downbeat_act_chunk)
                frame_positions.append(start_frame)

            # Combine chunks with proper overlap handling
            beat_activation = np.zeros(time_frames)
            downbeat_activation = np.zeros(time_frames)

            for i, (start_frame, beat_act, downbeat_act) in enumerate(zip(frame_positions, beat_activation_chunks, downbeat_activation_chunks)):
                chunk_len = len(beat_act)
                end_frame = start_frame + chunk_len

                if i == 0:
                    # First chunk - use all frames
                    beat_activation[start_frame:end_frame] = beat_act
                    downbeat_activation[start_frame:end_frame] = downbeat_act
                else:
                    # For subsequent chunks, blend the overlapping region
                    overlap_start = start_frame
                    overlap_end = min(frame_positions[i-1] + len(beat_activation_chunks[i-1]), end_frame)
                    non_overlap_end = end_frame

                    # Linear weights for blending (fade out previous chunk, fade in current chunk)
                    overlap_len = overlap_end - overlap_start
                    if overlap_len > 0:
                        weights = np.linspace(0, 1, overlap_len)

                        # Blend overlapping region
                        beat_activation[overlap_start:overlap_end] = (
                            beat_activation[overlap_start:overlap_end] * (1 - weights) +
                            beat_act[:overlap_len] * weights
                        )
                        downbeat_activation[overlap_start:overlap_end] = (
                            downbeat_activation[overlap_start:overlap_end] * (1 - weights) +
                            downbeat_act[:overlap_len] * weights
                        )

                        # Copy non-overlapping region
                        if overlap_end < non_overlap_end:
                            beat_activation[overlap_end:non_overlap_end] = beat_act[overlap_len:]
                            downbeat_activation[overlap_end:non_overlap_end] = downbeat_act[overlap_len:]

            # print(f"Successfully combined {len(beat_activation_chunks)} chunks")

        else:
            # For shorter audio, process the whole spectrogram at once
            with torch.no_grad():
                # Print shape information for debugging
                # print(f"Full spectrogram shape before model: {demixed_spec.shape}")

                # Ensure the input has the correct shape (batch, instr, time, mel_bins)
                # The model expects (batch, instr, time, mel_bins) but our spectrogram might be (instr, time, mel_bins)
                model_input = torch.from_numpy(demixed_spec).float()
                if len(model_input.shape) == 3:  # (instr, time, mel_bins)
                    model_input = model_input.unsqueeze(0)  # Add batch dimension

                # FIXED: Add context padding to allow predictions from the very beginning
                # The model needs ~30 frames of context due to dilated attention (attn_len=5, 9 layers)
                # Add padding at the beginning so the model can predict beats from frame 0
                batch, instr, time, mel_bins = model_input.shape
                context_frames = 32  # Slightly more than the observed 28-frame delay

                # Create padding with zeros (silence) at the beginning
                padding = torch.zeros((batch, instr, context_frames, mel_bins), dtype=model_input.dtype)
                model_input_padded = torch.cat([padding, model_input], dim=2)

                # Use the padded input for model inference
                model_input = model_input_padded

                # Ensure the model input has the correct shape
                # print(f"Full model input shape: {model_input.shape}")

                # Check if the input shape is valid for the model
                if model_input.shape[2] > 6000:
                    # print(f"Warning: Input time dimension ({model_input.shape[2]}) exceeds model limit (6000)")
                    # print("Truncating input to 6000 frames")
                    model_input = model_input[:, :, :6000, :]
                    # print(f"Truncated model input shape: {model_input.shape}")

                # Move to device
                model_input = model_input.to(device)

                try:
                    activation, _ = model(model_input)
                except RuntimeError as e:
                    # print(f"Error in model inference: {e}")
                    # print("Trying with reshaped input...")

                    # Try reshaping the input to match the expected shape
                    if len(model_input.shape) == 4:  # (batch, instr, time, mel_bins)
                        # Reshape to ensure dimensions are compatible
                        batch, instr, time, mel_bins = model_input.shape

                        # Ensure time dimension is a multiple of 256 (common requirement for transformer models)
                        pad_time = (256 - (time % 256)) % 256
                        if pad_time > 0:
                            # print(f"Padding time dimension by {pad_time} frames")
                            pad = torch.zeros((batch, instr, pad_time, mel_bins), device=device)
                            model_input = torch.cat([model_input, pad], dim=2)
                            # print(f"Padded model input shape: {model_input.shape}")

                        activation, _ = model(model_input)

                # Extract activations and remove the padding frames
                beat_activation_padded = torch.sigmoid(activation[0, :, 0]).detach().cpu().numpy()
                downbeat_activation_padded = torch.sigmoid(activation[0, :, 1]).detach().cpu().numpy()

                # Remove the padding frames to get activations aligned with original audio
                beat_activation = beat_activation_padded[context_frames:]
                downbeat_activation = downbeat_activation_padded[context_frames:]



        if MADMOM_AVAILABLE:
            # Initialize DBN processors from madmom for beat tracking
            # Allow a wider range of BPM values and more flexibility
            beat_tracker = DBNBeatTrackingProcessor(min_bpm=40.0, max_bpm=240.0, fps=44100/1024,
                                                    transition_lambda=100, observation_lambda=6,
                                                    num_tempi=None, threshold=0.1)

            # Initialize downbeat tracker with more time signature options
            # Include 2/4, 3/4, 4/4, 5/4, 6/8, 7/8, etc.
            downbeat_tracker = DBNDownBeatTrackingProcessor(beats_per_bar=[2, 3, 4, 5, 6, 7, 8],
                                                            min_bpm=40.0, max_bpm=240.0, fps=44100/1024,
                                                            transition_lambda=100, observation_lambda=6,
                                                            num_tempi=None, threshold=0.1)

            # Process beats with DBN
            # print(f"Running beat tracking on activation of length {len(beat_activation)}")

            # For very long audio, we need to process the beat tracking in chunks too
            if len(beat_activation) > 15000:  # If activation is very long
                # print("Beat activation is very long, processing in chunks...")

                # Process in chunks of 10000 frames with 2000 frame overlap
                chunk_size = 10000
                overlap = 2000
                all_beats = []

                for start_idx in range(0, len(beat_activation), chunk_size - overlap):
                    end_idx = min(start_idx + chunk_size, len(beat_activation))
                    # print(f"Processing beat tracking chunk {start_idx}-{end_idx}")

                    # Process chunk
                    chunk_activation = beat_activation[start_idx:end_idx]

                    chunk_beats = beat_tracker(chunk_activation)

                    # Convert to global frame indices
                    chunk_beats_global = chunk_beats + (start_idx / FRAMES_PER_SECOND)

                    # For all chunks except the first, remove beats in the overlap region
                    # that are too close to beats from the previous chunk
                    if start_idx > 0 and len(all_beats) > 0:
                        # Only keep beats that are at least 0.1s after the last beat of previous chunk
                        min_time = all_beats[-1] + 0.1
                        chunk_beats_global = chunk_beats_global[chunk_beats_global > min_time]

                    all_beats.extend(chunk_beats_global)

                dbn_beat_pred = np.array(all_beats)
                # print(f"Total beats after chunking: {len(dbn_beat_pred)}")
            else:
                # Process normally for shorter audio
                dbn_beat_pred = beat_tracker(beat_activation)



            # For downbeats, use the exact approach from the original implementation
            try:
                # For very long audio, we need to process the downbeat tracking in chunks too
                # print(f"Running downbeat tracking on activation of length {len(downbeat_activation)}")

                if len(downbeat_activation) > 15000:  # If activation is very long
                    # print("Downbeat activation is very long, processing in chunks...")

                    # Process in chunks of 10000 frames with 2000 frame overlap
                    chunk_size = 10000
                    overlap = 2000
                    all_downbeats = []

                    for start_idx in range(0, len(downbeat_activation), chunk_size - overlap):
                        end_idx = min(start_idx + chunk_size, len(downbeat_activation))
                        # print(f"Processing downbeat tracking chunk {start_idx}-{end_idx}")

                        # Extract chunk
                        beat_act_chunk = beat_activation[start_idx:end_idx]
                        downbeat_act_chunk = downbeat_activation[start_idx:end_idx]

                        # Create combined activation for this chunk
                        combined_act_chunk = np.concatenate((
                            np.maximum(beat_act_chunk - downbeat_act_chunk, np.zeros(beat_act_chunk.shape))[:, np.newaxis],
                            downbeat_act_chunk[:, np.newaxis]
                        ), axis=-1)  # (T, 2)

                        try:
                            # Process chunk
                            chunk_downbeats_with_labels = downbeat_tracker(combined_act_chunk)

                            # Filter to only include downbeats (where second column is 1)
                            chunk_downbeats = chunk_downbeats_with_labels[chunk_downbeats_with_labels[:, 1] == 1][:, 0]

                            # Convert to global time
                            chunk_downbeats_global = chunk_downbeats + (start_idx / FRAMES_PER_SECOND)

                            # For all chunks except the first, remove downbeats in the overlap region
                            # that are too close to downbeats from the previous chunk
                            if start_idx > 0 and len(all_downbeats) > 0:
                                # Only keep downbeats that are at least 0.5s after the last downbeat of previous chunk
                                min_time = all_downbeats[-1] + 0.5
                                chunk_downbeats_global = chunk_downbeats_global[chunk_downbeats_global > min_time]

                            all_downbeats.extend(chunk_downbeats_global)

                        except Exception as e:
                            print(f"Error processing downbeat chunk {start_idx}-{end_idx}: {e}")

                    dbn_downbeat_pred = np.array(all_downbeats)
                    # print(f"Total downbeats after chunking: {len(dbn_downbeat_pred)}")

                else:
                    # For shorter audio, process the whole activation at once
                    # Create combined activation exactly as in the original code
                    combined_act = np.concatenate((
                        np.maximum(beat_activation - downbeat_activation, np.zeros(beat_activation.shape))[:, np.newaxis],
                        downbeat_activation[:, np.newaxis]
                    ), axis=-1)  # (T, 2)

                    # Use the downbeat tracker with the exact same approach as the original
                    dbn_downbeat_pred_with_labels = downbeat_tracker(combined_act)

                    # Filter to only include downbeats (where second column is 1)
                    # This is exactly what the original code does
                    dbn_downbeat_pred = dbn_downbeat_pred_with_labels[dbn_downbeat_pred_with_labels[:, 1] == 1][:, 0]

                # Check if we have a reasonable number of downbeats
                if len(dbn_downbeat_pred) < 10:
                    # print(f"Too few downbeats detected ({len(dbn_downbeat_pred)}). Using every 4th beat.")
                    dbn_downbeat_pred = dbn_beat_pred[::4]

                # If we have too many downbeats, filter them
                if len(dbn_downbeat_pred) > 150:
                    # print(f"Too many downbeats detected ({len(dbn_downbeat_pred)}). Filtering to every 4th beat.")
                    # Keep only every 4th beat that is also a downbeat
                    beat_indices = np.arange(0, len(dbn_beat_pred), 4)
                    beat_times = dbn_beat_pred[beat_indices]

                    # Find the closest downbeat to each 4th beat
                    filtered_downbeats = []
                    for beat_time in beat_times:
                        closest_idx = np.argmin(np.abs(dbn_downbeat_pred - beat_time))
                        if abs(dbn_downbeat_pred[closest_idx] - beat_time) < 0.2:  # Within 200ms
                            filtered_downbeats.append(dbn_downbeat_pred[closest_idx])

                    if len(filtered_downbeats) >= 10:
                        dbn_downbeat_pred = np.array(filtered_downbeats)

            except Exception as e:
                # print(f"Error in DBN downbeat tracking: {e}")
                # print("Using every 4th beat as downbeat (standard 4/4 time).")
                dbn_downbeat_pred = dbn_beat_pred[::4] if len(dbn_beat_pred) >= 4 else np.array([0.0])

            except Exception as e:
                # print(f"Error in downbeat detection: {e}")
                # Use a simple fallback - every 4th beat is a downbeat (standard 4/4 time)
                dbn_downbeat_pred = dbn_beat_pred[::4] if len(dbn_beat_pred) >= 4 else np.array([0.0])
        else:
            # Use fallback if madmom not available
            dbn_beat_pred, dbn_downbeat_pred = fallback_beat_tracking(beat_activation, downbeat_activation)

        # Check if audio file exists
        audio_path = Path(audio_file)
        if not audio_path.exists():
            # print(f"Error: Audio file not found: {audio_file}")
            # print(f"Current working directory: {Path.cwd()}")
            # print(f"Absolute path: {audio_path.absolute()}")
            return

        # Load original audio with exception handling
        try:
            audio, sr = librosa.load(str(audio_path.absolute()), sr=None)
            # print(f"Successfully loaded audio: {audio_file}, shape: {audio.shape}, sr: {sr}")
        except Exception as e:
            # print(f"Error loading audio file {audio_file}: {e}")
            # print(f"Audio file absolute path: {audio_path.absolute()}")
            # Provide a dummy audio for visualization
            # print("Using dummy audio for visualization...")
            sr = 44100
            audio = np.zeros(int(max(dbn_beat_pred[-1] if len(dbn_beat_pred) > 0 else 10, 10) * sr))

        # Check if we got valid beat predictions
        if len(dbn_beat_pred) == 0:
            # print("Warning: No beats detected. Using librosa fallback.")
            tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
            dbn_beat_pred = librosa.frames_to_time(beats, sr=sr)
            # Simple downbeats (every 4 beats)
            dbn_downbeat_pred = dbn_beat_pred[::4] if len(dbn_beat_pred) >= 4 else np.array([0.0])

        # Create click tracks and overlay
        beats_click = librosa.clicks(times=dbn_beat_pred, sr=sr, click_freq=1000.0,
                                    click_duration=0.1, length=len(audio))
        downbeats_click = librosa.clicks(times=dbn_downbeat_pred, sr=sr, click_freq=1500.0,
                                        click_duration=0.15, length=len(audio))
        output_audio = 0.6*audio + 0.25*beats_click + 0.25*downbeats_click

        # Save the output audio with clicks
        output_file = "output_with_clicks.wav"
        try:
            import soundfile as sf
            sf.write(output_file, output_audio, sr)
            # print(f"Saved audio with beat clicks to {output_file}")
        except ImportError:
            # print("soundfile module not found. Install with: pip install soundfile")
            pass

        # Try to play the audio directly if sounddevice is available
        try:
            import sounddevice as sd
            # print("\nPlaying audio with clicks... Press Ctrl+C to stop.")
            # sd.play(output_audio, sr)
            # sd.wait()  # Wait until playback is done
        except ImportError:
            # print("\nTo play audio directly, install sounddevice: pip install sounddevice")
            pass
        except Exception as e:
            # print(f"\nCouldn't play audio: {e}")
            pass

        # Process downbeats - assign sequential measure numbers
        downbeats_with_positions = []
        for i, downbeat_time in enumerate(dbn_downbeat_pred):
            downbeats_with_positions.append({
                "time": round(downbeat_time, 2),
                "measureNum": i + 1  # Measure numbers start from 1
            })

        # Save downbeats as plain text file
        with open('downbeats_measures.txt', 'w') as f:
            for entry in downbeats_with_positions:
                f.write(f"time: {entry['time']:.2f} \t measureNum: {entry['measureNum']}\n")

        # print("\nFirst few downbeats (measures):")
        # for entry in downbeats_with_positions[:7]:
        #     print(f"time: {entry['time']:.2f} \t measureNum: {entry['measureNum']}")
        # print(f"Saved {len(downbeats_with_positions)} downbeats to downbeats_measures.txt")

        # Process regular beats - assign modular beat numbers and calculate BPM
        beats_with_positions = []

        # Calculate overall BPM
        if len(dbn_beat_pred) >= 2:
            # Calculate time differences between consecutive beats
            beat_diffs = np.diff(dbn_beat_pred)
            # Convert to BPM (beats per minute)
            beat_bpms = 60.0 / beat_diffs
            # Calculate average BPM
            avg_bpm = np.mean(beat_bpms)
            # Calculate median BPM (more robust to outliers)
            median_bpm = np.median(beat_bpms)

            # print(f"\nOverall tempo analysis:")
            # print(f"Average BPM: {avg_bpm:.2f}")
            # print(f"Median BPM: {median_bpm:.2f}")

            # Check for BPM changes
            # Use a sliding window to detect significant changes in tempo
            window_size = 8  # 8 beats per window
            if len(beat_bpms) >= window_size * 2:
                window_bpms = []
                for i in range(0, len(beat_bpms) - window_size + 1, window_size // 2):
                    window = beat_bpms[i:i+window_size]
                    window_bpms.append((i, np.median(window)))

                # Detect significant changes (more than 10% change)
                bpm_changes = []
                for i in range(1, len(window_bpms)):
                    prev_bpm = window_bpms[i-1][1]
                    curr_bpm = window_bpms[i][1]
                    change_pct = abs(curr_bpm - prev_bpm) / prev_bpm * 100

                    if change_pct > 10:  # More than 10% change
                        beat_idx = window_bpms[i][0]
                        beat_time = dbn_beat_pred[beat_idx]
                        bpm_changes.append((beat_time, prev_bpm, curr_bpm, change_pct))

                if bpm_changes:
                    # print("\nDetected BPM changes:")
                    # for time, from_bpm, to_bpm, pct in bpm_changes:
                    #     print(f"At {time:.2f}s: {from_bpm:.1f} → {to_bpm:.1f} BPM ({pct:.1f}% change)")
                    pass
                # else:
                #     print("\nNo significant BPM changes detected")
            # else:
            #     print("\nNot enough beats to analyze BPM changes")

        # Check if we have enough downbeats to do proper positioning
        if len(dbn_downbeat_pred) >= 2:
            # Find which measure each beat belongs to
            downbeat_times = sorted(dbn_downbeat_pred)

            # Determine the most common number of beats per measure (time signature)
            measure_lengths = []
            for j in range(len(downbeat_times) - 1):
                beats_in_this_measure = len([b for b in dbn_beat_pred
                                           if downbeat_times[j] <= b < downbeat_times[j+1]])
                if beats_in_this_measure > 0:
                    measure_lengths.append(beats_in_this_measure)

            # Use most common measure length (time signature)
            from collections import Counter
            if measure_lengths:
                common_length = Counter(measure_lengths).most_common(1)[0][0]
                # Ensure it's a reasonable value (2, 3, 4, 6, etc.)
                if common_length not in [2, 3, 4, 6, 8, 12]:
                    # Default to 4/4 time if unusual value
                    common_length = 4
            else:
                # Default to 4/4 time if we couldn't determine
                common_length = 4

            # print(f"\nDetected time signature with {common_length} beats per measure")

            # Check for time signature changes
            if len(measure_lengths) >= 3:
                time_sig_changes = []
                current_ts = measure_lengths[0]
                for i in range(1, len(measure_lengths)):
                    if measure_lengths[i] != current_ts and measure_lengths[i] in [2, 3, 4, 5, 6, 7, 8, 12]:
                        time_sig_changes.append((downbeat_times[i], current_ts, measure_lengths[i]))
                        current_ts = measure_lengths[i]

                # if time_sig_changes:
                #     print("\nDetected time signature changes:")
                #     for time, from_ts, to_ts in time_sig_changes:
                #         print(f"At {time:.2f}s: {from_ts}/4 → {to_ts}/4")
                # else:
                #     print("\nNo time signature changes detected")

            # TEMPORARY FIX: Apply timing offset compensation
            # Identify pickup beats (beats before the first downbeat)
            pickup_beats = []
            if len(downbeat_times) > 0:
                first_downbeat = downbeat_times[0]
                pickup_beats = [b for b in dbn_beat_pred if b < first_downbeat]

            # Calculate timing offset and missing beats
            first_beat_time = dbn_beat_pred[0] if len(dbn_beat_pred) > 0 else 0
            timing_offset = first_beat_time  # e.g., 0.65s

            # Estimate tempo from the first few beats
            if len(dbn_beat_pred) >= 3:
                intervals = np.diff(dbn_beat_pred[:5])  # Use first 5 beats for tempo
                avg_beat_interval = np.median(intervals)  # More robust than mean
            else:
                avg_beat_interval = 0.32  # Default fallback

            # Calculate how many beats were cut off initially
            missing_beats_count = int(round(timing_offset / avg_beat_interval))

            print(f"DEBUG - Timing compensation:")
            print(f"  - First beat at: {first_beat_time:.3f}s")
            print(f"  - Average beat interval: {avg_beat_interval:.3f}s")
            print(f"  - Estimated missing beats: {missing_beats_count}")
            print(f"  - Detected pickup beats: {len(pickup_beats)}")
            print(f"  - Time signature: {common_length}/4")

            # Add missing beats at the beginning (padded beats)
            padded_beats = []
            if missing_beats_count > 0:
                for i in range(missing_beats_count):
                    estimated_time = first_beat_time - (missing_beats_count - i) * avg_beat_interval
                    # Clip negative times to 0 to prevent negative timestamps
                    estimated_time = max(0.0, estimated_time)
                    padded_beats.append(estimated_time)

                print(f"  - Adding {missing_beats_count} padded beats before first detected beat")
                print(f"  - Padded beat times: {[f'{t:.3f}s' for t in padded_beats]}")

            # Combine padded beats with detected beats for processing
            all_beats = padded_beats + list(dbn_beat_pred)
            all_beat_sources = ['padded'] * len(padded_beats) + ['detected'] * len(dbn_beat_pred)

            # Recalculate pickup beats including padded ones
            if len(downbeat_times) > 0:
                first_downbeat = downbeat_times[0]
                all_pickup_beats = [b for b in all_beats if b < first_downbeat]
                all_pickup_sources = [all_beat_sources[i] for i, b in enumerate(all_beats) if b < first_downbeat]
            else:
                all_pickup_beats = []
                all_pickup_sources = []

            print(f"  - Total pickup beats (including padded): {len(all_pickup_beats)}")

            # For each beat (including padded), determine its position within its measure
            for i, beat_time in enumerate(all_beats):
                beat_source = all_beat_sources[i]
                # Handle pickup beats (including padded ones) with our strategy
                if beat_time in all_pickup_beats:
                    # Calculate beat number for pickup beats
                    pickup_index = all_pickup_beats.index(beat_time)
                    total_pickup_count = len(all_pickup_beats)
                    beat_num = common_length - total_pickup_count + pickup_index + 1

                    # Ensure beat number is valid (wrap around if necessary)
                    if beat_num <= 0:
                        beat_num = common_length + beat_num
                else:
                    # Find which measure this beat belongs to
                    measure_idx = 0
                    while measure_idx < len(downbeat_times) - 1 and beat_time >= downbeat_times[measure_idx + 1]:
                        measure_idx += 1

                    # Calculate beat number within measure
                    if measure_idx < len(downbeat_times) - 1:
                        # If not in the last measure, find position between this and next downbeat
                        next_downbeat_time = downbeat_times[measure_idx + 1]
                        curr_downbeat_time = downbeat_times[measure_idx]

                        # Find all beats within this measure
                        beats_in_measure = [b for b in dbn_beat_pred if curr_downbeat_time <= b < next_downbeat_time]

                        # If this is a downbeat, it's beat 1
                        if abs(beat_time - curr_downbeat_time) < 0.01:
                            beat_num = 1
                        else:
                            # Find this beat's position in the measure
                            try:
                                # First try with tolerance
                                TOLERANCE = 1e-6
                                position = next(i+1 for i, b in enumerate(beats_in_measure)
                                              if abs(b - beat_time) < TOLERANCE)
                            except StopIteration:
                                # Fall back to finding the closest match
                                position = 1 + beats_in_measure.index(min(beats_in_measure,
                                                                     key=lambda b: abs(b - beat_time)))

                            # Ensure beat numbers are within the time signature
                            beat_num = ((position - 1) % common_length) + 1
                    else:
                        # For beats in the last measure
                        if abs(beat_time - downbeat_times[measure_idx]) < 0.01:
                            # If this is a downbeat, it's beat 1
                            beat_num = 1
                        else:
                            # Calculate position in this measure
                            beats_after_last_downbeat = [b for b in dbn_beat_pred if b >= downbeat_times[measure_idx]]
                            position = beats_after_last_downbeat.index(beat_time) + 1

                            # Ensure beat numbers are within the time signature
                            beat_num = ((position - 1) % common_length) + 1

                beats_with_positions.append({
                    "time": round(beat_time, 2),
                    "beatNum": int(beat_num),
                    "source": beat_source  # Track whether this is 'padded' or 'detected'
                })
        else:
            # Simple fallback beat numbering (just modulo 4 for standard 4/4 time)
            for i, beat_time in enumerate(dbn_beat_pred):
                beats_with_positions.append({
                    "time": round(beat_time, 2),
                    "beatNum": ((i % 4) + 1)  # Simple 4/4 pattern
                })

        # Save beats as plain text file
        with open('beats_with_positions.txt', 'w') as f:
            for beat in beats_with_positions:
                source_marker = " (PADDED)" if beat.get('source') == 'padded' else ""
                f.write(f"time: {beat['time']:.2f} \t beatNum: {beat['beatNum']}{source_marker}\n")

        print("\nFirst 20 beats with modular positions (including timing compensation):")
        for beat in beats_with_positions[:20]:
            source_marker = " (PADDED)" if beat.get('source') == 'padded' else ""
            print(f"time: {beat['time']:.2f} \t beatNum: {beat['beatNum']}{source_marker}")

        # Count padded vs detected beats
        padded_count = sum(1 for beat in beats_with_positions if beat.get('source') == 'padded')
        detected_count = len(beats_with_positions) - padded_count

        print(f"Saved {len(beats_with_positions)} beats to beats_with_positions.txt")
        print(f"  - {padded_count} padded beats (timing compensation)")
        print(f"  - {detected_count} detected beats (from model)")

        print(f"\nDetected {len(dbn_beat_pred)} beats")
        print(f"Detected {len(dbn_downbeat_pred)} downbeats")



        # Apply max_time limit if specified
        if max_time is not None:
            print(f"Limiting results to maximum time of {max_time} seconds")
            dbn_beat_pred = dbn_beat_pred[dbn_beat_pred <= max_time]
            dbn_downbeat_pred = dbn_downbeat_pred[dbn_downbeat_pred <= max_time]
            print(f"After limiting: {len(dbn_beat_pred)} beats, {len(dbn_downbeat_pred)} downbeats")

        return dbn_beat_pred, dbn_downbeat_pred

    except Exception as e:
        print(f"Error in run_beat_tracking: {e}")
        import traceback
        traceback.print_exc()
        # Return empty beat and downbeat arrays as a fallback
        return np.array([]), np.array([])

if __name__ == "__main__":
    DEMIXED_SPEC_FILE = "./demixed_spectrogram.npy"
    AUDIO_FILE = "./test_audio/tst30.mp3"  # Update this path to your audio file
    PARAM_PATH = "./checkpoint/fold_4_trf_param.pt"  # Update this path to your model checkpoint
    run_beat_tracking(DEMIXED_SPEC_FILE, AUDIO_FILE, PARAM_PATH)
