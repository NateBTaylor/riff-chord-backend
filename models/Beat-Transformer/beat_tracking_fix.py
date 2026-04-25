"""
Fix for Beat-Transformer beat tracking to make it more reliable.
This module provides a wrapper around the original beat_tracking_demo.py
that handles file paths and error cases better.
"""

import os
import sys
import numpy as np
import tempfile
from pathlib import Path
import traceback

# Get the directory of this file
CURRENT_DIR = Path(__file__).parent

def run_beat_tracking_wrapper(demixed_spec_file, audio_file, param_path=None):
    """
    Wrapper around the original run_beat_tracking function that handles file paths
    and error cases better.
    
    Args:
        demixed_spec_file: Path to the demixed spectrogram file
        audio_file: Path to the audio file
        param_path: Path to the model checkpoint (optional)
        
    Returns:
        Tuple of (beat_times, downbeat_times, beats_with_positions, downbeats_with_measures)
    """
    try:
        # Import the original function
        sys.path.insert(0, str(CURRENT_DIR))
        from beat_tracking_demo import run_beat_tracking
        
        # Use default checkpoint if not provided
        if param_path is None:
            param_path = str(CURRENT_DIR / "checkpoint" / "fold_4_trf_param.pt")
        
        # Create temporary files for the output
        temp_dir = tempfile.mkdtemp()
        beats_file = os.path.join(temp_dir, "beats_with_positions.txt")
        downbeats_file = os.path.join(temp_dir, "downbeats_measures.txt")
        
        # Change to the temp directory to run the function
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        
        # Run the original function
        beat_times, downbeat_times = run_beat_tracking(
            demixed_spec_file=demixed_spec_file,
            audio_file=audio_file,
            param_path=param_path
        )
        
        # Read the beat positions from the generated file
        beats_with_positions = []
        try:
            with open('beats_with_positions.txt', 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        time_part = parts[0].strip().replace('time: ', '')
                        beat_num_part = parts[1].strip().replace('beatNum: ', '')
                        try:
                            time = float(time_part)
                            beat_num = int(beat_num_part)
                            beats_with_positions.append({
                                "time": time,
                                "beatNum": beat_num
                            })
                        except (ValueError, TypeError):
                            print(f"Warning: Could not parse line: {line}")
        except FileNotFoundError:
            print("Warning: beats_with_positions.txt not found")
            # Create simple beat positions if file not found
            beats_with_positions = [
                {"time": float(time), "beatNum": ((i % 4) + 1)}
                for i, time in enumerate(beat_times)
            ]
        
        # Read downbeat positions
        downbeats_with_measures = []
        try:
            with open('downbeats_measures.txt', 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        time_part = parts[0].strip().replace('time: ', '')
                        measure_part = parts[1].strip().replace('measureNum: ', '')
                        try:
                            time = float(time_part)
                            measure = int(measure_part)
                            downbeats_with_measures.append({
                                "time": time,
                                "measureNum": measure
                            })
                        except (ValueError, TypeError):
                            print(f"Warning: Could not parse line: {line}")
        except FileNotFoundError:
            print("Warning: downbeats_measures.txt not found")
            # Create simple downbeat positions if file not found
            downbeats_with_measures = [
                {"time": float(time), "measureNum": i + 1}
                for i, time in enumerate(downbeat_times)
            ]
        
        # Change back to the original directory
        os.chdir(original_dir)
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temporary files: {e}")
        
        return beat_times, downbeat_times, beats_with_positions, downbeats_with_measures
    
    except Exception as e:
        print(f"Error in run_beat_tracking_wrapper: {e}")
        traceback.print_exc()
        # Return empty arrays as a fallback
        return np.array([]), np.array([]), [], []
