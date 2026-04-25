from chordnet_ismir_naive import ChordNet,chord_limit,ChordNetCNN
from mir.nn.train import NetworkInterface
from extractors.cqt import CQTV2,SimpleChordToID
from mir import io,DataEntry
from extractors.xhmm_ismir import XHMMDecoder
import numpy as np
from io_new.chordlab_io import ChordLabIO
from settings import DEFAULT_SR,DEFAULT_HOP_LENGTH
import sys
import os

# Get the absolute path to the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define model paths with absolute paths
MODEL_NAMES = [os.path.join(current_dir, 'cache_data', f'joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s{i}.best.sdict') for i in range(5)]

def chord_recognition(audio_path, lab_path, chord_dict_name='submission'):
    # Use absolute paths for template files
    template_file = os.path.join(current_dir, 'data', f'{chord_dict_name}_chord_list.txt')
    hmm = XHMMDecoder(template_file=template_file)
    
    entry = DataEntry()
    entry.prop.set('sr', DEFAULT_SR)
    entry.prop.set('hop_length', DEFAULT_HOP_LENGTH)
    entry.append_file(audio_path, io.MusicIO, 'music')
    entry.append_extractor(CQTV2, 'cqt')
    
    probs = []
    for model_name in MODEL_NAMES:
        net = NetworkInterface(ChordNet(None), model_name, load_checkpoint=False)
        print(f'Inference: {model_name} on {audio_path}')
        probs.append(net.inference(entry.cqt))
    
    probs = [np.mean([p[i] for p in probs], axis=0) for i in range(len(probs[0]))]
    chordlab = hmm.decode_to_chordlab(entry, probs, False)
    entry.append_data(chordlab, ChordLabIO, 'chord')
    entry.save('chord', lab_path)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        chord_recognition(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        chord_recognition(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print('Usage: chord_recognition.py path_to_audio_file path_to_output_file [chord_dict=submission]')
        print('\tChord dict can be one of the following: full, ismir2017, submission, extended')
        exit(0)
