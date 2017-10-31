import sys
sys.path.append("../")
import os
from scipy.io import wavfile

import numpy as np
import modules.features.data_representation as data_rep
import modules.features.spectrogram as spectrogram
from matplotlib import pyplot as plt


if len(sys.argv) < 2:
    print ('this method needs 3 args RAW_DIR FEATURE_DIR NUM_CONTEXT')
    print ('RAW_DIR ~> directory of audio raw formatted .WAV')
else:
    raw_dir = sys.argv[1]
    if not os.path.exists(raw_dir):
        print ("RAW_DIR doesn't exist")
    else:
        for root, dirs, files in os.walk(raw_dir, topdown=False):
            for file in files:
                name, ext = file.split('.')
                if ext == 'wav':
                    print(name)
                    filename = os.path.join(root, file)
                    fs,audio = wavfile.read(filename)
                    mfcc_audio = data_rep.mfcc(audio,fs)
                    print(mfcc_audio)
                    normalized_audio = data_rep.normalize_to_db(audio,0)
                    mfcc_normalized_audio = data_rep.mfcc(normalized_audio,fs)
                    print(mfcc_normalized_audio)
                    wavfile.write(os.path.join(root,name+"-normalized.wav"),fs,normalized_audio)
