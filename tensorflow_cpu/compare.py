import sys
sys.path.append("../")
import os
import numpy as np
from python_speech_features import mfcc
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt


if len(sys.argv) < 5:
    print ('this method needs 4 args RAW_DIR FEATURE_DIR NUM_CONTEXT')
    print ('RAW_DIR ~> directory of audio raw formatted .WAV')
    print ('COMPARE_RESULT_DIR ~> directory of target compare files will be created (MFCC)')
    print ('FEATURE_1 ~> top side of compare')
    print ('FEATURE_2 ~> bot side of compare')
    print ('MFCC_TIME_WINDOW_IN_MILISECONDS ~> default to 25 ms')
else:
    raw_dir = sys.argv[1]
    compare_result_dir = sys.argv[2]
    feature_1 = sys.argv[3]
    feature_2 = sys.argv[4]
    time_window = 0.025
    if len(sys.argv) == 6:
        time_window = int(sys.argv[5]) / 1000.0
    if not os.path.exists(raw_dir):
        print ("RAW_DIR doesn't exist")
    else:

        if not os.path.exists(compare_result_dir):
            os.makedirs(compare_result_dir)

        for root, dirs, files in os.walk(raw_dir, topdown=False):
            for file in files:

                name, ext = file.split('.')
                if ext == 'wav':
                    filename = os.path.join(root, file)
                    fs, audio = wavfile.read(filename)
                    mfcc_feature = mfcc(audio,fs,time_window)
                    f, t, spectrogram_feature = signal.spectrogram(audio, fs)

                    if feature_1 == 'mfcc' and feature_2 == 'spectrogram':
                        fig, (top, bot) = plt.subplots(2, 1)
                        t = np.arange(0,mfcc_feature.shape[0],1)
                        f = np.arange(0,mfcc_feature.shape[1],1)
                        top.pcolormesh(t, f, mfcc_feature.T)

                        f = np.arange(0,spectrogram_feature.shape[0],1)
                        t = np.arange(0,spectrogram_feature.shape[1],1)
                        bot.pcolormesh(t, f, spectrogram_feature)
                    elif feature_1 == 'spectrogram' and feature_2 == 'mfcc':
                        fig, (top, bot) = plt.subplots(2, 1)
                        t = np.arange(0,mfcc_feature.shape[0],1)
                        f = np.arange(0,mfcc_feature.shape[1],1)
                        bot.pcolormesh(t, f, mfcc_feature.T)

                        f = np.arange(0,spectrogram_feature.shape[0],1)
                        t = np.arange(0,spectrogram_feature.shape[1],1)
                        top.pcolormesh(t, f, spectrogram_feature)
                    plt.show()