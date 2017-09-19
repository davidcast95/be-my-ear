import sys
sys.path.append("../")
import os
import numpy as np
from python_speech_features import mfcc
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt


if len(sys.argv) < 2:
    print ('this method needs 4 args RAW_DIR FEATURE_DIR NUM_CONTEXT')
    print ('NUMPY_FILE ~> file numpy array formatted .NPY')
else:
    npy_file = sys.argv[1]
    array = np.load(npy_file)
    fig, ax = plt.subplots()
    t = np.arange(0, array.shape[0], 1)
    f = np.arange(0, array.shape[1], 1)
    ax.pcolormesh(t, f, array.T)
    plt.show()
