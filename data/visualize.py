import sys
sys.path.append("../")
import os
import numpy as np
import modules.features.data_representation as data_rep
import modules.features.spectrogram as spectrogram
from matplotlib import pyplot as plt


if len(sys.argv) < 4:
    print ('this method needs 3 args RAW_DIR FEATURE_DIR NUM_CONTEXT')
    print ('RAW_DIR ~> directory of audio raw formatted .WAV')
    print ('FEATURE_TYPE ~> mffc / spectrogram')
    print ('NUM_CONTEXT ~> mfcc : number of past and future context, spectrogram : binsize ')
else:
    raw_dir = sys.argv[1]
    preprocess_type = sys.argv[2]
    num_context = int(sys.argv[3])
    if not os.path.exists(raw_dir):
        print ("RAW_DIR doesn't exist")
    else:
        if preprocess_type == 'mfcc':
            for root, dirs, files in os.walk(raw_dir, topdown=False):
                for file in files:
                    name, ext = file.split('.')
                    if ext == 'wav':
                        filename = os.path.join(root, file)
                        vector_feature = data_rep.audio_to_feature_representation(filename,num_context)
                        print(vector_feature.shape)
                        fig, ax = plt.subplots()
                        t = np.arange(0, vector_feature.shape[0], 1)
                        f = np.arange(0, vector_feature.shape[1], 1)
                        print(vector_feature.T.min())
                        print(vector_feature.T.max())
                        ax.pcolormesh(t, f, vector_feature.T, vmin=-100,vmax=100)
                        ax.set_title(name)
                        fig.savefig(os.path.join(raw_dir,name + '.png'))
                        # plt.show()

        if preprocess_type == 'spectrogram':
            for root, dirs, files in os.walk(raw_dir, topdown=False):
                for file in files:
                    name, ext = file.split('.')
                    if ext == 'wav':
                        print(file)
                        filename = os.path.join(root, file)
                        vector_feature = spectrogram.generate(filename,num_context)
                        vector_feature = data_rep.normalize(vector_feature,0,100)
                        print(vector_feature)
                        fig, ax = plt.subplots()
                        t = np.arange(0, vector_feature.shape[0], 1)
                        f = np.arange(0, vector_feature.shape[1], 1)
                        ax.pcolormesh(t,f, vector_feature.T, cmap="jet")
                        fig.savefig(os.path.join(raw_dir,name + '.png'))
                        # plt.show()