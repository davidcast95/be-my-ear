import sys
sys.path.append("../")
import os
import numpy as np
import modules.features.data_representation as data_rep
from scipy import signal
from scipy.io import wavfile


if len(sys.argv) < 5:
    print ('this method needs 3 args RAW_DIR FEATURE_DIR NUM_CONTEXT')
    print ('RAW_DIR ~> directory of audio raw formatted .WAV')
    print ('FEATURE_DIR ~> directory of target preprocessing files will be created (MFCC)')
    print ('FEATURE_TYPE ~> mffc / spectrogram')
    print ('NUM_CONTEXT ~> number of past and future context')
else:
    raw_dir = sys.argv[1]
    feature_dir = sys.argv[2]
    preprocess_type = sys.argv[3]
    num_context = int(sys.argv[4])
    if not os.path.exists(raw_dir):
        print ("RAW_DIR doesn't exist")
    else:
        if preprocess_type == 'mfcc':
            feature_dir = os.path.join(feature_dir, preprocess_type)
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)
            for root, dirs, files in os.walk(raw_dir, topdown=False):
                for file in files:
                    name, ext = file.split('.')
                    if ext == 'txt':
                        with open(os.path.join(raw_dir,file)) as filetarget:
                            target = filetarget.read()
                            target = target.replace("\n","")
                            indices_target = data_rep.text_to_indices(target)
                            np.save(os.path.join(feature_dir,'_' + name),indices_target)
                            filetarget.close()
                    if ext == 'wav':
                        filename = os.path.join(root, file)
                        vector_feature = data_rep.audio_to_feature_representation(filename,num_context)
                        target_vector_dir = os.path.join(feature_dir, name)
                        np.save(target_vector_dir, vector_feature)
                        print (name + " has been saved to " + target_vector_dir)
        if preprocess_type == 'spectrogram':
            feature_dir = os.path.join(feature_dir, preprocess_type)
            if not os.path.exists(feature_dir):
                os.makedirs(feature_dir)
            for root, dirs, files in os.walk(raw_dir, topdown=False):
                for file in files:
                    name, ext = file.split('.')
                    if ext == 'txt':
                        with open(os.path.join(raw_dir, file)) as filetarget:
                            target = filetarget.read()
                            target = target.replace("\n", "")
                            indices_target = data_rep.text_to_indices(target)
                            np.save(os.path.join(feature_dir,'_' + name),indices_target)
                            filetarget.close()
                    if ext == 'wav':
                        filename = os.path.join(root, file)
                        fs, audio = wavfile.read(filename)
                        f, t, vector_feature = signal.spectrogram(audio, fs)
                        target_vector_dir = os.path.join(feature_dir, name)
                        np.save(target_vector_dir, vector_feature.T)
                        print(name + " has been saved to " + target_vector_dir)


