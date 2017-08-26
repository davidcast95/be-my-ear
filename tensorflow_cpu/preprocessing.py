
import sys
sys.path.append("../")
import os
import numpy as np
import modules.features.data_representation as data_rep


if len(sys.argv) < 4:
    print ('this method needs 3 args RAW_DIR FEATURE_DIR NUM_CONTEXT')
    print ('RAW_DIR ~> directory of audio raw formatted .WAV')
    print ('FEATURE_DIR ~> directory of target preprocessing files will be created (MFCC)')
    print ('NUM_CONTEXT ~> number of past and future context')
else:
    raw_dir = sys.argv[1]
    feature_dir = sys.argv[2]
    num_context = sys.argv[3]
    preprocess_type = "mfcc"

    if preprocess_type == 'mfcc':
        feature_dir = feature_dir + '/' + preprocess_type
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)
        for root, dirs, files in os.walk(raw_dir, topdown=False):
            for file in files:
                name, ext = file.split('.')
                if ext == 'txt':
                    with open(raw_dir + '/' + file) as filetarget:
                        target = filetarget.read()
                        target = target.replace("\n","")
                        indices_target = data_rep.text_to_indices(target)
                        np.save(feature_dir+'/*-'+name,indices_target)
                        filetarget.close()
                if ext == 'wav':
                    filename = os.path.join(root, file)
                    vector_feature = data_rep.audio_to_feature_representation(filename,num_context)
                    np.save(feature_dir + '/' + name, vector_feature)
                    print (name + " has been saved to " + feature_dir + '/' + name, vector_feature)

