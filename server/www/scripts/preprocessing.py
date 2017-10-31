import sys
sys.path.append("../../../")
import os
import numpy as np
import modules.features.data_representation as data_rep
import modules.features.spectrogram as spectrogram

target_vector_dir = '../preprocessed_data'


def mffc_rep_wav(root,file, num_context):
    filename = os.path.join(root,file)
    fileattr = file.split('.')
    vector_feature = data_rep.mfcc_num_context(filename, num_context)
    save_target = os.path.join(target_vector_dir, fileattr[0])
    np.save(save_target, vector_feature)

    print("done convert " + filename + ' to mfcc with context : ' + str(num_context) + ' save to ' + save_target)