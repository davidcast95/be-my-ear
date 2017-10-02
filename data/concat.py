import sys
sys.path.append("../")
import os
import numpy as np
import modules.features.data_representation as data_rep
from scipy import signal
from scipy.io import wavfile


if len(sys.argv) < 4:
    print ('this method needs 3 args RAW_DIR FEATURE_DIR NUM_CONTEXT')
    print ('RAW_DIR ~> directory of audio raw formatted .WAV')
    print ('TARGET_DIR ~> directory of target directory')
    print ('NUM_CONCAT ~> number of WAV concat')
else:
    raw_dir = sys.argv[1]
    target_dir = sys.argv[2]
    num_concat = int(sys.argv[3])
    if not os.path.exists(raw_dir):
        print ("RAW_DIR doesn't exist")
    else:
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        for root, dirs, files in os.walk(raw_dir, topdown=False):
            text = ""
            wav = np.array([[0,0]])
            sr = 16000
            i = 0
            for file in files:
                name, ext = file.split('.')
                if i == num_concat:
                    i = 0
                    print(text)
                    with open(os.path.join(target_dir,name + '.txt'),"w") as filetarget:
                        filetarget.write(text)
                        filetarget.close()
                    wavfile.write(os.path.join(target_dir,name + '.wav'),sr,wav)
                    print(wav)
                    text = ""
                    wav = np.array([[0,0]])
                if ext == 'wav':
                    i+=1
                    with open(os.path.join(raw_dir,name + '.txt')) as filetarget:
                        target = filetarget.read()
                        target = target.replace("\n"," ")
                        text += target + " "
                        filetarget.close()
                    filename = os.path.join(root, file)
                    sr, readwav = wavfile.read(os.path.join(raw_dir,file))
                    print(readwav)
                    wav = np.concatenate((wav,readwav),axis=0)
