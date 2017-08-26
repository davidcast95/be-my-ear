# be-my-ear
This project is dedicated to final assignment purpose only

Goal of this project :
- a system that can create a Machine Learning Models which be able to configured and save every checkpoints (weight and biases)
- a Machine Learning Model which be able to understands words of Indonesian Language with learn from variety human voices.

Requirements for Windows :
- Python 3.5
- Anaconda 4.4.0
- numpy 1.13.1 or newer
- matplotlib 2.0.2 or newer
- Tensorflow CPU 1.2.1 or newer
- python_speech_features 0.5 by jameslyons (https://github.com/jameslyons/python_speech_features)
- scipy 0.19.1

Installation of Tensorflow using Anaconda :
https://www.tensorflow.org/versions/r0.12/get_started/os_setup#anaconda_installation

Installation of python_speech_features :
running in your conda environment
in this case conda environment named tensorflow
(tensorflow) C:> pip install python_speech_features


Requirements for Ubuntu / Mac OS X:
- Python 3.5
- numpy 1.13.1 or newer
- matplotlib 2.0.2 or newer
- virtualenv 15.1.0
- Tensorflow CPU 1.2.1 or newer
- python_speech_features 0.5 by jameslyons (https://github.com/jameslyons/python_speech_features)
- scipy 0.19.1

Installation of Tensorflow using virtualenv with python 3.5 :
https://www.tensorflow.org/versions/r0.12/get_started/os_setup#virtualenv_installation

Explanation running python in virtualenv :
https://virtualenv.pypa.io/en/stable/userguide/#using-virtualenv-without-bin-python


Run your first example
1. Preprocessing Data
    - prepare your raw data (this refer to dataset directory on this project)
    - raw data consist of two files : .wav and .txt (audio source and target source)
    - charset for target source : { a : 1, b : 2, ... z : 26, SPACE : 27 }
    - run python preprocessing.py RAW_DIR FEATURE_DIR NUM_CONTEXT
      example : python preprocessing.py "D:\Skripsi Dewe\be-my-ear-master\datasets\raw extended" "D:\Skripsi Dewe\feature" 3
    - this script will create dataset files of MFCC features with 3 context (past, now and future)
