import numpy as np
from python_speech_features import mfcc
from scipy.io import wavfile


#===================================Char and Integer Representation===================================

charset = ['','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','r','s','t','u','v','w','x','y','z',' ']

def text_to_indices(text):
    return np.array([charset.index(char) for char in text.lower()])

def indices_to_text(indices):
    str = ''
    for index in indices:
        str += charset[index]
    return str

#===================================Sparse Representation===================================

#This function get from https://github.com/tensorflow/tensorflow/tree/679f95e9d8d538c3c02c0da45606bab22a71420e/tensorflow/python/kernel_tests

def SimpleSparseTensorFrom(x):
  """Create a very simple SparseTensor with dimensions (batch, time).
  Args:
    x: a list of lists of type int
  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]

  return x_ix, x_val, x_shape

def maximum(x):
    max = len(x[0])
    for i in range(1,len(x)):
        if max < len(x[i]):
            max = len(x[i])
    return max

def sparse_dataset(x):
    max = maximum(x)
    n = len(x)
    sparse_datasets = np.zeros((n,max,x[0].shape[1]))
    for i in range(0,n):
        t = x[i].shape[0]
        sparse_datasets[i, 0:t] = x[i]
    return sparse_datasets

#===================================Audio Representation===================================

#This function get from https://svds.com/tensorflow-rnn-tutorial/

def audio_to_feature_representation(audio_filename, numcontext):
    # Load wav files
    fs, audio = wavfile.read(audio_filename)

    # Get mfcc coefficients
    if fs == 8000:
        numcep = 13
    else:
        numcep = 26
    orig_inputs = mfcc(audio, samplerate=fs, numcep=numcep)

    # We only keep every second feature (BiRNN stride = 2)
    orig_inputs = orig_inputs[::2]

    # For each time slice of the training set, we need to copy the context this makes
    # the numcep dimensions vector into a numcep + 2*numcep*numcontext dimensions
    # because of:
    #  - numcep dimensions for the current mfcc feature set
    #  - numcontext*numcep dimensions for each of the past and future (x2) mfcc feature set
    # => so numcep + 2*numcontext*numcep
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))

    # Prepare pre-fix post fix context
    empty_mfcc = np.array([])
    empty_mfcc.resize((numcep))

    # Prepare train_inputs with past and future contexts
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
    for time_slice in time_slices:
        # Reminder: array[start:stop:step]
        # slices from indice |start| up to |stop| (not included), every |step|

        # Add empty context data of the correct size to the start and end
        # of the MFCC feature matrix

        # Pick up to numcontext time slices in the past, and complete with empty
        # mfcc features
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_mfcc for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext):time_slice]

        # Pick up to numcontext time slices in the future, and complete with empty
        # mfcc features
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_mfcc for empty_slots in range(need_empty_future))
        data_source_future = orig_inputs[time_slice + 1:time_slice + numcontext + 1]

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
    return train_inputs
