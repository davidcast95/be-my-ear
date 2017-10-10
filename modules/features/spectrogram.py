import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from python_speech_features import sigproc
import modules.features.data_representation as data_rep

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    zero_samples = np.zeros(int(frameSize / 2),np.int16)
    samples = np.append(zero_samples, sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0])).copy()
    frames *= win
    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins - 1) / max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, int(scale[i]):int(scale[i + 1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale) - 1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i + 1])])]

    return newspec, freqs


def generate(audiopath, binsize=512, numcontext=0, istxt=False):
    if istxt:
        samples = np.loadtxt(audiopath,dtype=np.int16, delimiter=',')
        print(samples.shape)
        samplerate = 16000
    else:
        samplerate, samples = wav.read(audiopath)
        if len(samples.shape) == 2:
            samples = samples[:,0]

    samples = sigproc.preemphasis(samples, 0.97)


    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)  # amplitude to decibel

    # return ims
    _, numcep = ims.shape
    # For each time slice of the training set, we need to copy the context this makes
    # the numcep dimensions vector into a numcep + 2*numcep*numcontext dimensions
    # because of:
    #  - numcep dimensions for the current mfcc feature set
    #  - numcontext*numcep dimensions for each of the past and future (x2) mfcc feature set
    # => so numcep + 2*numcontext*numcep
    train_inputs = np.array([], np.float32)
    train_inputs.resize((ims.shape[0], numcep + 2 * numcep * numcontext))

    # Prepare pre-fix post fix context
    empty_spectrogram = np.array([])
    empty_spectrogram.resize((numcep))

    # Prepare train_inputs with past and future contexts
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
    for time_slice in time_slices:
        # Reminder: array[start:stop:step]
        # slices from indice |start| up to |stop| (not included), every |step|

        # Add empty context data of the correct size to the start and end

        # Pick up to numcontext time slices in the past, and complete with empty
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_spectrogram for empty_slots in range(need_empty_past))
        data_source_past = ims[max(0, time_slice - numcontext):time_slice]

        # Pick up to numcontext time slices in the future, and complete with empty
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(empty_spectrogram for empty_slots in range(need_empty_future))
        data_source_future = ims[time_slice + 1:time_slice + numcontext + 1]

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, numcontext * numcep)
        now = ims[time_slice]
        future = np.reshape(future, numcontext * numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
    return train_inputs

