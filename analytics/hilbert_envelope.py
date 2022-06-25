from scipy.signal import hilbert
import numpy as np


# extract the hilbert envelope
def hilbert_envelope(input_signal, sampling_rate, plot=None):
    hilbert_env_ = hilbert(input_signal)
    hilbert_env = np.absolute(hilbert_env_)

    if plot:
        plot(hilbert_env, sampling_rate)

    return hilbert_env
