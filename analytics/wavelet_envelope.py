from pywt import downcoef
import numpy as np


def wavelet_envelope(input_signal, sampling_rate, plot=None):
    cd = downcoef('d', input_signal, 'rbio3.9', level=3)
    cd = np.absolute(cd)
    if plot:
        plot(cd, sampling_rate)
    return cd
