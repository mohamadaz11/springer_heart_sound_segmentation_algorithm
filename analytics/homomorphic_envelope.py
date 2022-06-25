from scipy.signal import butter, sosfiltfilt
import numpy as np


def homomorphic_envelope(input_signal, sampling_rate,  plot=None):
    sampling_rate = float(sampling_rate)
    # first order
    order = 1
    sos = butter(order, 8, btype='low', analog=False, output='sos', fs=sampling_rate)
    env = np.exp(sosfiltfilt(sos, np.log(input_signal)))
    env[0] = env[1]
    if plot:
        plot(env, sampling_rate)

    return env
