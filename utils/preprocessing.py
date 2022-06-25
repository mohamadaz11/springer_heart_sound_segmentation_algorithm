from scipy.io import loadmat
from scipy.signal import resample, butter, filtfilt, spectrogram
import numpy as np


def load_mat_data(file_name):
    mat = loadmat(file_name)

    annotations = mat['example_data']['example_annotations'][0][0]

    # import the the PCG signals
    audio_signal = mat['example_data']['example_audio_data'][0][0]

    # make the signal in shape(n,)
    return audio_signal[0], annotations


# normalize the envelope
def normalize(input_signal):
    return (input_signal - np.mean(input_signal)) / np.std(input_signal)


def high_pass_filter(input_signal, order, cutoff, sampling_rate, plot=None):
    b, a = butter(order, cutoff / (sampling_rate / 2), 'highpass', output='ba')
    high_pass_filtered_signal = filtfilt(b, a, input_signal)

    if plot:
        plot(high_pass_filtered_signal, sampling_rate)

    return high_pass_filtered_signal


def low_pass_filter(input_signal, order, cutoff, sampling_rate, plot=None):
    b, a = butter(order, cutoff / (sampling_rate / 2), 'lowpass', output='ba')
    low_pass_filtered_signal = filtfilt(b, a, input_signal)

    if plot:
        plot(low_pass_filtered_signal, sampling_rate)

    return low_pass_filtered_signal


# downsample the extracted envelope
def downsample(envelope, feature_fs, sampling_rate, plot=None):
    number_of_samples = int(np.round(len(envelope) * float(feature_fs / sampling_rate)))
    signal = resample(envelope, number_of_samples)

    if plot:
        plot(signal, feature_fs)
    return signal


def schmidt_spike_removal(original_signal, fs, plot=None):
    window_size = int(np.round(fs/2))
    trailing_samples = np.mod(len(original_signal), window_size)
    sample_frames = np.reshape(original_signal, (-1, window_size))

    if trailing_samples:
        sample_frames = np.reshape(original_signal[0:-trailing_samples], (-1, window_size))
    maa = np.max(np.abs(sample_frames), axis=1)

    while len(maa[maa > 3 * np.median(maa)]):
        win_num = np.argmax(maa)
        spike_position = np.argmax(np.abs(sample_frames[win_num]))
        zero_crossings = np.where(np.abs(np.diff(np.sign(sample_frames[win_num]))) > 1)[0]

        spike_start = zero_crossings[:spike_position]
        if len(spike_start):
            spike_start = spike_start[-1]

        else:
            spike_start = 0
        spike_end = zero_crossings[spike_position:]
        if len(spike_end):
            spike_end = spike_end.min()
        else:
            spike_end = window_size - 1
        sample_frames[win_num, spike_start:spike_end] = 0.0001

        maa = np.max(np.abs(sample_frames), axis=1)

    if plot:
        plot(original_signal, fs)

    return np.concatenate((sample_frames.flatten(), original_signal[sample_frames.size:]))
