import numpy as np
from scipy.signal import butter, spectrogram
import matplotlib.pyplot as plt

from analytics.homomorphic_envelope import homomorphic_envelope
from analytics.hilbert_envelope import hilbert_envelope
from analytics.wavelet_envelope import wavelet_envelope
from utils.preprocessing import schmidt_spike_removal, normalize, \
    high_pass_filter, downsample, low_pass_filter


def label_pcg_states(envelope, s1, s2, signal, feature_fs, show_plot=None):
    states = np.zeros((len(envelope), 1))
    mean_s1 = 0.122 * feature_fs
    std_s1 = 0.022 * feature_fs
    mean_s2 = 0.092 * feature_fs
    std_s2 = 0.022 * feature_fs

    for s in s1:
        upper_bound = np.round(np.min([len(states), s + mean_s1]))
        states[int(np.max([1, s])) - 1:int(upper_bound)] = 1

    for s in s2:
        lower_bound = int(np.max([s - np.floor(mean_s2 + std_s2), 1]))
        upper_bound = int(np.min([len(states), np.ceil(s + np.floor(mean_s2 + std_s2))]))
        search_window = np.multiply(envelope[lower_bound:upper_bound], 1 - states[lower_bound:upper_bound])

        # find the index of the maximum value
        s2_index = np.argmax(search_window)
        s2_index = int(np.min([len(states), lower_bound + s2_index - 1]))
        upper_bound = np.min([len(states), np.ceil(s2_index + (mean_s2 / 2))])
        states[int(np.max([np.ceil(s2_index - (mean_s2 / 2)), 0])):int(upper_bound)] = 3

        diffs = s1 - s
        pos_diffs = diffs[diffs >= 0]

        if len(pos_diffs):
            index_m = np.argmin(pos_diffs)
            end_pos = s1[index_m] - 1
        else:
            end_pos = len(states)
        states[int(np.ceil(s2_index + (mean_s2 / 2))-1):int(end_pos)] = 4

    def get_index_before(index):
        x, y = index
        if x == 0:
            return False
        return [x - 1, y]

    def get_index_after(index, h):
        x, y = index
        if x == h:
            return False
        return [x + 1, 0]

    first_location_of_definite_state = np.transpose(np.nonzero(states))[0]
    first_location_of_undefined_state = get_index_before(first_location_of_definite_state)

    if not first_location_of_undefined_state:
        print("no zeros")
    if states[first_location_of_definite_state[0], first_location_of_definite_state[1]] == 1:
        for i in range(first_location_of_undefined_state[0] + 1):
            states[i, 0] = 4

    elif states[first_location_of_definite_state[0], first_location_of_definite_state[1]] == 3:
        for i in range(first_location_of_undefined_state[0] + 1):
            states[i, 0] = 2

    last_location_of_definite_state = np.transpose(np.nonzero(states))[-1]
    last_location_of_undefined_state = get_index_after(last_location_of_definite_state, len(states))

    if not last_location_of_undefined_state:
        print("no zeros")
    elif states[last_location_of_definite_state[0], last_location_of_definite_state[1]] == 1:
        for i in range(last_location_of_undefined_state[0]):
            states[i, 0] = 2
    elif states[last_location_of_definite_state[0], last_location_of_definite_state[1]] == 3:
        for i in range(last_location_of_undefined_state[0], len(states)):
            states[i, 0] = 4

    states[states == 0] = 2

    if show_plot:
        length_s = len(states)
        length_e = len(envelope)
        length_si = len(signal)
        T_s = (length_s - 1) / 50
        T_e = (length_e - 1) / 1000
        T_si = (length_si - 1) / 1000

        ts_s = np.linspace(0, T_s, length_s, endpoint=True)
        ts_e = np.linspace(0, T_e, length_e, endpoint=True)
        ts_si = np.linspace(0, T_si, length_si, endpoint=True)

        fig, ax = plt.subplots(figsize=(40, 20))
        ax.plot(ts_s, states)
        ax.plot(ts_si, signal)
        plt.show()
    return states


def get_duration(heart_rate, systolic_time, audio_seg_fs):
    mean_s1 = int(np.round(0.122 * audio_seg_fs))
    std_s1 = int(np.round(0.022 * audio_seg_fs))
    mean_s2 = int(np.round(0.094 * audio_seg_fs))
    std_s2 = int(np.round(0.022 * audio_seg_fs))

    mean_systole = int(np.round(systolic_time * audio_seg_fs)) - mean_s1
    std_systole = (25 / 1000) * audio_seg_fs

    mean_diastole = ((60 / heart_rate) - systolic_time - 0.094) * 50
    std_diastole = 0.07 * mean_diastole + (6 / 1000) * 50

    d_distributions = np.array([
        [mean_s1, std_s1 ** 2],
        [mean_systole, std_systole ** 2],
        [mean_s2, std_s2 ** 2],
        [mean_diastole, std_diastole ** 2]
    ])

    min_systole = mean_systole - 3 * (std_systole + std_s1)
    max_systole = mean_systole + 3 * (std_systole + std_s1)

    min_diastole = mean_diastole - 3 * std_diastole
    max_diastole = mean_diastole + 3 * std_diastole

    min_s1 = (mean_s1 - 3 * std_s2)
    if min_s1 < 1:
        min_s1 = 1

    min_s2 = (mean_s2 - 3 * std_s2)
    if min_s2 < 1:
        min_s2 = 1

    max_s1 = (mean_s1 + 3 * std_s1)
    max_s2 = (mean_s2 + 3 * std_s2)

    return {
        'd_distributions': d_distributions,
        'max_s1': max_s1,
        'min_s1': min_s1,
        'max_s2': max_s2,
        'min_s2': min_s2,
        'max_systole': max_systole,
        'min_systole': min_systole,
        'max_diastole': max_diastole,
        'min_diastole': min_diastole
    }


def get_heart_rate(input_signal, audio_fs):
    input_signal = low_pass_filter(input_signal, 2, 400, audio_fs)
    input_signal = high_pass_filter(input_signal, 2, 25, audio_fs)
    input_signal = schmidt_spike_removal(input_signal, audio_fs)

    hilbert_env = hilbert_envelope(input_signal, audio_fs)
    homomorphic_env = homomorphic_envelope(hilbert_env, audio_fs)

    auto_correlation = np.correlate(homomorphic_env, homomorphic_env, mode='full')
    auto_correlation = auto_correlation[auto_correlation.size // 2:]

    min_index = int(0.5 * audio_fs) - 1
    max_index = int(2 * audio_fs) - 1

    index = np.argmax(auto_correlation[min_index:max_index])
    true_index = index + min_index - 1

    heart_rate = 60 / (true_index / audio_fs)

    max_sys_duration = int(np.round(((60 / heart_rate) * audio_fs) / 2)) - 1
    min_sys_duration = int(np.round(0.2 * audio_fs)) - 1

    pos = np.argmax(auto_correlation[min_sys_duration:max_sys_duration])
    systolic_time_interval = (min_sys_duration + pos) / audio_fs

    return {
        'heart_rate': heart_rate,
        'systolic_time_interval': systolic_time_interval,
    }


def get_pcg_features(audio_data, features_fs, audio_fs, wavelet=False):
    audio_data = low_pass_filter(audio_data, 2, 400, audio_fs)
    audio_data = high_pass_filter(audio_data, 2, 25, audio_fs)
    audio_data = schmidt_spike_removal(audio_data, audio_fs)

    hilbert_env = hilbert_envelope(audio_data, audio_fs)
    normalized_hilbert = downsample(hilbert_env, features_fs, audio_fs)
    normalized_hilbert = normalize(normalized_hilbert)

    homomorphic_env = homomorphic_envelope(hilbert_env, audio_fs)
    normalized_homomorphic = downsample(homomorphic_env, features_fs, audio_fs)
    normalized_homomorphic = normalize(normalized_homomorphic)
    """
    psd = get_psd_feature(audio_data, features_fs, 40, 60)
    psd = downsample(psd, len(normalized_homomorphic), len(psd))
    psd = normalize(psd)"""

    num_of_dims = 2 # change to 3 to include psd
    if wavelet:
        num_of_dims = 3 # change to 4 to include psd
    pcg_features = np.zeros((len(normalized_homomorphic), num_of_dims))
    pcg_features[:, 0] = normalized_homomorphic
    pcg_features[:, 1] = normalized_hilbert
    # pcg_features[:, 2] = psd

    if wavelet:
        wavelet = wavelet_envelope(audio_data, audio_fs)
        wavelet = wavelet[1:len(normalized_homomorphic)]
        normalized_wavelet = downsample(wavelet, features_fs, audio_fs)
        normalized_wavelet = normalize(normalized_wavelet)
        pcg_features[:, 3] = normalized_wavelet

    return {
        'pcg_features': pcg_features,
        'fs': features_fs
    }


def get_psd_feature(data, sampling_frequency, frequency_limit_low, frequency_limit_high):
    f, t, sxx = spectrogram(x=data, window=[sampling_frequency / 40],
                            noverlap=round(sampling_frequency / 80),
                            fs=sampling_frequency, mode='complex', scaling='density')

    low_lim = min(abs(f - frequency_limit_low))
    high_lim = min(abs(f - frequency_limit_high))

    return np.mean(sxx[low_lim:high_lim, :])


def expand_qt(original_qt, old_fs, new_fs, new_length):
    original_qt = np.array(original_qt).T
    expanded_qt = np.zeros(new_length)
    indices_of_changes = np.where(np.abs(np.diff(original_qt)) > 0)

    indices_of_changes = np.append(indices_of_changes, len(original_qt))

    start_index = 0
    for i in range(len(indices_of_changes)):

        end_index = indices_of_changes[i]

        mid_point = round((end_index - start_index) / 2) + start_index

        value_at_mid_point = original_qt[mid_point]

        expanded_start_index = round(np.multiply(np.divide(start_index, old_fs), new_fs))
        expanded_end_index = round(np.multiply(np.divide(end_index, old_fs), new_fs))

        if expanded_end_index > new_length:
            expanded_qt[expanded_start_index:] = value_at_mid_point
        else:
            expanded_qt[expanded_start_index:expanded_end_index] = value_at_mid_point

        start_index = end_index

    return expanded_qt


def plot_qt(pred_state, state, test_recording, features_fs, audio_fs):
    length_s = len(pred_state)
    length_si = len(np.squeeze(test_recording))
    t_s = (length_s - 1) / features_fs
    t_si = (length_si - 1) / audio_fs

    ts_s = np.linspace(0, t_s, length_s, endpoint=True)
    ts_si = np.linspace(0, t_si, length_si, endpoint=True)
    x = normalize(np.squeeze(test_recording))
    fig, ax = plt.subplots(figsize=(40, 20))
    ax.plot(ts_si, x, label='PCG Signal')
    ax.plot(ts_s, pred_state, label='Predicted States', c='g', ls='dashdot')
    ax.plot(ts_s, state, label='States', c='black')

    x1, x2 = np.unique(pred_state, return_counts=True)
    x2_top4 = np.argpartition(x2, -4)[-4:]
    s1, systole, s2, diastole = np.sort(x1[x2_top4])

    ax.axhline(y=diastole, ls='dotted', label='diastole', c='blue')
    ax.axhline(y=s2, ls='dotted', label='s2', c='y')
    ax.axhline(y=systole, ls='dotted', label='systole', c='orange')
    ax.axhline(y=s1, ls='dotted', label='s1', c='r')
    ax.legend()

    plt.show()
