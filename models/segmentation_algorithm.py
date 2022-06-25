import numpy as np
from utils.tools import get_pcg_features, label_pcg_states, get_heart_rate, expand_qt, plot_qt
from models.viterbi_algorithm import viterbi_decode_pcg
from models.band_pi_matrices import train_band_pi_matrices


def train_segmentation_algorithm(train_recordings, train_annotations, features_fs, audio_fs, wavelet=False):
    number_of_states = 4
    num_pcgs = len(train_recordings)

    # state_observation_values = np.zeros((5, 4, 1))

    state_observation_values = []

    for pcgi in range(num_pcgs):
        state_observation_values.append([0, 0, 0, 0])
        pcg_audio = np.squeeze(train_recordings[pcgi])

        s1_locations = train_annotations[pcgi, 0]
        s2_locations = train_annotations[pcgi, 1]

        pcg_features = get_pcg_features(pcg_audio, features_fs, audio_fs, wavelet)['pcg_features']
        pcg_states = label_pcg_states(pcg_features[:, 0], s1_locations, s2_locations, pcg_audio, features_fs)
        for state_i in range(number_of_states):
            state_observation_values[pcgi][state_i] = pcg_features[np.where(pcg_states == state_i + 1)[0], :]

    bpm = train_band_pi_matrices(state_observation_values)

    return {
        'total_obs_distribution': bpm['total_obs_distribution'],
        'pi_vector': bpm['pi_vector'],
        'model': bpm['model']
    }


def run_segmentation_algorithm(test_recording, test_annotations, features_fs, audio_fs,
                               pi_vector, model, total_observation_distribution):
    test_recording = test_recording.flatten()
    pcg_features = get_pcg_features(test_recording, features_fs, audio_fs)['pcg_features']

    s1_locations = test_annotations[0]
    s2_locations = test_annotations[1]

    pcg_states = label_pcg_states(pcg_features[:, 0], s1_locations, s2_locations, test_recording, features_fs)

    heart_rate_info = get_heart_rate(test_recording, audio_fs)

    delta, psi, qt = viterbi_decode_pcg(pcg_features, pi_vector, model, total_observation_distribution,
                                        heart_rate_info['heart_rate'], heart_rate_info['systolic_time_interval'],
                                        features_fs)
    # plot_qt(qt, test_recording, features_fs, audio_fs)
    return expand_qt(qt, features_fs, audio_fs, len(test_recording)),\
           expand_qt(np.squeeze(pcg_states), features_fs, audio_fs, len(test_recording))
