import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from utils.tools import get_duration, normalize


def viterbi_decode_pcg(observation_sequence, pi_vector, model,
                       total_obs_distribution, heart_rate, systolic_time, feature_fs, use_mex=False):
    t = len(observation_sequence)
    n = 4

    max_duration_d = round(60 / heart_rate * feature_fs)

    delta = np.ones((t + max_duration_d - 1, n)) * -np.inf
    psi = np.zeros((t + max_duration_d - 1, n))
    psi_duration = np.zeros((t + max_duration_d - 1, n))
    observation_probs = np.zeros((t, n))

    for i in range(n):
        pihat = model[i].predict_proba(observation_sequence)[:, 1]
        po_correction = multivariate_normal.pdf(observation_sequence, total_obs_distribution[0],
                                                total_obs_distribution[1])
        observation_probs[:, i] = (pihat * po_correction) / pi_vector[i]

    durations = get_duration(heart_rate, systolic_time, feature_fs)

    duration_probs = np.zeros((n, 3 * feature_fs))
    duration_sum = np.zeros((n, 1))

    for state_j in range(n):
        for d in range(max_duration_d):
            duration_probs[state_j, d] = multivariate_normal.pdf(d, durations['d_distributions'][state_j, 0],
                                                                 durations['d_distributions'][state_j, 1])

            if (state_j == 0 and (d < int(durations['min_s1']) or d > int(durations['max_s1']))) or (
                    state_j == 1 and (d < int(durations['min_systole']) or d > int(durations['max_systole']))) or (
                    state_j == 2 and (d < int(durations['min_s2']) or d > int(durations['max_s2']))) or (
                    state_j == 3 and (d < int(durations['min_diastole']) or d > int(durations['max_diastole']))):
                duration_probs[state_j, d] = np.finfo(float).tiny

        duration_sum[state_j] = sum(duration_probs[state_j, :])

    if len(duration_probs) > 3 * feature_fs:
        duration_probs[:, (3 * feature_fs + 1):] = []

    qt = np.zeros((1, len(delta)))

    delta[0, :] = np.log(pi_vector) + np.log(observation_probs[0, :])

    psi[0, :] = -1

    a_matrix = np.zeros((4, 4))
    a_matrix[0, 1] = 1
    a_matrix[1, 2] = 1
    a_matrix[2, 3] = 1
    a_matrix[3, 0] = 1

    if use_mex:
        return viterbi(n, t, a_matrix, max_duration_d, delta, observation_probs, duration_probs, psi, duration_sum)

    for window_t in range(1, t + max_duration_d - 1):
        for j in range(n):
            for d in range(max_duration_d):

                start_t = min(max(0, window_t - d), t - 1)

                end_t = window_t
                if window_t > t:
                    end_t = t - 1

                max_delta = max(delta[start_t, :] + np.log(a_matrix[:, j]))
                max_index = np.argmax(np.log(a_matrix[:, j]) + delta[start_t, :])

                probs = np.prod(observation_probs[start_t:end_t, j])

                if not probs:
                    probs = np.finfo(float).tiny
                emission_probs = np.log(probs)

                if not emission_probs or np.isnan(emission_probs):
                    emission_probs = np.finfo(float).tiny

                delta_temp = max_delta + emission_probs + np.log(np.divide(duration_probs[j, d], duration_sum[j]))

                if delta_temp > delta[window_t, j]:
                    delta[window_t, j] = delta_temp
                    psi[window_t, j] = max_index + 1
                    psi_duration[window_t, j] = d

    temp_delta = np.array(delta[t:, :])

    pos, state = divmod(np.argmax(temp_delta), temp_delta.shape[1])
    pos = t + pos

    offset = pos
    preceding_state = psi[offset, state]

    onset = offset - psi_duration[offset, state]

    qt[0, int(onset)-1:int(offset)] = state + 1

    state = preceding_state
    count = 0

    while onset != 0 and count < 10000:
        offset = onset - 1
        preceding_state = psi[int(offset) - 1, int(state) - 1]
        onset = offset - psi_duration[int(offset) - 1, int(state) - 1]
        if onset < 1:
            onset = 1
        qt[0, int(onset) - 1:int(offset)] = state
        state = preceding_state
        count = count + 1

    qt = qt[0, :t]
    return delta, psi, qt
