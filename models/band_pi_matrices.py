import numpy as np
from sklearn.linear_model import LogisticRegression


def train_band_pi_matrices(state_observation_values):
    number_of_states = 4
    pi_vector = np.array([0.25, 0.25, 0.25, 0.25])
    state_values = []
    model = [0, 0, 0, 0]

    for i in range(number_of_states):
        state_values.append(np.vstack((state_observation_values[0][i], state_observation_values[1][i],
                                       state_observation_values[2][i], state_observation_values[3][i],
                                       state_observation_values[4][i])))

    total_observation_sequence = np.vstack((state_values[0], state_values[1], state_values[2], state_values[3]))

    total_obs_distribution = [np.mean(total_observation_sequence, axis=0),
                              np.cov(total_observation_sequence, rowvar=False)]

    for state in range(number_of_states):
        length_of_state_samples = len(state_values[state])
        length_per_other_state = np.floor(length_of_state_samples / (number_of_states - 1))

        min_length_other_class = np.inf

        for other_state in range(number_of_states):
            samples_in_other_state = len(state_values[other_state])

            if other_state != state:
                min_length_other_class = min(min_length_other_class, samples_in_other_state)

        if length_per_other_state > min_length_other_class:
            length_per_other_state = min_length_other_class

        training_data = [
            [0],
            [0, 0],
        ]

        for other_state in range(number_of_states):
            x = state_values[other_state]
            samples_in_other_state = len(state_values[other_state])

            if other_state == state:
                indices = np.random.permutation(samples_in_other_state)
                indices = indices[:int(length_per_other_state) * (4 - 1)]
                training_data[0] = state_values[other_state][indices, :]

            else:
                indices = np.random.permutation(samples_in_other_state)
                indices = indices[:int(length_per_other_state)]
                state_data = state_values[other_state][indices, :]
                training_data[1] = np.vstack((training_data[1], state_data))

        labels = np.zeros((len(training_data[0]) + len(training_data[1]), 1))
        labels[:len(training_data[0])] = 1

        all_data = np.vstack((training_data[0], training_data[1]))

        model[state] = LogisticRegression(multi_class='multinomial').fit(all_data, np.ravel(labels))

    return {
        'pi_vector': pi_vector,
        'total_obs_distribution': total_obs_distribution,
        'model': model
    }
