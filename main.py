import numpy as np
import warnings

from utils.tools import plot_qt
from utils.preprocessing import load_mat_data
from sklearn.metrics import classification_report
from models.segmentation_algorithm import train_segmentation_algorithm, run_segmentation_algorithm

warnings.filterwarnings("ignore")
## Constants
AUDIO_FS = 1000
FEATURES_FS = 50

example_audio_data, example_annotations = load_mat_data('data/example_data.mat')

train_recordings = example_audio_data[:5]
train_annotations = example_annotations[:5]

test_recordings = example_audio_data[55:58]
test_annotations = example_annotations[55:58]

info = train_segmentation_algorithm(train_recordings, train_annotations, FEATURES_FS, AUDIO_FS)
all_states = np.array([])
all_pred_states = np.array([])
for i in range(len(test_recordings)):
    pred_state, state = run_segmentation_algorithm(test_recordings[i], test_annotations[i], FEATURES_FS, AUDIO_FS,
                                                   info['pi_vector'], info['model'], info['total_obs_distribution'])
    plot_qt(pred_state, state, test_recordings[i].flatten(), AUDIO_FS, AUDIO_FS)
    all_states = np.concatenate((all_states, state))
    all_pred_states = np.concatenate((all_pred_states, pred_state))
    print(classification_report(state, pred_state))

print(classification_report(all_states, all_pred_states))
