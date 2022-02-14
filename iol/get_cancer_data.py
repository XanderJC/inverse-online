from iol.data_loading import get_processed_data, get_cancer_sim_data
from iol.constants import CHEMO_COEFF, RADIO_COEFF

from pkg_resources import resource_filename
import numpy as np
import torch

seed = 41310
torch.manual_seed(seed)
np.random.seed(seed)

pickle_map = get_cancer_sim_data(
    chemo_coeff=CHEMO_COEFF, radio_coeff=RADIO_COEFF, b_load=False, b_save=False
)

training_data = pickle_map["training_data"]
validation_data = pickle_map["validation_data"]
test_data = pickle_map["test_data_factuals"]
scaling_data = pickle_map["scaling_data"]

print(training_data["sequence_lengths"][:20])

training_processed = get_processed_data(training_data, scaling_data)
validation_processed = get_processed_data(validation_data, scaling_data)
test_processed = get_processed_data(test_data, scaling_data)

print(training_processed["current_covariates"])
print(training_processed["current_covariates"].shape)

print(training_processed["current_treatments"].shape)
print(training_processed["outputs"].shape)


training_data = (
    training_processed["current_covariates"],
    training_processed["current_treatments"],
    training_processed["outputs"],
    training_data["sequence_lengths"],
)

validation_data = (
    validation_processed["current_covariates"],
    validation_processed["current_treatments"],
    validation_processed["outputs"],
    validation_data["sequence_lengths"],
)

test_data = (
    test_processed["current_covariates"],
    test_processed["current_treatments"],
    test_processed["outputs"],
    test_data["sequence_lengths"],
)


path_training = resource_filename("src", "data_loading/data/prerun_cancer_training.npz")
np.savez(path_training, *training_data)

path_validation = resource_filename(
    "src", "data_loading/data/prerun_cancer_validation.npz"
)
np.savez(path_validation, *validation_data)

path_test = resource_filename("src", "data_loading/data/prerun_cancer_test.npz")
np.savez(path_test, *test_data)
