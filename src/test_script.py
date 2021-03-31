from src.data_loading import get_processed_data, get_cancer_sim_data
from src.constants import *


pickle_map = get_cancer_sim_data(
    chemo_coeff=CHEMO_COEFF, radio_coeff=RADIO_COEFF, b_load=False, b_save=False
)

training_data = pickle_map["training_data"]
validation_data = pickle_map["validation_data"]
scaling_data = pickle_map["scaling_data"]


training_processed = get_processed_data(training_data, scaling_data)

print(training_processed["current_covariates"])
print(training_processed["current_covariates"].shape)

print(training_processed["current_treatments"].shape)
print(training_processed["outputs"].shape)
