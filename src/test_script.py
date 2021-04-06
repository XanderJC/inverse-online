from src.data_loading import get_processed_data, get_cancer_sim_data, CancerDataset
from src.constants import *
from src.models import BeliefModel, AdaptiveLinearModel, MLPModel

from pkg_resources import resource_filename
import numpy as np
import torch


dataset = CancerDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
list_train = list(dataloader)
batch = list_train[0]

model = MLPModel(
    covariate_size=C_COV_DIM,
    action_size=C_ACT_DIM,
    outcome_size=C_OUT_DIM,
    hidden_size=64,
    num_layers=2,
)

model.fit(dataset, learning_rate=1e-5)


"""
model = AdaptiveLinearModel(
    covariate_size=C_COV_DIM,
    action_size=C_ACT_DIM,
    outcome_size=C_OUT_DIM,
    lstm_hidden_size=64,
    lstm_layers=1,
    lstm_dropout=0,
    summary_size=10,
    fc_hidden_size=64,
    fc_layers=2,
)

model.fit(dataset, learning_rate=1e-1)


model = BeliefModel(
    covariate_size=C_COV_DIM,
    action_size=C_ACT_DIM,
    outcome_size=C_OUT_DIM,
    lstm_hidden_size=64,
    lstm_layers=1,
    lstm_dropout=0,
    summary_size=10,
    pred_hidden_size=64,
    pred_layers=2,
)

model.fit(dataset)
"""
"""
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


training_data = (
    training_processed["current_covariates"],
    training_processed["current_treatments"],
    training_processed["outputs"],
)


path = resource_filename("src", "data_loading/data/prerun_cancer_training.npz")

np.savez(path, *training_data)
"""
