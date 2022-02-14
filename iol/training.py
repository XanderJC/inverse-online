from iol.constants import SIM_COV_DIM, SIM_ACT_DIM, SIM_OUT_DIM
from iol.models import (
    AdaptiveLinearModel,
    BehaviouralCloning,
    BehaviouralCloningLSTM,
    BehaviouralCloningDeep,
    RCAL,
    CIRL,
)  # noqa: F401
from iol.data_loading import generate_linear_dataset, get_centre_data, SupDataset
import torch
import numpy as np

training_centre = "CTR23901"

training_data = get_centre_data(training_centre, seq_length=50)
validation_data = get_centre_data("CTR279").get_whole_batch()
test_data = get_centre_data("CTR124").get_whole_batch()

hyperparams = {
    "covariate_size": 63,
    "action_size": 2,
    "outcome_size": 1,
    "memory_hidden_size": 32,
    "memory_layers": 1,
    "memory_dropout": 0,
    "memory_size": 16,
    "outcome_hidden_size": 32,
    "outcome_layers": 1,
    "inf_hidden_size": 16,
    "inf_layers": 1,
    "inf_dropout": 0.5,
    "inf_fc_size": 32,
}

model = AdaptiveLinearModel
model = model(**hyperparams)

model.fit(
    training_data,
    batch_size=100,
    epochs=20,
    learning_rate=0.01,
    validation_set=validation_data,
)

losses = model.validation(test_data)
print(losses)
model.save_model("analysis2")
