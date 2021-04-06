from src.data_loading import CancerDataset
from src.constants import C_COV_DIM, C_ACT_DIM, C_OUT_DIM

from src.models import BeliefModel, AdaptiveLinearModel, MLPModel

import torch


dataset = CancerDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
list_train = list(dataloader)
batch = list_train[0]

validation_set = CancerDataset(fold="validation").get_whole_batch()
# print(len(validation_set[0]))
test_set = CancerDataset(fold="test").get_whole_batch()

model = MLPModel(
    covariate_size=C_COV_DIM,
    action_size=C_ACT_DIM,
    outcome_size=C_OUT_DIM,
    hidden_size=64,
    num_layers=2,
)

model.fit(dataset, learning_rate=1e-5, validation_set=test_set)


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

# model.fit(dataset, learning_rate=1e-1)


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

# model.fit(dataset)
