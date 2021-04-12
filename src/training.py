from src.data_loading import CancerDataset
from src.constants import C_COV_DIM, C_ACT_DIM, C_OUT_DIM

from src.models import BeliefModel, AdaptiveLinearModel, MLPModel  # noqa: F401


dataset = CancerDataset()
validation_set = CancerDataset(fold="validation").get_whole_batch()

hyperparams = {
    "covariate_size": C_COV_DIM,
    "action_size": C_ACT_DIM,
    "outcome_size": C_OUT_DIM,
    "lstm_hidden_size": 16,
    "lstm_layers": 2,
    "lstm_dropout": 0.2,
    "summary_size": 32,
    "fc_hidden_size": 16,
    "fc_layers": 1,
    "pred_hidden_size": 16,
    "pred_layers": 1,
}

model = AdaptiveLinearModel
model = model(**hyperparams)
model.fit(dataset, epochs=20, learning_rate=1e-4, validation_set=validation_set)
model.save_best()
