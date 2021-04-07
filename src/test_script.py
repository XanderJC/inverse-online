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
    "lstm_layers": 1,
    "lstm_dropout": 0,
    "summary_size": 8,
    "fc_hidden_size": 16,
    "fc_layers": 1,
    "pred_layers": 2,
    "pred_hidden_size": 16,
}

model = BeliefModel
model = model(**hyperparams)
model.fit(dataset, epochs=5, learning_rate=1e-4, validation_set=validation_set)
model.save_model()
