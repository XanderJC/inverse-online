from src.constants import C_COV_DIM, C_ACT_DIM, C_OUT_DIM

from src.models import BeliefModel, AdaptiveLinearModel, MLPModel  # noqa: F401

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
model.load_model()

from src.data_loading import CancerDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

validation_set = CancerDataset(fold="validation").get_whole_batch()

# print(F.softmax(model.forward(*validation_set[:3])[0], 2))

params = model.forward(*validation_set[:3])[1]
patient_0_0 = np.array(params[0, :, :, 0].detach())

plt.plot(list(range(59)), patient_0_0)
plt.show()
