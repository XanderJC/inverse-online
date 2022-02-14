from iol.constants import SIM_COV_DIM, SIM_ACT_DIM, SIM_OUT_DIM
from iol.models import (
    AdaptiveLinearModel,
    BehaviouralCloning,
    BehaviouralCloningLSTM,
    BehaviouralCloningDeep,
    RCAL,
    CIRL,
)  # noqa: F401
from iol.data_loading import generate_linear_dataset, get_centre_data  # , SupDataset
import torch
import numpy as np

torch.manual_seed(41310)
np.random.seed(41310)

hyperparams = {
    "covariate_size": 5,
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
    "hidden_size": 64,
}

model = AdaptiveLinearModel
# model = BehaviouralCloningLSTM
model = model(**hyperparams)
"""
training_data = SupDataset("cf", max_seq_length=50)
validation_data = SupDataset("cf", max_seq_length=50, test=True).get_whole_batch()
test_data = SupDataset("cf", max_seq_length=50, test=True).get_whole_batch()
"""
"""
training_centre = "CTR23901"

training_data = get_centre_data(training_centre, seq_length=50)
validation_data = get_centre_data("CTR279").get_whole_batch()
test_data = get_centre_data("CTR124").get_whole_batch()
"""
# loss = model.loss(validation_data)
# print(loss)


training_data = generate_linear_dataset(10000, 50, seed=41310)
validation_data = generate_linear_dataset(1000, 50, seed=41311).get_whole_batch()
test_data = generate_linear_dataset(1000, 50, seed=41311).get_whole_batch()

# training_data.cut_start_sequence(2500)
# print(training_data.actions.float().mean(axis=1))

# validation_data = generate_linear_dataset(1000, 50, seed=28).get_whole_batch()
# test_data = generate_linear_dataset(1000, 50, seed=28).get_whole_batch()


model.fit(
    training_data,
    batch_size=100,
    epochs=5,
    learning_rate=0.01,
    validation_set=validation_data,
)

losses = model.validation(test_data)
print(losses)
# model.save_model("analysis")

# print(list(model.treatment_rule.parameters()))

# print(model.treatment_rule.alpha)

# print(training_data.actions)
# loss = model.loss(batch)
# print(loss)


# model.load_model()
# validation_set = CancerDataset(fold="validation").get_whole_batch()

# print(F.softmax(model.forward(*validation_set[:3])[0], 2))

# params = model.forward(*validation_set[:3])[1]
# patient_0_0 = np.array(params[0, :, :, 0].detach())

# plt.plot(list(range(59)), patient_0_0)
# plt.show()
