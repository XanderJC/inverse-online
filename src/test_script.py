from src.constants import SIM_COV_DIM, SIM_ACT_DIM, SIM_OUT_DIM
from src.models import (
    AdaptiveLinearModel,
    BehaviouralCloning,
    BehaviouralCloningLSTM,
)  # noqa: F401
from src.data_loading import generate_linear_dataset, get_centre_data


hyperparams = {
    # "covariate_size": SIM_COV_DIM,
    "covariate_size": 63,
    "action_size": SIM_ACT_DIM,
    "outcome_size": SIM_OUT_DIM,
    "memory_hidden_size": 64,
    "memory_layers": 2,
    "memory_dropout": 0,
    "memory_size": 32,
    "outcome_hidden_size": 64,
    "outcome_layers": 2,
    "inf_hidden_size": 16,
    "inf_layers": 2,
    "inf_dropout": 0.5,
    "inf_fc_size": 64,
}

model = AdaptiveLinearModel
# model = BehaviouralCloningLSTM
model = model(**hyperparams)

training_data = get_centre_data("CTR279")
validation_data = get_centre_data("CTR279").get_whole_batch()

# training_data = generate_linear_dataset(100, 50, seed=41310)
# training_data.cut_start_sequence(2500)
# print(training_data.actions.float().mean(axis=1))

# validation_data = generate_linear_dataset(100, 50, seed=41310).get_whole_batch()

model.fit(
    training_data,
    batch_size=100,
    epochs=100,
    learning_rate=0.001,
    validation_set=validation_data,
)

print(list(model.treatment_rule.parameters()))

print(model.treatment_rule.alpha)

print(training_data.actions)
# loss = model.loss(batch)
# print(loss)


# model.load_model()
# validation_set = CancerDataset(fold="validation").get_whole_batch()

# print(F.softmax(model.forward(*validation_set[:3])[0], 2))

# params = model.forward(*validation_set[:3])[1]
# patient_0_0 = np.array(params[0, :, :, 0].detach())

# plt.plot(list(range(59)), patient_0_0)
# plt.show()
