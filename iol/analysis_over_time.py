from iol.models import AdaptiveLinearModel
from iol.data_loading import generate_linear_dataset, get_centre_data

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.manual_seed(41310)

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
# model = BehaviouralCloningLSTM
model = model(**hyperparams)

training_centre = "CTR23901"

validation_data = get_centre_data(training_centre, seq_length=200).get_whole_batch()
model.load_model("analysis")

info = model.inspection(validation_data)

omega1 = info["omega_1_prior"]
omega1 = omega1.detach().numpy()

omega0 = info["omega_0_prior"]
omega0 = omega0.detach().numpy()

omega = omega1 - omega0

omega = np.abs(omega)
omega = omega / np.expand_dims(omega.sum(2), 2)


mean = omega.mean(axis=0)
num_trajs = omega.shape[0]
print(num_trajs)

# for i in range(num_trajs):
# plt.plot(mean[:, 6], color="red", alpha=1.0)
# plt.title(f"index {j}")
# plt.plot(mean[:, 7], color="blue", alpha=1.0)
# plt.plot(mean[:, 8], color="green", alpha=1.0)
# plt.plot(mu[i, :, 3], color="orange", alpha=0.01)

index = [3, 5, 21, 17]
# index = range(25)
for j in index:
    # plt.plot(omega[0, :, j], alpha=0.5)
    plt.plot(mean[:, j], alpha=0.5, label=f"{j}")
plt.legend()

"""
for j in range(1):
    plt.plot(omega[j, :, 4], color="red", alpha=0.05)
    # plt.title(f"index {j}")
    plt.plot(omega[j, :, 5], color="blue", alpha=0.05)
    plt.plot(omega[j, :, 6], color="green", alpha=0.05)
"""
plt.show()
