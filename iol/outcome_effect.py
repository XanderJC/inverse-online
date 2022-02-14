from iol.constants import SIM_COV_DIM, SIM_ACT_DIM, SIM_OUT_DIM
from iol.models import (
    AdaptiveLinearModel,
    BehaviouralCloning,
    BehaviouralCloningLSTM,
)  # noqa: F401
from iol.data_loading import generate_linear_dataset, get_centre_data

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import torch
from scipy import stats

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

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

# training_centre = "CTR23901"
training_centre = "CTR124"

validation_data = get_centre_data(training_centre, seq_length=200).get_whole_batch()
model.load_model("analysis")
info = model.inspection(validation_data)
omega1 = info["omega_1_posterior"]
omega1 = omega1.detach().numpy()

omega0 = info["omega_0_posterior"]
omega0 = omega0.detach().numpy()

omega = omega1  # - omega0

omega = torch.tensor(omega.sum(2))

covariates, actions, outcomes, mask = validation_data

outcomes = outcomes[:, :-1]
actions = actions[:, :-1]

omega = omega[:, 1:] - omega[:, :-1]

# pos = omega[(outcomes > 1)]

# neg = omega[(outcomes < 3)]

pos_0 = omega[(outcomes > 1.0) & (actions == 0)]

neg_0 = omega[(outcomes < -1.0) & (actions == 0)]


pos_1 = omega[(outcomes > 1.0) & (actions == 1)]

neg_1 = omega[(outcomes < -1.0) & (actions == 1)]

density = stats.kde.gaussian_kde(neg_0.numpy())
density2 = stats.kde.gaussian_kde(pos_0.numpy())
density3 = stats.kde.gaussian_kde(neg_1.numpy())
density4 = stats.kde.gaussian_kde(pos_1.numpy())
x = np.arange(-2, 2, 0.1)

print(neg_0.numpy().mean())
print(pos_0.numpy().mean())
print(neg_1.numpy().mean())
print(pos_1.numpy().mean())

fig, ax = plt.subplots()

ax.set_facecolor("whitesmoke")
ax.set_xlabel("Policy Shift")
ax.set_ylabel("Density")

ax.hist(neg_0.numpy(), alpha=0.2, density=True, color="blue")
ax.hist(pos_0.numpy(), alpha=0.2, density=True, color="orange")
# ax.hist(neg_1.numpy(), alpha=0.2, density=True, color="green")
# ax.hist(pos_1.numpy(), alpha=0.2, density=True, color="red")

ax.plot(x, density(x), label="Y -ive, A = 0", color="blue")
ax.plot(x, density2(x), label="Y +ive, A = 0", color="orange")
# ax.plot(x, density3(x), label="Y -ive, A = 1", color="green")
# ax.plot(x, density4(x), label="Y +ive, A = 1", color="red")
plt.xlim(-2, 2)
"""
ax.fill_between(x, density(x), alpha=0.2)
ax.fill_between(x, density2(x), alpha=0.2)
ax.fill_between(x, density3(x), alpha=0.2)
ax.fill_between(x, density4(x), alpha=0.2)
"""
plt.title("Outcome Effect")
plt.legend()
plt.tight_layout()
plt.savefig("policy_shift2.pdf")
# plt.boxplot([pos.numpy(), neg.numpy()])
# plt.hist(neg.numpy(), density=True, alpha=0.5)
# plt.hist(pos.numpy(), density=True, alpha=0.5)
plt.show()
