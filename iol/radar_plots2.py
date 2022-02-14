from iol.constants import SIM_COV_DIM, SIM_ACT_DIM, SIM_OUT_DIM
from iol.models import (
    AdaptiveLinearModel,
    BehaviouralCloning,
    BehaviouralCloningLSTM,
)  # noqa: F401
from iol.data_loading import generate_linear_dataset, get_centre_data

import matplotlib.pyplot as plt
import numpy as np

import plotly.graph_objects as go
import plotly.offline as pyo

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

validation_data = get_centre_data("CTR23901").get_whole_batch()

model.load_model("analysis2")

info = model.inspection(validation_data)

omega1 = info["omega_1_posterior"]
omega1 = omega1.detach().numpy()

omega0 = info["omega_0_posterior"]
omega0 = omega0.detach().numpy()

num_cov = 15
categories = [f"{i}" for i in range(num_cov)]
categories = [
    "Age",
    "Gender",
    "Height",
    "Weight",
    "ABO",
    "BMI",
    "Creatinine",
    "INR",
    "Total Bilirubin",
    "MELD Score",
    "Serum Sodium",
    "DIAL_TX",
    "MELD Status",
    "Status 1",
    "Albumin",
]
# categories = [*categories, categories[0]]

accept = omega1[0, 45, :num_cov]
reject = omega0[0, 0, :num_cov]

# restaurant_1 = mu[0, 0, :num_cov]
# restaurant_2 = mu[0, 25, :num_cov]
# restaurant_3 = mu[0, 49, :num_cov]

# restaurant_1 = [4, 4, 5, 4, 3]
# restaurant_2 = [5, 5, 4, 5, 2]
# restaurant_3 = [3, 4, 5, 3, 5]
# restaurant_1 = [*restaurant_1, restaurant_1[0]]
# restaurant_2 = [*restaurant_2, restaurant_2[0]]
# restaurant_3 = [*restaurant_3, restaurant_3[0]]


fig = go.Figure(
    data=[
        go.Scatterpolar(
            r=omega1[0, 0, :num_cov], theta=categories, fill="toself", name="Accept"
        ),
        # go.Scatterpolar(r=reject, theta=categories, fill="toself", name="Reject"),
    ],
    layout=go.Layout(
        polar={"radialaxis": {"visible": True}},
        showlegend=True,
    ),
)

# pyo.plot(fig)
fig.show()
# fig.write_image("radar_plot_0.pdf")
fig = go.Figure(
    data=[
        go.Scatterpolar(
            r=omega1[0, 25, :num_cov], theta=categories, fill="toself", name="Accept"
        ),
        # go.Scatterpolar(r=reject, theta=categories, fill="toself", name="Reject"),
    ],
    layout=go.Layout(
        polar={"radialaxis": {"visible": True}},
        showlegend=True,
    ),
)

# pyo.plot(fig)
fig.show()
# fig.write_image("radar_plot_25.pdf")
fig = go.Figure(
    data=[
        go.Scatterpolar(
            r=omega1[0, 45, :num_cov], theta=categories, fill="toself", name="Accept"
        ),
        # go.Scatterpolar(r=reject, theta=categories, fill="toself", name="Reject"),
    ],
    layout=go.Layout(
        polar={"radialaxis": {"visible": True}},
        showlegend=True,
    ),
)

# pyo.plot(fig)
fig.show()
# fig.write_image("radar_plot_45.pdf")


exit()


mu = mu.detach().numpy()
num_trajs = mu.shape[0]
print(num_trajs)
for j in range(1):
    for i in range(num_trajs):
        plt.plot(mu[i, :, j], color="red", alpha=0.05)
        plt.title(f"index {j}")
    # plt.plot(mu[i, :, 1], color="blue", alpha=0.01)
    # plt.plot(mu[i, :, 2], color="green", alpha=0.01)
    # plt.plot(mu[i, :, 3], color="orange", alpha=0.01)
    plt.show()
