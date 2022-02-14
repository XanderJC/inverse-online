# Place all your constants here
import os

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)

HYPERPARAMS = {
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

"""
Constants for the cancer simulation code
"""
CHEMO_COEFF = 2
RADIO_COEFF = 2

C_COV_DIM = 2
C_ACT_DIM = 4
C_OUT_DIM = 1


"""
Constants for the basic simulation code
"""

SIM_COV_DIM = 5
SIM_ACT_DIM = 2
SIM_OUT_DIM = 1
