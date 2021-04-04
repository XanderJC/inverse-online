# Place all your constants here
import os

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)

"""
Constants for the cancer simulation code
"""
CHEMO_COEFF = 2
RADIO_COEFF = 2

C_COV_DIM = 2
C_ACT_DIM = 4
C_OUT_DIM = 1
