"""
CODE FROM: https://github.com/ioanabica/Counterfactual-Recurrent-Network
# Copyright (c) 2020, Ioana Bica
"""

import numpy as np


def get_processed_data(raw_sim_data, scaling_params):
    """
    Create formatted data to train both encoder and seq2seq atchitecture.
    """
    mean, std = scaling_params

    horizon = 1
    offset = 1

    mean["chemo_application"] = 0
    mean["radio_application"] = 0
    std["chemo_application"] = 1
    std["radio_application"] = 1

    input_means = mean[
        ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
    ].values.flatten()
    input_stds = std[
        ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
    ].values.flatten()

    # Continuous values
    cancer_volume = (raw_sim_data["cancer_volume"] - mean["cancer_volume"]) / std[
        "cancer_volume"
    ]
    patient_types = (raw_sim_data["patient_types"] - mean["patient_types"]) / std[
        "patient_types"
    ]

    patient_types = np.stack(
        [patient_types for t in range(cancer_volume.shape[1])], axis=1
    )

    # Binary application
    chemo_application = raw_sim_data["chemo_application"]
    radio_application = raw_sim_data["radio_application"]
    sequence_lengths = raw_sim_data["sequence_lengths"]

    # Convert treatments to one-hot encoding

    treatments = np.concatenate(
        [
            chemo_application[:, :-offset, np.newaxis],
            radio_application[:, :-offset, np.newaxis],
        ],
        axis=-1,
    )

    one_hot_treatments = np.zeros(shape=(treatments.shape[0], treatments.shape[1], 4))
    for patient_id in range(treatments.shape[0]):
        for timestep in range(treatments.shape[1]):
            if (
                treatments[patient_id][timestep][0] == 0
                and treatments[patient_id][timestep][1] == 0
            ):
                one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
            elif (
                treatments[patient_id][timestep][0] == 1
                and treatments[patient_id][timestep][1] == 0
            ):
                one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
            elif (
                treatments[patient_id][timestep][0] == 0
                and treatments[patient_id][timestep][1] == 1
            ):
                one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
            elif (
                treatments[patient_id][timestep][0] == 1
                and treatments[patient_id][timestep][1] == 1
            ):
                one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]

    one_hot_previous_treatments = one_hot_treatments[:, :-1, :]

    current_covariates = np.concatenate(
        [
            cancer_volume[:, :-offset, np.newaxis],
            patient_types[:, :-offset, np.newaxis],
        ],
        axis=-1,
    )
    outputs = cancer_volume[:, horizon:, np.newaxis]

    output_means = mean[["cancer_volume"]].values.flatten()[
        0
    ]  # because we only need scalars here
    output_stds = std[["cancer_volume"]].values.flatten()[0]

    print(outputs.shape)

    # Add active entires
    active_entries = np.zeros(outputs.shape)

    for i in range(sequence_lengths.shape[0]):
        sequence_length = int(sequence_lengths[i])
        active_entries[i, :sequence_length, :] = 1

    raw_sim_data["current_covariates"] = current_covariates
    raw_sim_data["previous_treatments"] = one_hot_previous_treatments
    raw_sim_data["current_treatments"] = one_hot_treatments
    raw_sim_data["outputs"] = outputs
    raw_sim_data["active_entries"] = active_entries

    raw_sim_data["unscaled_outputs"] = (
        outputs * std["cancer_volume"] + mean["cancer_volume"]
    )
    raw_sim_data["input_means"] = input_means
    raw_sim_data["inputs_stds"] = input_stds
    raw_sim_data["output_means"] = output_means
    raw_sim_data["output_stds"] = output_stds

    return raw_sim_data
