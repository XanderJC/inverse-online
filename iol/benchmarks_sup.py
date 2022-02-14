from iol.models import (
    BehaviouralCloning,
    BehaviouralCloningDeep,
    AdaptiveLinearModel,
    RCAL,
)  # noqa: F401
from iol.data_loading import generate_linear_dataset, get_centre_data, SupDataset
import numpy as np
import pickle
import torch

torch.manual_seed(41310)

hyperparams = {
    "covariate_size": 76,
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
    "hidden_size": 32,
    "num_layers": 2,
}

"""
model_dict = {
    "logistic": BehaviouralCloning,
    "bc": BehaviouralCloningDeep,
    "rcal": RCAL,
}
"""
model_dict = {"pte": AdaptiveLinearModel}

num_runs = 10

training_data = SupDataset("cf", max_seq_length=50)
validation_data = SupDataset("cf", max_seq_length=50, test=True).get_whole_batch()
test_data = SupDataset("cf", max_seq_length=50, test=True).get_whole_batch()
results = {}

for models in model_dict.keys():

    model_results = []

    for run in range(num_runs):
        model = model_dict[models](**hyperparams)

        model.fit(
            training_data,
            batch_size=100,
            epochs=15,
            learning_rate=0.001,
            validation_set=validation_data,
        )

        result = model.validation(test_data)
        model_results.append(result)

    results[models] = model_results

results_array = np.zeros((len(model_dict), num_runs, 4))

for i, model in enumerate(results.keys()):

    result = results[model]
    for j, res in enumerate(result):
        results_array[i, j, 0] = res["ACC"]
        results_array[i, j, 1] = res["AUC"]
        results_array[i, j, 2] = res["APR"]
        results_array[i, j, 3] = res["NLL"]

    print(f"{model}:")
    print(
        f"ACC: {round(results_array.mean(1)[i,0],3)} +- {round(results_array.std(1)[i,0],3)}"
    )
    print(
        f"AUC: {round(results_array.mean(1)[i,1],3)} +- {round(results_array.std(1)[i,1],3)}"
    )
    print(
        f"APR: {round(results_array.mean(1)[i,2],3)} +- {round(results_array.std(1)[i,2],3)}"
    )
    print(
        f"NLL: {round(results_array.mean(1)[i,3],3)} +- {round(results_array.std(1)[i,3],3)}"
    )
"""
filename = "ward_bench_results.pkl"
with open(filename, "wb") as file:
    pickle.dump(results_array, file)
"""
