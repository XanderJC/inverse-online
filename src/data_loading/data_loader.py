import torch
import numpy as np
from pkg_resources import resource_filename


class CancerDataset(torch.utils.data.Dataset):
    def __init__(self, load_prerun=True, fold="training"):
        super(CancerDataset, self).__init__()

        if load_prerun:

            path_tail = f"data_loading/data/prerun_cancer_{fold}.npz"
            path = resource_filename("src", path_tail)

            data = np.load(path)

        else:
            raise NotImplementedError

        self.covariates = torch.tensor(data["arr_0"], dtype=torch.double)
        self.actions = torch.tensor(data["arr_1"], dtype=torch.double)
        self.outcomes = torch.tensor(data["arr_2"], dtype=torch.double)
        self.mask = torch.ones(self.outcomes.shape)

        self.N = len(self.covariates)

    def __len__(self):
        "Total number of samples"
        return self.N

    def __getitem__(self, index):
        "Generates one batch of data"
        return (
            self.covariates[index],
            self.actions[index],
            self.outcomes[index],
            self.mask[index],
        )

    def get_whole_batch(self):
        "Returns all data as a single batch"
        return self.covariates, self.actions, self.outcomes
