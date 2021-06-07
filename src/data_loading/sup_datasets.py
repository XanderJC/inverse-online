import torch
import numpy as np
import pandas as pd

from medkit.domains import CFDomain, ICUDomain, WardDomain
from medkit.bases import standard_dataset


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

        self.covariates = None
        self.actions = None
        self.outcomes = None
        self.mask = None
        self.N = None

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
        return self.covariates, self.actions, self.outcomes, self.mask

    def cut_start_sequence(self, num):

        self.covariates = self.covariates[:, num:, :]
        self.actions = self.actions[:, num:]
        self.outcomes = self.outcomes[:, num:]
        self.mask = self.mask[:, num:]


class SupDataset(BaseDataset):
    """
    Dataset to be passed to a torch DataLoader
    """

    def __init__(self, domain, max_seq_length=50, test=False):
        super(SupDataset, self).__init__()

        domain_dict = {"icu": ICUDomain, "cf": CFDomain, "ward": WardDomain}

        domain = domain_dict[domain](y_dim=2)
        data = standard_dataset(domain, test=test)

        self.covariates = data.X_series.double()
        self.actions = data.y_series.long()
        self.outcomes = torch.zeros(data.y_series.shape, dtype=torch.double)
        self.mask = data.X_mask.int()
        self.N = data.N


if __name__ == "__main__":

    training_data = SupDataset("cf", max_seq_length=50)

    print(training_data.actions.shape)
