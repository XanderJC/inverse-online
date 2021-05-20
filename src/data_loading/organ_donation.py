import numpy as np
import torch
import pickle
from pkg_resources import resource_filename


class OrganDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(OrganDataset, self).__init__()

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


def get_centre_data(centre, seq_length=50):

    path_tail = f"data_loading/centres_cleaned/{centre}_data.pkl"
    path = resource_filename("src", path_tail)

    with open(path, "rb") as file:
        data = pickle.load(file)

    X = data[0]
    Y = data[1]
    A = data[2]

    num_pairs, cov_size = X.shape
    num_trajs = int(num_pairs / seq_length)

    Xs = np.zeros((num_trajs, seq_length, cov_size))
    Ys = np.zeros((num_trajs, seq_length))
    As = np.zeros((num_trajs, seq_length))
    mask = np.ones((num_trajs, seq_length))

    for i in range(num_pairs):
        k = i % seq_length
        j = int(i / seq_length)
        if (j < num_trajs) & (k < seq_length):
            Xs[j, k, :] = X[i, :]
            Ys[j, k] = Y[i]
            As[j, k] = A[i]

    dataset = OrganDataset()

    dataset.covariates = torch.tensor(Xs)
    dataset.actions = torch.tensor(As, dtype=int)
    dataset.outcomes = torch.tensor(Ys)
    dataset.mask = torch.tensor(mask, dtype=int)

    dataset.N = len(dataset.covariates)

    return dataset
