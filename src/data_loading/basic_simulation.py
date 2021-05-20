import numpy as np
import torch

from scipy.special import expit

EPS = 0.00001


def simulate_x_and_pos(
    n,
    d: int = 5,
    covariate_model=None,
    covariate_model_params: dict = None,
    mu_0_model=None,
    mu_0_model_params: dict = None,
    mu_1_model=None,
    mu_1_model_params: dict = None,
    error_sd: float = 0.1,
    seed: int = 42,
    no_seed: bool = False,
):
    """ Simulate data matrix and all potential outcomes"""
    # set defaults
    if covariate_model is None:
        covariate_model = normal_covariate_model

    if covariate_model_params is None:
        covariate_model_params = {}

    if mu_0_model is None:
        mu_0_model = mu0_linear

    if mu_0_model_params is None:
        mu_0_model_params = {}

    if mu_1_model is None:
        mu_1_model = mu1_linear

    if mu_1_model_params is None:
        mu_1_model_params = {}

    if not no_seed:
        np.random.seed(seed)

    X = covariate_model(n=n, d=d, **covariate_model_params)
    mu_0, params0 = mu_0_model(X, **mu_0_model_params)
    mu_1, params1 = mu_1_model(X, mu_0=mu_0, **mu_1_model_params)
    cate = mu_1 - mu_0

    # generate observables
    err = np.random.normal(0, error_sd, n)
    y_0 = mu_0 + err
    y_1 = mu_1 + err

    pos = np.c_[y_0, y_1]

    return {"X": X, "pos": pos, "mu_0": mu_0, "mu_1": mu_1, "cate": cate}, [
        params0,
        params1,
    ]


def normal_covariate_model(n, d, rho=0.3, var=1 / 20):
    mean_vec = np.zeros(d)
    Sigma_x = (np.ones([d, d]) * rho + np.identity(d) * (1 - rho)) * var
    X = np.random.multivariate_normal(mean_vec, Sigma_x, n)
    return X


def mu0_linear(X, sparsity: float = 0.5, return_model_params: bool = True):
    # linear function of X
    n_cov = X.shape[1]
    # beta = np.random.choice(
    #    [0, 1], replace=True, size=n_cov, p=[1 - sparsity, sparsity]
    # )
    beta = np.array([1, 1, 0, 0, 1])
    mu0 = np.dot(X, beta)
    if return_model_params:
        return mu0, beta
    else:
        return mu0


def mu1_linear(X, mu_0, sparsity: float = 0.5, return_model_params: bool = True):
    # linear function of X, add to mu_0
    n_cov = X.shape[1]
    # beta = np.random.choice(
    #    [0, 1], replace=True, size=n_cov, p=[1 - sparsity, sparsity]
    # )
    beta = np.array([0, 1, 1, 0, 0])
    mu1 = np.dot(X, beta) + mu_0
    if return_model_params:
        return mu1, beta
    else:
        return mu1


def generate_samples_rational_linear_agent(
    data, alpha: float = 1, update_type: str = "regress", lr: float = 0.1
):
    X, pos = data["X"], data["pos"]

    # TODO add intercept

    n_samples, n_cov = X.shape

    # initialize
    beta_curr = np.random.normal(size=(n_cov, 2))  # np.zeros((n_cov, 2))

    # collect actions and outcomes
    a = np.zeros(n_samples)
    y = np.zeros(n_samples)
    pred_pos = np.zeros((n_samples, 2))
    pi = np.zeros(n_samples)
    betas = list()

    # agent (doctor) learns online
    for i in range(n_samples):
        # store current beta
        betas.append(beta_curr.copy())

        # get next observation
        x = X[i, :]

        # predict potential outcomes
        pred_0 = np.dot(x, beta_curr[:, 0])
        pred_1 = np.dot(x, beta_curr[:, 1])
        pred_pos[i, :] = np.array([pred_0, pred_1])

        # compute predicted relative effect
        pred_rel_cate = (pred_1 - pred_0) / (pred_0 + EPS)

        # randomly choose an action according to expected rel. effect:
        # compute prob/policy
        prob_1 = expit(alpha * pred_rel_cate)
        pi[i] = prob_1
        # choose action
        action = np.random.choice([0, 1], p=[1 - prob_1, prob_1])
        a[i] = action

        # observe outcome & save
        y_factual = pos[i, action]
        y[i] = y_factual

        # update belief
        if update_type == "regress":
            factual_error = pred_pos[i, action] - y_factual
            new_beta = beta_curr[:, action] - lr * factual_error * x
            beta_curr[:, action] = new_beta
        else:
            raise ValueError("Update type {} not implemented (yet)".format(update_type))

    return X, y, a, pi, betas


class SimDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(SimDataset, self).__init__()

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


def generate_linear_dataset(num_trajs, max_len, seed=41310):

    np.random.seed(seed)

    X = np.zeros((num_trajs, max_len, 5))
    A = np.zeros((num_trajs, max_len))
    Y = np.zeros((num_trajs, max_len))
    M = np.ones((num_trajs, max_len))

    for i in range(num_trajs):
        data, params = simulate_x_and_pos(n=max_len, no_seed=True)
        x, y, a, pi, betas = generate_samples_rational_linear_agent(data)

        X[i, :, :] = x
        A[i] = a
        Y[i] = y

    dataset = SimDataset()

    dataset.covariates = torch.tensor(X)
    dataset.actions = torch.tensor(A, dtype=int)
    # dataset.actions = torch.ones(dataset.actions.shape)
    # dataset.actions = torch.tensor(dataset.actions, dtype=int)
    dataset.outcomes = torch.tensor(Y)
    dataset.mask = torch.tensor(M, dtype=int)

    dataset.N = len(dataset.covariates)

    return dataset


if __name__ == "__main__":
    n_steps = 2500
    data, params = simulate_x_and_pos(n=n_steps, seed=5)
    X, y, a, pi, betas = generate_samples_rational_linear_agent(data)
    print(
        "True beta_0: {}, True beta_1: {} ".format(params[0], (params[0] + params[1]))
    )
    print(
        "Agent finds beta_0 {} and beta_1 {}".format(
            np.round(betas[n_steps - 1][:, 0], decimals=2),
            np.round(betas[n_steps - 1][:, 1], decimals=2),
        )
    )

    print(X.shape)
    print(y.shape)
    print(a.shape)

    X, A, Y, M = generate_linear_dataset(100, 50, seed=41310)

    print(X.shape)
    print(Y.shape)
    print(A.shape)
    print(M.shape)
