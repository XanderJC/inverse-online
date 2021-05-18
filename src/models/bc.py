import torch

from src.models import BaseModel
import torch.nn.functional as F


class BehaviouralCloning(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        **kwargs,
    ):
        super(BehaviouralCloning, self).__init__()
        self.covariate_size = covariate_size
        self.action_size = action_size

        self.linear = torch.nn.Linear(self.covariate_size, self.action_size)
        self.linear.double()

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        pred = F.softmax(self.linear(covariates), dim=2)

        dist = torch.distributions.Categorical(probs=pred)
        ll = dist.log_prob(actions)

        neg_log_likelihood = -ll.masked_select(mask.squeeze().bool()).mean()

        return neg_log_likelihood
