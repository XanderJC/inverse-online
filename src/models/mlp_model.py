import torch

from src.models import BaseModel, MLPNetwork
import torch.nn.functional as F


class MLPModel(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        outcome_size: int,
        hidden_size: int,
        num_layers: int,
        **kwargs,
    ):
        super(MLPModel, self).__init__()

        self.name = "MLP"

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.outcome_size = outcome_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.predictor = MLPNetwork(
            input_size=self.covariate_size,
            output_size=self.action_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        )

    def forward(self, x):

        return self.predictor(x)

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        pred = self.forward(covariates)
        pred = F.softmax(pred, 2)

        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(actions.argmax(dim=2))

        return -ll.masked_select(mask.squeeze().bool()).mean()
