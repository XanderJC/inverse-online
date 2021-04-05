import numpy as np
import torch

from src.constants import *
from src.models import BaseModel, SummaryNetwork
import torch.nn.functional as F


class AdaptiveLinearModel(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        outcome_size: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float,
        summary_size: int,
        fc_hidden_size: int,
        fc_layers: int,
    ):
        super(AdaptiveLinearModel, self).__init__()

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.outcome_size = outcome_size
        self.summary_size = summary_size

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers

        self.summary_network = SummaryNetwork(
            input_size=self.covariate_size + self.action_size + self.outcome_size,
            output_size=self.summary_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
        )

        self.linear_param_predictor = MLPNetwork(
            input_size=self.summary_size,
            output_size=(self.covariate_size + 1) * self.action_size,
            hidden_size=self.fc_hidden_size,
            num_layers=self.fc_layers,
        )

    def forward(self, covariates, actions, outcomes):

        batch_size, seq_len, _ = covariates.size()

        # Get summary - check axis
        concat_summary_in = torch.cat([covariates, actions, outcomes], axis=2)

        summary = self.summary_network(concat_summary_in)
        # out: tensor of shape (batch_size, seq_length + 1, output_size)

        summary = summary[:, :-1, :]

        linear_params = self.linear_param_predictor(summary)

        # Augment covariates with constant 1 for bias
        bias_covariate = torch.ones((batch_size, seq_len, 1))
        covariates = torch.cat((covariates, bias_covariate), axis=2)

        # Format the linear_params matrix
        linear_params = linear_params.reshape(
            (batch_size, seq_len, self.covariate_size + 1, self.action_size)
        )

        treatment_beliefs = torch.matmul(covariates.unsqueeze(2), linear_params)

        return treatment_beliefs.squeeze(), linear_params

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        treatment_beliefs, _ = self.forward(covariates, actions, outcomes)

        pred = F.softmax(treatment_beliefs, 2)
        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(actions.argmax(dim=2))

        return -ll.masked_select(mask.squeeze().bool()).mean()


class MLPNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super(MLPNetwork, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.in_layer = torch.nn.Linear(self.input_size, self.hidden_size)
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        )
        self.out_layer = torch.nn.Linear(self.hidden_size, self.output_size)

        self.in_layer.double()
        for layer in self.linears:
            layer.double()
        self.out_layer.double()

    def forward(self, x):

        x = self.in_layer(x)
        for layer in self.linears:
            x = layer(x)
        x = self.out_layer(x)

        return x
