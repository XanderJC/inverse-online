import numpy as np
import torch

from src.constants import *
from src.models import BaseModel


class AdaptiveLinearModel(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        num_treatments: int,
        summary_size: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float,
        fc_hidden_size: int,
        fc_layers: int,
    ):
        super(AdaptiveLinearModel, self).__init__()

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.num_treatments = num_treatments
        self.summary_size = summary_size

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        self.fc_hidden_size = fc_hidden_size
        self.fc_layers = fc_layers

        self.summary_network = SummaryNetwork(
            input_size=self.covariate_size + self.action_size + self.num_treatments,
            output_size=self.summary_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
        )

        self.linear_param_predictor = MLPNetwork(
            input_size=self.summary_size,
            output_size=(self.covariate_size + 1) * self.num_treatments,
            hidden_size=self.fc_hidden_size,
            num_layers=self.fc_layers,
        )

    def forward(self, covariates, actions, outcomes):

        # Get summary - check axis
        concat_summary_in = torch.concat([covariates, actions, outcomes], axis=1)
        print(concat_summary_in.size)

        summary = self.summary_network(concat_summary_in)
        # out: tensor of shape (batch_size, seq_length + 1, output_size)
        summary = summary[:, :-1, :]

        linear_params = self.linear_param_predictor(summary)

        # Augment covariates with constant 1 for bias
        # Format the linear_params matrix

        treatment_beliefs = torch.matmul(covariates, linear_params)

        return treatment_beliefs, linear_params

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        treatment_beliefs, _ = self.forward(covariates, actions, outcomes)

        pred = F.softmax(treatment_beliefs, 2)
        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(y_series)

        return -ll.masked_select(mask.bool()).mean()


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

        self.predictor = nn.ModuleList(
            [nn.Linear(self.input_size, self.hidden_size)]
            + [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
            + [nn.Linear(self.hidden_size, self.output_size)]
        )
