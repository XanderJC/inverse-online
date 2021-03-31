import numpy as np
import torch

from src.constants import *
from src.models import BaseModel


class BeliefModel(BaseModel):
    def __init__(self):
        super(BeliefModel, self).__init__()

        self.treatment_network = TreatNetwork()

    def loss(self, batch):
        return


class TreatNetwork(torch.nn.Module):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float,
    ):
        super(TreatNetwork, self).__init__()

        self.covariate_size = covariate_size
        self.action_size = action_size

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        self.pred_hidden_size = pred_hidden_size
        self.pred_layers = pred_layers

        self.h0 = torch.nn.Parameter(
            torch.zeros(self.lstm_layers, 1, self.lstm_hidden_size)
        )
        self.c0 = torch.nn.Parameter(
            torch.zeros(self.lstm_layers, 1, self.lstm_hidden_size)
        )

        self.lstm = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.lstm_layers,
            batch_first=True,
            dropout=self.lstm_dropout,
        )

        self.predictor = nn.ModuleList(
            [
                nn.Linear(
                    self.covariate_size + self.lstm_hidden_size, self.pred_hidden_size
                )
            ]
            + [
                nn.Linear(self.pred_hidden_size, self.pred_hidden_size)
                for _ in range(self.pred_layers)
            ]
            + [nn.Linear(self.pred_hidden_size, 2)]
        )

    def forward(self, covariates, actions, outcomes):

        # First get summary of previous history of everything with the LSTM
        h0 = self.h0.expand(self.lstm_layers, covariates.size(0), self.lstm_hidden_size)
        c0 = self.c0.expand(self.lstm_layers, covariates.size(0), self.lstm_hidden_size)

        # Need to check this - also maybe need to shift one step
        concat_lstm_in = torch.concat([covariates, actions, outcomes], axis=1)
        print(concat_lstm_in.size)

        out, _ = self.lstm(
            concat_lstm_in, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, lstm_hidden_size)

        # Now concatenate with covariates and predict belief over treatment effects
        concat_pred_in = torch.concat([covariates, out], axis=1)

        treatment_beliefs = self.predictor(concat_pred_in)

        return treatment_beliefs


class InfNetwork(torch.nn.Module):
    """
    Network that needs to
    """

    def __init__(self):
        super(InfNetwork, self).__init__()
