import numpy as np
import torch

from src.constants import *
from src.models import BaseModel


class BeliefModel(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        num_treatments: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float,
        summary_size: int,
        pred_hidden_size: int,
        pred_layers: int,
    ):
        super(BeliefModel, self).__init__()

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.num_treatments = num_treatments
        self.summary_size = summary_size

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        self.pred_hidden_size = pred_hidden_size
        self.pred_layers = pred_layers

        self.summary_network = SummaryNetwork(
            input_size = self.covariate_size+self.action_size+self.num_treatments,
            output_size = self.summary_size,
            hidden_size = self.lstm_hidden_size,
            num_layers = self.lstm_layers,
            dropout = self.lstm_dropout
        )

        self.treatment_network = TreatNetwork(
            covariate_size: self.covariate_size,
            summary_size: self.summary_size,
            hidden_size: self.pred_hidden_size,
            num_layers: self.pred_layers,
            num_treatments: self.num_treatments,
        )

    def forward(self, covariates, actions, outcomes):

        # Get summary - check axis
        concat_summary_in = torch.concat([covariates, actions, outcomes], axis=1)
        print(concat_summary_in.size)

        summary = self.summary_network(concat_summary_in)
        # out: tensor of shape (batch_size, seq_length + 1, output_size)

        # Now concatenate with covariates and predict belief over treatment effects
        concat_pred_in = torch.concat([covariates, summary], axis=1)

        treatment_beliefs = self.treatment_network(concat_pred_in)

        return treatment_beliefs

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        pred = F.softmax(self.forward(covariates, actions, outcomes), 2)
        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(y_series)

        return -ll.masked_select(mask.bool()).mean()


class SummaryNetwork(torch.nn.module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(SummaryNetwork, self).__init__()

        self.input_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout = dropout
        self.output_size = output_size

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

        self.fc = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):

        # First get summary of previous history of everything with the LSTM
        h0 = self.h0.expand(self.num_layers, x.size(0), self.hidden_size)
        c0 = self.c0.expand(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x, (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        init_out = self.fc(h0[-1,0,:].expand(x.size(0),1,self.hidden_size))
        out = torch.concat(init_out, out, axis=1)
        print(out.shape)

        out = self.fc(out)
        # out: tensor of shape (batch_size, seq_length+1, output_size)

        return out


class TreatNetwork(torch.nn.Module):
    def __init__(
        self,
        covariate_size: int,
        summary_size: int,
        hidden_size: int,
        num_layers: int,
        num_treatments: int,
    ):
        super(TreatNetwork, self).__init__()

        self.covariate_size = covariate_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.predictor = nn.ModuleList(
            [nn.Linear(self.covariate_size + self.summary_size, self.hidden_size)]
            + [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
            + [nn.Linear(self.hidden_size, self.num_treatments)]
        )

    def forward(self, covariates, summary):

        # Concatenate with covariates summary
        # Check axis
        concat_pred_in = torch.concat([covariates, summary], axis=1)

        treatment_beliefs = self.predictor(concat_pred_in)

        return treatment_beliefs


class InfNetwork(torch.nn.Module):
    """
    Network that needs to
    """

    def __init__(self):
        super(InfNetwork, self).__init__()
