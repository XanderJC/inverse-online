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


class BehaviouralCloningLSTM(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        **kwargs,
    ):
        super(BehaviouralCloningLSTM, self).__init__()
        self.covariate_size = covariate_size
        self.action_size = action_size

        self.lstm_hidden_size = 32
        self.lstm_layers = 1

        self.hidden_size = 32
        self.num_layers = 1

        self.lstm = torch.nn.LSTM(
            self.covariate_size,
            self.lstm_hidden_size,
            self.lstm_layers,
            batch_first=True,
        )

        self.h0 = torch.nn.Parameter(
            torch.zeros(
                (self.lstm_layers, 1, self.lstm_hidden_size), dtype=torch.double
            )
        )
        self.c0 = torch.nn.Parameter(
            torch.zeros(
                (self.lstm_layers, 1, self.lstm_hidden_size), dtype=torch.double
            )
        )

        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        )
        self.fc = torch.nn.Linear(self.hidden_size, self.action_size)

        self.lstm.double()
        for layer in self.linears:
            layer.double()
        self.fc.double()

    def forward(self, x):

        h0 = self.h0.expand(self.lstm_layers, x.size(0), self.hidden_size)
        c0 = self.c0.expand(self.lstm_layers, x.size(0), self.hidden_size)

        # Forward LSTM
        out, _ = self.lstm(
            x.double(), (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        for layer in self.linears:
            out = layer(out)
            out = F.elu(out)

        pred = self.fc(out)
        return pred

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        pred = F.softmax(self.forward(covariates), dim=2)

        dist = torch.distributions.Categorical(probs=pred)
        ll = dist.log_prob(actions)

        neg_log_likelihood = -ll.masked_select(mask.squeeze().bool()).mean()

        return neg_log_likelihood
