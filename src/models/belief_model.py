import torch

from src.models import BaseModel
from src.models.utils import reverse_sequence
from src.models.optimal_treatment_rules import OTR, NormalisedRatio
from src.models.cate_nets import CATENet, TreatNetwork

# import torch.nn.functional as F
from torch.distributions.kl import kl_divergence


class BeliefModel(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        outcome_size: int,
        lstm_hidden_size: int,
        lstm_layers: int,
        lstm_dropout: float,
        memory_size: int,
        pred_hidden_size: int,
        pred_layers: int,
        outcome_network: CATENet = TreatNetwork,
        treatment_rule: OTR = NormalisedRatio,
        **kwargs,
    ):
        super(BeliefModel, self).__init__()

        self.name = "Belief"

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.outcome_size = outcome_size
        self.memory_size = memory_size

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        self.pred_hidden_size = pred_hidden_size
        self.pred_layers = pred_layers

        self.memory_network = SummaryNetwork(
            input_size=self.covariate_size + self.action_size + self.outcome_size,
            output_size=self.memory_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.lstm_dropout,
        )

        self.outcome_network = outcome_network(
            covariate_size=self.covariate_size,
            action_size=self.action_size,
            memory_size=self.memory_size,
            hidden_size=self.pred_hidden_size,
            num_layers=self.pred_layers,
        )

        self.inference_network = InfNetwork(
            covariate_size=self.covariate_size,
            action_size=self.action_size,
            memory_size=self.memory_size,
            hidden_size=self.pred_hidden_size,
            num_layers=self.pred_layers,
        )

    def kl_div(self, posterior_params, prior_params):
        post_mean, post_lstd = posterior_params
        post_dist = torch.distributions.Normal(post_mean, torch.exp(post_lstd))

        prior_mean, prior_lstd = prior_params
        prior_dist = torch.distributions.Normal(prior_mean, torch.exp(prior_lstd))

        return kl_divergence(post_dist, prior_dist)

    def sample_beliefs(self, posterior_params):

        post_mean, post_lstd = posterior_params
        post_dist = torch.distributions.Normal(post_mean, torch.exp(post_lstd))

        memory_sample = post_dist.sample()

        return memory_sample

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        # Get summary of past at each step
        # ----------------------------------------------

        # Get summary - check axis
        concat_summary_in = torch.cat([covariates, actions, outcomes], axis=2)

        hidden_state = self.memory_network(concat_summary_in)
        # out: tensor of shape (batch_size, seq_length + 1, output_size)

        # Take summary so time step i doesn't include info from i
        hidden_state = hidden_state[:, :-1, :]

        # Predict prior with forward model
        # ----------------------------------------------

        prior_params = self.memory_network_fc(hidden_state)

        # predicted_outcomes = self.outcome_network(concat_pred_in)

        # Predict posterior with inference network
        # ----------------------------------------------

        posterior_params = self.inference_network(concat_summary_in, hidden_state, mask)

        # Calculate KL divergence
        # ----------------------------------------------

        kl_loss = self.kl_div(posterior_params, prior_params)
        kl_loss = kl_loss.masked_select(mask.squeeze().bool()).mean()

        # Sample from posterior and calculate likelihood
        # ----------------------------------------------

        # Concatenate with covariates summary

        memory = self.sample_memory(posterior_params)

        concat_pred_in = torch.cat((covariates, memory), axis=2)

        y_hat_1, y_hat_0 = self.outcome_network(concat_pred_in)

        pred = self.treatment_rule(y_hat_1, y_hat_0)
        dist = torch.distributions.Categorical(probs=pred)
        ll = dist.log_prob(actions.argmax(dim=2))

        neg_log_likelihood = -ll.masked_select(mask.squeeze().bool()).mean()

        return neg_log_likelihood + kl_loss


class SummaryNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(SummaryNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.h0 = torch.nn.Parameter(
            torch.zeros((self.num_layers, 1, self.hidden_size), dtype=torch.double)
        )
        self.c0 = torch.nn.Parameter(
            torch.zeros((self.num_layers, 1, self.hidden_size), dtype=torch.double)
        )

        self.lstm = torch.nn.LSTM(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.lstm.double()

    def forward(self, x):

        # First get summary of previous history of everything with the LSTM
        h0 = self.h0.expand(self.num_layers, x.size(0), self.hidden_size)
        c0 = self.c0.expand(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x.double(), (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        init_out = h0[-1, 0, :].expand(x.size(0), 1, self.hidden_size)
        out = torch.cat((init_out, out), axis=1)

        # out: tensor of shape (batch_size, seq_length+1, hidden_size)

        return out


class InfNetwork(torch.nn.Module):
    """
    Network that approximates posterior distribution of the memory.
    """

    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        summary_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super(InfNetwork, self).__init__()

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.summary_size = summary_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.in_layer = torch.nn.Linear(
            self.covariate_size + self.action_size + 1 + self.summary_size,
            self.hidden_size,
        )
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        )
        self.out_layer = torch.nn.Linear(self.hidden_size, self.action_size)

        self.in_layer.double()
        for layer in self.linears:
            layer.double()
        self.out_layer.double()

    def forward(self, x, memory_embedding, mask):

        # Reverse ordering of x
        rev_x = reverse_sequence(x, mask)

        # First get summary of previous history of everything with the LSTM
        h0 = self.h0.expand(self.num_layers, x.size(0), self.hidden_size)
        c0 = self.c0.expand(self.num_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(
            rev_x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        out = self.linear(out)
        # Reverse hidden state again
        re_rev_out = reverse_sequence(out, mask)

        full_embedding = torch.cat((memory_embedding, re_rev_out), 2)

        x = self.in_layer(full_embedding)
        for layer in self.linears:
            x = layer(x)
        variational_params = self.out_layer(x)

        return variational_params
