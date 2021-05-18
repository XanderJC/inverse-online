import torch

from src.models import BaseModel
from src.models.utils import reverse_sequence

# from src.models.optimal_treatment_rules import OTR, NormalisedRatio
# from src.models.cate_nets import CATENet, TreatNetwork

import torch.nn.functional as F
from torch.distributions.kl import kl_divergence


class AdaptiveLinearModel(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        outcome_size: int,
        # Memory network params
        memory_hidden_size: int,
        memory_layers: int,
        memory_dropout: float,
        memory_size: int,
        # Outcome network params
        outcome_hidden_size: int,
        outcome_layers: int,
        # Inf network params
        inf_hidden_size: int,
        inf_layers: int,
        inf_dropout: float,
        inf_fc_size: int,
        **kwargs,
    ):
        super(AdaptiveLinearModel, self).__init__()

        self.name = "AdaptiveLinear"

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.outcome_size = outcome_size
        self.memory_size = memory_size

        self.memory_hidden_size = memory_hidden_size
        self.memory_layers = memory_layers
        self.memory_dropout = memory_dropout

        self.outcome_hidden_size = outcome_hidden_size
        self.outcome_layers = outcome_layers

        self.inf_hidden_size = inf_hidden_size
        self.inf_layers = inf_layers
        self.inf_dropout = inf_dropout
        self.inf_fc_size = inf_fc_size

        self.memory_network = SummaryNetwork(
            input_size=self.covariate_size + self.action_size + self.outcome_size,
            memory_size=self.memory_size,
            hidden_size=self.memory_hidden_size,
            num_layers=self.memory_layers,
            dropout=self.memory_dropout,
        )

        self.outcome_network = TENetwork(
            input_size=self.covariate_size,
            memory_size=self.memory_size,
            hidden_size=self.outcome_hidden_size,
            num_layers=self.outcome_layers,
        )

        self.treatment_rule = TreatRule()

        self.inference_network = InfNetwork(
            covariate_size=self.covariate_size,
            action_size=self.action_size,
            memory_size=self.memory_size,
            memory_hidden_size=self.memory_hidden_size,
            hidden_size=self.inf_hidden_size,
            num_layers=self.inf_layers,
        )

    def kl_div(self, posterior_params, prior_params):
        post_mean = posterior_params[:, :, : self.memory_size]
        post_lstd = posterior_params[:, :, self.memory_size :]
        post_dist = torch.distributions.Normal(post_mean, torch.exp(post_lstd))

        prior_mean = prior_params[:, :, : self.memory_size]
        prior_lstd = prior_params[:, :, self.memory_size :]
        prior_dist = torch.distributions.Normal(prior_mean, torch.exp(prior_lstd))

        return kl_divergence(post_dist, prior_dist)

    def sample_memory(self, posterior_params):

        post_mean = posterior_params[:, :, : self.memory_size]
        post_lstd = posterior_params[:, :, self.memory_size :]
        post_dist = torch.distributions.Normal(post_mean, torch.exp(post_lstd))

        memory_sample = post_dist.rsample()

        return memory_sample

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        # Get summary of past at each step
        # ----------------------------------------------

        # Get summary - check axis
        # print("covariates:")
        # print(covariates.shape)

        actions = F.one_hot(actions)

        # print("actions:")
        # print(actions.shape)

        # print("outcomes:")
        # print(outcomes.shape)

        history = torch.cat([covariates, actions, outcomes.unsqueeze(2)], axis=2)

        # print("history:")
        # print(history.shape)

        hidden_state, prior_params = self.memory_network(history)
        # out: tensor of shape (batch_size, seq_length + 1, output_size)

        # Take summary so time step i doesn't include info from i
        prior_params = prior_params[:, :-1, :]
        hidden_state = hidden_state[:, :-1, :]

        # print("prior_params:")
        # print(prior_params.shape)

        # print("hidden_state:")
        # print(hidden_state.shape)
        # Predict prior with forward model
        # ----------------------------------------------

        # prior_params = self.memory_network_fc(hidden_state)

        # predicted_outcomes = self.outcome_network(concat_pred_in)

        # Predict posterior with inference network
        # ----------------------------------------------

        posterior_params = self.inference_network(history, hidden_state, mask)

        # print("posterior_params:")
        # print(posterior_params.shape)

        # Calculate KL divergence
        # ----------------------------------------------

        kl_loss = self.kl_div(posterior_params, prior_params)
        kl_loss = kl_loss.mean(axis=2)
        # print("kl_loss:")
        # print(kl_loss.shape)

        kl_loss = kl_loss.masked_select(mask.squeeze().bool()).mean()

        # Sample from posterior and calculate likelihood
        # ----------------------------------------------

        # Concatenate with covariates summary

        memory = self.sample_memory(posterior_params)

        # memory = posterior_params

        # print("memory:")
        # print(memory.shape)

        treatment_effect = self.outcome_network(covariates, memory)

        # print("treatment_effect:")
        # print(treatment_effect.shape)

        pred = self.treatment_rule(treatment_effect)

        # print("pred:")
        # print(pred.shape)

        dist = torch.distributions.Categorical(probs=pred)
        ll = dist.log_prob(actions.argmax(dim=2))

        # print(pred)
        # print(actions.argmax(dim=2))

        neg_log_likelihood = -ll.masked_select(mask.squeeze().bool()).mean()
        # print(neg_log_likelihood)
        # print(kl_loss)
        return neg_log_likelihood + kl_loss


class TreatRule(torch.nn.Module):
    def __init__(
        self,
    ):
        super(TreatRule, self).__init__()
        self.alpha = torch.nn.Parameter(torch.ones((1), dtype=torch.double))
        self.beta = torch.nn.Parameter(torch.zeros((1), dtype=torch.double))

    def forward(self, treatment_effect):

        keep_pos = torch.nn.Softplus()
        logit = keep_pos(self.alpha) * (treatment_effect - self.beta)

        pred = torch.sigmoid(logit)

        zero_pred = 1 - pred

        pred = torch.cat((zero_pred, pred), axis=2)

        return pred


class TENetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        memory_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super(TENetwork, self).__init__()

        self.input_size = input_size
        self.memory_size = memory_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.in_layer = torch.nn.Linear(self.memory_size, self.hidden_size)
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        )
        self.weight_layer_0 = torch.nn.Linear(self.hidden_size, self.input_size)
        self.weight_layer_1 = torch.nn.Linear(self.hidden_size, self.input_size)

        self.bias_layer_0 = torch.nn.Linear(self.hidden_size, 1)
        self.bias_layer_1 = torch.nn.Linear(self.hidden_size, 1)

        self.in_layer.double()
        for layer in self.linears:
            layer.double()
        self.weight_layer_0.double()
        self.weight_layer_1.double()
        self.bias_layer_0.double()
        self.bias_layer_1.double()

    def forward(self, covariates, memory):

        batch_size, seq_length, dim = covariates.shape

        x = self.in_layer(memory)
        for layer in self.linears:
            x = F.elu(layer(x))

        omega_0 = self.weight_layer_0(x)
        omega_1 = self.weight_layer_1(x)

        b_0 = self.bias_layer_0(x)
        b_1 = self.bias_layer_1(x)

        # omega_0 = omega_0.reshape((batch_size * seq_length, dim, 1))
        # omega_1 = omega_1.reshape((batch_size * seq_length, dim, 1))
        # covariates = covariates.reshape((batch_size * seq_length, 1, dim))

        # print("omega_0:")
        # print(omega_0.shape)

        # print("covariates:")
        # print(covariates.shape)

        # mu_0 = torch.bmm(covariates, omega_0)
        # mu_1 = torch.bmm(covariates, omega_1)

        mu_0 = (omega_0 * covariates).sum(axis=2)
        mu_1 = (omega_1 * covariates).sum(axis=2)

        mu_0 = mu_0.unsqueeze(2) + b_0
        mu_1 = mu_1.unsqueeze(2) + b_1

        # print("mu_0:")
        # print(mu_0.shape)

        treatment_effect = mu_1 - mu_0

        # print("treatment_effect:")
        # print(treatment_effect.shape)

        treatment_effect = treatment_effect.reshape((batch_size, seq_length, 1))
        return treatment_effect


class SummaryNetwork(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        memory_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(SummaryNetwork, self).__init__()

        self.input_size = input_size
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

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

        self.fc_layer = torch.nn.Linear(self.hidden_size, self.memory_size * 2)
        self.fc_layer.double()

    def forward(self, x):

        # First get summary of previous history of everything with the LSTM
        h0 = self.h0.expand(self.num_layers, x.size(0), self.hidden_size)
        c0 = self.c0.expand(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.lstm(x.double(), (h0, c0))
        # out: tensor of shape (batch_size, seq_length, hidden_size)

        init_out = h0[-1, 0, :].expand(x.size(0), 1, self.hidden_size)
        out = torch.cat((init_out, out), axis=1)

        # print("out")
        # print(out.shape)
        # out: tensor of shape (batch_size, seq_length+1, hidden_size)

        return out, self.fc_layer(out)


class InfNetwork(torch.nn.Module):
    """
    Network that approximates posterior distribution of the memory.
    """

    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        memory_size: int,
        memory_hidden_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super(InfNetwork, self).__init__()

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory_hidden_size = memory_hidden_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_size = self.covariate_size + self.action_size + 1

        self.in_layer = torch.nn.Linear(
            self.memory_hidden_size + self.hidden_size,
            self.hidden_size,
        )
        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        )
        self.out_layer = torch.nn.Linear(self.hidden_size, self.memory_size * 2)

        self.in_layer.double()
        for layer in self.linears:
            layer.double()
        self.out_layer.double()

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
        )
        self.lstm.double()

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

        # out = self.linear(out)
        # Reverse hidden state again
        re_rev_out = reverse_sequence(out, mask)

        full_embedding = torch.cat((memory_embedding, re_rev_out), 2)

        # print("memory_embedding")
        # print(memory_embedding.shape)

        # print("re_rev_out")
        # print(re_rev_out.shape)

        x = F.elu(self.in_layer(full_embedding))
        for layer in self.linears:
            x = F.elu(layer(x))
        variational_params = self.out_layer(x)

        return variational_params
