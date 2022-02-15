import torch

from iol.models import BaseModel
from iol.models.utils import reverse_sequence

import torch.nn.functional as F
from torch.distributions.kl import kl_divergence

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


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
        mem_reg: float = 0.0,
        spread: int = 2,
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

        self.mem_reg = mem_reg
        self.spread = spread
        self.memory_network = SummaryNetwork(
            input_size=self.memory_size
            + self.covariate_size
            + self.action_size
            + self.outcome_size,
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

        self.prior = torch.nn.Parameter(
            torch.ones((1, 1, self.memory_size * 2), dtype=torch.double)
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

        batch_size, seq_length, output_size = covariates.shape
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

        prior_params = torch.zeros((batch_size, seq_length, self.memory_size * 2))
        posterior_params = torch.zeros((batch_size, seq_length, self.memory_size * 2))

        memories = torch.zeros((batch_size, seq_length, self.memory_size))

        # print("self.prior:")
        # print(self.prior.shape)

        batch_prior = self.prior.expand(batch_size, 1, self.memory_size * 2)

        # print("batch_prior:")
        # print(batch_prior.shape)

        prior_params[:, 0, :] = batch_prior.squeeze()

        future_state = self.inference_network.summarise_future(history, mask)

        # print("future_state:")
        # print(future_state.shape)

        posterior = self.inference_network(
            torch.zeros(batch_size, self.memory_size), future_state[:, 0, :]
        )

        # print("posterior:")
        # print(posterior.shape)

        memory = self.sample_memory(posterior.unsqueeze(1))

        # print("memory:")
        # print(memory.shape)

        posterior_params[:, 0, :] = posterior
        memories[:, 0, :] = memory.squeeze()

        for t in range(seq_length - 1):

            previous_memory = memories[:, t, :]

            # print("previous_memory:")
            # print(previous_memory.shape)

            concat = torch.cat((previous_memory, history[:, t, :]), 1)

            # print("concat:")
            # print(concat.shape)

            next_prior = self.memory_network(concat)

            posterior = self.inference_network(previous_memory, future_state[:, t, :])

            memory = self.sample_memory(posterior.unsqueeze(1))

            prior_params[:, t + 1, :] = next_prior
            posterior_params[:, t + 1, :] = posterior

            memories[:, t + 1, :] = memory.squeeze()

        # print("prior_params:")
        # print(prior_params.shape)

        # print("hidden_state:")
        # print(hidden_state.shape)
        # Predict prior with forward model
        # ----------------------------------------------

        # prior_params = self.memory_network_fc(hidden_state)

        # predicted_outcomes = self.outcome_network(concat_pred_in)

        # print("posterior_params:")
        # print(posterior_params.shape)

        memory_diff = memories[:, :-1, :] - memories[:, 1:, :]
        memory_reg = (memory_diff ** 2).mean(axis=2)
        memory_reg = memory_reg.masked_select(mask[:, 1:].squeeze().bool()).mean()


        # Calculate KL divergence
        # ----------------------------------------------

        kl_loss = self.kl_div(posterior_params, prior_params)
        kl_loss = kl_loss.mean(axis=2)
        # print("kl_loss:")
        # print(kl_loss)

        kl_loss = kl_loss.masked_select(mask.squeeze().bool()).mean()

        # Sample from posterior and calculate likelihood
        # ----------------------------------------------

        # memory = posterior_params

        # print("memory:")
        # print(memory.shape)

        treatment_effect = self.outcome_network(covariates, memories)

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

        pred_loss = 0
        pred_loss += neg_log_likelihood

        for i in range(self.spread - 1):
            i = i + 1
            treatment_effect = self.outcome_network(covariates[:, :-i], memories[:, i:])
            pred = self.treatment_rule(treatment_effect)
            dist = torch.distributions.Categorical(probs=pred)
            ll = dist.log_prob(actions[:, :-i].argmax(dim=2))
            neg_log_likelihood = -ll.masked_select(mask[:, :-i].squeeze().bool()).mean()

            pred_loss += neg_log_likelihood

        pred_loss = pred_loss / self.spread
        # print(neg_log_likelihood)
        # print(kl_loss)
        return pred_loss + (1.0 * kl_loss) + (self.mem_reg * memory_reg)

    def validation(self, batch, prior = False):
        covariates, actions, outcomes, mask = batch

        batch_size, seq_length, output_size = covariates.shape

        actions = F.one_hot(actions)

        history = torch.cat([covariates, actions, outcomes.unsqueeze(2)], axis=2)

        prior_params = torch.zeros((batch_size, seq_length, self.memory_size * 2))
        posterior_params = torch.zeros((batch_size, seq_length, self.memory_size * 2))

        memories = torch.zeros((batch_size, seq_length, self.memory_size))

        prior_memories = torch.zeros((batch_size, seq_length, self.memory_size))

        batch_prior = self.prior.expand(batch_size, 1, self.memory_size * 2)

        prior_params[:, 0, :] = batch_prior.squeeze()

        future_state = self.inference_network.summarise_future(history, mask)

        posterior = self.inference_network(
            torch.zeros(batch_size, self.memory_size), future_state[:, 0, :]
        )

        memory = self.sample_memory(posterior.unsqueeze(1))
        prior_memory = self.sample_memory(batch_prior)

        posterior_params[:, 0, :] = posterior
        memories[:, 0, :] = memory.squeeze()
        prior_memories[:, 0, :] = prior_memory.squeeze()

        for t in range(seq_length - 1):

            previous_memory = memories[:, t, :]

            concat = torch.cat((previous_memory, history[:, t, :]), 1)

            next_prior = self.memory_network(concat)

            posterior = self.inference_network(previous_memory, future_state[:, t, :])

            memory = self.sample_memory(posterior.unsqueeze(1))
            prior_memory = self.sample_memory(next_prior.unsqueeze(1))

            prior_params[:, t + 1, :] = next_prior
            posterior_params[:, t + 1, :] = posterior

            memories[:, t + 1, :] = memory.squeeze()
            prior_memories[:, t + 1, :] = prior_memory.squeeze()

        # memories = self.sample_memory(posterior_params)
        # memories = self.sample_memory(prior_params)

        treatment_effect = self.outcome_network(covariates, memories)

        if prior:
            treatment_effect = self.outcome_network(covariates, prior_memories)

        pred = self.treatment_rule(treatment_effect)

        dist = torch.distributions.Categorical(probs=pred)
        ll = dist.log_prob(actions.argmax(dim=2))

        neg_log_likelihood = -ll.masked_select(mask.squeeze().bool()).mean()

        acc = ((pred * actions).sum(axis=2) > 0.5).float().mean()

        pred = pred.reshape(batch_size * seq_length, 2)
        actions = actions.reshape(batch_size * seq_length, 2)

        auc = roc_auc_score(y_true=actions, y_score=pred.detach())
        apr = average_precision_score(y_true=actions, y_score=pred.detach())

        loss_dict = {
            "ACC": acc.detach(),
            "AUC": auc,
            "APR": apr,
            "NLL": neg_log_likelihood.detach(),
        }
        return loss_dict

    def inspection(self, batch):
        covariates, actions, outcomes, mask = batch

        batch_size, seq_length, output_size = covariates.shape

        actions = F.one_hot(actions)

        history = torch.cat([covariates, actions, outcomes.unsqueeze(2)], axis=2)

        prior_params = torch.zeros((batch_size, seq_length, self.memory_size * 2))
        posterior_params = torch.zeros((batch_size, seq_length, self.memory_size * 2))

        memories = torch.zeros((batch_size, seq_length, self.memory_size))

        batch_prior = self.prior.expand(batch_size, 1, self.memory_size * 2)

        prior_params[:, 0, :] = batch_prior.squeeze()

        future_state = self.inference_network.summarise_future(history, mask)

        posterior = self.inference_network(
            torch.zeros(batch_size, self.memory_size), future_state[:, 0, :]
        )

        memory = self.sample_memory(posterior.unsqueeze(1))

        posterior_params[:, 0, :] = posterior
        memories[:, 0, :] = memory.squeeze()

        for t in range(seq_length - 1):

            previous_memory = memories[:, t, :]

            concat = torch.cat((previous_memory, history[:, t, :]), 1)

            next_prior = self.memory_network(concat)

            posterior = self.inference_network(previous_memory, future_state[:, t, :])

            memory = self.sample_memory(next_prior.unsqueeze(1))

            prior_params[:, t + 1, :] = next_prior
            posterior_params[:, t + 1, :] = posterior

            memories[:, t + 1, :] = memory.squeeze()

        memory_prior = self.sample_memory(prior_params)

        memory_posterior = self.sample_memory(posterior_params)

        mu_1_prior, mu_0_prior, omega_1_prior, omega_0_prior = self.outcome_network(
            covariates, memory_prior, return_mu=True
        )

        (
            mu_1_posterior,
            mu_0_posterior,
            omega_1_posterior,
            omega_0_posterior,
        ) = self.outcome_network(covariates, memory_posterior, return_mu=True)

        return {
            "prior_params": prior_params,
            "posterior_params": posterior_params,
            "mu_1_prior": mu_1_prior,
            "mu_0_prior": mu_0_prior,
            "mu_1_posterior": mu_1_posterior,
            "mu_0_posterior": mu_0_posterior,
            "omega_1_prior": omega_1_prior,
            "omega_0_prior": omega_0_prior,
            "omega_1_posterior": omega_1_posterior,
            "omega_0_posterior": omega_0_posterior,
        }


class TreatRule(torch.nn.Module):
    def __init__(
        self,
    ):
        super(TreatRule, self).__init__()

        # self.alpha = torch.nn.Parameter(torch.ones((1), dtype=torch.double))
        # self.beta = torch.nn.Parameter(torch.zeros((1), dtype=torch.double))

        self.alpha = torch.ones((1), dtype=torch.double)
        self.beta = torch.zeros((1), dtype=torch.double)

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

    def forward(self, covariates, memory, return_mu=False):

        batch_size, seq_length, dim = covariates.shape

        x = self.in_layer(memory.double())
        for layer in self.linears:
            x = F.elu(layer(x))

        omega_0 = self.weight_layer_0(x)
        omega_1 = self.weight_layer_1(x)

        b_0 = self.bias_layer_0(x)
        b_1 = self.bias_layer_1(x)

        b_0 = 0
        b_1 = 0

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

        treatment_effect = mu_1  # - mu_0

        # print("treatment_effect:")
        # print(treatment_effect.shape)

        treatment_effect = treatment_effect.reshape((batch_size, seq_length, 1))

        if not return_mu:
            return treatment_effect
        else:
            return mu_1, mu_0, omega_1, omega_0


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

        self.in_layer = torch.nn.Linear(self.input_size, self.hidden_size)
        self.in_layer.double()

        self.linears = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        )
        for layer in self.linears:
            layer.double()

        self.fc_layer = torch.nn.Linear(self.hidden_size, self.memory_size * 2)
        self.fc_layer.double()

    def forward(self, x):

        x = torch.sigmoid(self.in_layer(x))
        for layer in self.linears:
            x = torch.sigmoid(layer(x))

        x = self.fc_layer(x)

        return x


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
            self.memory_size + self.hidden_size,
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

    def summarise_future(self, x, mask):

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

        return re_rev_out

    def forward(self, memory, future_state):

        full_embedding = torch.cat((memory, future_state), 1)

        x = F.elu(self.in_layer(full_embedding))
        for layer in self.linears:
            x = F.elu(layer(x))
        variational_params = self.out_layer(x)

        return variational_params
