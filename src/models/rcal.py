import torch

from src.models import BaseModel
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


class RCAL(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        hidden_size: int,
        **kwargs,
    ):
        super(RCAL, self).__init__()

        self.name = "RCAL"
        self.covariate_size = covariate_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.linear1 = torch.nn.Linear(self.covariate_size, self.hidden_size)
        self.linear2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = torch.nn.Linear(self.hidden_size, self.action_size)
        self.linear1.double()
        self.linear2.double()
        self.linear3.double()

    def forward(self, x):

        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        x = self.linear3(x)

        return x

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        q_values = self.forward(covariates)
        pred = F.softmax(q_values, dim=2)

        dist = torch.distributions.Categorical(probs=pred)
        ll = dist.log_prob(actions)

        neg_log_likelihood = -ll.masked_select(mask.squeeze().bool()).mean()

        next_q_values = q_values.logsumexp(2) / 2

        q_diff = q_values[:, :-1, :] - next_q_values[:, 1:].unsqueeze(2)

        loss_reg = torch.abs(q_diff).mean() * 0.1

        return neg_log_likelihood + loss_reg

    def validation(self, batch):
        covariates, actions, outcomes, mask = batch

        batch_size, seq_length = actions.shape

        pred = F.softmax(self.forward(covariates), dim=2)

        dist = torch.distributions.Categorical(probs=pred)
        ll = dist.log_prob(actions)

        neg_log_likelihood = -ll.masked_select(mask.squeeze().bool()).mean()

        actions = F.one_hot(actions)
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
