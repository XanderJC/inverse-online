import torch

from iol.models import BaseModel
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score


class CIRL(BaseModel):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        hidden_size: int,
        **kwargs,
    ):
        super(CIRL, self).__init__()

        self.name = "CIRL"
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

        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        x = self.linear3(x)

        return x

    def loss(self, batch):
        covariates, actions, outcomes, mask = batch

        pred = self.forward(covariates)
        actions = F.one_hot(actions)

        pred = pred * actions

        dist = pred.sum(axis=2) - outcomes

        sq_diff = (dist ** 2).mean()

        return sq_diff

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
