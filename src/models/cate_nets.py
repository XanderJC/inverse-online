import torch


class CATENet(torch.nn.Module):
    def __init__(self):
        super(CATENet, self).__init__()
        self.name = "name"


class TreatNetwork(CATENet):
    def __init__(
        self,
        covariate_size: int,
        action_size: int,
        summary_size: int,
        hidden_size: int,
        num_layers: int,
    ):
        super(TreatNetwork, self).__init__()

        self.covariate_size = covariate_size
        self.action_size = action_size
        self.summary_size = summary_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.in_layer = torch.nn.Linear(
            self.covariate_size + self.summary_size, self.hidden_size
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

    def forward(self, x):

        x = self.in_layer(x)
        for layer in self.linears:
            x = layer(x)
        x = self.out_layer(x)

        return x
