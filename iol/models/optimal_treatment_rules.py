import torch


class OTR():
    def __init__(self):
        self.name = "name"


class NormalisedRatio(OTR):
    def __init__(self):
        super(NormalisedRatio, self).__init__()

    def __call__(self, y_1, y_0):

        ratio = (y_1 - y_0) / y_0

        return torch.sigmoid(ratio, 2)
