import numpy as np
import torch
import time

from pkg_resources import resource_filename

from src.constants import *


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        name = "Base"

    def fit(
        self,
        dataset,
        epochs=10,
        batch_size=128,
        learning_rate=1e-3,
        validation_set=None,
    ):
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, betas=(0.9, 0.9)
        )

        for epoch in range(epochs):
            self.train()
            running_loss = 0
            start = time.time()
            for i, batch in enumerate(data_loader):

                optimizer.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                optimizer.step()

                running_loss += loss
            end = time.time()
            average_loss = round((running_loss.detach().numpy() / (i + 1)), 5)
            print(
                f"Epoch {epoch+1} average loss: {average_loss} ({round(end-start,2)} seconds)"
            )

            if validation_set is not None:
                self.eval()
                validation_loss = round(float(self.loss(validation_set).detach()), 5)
                print(f"Epoch {epoch+1} validation loss: {validation_loss}")

        return

    def save_model(self):
        path = resource_filename("src", f"models/saved_models/{self.name}.pth")
        torch.save(self.state_dict(), path)
