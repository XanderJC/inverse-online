import torch
import time
import datetime
from pkg_resources import resource_filename


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.name = "Base"

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
                f"Epoch {epoch+1} average loss: {average_loss}"
                + f" ({round(end-start,2)} seconds)"
            )

            if validation_set is not None:
                self.eval()
                metrics = self.validation(validation_set)
                for metric in metrics.keys():
                    met = round(float(metrics[metric]), 5)
                    print(f"Epoch {epoch+1} {metric}: {met}")

        return

    def save_model(self, name=None):

        if name is None:
            x = datetime.datetime.now()
            date = x.strftime("%x").replace("/", "")
            clock = x.strftime("%X")
            path_tail = f"models/saved_models/{self.name}_{date}_{clock}.pth"
        else:
            path_tail = f"models/saved_models/{self.name}_{name}.pth"
        path = resource_filename("iol", path_tail)
        torch.save(self.state_dict(), path)

        return

    def load_model(self, name=None):

        if name is None:
            path_tail = f"models/saved_models/{self.name}_best.pth"
        else:
            path_tail = f"models/saved_models/{self.name}_{name}.pth"
        path = resource_filename("iol", path_tail)
        self.load_state_dict(torch.load(path))

        return

    def save_best(self):

        path = resource_filename("iol", f"models/saved_models/{self.name}_best.pth")
        torch.save(self.state_dict(), path)

        return
