import torch

class ErrorEstimator(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = torch.nn.Sequential(
            torch.nn.Linear(cfg.latent_dim + cfg.action_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)

        error = self.model(x)
        return error

    def save(self, path):
        torch.save(self.state_dict(), path)