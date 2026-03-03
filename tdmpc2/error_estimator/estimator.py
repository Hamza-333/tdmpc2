import torch

class ErrorEstimator(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder_block1 = torch.nn.Sequential(
            torch.nn.Linear(cfg.latent_dim + cfg.action_dim, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
        )
        self.encoder_block_2 = torch.nn.Sequential(
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.LeakyReLU(),
        )
        self.decoder_block_1 = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.LayerNorm(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 512)
        )
        self.decoder_block_2 = torch.nn.Sequential(
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, cfg.latent_dim)
        )
        
        # self.module_1 = torch.nn.Sequential(
        #     torch.nn.Linear(cfg.latent_dim + cfg.action_dim, 256),
        #     torch.nn.LayerNorm(256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 128),
        #     torch.nn.LayerNorm(128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 64),
        #     ## comment out below for error magnitude prediction model and change above to (128, 1)
        #     # torch.nn.LayerNorm(64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 128),
        #     # torch.nn.LayerNorm(128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 256),
        #     # torch.nn.LayerNorm(256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, cfg.latent_dim)
        # )

        if self.cfg.error_model_mode == 'eval':
            self.load(self.cfg.error_model_checkpoint)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)

        # x = self.module_1(x)
        # error = self.module_2(x)
        # return error + self.down_sample(x)
        out1 = self.encoder_block1(x)
        out2 = self.encoder_block_2(out1)
        out3 = self.decoder_block_1(out2)
        resid_connection = out1 + out3
        out = self.decoder_block_2(resid_connection)
        return out

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))