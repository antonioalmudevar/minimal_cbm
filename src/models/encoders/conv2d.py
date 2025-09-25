import torch.nn as nn


class Conv2DEncoder(nn.Module):
    def __init__(
            self, 
            latent_dim, 
        ):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, latent_dim),
            nn.ReLU(),
        )

        
    def forward(self, x):
        x = self.encoder(x)
        return x