import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):

        super(MLPEncoder, self).__init__()

        self.latent_dim = latent_dim
        
        # Build the layers dynamically
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        # Wrap in nn.Sequential
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder(x)