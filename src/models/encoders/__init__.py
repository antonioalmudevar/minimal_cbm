from typing import Tuple, Union

from torch import nn
from torch.distributions import MultivariateNormal
from torchvision.models.resnet import *
from torchvision.models.inception import *

from .cifar_resnet import *
from .vit import *
from .mlp import MLPEncoder
from .conv2d import Conv2DEncoder


CIFAR_RESNETS = [
    'resnet8', 
    'resnet20', 
    'resnet32', 
    'resnet44', 
    'resnet56', 
    'resnet110', 
    'resnet1202',
]

RESNETS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
]

VITS = [
    "tiny_vit"
]


def select_model(arch, **kwargs) -> Union[CifarResNet, ResNet, Inception3, TinyViT]:
    return eval(arch)(**kwargs)


def get_main_encoder(arch, ch_in=1, image_size=32, **kwargs) -> Tuple[nn.Module, int]:
    if arch in CIFAR_RESNETS:
        model = select_model(arch=arch, ch_in=ch_in, **kwargs)
        latent_dim = model.in_planes
    elif arch in RESNETS:
        model = select_model(arch, **kwargs)
        latent_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif arch in VITS:
        model = select_model(arch, ch_in=ch_in, image_size=image_size, **kwargs)
        latent_dim = model.size_code
    elif arch == "mlp":
        input_dim = ch_in * image_size ** 2
        model = MLPEncoder(input_dim, **kwargs)
        latent_dim = model.latent_dim
    elif arch == "conv2d":
        model = Conv2DEncoder(**kwargs)
        latent_dim = model.latent_dim
    elif arch == "inception_v3":
        model = inception_v3(**kwargs)
        latent_dim = model.fc.in_features
        model.fc = nn.Identity()
        model.aux_logits = False
    else:
        raise ValueError
    return model, latent_dim


def get_encoder(arch, dim_z, hidden_dim=None, **kwargs) -> nn.Module:
    encoder_main, latent_dim = get_main_encoder(arch=arch, **kwargs)
    if hidden_dim is None:
        projector = nn.Linear(latent_dim, sum(dim_z))
    else:
        projector = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim, bias=False), 
            nn.ReLU(),
            nn.Linear(hidden_dim, sum(dim_z), bias=False)
        )
    return nn.Sequential(encoder_main, projector)


def get_gaussian_encoder(arch, dim_z, cov_type="global", hidden_dim=None, **kwargs):
    return GaussianEncoder(arch, dim_z, cov_type, hidden_dim, **kwargs)


class GaussianEncoder(nn.Module):

    def __init__(
            self, 
            arch, 
            dim_z, 
            cov_type="global", 
            hidden_dim=None, 
            **kwargs
        ):

        super().__init__()

        self.cov_type = cov_type
        self.dim_z_total = sum(dim_z)

        self.encoder_main, latent_dim = get_main_encoder(arch=arch, **kwargs)

        if hidden_dim is None:
            self.projector_mu = nn.Linear(latent_dim, sum(dim_z))
        else:
            self.projector_mu = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim, bias=False), 
                nn.ReLU(),
                nn.Linear(hidden_dim, sum(dim_z), bias=False)
            )

        if cov_type == "global":
            self.projector_sigma = nn.Parameter(
                torch.zeros(int(sum(dim_z) * (sum(dim_z) + 1) / 2))
            )
        elif cov_type == "empirical":
            self.projector_sigma = torch.zeros(
                int(sum(dim_z) * (sum(dim_z) + 1) / 2)
            )
        else:
            self.projector_sigma = nn.Linear(
                latent_dim, int(sum(dim_z) * (sum(dim_z) + 1) / 2),
            )
            self.projector_sigma.weight.data *= 0.01
    

    def forward(self, x, n_samples=1):
        intermediate = self.encoder_main(x)
        z_mu = self.projector_mu(intermediate)
        if self.cov_type == "global":
            z_sigma = self.projector_sigma.repeat(z_mu.size(0), 1)
        elif self.cov_type == "empirical":
            z_sigma = self.projector_sigma.unsqueeze(0).repeat(z_mu.size(0), 1, 1)
        else:
            z_sigma = self.projector_sigma(intermediate)

        if self.cov_type == "empirical":
            z_triang_cov = z_sigma
        else:
            z_triang_cov = torch.zeros(
                (z_sigma.shape[0], self.dim_z_total, self.dim_z_total),
                device=z_sigma.device,
            )
            rows, cols = torch.tril_indices(
                row=self.dim_z_total, col=self.dim_z_total, offset=0
            )
            diag_idx = rows == cols
            z_triang_cov[:, rows, cols] = z_sigma
            z_triang_cov[:, range(self.dim_z_total), range(self.dim_z_total)] = (
                F.softplus(z_sigma[:, diag_idx]) + 1e-6
            )
        if n_samples==0:
            return z_mu, z_mu, z_triang_cov
        else:
            z_dist = MultivariateNormal(z_mu, scale_tril=z_triang_cov)
            z = z_dist.rsample([n_samples]).movedim(0, -1)
            return torch.squeeze(z, -1), z_mu, z_triang_cov