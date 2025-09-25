from typing import List, Dict, Union

import torch
from torch import nn, Tensor

from .encoders import get_encoder
from .utils import SqueezeBCEWithLogitsLoss, SqueezeMSELoss


class VanillaModel(nn.Module):

    def __init__(
            self,
            n_concepts: int,
            dim_y: int,
            dim_c: Union[int, List[int]],
            continuous_y: bool,
            continuous_c: Union[bool, List[bool]],
            encoder: Dict,
            hidden_dims_y: Union[int, List[int]],
            ch_in: int=3,
            image_size: int=32,
            imbalance_ratio: List[float]=None,
            dim_z: Union[int, List[int]]=None,
            var_z: float=0.0,
        ) -> None:
        
        super().__init__()

        if isinstance(dim_c, int):
            dim_c = [dim_c] * n_concepts
        if dim_z is not None and isinstance(dim_z, int):
            dim_z = [dim_z] * n_concepts
        assert n_concepts==len(dim_c), "n_concepts must be equal to the length of dim_c"
        if isinstance(continuous_c, bool):
            continuous_c = [continuous_c] * n_concepts
        if isinstance(hidden_dims_y, int):
            hidden_dims_y = [hidden_dims_y]
        
        self.n_concepts = n_concepts
        self.dim_y = dim_y
        self.dim_c = dim_c
        self.dim_z = dim_c if dim_z is None else dim_z
        self.continuous_y = continuous_y
        self.continuous_c = continuous_c
        self.continuous_z = [True] * self.n_concepts    # z is assumed to be always continuous

        self.cfg_encoder = {**encoder, **{'ch_in': ch_in, 'image_size': image_size}}
        self.hidden_dims_y = hidden_dims_y
        self.var_z = var_z
        self.imbalance_ratio = imbalance_ratio

        self.set_encoder()
        self.set_head_y()

        self.has_concepts = False
        self.forward_returns = ['z', 'y_preds']
        self.losses = ['task', 'total']


    #==========Encoder p_θ(z|x)==========
    def set_encoder(self):
        self.encoder = get_encoder(dim_z=self.dim_z, **self.cfg_encoder)

    def p_z_x(self, x: Tensor) -> Tensor:
        return self.encoder(x)


    #==========Task Head q_ϕ(y|z)==========
    def set_head_y(self):
        self.mlp_y, self.act_y = self._build_head(
            input_dim=sum(self.dim_z), 
            hidden_dims=self.hidden_dims_y, 
            output_dim=self.dim_y,
            continuous=self.continuous_y
        )
        self.loss_y = self._get_loss_fn(self.dim_y, self.continuous_y)

    def q_y_z(self, z: Tensor) -> Tensor:
        y_logits = self.mlp_y(z)
        y_preds = self.act_y(y_logits)
        return y_logits, y_preds
    
    def get_loss_y(self, y: Tensor, y_logits: Tensor) -> Tensor:
        return self.loss_y(y_logits, y)


    #==========Forward==========
    def forward(self, x: Tensor, c: Tensor, sampling: bool=False) -> Dict[str, Tensor]:
        z = self.p_z_x(x)
        y_logits, y_preds = self.q_y_z(z)
        return {
            'z':        z,
            'y_logits': y_logits,
            'y_preds':  y_preds,
        }
    
    def get_loss(
            self, 
            y: Tensor, 
            y_logits: Tensor, 
            **kwargs
        ) -> Dict[str, Tensor]:
        y_loss = self.get_loss_y(y, y_logits)
        loss = y_loss
        return {
            'task':     y_loss,   
            'total':    loss,
        }

    #==========Miscellanea==========
    def _reparameterize(self, mean_z: Tensor, n_samples: int=None) -> Tensor:
        if n_samples==0 or self.var_z==0:
            return mean_z
        else:
            mean_z = mean_z.unsqueeze(0).expand(n_samples, *mean_z.shape)  
            eps = torch.randn_like(mean_z)
            z = mean_z + torch.sqrt(self.var_z) * eps
            return z.view(-1, mean_z.shape[-1])
        
    def _build_head(
            self, 
            input_dim: int, 
            hidden_dims: List[int], 
            output_dim: int,
            continuous: bool,
        ) -> nn.Module:
        if hidden_dims==None:
            mlp = nn.Identity()
        else:
            layers = []
            dims = [input_dim] + hidden_dims + [output_dim]
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(nn.ReLU())
            mlp = nn.Sequential(*layers)
        if continuous:
            act = nn.Identity()
        else:
            act = nn.Sigmoid() if output_dim==1 else nn.Softmax(dim=1)
        return mlp, act

    def _get_loss_fn(self, dim, continuous, **kwargs):
        if continuous:
            return SqueezeMSELoss(**kwargs)
        else:
            return SqueezeBCEWithLogitsLoss(**kwargs) if dim==1 else nn.CrossEntropyLoss(**kwargs)
