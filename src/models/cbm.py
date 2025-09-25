from typing import List, Union, Dict

import torch
from torch import nn, Tensor

from .vanilla import VanillaModel


class ConceptBottleneckModel(VanillaModel):

    def __init__(
            self,
            hidden_dims_c: Union[int, List[int], List[List[int]]],
            beta: float=1.,
            **kwargs
        ) -> None:
        
        super().__init__(**kwargs)

        if isinstance(hidden_dims_c, int):
            hidden_dims_c = [hidden_dims_c]
        if hidden_dims_c==[] or hidden_dims_c is None or isinstance(hidden_dims_c[0], int):
            hidden_dims_c = [hidden_dims_c] * self.n_concepts
        
        self.hidden_dims_c = hidden_dims_c
        self.beta = beta

        self.idxs_z = [list(range(sum(self.dim_z[:i]), sum(self.dim_z[:i]) + size)) \
            for i, size in enumerate(self.dim_z)]
        self.idxs_c = [list(range(sum(self.dim_c[:i]), sum(self.dim_c[:i]) + size)) \
            for i, size in enumerate(self.dim_c)]
        
        self.set_heads_c()

        self.has_concepts = True
        self.forward_returns = ['z', 'y_preds', 'c_preds']
        self.losses = ['task', 'concepts',  'total']


    #==========Concept Heads q_ϕ(c_j|z_j)==========
    def set_heads_c(self):
        assert len(self.dim_c)==len(self.continuous_c)==len(self.hidden_dims_c), \
            "dim_c, continuous_c and hidden_dims_c must have the same length"
        mlp_c, act_c, loss_c = [], [], []
        for j in range(self.n_concepts):
            mlp_c_j, act_c_j = self._build_head(
                input_dim=self.dim_z[j], 
                hidden_dims=self.hidden_dims_c[j], 
                output_dim=self.dim_c[j],
                continuous=self.continuous_c[j]
            )
            loss_c_j = self._get_loss_fn(
                dim=self.dim_c[j], 
                continuous=self.continuous_c[j],
                pos_weight=torch.FloatTensor([self.imbalance_ratio[j]]) if \
                    self.imbalance_ratio is not None else None
            )
            mlp_c.append(mlp_c_j)
            act_c.append(act_c_j)
            loss_c.append(loss_c_j)
        self.mlp_c = nn.ModuleList(mlp_c)
        self.act_c = nn.ModuleList(act_c)
        self.loss_c = nn.ModuleList(loss_c)

    def q_c_z(self, z: Tensor) -> Tensor:
        c_logits = torch.stack([
            self.mlp_c[j](z[:,self.idxs_z[j]]) for j in range(self.n_concepts)
        ]).permute(1, 0, 2)
        c_preds = torch.stack([
            self.act_c[j](c_logits[:,j]) for j in range(self.n_concepts)
        ]).permute(1, 0, 2)
        return c_logits, c_preds

    def get_loss_c(self, c: Tensor, c_logits: Tensor) -> Tensor:
        loss_c = 0
        for j in range(self.n_concepts):
            loss_c += self.loss_c[j](c_logits[:,j], c[:,j])
        return loss_c
    

    #==========Forward==========
    def forward(self, x: Tensor, c: Tensor, sampling: bool=False):
        z = self.p_z_x(x)
        y_logits, y_preds = self.q_y_z(z)
        c_logits, c_preds = self.q_c_z(z)
        return {
            'z':        z,
            'y_logits': y_logits,
            'y_preds':  y_preds,
            'c_logits': c_logits,
            'c_preds':  c_preds,
        }
    
    def get_loss(
            self, 
            y: Tensor, 
            c: Tensor, 
            y_logits: Tensor, 
            c_logits: Tensor, 
            **kwargs
        ) -> Dict[str, Tensor]:
        y_loss = self.get_loss_y(y, y_logits)
        c_loss = self.get_loss_c(c, c_logits)
        loss = y_loss + self.beta * c_loss
        return {
            'task':     y_loss,
            'concepts': c_loss,   
            'total':    loss,
        }
    
    #==========Interventions==========
    def prepare_interventions(self, zs: Tensor, cs: Tensor, quantiles: List[float]=[0.05, 0.95]):
        self.quantiles = []
        for j in range(self.n_concepts):
            if self.continuous_c[j]:
                self.quantiles.append(None)
            else:
                if self.dim_c[j]==1:
                    z_j = zs[:,self.idxs_c[j]]
                    """
                    self.quantiles.append([
                        torch.quantile(z_j, quantiles[0]),
                        torch.quantile(z_j, quantiles[1])
                    ])
                    """
                    c_j = cs[:,self.idxs_c[j]]
                    z_j_pos = z_j[c_j==1]
                    z_j_neg = z_j[c_j==0]
                    self.quantiles.append([
                        z_j_neg.mean(),
                        z_j_pos.mean()
                    ])
                else:
                    raise ValueError("Multiclass concepts cannot be intervened in CBMs")

    def _intervene_kj(self, c: Tensor, k: int, j: int):
        assert self.hidden_dims_c==[None] * self.n_concepts, \
            "Interventions in CBMs cannot be performed for non-invertible concept heads"
        assert c.shape[1]==self.n_concepts, \
            "Length of c must be equal to the number of concepts"
        if self.continuous_c[j]:
            return c[k,self.idxs_z[j]]
        else:
            return self.quantiles[j][0] if c[k,self.idxs_z[j]]==0 else self.quantiles[j][1]

    def intervene(self, x: Tensor, c: Tensor):
        z = self.p_z_x(x)
        y_logits, y_preds = self.q_y_z(z)
        c_logits, c_preds = self.q_c_z(z)
        z_copy = z.clone()
        for k in range(x.shape[0]):
            for j in range(self.n_concepts):
                if not torch.isnan(c[k,self.idxs_c[j]]):
                    z_copy[k,self.idxs_z[j]] = self._intervene_kj(c, k, j)
        y_logits, y_preds = self.q_y_z(z_copy)
        c_logits, c_preds = self.q_c_z(z_copy)
        return {
            'z':        z,
            'y_logits': y_logits,
            'y_preds':  y_preds,
            'c_logits': c_logits,
            'c_preds':  c_preds,
        }