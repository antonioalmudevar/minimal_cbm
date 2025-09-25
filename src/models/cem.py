from typing import List, Union

import torch
from torch import Tensor

from .cbm import ConceptBottleneckModel


class ConceptEmbeddingModel(ConceptBottleneckModel):

    def __init__(
            self,
            dim_c: Union[int, List[int]],
            **kwargs
        ) -> None:

        dim_z = dim_c * 2 if isinstance(dim_c, int) else [dcj * 2 for dcj in dim_c]
        super().__init__(dim_c=dim_c, dim_z=dim_z, **kwargs)

    #==========Task Head q_ϕ(y|z)==========
    def set_head_y(self):
        self.mlp_y, self.act_y = self._build_head(
            input_dim=sum(self.dim_z)//2, 
            hidden_dims=self.hidden_dims_y, 
            output_dim=self.dim_y,
            continuous=self.continuous_y
        )
        self.loss_y = self._get_loss_fn(self.dim_y, self.continuous_y)

    def q_y_zcpreds(self, z: Tensor, c_preds: Tensor) -> Tensor:
        #In CEM, $q(\hat{y}|z,\hat{c})$ is used instead of $q(\hat{y}|z)$
        z_mod = torch.stack([
            c_preds[:,j,0] * z[:,self.idxs_z[j]][:,0] + \
                (1-c_preds[:,j,0]) * z[:,self.idxs_z[j]][:,1] for j in range(self.n_concepts)
        ]).permute(1, 0)
        y_logits = self.mlp_y(z_mod)
        y_preds = self.act_y(y_logits)
        return y_logits, y_preds
    
    
    #==========Forward==========
    def forward(self, x: Tensor, c: Tensor, sampling: bool=False):
        z = self.p_z_x(x)
        c_logits, c_preds = self.q_c_z(z)
        y_logits, y_preds = self.q_y_zcpreds(z, c_preds)
        return {
            'z':        z,
            'y_logits': y_logits,
            'y_preds':  y_preds,
            'c_logits': c_logits,
            'c_preds':  c_preds,
        }