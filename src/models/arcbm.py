import torch
from torch import nn, Tensor

from .cbm import ConceptBottleneckModel


class AutoregressiveConceptBottleneckModel(ConceptBottleneckModel):

    def __init__(self, **kwargs) -> None:

        super().__init__(**kwargs)

    
    def set_heads_c(self):
        self.mlp_c = nn.ModuleList()
        self.act_c = nn.ModuleList()
        self.loss_c = nn.ModuleList()
        for j in range(self.n_concepts):
            input_dim = self.dim_z[j] + sum(self.dim_c[:j])
            mlp_c_j, act_c_j = self._build_head(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims_c[j],
                output_dim=self.dim_c[j],
                continuous=self.continuous_c[j]
            )
            loss_c_j = self._get_loss_fn(
                self.dim_c[j], 
                self.continuous_c[j],
                pos_weight=torch.FloatTensor([self.imbalance_ratio[j]]) if \
                    self.imbalance_ratio is not None else None
            )
            self.mlp_c.append(mlp_c_j)
            self.act_c.append(act_c_j)
            self.loss_c.append(loss_c_j)


    def q_c_z(self, z: Tensor) -> Tensor:
        c_logits = []
        c_preds = []
        prev = None
        for j in range(self.n_concepts):
            z_j = z[:, self.idxs_z[j]]
            inp = torch.cat((z_j, prev), dim=1) if prev is not None else z_j
            logits_j = self.mlp_c[j](inp)
            preds_j = self.act_c[j](logits_j)
            c_logits.append(logits_j)
            c_preds.append(preds_j)
            prev = torch.cat(c_preds, dim=1)
        c_logits = torch.cat(c_logits, dim=1).unsqueeze(-1)
        c_preds = torch.cat(c_preds, dim=1).unsqueeze(-1)
        return c_logits, c_preds
