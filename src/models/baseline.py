from typing import Dict

from torch import Tensor

from .vanilla import VanillaModel


class BaselineModel(VanillaModel):

    def __init__(self, **kwargs) -> None:
        
        super().__init__(**kwargs, encoder={})

        self.forward_returns = ['y_preds']
        print(self.mlp_y)


    #==========Encoder p_θ(z|x)==========
    def set_encoder(self):
        pass

    #==========Forward==========
    def forward(self, x: Tensor, c: Tensor, sampling: bool=False) -> Dict[str, Tensor]:
        y_logits, y_preds = self.q_y_z(c)
        return {
            'y_logits': y_logits,
            'y_preds':  y_preds,
        }