from typing import OrderedDict

from torch import nn, Tensor


class SqueezeBCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input.squeeze(1), target)
    
"""
class SqueezeBCEWithLogitsLoss(nn.Module):

    def __init__(self, weight: float=1.):
        super().__init__()
        self.register_buffer('weight', weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        preds = torch.sigmoid(input.squeeze(1)).to(input.device) + 1e-6
        loss = - target * torch.log(preds) - self.weight * (1-target) * torch.log(1-preds)
        return loss.mean()
"""

class SqueezeMSELoss(nn.MSELoss):

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input.squeeze(1), target)
    

class ModelParallel(nn.DataParallel):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.has_concepts = self.module.has_concepts
        self.n_concepts = self.module.n_concepts
        self.losses = self.module.losses
        self.forward_returns = self.module.forward_returns

    def load_state_dict(
            self, 
            state_dict: OrderedDict[str, Tensor], 
            strict: bool = True,
        ):
        return self.module.load_state_dict(state_dict, strict)

    def state_dict(self):
        return self.module.state_dict()

    def get_loss(self, *args, **kwargs) -> Tensor:
        return self.module.get_loss(*args, **kwargs)