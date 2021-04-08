import torch
from torch import nn
from torch.nn import functional as F

from allennlp.nn import util
from src.modules.pooling.pooling import Pooling

import torchsnooper

@Pooling.register('attention')
class AttentionPooling(Pooling):
    def __init__(
        self,
        num_models: int,
        **kwargs
    ):
        super().__init__()
        self._vec = nn.Parameter(torch.randn(1, num_models), requires_grad=True)

    def forward(self, logits_ensemble: torch.Tensor): 
        attn = F.softmax(self._vec, dim=-1).unsqueeze(-1)
        return (attn * logits_ensemble).sum(dim=1)