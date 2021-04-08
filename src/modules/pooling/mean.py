import torch
from src.modules.pooling.pooling import Pooling

@Pooling.register('mean')
class MeanPooling(Pooling):
    def forward(self, logits_ensemble: torch.Tensor):
        return logits_ensemble.mean(dim=1)