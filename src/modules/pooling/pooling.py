import torch
from torch import nn
from allennlp.common import Registrable

class Pooling(Registrable, nn.Module):
    default_implementation = 'mean'

    def forward(self, logits_ensemble: torch.Tensor):
        raise NotImplementedError()