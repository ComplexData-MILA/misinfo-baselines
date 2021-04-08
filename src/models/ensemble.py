# See:
# https://github.com/allenai/allennlp/blob/v0.9.0/allennlp/models/ensemble.py
# https://github.com/allenai/allennlp-models/blob/2b9ca77aebf21cb965207752989a4201a96d42f5/allennlp_models/rc/models/bidaf_ensemble.py

from typing import Dict, Optional, List

from overrides import overrides
import torch
from torch import nn

from allennlp.common.checks import ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

from src.modules.pooling.pooling import Pooling
from src.training.multilabel_f1 import SKLearnF1Score

import torchsnooper

@Model.register("ensemble")
class EnsembleClassifier(Model):
    def __init__(
        self, 
        models: List[Model],
        pooling_strategy: Pooling,
        dropout: float = 0.,
        f1_average: str = "weighted", 
        wandb_name: str = "Random"
    ) -> None:
        vocab = models[0].vocab
        self._num_labels = models[0]._num_labels
        
        for submodel in models:
            if submodel.vocab != vocab:
                raise ConfigurationError("Vocabularies in ensemble differ")
            if submodel._num_labels != self._num_labels:
                raise ConfigurationError("Output sizes of models in ensemble differ")

        super().__init__(vocab, None)
        
        self.submodels = nn.ModuleList(models)
        self._pooling = pooling_strategy
        self._loss = torch.nn.CrossEntropyLoss()
        
        self._accuracy = CategoricalAccuracy()
        self._f1 = SKLearnF1Score(average=f1_average)
        self._dropout = nn.Dropout(dropout)

    # @overrides
    # @torchsnooper.snoop(watch='tokens.keys()')
    def forward(self, tokens: TextFieldTensors, label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        logit_ensemble = [predict_submodel(tokens, model) for model in self.submodels] # module(tokens)['logits']
        logit_ensemble = torch.stack(logit_ensemble, dim=1) # B x len(submodels) x C
        logit_ensemble = self._dropout(logit_ensemble.transpose(1, 2)).transpose(1, 2) # dropout on submodels
        
        logits = self._pooling(logit_ensemble)
        probs = nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "probs": probs}
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            self._f1(logits, label)

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {"accuracy": self._accuracy.get_metric(reset)}
        metrics.update(**self._f1.get_metric(reset))
        return metrics

    default_predictor = "text_classifier"

def predict_submodel(tokens: TextFieldTensors, model: Model):
    aligned_input = {model._namespace: tokens[model._namespace]}
    return model(aligned_input)['logits']