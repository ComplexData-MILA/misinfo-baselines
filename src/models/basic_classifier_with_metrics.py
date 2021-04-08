from typing import Dict, Optional

from overrides import overrides
import torch

from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models import Model, BasicClassifier
from allennlp.nn import util
from src.training.multilabel_f1 import SKLearnF1Score

import torchsnooper

@Model.register("metrics_classifier")
class BasicClassifierWithMetrics(BasicClassifier):
    def __init__(self, f1_average: str = "weighted", **kwargs) -> None:
        super().__init__(**kwargs)
        self._f1 = SKLearnF1Score(average=f1_average)

    def _get_logits(self, tokens: TextFieldTensors) -> torch.Tensor:
        embedded_text = self._text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits = self._classification_layer(embedded_text)
        return logits, embedded_text

    @overrides
    def forward(self, tokens: TextFieldTensors, label: torch.IntTensor = None) -> Dict[str, torch.Tensor]:
        logits, embedded_text = self._get_logits(tokens)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"probs": probs, "true_label":label.float()} # "embedding": embedded_text,
        if label is not None:
            loss = self._loss(logits, label.long().view(-1))
            output_dict["loss"] = loss
            self._accuracy(logits, label)
            self._f1(logits, label)
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = super().get_metrics(reset)
        metrics.update(**self._f1.get_metric(reset))
        return metrics

    default_predictor = "text_classifier"