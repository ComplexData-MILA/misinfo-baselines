from allennlp.predictors.predictor import Predictor

from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


from allennlp.data import Instance, Vocabulary
from allennlp.data.batch import Batch
from allennlp.nn import util
import torch
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.float32):
            return obj.astype(float)
        return json.JSONEncoder.default(self, obj)

@Predictor.register('metrics_predictor')
class metrics_predictor(Predictor):
    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        return json.dumps(outputs, cls=NumpyEncoder) + "\n"

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        sentence = json_dict["sentence"]
        reader_has_tokenizer = (
            getattr(self._dataset_reader, "tokenizer", None) is not None
            or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = SpacyTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(
        self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label = numpy.argmax(outputs["probs"])
        new_instance.add_field("label", LabelField(int(label), skip_indexing=True))
        return [new_instance]
    
    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        #outputs = self._model.forward(instance)
        #return sanitize(outputs)

        """
        Takes a list of `Instances`, converts that text into arrays using this model's `Vocabulary`,
        passes those arrays through `self.forward()` and `self.make_output_human_readable()` (which
        by default does nothing) and returns the result.  Before returning the result, we convert
        any `torch.Tensors` into numpy arrays and separate the batched output into a list of
        individual dicts per instance. Note that typically this will be faster on a GPU (and
        conditionally, on a CPU) than repeated calls to `forward_on_instance`.
        # Parameters
        instances : `List[Instance]`, required
            The instances to run the model on.
        # Returns
        A list of the models output for each instance.
        """
        instances = [instance]
        batch_size = len(instances)
        with torch.no_grad():
            cuda_device = self._model._get_prediction_device()
            dataset = Batch(instances)
            dataset.index_instances(self._model.vocab)
            model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            outputs = self.make_output_human_readable(self._model(**model_input))

            instance_separated_output: List[Dict[str, numpy.ndarray]] = [
                {} for _ in dataset.instances
            ]
            for name, output in list(outputs.items()):
                if isinstance(output, torch.Tensor):
                    # NOTE(markn): This is a hack because 0-dim pytorch tensors are not iterable.
                    # This occurs with batch size 1, because we still want to include the loss in that case.
                    if output.dim() == 0:
                        output = output.unsqueeze(0)

                    if output.size(0) != batch_size:
                        self._maybe_warn_for_unseparable_batches(name)
                        continue
                    output = output.detach().cpu().numpy()
                elif len(output) != batch_size:
                    self._maybe_warn_for_unseparable_batches(name)
                    continue
                for instance_output, batch_element in zip(instance_separated_output, output):
                    instance_output[name] = batch_element
            return instance_separated_output
    
    
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        predictions = output_dict["probs"]
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self._model.vocab.get_index_to_token_vocabulary(self._model._label_namespace).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        output_dict["label"] = classes
        tokens = []
        '''
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self._model.vocab.get_token_from_index(token_id.item(), namespace=self._model._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        '''
        return output_dict