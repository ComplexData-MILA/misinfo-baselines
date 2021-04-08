import sys
import subprocess
import json

from omegaconf import OmegaConf
from allennlp.commands import main

import uuid

if __name__ == '__main__':
    overrides = OmegaConf.to_container(OmegaConf.from_cli())
    overrides = dict((key.replace('--',''), value) for (key, value) in overrides.items())
    LM_name = overrides.pop('LM_name')
    data_number = overrides.pop('data_number')
    data_path = overrides.pop('data_path')
    dataset_name = data_path.split('/')[-1]

    #overrides["model.seq2vec_encoder.pretrained_model"] = LM_name
    #overrides["model.text_field_embedder.token_embedders.tokens.model_name"] = LM_name
    #overrides["dataset_reader.token_indexers.tokens.model_name"] = LM_name
    #overrides["dataset_reader.tokenizer.model_name"] = LM_name
    overrides["train_data_path"] = f"{data_path}/train_{data_number}.jsonl"
    overrides["validation_data_path"] = f"{data_path}/test_{data_number}.jsonl"
    override_args = ('--overrides', json.dumps(overrides)) if overrides else []
    print(override_args)
    sys.argv = [
        'allennlp',
        'train', 
        'conf/elmo_fakenewsnet.jsonnet',
        '-fs',
        f'chkpt/{dataset_name}_{LM_name}_{data_number}',
        *override_args
    ]

    main()

    
    if '/' in LM_name:
        name_no_slash = LM_name.split('/')[1]
    else:
        name_no_slash = LM_name

    from subprocess import PIPE, run

    result = run(f'allennlp predict chkpt/{dataset_name}_{LM_name}_{data_number} {overrides["validation_data_path"]} --output-file preds/{dataset_name}_{name_no_slash}_test_{data_number} --use-dataset-reader --predictor metrics_predictor --include-package custom_predictor --silent --weights-file chkpt/{dataset_name}_{LM_name}_{data_number}/model_state_epoch_1.th --cuda-device 0 --overrides \'{{"data_loader.batch_size" : 256}}\'',
                    stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    print(result.returncode, result.stdout, result.stderr)
    