local data_root = "./%s.jsonl";

{
    "dataset_reader": {
        "type": "text_classification_json",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "train_data_path": data_root % "train",
    "validation_data_path": data_root % "valid",
    "model": {
        "type": "src.models.basic_classifier_with_metrics.BasicClassifierWithMetrics",
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {
                    "type": "elmo_token_embedder",
                    "options_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                    "weight_file": "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5",
                    "do_layer_norm": false,
                    "dropout": 0.0
                }
            }
        },
        "seq2vec_encoder": {
            "type": "lstm",
            "input_size": 1024,
            "hidden_size": 300,
            "num_layers": 1,
            "bidirectional": true
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": false
    },
    "trainer": {
        "num_epochs": 2,
        "validation_metric": "+accuracy",
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 2,
            "num_steps_per_epoch": 500,
            "cut_frac": 0.06
        },
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-5,
            "weight_decay": 0.1,
        }
    }
}