local data_root = "./%s.jsonl";

local transformer_model = "google/bert_uncased_L-2_H-128_A-2";
local transformer_dim = 128;

{
    "dataset_reader": {
        "type": "text_classification_json",
        "token_indexers": {
            "tokens": {
                "type": "pretrained_transformer",
                "model_name"+: $.model.text_field_embedder.token_embedders.tokens.model_name
            }
        },
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name"+: $.model.text_field_embedder.token_embedders.tokens.model_name
        }
    },

    "train_data_path": data_root % "train",
    "validation_data_path": data_root % "valid",

    "model": {
        "type": "src.models.basic_classifier_with_metrics.BasicClassifierWithMetrics",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer",
                    "model_name"+: transformer_model
                },
            }
        },
        "seq2vec_encoder": {
            "type": "bert_pooler",
            "pretrained_model"+: transformer_model,
            "dropout": 0.1,
        }
    },
    
    "data_loader": {
        "batch_size": 32,
        "shuffle": false
    },
    "trainer": {
        "num_epochs": 50,
        "validation_metric": "+accuracy",
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": 50,
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