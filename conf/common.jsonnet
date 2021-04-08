local lm_clf = function(lm, dropout=0.1){
    "type": "metrics_classifier",
    "namespace": lm.tokens,

    "text_field_embedder": {
        "token_embedders": {
            [lm.tokens]: {
                "type": "pretrained_transformer_mismatched",
                "model_name": lm.name
            },
        }
    },
    "seq2vec_encoder": if (lm.encoder != null) then lm.encoder else {
        "type": "bert_pooler",
        "pretrained_model": lm.name,
        "dropout": dropout,
    }
};

{
    lm_tuple::function(t){
        "name": t[0],
        "tokens": t[1],
        "encoder": if std.length(t) > 2 then t[2] else null
    },
    lm_classifier::lm_clf,
    lm_token_indexers::function(lms){
        [lm.tokens]: {
            "type": "pretrained_transformer_mismatched",
            "model_name": lm.name
        }
        for lm in lms
    },
    wandb_callback::function(name=null){
        "type": "wandb",
        "project": std.extVar("WANDB_PROJECT"),
        "entity": std.extVar("WANDB_ENTITY")
    },
}