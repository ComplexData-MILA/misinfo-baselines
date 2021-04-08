#!/bin/bash

#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/slurm-%j.out
#SBATCH --error=slurm/slurm-%j.err

module load anaconda/3

conda deactivate
conda activate env_name

allennlp predict chkpt/politifact_funnel-transformer/xlarge-base_3 data/politifact/test_3.jsonl --output-file preds/politifact_funnel_3 --use-dataset-reader --predictor metrics_predictor --include-package package_test --silent --weights-file chkpt/politifact_funnel-transformer/xlarge-base_3/model_state_epoch_1.th --cuda-device 0 --overrides '{"data_loader.batch_size" : 256}'
