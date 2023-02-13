#!/bin/bash
#SBATCH --job-name=age
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v2,rtx6000
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=50GB
#SBATCH --exclude=gpu109

checkpoint_path=$1
run_name=$2
dataset=$3

python tools/get_class_embedding.py \
--class_embedding_path=class_embeds/$run_name \
--psp_checkpoint_path=pretrained_models/$checkpoint_path \
--n_distribution_path=n_distribution/$run_name
--train_data_path=data/$dataset/train \
--test_batch_size=4 \
--test_workers=4