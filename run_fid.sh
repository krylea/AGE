#!/bin/bash
#SBATCH --job-name=AGE
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=rtx6000,t4v2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=25GB
#SBATCH --exclude=gpu109

dataset=$1

class_embedding_path="class_embeds/${dataset}-pretrained"
n_distribution_path="n_distribution/${dataset}-pretrained"
psp_checkpoint_path="pretrained_models/psp_${dataset}.pt"
age_checkpoint_path="pretrained_models/age_${dataset}.pt"

train_data_path="../setgan2/datasets/${dataset}/train_all"
test_data_path="../setgan2/datasets/${dataset}/test"

python tools/fid.py \
--dataset_type="${dataset}_encode" \
--class_embedding_path=$class_embedding_path \
--n_distribution_path=$n_distribution_path \
--psp_checkpoint_path=$psp_checkpoint_path \
--checkpoint_path=$age_checkpoint_path \
--train_data_path=$train_data_path \
--test_data_path=$test_data_path \
--test_batch_size=4 \
--test_workers=4 \
--alpha=1 \
--beta=0.005 \
--n_images 128