#!/bin/bash
#SBATCH --job-name=age
#SBATCH --output=logs/slurm-%j.txt
#SBATCH --open-mode=append
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=t4v2,rtx6000
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=50GB
#SBATCH --exclude=gpu109

export PYTHONPATH="$PYTHONPATH:./"

#psp_checkpoint_path=$1
age_checkpoint_path=$1
run_name=$2
dataset=$3

class_embedding_path="class_embeds/${run_name}"
n_distribution_path="n_distribution/${run_name}/n_distribution.npy"

python tools/generate_images_for_metrics.py \
--output_path=outputs/$run_name \
--checkpoint_path=pretrained_models/$age_checkpoint_path \
--test_data_path=../setgan2/datasets/$dataset/test \
--train_data_path=../setgan2/datasets/$dataset/train \
--class_embedding_path=$class_embedding_path \
--n_distribution_path=$n_distribution_path \
--test_batch_size=4 \
--test_workers=4 \
--n_images=128 \
--alpha=1 \
--beta=0.005