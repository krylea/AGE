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

export PYTHONPATH="$PYTHONPATH:./"

psp_checkpoint_path=$1
age_checkpoint_path=$2
run_name=$3
dataset=$4

class_embedding_path="class_embeds/${run_name}"
n_distribution_path="n_distribution/${run_name}"

if [ ! -f "${class_embedding_path}/class_embeddings.pt" ]
then
    python tools/get_class_embedding.py \
    --class_embedding_path=$class_embedding_path \
    --psp_checkpoint_path=pretrained_models/$psp_checkpoint_path \
    --train_data_path=data/$dataset/train \
    --test_batch_size=4 \
    --test_workers=4
fi

if [ ! -f "${n_distribution_path}/n_distribution.npy" ]
then
    python tools/get_n_distribution.py \
    --class_embedding_path=$class_embedding_path \
    --checkpoint_path=pretrained_models/$age_checkpoint_path \
    --train_data_path=data/$dataset/train \
    --test_batch_size=4 \
    --test_workers=4
fi
