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
output_size=$5

class_embedding_path="class_embeds/${run_name}/class_embeddings_${output_size}.pt"
n_distribution_path="n_distribution/${run_name}/n_distribution_${output_size}.npy"

if [ ! -f $class_embedding_path ]
then
    python tools/get_class_embedding.py \
    --class_embedding_path=$class_embedding_path \
    --psp_checkpoint_path=pretrained_models/$psp_checkpoint_path \
    --train_data_path=data/$dataset/train \
    --test_batch_size=4 \
    --test_workers=4 \
    --output_size=$output_size
fi

if [ ! -f $n_distribution_path ]
then
    python tools/get_n_distribution.py \
    --class_embedding_path=$class_embedding_path \
    --n_distribution_path=$n_distribution_path \
    --checkpoint_path=pretrained_models/$age_checkpoint_path \
    --train_data_path=data/$dataset/train \
    --test_batch_size=4 \
    --test_workers=4 \
    --output_size=$output_size
fi
