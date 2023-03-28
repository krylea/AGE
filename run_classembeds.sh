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

run_name=$1
dataset=$2
output_size=$3

psp_checkpoint_path="pretrained_models/psp_${dataset}.pt"
age_checkpoint_path="pretrained_models/age_${dataset}.pt"

data_path="../setgan2/datasets/animal_faces/train"

class_embedding_path="class_embeds/${run_name}/class_embeddings_${output_size}.pt"
n_distribution_path="n_distribution/${run_name}/n_distribution_${output_size}.npy"

if [ ! -f $class_embedding_path ]
then
    python tools/get_class_embedding.py \
    --class_embedding_path=$class_embedding_path \
    --psp_checkpoint_path=$psp_checkpoint_path \
    --train_data_path=$data_path \
    --test_batch_size=4 \
    --test_workers=4 \
    --output_size=$output_size
fi

if [ ! -f $n_distribution_path ]
then
    python tools/get_n_distribution.py \
    --class_embedding_path=$class_embedding_path \
    --n_distribution_path=$n_distribution_path \
    --checkpoint_path=$age_checkpoint_path \
    --train_data_path=$data_path \
    --test_batch_size=4 \
    --test_workers=4 \
    --output_size=$output_size
fi
