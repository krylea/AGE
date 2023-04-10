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
export CUDA_HOME="/pkgs/cuda-11.7/"

name=$1
dataset=$2
n_ref=$3
image_size=$4
ndist_suffix=$5

run_name="${dataset}-pretrained"
pretrained_model_dir="pretrained_models"

psp_checkpoint_path="${pretrained_model_dir}/psp_${dataset}.pt"
age_checkpoint_path="${pretrained_model_dir}/age_${dataset}.pt"

n_distribution_path="n_distribution/${run_name}/n_distribution${ndist_suffix}.npy"

#dataset_path="../setgan2/datasets/animal_faces"

argstring="--name=$name \
--output_path=eval \
--checkpoint_path=$age_checkpoint_path \
--n_distribution_path=$n_distribution_path \
--n_images=128 \
--n_ref=$n_ref \
--alpha=1 \
--beta=0.005"


python3 tools/get_scores.py $argstring
