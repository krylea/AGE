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
output_size=$3
num_images=$4
kshot=$5
renorm=$6
resize_outputs=$7

run_name="${dataset}-pretrained"
pretrained_model_dir="pretrained_models"

psp_checkpoint_path="${pretrained_model_dir}/psp_${dataset}.pt"
age_checkpoint_path="${pretrained_model_dir}/age_${dataset}.pt"

class_embedding_path="class_embeds/${run_name}/class_embeddings_${output_size}.pt"
n_distribution_path="n_distribution/${run_name}/n_distribution_${output_size}.npy"

dataset_path="../setgan2/datasets/animal_faces"

argstring="--name=$name \
--output_path=eval \
--checkpoint_path=$age_checkpoint_path \
--test_data_path=$dataset_path/test \
--train_data_path=data/$dataset_path/train \
--dataset_type=${dataset}_encode \
--class_embedding_path=$class_embedding_path \
--n_distribution_path=$n_distribution_path \
--test_batch_size=4 \
--test_workers=4 \
--n_images=$num_images \
--alpha=1 \
--beta=0.005 \
--output_size=$output_size
--kshot=$kshot
--combine_fid"

if [ $renorm -eq 1 ]
then
    argstring="$argstring --renorm"
fi
if [ $resize_outputs -eq 1 ]
then
    argstring="$argstring --resize_outputs"
fi

python3 tools/get_scores.py $argstring
