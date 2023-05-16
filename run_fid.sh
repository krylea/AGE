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
name=$2
n_ref=$3
image_size=$4
n_exps=$5
n_eval=$6
sample_eval=${7:-1}
resize_outputs=${8:-1}
randomize_noise=${9:-0}

class_embedding_path="class_embeds/${dataset}-pretrained"
n_distribution_path="n_distribution/${dataset}-pretrained"
psp_checkpoint_path="pretrained_models/psp_${dataset}.pt"
age_checkpoint_path="pretrained_models/age_${dataset}.pt"

train_data_path="../setgan2/datasets/${dataset}/train"
test_data_path="../setgan2/datasets/${dataset}/test"

if [ $dataset = "animalfaces" ]
then
    train_data_path="../setgan2/datasets/${dataset}/train_all"
fi

real_dir="outputs/${name}_real"
fake_dir="outputs/${name}_fake"

argstring="--name=$name \
--n_distribution_path=$n_distribution_path \
--checkpoint_path=$age_checkpoint_path \
--real_dir=$real_dir \
--fake_dir=$fake_dir \
--dataset=$dataset \
--test_data_path=$test_data_path \
--alpha=1 \
--beta=0.005 \
--n_images=128 \
--n_ref=$n_ref \
--image_size=$image_size \
--n_exps=$n_exps \
--n_eval=$n_eval"

if [ $resize_outputs -eq 1 ]
then
    argstring="${argstring} --resize_outputs"
fi
if [ $randomize_noise -eq 1 ]
then
    argstring="${argstring} --randomize_noise"
fi
if [ $sample_eval -eq 1 ]
then
    argstring="${argstring} --sample_eval"
fi
#if [ $cleanup -eq 1 ]
#then
#    argstring="${argstring} --cleanup"
#fi


python tools/fid.py $argstring