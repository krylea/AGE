#!/bin/bash

NREF=(1 5 10 30)
NEVAL=128
IMGSIZE=128
NIMG=128
NEXP=3

dataset=$1

for nref in "${NREF[@]}"
do
    sbatch run_fid.sh $dataset "${dataset}_${nref}_${IMGSIZE}_${NEVAL}" $nref $IMGSIZE $NEXP $NEVAL
done