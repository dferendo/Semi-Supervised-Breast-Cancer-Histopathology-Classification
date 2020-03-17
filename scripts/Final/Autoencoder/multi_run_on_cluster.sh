#!/bin/sh

export DATASET_DIR="../../data/BreaKHis_v1/"

magnifications=("40X" "100X" "200X" "400X")
seeds=(4324 342432 764365)

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    experiment_result_location="./experiments/autoencoder_${seed}_${magnification}"

    sbatch mlp_cluster_train.sh ${seed} "${magnification}" "${experiment_result_location}"
  done
done
