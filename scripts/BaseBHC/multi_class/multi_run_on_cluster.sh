#!/bin/sh

magnifications=("40X" "100X" "200X" "400X")
unlabelled_splits=(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for magnification in "${magnifications[@]}"
do
  for unlabelled_split in "${unlabelled_splits[@]}"
  do
    experiment_result_location="./experiments/base_tune_test_subclasses_3_${magnification}_${unlabelled_split}"

    erf_sched_alpha=-3
    erf_sched_beta=3

    sbatch mlp_cluster_train.sh $experiment_result_location $magnification $unlabelled_split $erf_sched_alpha $erf_sched_beta
  done
done
