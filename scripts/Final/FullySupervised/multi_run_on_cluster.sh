#!/bin/sh

seeds=(4545 342432 764365)
magnifications=("40X")

dropout_values=(0. 0.1 0.2)
weight_decay_values=(0.00001 0.0001 0.001)
learning_rates=(0.1 0.01)

for seed in "${seeds[@]}"
do
  for magnification in "${magnifications[@]}"
  do
    for dropout_value in "${dropout_values[@]}"
    do
      for weight_decay_value in "${weight_decay_values[@]}"
      do
        for learning_rate in "${learning_rates[@]}"
        do
          experiment_result_location="./experiments/basemixmatch_${seed}_${magnification}_${dropout_value}_${weight_decay_value}_${learning_rate}"

          sbatch mlp_cluster_train.sh ${seed} "${magnification}" ${dropout_value} ${weight_decay_value} ${learning_rate} "${experiment_result_location}"
        done
      done
    done
  done
done
